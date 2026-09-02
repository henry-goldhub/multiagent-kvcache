"""Three-stage append-only inference with exact and adapted cache reuse."""

from __future__ import annotations

import random
import time
from typing import Any, ClassVar

import torch

from .adapter import AdapterError, KVAdapter, build_adapter
from .cache import CacheState
from .compat import (
    eos_token_ids,
    forward_with_cache,
    model_input_device,
    synchronize,
    timed_forward,
)
from .models import ModelBundle
from .prompts import PromptState


class Pipeline:
    """Run extract → plan → compute with exact or experimental cache reuse."""

    VALID_CACHE_POLICIES: ClassVar[set[str]] = {"disabled", "same_model_only", "cross_model"}

    def __init__(self, models: dict[str, ModelBundle], config: dict[str, Any] | None = None):
        self.models = models
        self.config = config or {}
        requested_device = self.config.get("device", "auto")
        if requested_device != "auto":
            device = torch.device(requested_device)
            for bundle in self.models.values():
                device_map = getattr(bundle.model, "hf_device_map", None)
                if device_map and len({str(value) for value in device_map.values()}) > 1:
                    raise ValueError(
                        "An explicit device cannot be applied to a model sharded by device_map; "
                        "use device='auto'"
                    )
                bundle.model.to(device)
        self.cache_policy = self.config.get("cache_policy", "same_model_only")
        if self.cache_policy not in self.VALID_CACHE_POLICIES:
            allowed = ", ".join(sorted(self.VALID_CACHE_POLICIES))
            raise ValueError(f"cache_policy must be one of: {allowed}")
        adapter_config = self.config.get("adapter")
        if adapter_config is None and self.cache_policy == "cross_model":
            adapter_config = {"type": "ridge"}
        self.adapter: KVAdapter = build_adapter(adapter_config)
        self.seed = int(self.config.get("seed", 42))
        self.degradation_threshold = float(
            self.config.get("degradation", {}).get("threshold", 0.15)
        )

    def run(self, task_input: str, step_assignment: list[str]) -> tuple[str, dict[str, Any]]:
        if len(step_assignment) != 3:
            raise ValueError("step_assignment must contain exactly three model names")
        if any(name not in self.models for name in step_assignment):
            raise KeyError("step_assignment contains a model that was not supplied")

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        prompt = PromptState.from_question(task_input)
        previous_cache: CacheState | None = None
        outputs: list[str] = []
        logs: dict[str, Any] = {"steps": []}

        for step_index, model_name in enumerate(step_assignment):
            bundle = self.models[model_name]
            raw_output, cache, step_log = self._run_step(bundle, prompt, previous_cache)
            outputs.append(raw_output.strip())
            logs["steps"].append(step_log)
            prompt = prompt.after_output(raw_output, step_index + 1 if step_index < 2 else None)
            previous_cache = cache

        logs["final_prompt"] = prompt.text
        return outputs[-1], logs

    @staticmethod
    def _tokenize(
        tokenizer: Any,
        text: str,
        device: torch.device,
        *,
        add_special_tokens: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=add_special_tokens,
            )
        except TypeError:
            encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask.to(device)

    @staticmethod
    def _decode(tokenizer: Any, token_ids: torch.Tensor) -> str:
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _step_tokens(
        self,
        bundle: ModelBundle,
        prompt: PromptState,
        previous_cache: CacheState | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return logical full tokens plus separately encoded prefix/suffix."""
        if previous_cache is None:
            full_ids, full_mask = self._tokenize(
                bundle.tokenizer, prompt.text, device, add_special_tokens=True
            )
            return full_ids, full_mask, None, None

        suffix_text = prompt.suffix_text
        if suffix_text is None or not prompt.text.startswith(previous_cache.prompt_text):
            full_ids, full_mask = self._tokenize(
                bundle.tokenizer, prompt.text, device, add_special_tokens=True
            )
            return full_ids, full_mask, None, None
        suffix_ids, suffix_mask = self._tokenize(
            bundle.tokenizer, suffix_text, device, add_special_tokens=False
        )
        if previous_cache.model_name == bundle.name:
            prefix_ids = previous_cache.token_ids.to(device)
            prefix_mask = previous_cache.attention_mask
            if prefix_mask is None:
                prefix_mask = torch.ones_like(prefix_ids)
            else:
                prefix_mask = prefix_mask.to(device)
        else:
            prefix_ids, prefix_mask = self._tokenize(
                bundle.tokenizer,
                previous_cache.prompt_text,
                device,
                add_special_tokens=True,
            )
        return (
            torch.cat((prefix_ids, suffix_ids), dim=-1),
            torch.cat((prefix_mask, suffix_mask), dim=-1),
            prefix_ids,
            suffix_ids,
        )

    def _run_step(
        self,
        bundle: ModelBundle,
        prompt: PromptState,
        previous_cache: CacheState | None,
    ) -> tuple[str, CacheState, dict[str, Any]]:
        tokenizer = bundle.tokenizer
        model = bundle.model
        device = model_input_device(model)
        input_ids, attention_mask, prefix_ids, suffix_ids = self._step_tokens(
            bundle, prompt, previous_cache, device
        )
        same_model = previous_cache is not None and previous_cache.model_name == bundle.name
        cache_hit_tokens = 0
        cold_prefill = True
        adapter_used = False
        adapter_accepted = False
        adapter_seconds = 0.0
        quality_gate_seconds = 0.0
        fallback_reason: str | None = None
        cache_decision = "initial_cold_prefill" if previous_cache is None else "policy_disabled"
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "cache_position": torch.arange(input_ids.shape[-1], device=device),
        }

        if previous_cache is not None and self.cache_policy != "disabled":
            if same_model and prefix_ids is not None and suffix_ids is not None:
                cache_hit_tokens = int(prefix_ids.shape[-1])
                model_inputs["input_ids"] = suffix_ids
                model_inputs["past_key_values"] = previous_cache.past_key_values
                model_inputs["cache_position"] = torch.arange(
                    cache_hit_tokens,
                    cache_hit_tokens + suffix_ids.shape[-1],
                    device=device,
                )
                cold_prefill = False
                cache_decision = "same_model_hit"
            elif same_model:
                cache_decision = "same_model_prefix_mismatch"
                fallback_reason = "prompt_not_append_only"
            elif self.cache_policy == "same_model_only":
                cache_decision = "cross_model_not_allowed"
            elif self.cache_policy == "cross_model" and prefix_ids is not None and suffix_ids is not None:
                started = time.perf_counter()
                adapter_used = True
                try:
                    adapted = self.adapter.adapt(
                        previous_cache,
                        previous_cache.model_config,
                        model.config,
                        bundle.name,
                        target_token_ids=prefix_ids,
                        target_attention_mask=attention_mask[:, : prefix_ids.shape[-1]],
                    )
                    gate_started = time.perf_counter()
                    runtime_accepted = adapted.accepted
                    if (
                        adapted.degradation_score is not None
                        and adapted.degradation_score > self.degradation_threshold
                    ):
                        runtime_accepted = False
                        adapted.rejection_reason = "runtime_degradation_threshold_exceeded"
                    quality_gate_seconds = time.perf_counter() - gate_started
                    adapter_accepted = runtime_accepted
                    if runtime_accepted and adapted.past_key_values is not None:
                        cache_hit_tokens = adapted.transferred_tokens
                        model_inputs["input_ids"] = suffix_ids
                        model_inputs["past_key_values"] = adapted.past_key_values
                        model_inputs["cache_position"] = torch.arange(
                            cache_hit_tokens,
                            cache_hit_tokens + suffix_ids.shape[-1],
                            device=device,
                        )
                        cold_prefill = False
                        cache_decision = "cross_model_adapter_hit"
                    else:
                        cache_decision = "cross_model_quality_fallback"
                        fallback_reason = adapted.rejection_reason or "quality_gate_rejected"
                except AdapterError as error:
                    cache_decision = "cross_model_adapter_fallback"
                    fallback_reason = str(error)
                except Exception as error:  # noqa: BLE001 - mandatory safety boundary
                    cache_decision = "cross_model_adapter_fallback"
                    fallback_reason = f"adapter_runtime_error:{type(error).__name__}"
                finally:
                    adapter_seconds = time.perf_counter() - started
            else:
                cache_decision = "cross_model_prefix_mismatch"
                fallback_reason = "prompt_not_append_only"

        prefill = timed_forward(model, device, **model_inputs)
        outputs = prefill.output
        raw_text, generated_ids, completed_cache, decode_seconds = self._greedy_decode(
            model=model,
            tokenizer=tokenizer,
            first_outputs=outputs,
            max_new_tokens=int(self.config.get("max_new_tokens", 96)),
            attention_mask=attention_mask,
            initial_cache_position=int(input_ids.shape[-1]),
            device=device,
        )
        complete_token_ids = torch.cat((input_ids, generated_ids), dim=-1)
        complete_attention_mask = torch.cat(
            (attention_mask, torch.ones_like(generated_ids, device=device)), dim=-1
        )
        cache = CacheState(
            model_name=bundle.name,
            token_ids=complete_token_ids,
            prompt_text=f"{prompt.text}{raw_text}",
            past_key_values=completed_cache,
            model_config=model.config,
            attention_mask=complete_attention_mask,
            next_cache_position=int(complete_token_ids.shape[-1]),
        )
        log = {
            "model": bundle.name,
            "prefill_seconds": prefill.seconds,
            "decode_seconds": decode_seconds,
            "adapter_seconds": adapter_seconds,
            "quality_gate_seconds": quality_gate_seconds,
            "cache_hit_tokens": cache_hit_tokens,
            "prompt_tokens": int(input_ids.shape[-1]),
            "prefill_input_tokens": int(model_inputs["input_ids"].shape[-1]),
            "cache_policy": self.cache_policy,
            "cache_decision": cache_decision,
            "adapter_used": adapter_used,
            "adapter_accepted": adapter_accepted,
            "cold_prefill": cold_prefill,
            "fallback_reason": fallback_reason,
        }
        return raw_text, cache, log

    def _greedy_decode(
        self,
        model: Any,
        tokenizer: Any,
        first_outputs: Any,
        max_new_tokens: int,
        attention_mask: torch.Tensor,
        initial_cache_position: int,
        device: torch.device,
    ) -> tuple[str, torch.Tensor, Any, float]:
        if max_new_tokens <= 0:
            empty = torch.empty((1, 0), dtype=torch.long, device=first_outputs.logits.device)
            return "", empty, first_outputs.past_key_values, 0.0

        stop_ids = eos_token_ids(model, tokenizer)
        generated: list[torch.Tensor] = []
        outputs = first_outputs
        current_attention_mask = attention_mask
        completed_outputs = first_outputs
        synchronize(device)
        started = time.perf_counter()
        for token_index in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            current_attention_mask = torch.cat(
                (
                    current_attention_mask,
                    current_attention_mask.new_ones((current_attention_mask.shape[0], 1)),
                ),
                dim=-1,
            )
            with torch.inference_mode():
                completed_outputs = forward_with_cache(
                    model,
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    cache_position=torch.tensor(
                        [initial_cache_position + token_index], device=next_token.device
                    ),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
            if int(next_token[0, 0]) in stop_ids:
                break
            outputs = completed_outputs
        synchronize(device)
        decode_seconds = time.perf_counter() - started
        generated_ids = torch.cat(generated, dim=-1)
        raw_text = self._decode(tokenizer, generated_ids[0])
        return raw_text, generated_ids, completed_outputs.past_key_values, decode_seconds
