"""The three-stage append-only inference pipeline."""

from __future__ import annotations

import random
import time
from typing import Any, ClassVar

import torch

from .adapter import KVAdapter, UnsupportedAdapter
from .cache import CacheState, is_exact_token_prefix
from .models import ModelBundle
from .prompts import append_step, build_initial_prompt


class Pipeline:
    """Run extract → plan → compute with same-model cache reuse where possible."""

    VALID_CACHE_POLICIES: ClassVar[set[str]] = {"disabled", "same_model_only", "cross_model"}

    def __init__(self, models: dict[str, ModelBundle], config: dict[str, Any] | None = None):
        self.models = models
        self.config = config or {}
        self.cache_policy = self.config.get("cache_policy", "same_model_only")
        if self.cache_policy not in self.VALID_CACHE_POLICIES:
            allowed = ", ".join(sorted(self.VALID_CACHE_POLICIES))
            raise ValueError(f"cache_policy must be one of: {allowed}")
        self.adapter: KVAdapter = self.config.get("adapter", UnsupportedAdapter())
        self.seed = int(self.config.get("seed", 42))

    def run(self, task_input: str, step_assignment: list[str]) -> tuple[str, dict[str, Any]]:
        if len(step_assignment) != 3:
            raise ValueError("step_assignment must contain exactly three model names")
        if any(name not in self.models for name in step_assignment):
            raise KeyError("step_assignment contains a model that was not supplied")

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        prompt = build_initial_prompt(task_input)
        previous_cache: CacheState | None = None
        outputs: list[str] = []
        logs: dict[str, Any] = {"steps": []}

        for step_index, model_name in enumerate(step_assignment):
            bundle = self.models[model_name]
            output, cache, step_log = self._run_step(bundle, prompt, previous_cache)
            outputs.append(output)
            logs["steps"].append(step_log)
            prompt = append_step(prompt, output, step_index + 1 if step_index < 2 else None)
            previous_cache = cache

        logs["final_prompt"] = prompt
        return outputs[-1], logs

    def _run_step(
        self,
        bundle: ModelBundle,
        prompt: str,
        previous_cache: CacheState | None,
    ) -> tuple[str, CacheState, dict[str, Any]]:
        tokenizer = bundle.tokenizer
        model = bundle.model
        device = next(model.parameters()).device
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = getattr(encoded, "attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)
        use_same_model_cache = previous_cache is not None and previous_cache.model_name == bundle.name
        cache_hit_tokens = 0
        cold_prefill = True
        adapter_used = False
        adapter_seconds = 0.0
        quality_gate_seconds = 0.0
        fallback_reason: str | None = None
        cache_decision = "initial_cold_prefill" if previous_cache is None else ""
        model_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": True,
            "cache_position": torch.arange(input_ids.shape[-1], device=device),
        }

        if self.cache_policy == "disabled" and previous_cache is not None:
            cache_decision = "policy_disabled"
        elif use_same_model_cache and is_exact_token_prefix(previous_cache.token_ids, input_ids):
            cache_hit_tokens = previous_cache.token_count
            model_inputs["input_ids"] = input_ids[:, cache_hit_tokens:]
            model_inputs["past_key_values"] = previous_cache.past_key_values
            # For cached forwards, Hugging Face requires an attention mask
            # covering both past and newly supplied tokens. cache_position is
            # absolute and begins after the prior completed prompt/response.
            start_position = previous_cache.next_cache_position or cache_hit_tokens
            suffix_length = model_inputs["input_ids"].shape[-1]
            model_inputs["cache_position"] = torch.arange(
                start_position, start_position + suffix_length, device=device
            )
            cold_prefill = False
            cache_decision = "same_model_hit"
        elif use_same_model_cache:
            cache_decision = "same_model_prefix_mismatch"
        elif previous_cache is not None and self.cache_policy == "same_model_only":
            cache_decision = "cross_model_not_allowed"
        elif previous_cache is not None and self.cache_policy == "cross_model":
            started = time.perf_counter()
            try:
                self.adapter.adapt(
                    previous_cache, previous_cache.model_config, model.config, bundle.name
                )
                adapter_used = True
                # An adapted cache must first state which *target tokenizer*
                # positions it covers and pass a continuation quality gate.
                # Until that interface exists, safely use the cold baseline.
                cache_decision = "cross_model_quality_fallback"
                fallback_reason = "quality_gate_unavailable"
            except NotImplementedError:
                cache_decision = "cross_model_adapter_error"
                fallback_reason = "adapter_not_configured"
            adapter_seconds = time.perf_counter() - started

        prefill_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(**model_inputs)
        prefill_seconds = time.perf_counter() - prefill_started

        decode_started = time.perf_counter()
        text, generated_ids, completed_cache = self._greedy_decode(
            model=model,
            tokenizer=tokenizer,
            first_outputs=outputs,
            max_new_tokens=int(self.config.get("max_new_tokens", 96)),
            attention_mask=attention_mask,
            initial_cache_position=int(input_ids.shape[-1]),
        )
        decode_seconds = time.perf_counter() - decode_started
        # The cache must include the generated answer.  That makes the next
        # append-only prompt's token IDs start with ``cache.token_ids``.
        complete_token_ids = torch.cat((input_ids, generated_ids), dim=-1)
        cache = CacheState(
            model_name=bundle.name,
            token_ids=complete_token_ids,
            prompt_text=f"{prompt}{text}",
            past_key_values=completed_cache,
            model_config=model.config,
            attention_mask=torch.cat(
                (attention_mask, torch.ones_like(generated_ids, device=device)), dim=-1
            ),
            next_cache_position=int(input_ids.shape[-1] + generated_ids.shape[-1]),
        )
        log = {
            "model": bundle.name,
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "adapter_seconds": adapter_seconds,
            "quality_gate_seconds": quality_gate_seconds,
            "cache_hit_tokens": cache_hit_tokens,
            "prompt_tokens": int(input_ids.shape[-1]),
            "prefill_input_tokens": int(model_inputs["input_ids"].shape[-1]),
            "cache_policy": self.cache_policy,
            "cache_decision": cache_decision,
            "adapter_used": adapter_used,
            "cold_prefill": cold_prefill,
            "fallback_reason": fallback_reason,
        }
        return text, cache, log

    @staticmethod
    def _eos_token_ids(tokenizer: Any) -> set[int]:
        """Normalize Transformers' scalar/list EOS configuration."""
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return set(eos)

    def _greedy_decode(
        self,
        model: Any,
        tokenizer: Any,
        first_outputs: Any,
        max_new_tokens: int,
        attention_mask: torch.Tensor,
        initial_cache_position: int,
    ) -> tuple[str, torch.Tensor, Any]:
        """Greedily decode while retaining a cache for every emitted token.

        ``first_outputs`` comes from a full prefill (or a cached suffix
        prefill).  Its final logits predict the first generated token.  Each
        subsequent forward pass receives exactly one new token and the prior
        cache, so it never recomputes attention for the growing response.
        A final one-token forward appends the last emitted token to the cache;
        this is what makes that cache reusable by the following pipeline step.
        """
        if max_new_tokens <= 0:
            empty = torch.empty((1, 0), dtype=torch.long, device=first_outputs.logits.device)
            return "", empty, first_outputs.past_key_values

        eos_token_ids = self._eos_token_ids(tokenizer)
        generated: list[torch.Tensor] = []
        outputs = first_outputs
        current_attention_mask = attention_mask
        completed_outputs = first_outputs
        for token_index in range(max_new_tokens):
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            current_attention_mask = torch.cat(
                (current_attention_mask, current_attention_mask.new_ones((1, 1))), dim=-1
            )
            # Forward every selected token once: this both computes the next
            # logits and appends the selected token to the cache. The full
            # attention mask is required to cover past + current KV positions.
            with torch.inference_mode():
                completed_outputs = model(
                    input_ids=next_token,
                    attention_mask=current_attention_mask,
                    cache_position=torch.tensor(
                        [initial_cache_position + token_index], device=next_token.device
                    ),
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                )
            if int(next_token[0, 0]) in eos_token_ids:
                break
            outputs = completed_outputs

        generated_ids = torch.cat(generated, dim=-1)
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        return text, generated_ids, completed_outputs.past_key_values
