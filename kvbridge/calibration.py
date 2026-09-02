"""Offline ridge fitting and held-out degradation-gate orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .adapter import PairProjection, RidgeKVAdapter
from .cache import CacheState, LayerKV, legacy_cache_from_layers, normalize_cache
from .compat import forward_with_cache, model_input_device
from .evaluation import fixed_subset
from .models import ModelBundle
from .quality import QualityResult, apply_quality_gate


@dataclass(frozen=True)
class CalibrationSplit:
    fit: list[dict[str, Any]]
    validation: list[dict[str, Any]]


def calibration_split(
    dataset: list[dict[str, Any]], fit_size: int = 16, validation_size: int = 8, seed: int = 42
) -> CalibrationSplit:
    selected = fixed_subset(dataset, fit_size + validation_size, seed)
    if len(selected) < fit_size + validation_size:
        raise ValueError("Calibration dataset is smaller than the requested fit/validation split")
    return CalibrationSplit(selected[:fit_size], selected[fit_size:])


def calibrate_pair(
    adapter: RidgeKVAdapter,
    source_samples: list[list[LayerKV]],
    target_samples: list[list[LayerKV]],
    cold_validation_logits: torch.Tensor,
    adapted_validation_logits: torch.Tensor,
    *,
    source_model: str,
    target_model: str,
    source_config: Any,
    target_config: Any,
    threshold: float = 0.15,
    validation_examples: int = 8,
) -> tuple[PairProjection, QualityResult]:
    """Fit, gate, and persist one pair from externally collected cache/logit samples."""
    projection = adapter.fit_pair(
        source_samples,
        target_samples,
        source_model=source_model,
        target_model=target_model,
        source_config=source_config,
        target_config=target_config,
    )
    result = apply_quality_gate(
        projection,
        cold_validation_logits,
        adapted_validation_logits,
        threshold=threshold,
        validation_examples=validation_examples,
    )
    adapter.save_pair(projection)
    return projection, result


@dataclass
class CollectedCache:
    state: CacheState
    layers: list[LayerKV]


def _tokenize(bundle: ModelBundle, text: str, *, add_special_tokens: bool) -> tuple[torch.Tensor, torch.Tensor]:
    device = model_input_device(bundle.model)
    encoded = bundle.tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded.input_ids.to(device)
    attention_mask = getattr(encoded, "attention_mask", torch.ones_like(input_ids)).to(device)
    return input_ids, attention_mask


def collect_prefill(bundle: ModelBundle, text: str) -> CollectedCache:
    """Collect one full-prefix cache and move it to CPU for sequential model loading."""
    input_ids, attention_mask = _tokenize(bundle, text, add_special_tokens=True)
    device = input_ids.device
    with torch.inference_mode():
        outputs = forward_with_cache(
            bundle.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=torch.arange(input_ids.shape[-1], device=device),
            use_cache=True,
        )
    layers = [
        LayerKV(layer.key.detach().cpu().clone(), layer.value.detach().cpu().clone())
        for layer in normalize_cache(outputs.past_key_values)
    ]
    state = CacheState(
        model_name=bundle.name,
        token_ids=input_ids.detach().cpu(),
        prompt_text=text,
        past_key_values=legacy_cache_from_layers(layers),
        model_config=bundle.model.config,
        attention_mask=attention_mask.detach().cpu(),
        next_cache_position=int(input_ids.shape[-1]),
    )
    return CollectedCache(state, layers)


def calibrate_model_pair(
    adapter: RidgeKVAdapter,
    source_bundle: ModelBundle,
    target_bundle: ModelBundle,
    fit_texts: list[str],
    validation_prefixes: list[str],
    *,
    probe_text: str = "\nContinue:",
    probe_tokens: int = 8,
    threshold: float = 0.15,
) -> tuple[PairProjection, QualityResult]:
    """Collect caches, fit projections, compare adapted and cold target logits, and save."""
    source_fit = [collect_prefill(source_bundle, text).layers for text in fit_texts]
    target_fit = [collect_prefill(target_bundle, text).layers for text in fit_texts]
    projection = adapter.fit_pair(
        source_fit,
        target_fit,
        source_model=source_bundle.name,
        target_model=target_bundle.name,
        source_config=source_bundle.model.config,
        target_config=target_bundle.model.config,
    )
    # Permit the new in-memory projection only for held-out measurement. The
    # gate below immediately replaces this provisional status.
    projection.metadata.accepted = True
    cold_logits: list[torch.Tensor] = []
    adapted_logits: list[torch.Tensor] = []
    target_device = model_input_device(target_bundle.model)
    for prefix in validation_prefixes:
        source = collect_prefill(source_bundle, prefix).state
        target_prefix_ids, target_prefix_mask = _tokenize(
            target_bundle, prefix, add_special_tokens=True
        )
        probe_ids, probe_mask = _tokenize(
            target_bundle, probe_text, add_special_tokens=False
        )
        if probe_tokens <= 0:
            raise ValueError("probe_tokens must be positive")
        probe_ids = probe_ids[:, :probe_tokens]
        probe_mask = probe_mask[:, :probe_tokens]
        if probe_ids.shape[-1] == 0:
            raise ValueError("probe_text did not produce any target tokens")
        full_ids = torch.cat((target_prefix_ids, probe_ids), dim=-1)
        full_mask = torch.cat((target_prefix_mask, probe_mask), dim=-1)
        adapted = adapter.adapt(
            source,
            source_bundle.model.config,
            target_bundle.model.config,
            target_bundle.name,
            target_token_ids=target_prefix_ids,
            target_attention_mask=target_prefix_mask,
        )
        with torch.inference_mode():
            cold = forward_with_cache(
                target_bundle.model,
                input_ids=full_ids,
                attention_mask=full_mask,
                cache_position=torch.arange(full_ids.shape[-1], device=target_device),
                use_cache=True,
            )
            adapted_output = forward_with_cache(
                target_bundle.model,
                input_ids=probe_ids,
                attention_mask=full_mask,
                cache_position=torch.arange(
                    target_prefix_ids.shape[-1], full_ids.shape[-1], device=target_device
                ),
                past_key_values=adapted.past_key_values,
                use_cache=True,
            )
        cold_logits.append(cold.logits[:, -probe_ids.shape[-1] :].detach().cpu())
        adapted_logits.append(adapted_output.logits.detach().cpu())
    result = apply_quality_gate(
        projection,
        torch.cat(cold_logits, dim=1),
        torch.cat(adapted_logits, dim=1),
        threshold=threshold,
        validation_examples=len(validation_prefixes),
    )
    adapter.save_pair(projection)
    return projection, result
