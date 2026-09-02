"""Cross-model KV-cache adapters and calibrated projection artifacts."""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .cache import (
    CacheLayout,
    CacheState,
    LayerKV,
    config_cache_dimensions,
    dynamic_cache_from_layers,
    inspect_cache,
    legacy_cache_from_layers,
    normalize_cache,
)


class AdapterError(RuntimeError):
    """Base class for safe, logged cross-model adaptation failures."""


class AdapterUnavailable(AdapterError):
    """No usable calibrated artifact exists for a model pair."""


@dataclass
class AdaptedCache:
    past_key_values: Any | None
    source_model: str
    target_model: str
    transferred_tokens: int
    target_token_ids: torch.Tensor
    target_attention_mask: torch.Tensor
    layer_mapping: tuple[int, ...]
    source_layout: CacheLayout
    target_layout: CacheLayout | None
    accepted: bool
    degradation_score: float | None = None
    rejection_reason: str | None = None


class KVAdapter(ABC):
    @abstractmethod
    def adapt(
        self,
        source_cache: CacheState,
        source_config: Any,
        target_config: Any,
        target_model_name: str,
        *,
        target_token_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> AdaptedCache:
        """Return target-shaped cache metadata; rejected results trigger fallback."""

    @staticmethod
    def inspect_source(source_cache: CacheState) -> CacheLayout:
        return inspect_cache(source_cache.past_key_values, source_cache.model_config)


class UnsupportedAdapter(KVAdapter):
    def adapt(self, *args: Any, **kwargs: Any) -> AdaptedCache:
        raise AdapterUnavailable("adapter_not_configured")


def normalized_layer_mapping(source_layers: int, target_layers: int) -> tuple[int, ...]:
    if source_layers <= 0 or target_layers <= 0:
        raise ValueError("source_layers and target_layers must be positive")
    if target_layers == 1:
        return (0,)
    return tuple(
        round(index * (source_layers - 1) / (target_layers - 1))
        for index in range(target_layers)
    )


def resize_kv_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Pool proportional groups when shrinking and repeat proportionally when growing."""
    source_heads = int(tensor.shape[1])
    if target_heads <= 0:
        raise ValueError("target_heads must be positive")
    if source_heads == target_heads:
        return tensor.clone()
    if source_heads < target_heads:
        indices = torch.div(
            torch.arange(target_heads, device=tensor.device) * source_heads,
            target_heads,
            rounding_mode="floor",
        )
        return tensor.index_select(1, indices)
    pooled = []
    for target_index in range(target_heads):
        start = math.floor(target_index * source_heads / target_heads)
        end = math.floor((target_index + 1) * source_heads / target_heads)
        pooled.append(tensor[:, start : max(end, start + 1)].mean(dim=1, keepdim=True))
    return torch.cat(pooled, dim=1)


def resize_sequence(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if tensor.shape[-2] == target_tokens:
        return tensor.clone()
    original_dtype = tensor.dtype
    batch, heads, _, head_dim = tensor.shape
    values = tensor.float().permute(0, 1, 3, 2).reshape(batch * heads * head_dim, 1, -1)
    values = F.interpolate(values, size=target_tokens, mode="linear", align_corners=False)
    return (
        values.reshape(batch, heads, head_dim, target_tokens)
        .permute(0, 1, 3, 2)
        .to(original_dtype)
    )


def resize_head_dim(tensor: torch.Tensor, target_head_dim: int) -> torch.Tensor:
    if target_head_dim <= 0:
        raise ValueError("target_head_dim must be positive")
    if tensor.shape[-1] == target_head_dim:
        return tensor.clone()
    original_dtype = tensor.dtype
    flat = tensor.float().reshape(-1, 1, tensor.shape[-1])
    flat = F.interpolate(flat, size=target_head_dim, mode="linear", align_corners=False)
    return flat.reshape(*tensor.shape[:-1], target_head_dim).to(original_dtype)


def _prepare_base_tensor(
    tensor: torch.Tensor, target_heads: int, target_tokens: int
) -> torch.Tensor:
    return resize_sequence(resize_kv_heads(tensor, target_heads), target_tokens)


class ShapeOnlyKVAdapter(KVAdapter):
    """Deterministic shape baseline; rejected unless explicitly allowed."""

    def __init__(self, allow_uncalibrated: bool = False):
        self.allow_uncalibrated = allow_uncalibrated

    def _target_layers(
        self,
        source_layers: list[LayerKV],
        target_config: Any,
        target_tokens: int,
    ) -> tuple[list[LayerKV], tuple[int, ...]]:
        dimensions = config_cache_dimensions(target_config)
        mapping = normalized_layer_mapping(len(source_layers), dimensions["num_layers"])
        target_layers: list[LayerKV] = []
        for source_index in mapping:
            source = source_layers[source_index]
            key = _prepare_base_tensor(
                source.key, dimensions["num_key_value_heads"], target_tokens
            )
            value = _prepare_base_tensor(
                source.value, dimensions["num_key_value_heads"], target_tokens
            )
            target_layers.append(
                LayerKV(
                    key=resize_head_dim(key, dimensions["head_dim"]),
                    value=resize_head_dim(value, dimensions["head_dim"]),
                )
            )
        return target_layers, mapping

    def adapt(
        self,
        source_cache: CacheState,
        source_config: Any,
        target_config: Any,
        target_model_name: str,
        *,
        target_token_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> AdaptedCache:
        source_layout = inspect_cache(source_cache.past_key_values, source_config)
        layers, mapping = self._target_layers(
            normalize_cache(source_cache.past_key_values),
            target_config,
            int(target_token_ids.shape[-1]),
        )
        layers = [
            LayerKV(
                key=layer.key.to(target_token_ids.device),
                value=layer.value.to(target_token_ids.device),
            )
            for layer in layers
        ]
        try:
            target_cache = dynamic_cache_from_layers(layers, target_config)
            target_layout = inspect_cache(target_cache, target_config)
        except (TypeError, ValueError, AttributeError):
            target_cache = legacy_cache_from_layers(layers)
            target_layout = inspect_cache(target_cache, target_config)
        accepted = self.allow_uncalibrated
        return AdaptedCache(
            past_key_values=target_cache,
            source_model=source_cache.model_name,
            target_model=target_model_name,
            transferred_tokens=int(target_token_ids.shape[-1]),
            target_token_ids=target_token_ids,
            target_attention_mask=target_attention_mask,
            layer_mapping=mapping,
            source_layout=source_layout,
            target_layout=target_layout,
            accepted=accepted,
            rejection_reason=None if accepted else "uncalibrated_shape_adapter",
        )


@dataclass
class PairMetadata:
    source_model: str
    target_model: str
    ridge_lambda: float
    layer_mapping: list[int]
    accepted: bool = False
    degradation_metric: str = "mean_kl"
    degradation_score: float | None = None
    degradation_threshold: float = 0.15
    fit_examples: int = 0
    validation_examples: int = 0


@dataclass
class PairProjection:
    metadata: PairMetadata
    tensors: dict[str, torch.Tensor]


class RidgeKVAdapter(KVAdapter):
    """Registry of per-model-pair affine ridge projections."""

    def __init__(self, artifact_dir: str | Path = "artifacts/adapters", ridge_lambda: float = 1e-3):
        self.artifact_dir = Path(artifact_dir)
        self.ridge_lambda = float(ridge_lambda)
        self._pairs: dict[tuple[str, str], PairProjection] = {}

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    def _paths(self, source_model: str, target_model: str) -> tuple[Path, Path]:
        stem = f"{self._safe_name(source_model)}__to__{self._safe_name(target_model)}"
        return self.artifact_dir / f"{stem}.safetensors", self.artifact_dir / f"{stem}.json"

    def register(self, projection: PairProjection) -> None:
        key = (projection.metadata.source_model, projection.metadata.target_model)
        self._pairs[key] = projection

    def save_pair(self, projection: PairProjection) -> None:
        tensor_path, metadata_path = self._paths(
            projection.metadata.source_model, projection.metadata.target_model
        )
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in projection.tensors.items()},
            str(tensor_path),
        )
        metadata_path.write_text(json.dumps(asdict(projection.metadata), indent=2), encoding="utf-8")
        self.register(projection)

    def load_pair(self, source_model: str, target_model: str) -> PairProjection:
        key = (source_model, target_model)
        if key in self._pairs:
            return self._pairs[key]
        tensor_path, metadata_path = self._paths(source_model, target_model)
        if not tensor_path.exists() or not metadata_path.exists():
            raise AdapterUnavailable("missing_calibration")
        try:
            metadata = PairMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
            projection = PairProjection(metadata=metadata, tensors=load_file(str(tensor_path)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise AdapterUnavailable("malformed_calibration") from error
        if (metadata.source_model, metadata.target_model) != key:
            raise AdapterUnavailable("calibration_pair_mismatch")
        self.register(projection)
        return projection

    @staticmethod
    def _ridge_fit(
        x: torch.Tensor, y: torch.Tensor, ridge_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        y = y.float()
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        augmented = torch.cat((x, ones), dim=1)
        identity = torch.eye(augmented.shape[1], dtype=x.dtype, device=x.device)
        identity[-1, -1] = 0.0
        coefficients = torch.linalg.solve(
            augmented.T @ augmented + ridge_lambda * identity,
            augmented.T @ y,
        )
        return coefficients[:-1], coefficients[-1]

    def fit_pair(
        self,
        source_samples: Iterable[list[LayerKV]],
        target_samples: Iterable[list[LayerKV]],
        *,
        source_model: str,
        target_model: str,
        source_config: Any,
        target_config: Any,
    ) -> PairProjection:
        source_samples = list(source_samples)
        target_samples = list(target_samples)
        if not source_samples or len(source_samples) != len(target_samples):
            raise ValueError("source_samples and target_samples must have equal non-zero length")
        target_dimensions = config_cache_dimensions(target_config)
        mapping = normalized_layer_mapping(
            config_cache_dimensions(source_config)["num_layers"],
            target_dimensions["num_layers"],
        )
        tensors: dict[str, torch.Tensor] = {}
        for target_index, source_index in enumerate(mapping):
            for kind in ("key", "value"):
                xs: list[torch.Tensor] = []
                ys: list[torch.Tensor] = []
                for source_layers, target_layers in zip(source_samples, target_samples, strict=True):
                    source_tensor = getattr(source_layers[source_index], kind)
                    target_tensor = getattr(target_layers[target_index], kind)
                    base = _prepare_base_tensor(
                        source_tensor,
                        target_dimensions["num_key_value_heads"],
                        int(target_tensor.shape[-2]),
                    )
                    xs.append(base.reshape(-1, base.shape[-1]))
                    ys.append(target_tensor.reshape(-1, target_tensor.shape[-1]))
                weight, bias = self._ridge_fit(torch.cat(xs), torch.cat(ys), self.ridge_lambda)
                tensors[f"layer.{target_index}.{kind}.weight"] = weight.cpu()
                tensors[f"layer.{target_index}.{kind}.bias"] = bias.cpu()
        projection = PairProjection(
            metadata=PairMetadata(
                source_model=source_model,
                target_model=target_model,
                ridge_lambda=self.ridge_lambda,
                layer_mapping=list(mapping),
                fit_examples=len(source_samples),
            ),
            tensors=tensors,
        )
        self.register(projection)
        return projection

    def adapt(
        self,
        source_cache: CacheState,
        source_config: Any,
        target_config: Any,
        target_model_name: str,
        *,
        target_token_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> AdaptedCache:
        projection = self.load_pair(source_cache.model_name, target_model_name)
        metadata = projection.metadata
        source_layout = inspect_cache(source_cache.past_key_values, source_config)
        mapping = tuple(metadata.layer_mapping)
        if not metadata.accepted:
            return AdaptedCache(
                past_key_values=None,
                source_model=source_cache.model_name,
                target_model=target_model_name,
                transferred_tokens=0,
                target_token_ids=target_token_ids,
                target_attention_mask=target_attention_mask,
                layer_mapping=mapping,
                source_layout=source_layout,
                target_layout=None,
                accepted=False,
                degradation_score=metadata.degradation_score,
                rejection_reason="degradation_threshold_exceeded",
            )
        source_layers = normalize_cache(source_cache.past_key_values)
        dimensions = config_cache_dimensions(target_config)
        target_tokens = int(target_token_ids.shape[-1])
        layers: list[LayerKV] = []
        for target_index, source_index in enumerate(mapping):
            converted: dict[str, torch.Tensor] = {}
            for kind in ("key", "value"):
                source_tensor = getattr(source_layers[source_index], kind)
                base = _prepare_base_tensor(
                    source_tensor, dimensions["num_key_value_heads"], target_tokens
                ).to(target_token_ids.device)
                weight = projection.tensors[f"layer.{target_index}.{kind}.weight"].to(
                    device=base.device, dtype=torch.float32
                )
                bias = projection.tensors[f"layer.{target_index}.{kind}.bias"].to(
                    device=base.device, dtype=torch.float32
                )
                converted[kind] = (base.float() @ weight + bias).to(base.dtype)
            layers.append(LayerKV(key=converted["key"], value=converted["value"]))
        target_cache = dynamic_cache_from_layers(layers, target_config)
        target_layout = inspect_cache(target_cache, target_config)
        return AdaptedCache(
            past_key_values=target_cache,
            source_model=source_cache.model_name,
            target_model=target_model_name,
            transferred_tokens=target_tokens,
            target_token_ids=target_token_ids,
            target_attention_mask=target_attention_mask,
            layer_mapping=mapping,
            source_layout=source_layout,
            target_layout=target_layout,
            accepted=True,
            degradation_score=metadata.degradation_score,
        )


def build_adapter(config: Any) -> KVAdapter:
    if isinstance(config, KVAdapter):
        return config
    if config is None:
        return UnsupportedAdapter()
    if not isinstance(config, dict):
        raise TypeError("adapter must be a KVAdapter or configuration dictionary")
    adapter_type = config.get("type", "ridge")
    if adapter_type == "ridge":
        return RidgeKVAdapter(
            artifact_dir=config.get("artifact_dir", "artifacts/adapters"),
            ridge_lambda=float(config.get("ridge_lambda", 1e-3)),
        )
    if adapter_type == "shape_only":
        return ShapeOnlyKVAdapter(bool(config.get("allow_uncalibrated", False)))
    if adapter_type == "unsupported":
        return UnsupportedAdapter()
    raise ValueError(f"Unknown adapter type: {adapter_type}")
