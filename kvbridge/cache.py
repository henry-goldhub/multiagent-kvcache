"""Model-agnostic inspection and normalization of Hugging Face KV caches.

KVBridge operates on a canonical layout of ``[batch, kv_heads, tokens,
head_dim]`` for every layer. Hugging Face commonly exposes this data as either
the legacy ``tuple[(key, value), ...]`` format or a ``DynamicCache`` object.
Keeping format-specific handling here prevents the adapter from depending on a
particular Transformers release.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CacheState:
    """A cache generated for one exact model prompt token sequence."""

    model_name: str
    token_ids: Any
    prompt_text: str
    past_key_values: Any
    model_config: Any
    attention_mask: Any | None = None
    next_cache_position: int | None = None

    @property
    def token_count(self) -> int:
        """Return the cached sequence length for common tensor-like token IDs."""
        return int(self.token_ids.shape[-1])


def is_exact_token_prefix(prefix: Any, full: Any) -> bool:
    """True only when ``prefix`` is an exact prefix of ``full`` token IDs."""
    prefix_length = int(prefix.shape[-1])
    if prefix_length > int(full.shape[-1]):
        return False
    return bool((prefix == full[..., :prefix_length]).all().item())


@dataclass(frozen=True)
class CacheLayout:
    """Architecture and tensor dimensions relevant to a model KV cache."""

    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    sequence_lengths: tuple[int, ...]
    cache_format: str

    @property
    def sequence_length(self) -> int:
        """Largest valid per-layer sequence length (legacy convenience API)."""
        return max(self.sequence_lengths)


@dataclass
class LayerKV:
    """One layer of canonical key/value tensors.

    Both tensors have shape ``[batch, kv_heads, tokens, head_dim]``.
    """

    key: torch.Tensor
    value: torch.Tensor


def config_cache_dimensions(config: Any) -> dict[str, int]:
    """Extract attention dimensions across common Hugging Face model configs.

    Qwen, Mistral, and Phi use the first names in each lookup, while the
    alternatives make this utility useful for smaller stand-in architectures.
    """
    hidden_size = _required_config_value(config, "hidden_size", "n_embd", "d_model")
    attention_heads = _required_config_value(
        config, "num_attention_heads", "n_head", "num_heads"
    )
    kv_heads = _optional_config_value(
        config, "num_key_value_heads", "num_kv_heads", default=attention_heads
    )
    head_dim = _optional_config_value(
        config, "head_dim", default=hidden_size // attention_heads
    )
    layers = _required_config_value(config, "num_hidden_layers", "n_layer", "num_layers")
    return {
        "num_layers": layers,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
    }


def normalize_cache(cache: Any) -> list[LayerKV]:
    """Convert a legacy cache or DynamicCache-like object to canonical layers.

    The returned tensors are references to the original tensors, not copies.
    Adaptation code should clone them before applying in-place transformations.
    """
    layers = list(_iter_cache_layers(cache))
    normalized: list[LayerKV] = []
    for layer_index, layer in enumerate(layers):
        key, value = _unpack_layer(layer, layer_index)
        _validate_layer_tensors(key, value, layer_index)
        valid_length = _effective_sequence_length(cache, layer_index, int(key.shape[-2]))
        # Static caches expose their allocated capacity in tensor.shape. Slice
        # to valid positions before adaptation; sliding-window layers can also
        # have a shorter valid history than ordinary layers.
        key = key[..., :valid_length, :]
        value = value[..., :valid_length, :]
        normalized.append(LayerKV(key=key, value=value))
    if not normalized:
        raise ValueError("Cannot normalize an empty KV cache")
    return normalized


def inspect_cache(cache: Any, config: Any) -> CacheLayout:
    """Return architecture dimensions and validate cache/config consistency."""
    layers = normalize_cache(cache)
    dimensions = config_cache_dimensions(config)
    first = layers[0]
    kv_heads = int(first.key.shape[-3])
    head_dim = int(first.key.shape[-1])

    if len(layers) != dimensions["num_layers"]:
        raise ValueError(
            f"Cache has {len(layers)} layers but config declares {dimensions['num_layers']}"
        )
    if kv_heads != dimensions["num_key_value_heads"]:
        raise ValueError(
            f"Cache has {kv_heads} KV heads but config declares "
            f"{dimensions['num_key_value_heads']}"
        )
    if head_dim != dimensions["head_dim"]:
        raise ValueError(
            f"Cache head_dim is {head_dim} but config declares {dimensions['head_dim']}"
        )
    for layer_index, layer in enumerate(layers[1:], start=1):
        if layer.key.shape[0] != first.key.shape[0]:
            raise ValueError(f"Layer {layer_index} batch size differs from layer 0")
        if layer.key.shape[-3] != kv_heads:
            raise ValueError(f"Layer {layer_index} KV-head count differs from layer 0")
        if layer.key.shape[-1] != head_dim:
            raise ValueError(f"Layer {layer_index} head dimension differs from layer 0")

    return CacheLayout(
        **dimensions,
        sequence_lengths=tuple(int(layer.key.shape[-2]) for layer in layers),
        cache_format=_cache_format_name(cache),
    )


def legacy_cache_from_layers(layers: Iterable[LayerKV]) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build the portable legacy tuple form from canonical layer tensors.

    Modern Transformers versions can generally convert this form to a
    ``DynamicCache`` at the model boundary. A later adapter step can add an
    explicit version-specific DynamicCache constructor here if required.
    """
    return tuple((layer.key, layer.value) for layer in layers)


def dynamic_cache_from_layers(layers: Iterable[LayerKV], config: Any) -> Any:
    """Build a current Hugging Face ``DynamicCache`` from canonical layers.

    This helper is intentionally lazy-imported so cache-shape unit tests do not
    require a particular Transformers cache class at import time.
    """
    import inspect

    from transformers import DynamicCache

    parameters = inspect.signature(DynamicCache).parameters
    if "config" in parameters:
        try:
            cache = DynamicCache(config=config)
        except (AttributeError, TypeError, ValueError):
            # Minimal/custom model configs can describe cache dimensions without
            # implementing the full PreTrainedConfig protocol expected by 5.x.
            cache = DynamicCache()
    else:
        cache = DynamicCache()
    for layer_index, layer in enumerate(layers):
        cache.update(layer.key, layer.value, layer_index)
    return cache


def _iter_cache_layers(cache: Any) -> Iterable[Any]:
    """Yield layer entries from supported public cache representations."""
    if isinstance(cache, (tuple, list)):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    # Some DynamicCache versions expose a public iterable cache-layers API.
    if hasattr(cache, "layers"):
        return cache.layers
    raise TypeError(
        "Unsupported cache type. Expected a legacy tuple/list or a cache object "
        "with to_legacy_cache() or layers."
    )


def _unpack_layer(layer: Any, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(layer, (tuple, list)) and len(layer) >= 2:
        return layer[0], layer[1]
    if hasattr(layer, "keys") and hasattr(layer, "values"):
        return layer.keys, layer.values
    if hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
        return layer.key_cache, layer.value_cache
    raise TypeError(f"Cannot extract key/value tensors from cache layer {layer_index}")


def _validate_layer_tensors(key: Any, value: Any, layer_index: int) -> None:
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError(f"Cache layer {layer_index} key and value must be torch tensors")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            f"Cache layer {layer_index} tensors must be rank 4 "
            "[batch, kv_heads, tokens, head_dim]"
        )
    if tuple(key.shape) != tuple(value.shape):
        raise ValueError(f"Cache layer {layer_index} key and value shapes differ")


def _effective_sequence_length(cache: Any, layer_index: int, tensor_length: int) -> int:
    """Read valid tokens rather than static-cache capacity when possible."""
    get_seq_length = getattr(cache, "get_seq_length", None)
    if get_seq_length is None:
        return tensor_length
    try:
        length = get_seq_length(layer_index)
    except TypeError:
        length = get_seq_length()
    if isinstance(length, torch.Tensor):
        length = length.item()
    length = int(length)
    if not 0 <= length <= tensor_length:
        raise ValueError(
            f"Cache layer {layer_index} reports invalid sequence length {length}; "
            f"tensor capacity is {tensor_length}"
        )
    return length


def _cache_format_name(cache: Any) -> str:
    if isinstance(cache, tuple):
        return "legacy_tuple"
    if isinstance(cache, list):
        return "legacy_list"
    return type(cache).__name__


def _required_config_value(config: Any, *names: str) -> int:
    missing = object()
    value = _optional_config_value(config, *names, default=missing)
    if value is missing:
        formatted = ", ".join(names)
        raise ValueError(f"Model config has none of the required fields: {formatted}")
    return int(value)


def _optional_config_value(config: Any, *names: str, default: Any) -> Any:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default
