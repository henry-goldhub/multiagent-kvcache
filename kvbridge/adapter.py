"""Experimental interfaces for adapting KV caches between models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .cache import CacheLayout, CacheState, inspect_cache


@dataclass
class AdaptedCache:
    """The adapter result plus metadata needed for quality-gate logging."""

    past_key_values: Any
    source_model: str
    target_model: str
    transferred_tokens: int


class KVAdapter(ABC):
    """Adapt a source model cache to a target model cache shape."""

    @abstractmethod
    def adapt(
        self,
        source_cache: CacheState,
        source_config: Any,
        target_config: Any,
        target_model_name: str,
    ) -> AdaptedCache:
        """Return a target-compatible cache or raise if adaptation is unsupported."""

    @staticmethod
    def inspect_source(source_cache: CacheState) -> CacheLayout:
        """Inspect and validate a source cache before adapting it."""
        return inspect_cache(source_cache.past_key_values, source_cache.model_config)


class UnsupportedAdapter(KVAdapter):
    """Safe default adapter: intentionally refuses cross-model transfer."""

    def adapt(
        self,
        source_cache: CacheState,
        source_config: Any,
        target_config: Any,
        target_model_name: str,
    ) -> AdaptedCache:
        raise NotImplementedError(
            "Cross-model cache adaptation is not configured. "
            "Use a concrete KVAdapter or allow the cold-prefill fallback."
        )
