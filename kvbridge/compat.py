"""Compatibility helpers for Hugging Face causal language models."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TimedOutput:
    output: Any
    seconds: float


def model_input_device(model: Any) -> torch.device:
    """Return the device owning input embeddings, including sharded models."""
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if get_embeddings is not None:
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and weight.device.type != "meta":
            return weight.device
    device = getattr(model, "device", None)
    if device is not None and torch.device(device).type != "meta":
        return torch.device(device)
    return next(model.parameters()).device


def supports_forward_argument(model: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def forward_with_cache(model: Any, **kwargs: Any) -> Any:
    """Call a model while dropping optional kwargs unsupported by its API."""
    for name in ("cache_position", "position_ids"):
        if name in kwargs and not supports_forward_argument(model, name):
            kwargs.pop(name)
    return model(**kwargs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_forward(model: Any, device: torch.device, **kwargs: Any) -> TimedOutput:
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = forward_with_cache(model, **kwargs)
    synchronize(device)
    return TimedOutput(output=output, seconds=time.perf_counter() - started)


def eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Collect scalar/list EOS IDs from generation config and tokenizer."""
    candidates = [
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ]
    result: set[int] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, int):
            result.add(candidate)
        else:
            result.update(int(value) for value in candidate)
    return result
