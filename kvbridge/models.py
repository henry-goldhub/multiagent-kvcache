"""Model container types used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelBundle:
    """A causal language model and its matching tokenizer."""

    model: Any
    tokenizer: Any
    name: str
