"""Calibration quality metrics and persisted degradation decisions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .adapter import PairProjection


@dataclass(frozen=True)
class QualityResult:
    metric: str
    score: float
    threshold: float
    accepted: bool


def mean_kl_divergence(cold_logits: torch.Tensor, adapted_logits: torch.Tensor) -> float:
    if cold_logits.shape != adapted_logits.shape:
        raise ValueError("cold and adapted logits must have identical shapes")
    cold_log_probs = F.log_softmax(cold_logits.float(), dim=-1)
    adapted_log_probs = F.log_softmax(adapted_logits.float(), dim=-1)
    cold_probs = cold_log_probs.exp()
    kl = (cold_probs * (cold_log_probs - adapted_log_probs)).sum(dim=-1)
    return float(kl.mean().item())


def apply_quality_gate(
    projection: PairProjection,
    cold_logits: torch.Tensor,
    adapted_logits: torch.Tensor,
    *,
    threshold: float = 0.15,
    validation_examples: int = 0,
) -> QualityResult:
    score = mean_kl_divergence(cold_logits, adapted_logits)
    accepted = score <= threshold
    projection.metadata.degradation_metric = "mean_kl"
    projection.metadata.degradation_score = score
    projection.metadata.degradation_threshold = threshold
    projection.metadata.accepted = accepted
    projection.metadata.validation_examples = validation_examples
    return QualityResult("mean_kl", score, threshold, accepted)
