"""Dataset evaluation and report aggregation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .pipeline import Pipeline


def extract_numeric_answer(text: str) -> str | None:
    """Extract GSM8K's preferred #### answer, otherwise the last number."""
    tagged = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    numbers = tagged or re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[dict[str, Any]],
    step_assignments: list[list[str]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare the configured policy against an identical no-cache baseline."""
    examples = list(dataset)
    effective_config = {**pipeline.config, **(config or {})}
    cached_pipeline = Pipeline(pipeline.models, effective_config)
    baseline_pipeline = Pipeline(
        pipeline.models, {**effective_config, "cache_policy": "disabled"}
    )
    report: dict[str, Any] = {"settings": {}}
    for assignment in step_assignments:
        key = " -> ".join(assignment)
        baseline = _evaluate_assignment(baseline_pipeline, examples, assignment)
        cached = _evaluate_assignment(cached_pipeline, examples, assignment)
        cached_prefill = cached["prefill_seconds"]
        report["settings"][key] = {
            "baseline": baseline,
            "cached": cached,
            "speedup": {
                "prefill_speedup": (
                    baseline["prefill_seconds"] / cached_prefill if cached_prefill else None
                )
            },
        }
    return report


def _evaluate_assignment(
    pipeline: Pipeline, examples: list[dict[str, Any]], assignment: list[str]
) -> dict[str, Any]:
    totals: defaultdict[str, float] = defaultdict(float)
    correct = 0
    for example in examples:
        question = example.get("question", example.get("task_input"))
        gold = extract_numeric_answer(str(example.get("answer", example.get("gold_answer", ""))))
        result, logs = pipeline.run(question, assignment)
        correct += extract_numeric_answer(result) == gold
        for step in logs["steps"]:
            for key in (
                "prefill_seconds",
                "decode_seconds",
                "adapter_seconds",
                "quality_gate_seconds",
                "cache_hit_tokens",
                "prompt_tokens",
            ):
                totals[key] += float(step[key])
    prompt_tokens = totals["prompt_tokens"]
    return {
        "accuracy": correct / len(examples) if examples else 0.0,
        "prefill_seconds": totals["prefill_seconds"],
        "decode_seconds": totals["decode_seconds"],
        "adapter_seconds": totals["adapter_seconds"],
        "quality_gate_seconds": totals["quality_gate_seconds"],
        "cache_hit_tokens": int(totals["cache_hit_tokens"]),
        "cache_hit_rate": totals["cache_hit_tokens"] / prompt_tokens if prompt_tokens else 0.0,
    }
