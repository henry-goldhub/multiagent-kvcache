"""GSM8K evaluation, aggregation, and portable report formatting."""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .pipeline import Pipeline


def extract_numeric_answer(text: str) -> str | None:
    tagged = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    numbers = tagged or re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def fixed_subset(
    dataset: Iterable[dict[str, Any]], size: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Select a stable subset without depending on dataset iteration order."""
    import random

    examples = list(dataset)
    if size < 0:
        raise ValueError("size must be non-negative")
    generator = random.Random(seed)
    indices = list(range(len(examples)))
    generator.shuffle(indices)
    return [examples[index] for index in indices[:size]]


def _validated_example(example: dict[str, Any]) -> tuple[str, str]:
    question = example.get("question", example.get("task_input"))
    answer = example.get("answer", example.get("gold_answer"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each evaluation example requires a non-empty question/task_input")
    if answer is None:
        raise ValueError("Each evaluation example requires answer/gold_answer")
    gold = extract_numeric_answer(str(answer))
    if gold is None:
        raise ValueError(f"Could not parse numeric gold answer for question: {question[:80]}")
    return question, gold


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _evaluate_assignment(
    pipeline: Pipeline, examples: list[dict[str, Any]], assignment: list[str]
) -> dict[str, Any]:
    correct = 0
    prefill: list[float] = []
    decode: list[float] = []
    adapter: list[float] = []
    quality: list[float] = []
    total: list[float] = []
    cache_hits = 0
    prompt_tokens = 0
    adapter_attempts = 0
    adapter_accepts = 0
    adapter_fallbacks = 0
    fallbacks = 0
    for example in examples:
        question, gold = _validated_example(example)
        result, logs = pipeline.run(question, assignment)
        predicted = extract_numeric_answer(result)
        correct += predicted is not None and predicted == gold
        example_total = 0.0
        for step in logs["steps"]:
            prefill.append(float(step["prefill_seconds"]))
            decode.append(float(step["decode_seconds"]))
            adapter.append(float(step["adapter_seconds"]))
            quality.append(float(step["quality_gate_seconds"]))
            example_total += sum(
                float(step[key])
                for key in (
                    "prefill_seconds",
                    "decode_seconds",
                    "adapter_seconds",
                    "quality_gate_seconds",
                )
            )
            cache_hits += int(step["cache_hit_tokens"])
            prompt_tokens += int(step["prompt_tokens"])
            adapter_attempts += int(step["adapter_used"])
            adapter_accepts += int(step["adapter_accepted"])
            adapter_fallbacks += int(step["adapter_used"] and not step["adapter_accepted"])
            fallbacks += int(step["fallback_reason"] is not None)
        total.append(example_total)
    step_count = len(examples) * len(assignment)
    return {
        "examples": len(examples),
        "accuracy": correct / len(examples) if examples else 0.0,
        "latency_seconds": {
            "prefill_mean": _mean(prefill),
            "prefill_p50": _median(prefill),
            "decode_mean": _mean(decode),
            "decode_p50": _median(decode),
            "adapter_mean": _mean(adapter),
            "adapter_p50": _median(adapter),
            "quality_gate_mean": _mean(quality),
            "total_mean": _mean(total),
            "total_p50": _median(total),
        },
        "cache_hit_tokens": cache_hits,
        "cache_hit_rate": cache_hits / prompt_tokens if prompt_tokens else 0.0,
        "adapter_attempt_rate": adapter_attempts / step_count if step_count else 0.0,
        "adapter_accept_rate": adapter_accepts / adapter_attempts if adapter_attempts else 0.0,
        "adapter_fallback_rate": (
            adapter_fallbacks / adapter_attempts if adapter_attempts else 0.0
        ),
        "fallback_rate": fallbacks / step_count if step_count else 0.0,
    }


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[dict[str, Any]],
    step_assignments: list[list[str]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare cross-step caching against cold-prefill-per-step baselines."""
    examples = list(dataset)
    for example in examples:
        _validated_example(example)
    effective_config = {**pipeline.config, **(config or {})}
    cached_pipeline = Pipeline(pipeline.models, effective_config)
    baseline_pipeline = Pipeline(
        pipeline.models, {**effective_config, "cache_policy": "disabled"}
    )
    report: dict[str, Any] = {
        "seed": int(effective_config.get("seed", 42)),
        "cache_policy": effective_config.get("cache_policy", "same_model_only"),
        "settings": {},
    }
    for assignment in step_assignments:
        if len(assignment) != 3:
            raise ValueError("Every evaluation assignment must contain exactly three models")
        key = " -> ".join(assignment)
        baseline = _evaluate_assignment(baseline_pipeline, examples, assignment)
        cached = _evaluate_assignment(cached_pipeline, examples, assignment)
        baseline_prefill = baseline["latency_seconds"]["prefill_mean"]
        cached_prefill = cached["latency_seconds"]["prefill_mean"]
        baseline_total = baseline["latency_seconds"]["total_mean"]
        cached_total = cached["latency_seconds"]["total_mean"]
        report["settings"][key] = {
            "baseline": baseline,
            "cached": cached,
            "speedup": {
                "prefill": baseline_prefill / cached_prefill if cached_prefill else None,
                "total": baseline_total / cached_total if cached_total else None,
            },
        }

    json_path = effective_config.get("report_json_path")
    markdown_path = effective_config.get("report_markdown_path")
    if json_path:
        save_report(report, json_path)
    if markdown_path:
        path = Path(markdown_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_to_markdown(report), encoding="utf-8")
    return report


def save_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def report_to_markdown(report: dict[str, Any]) -> str:
    headers = (
        "Setting",
        "Accuracy",
        "Prefill mean (s)",
        "Decode mean (s)",
        "Adapter mean (s)",
        "Cache-hit rate",
        "Adapter attempt rate",
        "Adapter accept rate",
        "Adapter fallback rate",
        "Prefill speedup",
        "Total speedup",
    )
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    for setting, values in report["settings"].items():
        cached = values["cached"]
        latency = cached["latency_seconds"]
        speedup = values["speedup"]["prefill"]
        total_speedup = values["speedup"]["total"]
        lines.append(
            "| "
            + " | ".join(
                (
                    setting,
                    f"{cached['accuracy']:.3f}",
                    f"{latency['prefill_mean']:.4f}",
                    f"{latency['decode_mean']:.4f}",
                    f"{latency['adapter_mean']:.4f}",
                    f"{cached['cache_hit_rate']:.3f}",
                    f"{cached['adapter_attempt_rate']:.3f}",
                    f"{cached['adapter_accept_rate']:.3f}",
                    f"{cached['adapter_fallback_rate']:.3f}",
                    "n/a" if speedup is None else f"{speedup:.2f}x",
                    "n/a" if total_speedup is None else f"{total_speedup:.2f}x",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
