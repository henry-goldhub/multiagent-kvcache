import pytest

from kvbridge import Pipeline, evaluate, report_to_markdown
from kvbridge.evaluation import extract_numeric_answer, fixed_subset
from kvbridge.synthetic import synthetic_models


def test_extracts_gsm8k_tagged_answer():
    assert extract_numeric_answer("Work shown here. #### 1,024") == "1024"


def test_fixed_subset_is_reproducible():
    examples = [{"question": str(index), "answer": "#### 1"} for index in range(10)]
    assert fixed_subset(examples, 4, 7) == fixed_subset(examples, 4, 7)


def test_evaluate_rejects_missing_or_unparseable_gold():
    pipeline = Pipeline(synthetic_models(), {"max_new_tokens": 1})
    with pytest.raises(ValueError):
        evaluate(pipeline, [{"question": "test"}], [["model_1"] * 3])
    with pytest.raises(ValueError):
        evaluate(pipeline, [{"question": "test", "answer": "unknown"}], [["model_1"] * 3])


def test_offline_end_to_end_runs_all_four_settings():
    models = synthetic_models()
    assignments = [
        ["model_1"] * 3,
        ["model_2"] * 3,
        ["model_3"] * 3,
        ["model_1", "model_2", "model_3"],
    ]
    config = {
        "cache_policy": "cross_model",
        "max_new_tokens": 2,
        "adapter": {"type": "shape_only", "allow_uncalibrated": False},
    }
    report = evaluate(
        Pipeline(models, config),
        [{"question": "One plus one?", "answer": "#### 11"}],
        assignments,
        config,
    )
    assert len(report["settings"]) == 4
    mixed = report["settings"]["model_1 -> model_2 -> model_3"]["cached"]
    assert mixed["fallback_rate"] > 0
    assert "Prefill speedup" in report_to_markdown(report)
