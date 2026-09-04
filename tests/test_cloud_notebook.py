"""Validate the notebook without installing Jupyter or downloading model weights."""

import gc
import json
import random
import shutil
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import pytest
import torch

from kvbridge import Pipeline, evaluate, report_to_markdown
from kvbridge.adapter import RidgeKVAdapter
from kvbridge.calibration import calibrate_model_pair
from kvbridge.evaluation import save_report
from kvbridge.prompts import PromptState
from kvbridge.synthetic import synthetic_models

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "examples" / "cloud_run.ipynb"


def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def cell_source(cell_id):
    cell = next(cell for cell in notebook()["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


def helpers():
    namespace = {
        "Path": Path,
        "json": json,
        "gc": gc,
        "random": random,
        "shutil": shutil,
        "time": time,
        "uuid": uuid,
        "torch": torch,
        "asdict": asdict,
        "Pipeline": Pipeline,
        "RidgeKVAdapter": RidgeKVAdapter,
        "calibrate_model_pair": calibrate_model_pair,
        "PromptState": PromptState,
        "evaluate": evaluate,
        "report_to_markdown": report_to_markdown,
        "save_report": save_report,
    }
    # Execute only this repository's trusted notebook cell, never remote content.
    exec(  # noqa: S102 - exercise the notebook's actual orchestration code
        compile(cell_source("workflow-helpers"), str(NOTEBOOK_PATH), "exec"), namespace
    )
    return namespace


def test_notebook_schema_syntax_and_safe_defaults():
    document = notebook()
    assert document["nbformat"] == 4
    assert document["nbformat_minor"] == 5
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    cell_ids = [cell["id"] for cell in document["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for cell in document["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            compile("".join(cell["source"]), f"notebook:{cell['id']}", "exec")
    settings = {}
    exec(  # noqa: S102 - validate the trusted local notebook's default settings
        compile(cell_source("settings"), "notebook:settings", "exec"), settings
    )
    assert settings["PROFILE"] == "standin"
    assert settings["NUM_EXAMPLES"] == 5
    assert settings["ENABLE_MODEL_DOWNLOADS"] is False
    assert settings["ENABLE_FULL_PROFILE"] is False
    assert cell_ids.index("cloud-permission-auth") < cell_ids.index("dataset-and-revisions")


def test_notebook_subset_selection_is_disjoint_and_repeatable():
    choose_rows = helpers()["choose_rows"]
    dataset = [{"question": str(index), "answer": "#### 1"} for index in range(30)]
    indices, rows = choose_rows(dataset, 24, 42)
    assert (indices, rows) == choose_rows(dataset, 24, 42)
    assert not set(indices[:16]) & set(indices[16:])
    with pytest.raises(ValueError):
        choose_rows(dataset, 31, 42)


def test_notebook_calibration_evaluation_fallback_and_export_offline(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        workflow = helpers()
        run_dir = tmp_path / "run"
        adapter_dir = run_dir / "adapters"
        adapter_dir.mkdir(parents=True)
        # A sibling cache must never end up in the downloadable results ZIP.
        (tmp_path / "hf-token.txt").write_text("not-an-export", encoding="utf-8")
        models = synthetic_models()
        rows = [
            {"question": "One plus one?", "answer": "#### 11"},
            {"question": "Two plus one?", "answer": "#### 11"},
        ]
        for source, target in (("model_1", "model_2"), ("model_2", "model_3")):
            result = workflow["calibrate_handoff"](
                models.__getitem__, source, target, rows, rows[:1],
                adapter_dir, 0.001, 0.15, 2,
            )
            assert result["quality"]["accepted"]
            assert result["probe_tokens"] == 2
        config = {
            "cache_policy": "cross_model",
            "seed": 42,
            "max_new_tokens": 2,
            "adapter": {"type": "ridge", "artifact_dir": str(adapter_dir)},
        }
        proof = workflow["prove_missing_artifact_fallback"](
            models, rows[0]["question"], config, run_dir
        )
        assert proof["logs"]["steps"][1]["fallback_reason"] == "missing_calibration"
        assignments = [
            ["model_1"] * 3, ["model_2"] * 3, ["model_3"] * 3,
            ["model_1", "model_2", "model_3"],
        ]
        report = workflow["evaluate_with_checkpoints"](
            models, rows[:1], assignments, config, run_dir
        )
        assert len(report["settings"]) == 4
        mixed = report["settings"]["model_1 -> model_2 -> model_3"]["cached"]
        assert mixed["adapter_accept_rate"] == 1.0
        status = json.loads((run_dir / "evaluation_status.json").read_text())
        assert status["complete"] and status["completed_assignments"] == 4
        assert "Prefill speedup" in (run_dir / "report.md").read_text()
        archive = workflow["archive_run"](run_dir)
        with ZipFile(archive) as exported:
            names = exported.namelist()
        assert "report.json" in names
        assert "forced_fallback.json" in names
        assert any(name.endswith(".safetensors") for name in names)
        assert "hf-token.txt" not in names
    finally:
        torch.set_num_threads(previous_threads)


def test_notebook_checkpoints_errors_without_claiming_completion(tmp_path):
    workflow = helpers()

    def failed_evaluation(*args, **kwargs):
        raise RuntimeError("simulated cloud forward failure")

    workflow["evaluate"] = failed_evaluation
    with pytest.raises(RuntimeError, match="simulated cloud forward failure"):
        workflow["evaluate_with_checkpoints"](
            synthetic_models(),
            [{"question": "One plus one?", "answer": "#### 11"}],
            [["model_1"] * 3],
            {"seed": 42, "cache_policy": "cross_model"},
            tmp_path,
        )
    status = json.loads((tmp_path / "evaluation_status.json").read_text())
    assert status["complete"] is False
    assert status["completed_assignments"] == 0
    assert status["error_type"] == "RuntimeError"
