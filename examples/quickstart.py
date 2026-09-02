"""Run KVBridge using offline, lightweight, or full model profiles."""

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvbridge import Pipeline, evaluate, report_to_markdown
from kvbridge.models import ModelBundle
from kvbridge.synthetic import synthetic_models

ASSIGNMENTS = [
    ["model_1", "model_1", "model_1"],
    ["model_2", "model_2", "model_2"],
    ["model_3", "model_3", "model_3"],
    ["model_1", "model_2", "model_3"],
]

MODEL_PROFILES = {
    "standin": {
        "model_1": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_2": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model_3": "distilbert/distilgpt2",
    },
    "full": {
        "model_1": "Qwen/Qwen2.5-7B-Instruct",
        "model_2": "mistralai/Mistral-7B-Instruct-v0.3",
        "model_3": "microsoft/Phi-3.5-mini-instruct",
    },
}


def load_bundle(alias: str, model_id: str, profile: str) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs: dict[str, Any] = {"device_map": "auto" if torch.cuda.is_available() else None}
    if profile == "full" and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer, name=alias)


def load_dataset_examples(profile: str, count: int) -> list[dict[str, str]]:
    if profile == "synthetic":
        return [
            {
                "question": f"Mia has {index + 1} apple and buys one more. How many apples?",
                "answer": "#### 11",
            }
            for index in range(count)
        ]
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(dataset[index]) for index in range(min(count, len(dataset)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("synthetic", "standin", "full"), default="synthetic")
    parser.add_argument("--num-examples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--cache-policy", choices=tuple(Pipeline.VALID_CACHE_POLICIES), default="cross_model")
    args = parser.parse_args()

    if args.profile == "synthetic":
        models = synthetic_models()
        adapter_config = {"type": "shape_only", "allow_uncalibrated": False}
    else:
        models = {
            alias: load_bundle(alias, model_id, args.profile)
            for alias, model_id in MODEL_PROFILES[args.profile].items()
        }
        adapter_config = {
            "type": "ridge",
            "artifact_dir": "artifacts/adapters",
            "ridge_lambda": 0.001,
        }

    config = {
        "cache_policy": args.cache_policy,
        "seed": 42,
        "device": "auto",
        "max_new_tokens": args.max_new_tokens,
        "step_assignments": ASSIGNMENTS,
        "adapter": adapter_config,
        "degradation": {"metric": "mean_kl", "threshold": 0.15, "probe_tokens": 8},
    }
    pipeline = Pipeline(models, config)
    report = evaluate(
        pipeline,
        load_dataset_examples(args.profile, args.num_examples),
        ASSIGNMENTS,
        config,
    )
    print(report_to_markdown(report))


if __name__ == "__main__":
    main()
