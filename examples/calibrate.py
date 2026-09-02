"""Fit and validate one pair-specific ridge KV adapter."""

import argparse

from quickstart import MODEL_PROFILES, load_bundle

from kvbridge.adapter import RidgeKVAdapter
from kvbridge.calibration import calibrate_model_pair, calibration_split
from kvbridge.prompts import PromptState
from kvbridge.synthetic import synthetic_models


def _records(profile: str, count: int) -> list[dict[str, str]]:
    if profile == "synthetic":
        return [
            {
                "question": f"Calibration arithmetic example {index}: what is {index} plus one?",
                "answer": f"#### {index + 1}",
            }
            for index in range(count)
        ]
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    return [dict(record) for record in dataset]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("synthetic", "standin", "full"), default="standin")
    parser.add_argument("--source", choices=("model_1", "model_2", "model_3"), required=True)
    parser.add_argument("--target", choices=("model_1", "model_2", "model_3"), required=True)
    parser.add_argument("--fit-examples", type=int, default=16)
    parser.add_argument("--validation-examples", type=int, default=8)
    parser.add_argument("--probe-tokens", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", default="artifacts/adapters")
    parser.add_argument("--ridge-lambda", type=float, default=0.001)
    args = parser.parse_args()

    if args.source == args.target:
        parser.error("--source and --target must identify different models")
    if args.profile == "synthetic":
        models = synthetic_models()
        source_bundle = models[args.source]
        target_bundle = models[args.target]
    else:
        profile = MODEL_PROFILES[args.profile]
        source_bundle = load_bundle(args.source, profile[args.source], args.profile)
        target_bundle = load_bundle(args.target, profile[args.target], args.profile)

    requested = args.fit_examples + args.validation_examples
    split = calibration_split(
        _records(args.profile, max(requested, 64)),
        fit_size=args.fit_examples,
        validation_size=args.validation_examples,
        seed=args.seed,
    )
    fit_texts = [PromptState.from_question(record["question"]).text for record in split.fit]
    validation_texts = [
        PromptState.from_question(record["question"]).text for record in split.validation
    ]
    adapter = RidgeKVAdapter(args.artifact_dir, ridge_lambda=args.ridge_lambda)
    projection, quality = calibrate_model_pair(
        adapter,
        source_bundle,
        target_bundle,
        fit_texts,
        validation_texts,
        probe_tokens=args.probe_tokens,
        threshold=args.threshold,
    )
    print(
        f"{args.source} -> {args.target}: accepted={quality.accepted}, "
        f"mean_kl={quality.score:.6f}, threshold={quality.threshold:.6f}, "
        f"fit_examples={projection.metadata.fit_examples}, "
        f"validation_examples={projection.metadata.validation_examples}"
    )


if __name__ == "__main__":
    main()
