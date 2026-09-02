"""Minimal local smoke test. Install optional model dependencies before running."""

from transformers import AutoModelForCausalLM, AutoTokenizer

from kvbridge import Pipeline, evaluate
from kvbridge.models import ModelBundle


def load_bundle(name: str) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return ModelBundle(model=model, tokenizer=tokenizer, name=name)


if __name__ == "__main__":
    # Use a tiny model for API validation. Replace with the three required models for final runs.
    name = "sshleifer/tiny-gpt2"
    bundle = load_bundle(name)
    pipeline = Pipeline({"model_1": bundle}, {"max_new_tokens": 16, "seed": 42})
    dataset = [{"question": "Mia has 2 apples and buys 3 more. How many apples?", "answer": "#### 5"}]
    report = evaluate(pipeline, dataset, [["model_1", "model_1", "model_1"]])
    print(report)
