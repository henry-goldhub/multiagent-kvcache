from types import SimpleNamespace

import torch

from kvbridge.cache import is_exact_token_prefix
from kvbridge.models import ModelBundle
from kvbridge.pipeline import Pipeline


def test_exact_token_prefix():
    assert is_exact_token_prefix(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3]]))
    assert not is_exact_token_prefix(torch.tensor([[1, 3]]), torch.tensor([[1, 2, 3]]))


class CharacterTokenizer:
    """Small tokenizer whose encoding is guaranteed to preserve text prefixes."""

    eos_token_id = None

    def __call__(self, text, return_tensors):
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[ord(char) for char in text]]))

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in token_ids)


class RecordingModel(torch.nn.Module):
    """Toy causal LM that records whether a prefix cache was supplied."""

    def __init__(self):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace()
        self.calls = []

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        self.calls.append(
            {
                "input_length": input_ids.shape[-1],
                "had_cache": past_key_values is not None,
                "cache_length": 0 if past_key_values is None else past_key_values.shape[-1],
            }
        )
        cache = input_ids if past_key_values is None else torch.cat((past_key_values, input_ids), dim=-1)
        logits = torch.zeros((*input_ids.shape, 128), device=input_ids.device)
        logits[..., ord("A")] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def test_same_model_steps_prefill_only_appended_tokens():
    model = RecordingModel()
    pipeline = Pipeline(
        {"model_1": ModelBundle(model=model, tokenizer=CharacterTokenizer(), name="model_1")},
        {"max_new_tokens": 2},
    )

    _, logs = pipeline.run("One plus one", ["model_1", "model_1", "model_1"])

    assert logs["steps"][0]["cache_hit_tokens"] == 0
    assert logs["steps"][1]["cache_hit_tokens"] > 0
    assert logs["steps"][2]["cache_hit_tokens"] > 0
    # Calls 0, 3, and 6 begin the three step prefills. Step 2/3 pass a cache
    # and their input is strictly shorter than the reused prefix.
    for call_index in (3, 6):
        call = model.calls[call_index]
        assert call["had_cache"]
        assert call["input_length"] < call["cache_length"]


def test_disabled_policy_never_reuses_cache():
    model = RecordingModel()
    pipeline = Pipeline(
        {"model_1": ModelBundle(model=model, tokenizer=CharacterTokenizer(), name="model_1")},
        {"cache_policy": "disabled", "max_new_tokens": 1},
    )

    _, logs = pipeline.run("One plus one", ["model_1", "model_1", "model_1"])

    assert all(step["cache_hit_tokens"] == 0 for step in logs["steps"])
    assert all(step["cold_prefill"] for step in logs["steps"])
    assert logs["steps"][1]["cache_decision"] == "policy_disabled"


def test_same_model_only_refuses_model_switches():
    first = RecordingModel()
    second = RecordingModel()
    pipeline = Pipeline(
        {
            "first": ModelBundle(model=first, tokenizer=CharacterTokenizer(), name="first-model"),
            "second": ModelBundle(model=second, tokenizer=CharacterTokenizer(), name="second-model"),
        },
        {"cache_policy": "same_model_only", "max_new_tokens": 1},
    )

    _, logs = pipeline.run("One plus one", ["first", "second", "first"])

    assert [step["cache_decision"] for step in logs["steps"]] == [
        "initial_cold_prefill",
        "cross_model_not_allowed",
        "cross_model_not_allowed",
    ]


def test_cross_model_policy_falls_back_without_adapter():
    first = RecordingModel()
    second = RecordingModel()
    pipeline = Pipeline(
        {
            "first": ModelBundle(model=first, tokenizer=CharacterTokenizer(), name="first-model"),
            "second": ModelBundle(model=second, tokenizer=CharacterTokenizer(), name="second-model"),
        },
        {"cache_policy": "cross_model", "max_new_tokens": 1},
    )

    _, logs = pipeline.run("One plus one", ["first", "second", "first"])

    assert logs["steps"][1]["cache_decision"] == "cross_model_adapter_error"
    assert logs["steps"][1]["fallback_reason"] == "adapter_not_configured"
