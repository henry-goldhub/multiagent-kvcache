from types import SimpleNamespace

import torch

from kvbridge.adapter import AdaptedCache, KVAdapter
from kvbridge.cache import is_exact_token_prefix
from kvbridge.compat import forward_with_cache
from kvbridge.models import ModelBundle
from kvbridge.pipeline import Pipeline
from kvbridge.synthetic import synthetic_models


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


class NonCompositionalTokenizer:
    """Records special-token handling while deliberately ignoring text composition."""

    eos_token_id = None

    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        # A whole-string encode is intentionally unrelated to concatenated suffix
        # encodes, like a BPE tokenizer whose boundary merges can change.
        ids = [2, len(text) % 97] if add_special_tokens else [3, (len(text) * 7) % 97]
        return SimpleNamespace(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
        )

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        return "A"


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


def test_same_model_growth_tokenizes_initial_prompt_once_and_suffixes_without_bos():
    tokenizer = NonCompositionalTokenizer()
    pipeline = Pipeline(
        {"model_1": ModelBundle(model=RecordingModel(), tokenizer=tokenizer, name="model_1")},
        {"max_new_tokens": 1},
    )

    pipeline.run("Boundary-sensitive text", ["model_1", "model_1", "model_1"])

    assert [add_special_tokens for _, add_special_tokens in tokenizer.calls] == [
        True,
        False,
        False,
    ]


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


def test_cross_model_policy_defaults_to_ridge_and_falls_back_without_calibration():
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

    assert logs["steps"][1]["cache_decision"] == "cross_model_adapter_fallback"
    assert logs["steps"][1]["fallback_reason"] == "missing_calibration"
    assert logs["steps"][1]["adapter_used"]


def test_cross_model_success_forwards_only_the_unmatched_target_suffix():
    pipeline = Pipeline(
        synthetic_models(),
        {
            "cache_policy": "cross_model",
            "max_new_tokens": 1,
            "adapter": {"type": "shape_only", "allow_uncalibrated": True},
        },
    )

    _, logs = pipeline.run("One plus one", ["model_1", "model_2", "model_3"])

    for step in logs["steps"][1:]:
        assert step["cache_decision"] == "cross_model_adapter_hit"
        assert step["adapter_accepted"]
        assert step["prefill_input_tokens"] < step["prompt_tokens"]


class HighScoreAdapter(KVAdapter):
    def adapt(
        self,
        source_cache,
        source_config,
        target_config,
        target_model_name,
        *,
        target_token_ids,
        target_attention_mask,
    ):
        return AdaptedCache(
            past_key_values=source_cache.past_key_values,
            source_model=source_cache.model_name,
            target_model=target_model_name,
            transferred_tokens=target_token_ids.shape[-1],
            target_token_ids=target_token_ids,
            target_attention_mask=target_attention_mask,
            layer_mapping=(),
            source_layout=SimpleNamespace(),
            target_layout=None,
            accepted=True,
            degradation_score=0.5,
        )


def test_runtime_threshold_forces_a_logged_cold_fallback():
    first = RecordingModel()
    second = RecordingModel()
    pipeline = Pipeline(
        {
            "first": ModelBundle(model=first, tokenizer=CharacterTokenizer(), name="first-model"),
            "second": ModelBundle(model=second, tokenizer=CharacterTokenizer(), name="second-model"),
        },
        {
            "cache_policy": "cross_model",
            "max_new_tokens": 1,
            "adapter": HighScoreAdapter(),
            "degradation": {"threshold": 0.15},
        },
    )

    _, logs = pipeline.run("One plus one", ["first", "second", "second"])

    switched = logs["steps"][1]
    assert switched["cache_decision"] == "cross_model_quality_fallback"
    assert switched["fallback_reason"] == "runtime_degradation_threshold_exceeded"
    assert switched["cold_prefill"]


def test_compatibility_wrapper_omits_unsupported_cache_position():
    class StrictModel:
        def forward(self, input_ids, use_cache=True):
            return input_ids

        __call__ = forward

    ids = torch.tensor([[1]])
    assert torch.equal(
        forward_with_cache(StrictModel(), input_ids=ids, use_cache=True, cache_position=ids),
        ids,
    )
