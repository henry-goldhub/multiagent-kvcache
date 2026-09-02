from types import SimpleNamespace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from kvbridge.cache import LayerKV, dynamic_cache_from_layers, inspect_cache, normalize_cache
from kvbridge.compat import forward_with_cache


def config(layers=2, heads=2, head_dim=4):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=head_dim,
        hidden_size=heads * head_dim,
    )


class Layer:
    def __init__(self, capacity):
        self.keys = torch.ones((1, 2, capacity, 4))
        self.values = torch.ones((1, 2, capacity, 4))


class StaticLikeCache:
    def __init__(self):
        self.layers = [Layer(8), Layer(8)]

    def get_seq_length(self, layer_index=0):
        return (5, 3)[layer_index]


def test_static_and_sliding_lengths_are_sliced_per_layer():
    cache = StaticLikeCache()
    layers = normalize_cache(cache)
    assert [layer.key.shape[-2] for layer in layers] == [5, 3]
    layout = inspect_cache(cache, config())
    assert layout.sequence_lengths == (5, 3)


def test_real_dynamic_cache_rebuild_and_normalization_round_trip():
    model_config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
    )
    original = [
        LayerKV(
            key=torch.randn(1, 2, 3, 4),
            value=torch.randn(1, 2, 3, 4),
        )
        for _ in range(2)
    ]

    rebuilt = dynamic_cache_from_layers(original, model_config)
    normalized = normalize_cache(rebuilt)

    assert len(normalized) == 2
    for expected, actual in zip(original, normalized, strict=True):
        assert torch.equal(expected.key, actual.key)
        assert torch.equal(expected.value, actual.value)
    assert inspect_cache(rebuilt, model_config).cache_format == "DynamicCache"


def test_tiny_hf_model_cached_suffix_logits_match_cold_prefill():
    torch.manual_seed(7)
    model_config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = LlamaForCausalLM(model_config).eval()
    prefix = torch.tensor([[1, 4, 7, 9]])
    suffix = torch.tensor([[3, 5]])
    full = torch.cat((prefix, suffix), dim=-1)
    full_mask = torch.ones_like(full)

    with torch.inference_mode():
        cold = forward_with_cache(
            model,
            input_ids=full,
            attention_mask=full_mask,
            cache_position=torch.arange(full.shape[-1]),
            use_cache=True,
        )
        prefix_output = forward_with_cache(
            model,
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            cache_position=torch.arange(prefix.shape[-1]),
            use_cache=True,
        )
        cached = forward_with_cache(
            model,
            input_ids=suffix,
            attention_mask=full_mask,
            cache_position=torch.arange(prefix.shape[-1], full.shape[-1]),
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        )

    assert torch.allclose(cold.logits[:, -suffix.shape[-1] :], cached.logits, atol=1e-5)
