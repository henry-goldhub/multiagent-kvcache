from types import SimpleNamespace

import pytest
import torch

from kvbridge.adapter import (
    AdapterUnavailable,
    RidgeKVAdapter,
    ShapeOnlyKVAdapter,
    UnsupportedAdapter,
    normalized_layer_mapping,
    resize_kv_heads,
)
from kvbridge.cache import CacheState, LayerKV, normalize_cache
from kvbridge.quality import apply_quality_gate


def test_default_adapter_refuses_cross_model_transfer():
    with pytest.raises(AdapterUnavailable):
        UnsupportedAdapter().adapt(None, None, None, "target")


def config(layers, attention_heads, kv_heads, head_dim):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_attention_heads=attention_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=attention_heads * head_dim,
    )


def legacy_cache(layers, heads, tokens, head_dim):
    return tuple(
        (
            torch.full((1, heads, tokens, head_dim), float(index)),
            torch.full((1, heads, tokens, head_dim), float(index + 10)),
        )
        for index in range(layers)
    )


def test_normalized_depth_layer_mapping():
    assert normalized_layer_mapping(4, 6) == (0, 1, 1, 2, 2, 3)
    assert normalized_layer_mapping(4, 1) == (0,)


def test_head_pooling_and_repetition_values():
    source = torch.arange(8.0).view(1, 8, 1, 1)
    pooled = resize_kv_heads(source, 4)
    assert torch.allclose(pooled.flatten(), torch.tensor([0.5, 2.5, 4.5, 6.5]))
    repeated = resize_kv_heads(torch.arange(4.0).view(1, 4, 1, 1), 8)
    assert torch.equal(repeated.flatten(), torch.tensor([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]))


def test_shape_adapter_returns_target_shaped_cache_without_mutating_source():
    source_config = config(4, 8, 4, 6)
    target_config = config(6, 6, 3, 4)
    source_past = legacy_cache(4, 4, 5, 6)
    original = tuple((key.clone(), value.clone()) for key, value in source_past)
    state = CacheState("source", torch.ones((1, 5), dtype=torch.long), "text", source_past, source_config)
    target_ids = torch.ones((1, 7), dtype=torch.long)

    adapted = ShapeOnlyKVAdapter().adapt(
        state,
        source_config,
        target_config,
        "target",
        target_token_ids=target_ids,
        target_attention_mask=torch.ones_like(target_ids),
    )

    layers = normalize_cache(adapted.past_key_values)
    assert len(layers) == 6
    assert all(layer.key.shape == (1, 3, 7, 4) for layer in layers)
    assert not adapted.accepted
    for (key, value), (original_key, original_value) in zip(source_past, original, strict=True):
        assert torch.equal(key, original_key)
        assert torch.equal(value, original_value)


def test_ridge_fit_recovers_known_affine_projection(tmp_path):
    source_config = config(1, 2, 2, 3)
    target_config = config(1, 2, 2, 2)
    source = torch.randn(1, 2, 12, 3)
    weight = torch.tensor([[1.0, 0.5], [-0.5, 2.0], [0.25, -1.0]])
    bias = torch.tensor([0.2, -0.3])
    target = source @ weight + bias
    source_samples = [[LayerKV(source, source + 1)]]
    target_samples = [[LayerKV(target, (source + 1) @ weight + bias)]]
    adapter = RidgeKVAdapter(tmp_path, ridge_lambda=1e-6)

    projection = adapter.fit_pair(
        source_samples,
        target_samples,
        source_model="source",
        target_model="target",
        source_config=source_config,
        target_config=target_config,
    )
    predicted = source @ projection.tensors["layer.0.key.weight"] + projection.tensors["layer.0.key.bias"]
    assert torch.allclose(predicted, target, atol=1e-3)

    result = apply_quality_gate(
        projection,
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 2, 4)),
        threshold=0.15,
        validation_examples=1,
    )
    assert result.accepted
    adapter.save_pair(projection)
    loaded = RidgeKVAdapter(tmp_path).load_pair("source", "target")
    assert loaded.metadata.accepted


def test_quality_gate_rejects_degraded_logits():
    adapter = RidgeKVAdapter()
    projection = adapter.fit_pair(
        [[LayerKV(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))]],
        [[LayerKV(torch.randn(1, 1, 3, 2), torch.randn(1, 1, 3, 2))]],
        source_model="a",
        target_model="b",
        source_config=config(1, 1, 1, 2),
        target_config=config(1, 1, 1, 2),
    )
    cold = torch.tensor([[[10.0, -10.0]]])
    adapted = torch.tensor([[[-10.0, 10.0]]])
    assert not apply_quality_gate(projection, cold, adapted, threshold=0.15).accepted


def test_malformed_artifact_is_rejected_with_a_precise_reason(tmp_path):
    adapter = RidgeKVAdapter(tmp_path)
    tensor_path, metadata_path = adapter._paths("source", "target")
    tensor_path.write_bytes(b"not-a-safetensor")
    metadata_path.write_text("{broken-json", encoding="utf-8")

    with pytest.raises(AdapterUnavailable, match="malformed_calibration"):
        adapter.load_pair("source", "target")
