"""Tiny deterministic models used for offline integration tests and demos."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from .cache import normalize_cache
from .models import ModelBundle


class ByteTokenizer:
    """UTF-8 byte tokenizer with a single BOS token and no downloads."""

    bos_token_id = 256
    eos_token_id = None
    vocab_size = 257

    def __call__(
        self, text: str, return_tensors: str, add_special_tokens: bool = True
    ) -> SimpleNamespace:
        if return_tensors != "pt":
            raise ValueError("ByteTokenizer supports return_tensors='pt' only")
        ids = list(text.encode("utf-8"))
        if add_special_tokens:
            ids.insert(0, self.bos_token_id)
        input_ids = torch.tensor([ids], dtype=torch.long)
        return SimpleNamespace(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    def decode(self, token_ids: Any, skip_special_tokens: bool = True, **kwargs: Any) -> str:
        values = [int(value) for value in token_ids]
        if skip_special_tokens:
            values = [value for value in values if value < 256]
        return bytes(values).decode("utf-8", errors="replace")


class SyntheticCausalLM(torch.nn.Module):
    """A deterministic causal LM emitting valid per-layer KV tensors."""

    def __init__(self, name: str, layers: int, attention_heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.name = name
        self.embedding = torch.nn.Embedding(257, attention_heads * head_dim)
        self.config = SimpleNamespace(
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            hidden_size=attention_heads * head_dim,
        )
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = True,
        cache_position: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        batch, tokens = input_ids.shape
        layers = []
        normalized_past = normalize_cache(past_key_values) if past_key_values is not None else None
        for layer_index in range(self.config.num_hidden_layers):
            values = input_ids.float().view(batch, 1, tokens, 1)
            values = values.expand(
                batch,
                self.config.num_key_value_heads,
                tokens,
                self.config.head_dim,
            )
            current_key = values / 256.0 + layer_index
            current_value = values / 512.0 + layer_index
            if normalized_past is not None:
                past = normalized_past[layer_index]
                current_key = torch.cat((past.key, current_key), dim=-2)
                current_value = torch.cat((past.value, current_value), dim=-2)
            layers.append((current_key, current_value))
        logits = torch.zeros((batch, tokens, 257), device=input_ids.device)
        logits[..., ord("1")] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=tuple(layers))


def synthetic_models() -> dict[str, ModelBundle]:
    tokenizer = ByteTokenizer()
    specifications = {
        "model_1": (2, 4, 2, 4),
        "model_2": (3, 3, 1, 6),
        "model_3": (4, 6, 3, 2),
    }
    return {
        name: ModelBundle(
            model=SyntheticCausalLM(name, *dimensions),
            tokenizer=tokenizer,
            name=name,
        )
        for name, dimensions in specifications.items()
    }
