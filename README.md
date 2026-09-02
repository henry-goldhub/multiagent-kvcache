# KVBridge

KVBridge is an experimental Python package for reuse of Hugging Face KV caches
in append-only multi-step language-model pipelines.

## Current status

This repository currently contains an installable package, append-only GSM8K
prompts, exact same-model cache reuse, cache metadata, a safe cross-model
adapter interface, baseline evaluation helpers, and lightweight unit tests.
Concrete cross-model projection, quality gating, and full GSM8K experiments are
the next implementation milestones.

## How same-model reuse works

At the end of a stage, KVBridge preserves the model's `past_key_values` and the
exact token IDs of the full stage prompt *plus its generated answer*. The next
stage's append-only prompt is tokenized again. If those saved token IDs are an
exact prefix of the new token IDs and the model is unchanged, KVBridge sends
only the added suffix together with `past_key_values`. The model therefore
attends to the cached prefix instead of prefilling it again.

During generation, KVBridge performs greedy decoding one token at a time. Each
new token is forwarded with the prior cache, then the final generated token is
also forwarded once to make sure the saved cache includes the complete answer.
This extra forward is necessary: logits predict a token before that token has
been incorporated into the cache.

## Cache policies

| Policy | Same-model hand-off | Cross-model hand-off |
| --- | --- | --- |
| `disabled` | Cold prefill | Cold prefill |
| `same_model_only` (default) | Exact token-prefix reuse | Cold prefill |
| `cross_model` | Exact token-prefix reuse | Attempt adapter, then safely fall back until a quality gate accepts it |

`disabled` is the evaluation baseline. `same_model_only` is exact because the
model and prefix token IDs must match. `cross_model` is deliberately
conservative: an adapter result is never used until it can identify the target
token positions it covers and pass a continuation-quality check against cold
prefill.

## Cache normalization

`kvbridge.cache` converts either Hugging Face's legacy tuple cache or a
DynamicCache-compatible object into a per-layer canonical representation:
`[batch, num_key_value_heads, tokens, head_dim]` for both keys and values.
`inspect_cache` validates that the cache matches its model configuration and
reports layer count, attention/KV heads, head dimension, per-layer valid
sequence lengths, and the source cache format. Static-cache capacity is sliced
to its valid token length, and sliding-window layers may have shorter caches.
The future cross-model adapter will use this boundary to map source tensors
into a target architecture.

## Install

```powershell
python -m pip install -e ".[dev,eval]"
```

## Intended API

```python
from kvbridge import Pipeline, evaluate

pipeline = Pipeline(models, config)
result, logs = pipeline.run(task_input, step_assignment)
report = evaluate(pipeline, dataset, step_assignments, config)
```

## Project layout

- `kvbridge/`: package implementation
- `examples/quickstart.py`: small smoke-test example
- `tests/`: unit and integration tests

## Research constraint

Same-model KV reuse is exact only when the earlier model token IDs are an exact
prefix of the later prompt. Cross-model KV transfer is approximate because model
architectures and tokenizers differ. A concrete adapter must therefore compare
against cold-prefill quality and safely fall back when degradation exceeds a
configured threshold.
