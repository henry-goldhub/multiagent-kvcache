# KVBridge

KVBridge is an experimental toolbox for reusing Hugging Face KV caches in
append-only, multi-step LLM pipelines. It performs exact same-model prefix
reuse and provides calibrated, quality-gated cross-model cache adaptation.

The original assignment is preserved in [TASK_SPEC.md](TASK_SPEC.md).

## Status

- Exact same-model reuse is implemented and tested offline.
- Cross-model shape conversion and affine ridge projection are implemented.
- Pair-level degradation decisions are persisted and enforced at runtime.
- Missing, rejected, malformed, or failing adapters safely cold-prefill.
- The synthetic profile runs all four required assignments without downloads.
- Lightweight stand-in and named-model profiles are implemented but require
  model downloads. Named-model results remain unverified until a cloud run.

## Install

```powershell
python -m pip install -e ".[dev,eval]"
```

For the 4-bit full-model profile:

```powershell
python -m pip install -e ".[dev,eval,full]"
```

## Public API

```python
from kvbridge import Pipeline, evaluate

pipeline = Pipeline(models, config)
result, logs = pipeline.run(task_input, step_assignment)
report = evaluate(pipeline, dataset, step_assignments, config)
```

The no-cache baseline disables reuse *between* stages but retains normal KV
caching during autoregressive decoding. Disabling decode caching would create
an artificially slow baseline unrelated to the optimization being measured.

## Quickstart

No downloads:

```powershell
python examples/quickstart.py --profile synthetic --num-examples 2
```

Openly licensed lightweight stand-ins:

```powershell
python examples/quickstart.py --profile standin --num-examples 5
```

Named assignment models on a suitable cloud GPU:

```powershell
python examples/quickstart.py --profile full --num-examples 200 --max-new-tokens 96
```

The stand-in profile uses Qwen2.5-0.5B-Instruct, SmolLM2-135M-Instruct,
and DistilGPT2. They are different architectural families and tokenizers, but
their GSM8K accuracy is not a substitute for the requested 7B-model results.

### Cloud notebook

Use [examples/cloud_run.ipynb](examples/cloud_run.ipynb) in Colab or a GPU-backed
Jupyter session. Commit and push the latest source first: a cloud clone cannot
see your uncommitted local changes. Upload the notebook to Colab, select a GPU
runtime, and run its cells in order. Set `ENABLE_MODEL_DOWNLOADS=True` when
ready to download pretrained weights and GSM8K.

The default is a five-example stand-in run with 16 fitting and eight validation
examples per hand-off. It records the source/model revisions and dataset row
indices, calibrates both pairs, proves a missing-artifact fallback, and evaluates
all four assignments. Each run gets isolated adapters and checkpointed reports
under `results/cloud/`; the export cell downloads a ZIP containing results and
artifacts but no model weights or credentials. Keep the committed notebook
template output-free and publish measured report files separately.

The full profile is an explicit opt-in. Calibration loads two models at a time;
mixed evaluation still holds all three models on one GPU, so 4-bit loading does
not guarantee a fit. The notebook has offline synthetic workflow tests, but
pretrained cloud results remain unverified until you execute it. Download the
export before disconnecting the runtime.

## Configuration

```python
config = {
    "cache_policy": "cross_model",
    "seed": 42,
    "device": "auto",
    "max_new_tokens": 96,
    "step_assignments": [
        ["model_1", "model_1", "model_1"],
        ["model_2", "model_2", "model_2"],
        ["model_3", "model_3", "model_3"],
        ["model_1", "model_2", "model_3"],
    ],
    "adapter": {
        "type": "ridge",
        "artifact_dir": "artifacts/adapters",
        "ridge_lambda": 0.001,
    },
    "degradation": {
        "metric": "mean_kl",
        "threshold": 0.15,
        "probe_tokens": 8,
    },
}
```

Cache policies:

| Policy | Same model | Different model |
| --- | --- | --- |
| `disabled` | Cold stage prefill | Cold stage prefill |
| `same_model_only` | Exact prefix cache | Cold stage prefill |
| `cross_model` | Exact prefix cache | Accepted calibrated adapter or safe fallback |

## Token-exact prompt growth

The pipeline does not re-tokenize accumulated text for a same-model hand-off.
It saves the exact prompt and generated token IDs in `CacheState`, tokenizes
only the newly appended instruction with `add_special_tokens=False`, and sends
that suffix with the old cache. This avoids BPE boundary changes and repeated
BOS tokens. Generated text is preserved exactly internally; whitespace is
stripped only from the user-facing final result.

All cached forwards carry an attention mask covering past plus current tokens.
Absolute cache positions are passed only when the model's `forward` method
supports them. Input tensors are placed on the input-embedding device, and CUDA
timings are synchronized.

## Cross-model adapter

KVBridge normalizes each cache layer to:

```text
[batch, num_key_value_heads, tokens, head_dim]
```

The shape-only baseline performs four operations:

1. **Layer mapping:** target layer `j` uses
   `round(j * (source_layers - 1) / (target_layers - 1))`. This preserves
   normalized network depth, mapping early, middle, and late layers to roughly
   corresponding depths.
2. **KV-head mapping:** proportional groups are averaged when reducing heads;
   source heads are repeated proportionally when expanding.
3. **Token-position mapping:** the source sequence axis is linearly resampled
   to the number of tokens in the target tokenizer's prefix.
4. **Head-dimension mapping:** vectors are linearly interpolated in float32 and
   returned to their original dtype.

The ridge adapter replaces step 4 with independent learned affine projections
for every target layer and for keys and values separately:

```text
target = mapped_source @ W + b
```

It solves ridge regression with default regularization `1e-3`. Projection
weights are stored as Safetensors; model-pair metadata and gate results are
stored as JSON under `artifacts/adapters/`.

## Calibration and degradation fallback

The intended calibration split is 16 fitting and 8 held-out GSM8K training
examples per source→target pair. `kvbridge.calibration.calibrate_model_pair`
collects source and cold-target prefix caches, fits the ridge projections, and
compares target continuation logits over a held-out probe.

The gate uses:

```text
mean KL(cold target logits || adapted-cache target logits)
```

The default threshold is `0.15`. The decision is stored per model pair so the
runtime does not perform an expensive cold reference at every hand-off. A
rejected or missing calibration artifact triggers a cold prefill and records
the reason in the step log.

Calibrate the two hand-offs used by the mixed stand-in assignment with:

```powershell
python examples/calibrate.py --profile standin --source model_1 --target model_2
python examples/calibrate.py --profile standin --source model_2 --target model_3
```

Rejected artifacts are still saved with their measured KL score, making the
runtime fallback reproducible and auditable.

## Reporting

`evaluate` runs every assignment under both the selected policy and the
cross-step-cache-disabled baseline. It reports accuracy, mean/p50 prefill and
decode latency, adapter overhead, cache-hit rate, adapter attempt/accept rate,
fallback rate, and prefill/total speedup. `report_to_markdown` renders the
comparison table; optional config paths write JSON and Markdown artifacts.

### Synthetic smoke-test result

Measured locally on two deterministic synthetic examples with two generated
tokens per stage. These numbers validate execution and accounting only; they
are not language-model quality results.

| Setting | Accuracy | Cache-hit rate | Fallback rate |
| --- | ---: | ---: | ---: |
| model_1 → model_1 → model_1 | 1.000 | 0.553 | 0.000 |
| model_2 → model_2 → model_2 | 1.000 | 0.553 | 0.000 |
| model_3 → model_3 → model_3 | 1.000 | 0.553 | 0.000 |
| model_1 → model_2 → model_3 | 1.000 | 0.000 | 0.667 |

Stand-in and named-model latency/accuracy tables must be added only after those
profiles are actually run. No results are inferred or fabricated.

## Tests

```powershell
python -m pytest
python -m ruff check kvbridge tests examples
```

The default suite is offline. Downloadable tests should use the `integration`
marker; named-model cloud tests use `full_models`.

## Limitations and failure modes

- Cross-model KV transfer is not lossless or established in the literature.
- Relative-depth layers need not share semantic representations.
- Different tokenizers have no exact token-position correspondence; linear
  sequence resampling is only an approximation.
- Keys may already contain architecture-specific positional transformations.
- Head pooling discards information; repetition creates redundant heads.
- Ridge projections are model-pair-specific and may not generalize beyond the
  calibration distribution.
- Mistral sliding-window layers can retain shorter histories than other layers;
  KVBridge records per-layer valid sequence lengths.
- Quantized and offloaded cache adaptation is not supported in this version.
- An adapter may be slower than cold prefill for short contexts. The report
  keeps adapter overhead separate so this is visible.
- Frequent quality-gate rejection is a valid research result. KVBridge favors
  correct fallback over claiming an unsafe speedup.
