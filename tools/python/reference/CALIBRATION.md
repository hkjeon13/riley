# FP32/BF16 correctness calibration

This workflow creates two Hugging Face oracle artifacts and, in a later native
PR, compares two `rustinfer-native` execution variants against them. It is not
the PR 01 golden fixture and the HF BF16 run must never be presented as a
native E0 candidate.

Before either HF producer imports or loads a model, the shared standard-library
host probe validates the exact primary host and the transient idle-GPU
preconditions. Each manifest records a strict
`provenance.observed_environment` snapshot and binds the probe implementation
as a Git source. FP32/BF16 comparison additionally requires identical power
limit, application graphics/memory clocks, persistence mode, and CPU governor
profile; drift fails calibration instead of producing a report.

The language-neutral authority is
`benchmarks/correctness/smollm2-fp32-bf16-native-e0-v1.json`, validated by
`benchmarks/schemas/correctness-gate.schema.json`. Its gate ID is
`smollm2-fp32-bf16-native-e0-v1`. The threshold status is an immutable
condition: a candidate gate requires a passing, raw-sidecar-replayed HF report
covering all 31 ordered prompts. The gate file is not edited from “pending” to
“active” after oracle generation; doing that would invalidate its own recorded
provenance.

## Produce the two HF oracles

Use a clean committed checkout on the primary RTX 4090 environment. Outputs
must be new sibling files outside the repository. Run FP32 and BF16 in separate
fresh processes so they cannot share model state or retain device memory:

```sh
mkdir -p /var/tmp/rustinfer-calibration
UV_BIN=/absolute/path/to/pinned/uv
export UV_PROJECT_ENVIRONMENT=/var/tmp/rustinfer-project-envs/reference-calibration-001
test ! -e "$UV_PROJECT_ENVIRONMENT"
test "$("$UV_BIN" --version)" = 'uv 0.12.5 (x86_64-unknown-linux-gnu)'
test "$(sha256sum "$UV_BIN" | awk '{print $1}')" = \
  b65f23a420c4acc96427efb30e5ed9bc0f7e25d2d712000f6ede77c1a0de5f46
MANAGED_PYTHON="$(UV_PYTHON_DOWNLOADS=never "$UV_BIN" python find 3.13.15)"
test "$(sha256sum "$MANAGED_PYTHON" | awk '{print $1}')" = \
  ce20f82411f2b0ccdf3e2212ca62303519521d73d25178588f1a9c8d4935c866
UV_PYTHON=3.13.15 UV_PYTHON_DOWNLOADS=never \
  "$UV_BIN" sync --frozen --offline --project tools/python/reference

"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference \
  calibrate-produce \
  --role fp32 \
  --prompts benchmarks/prompts.jsonl \
  --manifest /var/tmp/rustinfer-calibration/fp32-manifest.json \
  --sidecar /var/tmp/rustinfer-calibration/fp32.safetensors \
  --repo-root .

"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference \
  calibrate-produce \
  --role bf16 \
  --prompts benchmarks/prompts.jsonl \
  --manifest /var/tmp/rustinfer-calibration/bf16-manifest.json \
  --sidecar /var/tmp/rustinfer-calibration/bf16.safetensors \
  --repo-root .
```

Both commands are cache-only by default. `--allow-download` permits only the
immutable model revision. Before loading the model, the producer verifies
`config.json`, `model.safetensors`, and every tokenizer file against fixed
SHA-256 values. `trust_remote_code` is always false.

The sidecar stores, for each ordered prompt:

- the full first transformer-layer output for every valid token position;
- full-vocabulary logits at the last valid input position; and
- full-vocabulary FP32 log-softmax values under
  `log-softmax-fp32-v1`.

First, middle, and last token positions are recorded as anchors; they do not
replace the full hidden tensor used by the numeric gate. Prompts with
`target_prompt_tokens` are repeat/truncated after tokenization to exactly that
many input IDs. The compact 8,064-token boundary row therefore remains exact
without storing a huge prompt string. The producer transfers only the selected
first-layer output and final vectors to CPU, deletes GPU temporaries, and calls
`empty_cache` between cases. FP32 and BF16 use separate processes.

The BF16 artifact additionally records 32-step greedy cache-on and cache-off
paths. EOS may shorten a path. The `early-eos` row is valid only when both
paths generate EOS as output step zero and record `stop_reason=eos`; its input
does not need to begin with EOS.

## Activate threshold evidence

The thresholds are conservative ceilings predeclared from a non-activating
five-prompt RTX 4090 probe. The gate records those observed worst values and
their headroom explicitly; the probe itself is not activation evidence and the
ceilings cannot be silently relaxed. A passing full-31 report is the activation
evidence:

```sh
"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference \
  calibrate-oracles \
  --fp32-manifest /var/tmp/rustinfer-calibration/fp32-manifest.json \
  --bf16-manifest /var/tmp/rustinfer-calibration/bf16-manifest.json \
  --output /var/tmp/rustinfer-calibration/oracle-calibration-report.json \
  --repo-root .

"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference \
  calibrate-validate-oracles \
  /var/tmp/rustinfer-calibration/oracle-calibration-report.json \
  --fp32-manifest /var/tmp/rustinfer-calibration/fp32-manifest.json \
  --bf16-manifest /var/tmp/rustinfer-calibration/bf16-manifest.json \
  --repo-root .
```

The comparator reopens the sidecars and computes every scalar itself. For an
FP32 value `a` and comparison value `b`, relative error is
`abs(a-b)/max(abs(a),1)`. Full hidden tensors are reduced in bounded FP32
chunks with FP64 scalar sums, rather than expanded into Python-float lists.

| Tensor | max abs | mean abs | max relative | mean relative | cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full first-layer hidden | 0.30 | 0.0085 | 0.09 | 0.0055 | at least 0.99998 |
| Final logits | 5.6 | 1.15 | 1.0 | 0.125 | at least 0.9990 |
| Final log probabilities | 4.8 | 0.60 | 0.56 | 0.047 | at least 0.9988 |

If the 31-prompt run fails, the output remains a gate failure until a reviewed
contract revision explains and versions any change. The oracle calibration
report always contains `e0_candidate_evidence=false`.

## Native candidate contract

A future candidate manifest has runtime dependency class `native-production`,
uses the `rustinfer-native` lane and `Cargo.lock`, and records a clean candidate
Git revision independently of the older oracle revision. It must bundle and
hash the exact `rustinfer-native` executable beside its manifest, echo the
locked release build argv, and record the native calibration capture argv. The
capture argv binds the gate, prompt corpus, output manifest and sidecar, plus
both ordered reduction variants:

- `canonical-v1`, the production-default execution; and
- `fixed-contiguous-37-balanced-v1`, the alternate execution.

The alternate applies to every floating-point reduction contributing to the
captured tensors or greedy logits: matmul dot-product sums, RMSNorm sums of
squares, attention softmax max/sum, and final log-softmax max/sum. The logical
reduction axis is traversed in ascending order, split into contiguous
37-element chunks with one short final chunk, reduced by an ascending local
left fold, then merged as adjacent chunk partials in an ascending deterministic
balanced binary tree; an unpaired final partial is carried to the next level.
The accumulator dtype policy remains the canonical operator policy.

Each candidate variant supplies independent tensors and cache-on/cache-off
semantic paths. Numeric values are compared to FP32. Semantic values are
path-matched to HF BF16 (`cache-on` to `cache-on`, `cache-off` to `cache-off`).
Generated IDs and top-1 are ordered exact. Top-k is exact as a set, not as a
ranked list. Cache paths must match through the predeclared 16-token window,
and the zero-based first divergence step is recomputed from raw token IDs.
Both variants and every prompt must pass.

The future comparison command is:

```sh
"$UV_BIN" run --frozen --offline --no-sync --project tools/python/reference rustinfer-reference \
  calibrate-compare \
  --fp32-manifest /var/tmp/rustinfer-calibration/fp32-manifest.json \
  --bf16-manifest /var/tmp/rustinfer-calibration/bf16-manifest.json \
  --oracle-report /var/tmp/rustinfer-calibration/oracle-calibration-report.json \
  --candidate-manifest /var/tmp/rustinfer-calibration/candidate-manifest.json \
  --output /var/tmp/rustinfer-calibration/correctness-report.json \
  --repo-root .
```

`calibrate-validate-report` takes the same four raw inputs plus the report and
reruns the entire comparison. Merely parsing a report, recomputing its nested
booleans, or checking a sibling filename is not approval.

## Evidence bundle and hashes

The JSON manifests bind model revision, config and weights, tokenizer, ordered
prompt rows, matrix, gate manifest, environment, lane, dependency lock, Git
revision/status, sidecar path/hash/shape/dtype, and producer versions. Oracle
and candidate Git revisions are deliberately separate. A future E0 result is
acceptable only with the replayable bundle named by the gate: both oracle
manifests and sidecars, the passing oracle report, candidate manifest/sidecar/
executable, and candidate correctness report. PR 01 marks the native lane
`contract-only` and rejects E0 success rows until that raw replay integration
exists.

`correctness_report_sha256` is SHA-256 of the exact report file bytes. Reports
contain no self-hash. The candidate manifest hash transitively binds its
executable hash and build/capture argv; the correctness report also exposes
their canonical hashes for result validators.

Tokenizer binding uses these five immutable raw files:
`merges.txt`, `special_tokens_map.json`, `tokenizer.json`,
`tokenizer_config.json`, and `vocab.json`. First compute raw SHA-256 per file.
Then compute SHA-256 of the UTF-8 canonical JSON hash map with lexicographically
sorted keys, separators `,` and `:`, no ASCII escaping, no NaN, and no trailing
newline. The aggregate is
`51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db`.

Sidecars and executable bundles are intentionally not committed. The schemas
are `correctness-calibration-manifest.schema.json`,
`oracle-calibration-report.schema.json`, and `correctness-report.schema.json`.
The deterministic fake `CalibrationFixture` in `tests/test_calibration.py`
constructs all 31 rows, both candidate variants, raw sidecars, two distinct Git
revisions, and a bundled fake executable. It is the checked-in offline fixture;
there is intentionally no standalone synthetic “passing” report that could be
mistaken for E0 evidence.
