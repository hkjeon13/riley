# FP32/BF16 correctness calibration

This workflow creates two Hugging Face oracle artifacts, produces a Python-free
native candidate under the explicit CUDA feature, and compares its
release-qualified `rustinfer-native` execution against the oracles with
separate Python comparison and raw-replay commands. It is not the PR 01 golden
fixture and the HF BF16 run must never be presented as a native E0 candidate.

Before either HF producer imports or loads a model, the shared standard-library
host probe validates the exact primary host and the transient idle-GPU
preconditions. Each manifest records a strict
`provenance.observed_environment` snapshot and binds the probe implementation
as a Git source. FP32/BF16 comparison additionally requires identical power
limit, application graphics/memory clocks, persistence mode, and CPU governor
profile; drift fails calibration instead of producing a report.

The frozen oracle authority is
`benchmarks/correctness/smollm2-fp32-bf16-native-e0-v2.json`, validated by
`benchmarks/schemas/correctness-gate.schema.json`. The release candidate
authority is
`benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json`, validated by
`benchmarks/schemas/correctness-gate-v3.schema.json`. V3 changes only the
required candidate reduction inventory to `canonical-v1`; it inherits the
byte-exact v2 oracle manifests, all thresholds, roles, corpus, and provenance
rules through its explicit lineage object. The original v1 and v2 files remain
frozen historical/oracle evidence. A v1 report cannot activate v2, and a v2
candidate report cannot authorize the v3 release. Gate files are not edited
after evidence generation because doing that would invalidate their recorded
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

The v1 full-31 calibration report was reviewed at Git revision
`8ab7490bfdf9efd1d7c7d831204b8e67c0c7c5b9`. Its raw report SHA-256 is
`ca13c033af2ddce5cfbf280fc1f4d2f95d0cba0e242bda8c59f2592946cec726`;
it failed 12 of 31 cases while its semantic self-check passed. It remains
predeclaration calibration evidence, not activation evidence. The exact 48,625
byte report is versioned at
`benchmarks/correctness/evidence/smollm2-fp32-bf16-native-e0-v1-failed-oracle-report.json`;
the contract validator binds its path, size, SHA-256, report identity, summary,
and source revision before accepting the v2 gate.

V2 applies one uniform 15% outward margin to every recorded aggregate metric:
upper bounds are `observed * 1.15`, while cosine lower bounds are
`1 - (1 - observed) * 1.15`. These values are frozen before the v2 replay.
Any later data-dependent adjustment requires another gate version. Only an
independent, passing, raw-sidecar-replayed full-31 v2 report can activate v2:

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
| Full first-layer hidden | 0.3884272575378418 | 0.008509292567237658 | 0.13578447438776492 | 0.005414661057131772 | at least 0.999983706829855 |
| Final logits | 5.852936458587647 | 1.151280319263363 | 1.1707394897937775 | 0.13616598220459955 | at least 0.9979035305495393 |
| Final log probabilities | 4.998420619964599 | 0.6007178144163239 | 0.5767027348279953 | 0.04668832837569344 | at least 0.9987779663298298 |

If the 31-prompt v2 run fails, the output remains a gate failure until a
reviewed, version-bumped contract explains any change. The oracle calibration
report always contains `e0_candidate_evidence=false`.

## Native candidate contract

Native candidate contract v3 is owned by the non-default development workspace
member `crates/rustinfer-native`. The crate owns both its library and the
`rustinfer-native` binary. Its default feature set is empty, and the binary
declares `required-features = ["cuda"]`; the producer and native runtime
dependencies are selected only by the explicit `cuda` feature. The feature-off
library retains the side-effect-free strict ABI parser, while evidence
production requires defaults disabled and exactly `cuda` enabled.

The exact locked build argv bound into candidate evidence is:

```sh
cargo build --locked --release --package rustinfer-native \
  --no-default-features --features cuda --bin rustinfer-native
```

Adding `--package` is mandatory because `rustinfer-native` is not a root
default member. Keeping CUDA explicit also preserves the CUDA-free
`--workspace --no-default-features` CPU gate. An excluded `tools/native`
package or a server-owned calibration binary would not satisfy the root
`Cargo.lock`, non-default ownership, and release-artifact boundaries.

A candidate manifest has runtime dependency class `native-production`, uses
the `rustinfer-native-v3` lane and `Cargo.lock`, and records a clean candidate
Git revision independently of the older oracle revision. The producer requires a
clean, revision-bound release build and a clean runtime checkout at the exact
Git revision embedded at build time, then rechecks repository provenance and
bound source hashes after capture. It performs the CUDA/NVML and primary-host
preflight, binds the CUDA and NVML observations, and validates the pinned model
revision, configuration, weights, tokenizer files, and aggregate tokenizer
hash.

The producer emits a create-only manifest, safetensors sidecar, and bundled
`rustinfer-native` executable as sibling files outside the repository. It
refuses to overwrite any of them, hashes the bundled executable and sidecar,
echoes the locked release build argv above, and records the native calibration
capture argv. Its exact ordered contract-v3 shape is:

```sh
rustinfer-native calibrate \
  --repository-root /workspace/rustinfer \
  --model /models/smollm2 \
  --gate-manifest benchmarks/correctness/smollm2-fp32-bf16-native-e0-v3.json \
  --prompts benchmarks/prompts.jsonl \
  --manifest candidate-manifest.json \
  --sidecar candidate-sidecar.safetensors \
  --reduction-variant canonical-v1
```

Repository and model roots are normalized absolute POSIX paths; outputs are
normalized sibling filenames. Unknown, reordered, duplicated, or positional
arguments fail closed. The producer must bind the checkpoint it actually loads
to the manifest's model revision and config/weights/tokenizer hashes. The
capture argv also binds the gate, prompt corpus, and the sole release-qualified
`canonical-v1` production-default reduction variant.

The historical `fixed-contiguous-37-balanced-v1` selector remains available
only for development compatibility and optimizer diagnostics. It is excluded
from v3 candidate manifests and cannot contribute release evidence. Historical
v2 manifests, reports, and raw archives retain their two-variant replay contract
and are never rewritten as v3 evidence.

One v3 capture covers exactly 31 ordered prompts, one reduction variant, and
three tensors per prompt: full first-layer hidden output, final logits, and
final FP32 log probabilities. The sidecar inventory is therefore
`31 * 1 * 3 = 93` tensors. Tensor capture uses the cache-off path; the candidate
also supplies independent cache-on and cache-off semantic paths.

Numeric values are compared to FP32. Semantic values are path-matched to HF
BF16 (`cache-on` to `cache-on`, `cache-off` to `cache-off`). Generated IDs and
top-1 are ordered exact. Top-k is exact as a set, not as a ranked list. Cache
paths must match through the predeclared 16-token window, and the zero-based
first divergence step is recomputed from raw token IDs. The canonical variant
and every prompt must pass.

Candidate production does not run the comparator. The separate Python
comparison command is:

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

The native lane remains `contract-only`; implementing or running the producer
does not promote it. It stays contract-only until the separate Python
comparator and validator complete a passing full raw-sidecar replay for all 31
prompts for the canonical candidate under the v3 release gate, using the
immutable v2 oracle lineage.

## Evidence bundle and hashes

The oracle JSON manifests bind model revision, config and weights, tokenizer,
ordered prompt rows, the frozen matrix and v2 gate, environment, lane,
dependency lock, Git revision/status, sidecar path/hash/shape/dtype, and
producer versions. The v3 candidate manifest binds the same immutable model,
prompt, and environment identity plus its distinct v3 gate and lane; it does
not claim that the frozen matrix was produced at the newer candidate revision.
Oracle and candidate Git revisions are deliberately separate. An E0 result is
acceptable only with the replayable bundle named by the gate: both oracle
manifests and sidecars, the passing oracle report, candidate manifest/sidecar/
executable, and candidate correctness report. The native lane remains
`contract-only` and rejects E0 success rows until that full raw replay passes.

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

Sidecars and executable bundles are intentionally not committed. Frozen v2
oracle manifests use `correctness-calibration-manifest.schema.json`; v3
candidate manifests and reports use
`correctness-calibration-manifest-v3.schema.json` and
`correctness-report-v3.schema.json`. The oracle activation report continues to
use `oracle-calibration-report.schema.json`. The deterministic fake
`CalibrationFixture` in `tests/test_calibration.py` constructs all 31 rows, a
canonical v3 candidate, raw sidecars, two distinct Git revisions, and a bundled
fake executable; separate compatibility cases replay historical v2 two-variant
artifacts. There is intentionally no standalone synthetic “passing” report
that could be mistaken for E0 evidence.
