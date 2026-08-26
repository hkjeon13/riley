# Python-free clean-runtime E2E gate

This PR16 gate runs only on the authorized Linux GPU host. It starts the final
immutable release image—not a builder or test image—with Docker network mode
`none`, a read-only root filesystem, one read-only real-checkpoint mount, and a
small writable evidence mount. The server remains bound to container loopback.
Host-driven probes enter that same network namespace with `docker exec` and
Bash `/dev/tcp`; the production image needs no HTTP client or Python runtime.

The approved correctness golden is a reviewed, separately checksummed JSON
file with this closed schema:

```json
{
  "schema_version": "rustinfer.python-free-release-e2e-golden.v1",
  "correctness_gate_id": "smollm2-fp32-bf16-native-e0-v2",
  "correctness_report_sha256": "reviewed-lowercase-sha256",
  "source_revision": "clean-40-character-candidate-revision",
  "model_id": "HuggingFaceTB/SmolLM2-135M",
  "model_revision": "immutable-model-revision",
  "config_sha256": "reviewed-lowercase-sha256",
  "weights_sha256": "reviewed-lowercase-sha256",
  "tokenizer_aggregate_sha256": "51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db",
  "tokenizer_json_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
  "prompt": "One-line fixed release prompt",
  "max_tokens": 8,
  "expected_greedy_text_sha256": "reviewed-lowercase-sha256"
}
```

The text digest is SHA-256 over the exact UTF-8 completion bytes, without JSON
quotes or an added newline. `model_tree_sha256` is SHA-256 over concatenated,
bytewise sorted lines of the form
`<file-sha256><two spaces><safe-ASCII relative POSIX path><newline>`. The path
alphabet is `A-Z a-z 0-9 . _ / + @ = -`; symlinks, special files, empty trees,
and other names fail.

Set all reviewed bindings before invoking the driver on the remote GPU host:

```sh
export RUSTINFER_E2E_OUTPUT=/append-only-evidence/pr16-python-free-e2e
export RUSTINFER_E2E_IMAGE_ID=sha256:<immutable-image-id>
export RUSTINFER_E2E_SOURCE_REVISION=<clean-40-character-revision>
export RUSTINFER_E2E_SOURCE_ARCHIVE=/artifacts/source.tar
export RUSTINFER_E2E_SOURCE_ARCHIVE_SHA256=<reviewed-digest>
export RUSTINFER_E2E_RELEASE_BINARY=/artifacts/rustinfer
export RUSTINFER_E2E_RELEASE_BINARY_SHA256=<reviewed-digest>
export RUSTINFER_E2E_RELEASE_BUNDLE=/artifacts/rustinfer.tar.gz
export RUSTINFER_E2E_RELEASE_BUNDLE_SHA256=<reviewed-digest>
export RUSTINFER_E2E_MODEL_DIR=/models/reviewed-checkpoint
export RUSTINFER_E2E_MODEL_TREE_SHA256=<reviewed-digest>
export RUSTINFER_E2E_MODEL_REVISION=<immutable-model-revision>
export RUSTINFER_E2E_CONFIG_SHA256=1d556eab73b69c7f11f64c557a2f9c6f440bd4c6b89bb2584a6b498c92603843
export RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH=model.safetensors
export RUSTINFER_E2E_WEIGHTS_SHA256=<reviewed-digest>
export RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH=tokenizer.json
export RUSTINFER_E2E_TOKENIZER_JSON_SHA256=9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
export RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256=51666963fa4cef6fbd450fc7ec5f70e483717757e0fcc2a5956f097d3915c4db
export RUSTINFER_E2E_CORRECTNESS_GOLDEN=/artifacts/python-free-e2e-golden.json
export RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256=<reviewed-digest>
export RUSTINFER_E2E_CORRECTNESS_REPORT=/artifacts/native-e0-correctness-report.json
export RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256=<reviewed-digest>
ci/run_python_free_release_e2e.sh
```

The output directory must not exist. The driver verifies all inputs before
starting CUDA, refuses mutable image references, and creates raw evidence,
the release-gate attestation, post-SIGTERM shutdown metrics, and SHA256SUMS.
Before attestation, it creates an uncompressed deterministic raw-evidence v2
USTAR. Its fixed inventory contains 34 files plus `SHA256SUMS`: the summary and
golden, both shutdown metrics, raw Docker image inspect, the executable copied
from the image, native-manifest/`ldd`/`readelf` output, Python scan output, two
containers' pre/runtime/post inspect snapshots and pre/runtime process tables,
the exact greedy/sampling requests, raw JSON/SSE/metrics HTTP responses, and
the exact cancellation request plus admitted-response prefix. Every regular
file in the supplied model directory is also archived below `model/`; those
bytes must exactly match `model-SHA256SUMS`. The checksum manifest closes over
the complete fixed and dynamic inventory. The attestation remains outside the
tar and binds the tar SHA-256, avoiding a circular self-hash.
The container itself proves real checkpoint load through `/readyz` and
`/v1/models`; greedy prefill/decode and non-stream/SSE parity against the
approved golden; fixed-seed sampling repeatability across two clean starts
(the request ID is part of the RNG stream derivation); disconnect-triggered
cancellation and reclamation; and graceful SIGTERM shutdown with all final KV,
device, and pinned allocation gauges at zero.

The real `serve` invocation intentionally omits `--execution-completion`,
`--residual-rmsnorm`, and `--reduction-profile`. This gate therefore exercises
the production binary's stable optimization defaults rather than restating
their current values on the command line. The checker requires the archived
Docker `Args` array to match that exact default-path invocation and rejects
evidence containing any of those three selectors.
The embedded release manifest must name the same three stable defaults and the
reviewed SHA-256 of `crates/rustinfer-server/src/main.rs`; release preflight
fails if that resolver source drifts. This closes the default-path gate over
the Rust implementation rather than only the launcher/checker argument lists.

It also records the runtime process inventory, reviewed native dependency
manifest, dynamic-loader resolution, absence of Python-family executables and
artifacts, and absence of Python-family processes. The checker parses the
archived ELF instead of trusting a producer digest, cross-checks its DT_NEEDED
entries with both the copied native manifest and `readelf`/`ldd`, replays the
Docker state transitions, parses the raw HTTP JSON and SSE frames, and derives
the completion/cancellation claims from those transcripts. It re-hashes the
source archive, standalone binary, release bundle, every archived and supplied
model file, native E0 correctness report, golden, shutdown files, and raw
evidence; validates the closed schemas; and emits
the exact `rustinfer.release-gate-attestation.v1` check set consumed by the
final release-candidate gate.

The golden cannot self-authorize a new output: it must bind the exact passing
`smollm2-fp32-bf16-native-e0-v2` report, clean candidate revision, immutable
model revision, weights, the native five-file tokenizer aggregate, and the
exact runtime `tokenizer.json`. In addition, any final-candidate replay must
pass the independently reviewed golden SHA-256 to
`validate_bound_raw_archive(..., correctness_golden_sha256=...)`; omitting it
fails closed. The closed raw archive repeats those bindings;
the checker enforces its exact inventory and metadata, validates its internal
checksums, recomputes the five-file tokenizer aggregate from the preserved
model manifest, and is itself SHA-bound by the attestation. The attestation root stays
compatible with the final candidate's closed schema; the transitive provenance
is therefore exposed through `raw_evidence_sha256`, not an unreviewed root
extension.

Local validation is limited to `bash -n` and standard-library unit tests. Do
not run this driver on a developer machine or without the reviewed remote GPU
lane and append-only evidence destination.

Before the final golden is materialized, the development-only independent HF
review follows [`ci/release/HF_GOLDEN_ORACLE_RECEIPTS.md`](../../ci/release/HF_GOLDEN_ORACLE_RECEIPTS.md).
It requires two fresh remote-GPU BF16 eager processes and exact parity across
their four cache-on/cache-off paths. Those receipts and their create-only
approval stay outside the closed E2E, soak, and final-candidate evidence roots.
They approve only the expected completion tokens/text; the final golden is
created later and only after it can bind the clean final `REV` and the exact
passing final native E0 correctness-report SHA-256.
