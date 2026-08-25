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
The container itself proves real checkpoint load through `/readyz` and
`/v1/models`; greedy prefill/decode and non-stream/SSE parity against the
approved golden; fixed-seed sampling repeatability across two clean starts
(the request ID is part of the RNG stream derivation); disconnect-triggered
cancellation and reclamation; and graceful SIGTERM shutdown with all final KV,
device, and pinned allocation gauges at zero.

It also records the runtime process inventory, reviewed native dependency
manifest, dynamic-loader resolution, absence of Python-family executables and
artifacts, and absence of Python-family processes. The checker re-hashes the
source archive, standalone binary, release bundle, complete model tree,
weights, tokenizer, native E0 correctness report, golden, shutdown files, and
raw evidence; validates the closed schemas; and emits
the exact `rustinfer.release-gate-attestation.v1` check set consumed by the
final release-candidate gate.

The golden cannot self-authorize a new output: it must bind the exact passing
`smollm2-fp32-bf16-native-e0-v2` report, clean candidate revision, immutable
model revision, weights, the native five-file tokenizer aggregate, and the
exact runtime `tokenizer.json`. The closed raw evidence repeats those
bindings and is itself SHA-bound by the attestation. The attestation root stays
compatible with the final candidate's closed schema; the transitive provenance
is therefore exposed through `raw_evidence_sha256`, not an unreviewed root
extension.

Local validation is limited to `bash -n` and standard-library unit tests. Do
not run this driver on a developer machine or without the reviewed remote GPU
lane and append-only evidence destination.
