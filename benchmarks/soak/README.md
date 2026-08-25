# PR-16 reliability soak

`reliability-soak-v1.json` is the versioned release workload and
`ci/run_release_soak.sh` is its Python-free host driver.  The driver starts the
real `rustinfer serve` executable, changes only `--execution-completion` for the
two rollback arms, sends real HTTP requests, and writes create-only `run.json`
plus append-only `events.jsonl`.  It does not invoke a reference framework or
run an in-process mock.

The checked-in manifest is intentionally not executable evidence: both golden
digests are all-zero placeholders.  Before a release run, copy it outside the
checkout and replace:

- `golden.generated_sha256` with the SHA-256 of the exact UTF-8 completion text
  approved by the correctness gate for the `short` request;
- `golden.provenance_sha256` with the SHA-256 of that immutable correctness
  report.

The checker and runner reject the placeholder.  The materialized manifest's
byte-level SHA-256 is recorded in `run.json`, so changing a request, threshold,
scenario, or golden requires a new run.

## Target contract

Run the driver in the production process namespace.  For a CLI artifact this
is the release host; for an image it is a Python-free test layer derived from
the exact immutable release image.  That layer may add Bash, jq, curl, procps,
util-linux, coreutils and `nvidia-smi`, but must not replace the production
binary or model.  Set `target.kind` to `container` in the materialized manifest
for the latter case.

`GET /metrics` must return this closed JSON object (all values are nonnegative
integers):

```json
{
  "active_requests": 0,
  "waiting_requests": 0,
  "kv_allocated_blocks": 0,
  "allocation": {
    "device_live_count": 0,
    "device_live_bytes": 0,
    "pinned_live_count": 0,
    "pinned_live_bytes": 0
  },
  "counters": {
    "cancellations": 0,
    "disconnects": 0,
    "overloads": 0,
    "dropped_samples": 0
  }
}
```

Counters are monotonic for one server lifetime.  For only the last server
lifetime, the driver maps the new absolute `RUSTINFER_SOAK_FINAL_METRICS_JSON`
path to `RUSTINFER_SHUTDOWN_METRICS_PATH`.  On final shutdown the CLI must
create (never replace) the same metric shape there, after
stream/context/allocation close has completed.  The driver refuses a stale
path and refuses to synthesize allocation zeros.

Every lifecycle transition uses `SIGTERM` and requires exit status zero within
the manifest deadline.  Thus a process that merely dies from the signal cannot
produce passing graceful-restart evidence.  The `near_kv` profile expands its
manifest-bound prompt repetition before transmission to exercise a context
near the 8192-token release limit without checking a huge string into Git.

Each sample includes the target PID and its `/proc` RSS/HWM/fd/thread values, its complete
descendant process inventory, summed `nvidia-smi` per-process VRAM, and the
service metric snapshot.  Each request record includes its client-observed
outcome, HTTP status, latency, and completion-text digest.  Concurrent writers
use a lock to preserve a gap-free sequence and strictly increasing Linux
monotonic timestamps.

## Run and check

Use a new repository-external output directory and provide exact bindings:

```bash
RUSTINFER_SOAK_MANIFEST=/var/tmp/rustinfer-soak-manifest.json \
RUSTINFER_SOAK_OUTPUT=/var/tmp/rustinfer-soak-run001 \
RUSTINFER_SOURCE_REVISION=<full-clean-commit> \
RUSTINFER_SOURCE_ARCHIVE_SHA256=<sha256> \
RUSTINFER_BINARY_SHA256=<sha256> \
RUSTINFER_IMAGE_SHA256=<sha256> \
RUSTINFER_MODEL_SHA256=<sha256> \
RUSTINFER_MODEL_ID=HuggingFaceTB/SmolLM2-135M \
RUSTINFER_MODEL_REVISION=<immutable-revision> \
RUSTINFER_SOAK_FINAL_METRICS_JSON=/var/tmp/rustinfer-final-metrics.json \
ci/run_release_soak.sh

python3 benchmarks/scripts/check_reliability_soak.py \
  --manifest /var/tmp/rustinfer-soak-manifest.json \
  --run-directory /var/tmp/rustinfer-soak-run001 \
  --report /var/tmp/rustinfer-soak-run001.report.json
```

The checker runs outside the production dependency boundary and uses only the
Python standard library.  It fails closed on malformed/duplicate JSON,
non-contiguous events, clock or sample gaps, missing scenarios, request or
failure events outside each scenario's policy, Python descendants, nonzero
final active/waiting/KV/native allocation state, insufficient cancellation,
disconnect or overload evidence, RSS/VRAM plateau or slope excess, dropped
samples, unbounded restart, and restart/rollback mismatch with the bound
golden.
