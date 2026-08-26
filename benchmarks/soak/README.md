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
  from the independently reviewed Python-free E2E correctness golden for the
  exact `short` request; and
- `golden.provenance_sha256` with the byte SHA-256 of the submitted passing
  native E0 correctness report.

The checker and runner reject the placeholder. Every standalone check,
package, and raw replay also requires both trusted files. The checker derives
both hashes rather than accepting caller-supplied digest strings, requires the
E2E golden to hash the same native report, and cross-binds source, model,
prompt, max-token, and greedy-generation identity. A manifest and event stream
rewritten around an arbitrary completion therefore cannot self-authorize. The
checker also pins the
canonical checked-in contract after normalizing only those two golden fields;
changing a request, threshold, duration, scenario, or target cannot define an
easier release lane. The exact materialized manifest SHA-256 is recorded in
`run.json`, so changing either approved golden digest requires a new run.

`cycle_interval_ms` bounds raw request evidence growth without sampling or
aggregating away any request.  It is part of the bound manifest; burst-idle
requires at least one second, while long and near-KV requests naturally
dominate their configured interval. The overload arm uses 96 concurrent
requests, exceeding the release defaults of 8 active plus 64 waiting slots, so
the required client- and service-observed 429 responses are actually induced.

## Target contract

Run the driver in the production process namespace. For a CLI artifact this
is the release host; for an image it is a Python-free test layer derived from
the exact immutable release image. That layer may add Bash, jq, curl, procps,
util-linux, coreutils and `nvidia-smi`, but must not replace the production
binary or model. The reviewed v1 manifest keeps `target.kind=process` because
the driver runs inside that test layer and observes the real server process
directly; changing the target kind would change the pinned release contract.

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
    "dropped_observations": 0
  }
}
```

Counters are monotonic for one server lifetime. `dropped_observations` is the
number of request summaries evicted from the bounded diagnostics ring; a
nonzero value is expected under a long load and is not a monitoring sample
loss. Evidence sample loss is checked independently through sequence numbers,
monotonic time gaps, and each event's `sample_dropped` flag. For only the last server
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

`RUSTINFER_MODEL_SHA256` is the same canonical model-tree digest used by the
Python-free E2E gate: SHA-256 of bytewise path-sorted lines formatted as
`<file-sha256><two spaces><relative POSIX path>\n`. The driver recomputes it
from a symlink-free regular-file tree before starting the server, and the final
candidate requires the soak value to equal the E2E model-tree binding.

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
  --correctness-golden /evidence/python-free-e2e-golden.json \
  --native-correctness-report /evidence/native-correctness-report.json \
  --report /var/tmp/rustinfer-soak-run001.report.json

python3 benchmarks/scripts/package_reliability_soak_evidence.py \
  --manifest /var/tmp/rustinfer-soak-manifest.json \
  --run-directory /var/tmp/rustinfer-soak-run001 \
  --correctness-golden /evidence/python-free-e2e-golden.json \
  --native-correctness-report /evidence/native-correctness-report.json \
  --output /var/tmp/rustinfer-soak-run001.evidence.tar
```

The create-only packager refuses a non-passing run. It preserves exactly the
materialized manifest plus the run directory's `run.json` and `events.jsonl`
in a deterministic uncompressed USTAR with canonical ownership, mode, order,
and timestamps. An internal bytewise-sorted `SHA256SUMS` covers all three raw
payloads. The loader rejects additional files, links, special members,
noncanonical metadata or tar encoding, checksum drift, and oversized inputs.
It then reconstructs the existing run-directory contract and recomputes the
report; the final release-candidate gate requires that result to equal the
separately submitted report exactly. Report v2 records the trusted E2E golden,
generated-text, and native-report hashes in `bindings.trusted_correctness`.
Legacy v1 reports and replay calls without both trusted artifacts fail closed.

The checker runs outside the production dependency boundary and uses only the
Python standard library.  It fails closed on malformed/duplicate JSON,
non-contiguous events, clock or sample gaps, missing scenarios, request or
failure events outside each scenario's policy, Python descendants, nonzero
final active/waiting/KV/native allocation state, insufficient cancellation,
disconnect or overload evidence, RSS/VRAM plateau or slope excess, dropped
samples, a scenario shorter than its reviewed duration, samples that do not
span that duration, drift in every success from a golden-only scenario,
unbounded restart, and restart/rollback mismatch with the bound golden.
