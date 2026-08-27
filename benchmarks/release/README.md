# Release performance regression gate

`performance-baseline-v1.json` freezes the accepted PR15 command-batch result
for the single supported release lane: SmolLM2-135M BF16, concurrency 1,
prompt 128/output 32 greedy, on the pinned `server-4096` RTX 4090 environment.
It binds the measurement source archive, profiler binary, container image,
model revision/files, correctness report, runtime flags, workload, metrics, and
the append-only PR15 evidence root. The later CLI-default promotion source is
recorded separately and is not misrepresented as a performance measurement.
The metric baseline is the accepted `iteration-batch` candidate arm—not the
rejected-for-release `per-operation` comparison arm—and preserves the exact
aggregate values from the append-only PR15 pair report. The reviewed baseline
also pins the canonical `native_profile._request_identity` SHA-256 to
`e6a99a749c41a8227574c96a1d23f8b7d877d6e75b0df4d99154db1b1921a2e6`;
matching model/workload summaries with different prompt/generated-token
identity are incomparable.

The PR16 candidate document uses schema version
`riley.release-performance-candidate.v1` and is intentionally closed. See
the fixture builder in `benchmarks/scripts/tests/test_check_release_performance.py`
for the exact shape. It must contain:

- a SHA-256 of the baseline file bytes;
- clean candidate git/source-archive, exact `riley-profile` producer binary
  and producer image, release `riley` binary and runtime image,
  correctness-report, weights, and tokenizer bindings;
- the exact baseline model/environment/workload records;
- exactly five SHA-bound raw `riley.native-profile-run.v1` candidate files.
  The checker validates every closed raw record, source/model/environment/
  workload binding, status, trace, warmup/iteration count, and token identity;
- TTFT p95, TPOT p95, E2E median, and median output throughput exactly
  recomputed from those five raw files. Self-asserted aggregates are rejected.

Produce the five candidate raw files with the fail-closed, remote-only runner
documented in
[`ci/release/RELEASE_PERFORMANCE_RUNNER.md`](../../ci/release/RELEASE_PERFORMANCE_RUNNER.md).
It accepts only a clean frozen revision and externally reviewed artifact
digests, runs five fresh network-disabled GPU containers on `server-4096`, and
validates the actual host/container facts plus all five Docker inspect receipt
pairs. The manifest uses the exact reviewed server tool path/digest map, and
its model-tree digest must equal `model.manifest_sha256` in the submitted
optimizer correctness report. Local use is limited to its CPU/static contract
tests; do not run the measurement on a machine without the designated GPU
lane.

Create the candidate, checked report, and canonical raw archive together. The
candidate ID must use the final-gate form
`riley-<major>.<minor>.<patch>-rc<positive integer>`:

```sh
python3 benchmarks/scripts/package_release_performance_evidence.py \
  --baseline benchmarks/release/performance-baseline-v1.json \
  --candidate-id riley-0.1.0-rc1 \
  --recorded-at-utc 2026-08-26T12:34:56Z \
  --source-archive /evidence/source.tar \
  --profile-binary /evidence/riley-profile \
  --release-binary /release/riley \
  --weights /models/smollm2/model.safetensors \
  --tokenizer /models/smollm2/tokenizer.json \
  --correctness-report /evidence/optimization-correctness-report.json \
  --profile-image-id sha256:<measurement-image-digest> \
  --release-image-id sha256:<runtime-image-digest> \
  --run /runner-receipts/run-{1,2,3,4,5}/candidate.json \
  --runner-receipt-root /runner-receipts \
  --output-directory /evidence/release-performance
```

The output directory is published atomically without replacing an existing
path and contains exactly:

- `release-performance-candidate.json`;
- `release-performance-report.json`;
- `release-performance-evidence.tar`, a deterministic sorted, fixed-metadata
  uncompressed USTAR containing the closed v3 runner receipt inventory:
  `runner-manifest.json`, `gpu.csv`, image inspections before/after, each of
  five distinct runs' preflight/container-before/container-after/GPU-monitor/
  candidate/execution receipts, and `SHA256SUMS`. Each canonical execution
  receipt cross-binds a unique capture/container/run identity, the five
  constituent receipt hashes, and Docker Created/StartedAt/FinishedAt,
  exit-zero, and OOM-false state.

Packaging reopens the receipt root with no-follow bounded file descriptors,
checks its checksum manifest, replays the exact Docker command/full
environment/isolation/GPU/mount/state contract, derives 5 x (5 warmups + 30
measured iterations), compares the five `--run` bytes, and then replays the
new archive while preserving the runner manifest for the final RC gate. A
legacy five-JSON, alternate self-authorized tool map, foreign CUDA process, or
self-asserted model-tree receipt fails closed.

Exit `0` means the comparable threshold checks passed. Exit `1` means the
evidence was structurally valid and comparable but at least one threshold
failed; the same three files are preserved for review. Exit `2` means the
inputs were invalid or incomparable, or the create-only publish could not be
completed; no new output directory is intentionally published. A failed or
raced publish may deliberately leave a partial or complete private
`.staging-*` directory or a partial mode-`0600` raw staging file. The packager
never removes or truncates a failed staging path during rollback.
Once the atomic directory rename succeeds, any later I/O or verification
failure preserves the published directory exactly as observed, including any
externally replaced or added entry, for explicit operator inspection; it never
rolls the publication back file by file.

Run the standard-library-only checker on the authorized GPU host after the
candidate measurement has completed. The checker does not start CUDA or load a
model; it hashes the already-produced artifacts and evaluates the document:

```sh
python3 benchmarks/scripts/check_release_performance.py \
  --baseline benchmarks/release/performance-baseline-v1.json \
  --candidate /evidence/release-performance-candidate.json \
  --source-archive /evidence/source.tar \
  --profile-binary /evidence/riley-profile \
  --release-binary /release/riley \
  --weights /models/smollm2/model.safetensors \
  --tokenizer /models/smollm2/tokenizer.json \
  --correctness-report /evidence/correctness-report.json \
  --profile-image-id sha256:<measurement-image-digest> \
  --release-image-id sha256:<runtime-image-digest> \
  --run /runner-receipts/run-{1,2,3,4,5}/candidate.json \
  --runner-receipt-root /runner-receipts \
  --report /evidence/release-performance-report.json
```

Only `status=passed` exits zero. Missing/extra fields, non-finite numbers,
baseline-byte drift, artifact digest mismatch, dirty source, incomplete runs,
or unknown semantic/runtime settings produce `status=error`. Model,
environment, or workload drift produces `status=incomparable`; it cannot be
treated as a regression pass. A comparable candidate fails if either latency
ratio exceeds `1.05` or throughput falls below `0.95`. The output path is
exclusive and is never overwritten.

This baseline makes no claim for other GPUs, drivers/toolkits, models,
concurrency, or sequence lengths. Such a change needs a separately reviewed
baseline rather than relaxing this checker.

The independent real-model clean-runtime release gate and its remote-only
execution contract are documented in [PYTHON_FREE_E2E.md](PYTHON_FREE_E2E.md).
