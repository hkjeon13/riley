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
aggregate values from the append-only PR15 pair report.

The PR16 candidate document uses schema version
`rustinfer.release-performance-candidate.v1` and is intentionally closed. See
the fixture builder in `benchmarks/scripts/tests/test_check_release_performance.py`
for the exact shape. It must contain:

- a SHA-256 of the baseline file bytes;
- clean candidate git/source-archive, exact `rustinfer-profile` producer binary
  and producer image, release `rustinfer` binary and runtime image,
  correctness-report, weights, and tokenizer bindings;
- the exact baseline model/environment/workload records;
- exactly five SHA-bound raw `rustinfer.native-profile-run.v1` candidate files.
  The checker validates every closed raw record, source/model/environment/
  workload binding, status, trace, warmup/iteration count, and token identity;
- TTFT p95, TPOT p95, E2E median, and median output throughput exactly
  recomputed from those five raw files. Self-asserted aggregates are rejected.

Run the standard-library-only checker on the authorized GPU host after the
candidate measurement has completed. The checker does not start CUDA or load a
model; it hashes the already-produced artifacts and evaluates the document:

```sh
python3 benchmarks/scripts/check_release_performance.py \
  --baseline benchmarks/release/performance-baseline-v1.json \
  --candidate /evidence/release-performance-candidate.json \
  --source-archive /evidence/source.tar \
  --profile-binary /evidence/rustinfer-profile \
  --release-binary /release/rustinfer \
  --weights /models/smollm2/model.safetensors \
  --tokenizer /models/smollm2/tokenizer.json \
  --correctness-report /evidence/correctness-report.json \
  --profile-image-id sha256:<measurement-image-digest> \
  --release-image-id sha256:<runtime-image-digest> \
  --run /evidence/candidate-{1,2,3,4,5}.json \
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
