# Native paired-profile evidence

This directory defines the PR-15 native profiling evidence lane. It is
intentionally independent of the canonical PR-01 benchmark matrix, lane
manifests, and repeatability checker. Profiling establishes attribution and a
paired optimization decision; unprofiled end-to-end evidence remains the
authoritative product-latency result.

Each process writes one JSON document conforming to
`benchmarks/schemas/native-profile-run.schema.json`. A complete experiment has
five baseline and five candidate documents with `pair_index` values 1 through
5. Every pair must use the exact same clean source commit, executable bytes,
correctness gate, GPU/host/software environment, workload, and request token
identities. The implementation ID and value of the named runtime flag identify
the two arms; the flag name is common and its values must differ.

Each independent process must discard at least five warmup trials and retain at
least thirty measured trials. The checker rejects a shorter run, an incomplete
fixed-length response, or a trace whose retained count is not exactly the sum
of measured requests and measured scheduler iterations.
The v1 evidence bounds are concurrency 1–8, prompt length 1–8192, output length
2–512, warmups 5–100, and measured trials 30–100. At least two output tokens
are required so every measured request contains a real decode/TPOT interval.

`prompt_u32le_sha256` and `generated_u32le_sha256` are SHA-256 over the exact
concatenation of token IDs encoded individually as unsigned 32-bit
little-endian values, with no prefix or separator. Request records are closed:
they contain only a numeric `input_index`, those two hashes, token counts, and
TTFT/TPOT/E2E numbers. Raw prompts, generated text, token arrays, request IDs,
and pointer values are not evidence fields.

Build and run the native evidence producer on the target GPU host (the binary
has no Python runtime dependency):

```bash
cargo build --release --locked \
  --features bench,cuda \
  --bin riley-profile
target/release/riley-profile --help
```

Every source, GPU, host, software, and workload provenance field is a required
CLI value. The runner accepts two closed E0 comparisons: `residual_rmsnorm`
binds baseline/candidate to `separate`/`fused`, and `execution_completion`
binds them to `per-operation`/`iteration-batch`. The latter fixes residual
RMSNorm to `separate`, so the two arms differ only at the completion boundary.
The checker also binds these pairs to
`pr15-fused-residual-rmsnorm-exact-v1` and
`pr15-iteration-command-batch-exact-v1`, respectively; a performance report
cannot reuse the other candidate's correctness gate.
Use the exact same release binary for both arms. The runner verifies the declared model ID, revision,
dtype, tokenizer checksum, and single-shard weight checksum against the loaded
checkpoint manifest before initializing CUDA.

For each trial the runner selects `--concurrency` distinct corpus records whose
`target_prompt_tokens` equals `--prompt-tokens`, encodes each with the native
tokenizer and `add_special_tokens=true`, then repeats or truncates the encoded
ID sequence to the exact target length. Warmup traces are discarded. Measured
trials are flattened to `concurrency * measured_iterations` request rows;
unknown launch/copy/allocation counters are emitted as `unmeasured` with a JSON
`null` value. Output defaults to stdout. A non-`-` `--output` path is created
exclusively and is never overwritten. No success document is emitted unless
all trials and explicit zero-allocation cleanup succeed.

All host, CUDA, iteration, launch, transfer, and allocation aggregates carry a
`validity` discriminator. A measured value is numeric; an unmeasured value is
JSON `null`. The declared primary metric must be measured in all ten runs.
Trace storage is bounded by `trace.capacity`; a passing pair requires zero
failures and zero dropped trace records.

Run the standard-library-only checker with explicit evidence paths:

```bash
python3 benchmarks/scripts/check_native_profile_pair.py \
  --baseline /evidence/baseline-{1,2,3,4,5}.json \
  --candidate /evidence/candidate-{1,2,3,4,5}.json \
  --report /evidence/native-profile-pair-report.json
```

The checker emits a document conforming to
`benchmarks/schemas/native-profile-pair-report.schema.json`. It computes R7
medians and p95s for both arms, preserves per-run pair statistics, and fails
unless all of these hold:

- candidate TTFT p95 is at most `1.05 * baseline`;
- candidate TPOT p95 is at most `1.05 * baseline`;
- candidate median throughput is at least `0.95 * baseline`;
- both the arm-level median and median paired improvement in the declared
  primary metric are at least 5%;
- all ten runs succeeded without trace loss.

Malformed input returns `status=error`; a provenance, workload, runtime-binding,
or token-identity mismatch returns `status=incomparable`; a threshold failure
returns `status=failed`. Only `status=passed` exits zero. The optional report
path is created exclusively and is never overwritten.

The remote-only `benchmarks/scripts/profile_http_one_token.sh` sentinel records
kernel attribution for each arm. It requires the exact source revision, source
archive SHA-256, container image SHA-256, and correctness-report SHA-256 through
the corresponding `RILEY_PROFILE_*` environment variables. Both
`RILEY_PROFILE_RESIDUAL_RMSNORM` and
`RILEY_PROFILE_EXECUTION_COMPLETION` are recorded and passed explicitly.
Its final `SHA256SUMS` binds the executable, runtime flags, source/container provenance,
checkpoint file digests, GPU/toolkit environment, NCU CSV, the validated
fixed-token HTTP response, and server logs. It refuses to overwrite an evidence file and
uses bounded response and process-shutdown waits.
Run it once with `RILEY_PROFILE_OUTPUT_TOKENS=1` and once with `=2` for
each arm. The first trace isolates prefill-to-first-token work; the incremental
kernel inventory in the two-token trace contains one decode iteration and is
the decode/TPOT attribution sentinel. Each trace collects kernel duration,
registers per thread, register occupancy limit, achieved active-warps ratio,
and DRAM read/write bytes so launch saving, register pressure, occupancy, and
memory traffic can be reviewed together. The completion candidate intentionally
keeps the kernel inventory unchanged; its benefit is measured by the paired
host execution metric rather than inferred from kernel duration.
