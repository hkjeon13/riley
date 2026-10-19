# Owned inference completion on A100

This change keeps model arithmetic unchanged and moves completion waits from
individual operations to the enclosing iteration. The explicit
`owned-hf-eager-bf16-v1` profile retains its original per-operation default.
Select the optimized route with:

```sh
--numerics-profile owned-hf-eager-bf16-v1 --execution-completion iteration-batch
```

## What changed

The owned Qwen2.5 Q/K/V bias projections can now execute inside an iteration
command batch. Their prepared cuBLASLt algorithms, FP32 split-K reduction policy,
actual-request GEMM row counts, BF16 rounding and separate normalization remain
unchanged. The native command-batch ledger already retains each plan and its
input, weight, bias, output and workspace until confirmed completion. The runtime
no longer rejects that supported attention-bias path. Owned MLP biases and
unqualified model geometries remain rejected; full-step graph capture remains
unsupported. Subsequent opt-in piecewise Graph work is recorded in
[Graph and batch validation](owned-graph-batch-optimization.md).

Bias-free owned models already exposed iteration completion. Their measured
benefit motivated qualifying the attention-bias path before adding more fusion.
No public HTTP request or response format changes.

## Actual execution and bottleneck

The initially active remote job was a SmolLM2 comparison on physical A100 GPU 2,
already using iteration completion and process-local CUDA 12.9 libraries. It
was left running. The new work used the independently admitted physical GPU 3,
UUID `GPU-b7a1c292-6e3a-1bf2-6e6a-25307835853b`, on `aws-a100`
(`ldccai-a100-02`), with native CUDA 12.4 libraries and SM80 builds.

A separate Qwen3-1.7B C1 profile used identical prompts and sampled 48 generated
tokens across three requests after warmup. These are diagnostic profiler timings,
not HTTP performance acceptance numbers:

| Quantity | Per operation | Per iteration |
|---|---:|---:|
| GPU kernel launches | 35,448 | 35,448 |
| Sum of kernel durations | 261.822 ms | 261.887 ms |
| GPU span | 671.474 ms | 347.028 ms |
| GPU busy share of traced span | 40.57% | 78.52% |
| `cudaStreamSynchronize` calls | 26,086 | 454 |
| Time inside stream synchronization APIs | 384.464 ms | 65.049 ms |

API wait duration overlaps GPU execution and must not be added to kernel time.
The kernel counts and durations support a synchronization bottleneck in this
owned profile. The older 92.8%-busy/GEMM-heavy Mistral trace belongs to a different
optimized arithmetic profile and is not a description of this owned route.

## Numerical and lifecycle controls

- CUDA/runtime/server library and binary regression tests: 474 passed, 40 ignored.
- Independent, checksum-bound real-activation HF component oracle: 44 cases
  match BF16 bytes. The biased projections are checked with synchronous and
  command-batch completion using the same prepared plan.
- A dedicated integer oracle reuses one bias GEMM descriptor with different
  bias buffers in one batch, checks cross-stream exclusion of all operands
  and workspace, repeats execution without CUDA allocations, and verifies
  complete allocation release.
- Teacher-forced Qwen2.5-3B and SmolLM2 controls compare 26 full-vocabulary
  logit rows between completion modes: all 5,844,992 BF16 bytes match exactly.
  This is an equality comparison to the unchanged native route, not a claim
  that every model/library combination is HF-exact.
- HTTP checks compare prompt IDs, generated IDs, text, finish reason and usage
  to the independent pinned HF reference and across native arms. Explicit
  disconnect/retry checks assert KV reclamation and unchanged live allocations.

## Measured native improvement

Five paired fresh-process runs per concurrency, using warm HTTP/SSE requests.
Values below are medians across process runs; speedup is the median paired ratio.

| Concurrency | Before, tok/s | After, tok/s | Paired speedup | TPOT p50 before/after, ms |
|---:|---:|---:|---:|---:|
| 1 | 60.56 | 97.15 | 1.603x | 16.12 / 9.97 |
| 8 | 77.06 | 112.69 | 1.463x | 98.76 / 67.64 |
| 16 | 78.76 | 114.45 | 1.453x | 191.71 / 132.63 |
| 32 | 79.62 | 115.83 | 1.448x | 337.05 / 232.06 |

All twenty pairs improve throughput and all tested TTFT/TPOT p50/p95/p99.
Across forty arms, 1,920 measured responses and 300 separate correctness
responses retain prompt IDs, generated IDs, text, finish reason and usage.
All forty arms finish with zero active/waiting requests, zero allocated KV
blocks, and unchanged Riley-owned live allocation counts/bytes. Eight explicit
disconnects and twelve retries also pass the recovery controls.

The opt-in is validated for this model/device/workload. The original default
and per-operation rollback setting remain available.

## Measurement contract

The Qwen2.5-3B comparison pins checkpoint revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`, tokenizer, native EOS, BF16, greedy
sampling, the same GPU and CPU affinity, and CUDA 12.4 library resolution.
It uses C1/C8/C16/C32, five pairs of fresh server processes, alternating AB/BA,
12 excluded warmup requests and 48 measured requests per arm. The measured
`fixed` corpus contains twelve prompts of 115-130 input tokens with output
budgets of 8/16/24/32 tokens (20 output tokens per request on average); EOS is
not suppressed. Natural prompts are included in correctness controls separately.
TPOT is a per-response mean inferred from SSE arrivals, not an inter-token
latency distribution. P95/P99 are descriptive for this bounded sample.

The baseline is the immutable v14 binary, SHA-256
`513dd54ecb259797d1d7b03b06cbb93cb30135b77fe1e4f11c9d8b1c2aaf14b9`.
The candidate is `riley-iteration-bias-v1`, SHA-256
`31c40a2cb5b9b24a855e3e5ee84eaa3ad611f2aeb4b589d849eb8fb5d1d8e64c`.

Raw plans, requests, server logs, profiler exports, source snapshots and test
receipts are stored under `artifacts/inference-opt-20261016/` locally and
`/mnt/efs/nlp/psyche/riley-inference-opt-20261016/` on the remote host. The remote
build uses its own source copy and Cargo target. Source-hash reconciliation
against v14 also identifies two concurrent guard changes permitting optional
fused SiLU/multiply, plus its test fixture branch. That option is disabled in
every measured arm. The floating-point implementation files and CUDA kernels
match v14; this experiment changes the completion mode and its attention-bias
admission guard. The full compiled-source manifest records these distinctions.

The native comparison does not establish arbitrary-model numerical equivalence,
longer-context performance, RTX 4090 qualification, or soak stability. The
per-operation route remains available for rollback.


## Same-device vLLM comparison

The comparison uses installed vLLM 0.27.1, the same pinned Qwen2.5-3B checkpoint,
GPU 3, CPU affinity 16-23, BF16, native generation EOS, 1024-token sequence bound,
512-token batch/prefill budget, 16-token KV blocks and `concurrency * 64` KV blocks.
Prefix caching is disabled in both engines. Requests explicitly use greedy
sampling with repetition penalty 1.0, top-p 1.0, top-k disabled and zero frequency
and presence penalties. vLLM retains its standard compile/CUDA graph defaults
and FlashAttention backend; the strict Riley profile uses eager execution.
vLLM resolves the CUDA 12.9 wheel libraries while Riley uses CUDA 12.4. These
are actual installed engine routes, not an identical-kernel comparison.

Two fresh-process orders are measured per concurrency: Riley, standard vLLM,
modified vLLM diagnostic; then the reverse order. Each excludes 12 warmup
requests and measures 48 fixed-fixture requests. This is a bounded screening,
not the five-pair native-change acceptance or a broad engine-parity claim.

The process-local `vllm-event-spin` diagnostic changes CUDA event creation to
use nonblocking host waits. It is explicitly modified vLLM and never installed
into the shared environment. It is reported separately because standard vLLM
has exhibited roughly 50 ms/token host event waits on this host. This diagnostic
does not change model arithmetic, but its performance must not be attributed
to the unmodified installation.

The 0.6 vLLM GPU-memory-utilization setting is an allocator-sizing input, not
an equal total-VRAM ceiling. The explicit KV block override is confirmed in
server logs. NVML memory samples include startup as well as request execution;
peak memory is descriptive and is not a steady-state-only measurement.

An initial startup failed because the venv's Ninja executable was absent from
PATH. Another attempt inherited the checkpoint's repetition penalty 1.05 in
vLLM while Riley/reference used 1.0. Both attempts are preserved and excluded
from the aligned comparison. The aligned runner explicitly overrides the
sampling controls while retaining the checkpoint EOS list.

### Aligned results

Medians across two fresh-process runs per cell. Throughput is aggregate output
tokens divided by the measured request window, not per-user token speed. All
24 arms produced 960 measured output tokens from 48 requests.

| Concurrent requests | Riley, tok/s | Standard vLLM, tok/s | Riley / standard vLLM | Modified event-spin vLLM, tok/s | Riley / modified vLLM |
|---:|---:|---:|---:|---:|---:|
| 1 | 96.61 | 19.94 | 4.845x | 135.04 | 0.715x |
| 8 | 113.00 | 152.72 | 0.740x | 751.20 | 0.150x |
| 16 | 114.13 | 289.33 | 0.394x | 1159.93 | 0.098x |
| 32 | 115.66 | 454.65 | 0.254x | 1569.48 | 0.074x |

The C1 standard-vLLM result is affected by the approximately 52 ms response-mean
TPOT observed on this host. Selecting nonblocking event waits reduces C1 TPOT
to approximately 6.6 ms. The modified result exposes the host-wait sensitivity;
it is not a stock-vLLM result or a general engine ranking.

### Latency

Each cell is the median of the two process-run percentiles, in milliseconds.
TPOT is a response-mean SSE estimate; it is not a token-level ITL percentile.

| Concurrency | Engine | TTFT p50 | TTFT p95 | TTFT p99 | TPOT p50 | TPOT p95 | TPOT p99 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Riley | 15.25 | 17.68 | 17.82 | 10.03 | 10.23 | 10.30 |
| 1 | Standard vLLM | 22.91 | 23.76 | 24.86 | 51.69 | 54.24 | 54.33 |
| 1 | Modified event-spin vLLM | 21.78 | 23.45 | 26.30 | 6.62 | 6.65 | 6.80 |
| 8 | Riley | 127.00 | 140.69 | 158.33 | 67.62 | 68.76 | 69.95 |
| 8 | Standard vLLM | 109.04 | 150.56 | 160.29 | 40.24 | 52.40 | 54.21 |
| 8 | Modified event-spin vLLM | 37.91 | 59.51 | 59.93 | 7.95 | 9.14 | 9.46 |
| 16 | Riley | 252.85 | 395.89 | 470.84 | 132.93 | 134.65 | 136.35 |
| 16 | Standard vLLM | 67.95 | 135.36 | 137.34 | 35.19 | 47.91 | 50.30 |
| 16 | Modified event-spin vLLM | 50.02 | 95.10 | 95.86 | 9.45 | 10.74 | 11.46 |
| 32 | Riley | 520.86 | 1497.43 | 1640.03 | 232.96 | 262.74 | 264.47 |
| 32 | Standard vLLM | 99.99 | 194.95 | 195.34 | 26.51 | 45.63 | 50.35 |
| 32 | Modified event-spin vLLM | 84.14 | 172.00 | 172.51 | 12.01 | 15.62 | 16.69 |

### Response controls and memory

The 24-arm comparison adds 1,152 measured and 450 separate reference-control
responses. Every Riley response matches the pinned HF reference. Reference
agreement for vLLM is recorded below as exact generated-token sequence
agreement, not a semantic quality score. Standard and modified vLLM produce
the same total measured output lengths, but only 378/384 measured responses
match exactly by prompt across those two modes. Different asynchronous batch
schedules remain a source of rounding-dependent output differences; the
diagnostic is not claimed to preserve all vLLM responses byte-for-byte.

| Concurrency | Engine | Measured token matches | Separate control token matches | Peak sampled GPU memory, MiB |
|---:|---|---:|---:|---:|
| 1 | Riley | 96/96 | 15/15 | 7023 |
| 1 | Standard vLLM | 80/96 | 10/15 | 6701 |
| 1 | Modified event-spin vLLM | 80/96 | 10/15 | 6701 |
| 8 | Riley | 96/96 | 45/45 | 7279 |
| 8 | Standard vLLM | 81/96 | 33/45 | 7151 |
| 8 | Modified event-spin vLLM | 84/96 | 31/45 | 7069 |
| 16 | Riley | 96/96 | 45/45 | 7569 |
| 16 | Standard vLLM | 82/96 | 32/45 | 7341 |
| 16 | Modified event-spin vLLM | 85/96 | 34/45 | 7341 |
| 32 | Riley | 96/96 | 45/45 | 8149 |
| 32 | Standard vLLM | 84/96 | 33/45 | 8015 |
| 32 | Modified event-spin vLLM | 87/96 | 32/45 | 8015 |

The strict Riley path keeps actual-request GEMM row groups in
`execute_gemm_with_row_groups` and `execute_projection_with_row_groups`. Its
nearly flat throughput as concurrency rises is consistent with that batching
constraint, while vLLM also uses different kernels and graph execution. This
comparison does not quantify each remaining bottleneck. Cross-request GEMM
combination or graph support requires separate profiling and the same numerical
and lifetime gates before adoption.

Raw evidence: `artifacts/inference-opt-20261016/evidence/` contains the aligned
`qwen2-vllm-aligned-two-orders-attempt3/{summary,result,response-validation}.json`,
per-arm request/SSE data, effective argv, logs, loaded library maps and NVML
samples. Both excluded attempts and the pinned reference fixture are retained.
The comparison driver completed successfully and GPU 3 had no compute processes
after teardown.
