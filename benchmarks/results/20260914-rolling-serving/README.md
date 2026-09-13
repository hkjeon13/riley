# Rolling decode serving — C32, active32

Same-day prior/new-pair/new-rolling/vLLM comparison on RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, matched720MiB KV payload. Both Riley and vLLM enable prefix caching; Riley keeps512 unique cache pages within2048 physical pages. All Riley lanes use prefill-FFN, paired native transport and adaptive decode; query reuse is off. Only rolling uses `RILEY_ROLLING_DECODE=1`. New pair is the same new binary with that flag0, isolating the selected execution policy from the rebuild.

Shared workload:32 variants,498–518 prompt tokens. Unique workload:552–603 tokens with distinct first16-token pages. Every retained response has32 generated tokens. Per lane:64 warmup+256 retained requests, repeated in reverse order. Total4096 retained requests/131072 tokens. These are closed-loop, two-run screens without statistical significance or long-soak/open-loop qualification. No profiler/build runs during timed lanes. GUI retained; Blender down.

Throughput is output tokens/s. Latencies are ms; TTFT/TPOT P50 and E2E P95/P99. Values are medians of two run-level estimates, not pooled percentiles.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Previous Riley | 8,912.03 | 14.95 | 3.16 | 135.43 | 151.16 | 11.02 | 11.82 |
| shared | New Riley, pair | 8,930.74 | 14.85 | 3.15 | 134.82 | 151.11 | 10.86 | 11.73 |
| shared | New Riley, rolling | 10,027.73 | 14.85 | 2.74 | 123.43 | 142.05 | 10.85 | 12.36 |
| shared | vLLM | 12,349.82 | 29.14 | 1.67 | 89.43 | 91.15 | 3.18 | 6.16 |
| unique | Previous Riley | 3,475.55 | 126.27 | 5.67 | 349.99 | 382.99 | 10.32 | 12.51 |
| unique | New Riley, pair | 3,483.88 | 127.43 | 5.67 | 350.27 | 380.42 | 10.31 | 12.47 |
| unique | New Riley, rolling | 3,540.58 | 126.74 | 5.62 | 343.78 | 375.57 | 9.43 | 12.18 |
| unique | vLLM | 4,911.73 | 41.78 | 5.09 | 274.97 | 328.02 | 6.46 | 10.40 |

Token intervals use actual SSE token arrival timestamps, without interpolation. Multiple tokens observed in one frame can have zero intervals; no fabricated uniform arrival spacing is used.

shared: rolling throughput changes+12.52% versus prior and-18.80% versus vLLM. TPOT changes-13.27% versus prior.

unique: rolling throughput changes+1.87% versus prior and-27.92% versus vLLM. TPOT changes-0.85% versus prior.

All3072 retained Riley responses match the frozen prior binary in prompt/output tokens, text and finish reason. All4096 retained responses are protocol-valid. Each Riley lane executes32 stop cases,32 cancellations before32 tokens, and32 recovery requests outside timing. All stop and recovery outputs match the reference. All16 lanes exit0. vLLM token/text agreement is433/512 shared and158/512 unique; this is not a quality-loss estimate and cross-engine numerical equivalence remains unqualified.

[comparison.json](comparison.json) is recomputed from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_rolling_decode_serving.py benchmarks/results/20260914-rolling-serving`. The verifier checks raw rows, flags, equal KV budgets, controller/client hashes, positive rolling progress/drain counters, stop/cancel/recovery and exits. Binary hashes and launch commands are retained in preparation/launch evidence.

## Implementation and decision

The server keeps rolling state across worker ticks. It reads and samples the completed predecessor, settles its prefix without retiring successor pages, prepares/submits only the new successor while GPU work continues, and returns the committed token events on that tick. It never waits for an entire rolling run before returning output. Stop/cancel or admission/length drain fallbacks retain the first result and publication state until the second step settles, avoiding duplicate output. Runtime result reads bind retained iteration identity, output count/slots and vocabulary before sampling. Graph close/drain still precedes scheduler abort and page retirement.

Enable with `RILEY_ROLLING_DECODE=1` alongside `--decode-window paired-experimental-v1` on the supported V7 GPU-greedy configuration. Unset/0 keeps existing pairs; malformed values and unsupported base configuration are rejected. `RILEY_ROLLING_DECODE prepared=true` and final step/drain counters identify the rolling policy. The inherited base completion label remains the paired native transport label. Server host-phase timers currently cover the ordinary paths rather than all rolling tick work; do not use those partial counters as a complete rolling CPU profile.

Keep rolling opt-in. Shared C32 throughput improves12.52% and TPOT improves13.27% versus prior, but token-interval P99 increases4.53%. Throughput remains18.80% below vLLM shared and27.92% below unique. Cache-heavy C8 behavior is reported separately and must not be generalized to all workloads. Full goal and universal promotion remain unachieved.

The batch has now connected KV reservation extension, scheduler partial settlement, runtime ticket promotion and server streaming. Prior GPU correctness/failure gates are in [rolling-runtime](../20260914-rolling-runtime/README.md). This server step passes105 local server tests,9 CUDA-feature runtime state tests and the release build; actual HTTP serving supplies the new integration evidence. No CUDA kernel arithmetic or Rust/C ABI/CUDA language boundary changed. Remaining validation includes sustained/open-loop workloads, broader models/quality, shared-cache hardware fault injection and multi-GPU/Hopper/Blackwell runtime.

Rollback requires a new server instance with the flag0 after draining the running instance. Do not switch policy in the middle of live tickets. The preserved prior executable is `riley-before-rolling-v1` under the prepared remote root.

## Milestone across concurrency

Same binary in all three screens; C64 has active capacity32. Throughput units: output tokens/s.

| Client concurrency | Workload | Rolling Riley | vLLM | vs prior | vs vLLM |
|---:|---|---:|---:|---:|---:|
| 8 | shared | 5,092.23 | 4,475.51 | +12.50% | +13.78% |
| 8 | unique | 2,680.98 | 2,996.00 | +5.06% | -10.51% |
| 32 | shared | 10,027.73 | 12,349.82 | +12.52% | -18.80% |
| 32 | unique | 3,540.58 | 4,911.73 | +1.87% | -27.92% |
| 64 | shared | 10,054.22 | 14,841.06 | +12.98% | -32.25% |
| 64 | unique | 3,530.51 | 4,854.56 | +1.91% | -27.27% |

[C8 latency table](../20260914-rolling-serving-c8/README.md), [C64 queue/latency table](../20260914-rolling-serving-c64/README.md). Total12288 retained requests/393216 generated tokens;9216 Riley responses match prior. Across the three screens312 stop,312 cancellation and312 recovery cases pass. Final C64 compute inventory has no compute processes. C8 shared is the only tested condition exceeding vLLM throughput; this does not satisfy the overall goal.
