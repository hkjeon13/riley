# GQA staging serving — C32, active32

Same-day frozen rolling Riley / new binary control / new GQA staging / vLLM0.27.1. RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, equal720MiB KV payload and prefix caching enabled on both engines. Riley cache limit512 unique pages within2048 physical pages. Every Riley lane uses rolling decode, prefill FFN and adaptive decode; query reuse is off. Only GQA lane sets `RILEY_MIXED_GQA_STAGING=1`.

Shared32 variants have498–518 prompt tokens; unique prompts have552–603 tokens and distinct first16-token pages. Every retained request generates32 tokens. Per lane64 warmup+256 retained, two reverse orders,4096 retained requests/131072 tokens total. Closed-loop, two-run screen; no statistical significance or soak/open-loop qualification is claimed. Blender was paused during GPU measurement; the virtual display and public static viewers remained up. No profiler/build overlaps timing.

Throughput is output tokens/s; latency columns are ms. TTFT/TPOT are P50, E2E and actual token intervals include P95/P99. Values are medians of two run-level estimates, not pooled percentiles. Token arrivals use actual SSE timestamps with no interpolation.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Prior rolling Riley | 10,021.54 | 14.84 | 2.76 | 121.40 | 137.84 | 10.92 | 11.70 |
| shared | New Riley control | 9,941.24 | 14.93 | 2.75 | 122.48 | 138.60 | 10.96 | 11.91 |
| shared | GQA staging Riley | 9,870.37 | 15.20 | 2.78 | 126.00 | 145.70 | 11.04 | 12.09 |
| shared | vLLM | 11,979.72 | 30.05 | 1.70 | 96.50 | 98.17 | 3.44 | 6.27 |
| unique | Prior rolling Riley | 3,532.37 | 125.94 | 5.65 | 344.43 | 376.04 | 9.60 | 12.27 |
| unique | New Riley control | 3,538.78 | 125.47 | 5.61 | 344.35 | 375.77 | 9.50 | 12.18 |
| unique | GQA staging Riley | 3,498.92 | 126.38 | 5.54 | 350.81 | 380.54 | 9.49 | 12.79 |
| unique | vLLM | 4,842.16 | 41.69 | 5.11 | 287.22 | 348.18 | 6.71 | 10.87 |

## Decision and verification

shared: candidate throughput changes-1.51% against prior and-17.61% against vLLM.

unique: candidate throughput changes-0.95% against prior and-27.74% against vLLM.

No promotion. Native improvement did not carry into this serving screen. Actual candidate model execution must be profiled before assigning a cause; lower occupancy or tensor distributions are hypotheses, not measured explanations. Keep the opt-in backend and existing rolling control. Overall serving goal is unachieved.

All3072 retained Riley responses match prior prompt/output tokens, text and finish reason; all4096 retained responses are protocol-valid. Each of three Riley lanes passes32 stop,32 early cancellation and32 recovery cases. All16 measured server lanes exit0. The verifier reports cross-engine token agreement separately; greedy equivalence to prior does not prove broad numerical quality.

[comparison.json](comparison.json) is derived from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_gqa_staging_serving.py benchmarks/results/20260914-gqa-staging-serving`. The archive includes the exact measured client source and its preparation hash. Its imported prefix (imports plus all helper functions) is byte-identical to the canonical client; only the unused standalone `main` CLI differs. The verifier checks both facts rather than accepting an unexplained hash mismatch. The first controller attempt failed before server launch because of a tuple environment key; it was fixed before the final v2 screen. No failed measurements are mixed into the table.

The release hash and model tests are in [model integration](../20260914-gqa-staging-model/README.md). Actual GPU tests and full-model memcheck passed; initial Blender restoration failed separately and was corrected. Current lifecycle receipts verify scene-query recovery. C8 and client64/active32 comparisons are separate conditions. Additional model/hardware, Q/K fault injection and sustained/open-loop validation remain incomplete.

## C8/C32/C64 milestone

Same model and matched budgets within each condition; C64 has 64 clients and 32 active slots on both engines. Throughput in output tokens/s.

| Clients | Workload | Prior Riley | GQA Riley | vLLM | GQA vs prior | GQA vs vLLM |
|---:|---|---:|---:|---:|---:|---:|
| 8 | shared | 5,116.05 | 5,064.81 | 4,569.57 | -1.00% | +10.84% |
| 8 | unique | 2,671.36 | 2,684.32 | 2,965.71 | +0.49% | -9.49% |
| 32 | shared | 10,021.54 | 9,870.37 | 11,979.72 | -1.51% | -17.61% |
| 32 | unique | 3,532.37 | 3,498.92 | 4,842.16 | -0.95% | -27.74% |
| 64 | shared | 10,010.30 | 9,953.50 | 14,249.99 | -0.57% | -30.15% |
| 64 | unique | 3,527.72 | 3,476.44 | 4,745.00 | -1.45% | -26.73% |

Native graph gains of 2.30–16.90% did not carry into serving. Keep `RILEY_MIXED_GQA_STAGING` off by default. Total: 12,288 retained requests, 393,216 output tokens, 9,216 reference-exact Riley responses, and 312 each stop/cancel/recovery cases. All 48 measured server lanes exited zero. [C8 latency table](../20260914-gqa-staging-serving-c8/README.md) and [C64 latency table](../20260914-gqa-staging-serving-c64/README.md) retain tail regressions. The overall vLLM serving goal remains unmet.
