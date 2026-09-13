# GQA staging serving — C64, active32

Same-day frozen rolling Riley / new binary control / new GQA staging / vLLM0.27.1. RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, equal720MiB KV payload and prefix caching enabled on both engines. Riley cache limit512 unique pages within2048 physical pages. Every Riley lane uses rolling decode, prefill FFN and adaptive decode; query reuse is off. Only GQA lane sets `RILEY_MIXED_GQA_STAGING=1`.

Shared32 variants have498–518 prompt tokens; unique prompts have552–603 tokens and distinct first16-token pages. Every retained request generates32 tokens. Per lane64 warmup+256 retained, two reverse orders,4096 retained requests/131072 tokens total. Closed-loop, two-run screen; no statistical significance or soak/open-loop qualification is claimed. Blender was paused during GPU measurement; the virtual display and public static viewers remained up. No profiler/build overlaps timing.

Throughput is output tokens/s; latency columns are ms. TTFT/TPOT are P50, E2E and actual token intervals include P95/P99. Values are medians of two run-level estimates, not pooled percentiles. Token arrivals use actual SSE timestamps with no interpolation.

There are 64 clients and 32 active slots in both engines; this condition includes admission queueing.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Prior rolling Riley | 10,010.30 | 116.35 | 2.77 | 212.55 | 227.64 | 10.98 | 11.76 |
| shared | New Riley control | 9,949.94 | 116.29 | 2.77 | 220.02 | 236.31 | 11.09 | 11.93 |
| shared | GQA staging Riley | 9,953.50 | 116.50 | 2.77 | 216.55 | 231.86 | 11.15 | 11.95 |
| shared | vLLM | 14,249.99 | 76.38 | 1.90 | 156.03 | 161.03 | 4.49 | 7.18 |
| unique | Prior rolling Riley | 3,527.72 | 402.37 | 5.62 | 677.42 | 702.52 | 9.58 | 12.70 |
| unique | New Riley control | 3,528.44 | 403.91 | 5.62 | 672.02 | 698.03 | 9.57 | 12.27 |
| unique | GQA staging Riley | 3,476.44 | 409.64 | 5.72 | 686.95 | 711.68 | 9.65 | 12.70 |
| unique | vLLM | 4,745.00 | 251.25 | 5.26 | 505.77 | 563.51 | 7.15 | 12.33 |

## Decision and verification

No promotion: GQA staging has no consistent serving improvement over prior or the same-binary control. Two reversed-order runs are a screen, not statistical significance or sustained qualification.

All 3072 retained Riley responses exactly match prior prompt/output tokens, text and finish reason; all 4096 responses are protocol-valid. The three Riley lanes each pass 64 stop, 64 early-cancellation and 64 recovery cases. All 16 measured server lanes exit zero. Cross-engine agreement is reported separately.

Generate [comparison.json](comparison.json) from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_gqa_staging_serving.py benchmarks/results/20260914-gqa-staging-serving-c64`. The archive binds the exact measured client snapshot to its preparation hash and verifies byte-identical imported helper code against the canonical client. The combined C8/C64 lifecycle and successful Blender recovery receipts are in [the C64 directory](../20260914-gqa-staging-serving-c64/). Public static viewers stayed up.

See [model gate](../20260914-gqa-staging-model/README.md) for source/binary hashes and [milestone comparison](../20260914-gqa-staging-serving/README.md#c8c32c64-milestone). Additional models, hardware runtime, Q/K nonfinite injection and sustained/open-loop validation remain incomplete.
