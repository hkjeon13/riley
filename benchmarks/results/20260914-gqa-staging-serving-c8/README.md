# GQA staging serving — C8, active8

Same-day frozen rolling Riley / new binary control / new GQA staging / vLLM0.27.1. RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, equal720MiB KV payload and prefix caching enabled on both engines. Riley cache limit512 unique pages within2048 physical pages. Every Riley lane uses rolling decode, prefill FFN and adaptive decode; query reuse is off. Only GQA lane sets `RILEY_MIXED_GQA_STAGING=1`.

Shared32 variants have498–518 prompt tokens; unique prompts have552–603 tokens and distinct first16-token pages. Every retained request generates32 tokens. Per lane64 warmup+256 retained, two reverse orders,4096 retained requests/131072 tokens total. Closed-loop, two-run screen; no statistical significance or soak/open-loop qualification is claimed. Blender was paused during GPU measurement; the virtual display and public static viewers remained up. No profiler/build overlaps timing.

Throughput is output tokens/s; latency columns are ms. TTFT/TPOT are P50, E2E and actual token intervals include P95/P99. Values are medians of two run-level estimates, not pooled percentiles. Token arrivals use actual SSE timestamps with no interpolation.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Prior rolling Riley | 5,116.05 | 12.66 | 1.21 | 50.88 | 52.30 | 1.69 | 3.58 |
| shared | New Riley control | 5,071.60 | 13.08 | 1.21 | 51.56 | 52.88 | 1.73 | 3.59 |
| shared | GQA staging Riley | 5,064.81 | 13.26 | 1.21 | 52.64 | 53.55 | 1.70 | 3.63 |
| shared | vLLM | 4,569.57 | 14.78 | 1.29 | 61.69 | 68.41 | 1.63 | 2.35 |
| unique | Prior rolling Riley | 2,671.36 | 56.24 | 1.26 | 96.64 | 101.72 | 1.58 | 4.78 |
| unique | New Riley control | 2,674.08 | 56.29 | 1.27 | 96.32 | 100.20 | 1.51 | 4.77 |
| unique | GQA staging Riley | 2,684.32 | 54.00 | 1.31 | 95.95 | 96.86 | 3.13 | 5.34 |
| unique | vLLM | 2,965.71 | 27.71 | 1.86 | 98.16 | 106.36 | 4.96 | 5.60 |

## Decision and verification

No promotion: GQA staging has no consistent serving improvement over prior or the same-binary control. Two reversed-order runs are a screen, not statistical significance or sustained qualification.

Unique-workload token interval P95 rises from 1.58 ms to 3.13 ms against prior, despite lower TTFT. Shared-workload throughput exceeds vLLM, but token interval P99 remains worse; this is not an across-metric win.

All 3072 retained Riley responses exactly match prior prompt/output tokens, text and finish reason; all 4096 responses are protocol-valid. The three Riley lanes each pass 8 stop, 8 early-cancellation and 8 recovery cases. All 16 measured server lanes exit zero. Cross-engine agreement is reported separately.

Generate [comparison.json](comparison.json) from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_gqa_staging_serving.py benchmarks/results/20260914-gqa-staging-serving-c8`. The archive binds the exact measured client snapshot to its preparation hash and verifies byte-identical imported helper code against the canonical client. The combined C8/C64 lifecycle and successful Blender recovery receipts are in [the C64 directory](../20260914-gqa-staging-serving-c64/). Public static viewers stayed up.

See [model gate](../20260914-gqa-staging-model/README.md) for source/binary hashes and [milestone comparison](../20260914-gqa-staging-serving/README.md#c8c32c64-milestone). Additional models, hardware runtime, Q/K nonfinite injection and sustained/open-loop validation remain incomplete.
