# Projection extended serving — C32, active32

Same frozen SmolLM2-135M BF16 model, RTX4090, context1024, batch/chunk512, equal720MiB KV payload and prefix caching. Every Riley lane uses rolling decode, prefill FFN and adaptive decode; projection alone enables the new backend. Candidate binary00969752 and priorb51b1c4 are unchanged from the short screens. Actual full hashes and launch budgets are in comparison.json. The model runtime remains Rust → C ABI → CUDA.

Each lane executes256 warmup+8192 retained requests, with two reversed orders and shared/unique workloads:131,072 retained requests and4,194,304 output tokens. Shared uses32 prefix variants; unique uses distinct initial16-token pages. This fixed-count closed-loop comparison is separate from the earlier256-request screens and is not an open-loop/long-soak or other-hardware qualification.

All131,072 retained responses pass the protocol checks; all98,304 Riley responses match prior reference token IDs, text and finish reason. Stop/cancel/recovery each pass96 cases across the three Riley lanes. Every measured server lane exits0. The verifier independently reconstructs token/text/usage from captured SSE frames, compares the fixture identity and validates arrival timestamps against phase bounds. The transport [DONE] condition still relies on the hash-bound client because that marker is not saved among the frames.

## Aggregated comparison

Output tokens/s; all latency values ms. TTFT/TPOT P50 and E2E/actual token-interval tails are medians of two run-level estimates, not pooled percentiles or confidence intervals.

| Workload | Lane | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 9,360.07 | 14.91 | 2.79 | 117.08 | 362.44 | 11.12 | 12.09 |
| shared | control | 9,319.65 | 14.63 | 2.76 | 124.80 | 402.90 | 10.86 | 11.82 |
| shared | projection | 9,693.23 | 14.00 | 2.70 | 109.52 | 364.43 | 10.52 | 11.46 |
| shared | vllm | 10,719.27 | 30.10 | 1.80 | 111.17 | 394.62 | 3.95 | 7.22 |
| unique | prior | 3,401.94 | 71.79 | 7.20 | 346.20 | 499.70 | 10.08 | 11.85 |
| unique | control | 3,407.20 | 71.94 | 7.17 | 346.40 | 510.61 | 10.04 | 11.69 |
| unique | projection | 3,692.44 | 66.23 | 6.61 | 319.25 | 481.28 | 9.36 | 11.05 |
| unique | vllm | 4,607.67 | 48.70 | 5.18 | 288.68 | 477.31 | 6.73 | 11.87 |

| Workload | Projection vs prior | Projection vs control | Projection vs vLLM |
|---|---:|---:|---:|
| shared | +3.56% | +4.01% | -9.57% |
| unique | +8.54% | +8.37% | -19.86% |

## Individual runs and host observations

PSI counters describe the entire shared host; percentages are elapsed-phase delta ratios, not engine-specific stall fractions. They do not prove causality. Request timing excludes the before/after reads and result serialization.

| Run | tok/s | Phase seconds | CPU some % | IO some % | IO full % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 9355.48 | 28.027 | 2.96 | 6.81 | 5.49 |
| shared-p0-control | 9298.38 | 28.197 | 1.65 | 6.19 | 5.12 |
| shared-p0-projection | 9666.20 | 27.124 | 1.41 | 6.76 | 5.92 |
| shared-p0-vllm | 10713.38 | 24.477 | 1.26 | 6.37 | 5.39 |
| shared-p1-vllm | 10725.16 | 24.447 | 1.36 | 8.59 | 7.36 |
| shared-p1-projection | 9720.26 | 26.976 | 1.31 | 10.36 | 8.69 |
| shared-p1-control | 9340.93 | 28.070 | 1.46 | 12.40 | 10.31 |
| shared-p1-prior | 9364.66 | 27.997 | 1.36 | 10.41 | 8.86 |
| unique-p0-prior | 3395.80 | 77.201 | 1.14 | 19.82 | 17.26 |
| unique-p0-control | 3406.53 | 76.959 | 1.13 | 13.55 | 11.85 |
| unique-p0-projection | 3687.93 | 71.086 | 1.27 | 7.51 | 6.60 |
| unique-p0-vllm | 4657.91 | 56.286 | 1.24 | 10.78 | 9.41 |
| unique-p1-vllm | 4557.43 | 57.526 | 1.32 | 11.68 | 10.11 |
| unique-p1-projection | 3696.95 | 70.913 | 1.06 | 26.05 | 23.07 |
| unique-p1-control | 3407.87 | 76.929 | 1.10 | 24.24 | 21.42 |
| unique-p1-prior | 3408.09 | 76.923 | 1.19 | 16.76 | 14.72 |

## Evidence and remaining gates

No default promotion is made by this report. Assess the measured throughput and latency together; a native-kernel gain or one favorable latency statistic does not achieve the overall vLLM goal. Other concurrency conditions, longer stability/open-loop tests, other models and multi-GPU/Hopper/Blackwell runtime remain separate gates.

All three Blender instances were restored and scene queries succeeded; the public3d/3dsol/3dfable static viewers stayed up. No profiler or build overlapped measurement. `execution.json` and `blender-restored.json` preserve completion evidence.

Recompute comparison.json using `python3 benchmarks/analysis/export_projection_long_serving.py benchmarks/results/20260914-projection-long-serving`. The evidence directory contains17 bounded lossless archives with a manifest, source/client/binary identities, complete request evidence and per-phase host counters. Request counts, cache budgets, outputs and frames are checked rather than trusting success flags alone.

## C32 assessment

Actual retained phases lasted24.4–77.2seconds. Projection throughput improves3.56% versus frozen prior on shared and8.54% on unique; versus same-binary control the improvements are4.01% and8.37%. TTFT, TPOT, E2E P95 and token-interval tails improve versus prior in both workloads. Shared E2E P99 rises from362.44 to364.43ms (+0.55%); unique E2E P99 falls from499.70 to481.28ms. These two-order observations support proceeding to C8/C64 with the same fixed candidate, but do not provide a confidence interval or long-soak proof.

The projection candidate is9.57% below vLLM shared throughput and19.86% below unique throughput. Shared TTFT and E2E P95/P99 are lower, while TPOT and token-interval tails remain higher. Unique TTFT, TPOT and E2E tails remain higher than vLLM; token interval P99 alone is lower. No overall goal or default promotion is declared.

Prompt token ranges in this extended fixture are {'shared': [498, 518], 'unique': [552, 623]}. Unique prompt indices extend to8447, so compare engines within this extended fixture rather than directly merging with shorter screens.

Long shared E2E P99 is much larger than in the short screens for all engines. The client retains every decoded SSE frame across a whole phase; client allocation/collection or shared-host stalls may contribute, but this run has no GC timing evidence and does not identify the cause. Before extending expensive C8/C64 runs, instrument client pauses in a bounded diagnostic with the same frozen binary. Do not subtract inferred pauses or relabel these measured latencies.
