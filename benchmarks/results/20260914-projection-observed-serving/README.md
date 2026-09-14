# Projection C32 repeat with per-phase host observations

Candidate and frozen prior binary, model, fixtures and serving budgets are unchanged from [the first screen](../20260914-prefill-projection-serving/README.md). C32/active32, SmolLM2-135M BF16, RTX4090, batch/chunk512, equal720MiB KV payload, prefix caching, rolling/prefill-FFN/adaptive decode on all Riley lanes. The candidate alone enables the projection pipeline. Four lanes × two workloads × two reversed orders,64 warmup+256 retained each. The new controller reads Linux PSI counters immediately before/after each client phase, outside request timing, without launching a profiler.

All4096 retained responses pass protocol; all3072 Riley responses match the frozen reference. Stop/cancel/recovery each96 cases pass and all16 measured server lanes exit0. Three Blender sessions are restored with successful scene queries; the public viewers stay up. No runtime implementation changed for this repeat.

Latencies are ms. TTFT/TPOT P50 and E2E/token-interval P95/P99 are medians of two run estimates, not pooled percentiles.

| Workload | Lane | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 9,927.17 | 14.98 | 2.76 | 125.32 | 141.66 | 11.05 | 12.00 |
| shared | control | 9,407.99 | 18.69 | 2.81 | 134.27 | 148.05 | 10.95 | 13.42 |
| shared | projection | 10,275.01 | 14.21 | 2.68 | 119.41 | 136.73 | 10.47 | 11.73 |
| shared | vllm | 11,802.51 | 31.57 | 1.70 | 96.21 | 98.06 | 3.30 | 6.32 |
| unique | prior | 3,519.17 | 127.05 | 5.67 | 350.05 | 377.98 | 9.46 | 12.68 |
| unique | control | 3,519.23 | 129.48 | 5.63 | 349.08 | 374.76 | 9.58 | 12.33 |
| unique | projection | 3,797.72 | 120.09 | 5.10 | 321.97 | 343.45 | 9.42 | 12.41 |
| unique | vllm | 4,664.61 | 43.50 | 5.32 | 294.33 | 355.51 | 7.54 | 12.87 |

## Run duration and host pressure

Pressure percentages are cumulative PSI microsecond deltas divided by the observed phase duration. They describe the entire shared host, not Riley stall time; there is no causal attribution to a process or engine.

| Run | tok/s | Observed seconds | CPU some % | IO some % | IO full % |
|---|---:|---:|---:|---:|---:|
| shared-p0-prior | 9917.69 | 0.827 | 1.55 | 8.31 | 6.88 |
| shared-p0-control | 10036.39 | 0.818 | 1.33 | 18.79 | 15.56 |
| shared-p0-projection | 10289.99 | 0.798 | 1.57 | 2.20 | 1.99 |
| shared-p0-vllm | 12265.36 | 0.671 | 1.44 | 9.40 | 7.83 |
| shared-p1-vllm | 11339.65 | 0.724 | 0.78 | 7.15 | 6.02 |
| shared-p1-projection | 10260.04 | 0.800 | 3.17 | 3.81 | 3.35 |
| shared-p1-control | 8779.59 | 0.935 | 9.79 | 3.20 | 2.25 |
| shared-p1-prior | 9936.65 | 0.827 | 0.84 | 5.06 | 4.56 |
| unique-p0-prior | 3518.23 | 2.331 | 1.45 | 11.35 | 7.28 |
| unique-p0-control | 3543.07 | 2.315 | 1.01 | 3.42 | 3.12 |
| unique-p0-projection | 3752.04 | 2.185 | 8.11 | 2.18 | 1.56 |
| unique-p0-vllm | 4425.71 | 1.854 | 1.57 | 3.67 | 3.19 |
| unique-p1-vllm | 4903.51 | 1.674 | 1.41 | 2.83 | 2.61 |
| unique-p1-projection | 3843.41 | 2.135 | 1.28 | 2.56 | 2.36 |
| unique-p1-control | 3495.39 | 2.347 | 2.78 | 11.26 | 7.71 |
| unique-p1-prior | 3520.11 | 2.329 | 1.21 | 4.55 | 4.10 |

## Decision

Candidate throughput is3.50% above frozen prior on shared and7.92% above it on unique. The frozen prior is stable between orders in this repeat, but the same-binary shared control falls from10,036.4 to8,779.6 tok/s. Its lower-I/O-pressure order is slower, so the observations do not establish I/O pressure as the cause. Unique control and prior medians agree closely; candidate improvement is promising but still a short-screen observation.

Retained phases last only0.67–2.35seconds. Short request groups are vulnerable to transient scheduling/client/server delays and are inadequate as final stable-serving proof. Do not promote or declare the vLLM objective achieved. The candidate remains below vLLM throughput and has materially higher unique TTFT. Shared TPOT and token-interval tails also remain worse than vLLM.

Next, extend fixed matched request counts rather than tuning the implementation again. Use8192 retained requests per lane, retain both reverse orders, collect the same PSI observations, and report actual durations; do not mix short and long samples into one median. Preserve request-level evidence in bounded archive parts. Begin at C32, then C8 and client64/active32. This is still a closed-loop comparison; sustained/open-loop stability and other model/hardware qualification remain separate requirements.

Recompute [comparison.json](comparison.json) from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_projection_observed_serving.py benchmarks/results/20260914-projection-observed-serving`. The verifier checks the measured controller/client identities, reference outputs, lifecycle cases and PSI deltas. Binary hashes match the first screen and the model integration receipt.
