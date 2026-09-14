# Split FFN graph serving — C32

RTX4090, SmolLM2-135M BF16, context1024, client/active32, chunk512, prefix512, KV payload720MiB per engine. Each lane uses256 warmup +8192 retained requests. Shared reuses32 prompts; unique has distinct first16-token pages. Four engines run in each order for each workload.

Prior is the frozen unified adaptive FFN binary. Control uses original M16 FFN in the current binary. Split selects original M16 below192 total packed rows and separate M32 kernels at192 or above. All Riley lanes enable the same projection pipeline and rolling decode. The runtime remains Rust → C ABI → CUDA; Python is the external measurement client.

The first attempt stopped on vLLM readiness timeout and is preserved in [failed-v1](failed-v1/README.md). These results use a fresh complete V2 run with a common600-second readiness limit for every engine. No V1 partial measurements are merged. Client GC is disabled only during timed phases, and response evidence is spooled to tmpfs.

## Serving results

Values are medians of two run-level estimates, not pooled percentiles. Latencies use actual HTTP SSE arrivals without interpolation.

| Workload | Engine | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P95 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 10361.895 | 14.079 | 2.704 | 101.818 | 104.828 | 10.674 | 11.490 |
| shared | control | 10411.661 | 14.027 | 2.692 | 100.677 | 103.837 | 10.578 | 11.341 |
| shared | split | 10454.689 | 13.952 | 2.683 | 100.508 | 103.179 | 10.547 | 11.303 |
| shared | vllm | 12124.537 | 26.739 | 1.816 | 98.460 | 107.931 | 4.037 | 6.815 |
| unique | prior | 4025.985 | 58.991 | 6.248 | 277.055 | 288.944 | 9.292 | 10.810 |
| unique | control | 3791.741 | 62.113 | 6.640 | 294.067 | 306.983 | 9.482 | 11.001 |
| unique | split | 4056.258 | 58.361 | 6.211 | 274.280 | 285.854 | 8.942 | 10.433 |
| unique | vllm | 4876.897 | 47.838 | 5.090 | 235.787 | 276.870 | 6.455 | 10.805 |

## Descriptive changes

- shared, split versus prior: throughput +0.90%; increased latency metrics: none.
- shared, split versus control: throughput +0.41%; increased latency metrics: none.
- shared, split versus vllm: throughput -13.77%; increased latency metrics: tpot_0.5_ms, e2e_0.95_ms, itl_0.95_ms, itl_0.99_ms.
- unique, split versus prior: throughput +0.75%; increased latency metrics: none.
- unique, split versus control: throughput +6.98%; increased latency metrics: none.
- unique, split versus vllm: throughput -16.83%; increased latency metrics: ttft_0.5_ms, tpot_0.5_ms, e2e_0.95_ms, e2e_0.99_ms, itl_0.95_ms.

These ratios do not establish statistical significance, tail stability or overall qualification. Inspect both orders and host pressure below. Defaults are unchanged; future architecture runtime, other concurrency levels and open-loop sustained loads are not proven by this C32 run.

## Run-level measurements and host pressure

| Run | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P99 ms | IO some % | IO full % |
|---|---:|---:|---:|---:|---:|---:|
| shared-p0-prior | 10408.082 | 13.971 | 2.696 | 103.550 | 14.56 | 12.27 |
| shared-p0-control | 10392.710 | 14.064 | 2.696 | 104.573 | 19.22 | 16.25 |
| shared-p0-split | 10489.578 | 13.896 | 2.676 | 102.293 | 18.27 | 15.87 |
| shared-p0-vllm | 12300.260 | 25.254 | 1.814 | 105.129 | 15.05 | 12.42 |
| shared-p1-vllm | 11948.814 | 28.224 | 1.819 | 110.732 | 11.62 | 10.88 |
| shared-p1-split | 10419.800 | 14.008 | 2.689 | 104.066 | 9.45 | 8.51 |
| shared-p1-control | 10430.613 | 13.991 | 2.688 | 103.100 | 10.40 | 8.81 |
| shared-p1-prior | 10315.709 | 14.186 | 2.712 | 106.105 | 7.28 | 6.16 |
| unique-p0-prior | 4019.055 | 59.042 | 6.264 | 289.536 | 6.32 | 5.67 |
| unique-p0-control | 3790.127 | 61.894 | 6.640 | 305.946 | 6.04 | 5.37 |
| unique-p0-split | 4046.621 | 58.500 | 6.224 | 286.971 | 4.37 | 3.97 |
| unique-p0-vllm | 4881.293 | 47.988 | 5.114 | 264.654 | 16.96 | 11.81 |
| unique-p1-vllm | 4872.500 | 47.688 | 5.067 | 289.085 | 13.15 | 11.19 |
| unique-p1-split | 4065.894 | 58.221 | 6.198 | 284.737 | 27.37 | 23.70 |
| unique-p1-control | 3793.356 | 62.332 | 6.641 | 308.019 | 10.34 | 9.19 |
| unique-p1-prior | 4032.915 | 58.941 | 6.232 | 288.353 | 21.32 | 18.79 |

Host PSI is global and is not causal attribution to an engine. Readiness and client preparation are outside timed request phases.

## Startup observations

| Run | Launch to readiness ms | Server RSS KiB | Global GPU memory and temperature at readiness (nvidia-smi) |
|---|---:|---:|---|
| shared-p0-prior | 1002.343 | 794956 | 1900, 50 |
| shared-p0-control | 1002.161 | 795032 | 1900, 49 |
| shared-p0-split | 1001.854 | 804004 | 1916, 49 |
| shared-p0-vllm | 253932.459 | 1077772 | 1890, 44 |
| shared-p1-vllm | 197833.576 | 1077064 | 1890, 47 |
| shared-p1-split | 8513.091 | 803968 | 1916, 49 |
| shared-p1-control | 1001.807 | 795320 | 1900, 50 |
| shared-p1-prior | 1002.089 | 794948 | 1900, 49 |
| unique-p0-prior | 1001.966 | 795048 | 1900, 50 |
| unique-p0-control | 1002.188 | 794968 | 1900, 49 |
| unique-p0-split | 1002.270 | 804024 | 1916, 49 |
| unique-p0-vllm | 42078.674 | 1078384 | 1890, 46 |
| unique-p1-vllm | 29538.459 | 1078240 | 1890, 47 |
| unique-p1-split | 52564.633 | 803568 | 1916, 45 |
| unique-p1-control | 1002.223 | 794788 | 1900, 47 |
| unique-p1-prior | 1002.491 | 794840 | 1900, 49 |

Global GPU readings include display allocations. These are not isolated CUDA graph allocation measurements, and launch time includes dependency/model loading and compilation.

## Verification

All17 archives/184 regular files/4,735,499,038 uncompressed bytes passed archive and materialization hash checks, snapshot identity, lifecycle restoration and credential-pattern scans.

The semantic exporter checks131072 retained requests/4194304 output tokens and4096 warmup requests;98304 retained Riley responses must match the frozen reference. It verifies all16 exits, backend flags, KV capacity, rolling decode/drains, SSE-derived statistics,32 timed-phase GC receipts and96 each stop/cancel/recovery cases. Stored rows do not preserve the raw `[DONE]` marker; protocol completeness also relies on the hash-bound client. vLLM reference agreement is reported separately in comparison.json.

Reproduce with `verify_ffn_split_archives.py`, then `export_ffn_split_serving.py`, then this renderer against the result directory. The prior [model gate](../20260914-ffn-split-model/README.md) remains a separate full-logit and memcheck proof.

## Decision and next optimization area

Split adds less than1% throughput over the frozen unified adaptive backend in both workloads in this run. Against the current M16 control, unique improves6.98%, but shared improves only0.41% and changes direction across orders (+0.93%/−0.10%). All reported run-median latency metrics improve versus the unified prior, but candidate throughput remains13.77%/16.83% below vLLM. No broad performance goal is met.

The four-lane median launch-to-readiness is4757.68ms for split versus1002.17ms for control. Global GPU memory readings are1916 versus1900 and median server RSS is803986 versus795000KiB. These show a startup/resource cost in this experiment; global memory does not isolate graph allocations. Host IO PSI some ranges4.37–27.37% during retained phases, so small gains are not treated as stable causal proof.

Do not promote split by default and do not continue FFN threshold/tile micro-variants. Preserve the frozen binary and evidence as an experimental comparison point. The next meaningful area is the remaining PR06 attention work distribution: assess actual decode/prefill shape occupancy, bounded scratch/graph ownership, and arithmetic-order compatibility before implementing a linked batch. The existing128-token online-softmax recurrence and BF16 probability rounding mean mathematical split/merge equivalence does not prove the current numerical contract. An executed numerical failure stays a failure; tolerances are not relaxed after observing results.

Blender scene RPC on9876/9911/9887 was checked again after measurement, and local viewer endpoints31840/31970/32010 returned200. This is service verification, not a new public visual check.
