# Dense wire validation: C64 serving comparison

RTX 4090 / SmolLM2-135M BF16 / vLLM 0.29.0. Client concurrency 64; active sequence cap 32 for both engines. This measures queueing at C64, not 64 simultaneously executing sequences. Two reversed orders, 64 warmup + 2048 retained per lane. Values are medians of two run-level estimates, not pooled percentiles or confidence intervals.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 10304.978 | 113.230 | 2.713 | 201.757 | 209.804 |
| shared | candidate | 10719.717 | 108.738 | 2.612 | 194.803 | 196.931 |
| shared | vllm029 | 13819.795 | 81.837 | 2.032 | 164.479 | 174.431 |
| unique | prior | 4056.805 | 309.563 | 6.217 | 545.107 | 576.519 |
| unique | candidate | 4375.079 | 287.013 | 5.760 | 507.619 | 542.802 |
| unique | vllm029 | 4138.962 | 281.729 | 5.729 | 760.990 | 854.228 |

shared: candidate throughput versus prior +4.02%; versus vLLM -22.43%.
unique: candidate throughput versus prior +7.85%; versus vLLM +5.70%.

Independent verification reconstructs 25,344 warmup/retained responses, including 16,896 exact Riley references, plus vLLM preflight and stop/cancel/recovery checks. All 12 measured lane exits and lifecycle exit succeed; Blender restoration succeeds. Source snapshots and archive members are hash-bound. vLLM reference differences remain recorded and are not represented as exact Riley equivalence.

The earlier C64 v2 run failed the available-memory preflight before serving. This v3 run started after old Riley tmpfs artifacts were copied to persistent storage and SHA256-verified before removing their tmpfs copies. Failed runs are not pooled. Startup temperature threshold remains 48 C with a 600-second cooling limit.

The serving goal remains unachieved: shared is slower than vLLM across the listed metrics. Unique exceeds vLLM throughput and has lower E2E tail latency, but TTFT and TPOT remain higher. C8 shared regression also remains part of the overall assessment.

```sh
python3 benchmarks/analysis/verify_dense_wire_matrix.py benchmarks/results/20260914-dense-wire-matrix-c64 --concurrency 64 --retained 2048 --require-cooldown
```
