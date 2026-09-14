# Dense wire validation: C16 serving comparison

RTX 4090 / SmolLM2-135M BF16 / vLLM 0.29.0. Same frozen prior and dense-wire candidate as the C32 screen. Two reversed orders; 64 warmup + 2048 retained per lane. Values are medians of two run-level estimates, not pooled percentiles or confidence intervals.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 7686.162 | 13.738 | 1.682 | 71.079 | 73.541 |
| shared | candidate | 7969.000 | 13.064 | 1.638 | 67.111 | 72.979 |
| shared | vllm029 | 7571.755 | 19.412 | 1.510 | 76.792 | 83.812 |
| unique | prior | 3489.598 | 45.481 | 3.247 | 164.502 | 177.391 |
| unique | candidate | 3795.521 | 40.109 | 3.043 | 146.218 | 156.701 |
| unique | vllm029 | 3995.435 | 30.711 | 3.080 | 148.917 | 163.281 |

shared: candidate throughput versus prior +3.68%; versus vLLM +5.25%.
unique: candidate throughput versus prior +8.77%; versus vLLM -5.00%.


Independent verification reconstructs all 25,344 warmup/retained responses, including 16,896 exact Riley references, plus vLLM preflight and stop/cancel/recovery checks. All 12 measured lane exits and the lifecycle exit succeed; Blender restoration succeeds. Source snapshots and archive members are hash-bound. vLLM reference differences are retained, not represented as exact Riley equivalence.

The startup temperature threshold remains 48 C. Matrix v2 permits 600 seconds for cooling and records samples. C8 v1 stopped on the earlier 120-second cooling limit and is not pooled with this run. C64 v2 later failed the available-memory preflight before serving; this does not invalidate these completed conditions. Its separate rerun remains outstanding.

The broader serving goal remains unachieved. C8 shared regression must remain visible alongside C16/C32 gains; neither a universal improvement nor sustained multi-model stability is established.

```sh
python3 benchmarks/analysis/verify_dense_wire_matrix.py benchmarks/results/20260914-dense-wire-matrix-c16 --concurrency 16 --retained 2048 --require-cooldown
```
