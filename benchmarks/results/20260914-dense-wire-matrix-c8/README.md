# Dense wire validation: C8 serving comparison

RTX 4090 / SmolLM2-135M BF16 / vLLM 0.29.0. Same frozen prior and dense-wire candidate as the C32 screen. Two reversed orders; 64 warmup + 2048 retained per lane. Values are medians of two run-level estimates, not pooled percentiles or confidence intervals.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 5152.053 | 12.654 | 1.205 | 50.319 | 51.799 |
| shared | candidate | 5061.941 | 9.873 | 1.291 | 51.720 | 52.476 |
| shared | vllm029 | 4541.791 | 14.849 | 1.302 | 62.511 | 67.419 |
| unique | prior | 2909.595 | 46.027 | 1.345 | 89.317 | 90.411 |
| unique | candidate | 2967.842 | 46.098 | 1.285 | 87.382 | 87.993 |
| unique | vllm029 | 2914.550 | 28.979 | 1.849 | 102.057 | 115.591 |

shared: candidate throughput versus prior -1.75%; versus vLLM +11.45%.
unique: candidate throughput versus prior +2.00%; versus vLLM +1.83%.


Independent verification reconstructs all 25,344 warmup/retained responses, including 16,896 exact Riley references, plus vLLM preflight and stop/cancel/recovery checks. All 12 measured lane exits and the lifecycle exit succeed; Blender restoration succeeds. Source snapshots and archive members are hash-bound. vLLM reference differences are retained, not represented as exact Riley equivalence.

The startup temperature threshold remains 48 C. Matrix v2 permits 600 seconds for cooling and records samples. C8 v1 stopped on the earlier 120-second cooling limit and is not pooled with this run. C64 v2 later failed the available-memory preflight before serving; this does not invalidate these completed conditions. Its separate rerun remains outstanding.

The broader serving goal remains unachieved. C8 shared regression must remain visible alongside C16/C32 gains; neither a universal improvement nor sustained multi-model stability is established.

```sh
python3 benchmarks/analysis/verify_dense_wire_matrix.py benchmarks/results/20260914-dense-wire-matrix-c8 --concurrency 8 --retained 2048 --require-cooldown
```
