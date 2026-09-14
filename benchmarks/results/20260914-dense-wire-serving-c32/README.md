# Dense wire validation: C32 serving improvement

RTX 4090 / SmolLM2-135M BF16 / C32, prefix caching, identical 754,974,720-byte KV payload, context 1024 and batch/chunk budget 512. Prior is frozen split-FFN + rolling decode; candidate changes CPU wire validation and tentative encoding, preserving GPU arithmetic and serving policies. Comparator is vLLM 0.29.0. Two reversed orders, 64 warmup + 512 retained per lane; medians of two run-level estimates, not pooled percentiles.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms |
|---|---|---:|---:|---:|---:|---:|
| shared | prior | 10127.968 | 14.497 | 2.736 | 108.517 | 132.869 |
| shared | candidate | 10682.668 | 13.632 | 2.612 | 97.108 | 129.878 |
| shared | vllm029 | 11940.971 | 28.712 | 1.762 | 99.860 | 102.377 |
| unique | prior | 4122.596 | 70.118 | 5.721 | 289.401 | 316.600 |
| unique | candidate | 4382.591 | 64.154 | 5.299 | 274.699 | 300.327 |
| unique | vllm029 | 4856.279 | 42.285 | 5.224 | 231.174 | 332.215 |

shared: throughput versus prior +5.48%; versus vLLM -10.54%. TTFT/TPOT and E2E P95/P99 improve versus prior in the run-median summary.
unique: throughput versus prior +6.31%; versus vLLM -9.75%. TTFT/TPOT and E2E P95/P99 improve versus prior in the run-median summary.

Retain the implementation as a measured C32 improvement. The full goal is **not achieved**: throughput remains below vLLM, and broader concurrency/multi-model/hardware stability is unqualified. No confidence interval is inferred from two runs. Stage-level diagnostic timing is separate from these serving metrics.

Changes: bounded direct physical-owner/shared/alias tables, dense slot bits, and structural tentative replay validation without copying the immutable successor expectation. Every authority, duplicate owner, writable alias/COW, position, slot, cookie and replay check remains. A frozen tree-based validator under cfg(test) agrees on 9,600 deterministic normal/mutated cases, including identical errors. Six future-wire, 72 scheduler, 21 existing wire and nine CUDA-feature runtime ticket tests pass; CUDA server builds.

Independent serving proof checks 6,912 warmup/retained responses (4,608 exact Riley references), latest-vLLM preflight, and 64 each stop/cancel/recovery checks. All measured processes exit successfully and Blender restoration passes. Sources and candidate binary are bound to archived hashes. Candidate SHA256: `a525729d037b519e9c796b7574f960820fb6cbeb1e0d60e4a8a504c4cd616403`. vLLM package versions are retained.

```sh
python3 benchmarks/analysis/verify_dense_wire_serving.py
```

The follow-up bounded profile completes four traces / 512 exact responses, no owned processes remaining, and successful wrapper/restoration. Future preparation is 66.144ms / 103 calls (control shared) versus 28.105ms / 83 calls (candidate shared); unique is 48.127ms / 86 versus 9.149ms / 85. Call counts and shape mixes differ, so these are diagnostics, not a same-operation or serving speedup ratio. Candidate shared divides into 10.792ms authority construction and 17.309ms check/encode; unique is 2.594ms and 6.550ms. The separately timed serving screen above establishes the end-to-end improvement. Further unique-workload CPU micro-tuning has less headroom; broader concurrency validation and device work remain necessary. Raw Nsight/SQLite stays remote-private. Do not report the current screen as universal performance or use it to skip malformed-packet and ownership regression gates.
