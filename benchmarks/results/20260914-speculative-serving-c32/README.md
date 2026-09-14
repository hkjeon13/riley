# Speculative serving integration: correctness passes, performance regresses

Connected opt-in `RILEY_SPECULATIVE_DECODE=1` to wide verification with immutable prefix caching, request-processor fallback, multi-token detokenization, stop-token/string truncation, KV prefix settlement, and commit-before-publication. Requires GPU greedy, projection pipeline, and unbuffered execution; paired/rolling decode and split/adaptive FFN are explicitly excluded. Default remains disabled. Runtime execution remains Rust → C ABI → CUDA.

RTX 4090, SmolLM2-135M BF16, C32/active32, context1024, chunk512, 720MiB KV payload per engine, prefix cache enabled (Riley512 pages). Four lanes, two orders, shared and unique prompts. Each lane has 64 warmup and 512 retained requests. vLLM is **0.27.1**, not a latest-version claim.

| Workload | Lane | tok/s | TTFT P50 ms | TPOT P50 ms | E2E P95 ms | E2E P99 ms | ITL P99 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| shared | prior | 10309.224 | 14.071 | 2.694 | 104.814 | 129.599 | 11.542 |
| shared | control | 8841.911 | 14.419 | 3.241 | 117.924 | 150.635 | 11.674 |
| shared | speculative | 6316.548 | 14.808 | 4.605 | 182.108 | 197.231 | 12.802 |
| shared | vllm | 11550.778 | 32.492 | 1.694 | 100.716 | 103.751 | 6.735 |
| unique | prior | 4130.279 | 63.125 | 5.930 | 290.478 | 314.881 | 10.886 |
| unique | control | 3805.963 | 62.574 | 6.413 | 319.470 | 345.755 | 11.910 |
| unique | speculative | 3758.437 | 68.729 | 6.453 | 330.284 | 394.901 | 12.505 |
| unique | vllm | 5081.541 | 40.340 | 4.999 | 223.669 | 314.942 | 8.423 |

Prior is the frozen best split-FFN + rolling decode binary; control is the same new binary with unbuffered projection execution and speculation disabled. Speculation is **28.56% slower than control on shared** and **1.25% slower on unique**; versus prior, **38.73% / 9.00% slower**. Against vLLM it is **45.31% / 26.04% slower**. The loss of rolling/split optimizations is separated by the control lane. No default promotion or throughput improvement is claimed.

All 9,216 warmup/retained responses pass protocol reconstruction; all 6,912 Riley responses match frozen prior token/text references. Stop/cancel/recovery pass 32 cases each per Riley lane (96 each total). vLLM output agreement is reported separately and is not required to equal Riley numerical policy. Stop strings are validated from real serving requests, not only scheduler mocks. CPU: scheduler 72, runtime policy 7, server 69 pass (one pre-existing ignored). The server CUDA release build passes. The first controller failed before server startup due to a tuple environment key; failed evidence is retained, corrected controller completes all 16 lanes.

Metrics are medians of two run-level estimates, not pooled percentiles. This 512-request-per-lane screen does not establish stable P99 or broad serving qualification. Client Python is offline only; its GC is disabled inside timed phases for every lane. Requests use recorded SSE arrivals without interpolating token timestamps. CPU/IO pressure and launch receipts are retained. Blender is restored after measurement.

Four bounded Nsight traces completed with exact outputs on 512 total warmup/retained requests. Raw SQLite/Nsight files remain remote-private; `profile-numeric.tar.gz` contains curated counts, reconstructed requests and source digests. The middle 80% of graph launches includes warmup and is diagnostic only.

Verification averages 4.668ms per selected graph in shared and 4.730ms in unique. `riley_mixed_attention::mapped_attention` accounts for 56.4% / 59.6% of verification kernel time. The M256 head GEMM accounts for approximately 2.5%; it is not the dominant measured cost. Ordinary control decode averages 1.672ms / 1.764ms per selected graph, but these graphs do different amounts of work and this ratio is not a per-token speedup. Both tracing lifecycles restored Blender.

Next optimization batch: short multi-query verification attention, KV reuse across GQA heads/nearby query positions, and efficient treatment of one-input owners; preserve reduction order and strict generation gates. Validate profitable acceptance/work ratios with the same serving comparison. Do not start by tuning head buckets based on an assumed head bottleneck.

```sh
python3 benchmarks/analysis/verify_speculative_serving.py
```
