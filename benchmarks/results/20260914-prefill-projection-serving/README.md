# Prefill projection serving — C32 screen, high run variability

Same frozen SmolLM2-135M BF16 model on RTX4090; context1024, client32/active32, batch/chunk512, equal720MiB KV payload and prefix caching on both engines. Every Riley lane uses rolling decode, prefill FFN, adaptive decode and512 unique prefix pages; only projection enables `RILEY_PREFILL_PROJECTION_PIPELINE=1`. vLLM0.27.1 is freshly measured. Shared32 variants contain498–518 prompt tokens; unique prompts552–603 tokens with distinct initial16-token pages. Each retained response generates32 tokens. Each lane uses64 warmup+256 retained, with two reversed orders.

All4096 retained responses are protocol-valid; all3072 Riley responses match prior token IDs/text/finish. Three Riley lanes each pass32 stop,32 early-cancellation and32 recovery cases. All16 measured server lanes exit0, and all3 Blender instances are restored with successful scene queries. Static viewers remain up. Full-model correctness evidence is [separate](../20260914-prefill-projection-model/README.md).

**No promotion or confirmed speedup claim.** The prior shared lane changes from9,754.9 to7,041.5 tok/s between orders (−27.8%); the projection lane also changes from10,507.2 to9,372.8. A late-run host snapshot records elevated I/O pressure. It is not per-lane telemetry and does not establish a causal explanation. The same-binary control is more informative than the disturbed frozen-prior median, but repeated matched measurements are required before concluding an improvement. No failed run was dropped or replaced in this table.

Latency units are ms; TTFT/TPOT P50 and E2E/token-interval tails are medians of two run-level estimates, not pooled percentiles. Actual SSE arrival timestamps are used without interpolation.

| Workload | Lane | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | prior | 8,398.22 | 30.69 | 2.85 | 157.79 | 178.08 | 11.30 | 21.79 |
| shared | control | 9,769.91 | 14.93 | 2.75 | 127.06 | 141.79 | 10.86 | 12.22 |
| shared | projection | 9,940.02 | 15.66 | 2.74 | 118.32 | 136.06 | 10.90 | 12.80 |
| shared | vllm | 11,679.23 | 32.74 | 1.71 | 97.22 | 98.58 | 3.30 | 6.83 |
| unique | prior | 3,177.79 | 167.18 | 5.49 | 385.92 | 418.82 | 11.79 | 16.34 |
| unique | control | 3,416.65 | 126.97 | 5.79 | 363.24 | 397.58 | 10.23 | 14.06 |
| unique | projection | 3,761.10 | 121.41 | 5.26 | 322.06 | 350.62 | 9.12 | 11.88 |
| unique | vllm | 4,542.38 | 44.36 | 5.36 | 342.98 | 375.56 | 7.61 | 11.35 |

Observed throughput ratios, not qualified gains:

| Workload | Projection vs prior | Projection vs same-binary control | Projection vs vLLM |
|---|---:|---:|---:|
| shared | +18.36% | +1.74% | -14.89% |
| unique | +18.36% | +10.08% | -17.20% |

## Run-level throughput retained

| Run | tok/s |
|---|---:|
| shared-p0-prior | 9754.90 |
| shared-p0-control | 10019.53 |
| shared-p0-projection | 10507.21 |
| shared-p0-vllm | 12263.80 |
| shared-p1-vllm | 11094.66 |
| shared-p1-projection | 9372.82 |
| shared-p1-control | 9520.30 |
| shared-p1-prior | 7041.53 |
| unique-p0-prior | 2929.08 |
| unique-p0-control | 3343.81 |
| unique-p0-projection | 3822.53 |
| unique-p0-vllm | 4639.39 |
| unique-p1-vllm | 4445.37 |
| unique-p1-projection | 3699.67 |
| unique-p1-control | 3489.48 |
| unique-p1-prior | 3426.51 |

The candidate remains below vLLM throughput. Shared TTFT is lower but TPOT and token interval tails are worse than vLLM; unique median TTFT remains much higher even though TPOT and E2E tails are lower in this screen. The overall serving objective is not achieved. Next action is a C32 repeat with per-lane host pressure observations, followed by C8/C64 if the improvement survives; avoid changing the candidate while this comparison is unresolved. Other models/hardware and sustained/open-loop qualification remain incomplete.

Recompute [comparison.json](comparison.json) with `python3 benchmarks/analysis/export_projection_pipeline_serving.py benchmarks/results/20260914-prefill-projection-serving`. The [archive](evidence.tar.gz) includes exact measured client snapshot, controller hash, binary hashes, requests, launch/exit evidence and late host-pressure snapshot. The verifier checks that all imported client helpers match the canonical client; the unused standalone CLI can differ. No profiler or build overlapped GPU timing. Raw profiler/environment dumps are not included.
