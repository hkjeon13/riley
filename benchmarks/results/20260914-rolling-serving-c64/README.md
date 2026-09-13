# Rolling decode serving — C64, active32

Same-day prior/new-pair/new-rolling/vLLM comparison on RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, matched720MiB KV payload. Both Riley and vLLM enable prefix caching; Riley keeps512 unique cache pages within2048 physical pages. All Riley lanes use prefill-FFN, paired native transport and adaptive decode; query reuse is off. Only rolling uses `RILEY_ROLLING_DECODE=1`. New pair is the same new binary with that flag0, isolating the selected execution policy from the rebuild.

Shared workload:32 variants,498–518 prompt tokens. Unique workload:552–603 tokens with distinct first16-token pages. Every retained response has32 generated tokens. Per lane:64 warmup+256 retained requests, repeated in reverse order. Total4096 retained requests/131072 tokens. These are closed-loop, two-run screens without statistical significance or long-soak/open-loop qualification. No profiler/build runs during timed lanes. GUI retained; Blender down.

Throughput is output tokens/s. Latencies are ms; TTFT/TPOT P50 and E2E P95/P99. Values are medians of two run-level estimates, not pooled percentiles.

Client concurrency is64; both engines admit at most32 active sequences, exercising queued admission.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Previous Riley | 8,899.16 | 128.35 | 3.16 | 243.26 | 259.17 | 11.11 | 11.79 |
| shared | New Riley, pair | 8,889.90 | 127.56 | 3.15 | 246.73 | 264.01 | 10.94 | 12.11 |
| shared | New Riley, rolling | 10,054.22 | 114.91 | 2.74 | 213.84 | 231.65 | 10.86 | 11.69 |
| shared | vLLM | 14,841.06 | 73.90 | 1.77 | 149.52 | 152.99 | 4.07 | 6.66 |
| unique | Previous Riley | 3,464.25 | 399.45 | 5.69 | 689.57 | 726.48 | 10.30 | 12.51 |
| unique | New Riley, pair | 3,475.03 | 397.65 | 5.72 | 686.65 | 721.37 | 10.11 | 12.24 |
| unique | New Riley, rolling | 3,530.51 | 400.76 | 5.70 | 672.89 | 698.05 | 9.44 | 12.16 |
| unique | vLLM | 4,854.56 | 233.58 | 4.85 | 625.79 | 680.71 | 5.83 | 8.16 |

Actual SSE arrivals determine token intervals; no interpolation. All4096 retained requests are protocol-valid and all3072 Riley responses match prior prompt/output tokens, text and finish reason. Each Riley lane passes64 stop,64 cancellation and64 recovery cases outside timing. All16 lanes exit0. Cross-engine numerical equivalence and long-soak/open-loop stability remain unqualified.

Reproduce [comparison.json](comparison.json) from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_rolling_decode_serving.py benchmarks/results/20260914-rolling-serving-c64`. The verifier checks controller/client hashes, flags, KV budgets, output identity, rolling progress, safety cases and exits.

shared: rolling throughput changes+12.98% versus prior and-32.25% versus vLLM.

unique: rolling throughput changes+1.91% versus prior and-27.27% versus vLLM.

Queue pressure does not close the vLLM gap. Keep rolling opt-in. See [integration and qualification limits](../20260914-rolling-serving/README.md).
