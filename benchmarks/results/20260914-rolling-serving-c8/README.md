# Rolling decode serving — C8, active8

Same-day prior/new-pair/new-rolling/vLLM comparison on RTX4090, SmolLM2-135M BF16, context1024, batch/chunk512, matched720MiB KV payload. Both Riley and vLLM enable prefix caching; Riley keeps512 unique cache pages within2048 physical pages. All Riley lanes use prefill-FFN, paired native transport and adaptive decode; query reuse is off. Only rolling uses `RILEY_ROLLING_DECODE=1`. New pair is the same new binary with that flag0, isolating the selected execution policy from the rebuild.

Shared workload:32 variants,498–518 prompt tokens. Unique workload:552–603 tokens with distinct first16-token pages. Every retained response has32 generated tokens. Per lane:64 warmup+256 retained requests, repeated in reverse order. Total4096 retained requests/131072 tokens. These are closed-loop, two-run screens without statistical significance or long-soak/open-loop qualification. No profiler/build runs during timed lanes. GUI retained; Blender down.

Throughput is output tokens/s. Latencies are ms; TTFT/TPOT P50 and E2E P95/P99. Values are medians of two run-level estimates, not pooled percentiles.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 | Token interval P95 | Token interval P99 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared | Previous Riley | 4,526.39 | 11.28 | 1.46 | 58.62 | 62.33 | 2.84 | 5.49 |
| shared | New Riley, pair | 4,570.91 | 13.18 | 1.41 | 57.95 | 58.50 | 2.87 | 3.62 |
| shared | New Riley, rolling | 5,092.23 | 12.92 | 1.22 | 51.09 | 51.47 | 1.72 | 3.59 |
| shared | vLLM | 4,475.51 | 14.76 | 1.32 | 66.60 | 69.87 | 1.86 | 3.84 |
| unique | Previous Riley | 2,551.87 | 56.44 | 1.42 | 102.05 | 103.30 | 2.87 | 4.75 |
| unique | New Riley, pair | 2,556.66 | 56.06 | 1.42 | 101.73 | 104.72 | 2.87 | 4.79 |
| unique | New Riley, rolling | 2,680.98 | 56.07 | 1.26 | 96.80 | 97.32 | 1.56 | 4.78 |
| unique | vLLM | 2,996.00 | 25.92 | 1.88 | 96.56 | 107.38 | 4.88 | 5.54 |

Token intervals use actual SSE token arrival timestamps, without interpolation. Multiple tokens observed in one frame can have zero intervals; no fabricated uniform arrival spacing is used.

shared: rolling throughput changes+12.50% versus prior and+13.78% versus vLLM. TPOT changes-16.78% versus prior.

unique: rolling throughput changes+5.06% versus prior and-10.51% versus vLLM. TPOT changes-11.52% versus prior.

All3072 retained Riley responses match the frozen prior binary in prompt/output tokens, text and finish reason. All4096 retained responses are protocol-valid. Each Riley lane executes8 stop cases,8 cancellations before32 tokens, and8 recovery requests outside timing. All stop and recovery outputs match the reference. All16 lanes exit0. vLLM token/text agreement is435/512 shared and155/512 unique; this is not a quality-loss estimate and cross-engine numerical equivalence remains unqualified.

[comparison.json](comparison.json) is recomputed from [evidence.tar.gz](evidence.tar.gz) with `python3 benchmarks/analysis/export_rolling_decode_serving.py benchmarks/results/20260914-rolling-serving-c8`. The verifier checks raw rows, flags, equal KV budgets, controller/client hashes, positive rolling progress/drain counters, stop/cancel/recovery and exits. Binary hashes and launch commands are retained in preparation/launch evidence.

C8 shared beats vLLM in this screen, while unique throughput remains10.51% behind. Shared TTFT is14.56% above the prior binary (the same new binary in pair mode also has higher TTFT), so the measured baseline difference is retained rather than hidden. Two runs do not isolate its cause or establish statistical significance. Rolling remains opt-in; see the C32 report for implementation and broader qualification limits.
