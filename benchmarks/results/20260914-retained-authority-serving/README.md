# Retained authority — matched serving comparison

Measured throughput changes +0.51% on shared prefixes and +2.12% on unique prompts versus the preserved composed cache-on baseline. These two-run changes do not establish a large or statistically robust gain. Throughput/TPOT remain behind vLLM; the overall goal is unachieved.

RTX4090 / SmolLM2-135M BF16 / C32 and active32 / context1024 / batch and chunk512 / matched720 MiB KV payload. Cache-on Riley has2,048 physical pages and a512-page conservative cache budget; vLLM0.27.1 explicitly enables prefix caching with32,768 KV tokens. Every Riley lane uses prefill-FFN, paired-decode and adaptive projection; query reuse is off. Prior/new differ only in retained expectation handling. `off` disables the cache, not the new host pipeline.

Shared:32 variants,498–518 prompt tokens. Unique:552–603 tokens with distinct first16-token pages. Every response has32 generated tokens. Each lane has64 warmups+256 retained requests, repeated in reverse order:4,096 retained requests/131,072 output tokens. No profiler/build during the timed run. GUI retained; Blender down.

## Results

Output tokens/s; latency milliseconds; TTFT/TPOT P50, E2E includes queueing. Cells are medians of two run-level estimates, not pooled percentiles. No open-loop/soak or high-concurrency stability claim follows from C32 alone.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 |
|---|---|---:|---:|---:|---:|---:|
| shared | Prior composed + cache | 8,019.87 | 18.99 | 3.49 | 144.85 | 166.32 |
| shared | New, cache off | 4,043.12 | 73.13 | 5.58 | 320.74 | 346.74 |
| shared | New + cache | 8,061.00 | 18.85 | 3.46 | 142.00 | 164.79 |
| shared | vLLM + cache | 11,488.38 | 31.22 | 1.73 | 99.36 | 101.19 |
| unique | Prior composed + cache | 3,404.59 | 129.51 | 5.84 | 358.94 | 390.05 |
| unique | New, cache off | 3,570.10 | 128.87 | 5.43 | 338.87 | 366.56 |
| unique | New + cache | 3,476.86 | 126.50 | 5.66 | 350.55 | 381.75 |
| unique | vLLM + cache | 4,972.12 | 41.07 | 5.04 | 274.76 | 349.69 |

All3,072 retained Riley responses exactly match the prior release's prompt/output tokens, text and finish reason. All4,096 retained responses are protocol-valid with matching prompt counts, output lengths and finish reasons. After retained timing, prior/new-cache-off/new-cache-on each execute32 stop cases,32 cancellations and32 recovery requests. All96 stop outputs match the reference,96 clients disconnect at4–31 received tokens, and96 recovery responses are reference-exact. Logs confirm paired execution and all16 lanes exit0. The final GPU compute inventory is empty.

vLLM shared exact token/text agreement: 446/512. vLLM unique exact token/text agreement: 160/512. This is not a quality-loss estimate; cross-engine quality equivalence remains unqualified.

## Decision and next investigation

[Host diagnostics](../20260914-retained-authority/README.md) show less expectation work, but GPU wait increases and serving improves only slightly. Preserve the safe reduction in repeated work; do not keep refining small serialization/allocation costs as the primary route to the overall goal.

A separate structural limitation is visible: the32-variant shared workload ends with only16 cache entries at the512-page charge. `PrefixCache::publish` charges each export's full page list, even where exports refer to already-shared physical pages. This is conservative lease accounting, not unique physical memory. The recorded hit/reused-token counters do not by themselves prove how many distinct pages remain. Next measure unique physical residency and eviction/full-prefix reuse, then consider shared-page-aware cache accounting/indexing while preserving promise/eviction/reader/COW bounds. Do not just raise a cache limit and claim a structural improvement, or compare against a different physical KV budget.

## Evidence

[comparison.json](comparison.json) is rederived from [evidence.tar.gz](evidence.tar.gz) by `python3 benchmarks/analysis/export_retained_authority_serving.py benchmarks/results/20260914-retained-authority-serving`. The verifier checks raw rows, flags, matched KV budgets, reference counts, stop/cancel/recovery, cache telemetry and exits. Source/build/model evidence is in the implementation report. Raw completed experiment: `/data/riley-serving-260913-recovery/retained-authority-serving-c32-v1`. Broader models, unsupported hardware, full quality and sustained serving qualification remain outstanding.
