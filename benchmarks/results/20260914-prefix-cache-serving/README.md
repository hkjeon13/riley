# Automatic prefix cache — serving screen

**Shared-prefix throughput improves 90.0% over the current cache-off baseline, with TTFT down 70.8%. It remains 39.8% below vLLM throughput. Unique-prompt throughput regresses 4.3%, so the cache stays opt-in and the overall goal is not achieved.**

RTX 4090, SmolLM2-135M BF16, concurrency/active capacity 32, context limit 1,024, batch/chunk budget 512. Both engines receive a 720 MiB KV payload budget: Riley 2,048 pages × 16 tokens; vLLM explicitly reports 32,768 KV tokens. Riley's cache charge is capped at 512 pages/64 entries inside that same pool; vLLM uses its own eviction policy. GUI remains active and Blender stays down.

Each engine/workload has two reversed-order runs, each with 64 warmups and 256 retained requests. Shared workload: 32 distinct suffix variants with a common natural-language prefix, 498–518 input tokens. Unique workload: 320 distinct prompts per run, 552–603 tokens; all first 16-token pages are distinct, including warmup versus retained traffic. Every retained response from every engine has exactly 32 generated tokens. Total: 4,096 retained requests / 131,072 output tokens, zero protocol errors.

The prior binary and current cache-off/cache-on all use the same standard V7 GPU-greedy configuration, without the optional FFN or paired-decode variants. This isolates this cache batch; it is a different workload/configuration from the earlier adaptive-decode screen.

## Results

Throughput is output tokens/s. Latencies are milliseconds. TTFT/TPOT are P50; E2E tails include queueing. Each cell is the median of the two run-level estimates, not a percentile pooled across runs.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 |
|---|---|---:|---:|---:|---:|---:|
| Shared prefix | Prior Riley | 3,899.19 | 69.794 | 5.989 | 333.259 | 353.143 |
| Shared prefix | Current Riley, cache off | 3,973.13 | 65.495 | 5.977 | 326.056 | 352.528 |
| Shared prefix | Current Riley, cache on | **7,548.05** | **19.153** | 3.734 | 152.169 | 172.411 |
| Shared prefix | vLLM, cache on | **12,539.58** | 27.717 | **1.685** | **87.583** | **89.044** |
| Unique prompts | Prior Riley | 3,430.70 | 125.692 | 5.813 | 354.394 | 387.859 |
| Unique prompts | Current Riley, cache off | 3,469.95 | 124.271 | 5.728 | 351.267 | 385.149 |
| Unique prompts | Current Riley, cache on | 3,319.48 | 128.281 | 6.001 | 366.627 | 400.060 |
| Unique prompts | vLLM, cache on | **4,740.13** | **42.853** | **5.240** | **291.348** | **355.142** |

In the shared workload, cache-on also reduces Riley P99 by 51.1% and beats vLLM TTFT by 30.9%, but its TPOT is 121.6% higher than vLLM. Unique-prompt cache-on has no hits and increases Riley P99 by 3.9%. Cache logs confirm 306 hits per shared run, including warmup, versus zero unique-prompt hits. Final cache metadata payload capacity is 62,464 and 59,736 bytes respectively; this excludes allocator bookkeeping and other scheduler metadata.

## Correctness and comparison limits

All retained Riley responses match the prior Riley baseline in prompt tokens, generated tokens, text and finish reason: 512/512 per engine/workload. The [independent automatic-cache model gate](../20260914-automatic-prefix-cache/README.md) also verifies complete BF16 logits for its five-case fixture.

vLLM matches prompt tokens, finish reason and output length on every retained request. Exact greedy token/text agreement with prior Riley is **436/512 shared** and **160/512 unique**. Thus this is a serving workload comparison with equal input/output token counts, not a claim that the engines have identical arithmetic or a completed cross-engine quality evaluation.

Two closed-loop runs at one concurrency are a screen, not sustained-load/P99 stability qualification. Client SSE frame timestamps are used without interpolating per-token timestamps. Remaining high-concurrency, open-loop, soak, broader-model and hardware qualification is unchanged.

## Evidence and next action

[comparison.json](comparison.json) is recomputed from every archived retained response by `benchmarks/analysis/export_prefix_cache_serving.py`. [evidence.tar.gz](evidence.tar.gz) contains all raw reference/warmup/retained frames, fixtures, launch configurations, engine logs, exit receipts and completion markers. It contains no profiler environment dumps. The verifier checks reference agreement, response lengths, cache hits/misses, matching KV budgets, source/controller hashes and all 16 completed lanes.

Current binary SHA256: `f1ccef367d9329a13e8c19b499ec22bf5817a5f82692fbaf38c1c3b6021688ea`; prior: `d85368b3bc3af73593f28cac2ea7bcb01f09c45f2888e6a5f4fb6be5c9101bab`. vLLM logs identify version 0.27.1. All compute processes exited after the screen.

Keep cache opt-in. First qualify its composition with the previously tested prefill-FFN/paired-decode configuration; this screen deliberately used standard V7 to isolate the cache change. Then profile GPU execution and host/prefill-decode overlap in this longer-context workload before choosing the next execution optimization batch. The client TPOT gap alone does not identify a specific attention kernel as the cause. Cache-miss admission/index overhead is a separate observed regression to address, without treating small cache-only tuning as the overall solution. Captured-model partial-tail COW remains unfinished PR10 scope.
