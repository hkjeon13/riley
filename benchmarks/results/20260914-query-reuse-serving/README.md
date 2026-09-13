# Query reuse — matched serving screen: not promoted

The integrated candidate is correct on the executed model/serving fixtures, but does not establish a meaningful serving improvement: shared-prefix throughput changes −0.26%, unique-prompt throughput +0.96% versus the preserved composed cache-on binary. Keep the explicit option off by default. The overall vLLM performance goal remains unmet.

## Conditions

RTX 4090, SmolLM2-135M BF16, concurrency/active capacity32, context1024, batch/chunk512, matched 720 MiB KV payload. Both cache-on engines use page16; Riley cache cap512 pages/64 entries is inside its2,048-page pool, while vLLM0.27.1 uses its own eviction policy and explicitly reports32,768 KV tokens. Every Riley lane uses prefill-FFN, paired-decode and adaptive projection; only candidate lanes enable query reuse. The `off` lane disables the prefix cache, not query reuse.

Shared workload has32 prompt suffix variants,498–518 input tokens. Unique prompts have552–603 tokens and distinct first16-token pages. All requests generate exactly32 tokens. Each lane has64 warmups+256 retained requests; two reversed orders per engine/workload yield4,096 retained requests/131,072 output tokens. No build or profiler overlaps timed runs. GUI retained; Blender stays down.

## Results

Output tokens/s; all latencies milliseconds. TTFT/TPOT are P50, E2E includes queueing. Cells are medians of two run-level estimates, not pooled percentiles. Small differences from two closed-loop runs do not establish statistical significance or sustained-load stability.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 |
|---|---|---:|---:|---:|---:|---:|
| shared | Prior composed Riley + cache | 7,924.57 | 18.99 | 3.54 | 143.14 | 168.22 |
| shared | Query reuse, cache off | 4,069.15 | 79.28 | 5.38 | 314.85 | 346.41 |
| shared | Query reuse + cache | 7,904.09 | 19.01 | 3.53 | 145.52 | 166.88 |
| shared | vLLM + cache | 11,965.69 | 31.48 | 1.68 | 91.19 | 93.28 |
| unique | Prior composed Riley + cache | 3,405.26 | 129.62 | 5.85 | 358.32 | 387.87 |
| unique | Query reuse, cache off | 3,569.56 | 123.60 | 5.54 | 340.45 | 374.09 |
| unique | Query reuse + cache | 3,437.95 | 128.69 | 5.75 | 353.67 | 385.04 |
| unique | vLLM + cache | 4,961.80 | 41.68 | 5.08 | 278.58 | 342.20 |

The native long mixed-shape gain is not reproduced as a substantial application gain. The predeclared native serving-capacity probes already showed only1.9% improvement for pure512-token prefill and essentially none for a128-token cached suffix. Pure decode is unchanged. These observations help explain limited end-to-end impact but are not a measured attribution of every serving request to a kernel branch.

## Correctness and cleanup

All3,072 retained Riley responses match the preserved baseline in prompt tokens, generated tokens, text and finish reason. Every retained response from all engines is protocol-valid with matching prompt tokens, finish reason and output length. [Full-model gate](../20260914-query-reuse-model/README.md) independently verifies2,359,296 BF16 logits bytes for the candidate and its cache composition.

After timed requests in the first shared run, prior/candidate-cache-off/candidate-cache-on each execute32 stop cases,32 client cancellations and32 recovery requests. All96 stop outputs match the baseline exactly; all96 clients disconnect after receiving at least4 but fewer than32 tokens; all96 subsequent recovery responses are reference-exact. These checks occur after retained measurement, so they do not affect reported throughput/latency. Shutdown cache telemetry for those lanes includes the additional checks. This covers recovery, not a proof that every cancellation consumed zero extra GPU work.

vLLM exact token/text agreement on shared: 428/512. vLLM exact token/text agreement on unique: 160/512. These differences are not a quality-loss percentage. Cross-engine quality equivalence, broader models, open-loop/high-concurrency and soak remain unqualified.

All16 server lanes exit0. Candidate logs confirm successful query-reuse graph recording, no graph fallback and completed paired windows. The final GPU compute inventory was empty. No Python interpreter is in the Riley request path.

## Evidence and decision

[comparison.json](comparison.json) is independently recomputed from all retained responses in [evidence.tar.gz](evidence.tar.gz). The verifier checks controller/client hashes, flags, equal KV budgets, cache hits/misses, request lengths, summaries, stop/cancel/recovery and exits. Run `python3 benchmarks/analysis/export_query_reuse_serving.py benchmarks/results/20260914-query-reuse-serving`.

Prior binary SHA256:`f1ccef367d9329a13e8c19b499ec22bf5817a5f82692fbaf38c1c3b6021688ea`; candidate:`1fc55a483be57e8822bd45d24f2dbfa606cee2569c031272582a907aa223646e`. Build/source evidence is in the model report. The first controller attempt failed before server launch due to a tuple environment key; it is not a failed inference or a measured run. The fixed complete experiment is `/data/riley-serving-260913-recovery/query-reuse-serving-c32-v2`.

Do not promote query reuse or continue a string of tiny query-tile variants. Preserve this exact candidate as an optional control. The next structural choice must address dominant pure-prefill execution and/or the retained execution pipeline, with whole-workload benefit rather than the favorable mixed native subset. Existing failed POD and alternate-precision quality gates remain failed; neither this correctness pass nor a different metric silently overrides them. Further profiling should distinguish actual pure-prefill/resource cost and post-pair host work before selecting that batch. Multi-GPU/Hopper/Blackwell runtime qualification remains outstanding.
