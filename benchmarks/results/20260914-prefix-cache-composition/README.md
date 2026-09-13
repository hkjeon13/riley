# Prefix cache + execution composition — C32 serving screen

The existing prefill-FFN, paired-decode and adaptive projection paths compose with automatic prefix caching on the tested serving workload. Shared-prefix throughput improves 5.9% over standard cache-on, but remains 34.0% below vLLM. Unique-prompt composition remains 4.2% slower than optimized cache-off. Keep these features opt-in; the overall goal is not achieved.

## Matched conditions

RTX 4090 / SmolLM2-135M BF16 / concurrency and active capacity 32 / context 1,024 / batch and chunk 512 / 720 MiB KV payload per engine. Riley has 2,048 physical pages and a cache cap of 512 pages/64 entries; vLLM 0.27.1 enables prefix caching and explicitly allocates 32,768 KV tokens. GUI retained, Blender down. No runtime kernel/source changes were needed: this batch qualifies the previously implemented composition.

32 shared-prefix prompt variants have 498–518 input tokens; unique prompts have 552–603 tokens and distinct first 16-token pages. Each lane has 64 warmups and 256 retained requests, with exactly 32 output tokens. Two reversed-order runs per workload/engine produce 4,096 retained requests and 131,072 output tokens. Reference generation uses the frozen prior Riley binary before measurement.

## Results

Throughput is output tokens/s. Latencies are milliseconds; TTFT/TPOT are P50. Each cell is the median of two run-level estimates, not pooled percentiles. E2E includes queueing.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 |
|---|---|---:|---:|---:|---:|---:|
| shared | Riley standard + cache | 7,404.38 | 19.47 | 3.82 | 154.67 | 175.23 |
| shared | Riley FFN/paired/adaptive, cache off | 3,977.24 | 72.93 | 5.67 | 327.50 | 353.34 |
| shared | Riley FFN/paired/adaptive + cache | 7,839.56 | 19.47 | 3.52 | 144.60 | 163.04 |
| shared | vLLM cache on | 11,871.69 | 28.70 | 1.74 | 98.45 | 101.30 |
| unique | Riley standard + cache | 3,299.01 | 133.17 | 6.00 | 369.94 | 404.26 |
| unique | Riley FFN/paired/adaptive, cache off | 3,522.43 | 124.14 | 5.59 | 344.04 | 377.94 |
| unique | Riley FFN/paired/adaptive + cache | 3,373.90 | 127.34 | 5.98 | 361.43 | 393.47 |
| unique | vLLM cache on | 4,764.21 | 43.15 | 5.30 | 279.45 | 338.51 |

The comparison is concurrent-day and uses the same frozen binary for all current Riley lanes. Do not compare different-day vLLM numbers as an implementation delta. Two runs are screening evidence, not statistical or sustained-load qualification.

## Correctness and actual execution

All 3,072 retained Riley responses exactly match the prior Riley prompt tokens, generated tokens, text and finish reason. All 4,096 retained responses are protocol-valid with matched prompt/finish/output length. This is serving-output regression coverage, not a new full-logits proof for the combined configuration. The earlier automatic-cache full-logits gate remains a separate standard-session result.

vLLM shared exact token/text agreement: 434/512. vLLM unique exact token/text agreement: 160/512. Cross-engine quality equivalence remains unqualified.

The verifier checks launch options, graph preparation without fallback, and nonzero completed paired-decode windows in every optimized/composed lane. This proves paired execution occurred; individual kernel dispatch counts require a separate trace. Cache telemetry verifies shared hits and zero unique hits. Every lane exits 0; the remote compute-process inventory was empty after the controller completed.

## Evidence and next batch

[comparison.json](comparison.json) is recomputed from [evidence.tar.gz](evidence.tar.gz), which contains fixtures, raw SSE rows, launch arguments, allowlisted environment, logs, exit receipts, hashes and all 16 completion records. Reproduce with `python3 benchmarks/analysis/export_prefix_cache_composition.py benchmarks/results/20260914-prefix-cache-composition`. The controller and client hashes are checked against source. No profiler dumps are included.

Current binary SHA256: `f1ccef367d9329a13e8c19b499ec22bf5817a5f82692fbaf38c1c3b6021688ea`. This is the same release binary as the standard cache screen. Runtime remains Rust → C ABI → CUDA; Python only controls offline measurements.

Use the composed lane as the candidate for a separate bounded GPU/host trace on this longer-context shared/miss workload. Identify execution time by prefill/mixed/decode and kernel family before selecting a substantial execution batch (PR04 overlap or PR06 attention work partitioning). Client TPOT alone cannot identify the kernel bottleneck. Do not keep iterating tiny projection variants. Cache lookup/admission miss overhead remains an observed separate regression. Captured-model partial-tail COW, multi-GPU/Hopper/Blackwell execution, broader-model quality, open-loop/high-concurrency and soak qualification remain incomplete.
