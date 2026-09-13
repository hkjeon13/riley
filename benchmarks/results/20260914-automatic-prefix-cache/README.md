# PR10 automatic full-page prefix cache

Automatic scheduler reuse and the serving opt-in are implemented. The real-model correctness gate and a [matched C32 serving screen](../20260914-prefix-cache-serving/README.md) are complete. Shared-prefix throughput improves 90.0% versus current cache-off, but vLLM remains faster and unique-prompt throughput regresses 4.3%. Cache stays opt-in.

The bounded LRU cache publishes only completed full-page prompt prefixes. At least one prompt token is left for fresh logits. Admission finds the longest matching complete-page range, including a shorter range of a longer cached prompt. Range import verifies the original token digest before deriving the prefix and shares actual physical pages without copying. Cache entries retain immutable exports after the source request closes.

Cache-only owner tags come from the same monotonic namespace as request tags, preventing collisions in ordinary/paired authority. Cache page charges plus request promises stay within the physical pool; admission evicts old entries as needed, while live consumers keep their own page ownership. Shutdown drains cache leases, and an explicitly quiesced mutation-unknown abort invalidates the cache. Entry capacity, token arrays and leased block-handle payload capacity are reported separately from GPU pages. Lookup is a bounded scan of cached token prefixes, not a radix tree; its cache-miss overhead still needs serving measurement.

`RILEY_PREFIX_CACHE_PAGES=<positive page count>` enables the serving path, with at most 64 entries and the requested page budget. Zero/unset retains cache-off behavior. It requires V7 standard/adaptive decode, optionally prefill FFN. Unsupported backend combinations and invalid budgets are rejected. Identity is derived from the actual retained loaded-model catalog; it conservatively includes that catalog's configuration. No Python runtime bridge is added.

## Executed correctness gate

| Check | Result |
|---|---|
| Actual SmolLM2 cache-off vs automatic cache | All 1,474,560 BF16 logits bytes identical |
| Workload | 5 requests, 3 teacher-forced outputs each; repeated prompt, unique suffix, shorter prefix, miss |
| Prefill tokens, cache off / on | 157 / 77; 3 hits and 80 reused tokens |
| Final cached state before close | 3 entries / conservative charge 5 pages |
| Cleanup | All host pages and CUDA allocations reclaimed |
| Scheduler library with CUDA | 55 passed, 0 failed |
| CPU scheduler library/integration/doc suites | 155 passed, 0 failed |
| CPU runtime library | 349 passed, 0 failed; 1 existing timing diagnostic ignored |
| CUDA/server check and optimized serving build | Passed |

Prefill token reduction is a computation count for this fixture, not throughput or TTFT improvement. Cache correctness includes source/cache/consumer close ordering, suffix retry, admission eviction with a live reader, and mutation-unknown invalidation. The initial host test used a nonexistent close-output accessor; it was corrected before final tests. GPU tests were explicitly executed on RTX 4090, not skipped.

[Evidence](evidence/) and [receipt](receipt.json) preserve the final v2 gate. Release binary: `f1ccef367d9329a13e8c19b499ec22bf5817a5f82692fbaf38c1c3b6021688ea`. The previous binary was preserved as `riley-before-prefix-cache-v1`, SHA256 `d85368b3bc3af73593f28cac2ea7bcb01f09c45f2888e6a5f4fb6be5c9101bab`.

Reproduce correctness with `bash benchmarks/analysis/prefix_cache_model_gate.sh /data/riley-serving-260913-recovery <new-absolute-evidence-directory>`. The serving controller `prefix_cache_serving_screen.py` compares the preserved binary, current cache-off, current cache-on, and vLLM cache-on in reversed orders. It separates 32 shared-prefix/unique-suffix variants from prompts whose first 16 tokens are all unique, with equal 720 MiB KV payload limits. That screen is not an overall release qualification.

Remaining PR10 scope: captured-model partial-tail COW/drain, larger and natural quality corpora, concurrency/soak qualification, and measured serving comparison. Full-page automatic reuse is available independently of partial-tail COW. Multi-GPU/peer and Hopper/Blackwell runtime tests remain hardware-specific work.
