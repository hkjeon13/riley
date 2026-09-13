# Unique physical prefix-cache residency

The cache previously charged every exported lease page, including repeated leases on the same physical page. A bounded reproduction retained only two three-page entries under a six-page budget, despite only four physical pages being allocated. Three entries need only five physical pages. The new implementation retains all three under the unchanged limit.

The optimization batch separates unique physical residency from per-export lease ownership, recomputes incremental residency after each LRU eviction, and releases accounting only after the pool successfully releases the export. Per-physical-index state retains the full pool/generation block identity until the last cache lease disappears. All cache-owner/page pairs remain in the dispatch authority ledger; its reservation uses lease count rather than unique page count. Host-capacity reporting includes the residency index and per-entry page indices.

Admission still conservatively reserves promised request pages plus unique cached pages. Live-request/cache overlap is not subtracted. The pool capacity, cache page limit and entry limit are unchanged. Lookup remains a bounded longest-prefix scan; this is not a radix-tree implementation or partial-tail COW integration. Rust/C ABI/CUDA serving has no Python runtime dependency.

## Validation

Local scheduler tests: 157 passed, including 57 library tests and 2 doctests. Regression tests cover shared-page capacity, full-prefix reuse, all lease-owner entries, partial eviction, shutdown reclamation and physical-index reuse with a new generation. Existing live-consumer eviction, unknown-mutation invalidation and retry/authority tests also pass. Exact counts and GPU gates are retained under `model/`.

On RTX4090, both real-model tests pass: automatic cache versus uncached logits compare 1,474,560 bytes exactly; the composed query-reuse test compares 2,359,296 bytes exactly. CUDA scheduler, staged-session tests and release build pass. Query reuse remains disabled in the serving comparison. No CUDA kernel changed; this batch adds no Hopper/Blackwell/multi-GPU runtime qualification.

## Serving evidence

The matched C32 screen completed two reverse-order runs. Shared-prefix throughput improves 5.77%, TTFT improves 12.56%, and TPOT improves 5.94% versus the immediately preceding Riley binary. Unique throughput changes −0.07%, while TTFT increases 5.94%. These two-run measurements do not establish statistical significance or overall workload qualification. Prior binary is retained remotely as `riley-before-cache-residency-v1`. Both Riley cache-on lanes use 512 cache pages and 2,048 physical pages; vLLM uses the same 720 MiB KV payload with prefix caching enabled. The comparison includes cache-off and unique-prompt controls, output parity, stop/cancellation recovery, and process cleanup.

Rollback: restore the previous committed scheduler implementation or disable the automatic cache with `RILEY_PREFIX_CACHE_PAGES=0` for new server instances after draining the current instance. Cache enablement remains opt-in.

RTX4090 / SmolLM2-135M BF16 / C32 and active32 / context1024 / batch and chunk512. Shared workload has32 variants with498–518 prompt tokens; unique prompts have552–603 tokens and distinct first16-token pages. Each response emits32 tokens. Each lane has64 warmups and256 retained requests, with reverse-order repetition. GUI retained, Blender down, no profiler/build during timing.

Output tokens/s; latency milliseconds. TTFT/TPOT are P50; E2E includes queueing. Values are medians of two run-level estimates, not pooled percentiles.

| Workload | Engine | tok/s | TTFT | TPOT | E2E P95 | E2E P99 |
|---|---|---:|---:|---:|---:|---:|
| shared | Previous Riley + cache | 8,194.24 | 18.63 | 3.38 | 140.86 | 156.20 |
| shared | New Riley, cache off | 4,034.31 | 73.52 | 5.59 | 323.94 | 347.09 |
| shared | New Riley + cache | 8,666.72 | 16.29 | 3.18 | 135.86 | 151.59 |
| shared | vLLM + cache | 11,506.78 | 32.67 | 1.74 | 97.02 | 98.79 |
| unique | Previous Riley + cache | 3,472.51 | 125.95 | 5.64 | 350.30 | 381.23 |
| unique | New Riley, cache off | 3,559.95 | 130.75 | 5.35 | 341.77 | 375.25 |
| unique | New Riley + cache | 3,470.17 | 133.43 | 5.69 | 350.56 | 379.64 |
| unique | vLLM + cache | 4,885.77 | 40.69 | 5.05 | 305.10 | 364.47 |

All4,096 retained requests are protocol-valid; all3,072 Riley responses exactly match prior prompt/output tokens, text and finish reason. All96 stop cases and96 recovery requests match the reference;96 cancellation clients disconnect before the32-token budget. All16 engine lanes exit0 and the final GPU compute inventory is empty. vLLM token/text agreement is443/512 shared and160/512 unique; cross-engine numerical quality equivalence remains unqualified.

Cache telemetry confirms32 entries occupying489 unique pages instead of16 entries charged512 lease pages. Reused tokens increase168,384→200,800 in the first shared run and125,472→152,832 in the second. These counters include warmup and, in the first run, post-timing safety requests; they are not pure retained-phase counters. Host capacity rises62,464→194,152 bytes for shared and59,736→145,116 for unique due to the residency index, page indices and additional retained entries. With no shared pages, both versions retain13 entries/481 pages and have zero hits.

Keep the corrected bounded accounting in the opt-in cache; do not promote caching universally or claim vLLM parity. New Riley throughput is still24.68% below vLLM for shared and28.97% below for unique. Shared TTFT is lower, but TPOT and tail latency remain worse. The cache-capacity defect is addressed; repeatedly increasing this budget or tuning its counters is not the next primary optimization. Further work should return to the planned decode/GPU execution and prefill scheduling mechanisms, with a separately defined serving gate. Partial-tail COW, broader models, open-loop/soak, multi-GPU and Hopper/Blackwell runtime qualification remain outstanding.

[comparison.json](comparison.json) is regenerated from [evidence.tar.gz](evidence.tar.gz) by `python3 benchmarks/analysis/export_cache_residency_serving.py benchmarks/results/20260914-cache-residency`. The verifier checks per-request results, matching KV budgets and flags, controller/client hashes, stop/cancel/recovery and exits. [receipt.json](receipt.json) identifies the source and binary. Two final unit regressions were expanded after the GPU model gate; rebuilding with the final source produced the **same release SHA256** as the serving measurement. CUDA scheduler tests are rerun on that final source. GPU model results apply to unchanged production code.
