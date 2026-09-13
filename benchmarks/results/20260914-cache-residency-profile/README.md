# Post-residency serving trace: rolling decode pipeline selected

Current release SHA256 `4eeea8cc3b899c41352c0a28b098c42e45a987f472470a2e11503279ad669ece`, source baseline `ba9139bc`. RTX4090, SmolLM2-135M BF16, C32/active32, context1024, batch/chunk512, KV2048/cache512. Prefill FFN, paired decode and adaptive projection enabled; query reuse off. This is a diagnostic, not a new serving performance comparison.

Both traces execute64 warmup+64 checked requests, for256 total reference-exact responses. Both processes exit0 with no remaining owned processes; peak profiler-tree RSS is below1.6GB and the final compute inventory is empty. GUI retained, Blender down. Original reports and SQLite remain private remotely; the archive contains only fresh curated numeric CUDA databases, allowlisted public kernel/API names, and scoped request/launch receipts.

## Measured costs

The analyzer selects the middle80% of graph count, including portions of warmup and phase transitions. CPU launch gaps can include client pacing, profiler overhead and scheduling outside the engine. They are not all removable engine time. Kernel sums and graph spans use different denominators.

| Metric | Shared prefixes | Unique prompts |
|---|---:|---:|
| Prefill/mixed fraction of graph span | 40.80% | 83.55% |
| Ordinary decode median graph span | 1,642.41µs | 1,787.56µs |
| Future decode median graph span | 1,533.89µs | 1,687.56µs |
| After-pair median GPU gap | 1,116.93µs | 873.16µs |
| After-pair post-GPU until CPU launch | 1,106.87µs | 865.64µs |
| Score/value attention fraction of decode kernel sum | 35.71% | 41.69% |

The updated analyzer exports per-stage kernel costs, avoiding the prior ambiguity between total mixed-attention cost and pure-decode work. Pure-decode `independent_values` remains the largest single kernel group; prior softmax-sharing and monolithic attention attempts already regressed and are not selected again from these percentages alone. Unique prompts remain predominantly prefill/mixed and need a separate attention/prefill structural improvement; decode overlap alone is not a sufficient route to the full goal.

## Selected implementation direction

`VariableSession` retains one predecessor and one successor and fences both before pair confirmation. `Scheduler::plan_decode_window` reserves exactly two iterations; `try_decode_window` samples, settles and confirms the pair before returning. A new pair therefore starts after a full host boundary even when greedy decode can continue. This is a code-supported pipeline limitation, not proof that the entire measured gap is its cost.

Proceed with a bounded rolling, one-step-ahead decode pipeline, as specified in [PR20](../../../deploy/260913/20-rolling-decode-pipeline.md): retained append reservations with ordered prefix settlement; ticket/result-ring ownership; enqueue-next before processing the previous result; and token-by-token publication, stop/cancel draining and admission boundaries. Do not replace it with longer buffered output bursts or merely raise the fixed window length.

The principle is supported by [TensorRT-LLM overlap scheduling](https://nvidia.github.io/TensorRT-LLM/1.2.0/torch/features/overlap_scheduler.html) and [SGLang future-token scheduling](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/): CPU result processing overlaps the following GPU step. Riley's reservation and result ownership design is our implementation proposal, not a direct port or a claim of their reported speedups. Runtime remains Rust/C ABI/CUDA.

No runtime implementation or default changes in this diagnostic commit. Overall goal remains unachieved. Model parity and same-day prior/new/vLLM serving, including token inter-arrival tails, are required before promotion.

## Reproduce

Use `prefix_cache_composition_trace.py` with the latest serving fixtures and `--requests 64 --concurrency 32 --active-capacity 32 --prefill-ffn-pipeline --adaptive-decode`. Sanitize using `sanitize_serving_trace.py` before export. Recompute with `python3 benchmarks/analysis/export_cache_residency_profile.py benchmarks/results/20260914-cache-residency-profile`.

[receipt.json](receipt.json) contains hashes, per-stage kernel accounting and process receipts. [evidence.tar.gz](evidence.tar.gz) is the re-verifiable curated archive. Raw remote location: `/data/riley-serving-260913-recovery/cache-residency-trace-private-v1`; curated: `cache-residency-trace-export-v1`.
