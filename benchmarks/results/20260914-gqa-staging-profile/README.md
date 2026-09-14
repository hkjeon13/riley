# Same-binary rolling / GQA model profile

One bounded Nsight trace each in control-shared, GQA-shared, GQA-unique, control-unique order, using the serving-screen binary and fixtures. C32/active32, identical 2048 KV blocks and 512-page unique prefix cache. Each trace executes 64 warmup + 64 measured requests with 32 generated tokens: all 512 responses match the prior reference. All owned profiler/server processes exited; three Blender scene queries passed after restoration. Static public viewers stayed running. These are profiled diagnostics, not throughput measurements or repeated statistical estimates.

Accounting selects the middle 80% of graph launch count, including warmup and phase transitions. The shared pair has matching stage counts (18 ordinary, 76 future, 26 mixed); unique has 15 ordinary, 68 future, 133 mixed. Matching counts do not establish identical scheduling metadata. Rolling can execute consecutive future-decode graphs, so the analyzer uses explicit previous-stage → next-stage labels instead of requiring an ordinary graph before every future graph.

| Workload | Backend | Mixed graph sum ms | Mixed attention ms | Projection ms | FFN ms | All selected graph ms |
|---|---|---:|---:|---:|---:|---:|
| shared | control | 96.676 | 42.589 | 23.439 | 15.071 | 241.122 |
| shared | GQA | 99.724 | 44.353 | 23.930 | 15.349 | 247.287 |
| unique | control | 718.923 | 257.990 | 184.432 | 198.460 | 857.214 |
| unique | GQA | 705.698 | 244.883 | 184.319 | 198.336 | 843.773 |

In the selected unique region, attention falls 5.08%, mixed graph time 1.84%, and total graph time 1.57%. Across the whole raw trace (including excluded endpoints), attention falls about 4.50%; these are different scopes. In the selected shared region attention rises 4.14%, mixed graph time 3.15%. GQA is actually dispatched, so missing backend selection does not explain the failed serving gain. No causal claim about occupancy, HBM bandwidth or cache residency follows from these timings. Unchanged kernels also vary in shared, reinforcing the single-trace limitation.

The original mixed attention accounts for 35.89% of unique mixed graph time. Projection accounts for 25.65%, FFN 27.61%. Optimizing only this attention by 5% can at most remove roughly 1.8% of mixed time before other effects. This does not bridge the measured serving deficit. Keep GQA unpromoted; move to the remaining prefill projection mainloop as an area batch rather than tuning this staging candidate further.

Source inspection in `prefill_shape_projection.cuh` shows global operand loads inside each K16 MMA iteration; Q/K/V and output projection keep intermediate BF16 rounding at K192/K128 boundaries. [PR22](../../../deploy/260913/22-prefill-projection-pipeline.md) addresses operand reuse and pipelining while preserving that recurrence. It is a candidate, not a predicted speedup.

[evidence.tar.gz](evidence.tar.gz) contains only curated numeric analysis, safe launch/exit receipts and benchmark request evidence. Raw Nsight and SQLite files remain on the remote host. Recompute [receipt.json](receipt.json) with `python3 benchmarks/analysis/export_gqa_staging_profile.py benchmarks/results/20260914-gqa-staging-profile`. The verifier checks accounting, hashes, backend selection, reference responses and lifecycle receipts. Recomputing kernel timestamps themselves requires the hash-bound private remote databases and `gqa_staging_trace_analysis.py`; numeric consistency is not independent remeasurement. Existing [unprofiled serving comparisons](../20260914-gqa-staging-serving/README.md) remain authoritative for performance.
