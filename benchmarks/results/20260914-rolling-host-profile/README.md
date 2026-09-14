# Rolling host preparation census

The previous host timer skipped `try_rolling_decode` entirely. Opt-in instrumentation now records successful rolling prefix/drain calls without changing scheduler decisions, GPU work, tokens or ownership. It is enabled only by `RILEY_SERVING_PHASE_TIMING=1` and makes no runtime Python calls.

Four bounded Nsight traces compare frozen best split-FFN + rolling decode with the instrumented server on the same C32 shared/unique fixtures. All 512 warmup/retained responses match frozen references. All four profiler/server exits are zero, no owned process remains, and Blender restoration passes. Prefix step counts equal completed rolling extensions plus drains; measured drain counts equal scheduler drain counts.

| Instrumented workload | Rolling prefix calls | Continuation wall ms | Runtime future preparation ms | Runtime buffered submission ms |
|---|---:|---:|---:|---:|
| shared | 83 | 98.527 | 59.343 (83 calls) | 26.328 (150 calls) |
| unique | 85 | 83.782 | 48.219 (85 calls) | 49.850 (267 calls) |

Continuation includes next-window planning, promotion, preparation and GPU submission, or a drain decision; it is not pure CPU time. The nested runtime `future_prepare` timer ends before `submit_future_transfer` and covers the CPU authority callback, successor checks, ledger retention, future sidecar/packet construction and allocations. Submission totals include ordinary and rolling submissions, so they must not be subtracted from continuation. Nested intervals overlap and cannot be added as independent savings.

The instrumented rolling prepare/read wall time is 20.082ms prefix + 11.679ms drain for shared and 46.391ms + 16.426ms for unique. This includes GPU completion waits, cancellation/selection preparation and readback. Sampling/result and settlement counters are much smaller. Publication covers backend event construction, not socket delivery. Timers exclude failed/fallback calls, outer-worker idle and client readiness gaps; they are not total CPU utilization.

| Workload | Stage | Selected graph spans ms | Attention kernel ms |
|---|---|---:|---:|
| shared | decode | 142.838 | 48.330 |
| shared | prefill/mixed | 93.651 | 44.982 |
| unique | decode | 135.087 | 53.639 |
| unique | prefill/mixed | 603.092 | 258.751 |

GPU accounting selects the middle 80% of graph count, mixing warmup and retained requests with profiler overhead. Host timers cover the whole run. The different scopes and concurrent CPU/GPU work prohibit adding these totals or interpreting graph gaps as ready-work CPU stalls. Raw Nsight/SQLite stays remote-private; this artifact contains numeric exports only. No serving speedup is claimed from this diagnostic.

**Wrapper failure retained:** after successfully running the single profile experiment and restoring Blender, the wrapper incorrectly required two result entries left over from an older profile-plus-dispatch controller. Its exit is 1. `collection-audit.json` independently confirms all intended four traces and restoration; it does not rewrite the wrapper failure into success. No GPU run was repeated to conceal the failure.

Next batch targets repeated CPU future-wire preparation, not another short-query attention micro-variant. Split authority preparation from structural validation/encoding first, then group immutable validated-ownership reuse, avoiding successor structural clones, and bounded packet/scratch reuse where profiles support them. Every malformed owner/replay/cookie/page/progress rejection must remain intact. Validate native packet equivalence, model output, stop/cancel/drain, and serving versus frozen best and vLLM 0.29.0 before promotion.

```sh
python3 benchmarks/analysis/verify_rolling_host_profile.py
```
