# V47: widen decode execution to32 rows

Status: primitive and full serving integration validated; Round54 serving comparison is complete; global adoption is withheld. V46 remains the accepted measured baseline pending these results. The prototype findings below are historical evidence for the implemented batch.

Motivation: Round53 shows C32 natural throughput6757.8tok/s versus C16 7005.6, while the scheduler dispatches only16 GPU rows. Fresh V46 profiling identifies decode attention/projections as major residual cost after compact D2H. This batch targets calculation and dispatch width together.

Prototype changes: a warp reuses each loaded weight fragment across two ordered M16 MMA accumulators; all original BF16 interval-rounding and merge orders remain. Fused QKV offsets and gate/up/SwiGLU scratch cover32 rows. Attention launches32 row/head groups with unchanged per-row arithmetic. This differs from the rejected V45 grouped-value kernel: no cross-head arithmetic or recurrence changes.

Extended primitive run:612 configurations pass exact BF16 reference and output guards under both memcheck and racecheck: five projection shapes, three seeds, row counts0..33; fused gate/up; fused QKV; attention with mixed1..4096 contexts and permuted disjoint pages. All invalid/inactive rows preserve sentinels. This is not a full-model32-row test.

Four reversed-order graph timing pairs,40 warmups and200 graph replays each. Baseline processes the same rows with one or two existing16-row dispatches. Candidate handles them in one32-row kernel group. These are component GPU event times and exclude serving admission, model-wide KV transitions, sampling, transport and CPU overhead.

| Component,32 active rows | Existing16-row groups µs | Prototype32 µs | Change |
|---|---:|---:|---:|
| projection-576-576-192-0 (context0) | 12.598 | 8.192 | -34.97% |
| projection-192-576-192-0 (context0) | 11.602 | 7.055 | -39.19% |
| projection-576-576-128-0 (context0) | 11.890 | 7.593 | -36.14% |
| projection-1536-576-0-1 (context0) | 9.259 | 7.157 | -22.71% |
| projection-576-1536-320-1 (context0) | 12.006 | 7.880 | -34.37% |
| gate-up-swiglu (context0) | 11.360 | 7.671 | -32.47% |
| attention (context16) | 13.061 | 9.684 | -25.85% |
| attention (context128) | 15.662 | 11.779 | -24.80% |
| attention (context398) | 28.001 | 21.885 | -21.84% |
| attention (context1024) | 53.740 | 43.812 | -18.47% |
| attention (context4096) | 215.595 | 162.524 | -24.62% |
| attention (context-1) | 116.785 | 71.777 | -38.54% |
| qkv-fused (context0) | 9.548 | 5.796 | -39.30% |

The16-row and smaller cases regress in several projections/short attention cases. Preserve V46 for existing low-concurrency configurations. Do not enable32-row graphs universally or infer vLLM parity from this prototype.

Integration scope for the same optimization batch:

1. Add a versioned32-row wire/session contract with distinct request/result magic and exact extents, native packet validation, fresh staging and capacity-bound catalog identity. Keep8/16 compatibility and compact/full modes.
2. Connect the wider model sequence: QKV/rope/KV, attention, projection and gate/up kernels; validate parent extents, head GEMM support, scratch lifetimes and zero inactive hidden rows. Preserve full-logit fallback and poisoned-completion cleanup.
3. Extend scheduler execution shape, authorized descriptors, dense output slots and greedy workspace to32. Audit fixed16 argmax arrays in execution.rs before exposing32 full-logit outputs; do not merely increase scheduler capacity while leaving GPU width16. Add an explicit server profile, preserving V46 selection.
4. Full model exact-logit/compact-token tests with32 owners, partial row counts, prefill/decode transitions and outstanding-commit/abort ownership. Then concurrent HTTP greedy/fallback/disconnect validation and matched serving comparison against V46 and vLLM before acceptance.

Current contract surfaces inspected: scheduler config.rs/scheduler.rs; runtime executor/config.rs, multi_descriptor/variable_wire.rs, variable_session.rs, graph_decode_full.rs; server engine.rs/main.rs; native prefill_shape_packet.hpp, graph_resources.cu, prefill model wrappers, compact_shared_result.cuh and decode_shared16_model.cuh. Prepared shared-head GEMM binding is in kernels/src/gemm.cu (verify actual filename with rg). Physical pool maximum4096 remains a separate model constraint; the standalone attention probe uses8192 pages solely to isolate32 maximum-context rows.

Evidence:22 SHA-verified files, `raw/shared32-v47-manifest.json`; archive SHA256b88effa499444769c3f5a14ee65b18211182c644ad21b287c5d91ec69e85f892. Extended results are in `raw/shared32-qkv-v47`. All probe controllers are terminal; Blender remains stopped.


## Integrated candidate

Remote isolated commit `ead5c09f9cb7bd3d62ad39902e65801582ab6479`; frozen V47 SHA256 `0da2d8f74f3fb1fa7d5692fdecd6fa271519a328c6574227e94ef01b7f991090`. Local dirty application source remains untouched.

V5 uses a distinct version/magic and32-row exact wire extent, selected through `variable-smol-v5`. The native model sequence, shared head binding, scratch extents, compact/full result graphs, sealed retained session, scheduler policy and token/argmax workspace now cover32 rows. Existing V3/V4 remain available. The server reports the actual variable profile in its preparation log. A fixed32-entry argmax array keeps full-result handling bounded without a per-iteration argmax allocation.

Validation completed before freezing:
-612 primitive configurations: exact BF16 and guards; memcheck/racecheck clean.
-14 wire tests, including compact32 identity/status/inactive/publication corruption and existing8/16 tests.
-10 owned model tests: all prior8/16 coverage plus full32 and alternating compact/full32, each4,096 output checks, and partial prefill224 checks. The full32 tests reach32 live decode rows, validate scheduler commit identity and allocation-zero close. Total16,320 output checks across all ten tests; mixed compact tests compare tokens to immutable full-logit references, not returned full logits on compact iterations.
-Three V5 owned model tests under memcheck: zero errors.
-Final binary37 HTTP completions exact, including32 concurrent streaming/nonstreaming requests, invalid bound rejection and disconnect/recovery; clean shutdown.
-22 ordered SSE responses per CPU/GPU backend match request IDs, tokens, text and finish reasons, including stochastic fallback; two CLI profile tests pass.

Integration fixes: the initial generated Rust wrapper misread an array-type semicolon; corrected before the successful build. Synthetic32-row fixture pages initially exceeded the unchanged4096-block pool; fixture spacing now keeps its live reservations within the pool. The old unsupported32-capacity test now uses64, while16-row descriptors still reject requested capacity32. The initial mixed request fixture finished short requests before reaching32 active decode rows; additional fixed long-request runs prove full width32. These are test/connection corrections, not permission to weaken ownership or numerical checks.

Round54 runs V46 GPU16 versus V47 GPU16 atC16/GPU32 atC32 versus vLLM, fixed/natural workloads, two reversed orders,96 warmups and384 retained requests per lane. Same external model/workloads, admission/KV/token budgets; internal execution width is the optimization under test. No serving improvement is claimed until all24 lanes complete. Blender remains stopped.


## Round54 complete: mixed outcome

C32 natural throughput+26.36%, TPOT−22.79%, P99−20.07% versusV46, but C32 fixed TTFT7.997→59.466ms with throughput only+0.32%. All9,216 retained requests complete and6,144 Riley references match. Do not globally replaceV46. Fresh fixed traces show only13.22→14.48 mean published decode rows, and serving-client first-token wait backlog median1→13; the next batch must address multi-owner prefill supply and class scheduling. See `V47_SERVING_RESULTS.md`. All292 exported files are SHA-verified; Blender stays stopped.
