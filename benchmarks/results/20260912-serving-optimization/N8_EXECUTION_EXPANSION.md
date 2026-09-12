# N8 execution expansion

The current working serving candidate remains V6. The N8 branch has only qualified the strided GEMV primitive; its server still has a four-sequence limit. This is an implementation stage toward the serving objective, not a smaller success criterion.

## Evidence and decision

N2/N4/N8: 30 compact/padded cases, 90 captured replays, exact impulse outputs, invalid-span reuse, unchanged input and explicit/drop close pass. Dense N4/N8: 120 rows (40 N4, 80 N8) match independently executed M1 exactly across five projection shapes and two input/weight patterns; 60 captured replays pass. Two CPU configuration tests include supported eight-row storage and rejection of adjacent unsupported counts.

The nonexclusive dense graph trace has six observations per shape and batch. Median N4/N8 GPU spans in microseconds:

| Projection N,K | N4 | N8 | N8 / N4 |
|---|---:|---:|---:|
| 960,576 | 5.008 | 8.208 | 1.639 |
| 3072,576 | 11.808 | 22.432 | 1.900 |
| 576,576 | 3.616 | 5.408 | 1.496 |
| 576,1536 | 4.704 | 7.153 | 1.521 |
| 49152,576 | 150.945 | 271.282 | 1.797 |

N8 can reduce per-row GPU cost, but does not double throughput. Gate/up and head scaling are weak. These are synthetic projection measurements without whole-model cache history, HTTP timing or exclusive GPU control. They justify implementing and measuring N8, not accepting a serving improvement.

## One capacity-expansion batch

1. Version the private wire format and its independent fixtures. Use version2, eight 128-byte row slots after the 128-byte header, request size1792, result prefix1152, prefill compatibility view at1152 (600bytes), and reserved tail1752..1792. Header row capacity becomes8. Keep row field offsets and per-row block capacity10. Full results append bucket*98304bytes at1152. Buckets1/2/4/8 represent active counts1/2/3..4/5..8. Old V1 golden packets must be explicitly rejected; generate independent V2 golden packets and a pinned layout artifact instead of retaining the old contract digest.
2. Expand retained CUDA catalog and cold parents to N2/N4/N8. Add a third catalog slot and index3, update every close/read/replay bound and partial-capture cleanup path. Extend geometry, packet admission, per-row kernels and argmax to8, with up to80 unique KV blocks. Audit all constants and byte offsets in both Rust and CUDA. Update capture identity for the changed ABI. Preserve one outstanding submission and scheduler-confirmed completion semantics.
3. Expand runtime/scheduler/server authority to capacity8 with80 blocks. Update cold owner limits, output slots, cookie storage and preallocated workspaces. Preserve fullP128 dispatch and existing fairness policy initially so serving differences isolate capacity. Ensure descriptors for5..8 rows select N8, and padded rows neither mutate KV nor publish output.
4. Qualify all boundaries together: independent CPU wire fixtures and malformed final-row/slot/cookie/physical-block cases; actual-model full-logit and initialized-KV comparisons for shapes1..8; cancellation before/after GPU execution, physical page reuse, stale completion rejection and terminal cleanup. Then freeze a full server candidate and compare V6/V7/baseline/vLLM on the same fixed and mixed serving workloads. Keep high-concurrency vLLM numerical divergence visible rather than treating observations as accepted correctness comparisons.

## Files to audit together

- `crates/riley-runtime/src/llama/multi_descriptor/{mod.rs,tests.rs,fixtures}`
- `crates/riley-runtime/src/llama/graph_decode_{full,multi_parents,multi_session}.rs` and their capture helpers
- `crates/riley-cuda/src/{gemm,graph_resources}.rs` and graph bindings
- `kernels/src/{gemm.cu,graph_resources.cu,graph_multisequence_record.inc,graph_multisequence_catalog.inc,graph_multisequence_packet.inc,graph_multisequence_io.cu,graph_multisequence_precise.cu,batch_primitives.cu,graph_numerics.cu}`
- Scheduler authority/execution/config and server engine/CLI validation

The working source is `/tmp/riley-opt-260912/multisequence-n8-source-v7` on `ai-assistant`; its native-stage commit is recorded in the status log. Do not mutate frozen V6. Native target is `/tmp/riley-opt-260912/multisequence-n8-target-v7`. Continue with full integration; no user approval is pending.

Variable prompt lengths, stronger vLLM numerical qualification, larger concurrency and sustained tail/stability workloads remain necessary for the original goal. N8 success alone cannot satisfy it.
