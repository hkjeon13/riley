# G02P3 packed RoPE position span — 2026-09-10

Development bridge and actual-executor cold Q/K audit verified. Full G02P3
in-flight layer binding / G02H aggregate / G03 full decode graph remain pending.

## Implementation

The existing indexed-RoPE graph reads positions at allocation byte zero, while
iteration-batch executor metadata stores positions inside a packed slab. An
additive positions-span C ABI now takes an aligned byte offset and retains the
whole opaque parent allocation, rather than borrowing an unowned raw pointer.
The original C ABI delegates with offset zero. No kernel math changed.

Capture, captured graph and graph executable retain the immutable position
offset. Alignment and remaining parent capacity are checked without overflow
at begin and lifecycle validation; capture/graph identity comparison includes
the offset, transfer preserves it, and capture cleanup clears it. Five distinct
device owners and the stream keep their existing exclusive leases.

BorrowedRopeGraph borrows the actual input/cos/sin/metadata/output/stream through
native destruction. Public preflight validates shape, aliasing, context, host
position bounds and aligned parent span. Host mirror bounds alone are not device
metadata identity evidence.

The executor cold audit derives the real PackedIterationLayout from packed rows,
checks GPU position bytes against that iteration's mirror, and uses actual
hidden_projection -> hidden_rotary (Q) and key_raw -> key_rotary (K), with the
actual absolute tables and head dimensions. It runs 16 graph replays, compares
eager output bytes, verifies finite output, untouched inputs/tables/whole slab
and tails, then restores and reads back the original output. Errors poison the
audited executor. Stale-position input is explicitly rejected in the model tests.

This operates between completed M=1 iterations and audits the shared allocations.
It does not claim each layer's in-flight activation or full-graph swap schedule.
No retained full-chain graph owner, inventory promotion, hot dispatch or defaults
changed. Global inventory remains 7 Supported / 7 Unknown, aggregate Unknown.

## Verification

| Gate | Result |
|---|---|
| New borrowed span fixture | 1 GPU test passed |
| Direct native offset rejection | 1 GPU test passed |
| Actual HF + canonical executor continuation | 2 GPU tests passed |
| Existing graph lifecycle | 40 GPU tests passed |
| Existing parent attention | 4 GPU tests passed |
| CUDA CPU library / graph source contracts | 81 / 32 passed |
| Runtime CPU / architecture / inventory | 258 / 15 / 1 passed |
| C11 ABI signatures / format / diff | passed |

**48 GPU tests total**, all successful commands exited 0. Existing graph tests
were run after the native span change. Earlier failed compilation is preserved
in gpu-span.log: an edit incorrectly inserted an assignment into the clear-state
predicate; corrected before successful native/GPU runs. Initial lint warnings
and their corrected output are retained. Normal final Clippy exited 0 with no
new RoPE module diagnostics; baseline repository warnings remain.

The synthetic span fixture covers (rows,heads,offset) = (1,3,0), (1,9,64),
(3,3,128), (3,9,64), D64, eight table positions. Each case exercises explicit
close and Drop, 32 replays each, full parent/table/input preservation, output-tail
sentinels, invalid offset recovery, stable allocations and zero-resource close.
Direct native preflight rejects unaligned, end-of-parent and overflowing offsets.

The two actual-model tests also retain the prior all-layer norm audit. After a
one-token prefill and three decode steps, full logits and initialized KV match
an independent eager executor, all scratch is restored, allocation counts are
stable, and resources close to zero. SmolLM2 continuation remains
[808,2775,288,536]; canonical fixture [2,2,2,2]. Each model audits Q/K at four
positions, with 16 replays per audit. A stale host position after the final step
is rejected and the candidate is poisoned without allocation growth.

## Reproduction

Use source-overlay.tar.gz over base_revision in manifest.json. Source hashes
were compared to the remote tree. Binary/GPU identities and raw logs accompany
this receipt; the checkpoint declarations are copied from the preceding norm
receipt, and payloads are verified by the loader at every model load.

```sh
cargo test -p riley-cuda --features cuda --test graph_rope_span_gpu -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --lib native_rope_span -- --ignored --nocapture --test-threads=1
cargo test -p riley-runtime --features cuda --lib c07_layer_norm_model -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_gpu --test graph_parent_attention_gpu -- --include-ignored --test-threads=1
```

Remote scratch /tmp/riley-g01-native-260910, target
/tmp/riley-g01-native-260910-target, CUDA /data/cuda-12.8.1/bin and
CMAKE=/data/cmake-3.31.12/bin/cmake. Set RILEY_REAL_CHECKPOINT and
RILEY_CANONICAL_CHECKPOINT to the paths in checkpoint-identities.json.

SmolLM2-first matched vLLM benchmarking remains the selected comparison scope.
Shared GPU usage/high host I/O pressure exclude latency/throughput claims here.
Remaining metadata/KV-write/MLP/residual/GEMM/head/output ownership and aggregate
integration, full decode graph, runtime selection and qualification are not
closed by this receipt. No deployment or measured vLLM speedup.
