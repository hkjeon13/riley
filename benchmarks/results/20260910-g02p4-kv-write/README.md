# G02P4 packed parent KV write — 2026-09-10

The native/Rust bridge and actual-executor **cold resource audit** are implemented
and GPU-verified. In-flight layer activations, retained aggregate admission and
full decode graph integration remain pending; this is not performance evidence.

## Change

An additive packed-parent KV-write C ABI derives the D64/page16 layer span from
parent layer count/index, physical block count and KV head count. It retains
exclusive leases on both whole KV parents, both dense sources, the whole metadata
slab and the stream. The existing standalone C05-18 ABI keeps zero layer offset
and separate metadata allocations.

The shared ragged state already carries parent geometry and packed-field offsets
for attention. KV write now validates and retains the same fields through begin,
enqueue, graph/exec transfer and destruction, selecting its own key parent rather
than attention's query/output field mapping. Packed metadata acquires one lease,
not five; five aligned non-overlapping views remain immutable. The write kernel
uses derived K/V layer offsets and those metadata views. Kernel arithmetic and
raw-invalid-row behavior are unchanged.

The Rust borrowed owner checks exact whole-parent capacities, distinct/context-
matched idle resources, and the actual device metadata against a validated host
batch before capture. It does not upload replacement metadata. Replay waits for
completion; terminal failures keep reuse blocked and native destruction precedes
release of Rust borrows.

The actual executor audit derives the packed layout and every layer's geometry
from its own plan/KvLayout. It captures writes from actual key_rotary/value_raw
allocations, compares the **entire** K/V parents against an independent CPU scatter
oracle, restores and reads back the original parents, and verifies untouched
sources and the complete metadata slab. Any transaction failure poisons the
executor. It requires a completed healthy M=1 iteration and D64.

These are shared scratch sources sampled between iterations; they are not claimed
to contain each layer's historical in-flight activations. Full-chain binding and
buffer scheduling still need G02H/G03. Inventory and hot dispatch are unchanged.

## Verification

All successful test commands exited 0: **15 GPU tests**.

| Gate | Passed |
|---|---:|
| New packed write / CPU scatter / parent preservation | 1 |
| Direct native parent and metadata rejection | 1 |
| Actual SmolLM2 and canonical continuation | 2 |
| Existing C05-18 KV-write regression | 3 |
| Existing C05-19 attention regression | 4 |
| Existing parent attention regression | 4 |
| CUDA CPU library / graph source contracts | 81 / 32 |
| Runtime CPU / architecture / inventory | 258 / 15 / 1 |

The synthetic write fixture uses three layers, two physical blocks, two KV heads,
two rows at positions 1 and 17 and a nontrivial logical-to-physical block mapping.
All three layers are exercised with explicit close and Drop, each with 32 replays.
The CPU oracle verifies every parent byte, including other layers, other tokens
and blocks. Sources, metadata and allocation counters are unchanged. Stale host
metadata is rejected and correct metadata can subsequently capture and replay.
Native preflight separately rejects wrong layer index, parent size/count,
overlapping/misaligned/overflowing metadata views before capture.

SmolLM2 H576/L30: 30 layers × four iterations = 120 write/restore transactions,
eight replays each. Canonical H64/L2 contributes eight transactions. The existing
model tests also run the previous norm/RoPE audits. Independent executors agree
on full logits and all initialized KV, with continuation [808,2775,288,536] for
SmolLM2 and [2,2,2,2] for the fixture. All resources close to zero.

C11 ABI signature, format and diff checks passed. Normal Clippy exited 0; no
new KV-audit/graph implementation diagnostics. Existing repository warnings,
including the previous unused legacy RoPE FFI declaration, remain.

## Reproduce

Apply source-overlay.tar.gz to base_revision from manifest.json. Source hashes
were verified against the remote scratch and all six unrelated local model-loader
edits were preserved. Binary, checkpoint manifest and GPU identities accompany
the raw logs; the loader validates model payloads on each model test run.

Remote: ssh ai-assistant, scratch /tmp/riley-g01-native-260910, target
/tmp/riley-g01-native-260910-target; PATH includes /data/cuda-12.8.1/bin;
CMAKE=/data/cmake-3.31.12/bin/cmake. Model paths are in runtime-identities.json.
Set RILEY_REAL_CHECKPOINT and RILEY_CANONICAL_CHECKPOINT accordingly.

```sh
cargo test -p riley-cuda --features cuda --test graph_kv_parent_gpu -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --lib native_packed_kv_write -- --ignored --nocapture --test-threads=1
cargo test -p riley-runtime --features cuda --lib c07_layer_norm_model -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_c05_18_gpu --test graph_c05_19_gpu --test graph_parent_attention_gpu -- --include-ignored --test-threads=1
```

Next: MLP SiLU/gated multiply/residual ownership, remaining GEMM/head/output
capabilities and aggregate integration. SmolLM2-first matched vLLM benchmarking
remains pending after full decode graph correctness and candidate qualification.
No deployment, default promotion or measured speedup is claimed.
