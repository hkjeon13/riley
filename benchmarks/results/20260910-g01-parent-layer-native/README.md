# G01 native parent-layer bridge — 2026-09-10

Decision: **development GPU correctness verified / C07 admission pending**.

`3bc72c9` plus `source-overlay.tar.gz` reproduces the tested implementation. The
archive was applied to an isolated remote source tree, excluding all unrelated
local `crates/riley-model` changes. `manifest.json` records the overlay, ten source
file hashes, GPU identity and the final parent-attention test binary hash.

## Implemented

- Additive C ABI takes opaque allocation owners plus layer count/index. Native
  validation derives the exact BF16 `[layers,P,KVH,16,64]` parent span, rejects
  invalid/overflowing geometry and retains exclusive **whole-parent** leases.
- Both generic and shared-GQA kernels read the checked layer offset. Existing
  C05-18/C05-19 entry points preserve their zero-offset semantics.
- Safe Rust graph borrows the real stream, parent K/V, query, output and five
  metadata allocations until close. Replay waits for completion and any failure
  makes further replay terminal. Native leases also protect forgotten owners.
- The executor cold helper derives its descriptor from `KvLayout`, checks exact
  offset/stride/parent size and borrows actual executor buffers. It is not called
  by default dispatch. Packed-slab metadata is explicitly rejected.

## Verification

| Gate | Result |
|---|---|
| CUDA crate CPU unit tests | 77 passed |
| Existing graph CPU contracts | 32 passed |
| Runtime CPU unit tests | 257 passed |
| Executor architecture contracts | 15 passed |
| Existing G01 CPU boundary | 1 passed |
| New parent-layer GPU target | 2 passed, 0 ignored, **process exit 0** |
| Existing C05-19 attention GPU | 4 passed, **process exit 0** |
| Existing C05-18 KV-write GPU | 3 passed, **process exit 0** |
| Remote runtime `--features cuda` check | passed, process exit 0; 3 warnings in unchanged batch_executor.rs |
| `cargo fmt --all --check`, `git diff --check` | passed before report writing |
| Strict Clippy | not passed; unchanged HEAD reproduces existing diagnostics |

The two new GPU cases exercise QH=3/KVH=3 generic dispatch and QH=9/KVH=3
shared-GQA dispatch. Each captures the middle of three distinct layers twice,
replays 64 times per capture, compares output bytes with isolated eager, checks
both complete KV parents are unchanged, and closes all allocations to zero.
Wrong parent capacity and wrong context must reject before a subsequent valid
capture succeeds. This is synthetic correctness evidence, not model/TPOT evidence.

Strict Clippy was also run on an isolated unmodified HEAD: CUDA reports 210
existing diagnostics. New parent-attention module diagnostics were reduced to
zero; no existing lint gate was weakened. The baseline log is retained.

Remote environment: `ssh ai-assistant`, RTX 4090, driver 580.173.02, CUDA 12.8.1,
SM89. At 14:36 KST I/O PSI full avg10 was 60.86%; therefore this shared host is
not a controlled performance environment, despite these correctness passes.

Commands used with `PATH=/data/cuda-12.8.1/bin:$PATH`,
`CMAKE=/data/cmake-3.31.12/bin/cmake`,
`CARGO_TARGET_DIR=/tmp/riley-g01-native-260910-target`:

```sh
cargo test -p riley-cuda --features cuda --test graph_parent_attention_gpu -- --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_c05_19_gpu -- --ignored --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_c05_18_gpu -- --ignored --test-threads=1
cargo check -p riley-runtime --features cuda
```

Final GPU runs have 120–180 second outer timeouts; all three returned 0. The
initial one-case development run also returned 0 and is superseded by the final
two-case source overlay. The earlier pre-change test's exit-124 result is not
used as a pass for this implementation.

## Remaining G01 acceptance

- Bind the canonical packed metadata slab, including exact metadata identity and
  completion evidence, rather than the currently supported five-buffer transport.
- Join the pre-existing CPU identity/layout/lease contract and C07 evidence
  inventory with the native owner; keep Attention `Unknown` until that chain is
  complete.
- Exercise this owner through actual model executor capture/replay and validate
  complete model/token/KV continuation parity. The cold helper is source-checked,
  not a production dispatch result.
- Candidate qualification, full decode graphs, competitive campaigns and default
  promotion remain separate work. No performance improvement is claimed here.
