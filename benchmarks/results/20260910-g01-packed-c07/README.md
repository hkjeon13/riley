# G01 canonical packed metadata and live attention evidence — 2026-09-10

Decision: **implementation/development GPU verification complete; candidate qualification pending**.

The canonical packed slab is now connected to the native parent-layer graph.
A live C07 owner supplies Attention evidence only after actual same-resource
GPU graph/eager byte parity. Global C05 capability queries and normal executor
dispatch remain unchanged; the aggregate full-decode capability remains Unknown.

## Changes

- A packed native entry point retains the metadata allocation exactly once.
  Its five field offsets are copied into immutable graph state, checked for
  alignment, bounds and overlap, and retained across capture/graph/exec. Abort
  and close release one slab lease, preserving the existing separate-buffer ABI.
- Safe preparation compares every used metadata field against the validated
  batch witness, completes the exact payload H2D, then captures that same slab.
  No caller-provided freshness flag is accepted.
- The C07 transaction compares complete host/device/expected layouts before
  device work. It derives fields from the canonical layout and KV spans from
  `KvLayout`. A graph probe completes and closes before output/metadata readback;
  eager output is then compared byte-for-byte, non-finite results are rejected,
  and unchanged borrowed inputs are captured into the retained final owner.
- Attention evidence is scoped to the live owner, exact metadata/KV layouts,
  layer and query-head count. Mismatches remain Unknown and explicit Unsupported
  evidence is preserved. Other operation slots are unchanged.
- CUDA execution/close failures poison the enclosing executor. Validation-only
  rejection permits recovery. Explicit close and Drop settle the graph before
  releasing the executor's poison-state borrow.
- The executor cold helper uses actual model-owned rotary queries, parent KV and
  attention output with a canonical C07 device slab. No full decode graph or
  default runtime dispatch is introduced.

## Final verification

| Gate | Result |
|---|---|
| CUDA CPU unit tests | 79 passed |
| CUDA graph CPU contracts | 32 passed |
| Runtime CPU unit tests | 257 passed |
| Executor architecture contracts | 15 passed |
| Existing G01 and inventory boundary tests | 2 passed |
| Native GPU: parent attention + C05-18 + C05-19 | 11 passed, all processes exit 0 |
| Canonical C07 owner GPU test | 1 passed, exit 0 |
| Actual model continuation GPU test | 1 passed, exit 0 |
| Format and diff checks | passed |
| CUDA-feature runtime Clippy (normal warning mode) | exit 0; no diagnostics in the two new implementation modules |

Strict `-D warnings` is not a clean repository gate: the previous isolated HEAD
check reproduced existing diagnostics. This work does not weaken lint settings.
The final CUDA build retains the pre-existing unused `BF16_BYTES` warning.

### Actual model proof

Checkpoint: `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, BF16. Actual checkpoint file hashes
were independently read and match the manifest; see `remote-identities.txt`.

Two model executors run the same one-token prefill and three greedy decode steps.
After each decode, the candidate's **actual last-layer (29) query and KV** are
used for the canonical graph/eager parity transaction, followed by eight graph
replays. The test checks:

- Exact logits before and after the graph intervention and at subsequent steps.
- Greedy output tokens `[808, 2775, 288, 536]`.
- Every initialized KV token, across all 30 layers and every KV head, against
  the independent eager executor. Uninitialized capacity bytes are excluded
  from cross-allocation comparison.
- Stable live GPU allocation accounting and zero allocations after all closes.
- Per-step SHA-256 values for canonical metadata and attention output.

Separate native tests use distinct values across three parent layers, compare
whole-parent bytes, exercise generic/shared-GQA dispatch and verify single-slab
lease release. The canonical test rejects a different complete layout even when
size/alignment are similar and proves partial inventory remains Unknown.

This validates an M=1 attention owner and short real-model continuation. It does
**not** establish a full-model CUDA graph, long-context/concurrent/Qwen coverage,
TPOT gains, a vLLM win, HTTP behavior, release qualification or default promotion.

## Reproduction and retained failures

Start from `manifest.json`'s base Git revision and apply `source-overlay.tar.gz`.
All 15 included implementation files were SHA-256 compared with the actual remote
source tree. Unrelated local `crates/riley-model` modifications are excluded and
were verified unchanged during this turn. Binary/checkpoint hashes and GPU
identity are in `remote-identities.txt`.

Environment: `ssh ai-assistant`, RTX 4090, CUDA 12.8.1, driver 580.173.02, SM89.
With the CUDA toolkit on PATH, explicit CMake and an isolated Cargo target:

```sh
cargo test -p riley-cuda --features cuda --test graph_parent_attention_gpu --test graph_c05_18_gpu --test graph_c05_19_gpu -- --include-ignored --test-threads=1
cargo test -p riley-runtime --features cuda --lib c07_packed_attention_owner -- --ignored --nocapture --test-threads=1
RILEY_REAL_CHECKPOINT=/data/riley-benchmark/20260827T051948Z-d7ad713a/model cargo test -p riley-runtime --features cuda --lib c07_model_owned_packed_attention -- --ignored --nocapture --test-threads=1
```

The initial actual-model fixture incorrectly kept the block-table logical length
at 1 during decode. That run failed; its log is retained separately. The fixture
now advances logical length with the row's target length, and the corrected run
passes without relaxing any assertion. Development compilation-error logs are
also retained. Host I/O pressure remains high, so elapsed run times are not
performance evidence.

Next: remaining G02 owner/capability slices and qualification, followed by G03
full decode capture. G01's development implementation no longer waits on packed
metadata or an actual-model attention check.
