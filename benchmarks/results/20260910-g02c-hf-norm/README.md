# G02C-HF: SmolLM2 FinalNorm graph — 2026-09-10

**Development implementation and GPU correctness verification complete.**
The real SmolLM2 executor can now retain a profile-specific FinalNorm graph with
same-owner byte parity evidence. Full decode graph and qualification remain pending.

## Implementation

- Separate HF begin/enqueue C ABI entry points capture the existing
  `hugging_face_smollm2_rms_norm_kernel<false>` through the same launch helper used
  by eager execution. No kernel math, reduction order, default profile or hot
  runtime dispatch changes.
- Both Rust and native preflight require BF16, hidden size 576, rows 1..8192 and
  exactly the 1e-5 float bits. The legacy norm-family resource tag has a separate
  immutable profile field, checked in capture/graph/exec validation, cleared on
  ownership transfer and retained in the destination. Mixed-profile enqueue is
  rejected before launching a kernel.
- The existing exclusive input/weight/output/stream leases, abort recovery,
  graph instantiate/replay/completion and close lifecycle are reused. The C11
  ABI symbol checks include the new signatures.
- The actual M=1 executor helper derives its profile, geometry, final weight
  identity and buffers from its prepared plan. C07 binding includes the exact
  profile; canonical evidence cannot stand in for HF evidence. Fixed37 remains
  rejected. Global primitive inventory and aggregate admission are unchanged.
- A cold same-resource graph/eager parity transaction verifies finite output,
  unchanged input/weight, untouched output tails and allocation accounting before
  retaining an owner. Replay/close failure poisons the executor; validation-only
  preflight rejection remains recoverable.

## Verification

| Gate | Outcome |
|---|---|
| Actual HF model, canonical model, owner matrix | 3 GPU tests passed |
| Native profile swap/abort/recovery | 1 GPU test passed |
| Existing graph lifecycle suite | 40 GPU tests passed |
| C05-22 canonical norm/GEMM composite | 3 GPU tests passed |
| Existing parent attention graphs | 4 GPU tests passed |
| Runtime CPU library | 258 passed |
| CUDA CPU library | 80 passed |
| CUDA graph source contracts | 32 passed |
| Architecture + inventory boundaries | 15 + 2 passed |
| Format/diff checks | Passed |

GPU test commands completed with exit 0: **51 total GPU tests**, including the
existing regressions. This is a development correctness receipt, not release
qualification or timing evidence.

Real model: SmolLM2-135M BF16, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`. Two independent executors process a
one-token prefill and three decode steps. After each candidate step, 32 FinalNorm
graph replays preserve the actual normalized bytes, full logits and every
initialized KV token across all 30 layers and heads. Greedy continuation is
`[808, 2775, 288, 536]`. All resources close to zero.

The synthetic H64/L2 canonical fixture exercises the same executor helper and
continuation checks, producing `[2, 2, 2, 2]`; it is not model-quality evidence.
The per-owner fixture covers canonical (1,64), (3,257), (1,576) and HF H576 with
1, 3, 8, 16, 17 rows, two captures × 64 replays per case. It also verifies
wrong-context recovery, foreign weight evidence, non-finite rejection/poison,
explicit close and Drop. The maximum 8192-row admission boundary is CPU-validated;
this receipt does not claim an 8192-row GPU matrix.

One source-text contract initially expected the kernel launch inside the public
canonical enqueue wrapper. It was updated to verify the shared enqueue helper,
the exact canonical/HF profile argument and mismatch guard, retaining the
allocation/synchronization prohibitions. The initial failure log is preserved;
all 32 final contracts pass. Normal Clippy outcomes are in `lint-summary.json`;
existing repository warnings are not presented as a clean strict warnings gate.

## Reproduce and scope

Use the base revision in `manifest.json` plus `source-overlay.tar.gz`; the overlay
includes prior G01/G02C source dependencies and excludes unrelated local model
loader edits. `source-hashes.json` was compared to the remote source tree.
Checkpoint declarations and binary identities accompany the logs. The model
loader verifies declared payload checksums during each model test.

Remote environment: `ssh ai-assistant`, RTX 4090 SM89, CUDA 12.8.1,
`CMAKE=/data/cmake-3.31.12/bin/cmake`, CUDA toolkit on PATH and an isolated Cargo
target. Set `RILEY_REAL_CHECKPOINT` to the pinned model and
`RILEY_CANONICAL_CHECKPOINT` to the fixture generated with
`generate_canonical_fixture.py` from the preceding G02C receipt (copied here).

```sh
cargo test -p riley-runtime --features cuda --lib c07_final_norm_ -- --include-ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --lib hf_norm_native -- --ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_gpu --test graph_c05_22_gpu --test graph_parent_attention_gpu -- --include-ignored --test-threads=1
```

G02P2 per-layer normalization owner binding remains separate. Remaining G02
operations/aggregate, G03 full decode graph, qualification and matched vLLM
benchmarks are still pending. Global inventory stays 7 primitive Supported / 7
Unknown, aggregate Unknown. Shared GPU use and high host I/O pressure exclude
latency/throughput claims from these runs. No deployment or default promotion.
