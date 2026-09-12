# G02C FinalNorm owner binding — 2026-09-10

Canonical-path development implementation and GPU checks are complete.
**SmolLM2 FinalNorm remains unqualified/Unknown** because the real executor uses
its Hugging Face reduction profile, not the canonical RMSNorm primitive.

## What changed

- `BorrowedNormGraph` reuses the existing C05-12 canonical BF16 RMSNorm capture ABI
  with exclusive borrows of actual input/weight/output/stream owners. Native
  permanent leases protect forgotten owners. Capture abort, instantiate,
  completion, explicit close and Drop retain the existing fail-closed semantics.
- A cold `PreparedLlamaBatchOwner` helper derives the M=1 hidden geometry,
  physical weight ID, BF16 dtype, exact weight shape and epsilon from the actual
  prepared plan. It binds `hidden_current`, `hidden_norm` and the uploaded final
  norm tensor without copying/replacing executor weights or buffers.
- C07 compares same-owner graph/eager output bytes, whole input/weight preservation
  and untouched output tails before retaining a final graph. Input, weight and
  output SHA-256 are recorded. Non-finite output cannot produce evidence.
- Typed binding reasons distinguish profile, geometry, epsilon, foreign/other
  physical weight and terminal state. Only a matching live owner supplies
  `FinalNorm`; explicit Unsupported is preserved. Per-layer `Norm` is unchanged.
- Validation rejection before capture is recoverable. Failed CUDA completion,
  close, parity or non-finite results prevent unsafe reuse through executor poison.
  The evidence is retained with actual borrowed resources; hashes alone cannot
  construct it or authorize C06 admission.

## GPU evidence

All tests use `ssh ai-assistant`, RTX 4090 / SM89, CUDA 12.8.1 and driver
580.173.02. These are development correctness checks, not Q-track qualification.

- Canonical owner test: (rows, hidden) = (1, 64), (3, 257), (1, 576), two
  captures × 64 replays each. Byte parity, wrong-context rejection and recovery,
  foreign weight evidence, explicit Unsupported preservation, finite-output
  rejection/poison, explicit close/Drop, tail preservation and zero allocation
  accounting all pass.
- Synthetic canonical model test: a deterministic BF16 Llama H64/L2 checkpoint
  passes the real model loader and executor. Actual final norm owners are
  captured after prefill and three decode steps, with 32 replays per step.
  Normalized bytes and full logits/continuation match the independent eager
  executor. Tokens are `[2, 2, 2, 2]`; this fixture does not measure model quality.
- Real SmolLM2-135M test: the executor's HF profile is checked and canonical
  capture is rejected before CUDA work. No poison/allocation change occurs;
  independent eager logits and continuation `[808, 2775, 288, 536]` still match.
  This is **profile rejection evidence**, not SmolLM2 graph support.

Final GPU results: **3 new runtime tests + 2 existing C05-12 tests passed**, both
commands exited 0. CPU: **258 runtime + 79 CUDA unit + 32 graph contract + 15
architecture + 2 inventory boundary tests passed**. Format/diff checks passed.
Process outcomes and identities are in `manifest.json` and the retained logs.
The initial boundary command named a nonexistent test target; it was corrected
to `architecture_boundary` and all 17 boundary tests passed. The original command
error is retained separately; it was not a product-test failure.
Normal-mode Clippy is used; the repository's existing warnings are not a clean
strict `-D warnings` gate.

## Reproduce

Apply `source-overlay.tar.gz` to the base revision in `manifest.json`. It includes
prior G01 source dependencies and this G02C work, excluding unrelated local
`crates/riley-model` edits. Every included source file is SHA-256 compared with the
remote test tree; see `source-hashes.json`.

Generate a new synthetic fixture directory using the pinned original checkpoint:

```sh
python3 generate_canonical_fixture.py /path/to/pinned-smollm2 /tmp/new-g02c-fixture
```

With `PATH=/data/cuda-12.8.1/bin:$PATH`,
`CMAKE=/data/cmake-3.31.12/bin/cmake`, an isolated `CARGO_TARGET_DIR`,
`RILEY_CANONICAL_CHECKPOINT=/tmp/new-g02c-fixture` and
`RILEY_REAL_CHECKPOINT=/path/to/pinned-smollm2`:

```sh
cargo test -p riley-runtime --features cuda --lib c07_final_norm_ -- --include-ignored --nocapture --test-threads=1
cargo test -p riley-cuda --features cuda --test graph_gpu canonical_bf16_rms_norm -- --include-ignored --test-threads=1
```

## Remaining scope

The new G02C-HF card in `deploy/260904/02-c07-capability-completion.md` requires
capture/lifecycle/parity for the **existing** HF SmolLM2 RMSNorm kernel, followed
by profile-specific owner mapping. Fixed37/fused paths require separate evidence.
The same profile distinction must be audited for G02P2's per-layer normalization.

Global inventory remains 7 primitive Supported / 7 Unknown and aggregate Unknown.
Other G02 slices, G03 full decode capture, production dispatch, candidate
qualification and matched vLLM performance comparison remain pending. Host I/O
pressure and shared GPU processes make these elapsed test times unsuitable as
latency measurements. No default setting, server or deployment was changed.
