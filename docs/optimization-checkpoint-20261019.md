# Optimization checkpoint — 2026-10-19

The user paused optimization and requested a commit/push and deletion of remote branches except `main`. The local primary checkout's935 pinned source files matched `candidate-v164-actual-integrated-manifest.json` before this checkpoint. This commit preserves those runtime changes, tests and existing qualification documents; it does not adopt private SM80 candidates v225/v235 or claim the multi-model stock-vLLM target is achieved.

## Included work

- Dense Mistral config/tokenizer support, canonical HF normalization/projection paths and component/teacher regression probes.
- Explicit owned decode batching and segmented CUDA Graph reuse, including the geometry/environment-gated SM89 paths described in the accompanying optimization documents. Existing flags/defaults and unsupported-device fallbacks remain represented in source.
- Separately replayable SM80 QKV/Gate-Up proposal at `benchmarks/candidates/sm80-bounded-projections-v235/candidate.patch`. It is deliberately not applied to the committed runtime.
- Compact manifests and validation receipts in `benchmarks/evidence/20261019-optimization-checkpoint/`. Raw success/failure logs, model-derived BF16 data, binaries and8.6GB of local experiment artifacts are retained under the ignored local `artifacts/` directory, with remote evidence retained on the original hosts.

## Validation and limits

`cargo test --locked --workspace` completed on macOS/Rust1.85 with826 passed,0 failed and37 ignored across122 reported suites. CUDA-only targets are gated or ignored locally; these numbers are not GPU qualification. `git diff --check` passed. `cargo fmt --all -- --check` failed on existing formatting differences in28 Rust files. This is a preservation checkpoint, not a claim that every CI check is green; no broad formatting rewrite was mixed into the hash-bound runtime snapshot.

The SM80 Gate/Up-only v225 candidate previously passed fresh same-A100 HF104 full-vocabulary rows/model, full native before/after byte equality and actual Graph/captured GEMM-argument checks for both Qwen3-4B and Mistral7B. At batch32, Gate/Up submissions fell from64 to4 per layer; this is not a throughput ratio. Its source remains unapplied.

The v235 QKV extension built successfully on SM80 and passed518 CPU regressions. Qwen3-4B's fresh candidate C1 and all1..32-row changed-context Graph helper completed successfully before the pause. However, the independent actual-QKV-call auditor did not find expected Lt/GemmEx inspection records; that audit is failed/incomplete, not a numerical mismatch verdict and not proof of the fast path. Its raw failure is preserved. Mistral's new-candidate whole-model checks and heterogeneous API/performance comparisons were not started. Fresh15-prompt HF generation references for each model exist separately from teacher-forced references.

## Paused execution

Local controller25322 was stopped before further experiments could launch. Its already-started finite Qwen3 Graph job completed with exit0; inspection logs were copied from RAM to persistent storage and A100 GPU2/3 ownership was rechecked afterward. No foreign jobs, production services, linked worktrees or local branches were removed. The existing `riley-gpu` heartbeat was already paused. Resume only on user instruction, inspecting actual processes and receipts first; never restart a completed job simply because the controller was interrupted.

The only remote head observed before push was `origin/main`; there were no other remote branches to delete. This is distinct from local branches and linked worktrees, which are preserved.
