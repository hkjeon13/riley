# Riley main checkpoint — 2026-10-08 (Asia/Seoul)

This checkpoint records work through Batch12. It is not a completed performance goal or a release qualification.

## Main source and preserved experiment source

- Main fast-forwards the Qwen correctness/diagnostic chain through `8a43885a4c2f26ef2be1e4c0b057f6e6c7af7dea`. The ordinary HTTP own-chosen128 logit observation test and its Cargo registration are included. Private-route correctness does not prove ordinary HTTP correctness.
- The SmolLM2 experiment lineage is a recovered source tree with an independent root. Replacing main with it would remove newer Qwen diagnostics and checkpoint features. Instead, the exact Batch12 candidate tree is saved in `smollm2-batch12-source/`, bound by all 1,083 file SHA256 values in `smollm2-batch12-source-receipt.json`. `batch07-to-batch12.patch` preserves the actual experimental changes. This is source preservation in main, not a claim that main's normal runtime is the measured SmolLM2 binary.
- Candidate Batch12 commit: `da5ec52aefbcb4216d087ef2053974a720b54601`; serving binary SHA256: `cc23b7a4317c64df66c2655892702c143fc3dc6026efb57744af87fc5883e84b`.
- Existing branches, worktrees, untracked files and raw archives are retained. No branch deletion, force push, source reset or archive removal is part of this checkpoint.

## Verified results and limits

- Batch11 completed all96 serving lanes, with independent request replay and launch/host audits. It did not pass the all8 performance minimum and was not adopted. Its absolute throughput, TTFT/TPOT/E2E median/P95/P99, errors, memory and repeat sample SD are in [the absolute report](remote-evidence/kernel-batch11-absolute-report-attempt01/report.md).
- Batch11 profiling completed32 captures and independently replayed6,144 HTTP requests;678 raw file SHA256 values were verified. Kernel intervals and CPU samples are diagnostic. Overlapping times are not added to serving elapsed time; unresolved libcuda samples do not establish host-wait causality.
- Batch12 changes mapped prefill attention packed loads and V fragment lifetimes. Native3,161 cases were BF16/FP32 bit-exact, and HTTP60 requests across all8 cases passed exact-output checks plus independent raw SSE replay. Compiler register observations are not serving performance.
- Batch12 attempt1 failed the original300-second host start gate before any lane. Attempt2 completed24 of96 lanes (C1 fixed/natural), then failed the C8 fixed start gate. The failed attempts, source metadata, host samples and archive receipts are preserved. No full8-cell performance conclusion or improvement/adoption claim is justified. Unrun C8/C16/C32 serving conditions remain unexecuted; sustained/cancellation/source-owned resource qualification remains unexecuted.
- vLLM0.27.1 is a reconstructed comparison lane, distinct from older historical vLLM comparisons; versions are not mixed.
- Qwen ordinary HTTP own-chosen128 is repeatable but still differs from pinned HF at the fourth generated token. Its serving performance and full stability qualification remain unexecuted.

## Evidence and recoverability

`remote-evidence/TRANSFER_INVENTORY.json` binds the copied reports, manifests, controllers, verifiers, logs and summaries to their source SHA256. `REMOTE_ORIGINAL_INVENTORY.json` also records remote raw response files that remain outside Git. Original remote evidence lives under `ssh ai-assistant:/data/riley-serving-261007/`. `archive-summary.json` binds the principal serving archives.

`LOCAL_RAW_ARTIFACTS.jsonl` records existing local archives, profiler databases, model/tensor files and source packs without uploading those large binaries to Git. They remain in their original local locations. `local-staging-selection.json` documents the local artifact selection. The preexisting961 tracked response-file deletions are intentionally left unstaged; their working-tree state and archived originals are preserved.

## Checkpoint validation

See `checkpoint-validation.json` and its logs. Python reference checks pass4/4. CPU workspace tests have one failed architecture-boundary target; all other targets pass. The first failure loads a forward source and test that are byte-identical to baseline main `3d007e8a`; the hashes are recorded in `preexisting-cpu-contract-failure.json`. No test or tolerance was relaxed. Main CUDA compilation/serving equivalence is not established by these CPU checks.

## Work boundary and future goal

The user requested proceeding only through Batch12 and then deciding the broader goal. No Batch13 or further GPU stability/Qwen experiment is started by this checkpoint. Batch12 serving is incomplete due to the fixed host admission policy, not accepted as successful. A later resumption must preserve both failed attempts and remeasure the complete matrix under the original conditions.

The future goal in [FUTURE_GOAL.md](FUTURE_GOAL.md) is a proposal, not authorization to resume beyond Batch12.
