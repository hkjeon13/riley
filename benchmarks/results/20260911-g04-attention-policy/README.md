# Bounded attention numerical prototype

This is an isolated correctness experiment, not a promoted Riley implementation or performance result. `verify_solution.py` checks the final receipts and explicitly reports `production_riley_candidate_integrated: false`.

## Verified

- `mma-comparison-final.json`: custom CUDA attention is BF16-exact on all 960 captured actual serving calls.
- `graph-replay.json`: one captured attention graph, 930 dynamic-length decode replays, exact outputs and zero replay allocation deltas; invalid lengths 0/161 are rejected and a subsequent valid replay recovers.
- `random-comparison.json`: ten supported random shapes match; two unsupported cached multirow prefill shapes are rejected.
- `independent-mma-profile-final.json`: an independent full 30-layer model calculation using Torch-compiled model segments and the custom CUDA attention generates all 32 reference tokens exactly. It does not inject observed contexts. It is not the native Rust-owned model graph.

Supported prototype scope: SM89, BF16, head dimension 64, nine query heads, three KV heads, sequence length at most 160, single-row decode or complete uncached prefill. `mma_attention.cu` MODE=6 is compiled with `--use_fast_math`. Kernels and shared libraries used for GPU verification reside on the authorized remote host; local receipts include source, logs and JSON.

The attention policy uses BF16 MMA with FP32 score accumulation, reverse 128-token tiles, BF16 exponential weights, and delayed per-thread denominator reduction. Upstream implementation reference: [FlashAttention softmax](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/csrc/flash_attn/src/softmax.h). Exactness claims above come from the recorded local comparisons, not from similarity to upstream code.

Earlier ablation files intentionally retain failures. Use the explicitly named `*-final.json` receipts; passing attention alone does not qualify native full-model execution. Performance trials: zero.

The subsequent native integration is now completed for the explicit SmolLM2 P128 profile. See `../20260911-g04-vllm-profile/README.md`; the prototype's `production_riley_candidate_integrated: false` receipt remains unchanged because it describes this earlier standalone experiment.
