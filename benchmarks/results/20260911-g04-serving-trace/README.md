# Actual SmolLM2 serving attention trace

Correctness diagnosis only; no performance trials.

`capture.py` runs three identical greedy 128-input/32-output requests against vLLM 0.27.1: before observer installation, observed, and after observer removal. `serving_worker.py` wraps the actual loaded FlashAttentionImpl forward method after initialization. Inductor compilation remains enabled; CUDA graphs are disabled so every invocation is observable. All three token sequences match the separately recorded default CUDA-graph reference (`invariance.json`). This establishes output invariance for this workload, not general equivalence of all execution modes.

`serving-tensors.json` records 960 calls (30 layers × one prefill plus 31 decode calls). BF16 Q/K/V and context files contain the actual backend inputs and outputs. `aot/` contains the generated Inductor source used by this trace. In particular, its fused residual/norm path preserves the post-attention residual in FP32 through the MLP before the next input norm.

The additive `crates/riley-cuda/tests/serving_attention_replay_gpu.rs` test replays the first 270 cases through the existing Riley primitive with scheduler-style paged KV. `attention-comparison.json` is that baseline, not a pass receipt for the new prototype.

See the sibling `20260911-g04-attention-policy` directory for independent attention and generation experiments, and `20260911-g04-native-profile` for the still-incomplete native integration.
