# SM80 mixed MLP piecewise Graph + merged decode Gate/Up v13

Private `native-prefill-sm80-v5` + `segments-v1`, Qwen3-4B BF16, SM80/108SM, CUDA12.4/cuBLASLt120405. Strict native/HF literal equality criteria remain in force. No public numeric-mode relaxation.

Mixed requests use a stable decode prefix with original output slots and KV boundaries. Startup prepares layer/count MLP pieces; serving cache misses remain eager without capturing. Pure decode R2..32 keeps the whole-body Graph. Graph keys and leases retain stream and operand allocation identities. Cached graphs close before their plans/workspaces.

Merged raw Gate/Up uses N19456/K2560, pinned Lt recipe `31/15/17/0/0/0/0`, only R1 and R17..32. R2..16 keeps the measured fused kernel. The original BF16 Gate, Up, SiLU and product boundaries remain. Separate prefill weights remain resident, so merged copies add about3.59GB. Additional workspace is cold, bounded and registered during replay.

Code: `crates/riley-cuda/src/{gemm,owned_mlp}.rs`, `crates/riley-runtime/src/llama/{batch,batch_executor,forward}.rs`, `crates/riley-runtime/src/llama/executor/{owner,dispatch}.rs`, `kernels/src/cublas_gemm_probe.cu`.

[Implementation and final measurement report](../../../docs/qwen-piecewise-graph-merged-gateup-v13-20261020.md). [Evidence](../../evidence/20261020-piecewise-v13/).

Failed candidates and slower WMMA/pipeline variants remain on EFS. They are excluded from final measurements. No foreign GPU process is stopped; original physical GPU6/7 holder contexts stay alive.
