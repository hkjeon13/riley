#!/bin/sh
# Row-wise attention with eager-attention rounding (normalized BF16 probabilities): rebuild, gate 53, native tests, HF token comparison 08, C8..C32 steady.
cd /data/riley-serving-261007
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
sh ./qwen_rebuild_and_gate.sh 53 || { touch qwen-rowwise2.failed; exit 1; }
cp qwen-http-exact-target-attempt01/release/riley qwen-riley-rowwise2-candidate
sha256sum qwen-riley-rowwise2-candidate > qwen-riley-rowwise2-candidate.sha256
( cd qwen-http-exact-source-attempt01 && for t in rowwise_attention_gpu prompt_mma_attention_gpu; do $E cargo test --locked --release -p riley-cuda --features cuda --test $t -- --ignored --nocapture --test-threads=1; done; $E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged ) > qwen-rowwise2-native-test.log 2>&1
python3 qwen_hf_token_ref_rowwise2.py 8 > qwen-hf-token-ref-08.log 2>&1
touch qwen-rowwise2-hf.finished
export QRILEY=/data/riley-serving-261007/qwen-riley-rowwise2-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 82 rowwise2-c8-32 $COMMON --decode-attention-producer=rowwise-tensor-core-v1 > qwen-rowwise2-82.log 2>&1
touch qwen-rowwise2.finished
