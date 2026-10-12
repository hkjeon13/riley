#!/bin/sh
# Row-wise attention with full-precision softmax weights (three BF16 pieces): rebuild, gate 54, native tests, HF token comparison 09, margins, C8..C32 steady.
cd /data/riley-serving-261007
until [ -f qwen-rowwise2.finished ]; do sleep 10; done
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
sh ./qwen_rebuild_and_gate.sh 54 || { touch qwen-rowwise3.failed; exit 1; }
cp qwen-http-exact-target-attempt01/release/riley qwen-riley-rowwise3-candidate
sha256sum qwen-riley-rowwise3-candidate > qwen-riley-rowwise3-candidate.sha256
( cd qwen-http-exact-source-attempt01 && for t in rowwise_attention_gpu prompt_mma_attention_gpu; do $E cargo test --locked --release -p riley-cuda --features cuda --test $t -- --ignored --nocapture --test-threads=1; done; $E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged ) > qwen-rowwise3-native-test.log 2>&1
python3 qwen_hf_token_ref_rowwise3.py 9 > qwen-hf-token-ref-09.log 2>&1
touch qwen-rowwise3-hf.finished
for n in 7 8 9; do CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 vllm0271-venv/bin/python qwen_hf_margin_n.py $n > qwen-hf-margin-attempt0$n.log 2>&1; done
touch qwen-rowwise3-margins.finished
export QRILEY=/data/riley-serving-261007/qwen-riley-rowwise3-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 83 rowwise3-c8-32 $COMMON --decode-attention-producer=rowwise-tensor-core-v1 > qwen-rowwise3-83.log 2>&1
touch qwen-rowwise3.finished
