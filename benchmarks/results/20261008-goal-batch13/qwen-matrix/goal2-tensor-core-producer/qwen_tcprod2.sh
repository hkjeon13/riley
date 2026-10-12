#!/bin/sh
# Tensor-core page producer with sequentially accumulated scores: rebuild, GPU test, HF token comparison 12 + margins, steady C8..C32 and C1.
cd /data/riley-serving-261007
until [ -f qwen-tcprod1.finished ]; do sleep 10; done
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
( cd qwen-http-exact-source-attempt01 && $E cargo build --locked --release -p riley-server --features server,hf-exact-probe ) > qwen-rebuild-tcprod2.log 2>&1 || { touch qwen-tcprod2.failed; exit 1; }
cp qwen-http-exact-target-attempt01/release/riley qwen-riley-tcprod2-candidate
sha256sum qwen-riley-tcprod2-candidate > qwen-riley-tcprod2-candidate.sha256
( cd qwen-http-exact-source-attempt01 && $E cargo test --locked --release -p riley-cuda --features cuda --test tensor_core_producer_gpu -- --ignored --nocapture --test-threads=1 ) > qwen-tcprod2-native-test.log 2>&1
python3 qwen_hf_token_ref_tcprod2.py 12 > qwen-hf-token-ref-12.log 2>&1
touch qwen-tcprod2-hf.finished
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 vllm0271-venv/bin/python qwen_hf_margin_n.py 12 > qwen-hf-margin-attempt12.log 2>&1
export QRILEY=/data/riley-serving-261007/qwen-riley-tcprod2-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 99 tcprod2-c8-32 $COMMON --decode-attention-producer=tensor-core-v1 > qwen-tcprod2-99.log 2>&1
QCONC=1 python3 qwen_matrix_steady_main_env2.py 100 tcprod2-c1 $COMMON --decode-attention-producer=tensor-core-v1 > qwen-tcprod2-100.log 2>&1
touch qwen-tcprod2.finished
