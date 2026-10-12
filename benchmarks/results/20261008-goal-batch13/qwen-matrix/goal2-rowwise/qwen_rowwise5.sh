#!/bin/sh
# Row-wise attention with early cache-line touches (same arithmetic as rowwise3/4): rebuild, native tests, C8..C32 steady, C1 steady.
cd /data/riley-serving-261007
until [ -f qwen-rowwise4.finished ]; do sleep 10; done
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
( cd qwen-http-exact-source-attempt01 && $E cargo build --locked --release -p riley-server --features server,hf-exact-probe ) > qwen-rebuild-rowwise5.log 2>&1 || { touch qwen-rowwise5.failed; exit 1; }
cp qwen-http-exact-target-attempt01/release/riley qwen-riley-rowwise5-candidate
sha256sum qwen-riley-rowwise5-candidate > qwen-riley-rowwise5-candidate.sha256
( cd qwen-http-exact-source-attempt01 && $E cargo test --locked --release -p riley-cuda --features cuda --test rowwise_attention_gpu -- --ignored --nocapture --test-threads=1 ) > qwen-rowwise5-native-test.log 2>&1
export QRILEY=/data/riley-serving-261007/qwen-riley-rowwise5-candidate
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 93 rowwise5-c8-32 $COMMON --decode-attention-producer=rowwise-tensor-core-v1 > qwen-rowwise5-93.log 2>&1
touch qwen-rowwise5-c8.finished
QCONC=1 python3 qwen_matrix_steady_main_env2.py 94 rowwise5-c1 $COMMON --decode-attention-producer=rowwise-tensor-core-v1 > qwen-rowwise5-94.log 2>&1
touch qwen-rowwise5.finished
