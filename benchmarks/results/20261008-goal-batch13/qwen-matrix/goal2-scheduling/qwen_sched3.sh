#!/bin/sh
# Scheduling experiments on the row-wise attention candidate (binary qwen-riley-rowwise3-candidate), C8..C32, Riley only.
# Base = attempt 83 (budget 256, no plateau policy). Then a row-cap-512 diagnostic build for larger step budgets.
cd /data/riley-serving-261007
until [ -f qwen-rowwise3.finished ]; do sleep 20; done
export QRILEY=/data/riley-serving-261007/qwen-riley-rowwise3-candidate
B128=1,2,4,8,16,24,32,48,64,80,96,112,128
B192=$B128,144,160,176,192
B256=$B192,208,224,240,256
COMMON="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --decode-attention-merge=global-max-v1 --decode-step-graph=capture-v1 --mlp-activation=fused-silu-multiply-v1 --query-key-rope=fused-split-v1 --gemm-algorithm-selection=bit-equivalent-table-v1 --mlp-residual-norm=fused-next-norm-v1 --lm-head-rows=output-rows-v1 --kv-cache-write=fused-split-v1 --decode-attention-producer=rowwise-tensor-core-v1"
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 84 rw3-plateau128-c8-32 $COMMON --batch-shape-buckets=$B256 --prefill-plateau-rows=128 > qwen-sched3-84.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 85 rw3-plateau128x192-c8-32 $COMMON --batch-shape-buckets=$B256 --prefill-plateau-rows=128,192 > qwen-sched3-85.log 2>&1
touch qwen-sched3-a.finished
# Diagnostic build: native D128 row cap 256 -> 512 (two constants), everything else identical.
rm -rf qwen-rowwise-cap512-diagsrc && cp -a qwen-http-exact-source-attempt01 qwen-rowwise-cap512-diagsrc
sed -i "s/^const NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_MAX_BATCH_TOKENS: usize = 256;/const NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_MAX_BATCH_TOKENS: usize = 512;/" qwen-rowwise-cap512-diagsrc/crates/riley-server/src/main.rs
sed -i "s/^const NATIVE_D128_TWO_STAGE_MAX_DENSE_ROWS: usize = 256;/const NATIVE_D128_TWO_STAGE_MAX_DENSE_ROWS: usize = 512;/" qwen-rowwise-cap512-diagsrc/crates/riley-runtime/src/llama/executor/owner.rs
( cd qwen-rowwise-cap512-diagsrc && grep -n "TWO_STAGE_MAX_BATCH_TOKENS: usize\|TWO_STAGE_MAX_DENSE_ROWS: usize" crates/riley-server/src/main.rs crates/riley-runtime/src/llama/executor/owner.rs && env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=/data/riley-serving-261007/qwen-rowwise-cap512-diag-target RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64 cargo build --locked --release -p riley-server --features server,hf-exact-probe ) > qwen-rowwise-cap512-diag-build.log 2>&1 || { touch qwen-sched3-b.failed; exit 1; }
export QRILEY=/data/riley-serving-261007/qwen-rowwise-cap512-diag-target/release/riley
B512=$B192,208,224,256,320,384,448,512
B384=$B256,320,384
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 86 rw3-cap512-budget512-c8-32 $COMMON --batch-shape-buckets=$B512 --batch-token-budget=512 --prefill-chunk-tokens=256 > qwen-sched3-86.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 87 rw3-cap512-budget384-c8-32 $COMMON --batch-shape-buckets=$B384 --batch-token-budget=384 --prefill-chunk-tokens=256 > qwen-sched3-87.log 2>&1
QCONC=8,16,32 python3 qwen_matrix_steady_main_env2.py 88 rw3-cap512-budget256-c8-32 $COMMON --batch-shape-buckets=$B256 > qwen-sched3-88.log 2>&1
touch qwen-sched3.finished
