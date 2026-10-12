#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-http-exact-source-attempt01
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
$E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged > $R/qwen-globalmerge-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-globalmerge-native-gpu-test.log
cd $R
if sh ./qwen_rebuild_and_gate.sh 20; then
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256"
python3 qwen_matrix_steady_main_flags.py 1 adopted-rebase $M > qwen-steady-main-adopted-rebase.log 2>&1
python3 qwen_matrix_steady_main_flags.py 1 globalmerge $M --decode-attention-merge=global-max-v1 > qwen-steady-main-globalmerge.log 2>&1
python3 qwen_hf_token_ref_gm.py 3 > qwen-hf-token-ref-3.log 2>&1
fi
touch qwen-gm-pipeline.finished
