#!/bin/sh
R=/data/riley-serving-261007
while [ ! -f $R/qwen-fusednorm-diag.finished ]; do sleep 20; done
cd $R/qwen-http-exact-source-attempt01
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-http-exact-target-attempt01 RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
$E cargo test --locked --release -p riley-cuda --features cuda --test row_bias_gpu -- --ignored --test-threads=1 > $R/qwen-mergedqkv-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-mergedqkv-native-gpu-test.log
cd $R
if sh ./qwen_rebuild_and_gate.sh 17; then
python3 qwen_matrix_steady_main_flags.py 1 strict > qwen-steady-main-strict.log 2>&1
python3 qwen_matrix_steady_main_flags.py 1 mergedqkv --projection-bias-backend=merged-qkv-strict-staged-v1 > qwen-steady-main-mergedqkv.log 2>&1
fi
touch qwen-mergedqkv-pipeline.finished
