#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-algo-diagsrc
E="env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-algo-diagtarget RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64"
$E cargo test --locked --release -p riley-cuda --features cuda --test native_bf16_paged_split_gqa_gpu -- --ignored --test-threads=1 ragged > $R/qwen-packed-native-gpu-test.log 2>&1
echo "exit $?" >> $R/qwen-packed-native-gpu-test.log
$E cargo build --locked --release -p riley-server --features server,hf-exact-probe > $R/qwen-packed-diag-build.log 2>&1
echo "exit $?" >> $R/qwen-packed-diag-build.log
cd $R
if grep -q "^exit 0" qwen-packed-diag-build.log && grep -q "^exit 0" qwen-packed-native-gpu-test.log; then
python3 qwen_matrix_steady_producer.py 1 packed > qwen-producer-packed.log 2>&1
fi
touch qwen-packed-diag.finished
