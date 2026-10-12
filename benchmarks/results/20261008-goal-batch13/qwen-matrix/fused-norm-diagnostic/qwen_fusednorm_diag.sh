#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-algo-diagsrc
env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-algo-diagtarget RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64 cargo build --locked --release -p riley-server --features server,hf-exact-probe > $R/qwen-fusednorm-diag-build.log 2>&1
echo "exit $?" >> $R/qwen-fusednorm-diag-build.log
cd $R
python3 qwen_matrix_steady_flags.py 1 fusednorm --residual-rmsnorm=fused > qwen-steady-flags-fusednorm.log 2>&1
touch qwen-fusednorm-diag.finished
