#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-algo-diagsrc
env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-algo-diagtarget RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64 cargo build --locked --release -p riley-server --features server,hf-exact-probe > $R/qwen-phase-diag2-build.log 2>&1
echo "exit $?" >> $R/qwen-phase-diag2-build.log
cd $R
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256 --decode-attention-merge=global-max-v1"
python3 qwen_matrix_steady_diag_flags.py 1 phase-timing2 $M > qwen-steady-diag-phase-timing2.log 2>&1
touch qwen-phase-diag2.finished
