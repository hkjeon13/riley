#!/bin/sh
R=/data/riley-serving-261007
cd $R/qwen-algo-diagsrc
env -i HOME=/home/psyche LANG=C.UTF-8 LC_ALL=C.UTF-8 PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/usr/bin:/bin CUDA_HOME=/data/cuda-12.8.1 CUDAToolkit_ROOT=/data/cuda-12.8.1 CUDACXX=/data/cuda-12.8.1/bin/nvcc CMAKE=/data/cmake-3.31.12/bin/cmake CARGO_BUILD_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_TARGET_DIR=$R/qwen-algo-diagtarget RUSTUP_TOOLCHAIN=1.85.0 LD_LIBRARY_PATH=/data/cuda-12.8.1/lib64 cargo build --locked --release -p riley-server --features server,hf-exact-probe > $R/qwen-budget512b-build.log 2>&1
echo "exit $?" >> $R/qwen-budget512b-build.log
cd $R
M="--projection-bias-backend=merged-qkv-strict-staged-v1 --residual-rmsnorm=fused"
python3 qwen_matrix_steady_diag_flags.py 1 budget512-buckets24 $M --batch-token-budget=512 --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,160,192,224,256,288,320,352,384,416,448,512 > qwen-steady-diag-budget512-buckets24.log 2>&1
python3 qwen_matrix_steady_diag_flags.py 1 budget384-buckets24 $M --batch-token-budget=384 --batch-shape-buckets=1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,256,288,320,352,384 > qwen-steady-diag-budget384-buckets24.log 2>&1
touch qwen-budget512b.finished
