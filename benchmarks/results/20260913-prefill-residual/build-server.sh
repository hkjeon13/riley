#!/bin/sh
set -eu
export PATH=/home/psyche/.cargo/bin:/tmp/riley-opt-260912/flashinfer-compatibility/toolchain130/nvidia/cu13/bin:/usr/bin:/bin
export CUDA_HOME=/tmp/riley-opt-260912/flashinfer-compatibility/toolchain130/nvidia/cu13
export CUDAToolkit_ROOT="$CUDA_HOME"
export CMAKE=/data/cmake-3.31.12/bin/cmake CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/tmp/riley-opt-260912/prefill-residual-cargo-target-v1
export RILEY_FLASHINFER_DATA=/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/flashinfer/data
export LD_LIBRARY_PATH=/tmp/riley-opt-260912/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu:/data/riley-g04-cuda13/lib
export RILEY_REAL_CHECKPOINT=/data/riley-benchmark/20260827T051948Z-d7ad713a/model
export RILEY_PREFILL_NATURAL_SCREEN_DIR=/tmp/riley-opt-260912/prefill-residual-natural-v1
cd /tmp/riley-opt-260912/prefill-residual-model-source-v1
cargo build -p riley-server --release --features cuda,server --bin riley > /tmp/riley-opt-260912/prefill-residual-model-v1/server-build.log 2>&1
