#!/bin/bash
set -eu
ROOT=/data/riley-serving-260913-recovery
export PATH=/home/psyche/.cargo/bin:$ROOT/toolchain130/nvidia/cu13/bin:/usr/bin:/bin
export CUDA_HOME=$ROOT/toolchain130/nvidia/cu13
export CUDAToolkit_ROOT=$CUDA_HOME
export CMAKE=/data/cmake-3.31.12/bin/cmake
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=$ROOT/target
export LD_LIBRARY_PATH=$CUDA_HOME/lib:/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib
cd "$ROOT/source"
set +e
cargo build --release -p riley-server --features cuda,server > "$ROOT/logs/server-build.log" 2>&1
result=$?
echo "$result" > "$ROOT/logs/server-build.exit"
tail -20 "$ROOT/logs/server-build.log"
exit "$result"
