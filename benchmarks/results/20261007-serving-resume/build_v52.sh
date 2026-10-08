#!/bin/bash
set -eu
TASK=/data/riley-serving-261007
TOOL=/data/riley-serving-260913-recovery/toolchain130/nvidia/cu13
export PATH=/home/psyche/.cargo/bin:$TOOL/bin:/usr/bin:/bin
export CUDA_HOME=$TOOL CUDAToolkit_ROOT=$TOOL
export CMAKE=/data/cmake-3.31.12/bin/cmake
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=$TASK/v52-target
export LD_LIBRARY_PATH=$TOOL/lib
cd "$TASK/v52-source"
set +e
cargo build --locked --release -p riley-server --features cuda,server > "$TASK/v52-build.log" 2>&1
result=$?
printf '%s\n' "$result" > "$TASK/v52-build.exit"
if [ "$result" = 0 ]; then
  sha256sum "$CARGO_TARGET_DIR/release/riley" > "$TASK/v52-binary.sha256"
fi
tail -25 "$TASK/v52-build.log"
exit "$result"
