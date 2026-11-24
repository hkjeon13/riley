#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > build.exit' EXIT
stage=$(pwd -P)
base=$(realpath ../long-tile)
root=$(realpath ../../..)
mkdir src
cp -R "$base/src/." src/
(cd src && git apply --check ../register-layer-trace.patch && git apply ../register-layer-trace.patch)
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="$stage/target"
(cd src && taskset -c 24-27 cargo build --offline --locked --release -p riley-server --features server,cuda) >build.log 2>&1
sha256sum target/release/riley > binary.sha256
