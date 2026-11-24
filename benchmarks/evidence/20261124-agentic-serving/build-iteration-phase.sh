#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > phase-build.exit' EXIT
root=$(realpath ../../..)
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="$PWD/phase-target"
cp -R src phase-src
(cd phase-src && git apply --check ../iteration-phase-metrics.patch && git apply ../iteration-phase-metrics.patch && taskset -c 24-27 cargo build --locked --offline --release -p riley-server --features server,cuda --bin riley)
cp phase-target/release/riley phase-server
sha256sum phase-server > phase-server.sha256
