#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > build.exit' EXIT
root=$(realpath ../../..)
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="$PWD/target"
(cd src && git apply --check ../http-worker-queue.patch && git apply ../http-worker-queue.patch && python3 ../apply_worker_trace.py && taskset -c 24-27 cargo build --locked --offline --release -p riley-server --features server,cuda --bin riley)
cp target/release/riley trace-server
sha256sum trace-server > trace-server.sha256
