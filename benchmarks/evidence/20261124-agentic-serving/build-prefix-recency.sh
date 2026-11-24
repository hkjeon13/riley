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
cp -R ../long-tile/src src
mkdir -p src/crates/riley-scheduler/examples
cp prefix_recency_bench.rs src/crates/riley-scheduler/examples/
for arm in control candidate; do
    if [[ $arm == candidate ]]; then (cd src && git apply --check ../prefix-recency.patch && git apply ../prefix-recency.patch); fi
    (cd src && taskset -c 24-27 cargo build --release --offline --locked -p riley-server --features server,cuda --bin riley)
    cp target/release/riley "server-$arm"
    (cd src && taskset -c 24-27 cargo build --release --offline --locked -p riley-scheduler --example prefix_recency_bench)
    cp target/release/examples/prefix_recency_bench "bench-$arm"
done
sha256sum server-{control,candidate} bench-{control,candidate} > artifacts.sha256
for id in 0-control 1-candidate 2-candidate 3-control; do
    taskset -c 62 "./bench-${id#*-}" > "cpu-$id.json"
done
