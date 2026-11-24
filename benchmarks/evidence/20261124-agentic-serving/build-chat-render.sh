#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > build.exit' EXIT
base=$(realpath ../long-tile)
root=$(realpath ../../..)
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="$PWD/target"
cp -R "$base/src" src
mkdir -p src/crates/riley-server/examples
cp qwen_prompt_profile.rs src/crates/riley-server/examples/
for arm in control candidate; do
    if [[ $arm == candidate ]]; then (cd src && git apply --check ../chat-owned-render.patch && git apply ../chat-owned-render.patch); fi
    sha256sum src/crates/riley-server/src/chat.rs > "$arm.source.sha256"
    (cd src && taskset -c 24-27 cargo build --release --offline --locked -p riley-server --features server,cuda --bin riley --example qwen_prompt_profile) > "$arm.build.log" 2>&1
    cp target/release/riley "server-$arm"
    cp target/release/examples/qwen_prompt_profile "probe-$arm"
done
sha256sum server-* probe-* > binary.sha256
