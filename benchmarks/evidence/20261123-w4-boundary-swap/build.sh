#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w4-long-code-build-20261122.LO5rPFy0"
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-inputs.sha256
(cd "$previous"
    printf '0a20e738e713effdb8c955e4cb03e11200ad7fa0fe2c43e5cbcee1ec8660b8a5  patched-source.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c patched-source.sha256)
cp -r "$previous/src" src
sha256sum --strict -c "$previous/patched-source.sha256"
python3 -B prepare.py "$stage"
sha256sum src/crates/riley-runtime/src/llama/executor/dispatch.rs >patched-source.sha256
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >dependencies.txt
if grep -F 'not found' dependencies.txt; then exit 1; fi
sha256sum target/release/riley patched-source.sha256 toolchain.txt dependencies.txt >build-result.sha256
