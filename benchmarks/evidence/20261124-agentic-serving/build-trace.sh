#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test -s source.tar.gz
mkdir trace-src
tar -xzf source.tar.gz -C trace-src
(cd trace-src && git apply --check ../workflow-trace.patch && git apply ../workflow-trace.patch)
root=/mnt/efs/nlp/psyche/riley-gap-20261102
export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 CUDA_VISIBLE_DEVICES=
export CARGO_TARGET_DIR="$PWD/trace-target"
sha256sum source.tar.gz workflow-trace.patch >trace-inputs.sha256
(cd trace-src && cargo build --offline --locked --release -p riley-server --features server,cuda) >trace-build.log 2>&1
sha256sum trace-target/release/riley >trace-binary.sha256
