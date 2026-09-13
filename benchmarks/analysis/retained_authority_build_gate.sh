#!/usr/bin/env bash
set -euo pipefail
root=${1:?prepared root};evidence=${2:?existing model gate output}
export PATH="$root/toolchain130/nvidia/cu13/bin:/home/psyche/.cargo/bin:/usr/bin:/bin"
export CUDA_HOME="$root/toolchain130/nvidia/cu13" CUDAToolkit_ROOT="$root/toolchain130/nvidia/cu13"
export CMAKE=/data/cmake-3.31.12/bin/cmake CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR="$root/target" RILEY_FA3_SOURCE="$root/deps/flash-attention"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib"
cd "$root/source"
timeout 900 cargo test -p riley-runtime --features cuda --lib variable_session -- --nocapture > "$evidence/staged-session-tests.log" 2>&1
timeout 900 cargo build -p riley-server --release --features cuda,server,bench --bin riley > "$evidence/release-build.log" 2>&1
sha256sum "$root/target/release/riley" > "$evidence/release-sha256.txt"
sha256sum crates/riley-runtime/src/llama/multi_descriptor/variable_wire.rs crates/riley-runtime/src/llama/multi_descriptor/future_token.rs crates/riley-runtime/src/llama/variable_session.rs > "$evidence/changed-source-sha256.txt"
