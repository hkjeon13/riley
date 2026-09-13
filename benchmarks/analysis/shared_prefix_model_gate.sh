#!/usr/bin/env bash
set -euo pipefail
root=${1:?prepared remote root}
evidence=${2:?new evidence directory}
mkdir "$evidence"
export PATH="$root/toolchain130/nvidia/cu13/bin:/home/psyche/.cargo/bin:/usr/bin:/bin"
export CUDA_HOME="$root/toolchain130/nvidia/cu13" CUDAToolkit_ROOT="$root/toolchain130/nvidia/cu13"
export CMAKE=/data/cmake-3.31.12/bin/cmake CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR="$root/target" RILEY_FA3_SOURCE="$root/deps/flash-attention"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib"
export RILEY_REAL_CHECKPOINT=/data/riley-benchmark/20260827T051948Z-d7ad713a/model
cd "$root/source"
timeout 900 cargo test -p riley-runtime --features cuda --lib staged_window_tests --quiet > "$evidence/session-tests.log" 2>&1
timeout 900 cargo test -p riley-runtime --features cuda --test shared_prefix_model_gpu --no-run > "$evidence/build.log" 2>&1
timeout 180 cargo test -p riley-runtime --features cuda --test shared_prefix_model_gpu -- --ignored --nocapture --test-threads=1 > "$evidence/model-tests.log" 2>&1
timeout 900 cargo check -p riley-server --features cuda,server > "$evidence/server-feature-check.log" 2>&1
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$evidence/compute-after.csv"
sha256sum crates/riley-runtime/tests/shared_prefix_model_gpu.rs kernels/src/graph_resources.cu kernels/src/prefill_shape_packet.hpp > "$evidence/source-sha256.txt"
cat "$evidence/session-tests.log" "$evidence/model-tests.log"
