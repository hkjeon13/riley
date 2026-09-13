#!/bin/sh
set -eu
export PATH=/home/psyche/.cargo/bin:/tmp/riley-opt-260912/flashinfer-compatibility/toolchain130/nvidia/cu13/bin:/usr/bin:/bin
export CUDA_HOME=/tmp/riley-opt-260912/flashinfer-compatibility/toolchain130/nvidia/cu13
export CUDAToolkit_ROOT="$CUDA_HOME"
export CMAKE=/data/cmake-3.31.12/bin/cmake CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR=/tmp/riley-opt-260912/persistent-post-attention-cargo-target-v1
export RILEY_FLASHINFER_DATA=/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/flashinfer/data
export LD_LIBRARY_PATH=/tmp/riley-opt-260912/driver580173-runtime-20260901/extracted/usr/lib/x86_64-linux-gnu:/data/riley-g04-cuda13/lib
export RILEY_REAL_CHECKPOINT=/data/riley-benchmark/20260827T051948Z-d7ad713a/model
export RILEY_FFN_NATURAL_SCREEN_DIR=/tmp/riley-opt-260912/persistent-post-attention-natural-v1
cd /tmp/riley-opt-260912/persistent-post-attention-model-source-v1
cargo test -p riley-scheduler --release --features cuda --test ffn_pipeline_free_generation_gpu -- --ignored --nocapture > /tmp/riley-opt-260912/persistent-post-attention-model-v1/free-generation.log 2>&1
cargo test -p riley-scheduler --release --features cuda --test ffn_pipeline_natural_logits_gpu -- --ignored --nocapture > /tmp/riley-opt-260912/persistent-post-attention-model-v1/natural.log 2>&1
cargo build -p riley-server --release --features cuda,server --bin riley > /tmp/riley-opt-260912/persistent-post-attention-model-v1/server-build.log 2>&1
