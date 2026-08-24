#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

# The native `cuda` feature is intentionally tested by ci/cuda/Dockerfile.
# Every other server feature combination must stay CUDA-toolkit independent.
cargo check --locked --workspace --all-targets --no-default-features

for features in \
    server \
    bench \
    experimental \
    server,bench \
    server,experimental \
    bench,experimental \
    server,bench,experimental
do
    echo "checking rustinfer-server features: $features"
    cargo check \
        --locked \
        --package rustinfer-server \
        --all-targets \
        --no-default-features \
        --features "$features"
done
