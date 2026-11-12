#!/bin/bash
# fresh.sh LABEL: unpack src.tgz into a clean src/, build riley-checkpoint and riley-server -> bin/riley-LABEL
W=/mnt/efs/nlp/psyche/riley-gap-20261102
export RUSTUP_HOME=$W/tools/rustup CARGO_HOME=$W/tools/cargo PATH=$W/tools/cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 RILEY_CUDA_ARCHITECTURES=80 CARGO_TARGET_DIR=$W/target
set -o pipefail
rm -rf $W/src && mkdir -p $W/src $W/bin && tar xzf $W/src.tgz -C $W/src || exit 1
cd $W/src
cargo build --locked --release --manifest-path tools/riley-checkpoint/Cargo.toml --features hub 2>&1 | tail -3 || exit 1
cargo build --locked --release -p riley-server --features server,hf-exact-probe 2>&1 | grep -E "^error|^warning: unused|Finished" | head -30 || exit 1
cp $W/target/release/riley $W/bin/riley-$1 && sha256sum $W/bin/riley-$1 && echo BUILD_OK $1
