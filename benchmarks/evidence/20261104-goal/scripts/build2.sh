#!/bin/bash
# build2.sh LABEL PATCH: apply PATCH to src (reverting the previous one), build bin/riley-LABEL at low priority
W=/mnt/efs/nlp/psyche/riley-gap-20261102
export RUSTUP_HOME=$W/tools/rustup CARGO_HOME=$W/tools/cargo PATH=$W/tools/cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.4 RILEY_CUDA_ARCHITECTURES=80 CARGO_TARGET_DIR=$W/target CARGO_BUILD_JOBS=16
set -o pipefail
cd $W/src
if [ -f $W/applied.patch ]; then patch -R -p1 -s < $W/applied.patch || exit 1; fi
patch -p1 -s < $2 || exit 1; cp $2 $W/applied.patch
nice -n 19 cargo build --locked --release -p riley-server --features server,hf-exact-probe 2>&1 | grep -E "^error|^ *--> |Finished" | head -30 || exit 1
cp $W/target/release/riley $W/bin/riley-$1 && sha256sum $W/bin/riley-$1 && echo BUILD_OK $1
