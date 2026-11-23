#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w3-c32-http-workers-build-20261122.YHWcWWNk"
package="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
private="$evidence/w3-private-warm-build-20261121.xrfPErXB"
prefill="$evidence/w3-c32-prefill-build-20261123.nnAtltun"
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-inputs.sha256
(cd "$previous" && printf '7b9b58d114fddafd5692ec71a9b4b445f47d20a351cf9eff6bdce8c30c8cb7b6  external-inputs.sha256\nd31203649c3bb200b33910eec7fa58d699badf2fd124a3ae80f816151b1536e0  patched-source.sha256\n' | sha256sum --strict -c)
(cd "$evidence" && sha256sum --strict -c "$previous/external-inputs.sha256")
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
(
    cd src
    for patch in "$private/private-warmup.patch.gz" "$previous/http-queue.patch.gz"; do
        gzip -dc "$patch" | git apply --check -
        gzip -dc "$patch" | git apply -
    done
    sha256sum --strict -c "$previous/patched-source.sha256"
    gzip -dc "$stage/stage-trace.patch.gz" | git apply --check -
    gzip -dc "$stage/stage-trace.patch.gz" | git apply -
    python3 -B "$stage/prepare.py" "$PWD"
    printf '3ecdc82772588fe868516b2030c449c9418703b12f36cb0c30d4d4accee7f177  %s\n' "$prefill/patched-source.sha256" | sha256sum --strict -c
    sha256sum --strict -c "$prefill/patched-source.sha256"
    python3 -B "$stage/prepare-peak.py" "$PWD"
    sha256sum crates/riley-server/src/{lib,main,service,engine,w3_trace}.rs crates/riley-scheduler/src/scheduler.rs >"$stage/patched-source.sha256"
)
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
sha256sum target/release/riley toolchain.txt dependencies.txt patched-source.sha256 >build-result.sha256
