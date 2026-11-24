#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
parent="$evidence/w3-c32-prefill-tail-20261124.C7DHxpyN"
stock="$evidence/w4-branch-anchors-20261124.fpN03ecM"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c dispatch.sha256
test "$(cat "$parent/build.exit")" = 0
test "$(cat "$parent/campaign.exit")" = 0
test "$(cat "$stock/campaign.exit")" = 0
printf '%s  %s\n' \
    5776274be2c9359490596ad7ab294fb8e53c749a667377b855c907a124e8ba2f "$stock/source.tar" \
    a4084f0ecae2ff820600df09061a75c7d653c49b5baf89634b63c1c1040f6e58 "$stock/cpu-evidence/report.json" |
    sha256sum --strict -c
(cd "$parent" && sha256sum --strict -c build-result.sha256)
mkdir src
tar -xf "$stock/source.tar" -C src --no-same-owner
(cd src && sha256sum --strict -c ../scheduler-before.sha256 &&
    git apply --check --whitespace=error ../candidate.patch &&
    git apply --whitespace=error ../candidate.patch &&
    sha256sum --strict -c ../scheduler-after.sha256)
mkdir -p target/release
cp -a -- "$parent/target/release/"{deps,.fingerprint} target/release/
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$stage/target" CUDA_VISIBLE_DEVICES= RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
(cd src && taskset -c 24,25,26,27 cargo build --offline --locked --release -p riley-server --features server,cuda) >compiler.log 2>&1
install -m 0755 target/release/riley riley-diagnostic
ldd riley-diagnostic >dependencies.txt
! grep -Fq 'not found' dependencies.txt
! cmp -s riley-diagnostic "$parent/riley-diagnostic"
sha256sum riley-diagnostic dependencies.txt toolchain.txt scheduler-after.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
