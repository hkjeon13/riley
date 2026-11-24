#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
parent="$evidence/w3-c32-step-observation-20261124.SnmQ1EfS"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c dispatch.sha256
test "$(cat "$parent/build.exit")" = 0
test "$(cat "$parent/campaign.exit")" = 0
printf '%s  %s\n' \
    f94b7283dd42f9b6c24643b49e7d9d51adea8ccc5f6ca6f8bd09de5093425280 "$parent/build-result.sha256" \
    fd8f5e5aef0911698ed7d8ff353229848226f644531eff269908112a97e45682 "$parent/diagnostic-sources.sha256" |
    sha256sum --strict -c
(cd "$parent" && sha256sum --strict -c build-result.sha256)
cp -a -- "$parent/src" src
(cd src && sha256sum --strict -c "$parent/diagnostic-sources.sha256" &&
    sha256sum --strict -c ../parent-tail-sources.sha256 &&
    git apply --check --whitespace=error ../prefill-tail.patch &&
    git apply --whitespace=error ../prefill-tail.patch &&
    sha256sum --strict -c ../prefill-tail-sources.sha256)
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
sha256sum riley-diagnostic dependencies.txt toolchain.txt prefill-tail-sources.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
