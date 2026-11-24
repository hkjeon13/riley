#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w4-pointer-syntax-20261124.HgwYVqCf"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c dispatch.sha256
test "$(cat "$package/campaign.exit")" = 0
printf '%s  %s\n' \
    4398f3e068a79f043270bc0ec2df638719f43e8834e5c2d290c52b6d2101384b "$package/source.tar" \
    8436c243060357615f9ac5499edd98ba404a079e44d55a8df0f76bcf9ce0b6d8 "$package/cpu-evidence/report.json" |
    sha256sum --strict -c
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
scheduler=crates/riley-scheduler/src/scheduler.rs
(
    cd src
    printf '%s  %s\n' 5585da9cf2fe5ae0b8ea64c2a4903d4827c2d04b175ed8263d57935aa2fe9cbd "$scheduler" |
        sha256sum --strict -c
    git apply --check --whitespace=error ../prefill-admission-fifo.patch
    git apply --whitespace=error ../prefill-admission-fifo.patch
    printf '%s  %s\n' 681333b0dccb8b48891d263bfab4e8c10be2cc1cda029d994733ff448cc1593a "$scheduler" |
        sha256sum --strict -c
)
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$stage/target" CUDA_VISIBLE_DEVICES= RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    taskset -c 24-27 cargo test --offline --locked -p riley-scheduler --no-default-features
) >cpu-tests.log 2>&1
(
    cd src
    taskset -c 24-27 cargo build --offline --locked --release -p riley-server --features server,cuda
) >compiler.log 2>&1
install -m 0755 target/release/riley riley-diagnostic
ldd riley-diagnostic >dependencies.txt
! grep -Fq 'not found' dependencies.txt
sha256sum riley-diagnostic dependencies.txt toolchain.txt cpu-tests.log src/"$scheduler" >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
