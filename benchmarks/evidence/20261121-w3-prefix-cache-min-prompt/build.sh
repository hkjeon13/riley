#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-current-package-20261121.f6SEbjvA"
archive=429f0132fed714f5af133f756e4374f21733ed4f27074b05f69b3b9930f5b6b6
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum -c build-inputs.sha256
test "$(cat "$package/campaign.exit")" = 0
printf "%s  %s\n" "$archive" "$package/source.tar" | sha256sum -c
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
(
    cd src
    sha256sum -c "$stage/baseline-source.sha256"
    git apply --check "$stage/min-prompt.patch"
    git apply "$stage/min-prompt.patch"
    sha256sum crates/riley-scheduler/src/scheduler.rs crates/riley-scheduler/tests/simulation.rs \
        >"$stage/patched-source.sha256"
)

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 CARGO_TARGET_DIR="$stage/target"
export RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >dependencies.txt
if grep -F 'not found' dependencies.txt; then exit 1; fi
strings target/release/riley | grep -F 'RILEY_W3_PREFIX_CACHE_MIN_128' >switch.txt
sha256sum target/release/riley toolchain.txt dependencies.txt patched-source.sha256 switch.txt >build-result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
