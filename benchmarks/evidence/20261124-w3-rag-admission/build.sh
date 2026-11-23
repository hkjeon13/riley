#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
native="$evidence/w5-shard-readiness-20261124.Jdlojvu8"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c build-dispatch.sha256
test "$(cat "$stock/campaign.exit")" = 0
test "$(cat "$native/campaign.exit")" = 0
printf '%s  %s\n' \
    e6b5b72d03baada23554c0a08d32b7183742b2fc6484e640006c88127a0e0da6 "$stock/source.tar" \
    e6112712011c7fe31ff710b5929e3895e8b0dc743e6347bdc332880c46b43455 "$native/target/release/riley" |
    sha256sum --strict -c

mkdir src
tar --no-same-owner -xf "$stock/source.tar" -C src
gzip -dc export-observations.patch.gz >export-observations.patch
printf '%s  %s\n' 7bdf373429fa1538f2856e3d99c5d875d5e0a4dccb0c3be4f92dca84a00f73df export-observations.patch |
    sha256sum --strict -c
(cd src
    git apply --check ../export-observations.patch
    git apply ../export-observations.patch
    sha256sum crates/riley-server/src/service.rs) >patched-source.sha256
mkdir -p target/release
# Keep the CUDA build directory local to this checkout. CMake stores absolute
# source and binary paths in its cache.
cp -a -- "$native/target/release/"{deps,.fingerprint} target/release/

export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64
export CARGO_TARGET_DIR="$stage/target" CUDA_VISIBLE_DEVICES= RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=2 CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'BUILD_START %s\n' "$(date -u +%FT%TZ)"
(cd src && taskset -c 24,25,26,27 cargo build --offline --locked --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
install -m 0755 target/release/riley riley-diagnostic
ldd riley-diagnostic >dependencies.txt
! grep -Fq 'not found' dependencies.txt
! cmp -s riley-diagnostic "$native/target/release/riley"
sha256sum riley-diagnostic dependencies.txt toolchain.txt patched-source.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
