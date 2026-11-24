#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
native="$evidence/w3-aged-decode-native-20261124.tNZ3fCMo"
short="$evidence/w3-aged-native-short-20261124.DbM7JcgV"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c build-dispatch.sha256
test "$(cat "$native/build.exit")" = 0
test "$(cat "$short/campaign.exit")" = 0
(cd "$native" && sha256sum --strict -c "$stage/native-build.sha256" && sha256sum --strict -c build-result.sha256)
(cd "$short" && sha256sum --strict -c "$stage/short-result.sha256" && sha256sum --strict -c result.sha256 >/dev/null)
cp -a -- "$native/src" src
(cd src && sha256sum --strict -c "$native/patched-source.sha256" &&
    git apply --check --whitespace=error ../export-observations.patch &&
    git apply --whitespace=error ../export-observations.patch)
(cd src && sha256sum crates/riley-scheduler/src/scheduler.rs crates/riley-scheduler/tests/aged_decode_reservation.rs crates/riley-server/src/{engine,main,service}.rs) >patched-source.sha256
mkdir -p target/release
cp -a -- "$native/target/release/"{deps,.fingerprint} target/release/
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
! cmp -s riley-diagnostic "$native/riley-diagnostic"
sha256sum riley-diagnostic dependencies.txt toolchain.txt patched-source.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
