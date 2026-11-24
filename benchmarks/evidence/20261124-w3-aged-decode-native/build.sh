#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w4-uri-package-20261124.73IfvPre"
previous="$evidence/w3-aged-decode-reserve-20261124.UbReV73C"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c build-dispatch.sha256
test "$(cat "$stock/campaign.exit")" = 0
test "$(cat "$previous/build.exit")" = 0
(cd "$stock" && sha256sum --strict -c "$stage/package-source.sha256")
cp -a -- "$stock/src" src
(cd src && sha256sum --strict -c ../parent-source.sha256 &&
    git apply --check --whitespace=error ../aged-decode-option.patch &&
    git apply --whitespace=error ../aged-decode-option.patch)
(cd src && sha256sum crates/riley-scheduler/src/scheduler.rs crates/riley-scheduler/tests/aged_decode_reservation.rs crates/riley-server/src/{engine,main}.rs) >patched-source.sha256
mkdir -p target/release
cp -a -- "$previous/target/release/"{deps,.fingerprint} target/release/

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
./riley-diagnostic --help >help.txt
grep -Fq -- '--prefill-aging-reserve-decode' help.txt
sha256sum riley-diagnostic dependencies.txt toolchain.txt patched-source.sha256 help.txt >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
