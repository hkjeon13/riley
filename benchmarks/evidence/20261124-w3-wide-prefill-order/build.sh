#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
prior="$evidence/w3-rag-admission-retry-20261124.OaUPAucl"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c build-dispatch.sha256
test "$(cat "$stock/campaign.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
printf '%s  %s\n' \
    e6b5b72d03baada23554c0a08d32b7183742b2fc6484e640006c88127a0e0da6 "$stock/source.tar" \
    fe91258591ce3a77ab15fe54794214600701077b9829be665c5a549d45796c58 "$prior/build-result.sha256" |
    sha256sum --strict -c
(cd "$prior" && sha256sum --strict -c build-result.sha256)
mkdir src
tar --no-same-owner -xf "$stock/source.tar" -C src
python3 -B instrument.py src base-instrument.py
(cd src && sha256sum crates/riley-scheduler/src/scheduler.rs crates/riley-server/src/engine.rs) >patched-source.sha256
mkdir -p target/release
# CMake stores absolute paths, so its build directory is regenerated here.
cp -a -- "$prior/target/release/"{deps,.fingerprint} target/release/

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
! cmp -s riley-diagnostic "$prior/riley-diagnostic"
sha256sum riley-diagnostic dependencies.txt toolchain.txt patched-source.sha256 >build-result.sha256
printf 'BUILD_END %s\n' "$(date -u +%FT%TZ)"
