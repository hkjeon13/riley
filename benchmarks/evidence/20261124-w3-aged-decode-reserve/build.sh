#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w3-prefill-step-cost-20261124.N8fkgjBg"
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
trap 'printf "%s\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT

sha256sum --strict -c build-dispatch.sha256
test "$(cat "$prior/build.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
printf '%s  %s\n' d20afdd68fac81b1db338095407235ae14b565b8e3e42a2f837f3d9581fffada "$prior/build-result.sha256" |
    sha256sum --strict -c
(cd "$prior" && sha256sum --strict -c build-result.sha256)
cp -a -- "$prior/src" src
(cd src && sha256sum --strict -c "$prior/patched-source.sha256")
python3 -B reserve-aged-decode.py src
python3 -B trace-aged-turn.py src
(cd src && sha256sum crates/riley-scheduler/src/scheduler.rs crates/riley-server/src/{engine,service,lib,rag_steps}.rs) >patched-source.sha256
mkdir -p target/release
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
