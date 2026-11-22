#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
private="$evidence/w3-private-warm-build-20261121.xrfPErXB"
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-inputs.sha256
test "$(cat "$package/build.exit")" = 0
test "$(cat "$private/build.exit")" = 0
(cd "$evidence" && sha256sum --strict -c "$stage/external-inputs.sha256")
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
gzip -dc "$private/private-warmup.patch.gz" >private-warmup.patch
gzip -dc export-observations.patch.gz >export-observations.patch
(
    cd src
    git apply --check "$stage/private-warmup.patch"
    git apply "$stage/private-warmup.patch"
    sha256sum --strict -c "$stage/observation-base.sha256"
    git apply --check "$stage/export-observations.patch"
    git apply "$stage/export-observations.patch"
    sha256sum crates/riley-server/src/service.rs >"$stage/patched-source.sha256"
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
strings target/release/riley | grep -F -e RILEY_W3_PREFIX_CACHE_MIN_128 -e RILEY_W3_UNCACHED_STARTUP -e RILEY_W3_EXPORT_OBSERVATIONS >switches.txt
grep -qF RILEY_W3_EXPORT_OBSERVATIONS switches.txt
sha256sum target/release/riley toolchain.txt dependencies.txt patched-source.sha256 switches.txt >build-result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
