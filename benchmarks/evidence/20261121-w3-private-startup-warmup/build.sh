#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum --strict -c build-inputs.sha256
test "$(cat "$package/build.exit")" = 0
(cd "$evidence" && sha256sum --strict -c "$stage/package-input.sha256")
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
gzip -dc private-warmup.patch.gz >private-warmup.patch
(
    cd src
    sha256sum --strict -c "$stage/baseline-source.sha256"
    git apply --check "$stage/private-warmup.patch"
    git apply "$stage/private-warmup.patch"
    while read -r _ path; do sha256sum "$path"; done <"$stage/baseline-source.sha256" >"$stage/patched-source.sha256"
)
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 CARGO_TARGET_DIR="$stage/target"
export RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
rustc --version >toolchain.txt
nvcc --version >>toolchain.txt
printf 'CPU_TEST %s\n' "$(date -u +%FT%TZ)"
(cd src && RILEY_W3_PREFIX_CACHE_MIN_128=1 cargo test --locked --offline --release \
    -p riley-scheduler --test simulation w3_) >cpu-tests.log 2>&1
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && cargo build --locked --offline --release -p riley-server --features server,hf-exact-probe) >compiler.log 2>&1
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >dependencies.txt
if grep -F 'not found' dependencies.txt; then exit 1; fi
strings target/release/riley | grep -F -e RILEY_W3_PREFIX_CACHE_MIN_128 -e RILEY_W3_UNCACHED_STARTUP >switches.txt
sha256sum target/release/riley cpu-tests.log toolchain.txt dependencies.txt patched-source.sha256 switches.txt >build-result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
