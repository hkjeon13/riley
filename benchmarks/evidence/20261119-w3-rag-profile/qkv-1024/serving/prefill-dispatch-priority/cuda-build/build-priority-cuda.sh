#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
trace="$evidence/w3-request-stages-20261119.DWykKwJR"
qkv="$evidence/w3-qkv-serving-20261119.MhydzNsX"
prior="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$package/campaign.exit")" = 0
test "$(cat "$trace/campaign.exit")" = 0
test "$(cat "$qkv/campaign.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 "$package/source.tar" | sha256sum -c
mkdir src
tar -xf "$package/source.tar" -C src --no-same-owner
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4 CARGO_TARGET_DIR="$stage/target"
export RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CUDA_VISIBLE_DEVICES=
cd src
gzip -dc "$trace/source.patch.gz" | git apply --check -
gzip -dc "$trace/source.patch.gz" | git apply -
gzip -dc "$qkv/source.patch.gz" | git apply --check -
gzip -dc "$qkv/source.patch.gz" | git apply -
printf '%s  %s\n' 5585da9cf2fe5ae0b8ea64c2a4903d4827c2d04b175ed8263d57935aa2fe9cbd crates/riley-scheduler/src/scheduler.rs                   580fd5baabf02a6d9c273924837648fc0ab9394c88799d6b310e4d84e0e86d21 crates/riley-server/src/engine.rs | sha256sum -c
git apply "$stage/early-prefill-priority.patch"
git apply "$stage/wire-early-priority.patch"
sha256sum -c "$stage/runtime-source.sha256"
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
"$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe >"$stage/build.log" 2>&1
cd "$stage"
file target/release/riley >binary-type.txt
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ldd target/release/riley >binary-dependencies.txt
if grep -F 'not found' binary-dependencies.txt; then exit 1; fi
strings target/release/riley | grep -F 'RILEY_W3_EARLY_PREFILL_PRIORITY' >switch.txt
sha256sum target/release/riley binary-type.txt binary-dependencies.txt switch.txt >result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
