#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I/src"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-before.json"; done
mkdir src
cp -a -- "$previous/"{Cargo.toml,Cargo.lock,rust-toolchain.toml,crates,kernels,third_party} src/
(
    cd src
    sha256sum --strict -c "$stage/source.sha256"
    git apply --check "$stage/witness.patch"
    git apply "$stage/witness.patch"
    sha256sum --strict -c "$stage/patched.sha256"
) >source-verification.log

export PATH="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 CARGO_TARGET_DIR="$stage/target"
export RILEY_CUDA_ARCHITECTURES=80 CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 CUDA_VISIBLE_DEVICES=
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    rustc --version
    nvcc --version
    cargo test --locked --offline --release -p riley-cuda --features cuda --test flashinfer_attention_gpu --no-run
) >build.log 2>&1
artifacts=(target/release/deps/flashinfer_attention_gpu-*)
binary=
for artifact in "${artifacts[@]}"; do
    if [[ -x $artifact && -f $artifact && $artifact != *.d ]]; then
        test -z "$binary"
        binary="$artifact"
    fi
done
test -n "$binary"
sha256sum "$binary" >binary.sha256
printf 'TEST %s\n' "$(date -u +%FT%TZ)"
rc=0
"${control[@]}" --index 6 run -- "$stage/$binary" --ignored --nocapture >test.log 2>&1 || rc=$?
for index in 6 7; do "${control[@]}" --index "$index" status >"gpu$index-after.json"; done
printf 'END %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
grep -E '^long_context_page_visibility rows=20 query_heads=28 context=8192 kv_chunk=512 max_abs=' test.log
sha256sum source-verification.log build.log binary.sha256 test.log gpu{6,7}-{before,after}.json >result.sha256
