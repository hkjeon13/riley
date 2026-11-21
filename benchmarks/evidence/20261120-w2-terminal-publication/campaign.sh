#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w2-ordered-tables-20261119.e89GGsLO"
harness="$evidence/w5-serving-smoke-recheck-20261120.PsNfm6KN"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 "$base/source.tar" | sha256sum -c
printf '%s  %s\n' 655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f \
    "$root/models/Qwen2.5-7B-Instruct/riley-checkpoint.json" | sha256sum -c
mkdir src bin tools
cp -- "$prior/holder.py" .
cp -- "$prior/tools/compare_runner.py" tools/
cp -- "$harness/benchmarks/live/"{api_check,soak,load}.py tools/
cp -- run_publication.py tools/
sha256sum -c reused.sha256
tar -xf "$base/source.tar" -C src --no-same-owner

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 W5_GPU_INDEX=7
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$stage/source-before.sha256"
    git apply --check "$stage/terminal-order.patch"
    git apply "$stage/terminal-order.patch"
    sha256sum -c "$stage/source-after.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build.log 2>&1
install -m 0700 target/release/riley bin/riley
sha256sum bin/riley >candidate.sha256

reservations() {
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"$1/gpu$index-$2.json"
    done
    "$python" -B verify_publication.py --reservations "$1" "$2"
}

for mode in immediate-v1 after-submit-v1; do
    before="$stage/before-$mode"
    mkdir "$before"
    reservations "$before" before
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_publication.py" "$mode" >"$before/control.log" 2>&1 || rc=$?
    reservations "$stage/$mode" after
    mv -- "$before"/gpu{6,7}-before.json "$stage/$mode/"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_publication.py "$stage" >verified.json
sha256sum verified.json candidate.sha256 {immediate-v1,after-submit-v1}/{receipt,api,soak,shutdown-metrics,gpu6-before,gpu6-after,gpu7-before,gpu7-after}.json >result.sha256
