#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w2-ordered-tables-20261119.e89GGsLO"
base_package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
(cd "$prior" && sha256sum -c "$stage/reused.sha256")
printf '%s  %s\n' 592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753 \
    "$root/models/SmolLM2-1.7B-Instruct/riley-checkpoint.json" | sha256sum -c
mkdir tools src bin bin/base bin/candidate
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior/holder.py" .
cp -- logprob_probe.py tools/
ln -- "$prior/bin/base/riley" bin/base/riley
"$python" -B prepare_runner.py
sha256sum tools/{run_profile_retry,profile_workload,logprob_probe}.py >selected-tools.sha256
test "$(cat "$base_package/campaign.exit")" = 0
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$base_package/source.tar" | sha256sum -c
tar -xf "$base_package/source.tar" -C src --no-same-owner

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1
printf 'BUILD candidate %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$stage/source-before.sha256"
    git apply --check "$stage/move-committed-payload.patch"
    git apply "$stage/move-committed-payload.patch"
    sha256sum -c "$stage/source-after.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build.log 2>&1
install -m 0700 target/release/riley bin/candidate/riley
sha256sum bin/candidate/riley >candidate.sha256

reservations() {
    local folder=$1 time=$2
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"$folder/gpu$index-$time.json"
    done
    "$python" -B - "$folder" "$time" <<'PY'
from pathlib import Path
import sys
from verify_payload import status
for index in (6, 7):
    status(Path(sys.argv[1]), index, sys.argv[2])
PY
}

for mode in base-first candidate-first candidate-second base-second; do
    initial="$stage/before-$mode"
    mkdir "$initial"
    reservations "$initial" before
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_profile_retry.py" smol "$mode" >"$initial/control.log" 2>&1 || rc=$?
    folder="$stage/smol-$mode"
    mkdir -p "$folder"
    reservations "$folder" after
    mv -- "$initial/"gpu{6,7}-before.json "$folder/"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_payload.py "$stage" "$prior" >verified.json
sha256sum verified.json candidate.sha256 selected-tools.sha256 \
    smol-*/{receipt,plan,server-exit,gpu6-before,gpu6-after,gpu7-before,gpu7-after,reason-c32,long-c32,logprob-probe,token-id-probe}.json \
    >result.sha256
