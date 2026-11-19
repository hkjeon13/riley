#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
operations="$evidence/w3-qkv-1024-20261119.IZQF8D4T"
previous="$evidence/w3-mixed-step-20261119.VBAhw0Z5"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit.tmp"; mv -- "$run/campaign.exit.tmp" "$run/campaign.exit"' EXIT
cd "$run"
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$operations" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$operations/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$operations/campaign.pid")" | grep -Fq -- "$operations/campaign.sh"; then
        test -s "$operations/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 25
done
test "$(cat "$operations/campaign.exit")" = 0
test -s "$operations/result.json"
test "$(cat "$previous/campaign.exit")" = 0
test "$(cat "$package/campaign.exit")" = 0
python=/usr/bin/python3
holder_status() {
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status
}
holder_status >initial-reservation.json
"$python" -B - initial-reservation.json <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    idle = json.load(handle)
if (idle["phase"] != "idle" or idle["other_processes"]
    or idle["allocation_bytes"] != 20 * 1024**3):
    raise SystemExit("GPU 7 was not exclusively reserved with 20 GiB before the campaign")
PY

mkdir src tools
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$package/source.tar" | sha256sum -c
tar -xf "$package/source.tar" -C src --no-same-owner
for file in run_profile_retry.py profile_workload.py load.py profile_summary.py profile_process.py compare_runner.py; do
    cp -- "$previous/tools/$file" "tools/$file"
done
cp -- "$previous/holder.py" .
sha256sum -c "$previous/tools.sha256"
sha256sum -c tools-base.sha256
gzip -dc tools.patch.gz | git apply --check -
gzip -dc tools.patch.gz | git apply -
sha256sum -c tools.sha256

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$run/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$run/source-base.sha256"
    gzip -dc "$run/source.patch.gz" | git apply --check -
    gzip -dc "$run/source.patch.gz" | git apply -
    sha256sum -c "$run/source.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build.log 2>&1
sha256sum target/release/riley >binary.sha256
for mode in riley-first qkv-first qkv-second riley-second; do
    holder_status >"idle-before-$mode.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    status=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$run/tools/run_profile_retry.py" qwen14 "$mode" || status=$?
    holder_status >"qwen14-$mode/idle-after.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then exit "$status"; fi
done
