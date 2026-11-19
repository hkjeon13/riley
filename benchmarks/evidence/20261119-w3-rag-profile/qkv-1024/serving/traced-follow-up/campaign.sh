#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
qkv="$evidence/w3-qkv-serving-20261119.MhydzNsX"
trace="$evidence/w3-request-stages-20261119.DWykKwJR"
preceding="$evidence/w4-residual-replace-20261119.DZyMRtQ6"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit.tmp"; mv -- "$run/campaign.exit.tmp" "$run/campaign.exit"' EXIT
cd "$run"
sha256sum -c dispatch.sha256
test "$(cat "$qkv/campaign.exit")" = 0
test "$(cat "$trace/campaign.exit")" = 0
test "$(cat "$package/campaign.exit")" = 0
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
python=/usr/bin/python3
holder_status() {
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status
}
holder_status >initial-reservation.json
"$python" -B - <<'PY'
import json
from pathlib import Path

idle = json.loads(Path("initial-reservation.json").read_bytes())
if (idle["physical_index"] != 7 or idle["pid"] != 2431237 or idle["phase"] != "idle"
        or idle["other_processes"] or idle["allocation_bytes"] != 20 * 1024**3
        or idle["allocation_bytes"] != idle["torch_reserved_bytes"]):
    raise SystemExit("GPU 7 was not idle in the original 20 GiB reservation")
PY

mkdir src tools
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$package/source.tar" | sha256sum -c
tar -xf "$package/source.tar" -C src --no-same-owner
for file in run_profile_retry.py profile_workload.py load.py profile_summary.py profile_process.py compare_runner.py; do
    cp -- "$qkv/tools/$file" "tools/$file"
done
cp -- "$qkv/holder.py" .
sha256sum -c tools-base.sha256
gzip -dc traced-workload.patch.gz | git apply --check -
gzip -dc traced-workload.patch.gz | git apply -
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
    sha256sum -c "$run/trace-source-base.sha256"
    gzip -dc "$trace/source.patch.gz" | git apply --check -
    gzip -dc "$trace/source.patch.gz" | git apply -
    sha256sum -c "$run/trace-source.sha256"
    sha256sum -c "$qkv/source-base.sha256"
    gzip -dc "$qkv/source.patch.gz" | git apply --check -
    gzip -dc "$qkv/source.patch.gz" | git apply -
    sha256sum -c "$qkv/source.sha256"
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
