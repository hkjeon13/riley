#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
root=$(dirname "$evidence")
package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
previous="$evidence/w3-qkv-serving-20261119.MhydzNsX"
preceding="$evidence/w5-tcache-control-20261119.liqVpVJ7"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit.tmp"; mv -- "$run/campaign.exit.tmp" "$run/campaign.exit"' EXIT
cd "$run"
sha256sum -c dispatch.sha256
test "$(cat "$package/campaign.exit")" = 0
test "$(cat "$previous/campaign.exit")" = 0
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
printf 'PREDECESSOR_EXIT %s %s\n' "$(cat "$preceding/campaign.exit")" "$(date -u +%FT%TZ)"
python=/usr/bin/python3
holder_status() {
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for gpu in 6 7; do holder_status "$gpu" >"initial-reservation-$gpu.json"; done
"$python" -B - <<'PY'
import json
from pathlib import Path

for index, pid in ((6, 2431228), (7, 2431237)):
    idle = json.loads(Path(f"initial-reservation-{index}.json").read_bytes())
    if (idle["physical_index"] != index or idle["pid"] != pid or idle["phase"] != "idle"
            or idle["other_processes"] or idle["allocation_bytes"] != 20 * 1024**3
            or idle["allocation_bytes"] != idle["torch_reserved_bytes"]):
        raise SystemExit(f"GPU {index} is not in its original 20 GiB idle reservation")
PY

mkdir src tools
mkdir -p bin/base bin/ordered
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$package/source.tar" | sha256sum -c
tar -xf "$package/source.tar" -C src --no-same-owner
for file in run_profile_retry.py profile_workload.py load.py profile_summary.py profile_process.py compare_runner.py; do
    cp -- "$previous/tools/$file" "tools/$file"
done
cp -- "$previous/holder.py" .
sha256sum -c helpers.sha256 tools-base.sha256
gzip -dc serving-runner.patch.gz | git apply --check -
gzip -dc serving-runner.patch.gz | git apply -
sha256sum -c tools.sha256

model="$root/models/SmolLM2-1.7B-Instruct"
printf '%s  %s\n' 592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753 \
    "$model/riley-checkpoint.json" | sha256sum -c
toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4
export CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$run/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1
printf 'BUILD base %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    sha256sum -c "$run/source-base.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build-base.log 2>&1
install -m 0700 target/release/riley bin/base/riley
printf 'BUILD ordered %s\n' "$(date -u +%FT%TZ)"
(
    cd src
    git apply --check "$run/ordered-physical-tables.patch"
    git apply "$run/ordered-physical-tables.patch"
    sha256sum -c "$run/source-ordered.sha256"
    "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe
) >build-ordered.log 2>&1
install -m 0700 target/release/riley bin/ordered/riley
sha256sum bin/{base,ordered}/riley >binaries.sha256
test "$(cut -d' ' -f1 binaries.sha256 | sort -u | wc -l)" -eq 2
for mode in base-first ordered-first ordered-second base-second base-trace ordered-trace; do
    holder_status 7 >"idle-before-$mode.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    status=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$run/tools/run_profile_retry.py" smol "$mode" || status=$?
    holder_status 7 >"smol-$mode/idle-after.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then exit "$status"; fi
done
for mode in base-trace ordered-trace; do
    "$python" -B summarize_sync_gaps.py "smol-$mode/trace.sqlite" >"smol-$mode/sync-gaps.json"
done
