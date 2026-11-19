#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w5-periodic-trim-20261119.mI9aaaEp"
timing="$evidence/w5-trim-sample-timing-20261119.UtiF64Me"
preceding="$evidence/w3-qkv-serving-20261119.MhydzNsX"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
test -s "$timing/cache-off/gpu-evidence/report.json"
cp -a -- "$timing/harness" harness
sha256sum -c harness.sha256
cp -- "$prior/w5-periodic-trim.so" .
printf '%s  %s\n' 5463d8ef947b88f08165c71378ef8dc2910f35989ab5bcf529516c0042ed9edf w5-periodic-trim.so | sha256sum -c -
export PYTHONPATH="$stage:$base/src/ci"
export RILEY_W5_PERIODIC_TRIM_LIBRARY="$stage/w5-periodic-trim.so"
"$python" -B - <<'PY'
from pathlib import Path
from periodic_trim import PERIOD_MS, mount_options
from compare_sample_timing import trim_position
from verify_periodic_trim import CONTROLS, read_soak
assert PERIOD_MS == 7_000 and mount_options()[-1] == "RILEY_W5_PERIODIC_TRIM_MS=7000"
for p in CONTROLS.values():
    read_soak(Path.cwd().parent / p)
PY
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
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
for mode in cache-off cache-on; do
    mkdir "$mode"
    holder_status >"$mode/idle-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
        >"$mode/gpu-check.log" 2>&1 || rc=$?
    holder_status >"$mode/idle-after.json"
    printf '%d\n' "$rc" >"$mode/exit"
    "$python" -B verify_periodic_trim.py "$stage" >"$mode/original-gate.json"
    "$python" -B compare_sample_timing.py "$mode/gpu-evidence" >"$mode/timing-summary.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
done
sha256sum cache-{off,on}/{exit,idle-before.json,idle-after.json,original-gate.json,timing-summary.json} >result.sha256
