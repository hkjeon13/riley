#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
previous="$evidence/w3-phase-gpu-20261119.9BEN4gYI"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256 gpu-dispatch.sha256 preflight.sha256
test "$(cat "$previous/campaign.exit")" = 0
test "$(cat "$base/campaign.exit")" = 0
(cd "$harness" && sha256sum -c "$stage/inputs.sha256")
export PYTHONPATH="$harness/ci:$base/src/ci"
"$python" -B verify_zerodecay.py "$stage" "$evidence" --check-controls

holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

for mode in cache-on cache-off; do
    mkdir "$mode"
    holder_status 6 >"$mode/idle-before.json"
    holder_status 7 >"$mode/gpu7-before.json"
    "$python" -B verify_zerodecay.py "$stage" "$evidence" --check-admission "$mode"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    result=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_gpu_check.py" "$stage" "$base/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" "$mode" \
        >"$mode/gpu-check.log" 2>&1 || result=$?
    holder_status 6 >"$mode/idle-after.json"
    holder_status 7 >"$mode/gpu7-after.json"
    printf '%d\n' "$result" >"$mode/exit"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$result"
    "$python" -B verify_zerodecay.py "$stage" "$evidence" --mode "$mode" >"$mode/verification.json"
done
"$python" -B verify_zerodecay.py "$stage" "$evidence" >verified-comparison.json
sha256sum verified-comparison.json cache-{on,off}/{verification.json,exit,idle-before.json,idle-after.json,allocator-before.json,allocator-after.json,gpu7-before.json,gpu7-after.json} >result.sha256
