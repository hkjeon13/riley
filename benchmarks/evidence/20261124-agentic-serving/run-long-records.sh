#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
d=$(realpath ../long-tile)
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
test -s "$d/llama-tile/lease-after.json"
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index 6)
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048
export TRIAL_PREFIX_BLOCKS=2048 LONG_TOOL_RECORDS=1
for arm in tile page; do
    mode=page-v1
    if [[ $arm == tile ]]; then mode=fixed-tile-unsplit-v1; fi
    TRIAL_ATTENTION_MODE="$mode" "${ctl[@]}" run -- bash run-profile.sh "$arm" "$d/target/release/riley" 0 control > "$arm.log" 2>&1
    "${ctl[@]}" status > "$arm/lease-after.json"
done
touch released
