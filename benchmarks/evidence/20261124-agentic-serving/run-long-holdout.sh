#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
d=$(realpath ../long-tile)
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
for ((i=0; i<900; i++)); do
    [[ -s "$d/qwen-page/lease-after.json" ]] && break
    sleep 2
done
test -s "$d/qwen-page/lease-after.json"
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index 7)
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048
export TRIAL_PREFIX_BLOCKS=2048 LONG_TILE_HOLDOUT=1
for arm in page tile; do
    mode=page-v1
    if [[ $arm == tile ]]; then mode=fixed-tile-unsplit-v1; fi
    TRIAL_ATTENTION_MODE="$mode" "${ctl[@]}" run -- bash run-profile.sh "$arm" "$d/target/release/riley" 0 control > "$arm.log" 2>&1
    "${ctl[@]}" status > "$arm/lease-after.json"
done
touch released
