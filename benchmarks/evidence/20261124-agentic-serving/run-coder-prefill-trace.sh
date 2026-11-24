#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) order=(page tile);; 7) order=(tile page);; *) exit 2;; esac
until test -s build.exit; do sleep 2; done
test "$(cat build.exit)" = 0
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct FROZEN_INPUT_DIR="$PWD/frozen" CODER_PREFILL_TRACE=1
mkdir "gpu$gpu"
for mode in "${order[@]}"; do
    if [[ $mode == page ]]; then export TRIAL_ATTENTION_MODE=page-v1; else export TRIAL_ATTENTION_MODE=fixed-tile-unsplit-v1; fi
    out="gpu$gpu/$mode"
    "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
