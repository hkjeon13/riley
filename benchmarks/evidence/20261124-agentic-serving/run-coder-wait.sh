#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) order=(off history boundary);; 7) order=(boundary history off);; *) exit 2;; esac
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct FROZEN_INPUT_DIR="$PWD/frozen" CODER_PREFILL_TRACE=1
mkdir "gpu$gpu"
for mode in "${order[@]}"; do
    out="gpu$gpu/$mode"
    export CODER_WAIT_MODE="$mode"
    "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
