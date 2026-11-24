#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) modes=(off bounded headroom);; 7) modes=(headroom bounded off);; *) exit 2;; esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_PREFILL_TRACE=1
mkdir "gpu$gpu"
for mode in "${modes[@]}"; do
    out="gpu$gpu/$mode"
    export CODER_PRESSURE_MODE="$mode"
    "${ctl[@]}" run -- bash -c 'date +%s%N > "$1.start"; bash run-profile.sh "$1" stock 5091 wide && date +%s%N > "$1.end"' _ "$out" > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
