#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) modes=(full selected);; 7) modes=(selected full);; *) exit 2;; esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_PREFILL_TRACE=1
mkdir "gpu$gpu"
for mode in "${modes[@]}"; do
    out="gpu$gpu/$mode"
    export CODER_CATALOG_MODE="$mode"
    "${ctl[@]}" run -- bash run-profile.sh "$out" stock 5091 wide > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
