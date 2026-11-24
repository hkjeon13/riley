#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) modes=(model application);; 7) modes=(application model);; *) exit 2;; esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct
mkdir "gpu$gpu"
for mode in "${modes[@]}"; do
    out="gpu$gpu/$mode"
    export CODER_NAMED_MODE="$mode"
    "${ctl[@]}" run -- bash run-profile.sh "$out" stock 5091 wide > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
