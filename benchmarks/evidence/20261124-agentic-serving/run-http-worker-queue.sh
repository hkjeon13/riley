#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) workers=(8 16);; 7) workers=(16 8);; *) exit 2;; esac
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_PREFILL_TRACE=1 CODER_PRESSURE_MODE=off
mkdir "gpu$gpu"
for n in "${workers[@]}"; do
    out="gpu$gpu/w$n"
    unset TRIAL_HTTP_WORKERS
    if [[ $n != 8 ]]; then export TRIAL_HTTP_WORKERS="$n"; fi
    "${ctl[@]}" run -- bash -c 'date +%s%N > "$1.start"; bash run-profile.sh "$1" stock 5091 wide && date +%s%N > "$1.end"' _ "$out" > "$out.controller.log" 2>&1
    "${ctl[@]}" status > "$out/lease-after.json"
done
