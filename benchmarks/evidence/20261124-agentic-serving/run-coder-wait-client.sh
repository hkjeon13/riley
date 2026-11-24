#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6|7) ;; *) exit 2;; esac
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_WAIT_CLIENT=1
out=gpu$gpu
"${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide > "$out.controller.log" 2>&1
"${ctl[@]}" status > "$out/lease-after.json"
