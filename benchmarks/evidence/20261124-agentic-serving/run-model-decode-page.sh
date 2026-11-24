#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) model=Qwen2.5-3B-Instruct;; 7) model=Qwen2.5-Coder-7B-Instruct;; *) exit 2;; esac
until [[ -s gpu$gpu/reference-lease.json ]]; do sleep 3; done
export TRIAL_ATTENTION_MODE=page-v1 AGENT_DECODE_MODEL="$model" FROZEN_INPUT_DIR="$PWD/frozen"
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
"${ctl[@]}" run -- bash run-profile.sh "gpu$gpu/source-page" stock 761 wide > "gpu$gpu/source-page.controller.log" 2>&1
"${ctl[@]}" status > "gpu$gpu/source-page/lease-after.json"
