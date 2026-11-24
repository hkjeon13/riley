#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) previous=tile;; 7) previous=page;; *) exit 2;; esac
until test -s "gpu$gpu/$previous/lease-after.json"; do sleep 3; done
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct FROZEN_INPUT_DIR="$PWD/frozen" CODER_PREFILL_TRACE=1 TRIAL_ATTENTION_MODE=page-v1
export RILEY_NATIVE_D128_PROMPT_DIRECT=1 RILEY_NATIVE_D128_PROMPT_DIRECT_MAX_PAGES=0
out="gpu$gpu/direct"
env | grep '^RILEY_NATIVE_D128_PROMPT_DIRECT' > "$out.env"
"${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide > "$out.controller.log" 2>&1
"${ctl[@]}" status > "$out/lease-after.json"
