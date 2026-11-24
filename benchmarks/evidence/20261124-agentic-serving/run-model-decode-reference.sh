#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
case "$gpu" in 6) last=3-riley;cpus=24-27;; 7) last=3-vllm;cpus=34-37;; *) exit 2;; esac
until [[ -s gpu$gpu/$last/lease-after.json ]]; do sleep 3; done
root=$(realpath ../../..)
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${ctl[@]}" run -- taskset -c "$cpus" "$root/tools/vllm-venv/bin/python" -B model_decode_reference.py "$gpu" > "gpu$gpu/reference.log" 2>&1
"${ctl[@]}" status > "gpu$gpu/reference-lease.json"
