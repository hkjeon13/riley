#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
reference="$evidence/w4-decode-layer0-ops-20261120.hYAOtc0n"
riley_plan="$evidence/w4-qkv-plan-20261120.hpF6Lg0W"
hf_profile="$evidence/w4-hf-qkv-retry-20261120.e5y92toi"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
venv="$root/tools/vllm-venv/bin/python"
cuda=/usr/local/cuda-12.4
export PYTHONPATH="$reference/tools:$hf_profile/tools" CUDA_DEVICE_ORDER=PCI_BUS_ID
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for previous in "$reference" "$riley_plan" "$hf_profile"; do
    test "$(cat "$previous/campaign.exit")" = 0
    (cd "$previous" && sha256sum -c dispatch.sha256 result.sha256) >>predecessor-check.log
done
/usr/bin/python3 -B projection_crossover.py prepare operands "$model" "$reference" "$riley_plan" "$hf_profile" >preparation.json
"$cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_80 replay_riley_lt.cu -lcublasLt -o replay_riley_lt

reservations() {
    local point=$1
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json"
    done
    /usr/bin/python3 -B - "$point" <<'PY'
import json
from pathlib import Path
import sys

for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (row["physical_index"], row["pid"], row["start_ticks"], row["phase"]) != (index, pid, ticks, "idle") or \
            row["allocation_bytes"] != 20 * 1024**3 or row["torch_reserved_bytes"] != 20 * 1024**3 or row["other_processes"]:
        raise ValueError(f"GPU {index} is missing its original 20 GiB reservation")
PY
}
reservations before
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    /bin/bash -ec '
        env LD_LIBRARY_PATH="$1/lib64" ./replay_riley_lt operands
        "$2" -B projection_crossover.py torch operands preparation.json >torch.json
    ' _ "$cuda" "$venv" >gpu.log 2>&1 || rc=$?
reservations after
printf 'GPU exit=%d\n' "$rc"
test "$rc" = 0
/usr/bin/python3 -B projection_crossover.py compare operands preparation.json torch.json >verified.json
sha256sum preparation.json torch.json verified.json replay_riley_lt gpu.log operands/*.bf16 gpu{6,7}-{before,after}.json >result.sha256
