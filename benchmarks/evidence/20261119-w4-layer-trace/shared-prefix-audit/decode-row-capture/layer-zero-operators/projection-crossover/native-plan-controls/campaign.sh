#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
reference="$evidence/w4-decode-layer0-ops-20261120.hYAOtc0n"
crossover="$evidence/w4-projection-crossover-20261120.J9REWUkf"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cuda=/usr/local/cuda-12.4
export PYTHONPATH="$crossover:$reference/tools" CUDA_DEVICE_ORDER=PCI_BUS_ID
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$crossover/campaign.exit")" = 0
(cd "$crossover" && sha256sum -c dispatch.sha256 result.sha256) >predecessor-check.log
patch --batch --silent -o replay_native_plans.cu "$crossover/replay_riley_lt.cu" <native-plan-controls.patch
"$cuda/bin/nvcc" -std=c++17 -O2 -arch=sm_80 replay_native_plans.cu -lcublasLt -o replay_native_plans
mkdir operands
for name in 669-9 1023-67; do
    ln -- "$crossover/operands/$name."{input,riley,hf}.bf16 operands/
done
ln -- "$crossover/operands/weights.bf16" operands/

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
    env LD_LIBRARY_PATH="$cuda/lib64" ./replay_native_plans operands >plans.txt 2>gpu.log || rc=$?
reservations after
printf 'GPU exit=%d\n' "$rc"
test "$rc" = 0
/usr/bin/python3 -B verify_native_plans.py "$stage" "$crossover" >verified.json
sha256sum plans.txt verified.json replay_native_plans.cu replay_native_plans gpu.log operands/*.bf16 gpu{6,7}-{before,after}.json >result.sha256
