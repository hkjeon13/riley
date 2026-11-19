#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
final_rows="$evidence/w4-decode-final-retry-20261120.5bTpAh8k"
reference="$evidence/w4-decode-layer0-ops-20261120.hYAOtc0n"
riley_plan="$evidence/w4-qkv-plan-20261120.hpF6Lg0W"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
venv="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for previous in "$reference" "$riley_plan" "$final_rows"; do
    test "$(cat "$previous/campaign.exit")" = 0
    (cd "$previous" && sha256sum -c dispatch.sha256 result.sha256) >>predecessor-check.log
done
mkdir tools hf
cp -- "$final_rows/tools/"{long_accuracy.py,capture_reference.py,capture_final_rows.py,replay_final_head.py} tools/
cp -- trace_decode_layers.py profile_hf_qkv.py tools/
ln -s -- "$final_rows/fixtures" fixtures
sha256sum -c previous-tools.sha256

reservations() {
    local point=$1
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json"
    done
    /usr/bin/python3 -B - "$point" <<'PY'
import json
from pathlib import Path
import sys

for index, pid, ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (row["physical_index"],row["pid"],row["start_ticks"],row["phase"]) != (index,pid,ticks,"idle") or \
            row["allocation_bytes"] != 20*1024**3 or row["torch_reserved_bytes"] != 20*1024**3 or row["other_processes"]:
        raise ValueError(f"GPU {index} is missing its original 20 GiB reservation")
PY
}
reservations before
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
printf 'START hf %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    "$venv" -B tools/trace_decode_layers.py hf fixtures "$model" hf/rows "$final_rows/verified.json" \
    >hf/trace-receipt.json 2>hf/trace.log || rc=$?
reservations after
printf 'END hf %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$venv" -B verify_hf_kernels.py "$stage" "$reference" "$model" >verified.json
sha256sum hf/{trace-receipt.json,trace.log,rows/*.bf16} gpu{6,7}-{before,after}.json verified.json >result.sha256
