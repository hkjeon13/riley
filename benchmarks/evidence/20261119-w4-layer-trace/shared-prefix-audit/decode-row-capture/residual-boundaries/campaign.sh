#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w4-decode-final-retry-20261120.5bTpAh8k"
trace="$evidence/w4-decode-layer-trace-20261120.kl10bRNz"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
python=/usr/bin/python3
venv="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
reservations() {
    local point=$1
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"gpu6-$point.json" || return 1
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$point.json" || return 1
    "$python" -B - "$point" <<'PY' || return 1
import json
from pathlib import Path
import sys

for index, pid, ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
    state=json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["start_ticks"] != ticks
            or state["phase"] != "idle" or state["other_processes"]
            or state["allocation_bytes"] != state["torch_reserved_bytes"] or state["allocation_bytes"] != 20*1024**3):
        raise ValueError(f"GPU {index} is missing its original idle reservation")
PY
}


test "$(cat "$trace/campaign.exit")" = 0
test "$(cat "$previous/campaign.exit")" = 0
printf '%s  %s\n' dd96d3894c55f29776a67e29ffa81bfc4989f006f95f900d158ecaf19500591a \
    "$trace/result.sha256" | sha256sum -c
(cd "$trace" && sha256sum -c result.sha256) >trace-check.log
mkdir tools
cp -- "$trace/tools/"{capture_final_rows,capture_reference,long_accuracy,replay_final_head,trace_decode_layers}.py tools/
cp -- replay_decode_boundaries.py tools/
sha256sum -c tools.sha256
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
deadline=$((SECONDS + 1800))
until reservations before; do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'START %s\n' "$(date -u +%FT%TZ)"
rc=0
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    timeout --signal=TERM --kill-after=20s 900s "$venv" -B "$stage/tools/replay_decode_boundaries.py" \
    "$trace" "$previous/verified.json" "$model" >result.json 2>client.log || rc=$?
reservations after
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
sha256sum result.json client.log gpu{6,7}-{before,after}.json >result.sha256
