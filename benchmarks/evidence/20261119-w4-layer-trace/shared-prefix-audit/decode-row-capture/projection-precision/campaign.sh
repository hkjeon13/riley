#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source="$evidence/w4-matched-mlp-20261120.wOWcpjAr"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
python=/usr/bin/python3
venv="$root/tools/vllm-venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$source/campaign.exit")" = 0
printf '%s  %s\n' 4e4c659548d41384f7a3c1828794834bfbf41acc225bc1754eaa21d5bce22299 \
    "$source/result.sha256" | sha256sum -c
(cd "$source" && sha256sum -c result.sha256) >source-check.log
printf 'BIND %s\n' "$(date -u +%FT%TZ)"
"$venv" -B projection_oracle.py bind "$stage" "$source" "$model" >shard.json

reservations() {
    local point=$1
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json" || return 1
    done
    "$python" -B - "$point" <<'PY' || return 1
import json
from pathlib import Path
import sys
for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    state = json.loads(Path(f"gpu{gpu}-{sys.argv[1]}.json").read_bytes())
    assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
    assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
    assert not state["other_processes"]
PY
}

deadline=$((SECONDS+1800))
until reservations before; do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'START GPU %s\n' "$(date -u +%FT%TZ)"
rc=0
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    bash -c 'exec "$@" >gpu-receipt.json 2>gpu.log' diagnostic timeout --signal=TERM --kill-after=20s 600s \
    "$venv" -B "$stage/projection_oracle.py" gpu "$stage" "$source" "$model" || rc=$?
reservations after
printf 'END GPU %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
printf 'START CPU %s\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES= timeout --signal=TERM --kill-after=20s 600s \
    "$venv" -B projection_oracle.py cpu "$stage" "$source" "$model" >verified.json 2>cpu.log
printf 'END CPU %s\n' "$(date -u +%FT%TZ)"
sha256sum shard.json gpu-receipt.json gpu.log cpu.log verified.json gpu{6,7}-{before,after}.json gpu/*.bf16 >result.sha256
