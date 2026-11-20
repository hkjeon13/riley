#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source="$evidence/w4-matched-mlp-20261120.wOWcpjAr"
oracle="$evidence/w4-projection-oracle-20261120.RB5VX8Md"
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-native-projection-plans-20261120\.[A-Za-z0-9]+$ ]]; then exit 2; fi
native=$1
cuda=/usr/local/cuda-12.4
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
test "$(cat "$oracle/campaign.exit")" = 0
printf '%s  %s\n' 0fc62e2b3cffacfa2cc638bd73e6e54a77935a15a400b02728e41882b037f8f5 \
    "$oracle/result.sha256" | sha256sum -c
(cd "$oracle" && sha256sum -c result.sha256) >oracle-check.log
printf 'COMPILE %s\n' "$(date -u +%FT%TZ)"
"$cuda/bin/nvcc" -arch=sm_80 -std=c++17 -O2 replay_projection_plans.cu -lcublasLt -o replay_projection_plans
printf 'PACK %s\n' "$(date -u +%FT%TZ)"
"$venv" -B replay_table.py pack "$stage" "$native" "$oracle" "$model" "$source" >weights.json

deadline=$((SECONDS+1800))
while ! test -s "$native/campaign.exit"; do
    ps -o args= -p "$(cat "$native/campaign.pid")" | grep -Fq -- "$native/campaign.sh" || {
        test -s "$native/campaign.exit" && break
        exit 75
    }
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$native/campaign.exit")" = 0
(cd "$native" && sha256sum -c dispatch.sha256 result.sha256) >native-check.log

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
    env LD_LIBRARY_PATH="$cuda/lib64" \
    timeout --signal=TERM --kill-after=20s 600s "$venv" -B "$stage/replay_table.py" gpu \
    "$stage" "$native" "$oracle" "$model" "$source" >gpu.log 2>&1 || rc=$?
reservations after
printf 'END GPU %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
printf 'START CPU %s\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES= timeout --signal=TERM --kill-after=20s 600s \
    "$venv" -B replay_table.py evaluate "$stage" "$native" "$oracle" "$model" "$source" >verified.json 2>cpu.log
printf 'END CPU %s\n' "$(date -u +%FT%TZ)"
sha256sum weights.json replay_projection_plans gpu.log cpu.log verified.json gpu{6,7}-{before,after}.json \
    {down_proj,o_proj}/{plans.log,*.bf16} >result.sha256
