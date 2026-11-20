#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
built="$evidence/w3-kv-pressure-retry-20261120.FFahBvG8"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$built/campaign.exit")" = 1
test ! -d "$built/qwen14-riley-first"
grep -Fxq 'ValueError: GPU 6 lacks the original exclusive idle reservation' "$built/campaign.log"
(cd "$built" && sha256sum -c dispatch.sha256 binary.sha256 patched.sha256) >built-stage-check.log
mkdir -p tools target/release
cp -- "$built/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$built"/{holder.py,binary.sha256} .
grep '^.*  tools/' "$built/patched.sha256" >run-tools.sha256
sha256sum -c run-tools.sha256
ln -- "$built/target/release/riley" target/release/riley
sha256sum -c binary.sha256

wait_for_idle() {
    local mode=$1 deadline=$((SECONDS+7200))
    printf 'WAIT %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    while ! reservations "$mode" before "qwen14-$mode" >"wait-$mode.log" 2>&1; do
        if (( SECONDS >= deadline )); then
            cat "wait-$mode.log"
            return 75
        fi
        sleep 20
    done
}
reservations() {
    local mode=$1 point=$2 name=$3
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point-$mode.json" || return 1
    done
    "$python" -B - "$mode" "$point" <<'PY' || return 1
import json
from pathlib import Path
import sys
mode,point=sys.argv[1:]
for index,pid,ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
    state=json.loads(Path(f"gpu{index}-{point}-{mode}.json").read_bytes())
    if (state["physical_index"],state["pid"],state["start_ticks"],state["phase"]) != (index,pid,ticks,"idle") or \
            state["allocation_bytes"] != 20*1024**3 or state["torch_reserved_bytes"] != 20*1024**3 or state["other_processes"]:
        raise ValueError(f"GPU {index} lacks the original exclusive idle reservation")
PY
    if test "$point" = after && test -d "$name"; then
        cp "gpu7-after-$mode.json" "$name/idle-after.json"
    fi
}
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
for mode in riley-first riley-second; do
    name="qwen14-$mode"
    wait_for_idle "$mode"
    rc=0
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_profile_retry.py" qwen14 "$mode" >"$mode.controller.log" 2>&1 || rc=$?
    reservations "$mode" after "$name"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum run-tools.sha256 binary.sha256 gpu{6,7}-{before,after}-*.json \
  qwen14-*/{plan,receipt,server-exit,idle-after,idle-before,drain,wave-50ms,rag-c8-s2}.json \
  qwen14-*/{server.log,server.pid} >result.sha256
