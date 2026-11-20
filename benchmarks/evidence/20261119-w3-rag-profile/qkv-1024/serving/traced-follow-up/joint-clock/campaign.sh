#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
prior="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$prior/campaign.exit")" = 0
(cd "$prior" && sha256sum -c "$stage/reused.sha256" binary.sha256) >reused-check.log
mkdir -p tools target/release
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior"/{holder.py,binary.sha256} .
ln -- "$prior/target/release/riley" target/release/riley
"$python" -B enable_joint_trace.py tools/run_profile_retry.py
sha256sum tools/run_profile_retry.py >patched-driver.sha256

reservations() {
    local mode=$1 point=$2 name=$3
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point-$mode.json"
    done
    "$python" -B - "$mode" "$point" <<'PY'
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
for mode in riley-first qkv-first; do
    name="qwen14-$mode"
    reservations "$mode" before "$name"
    rc=0
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_profile_retry.py" qwen14 "$mode" >"$mode.controller.log" 2>&1 || rc=$?
    reservations "$mode" after "$name"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum patched-driver.sha256 binary.sha256 gpu{6,7}-{before,after}-*.json \
    qwen14-*/{plan,receipt,server-exit,idle-after,wave-50ms,rag-c8-s2,wave-50ms.profile,rag-c8-s2.profile}.json \
    qwen14-*/{trace.sqlite,trace.nsys-rep,server.log,server.pid} >result.sha256
