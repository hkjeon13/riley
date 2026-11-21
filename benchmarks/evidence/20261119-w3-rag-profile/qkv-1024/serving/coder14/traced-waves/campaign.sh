#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
instrumented="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
baseline="$evidence/w3-coder14-qkv-20261120.Mx2OpwKm"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$instrumented/campaign.exit")" = 0
test "$(cat "$baseline/campaign.exit")" = 0
(cd "$instrumented" && sha256sum -c "$stage/reused.sha256")
(cd "$baseline" && sha256sum -c result.sha256 >/dev/null)
printf '%s  %s\n' 712390666d906f9df399f95870037cb4b5112eb6b8607bd3834f661e579c442f "$baseline/verified.json" | sha256sum -c
(cd "$root/models/Qwen2.5-Coder-14B-Instruct" && sha256sum -c "$baseline/model.sha256")
mkdir -p tools target/release
cp -- "$instrumented/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$instrumented/holder.py" .
ln -- "$instrumented/target/release/riley" target/release/riley
"$python" -B - <<'PY'
from pathlib import Path
path = Path("tools/run_profile_retry.py")
source = path.read_text()
for old, new in (
    ('if model != "qwen14":', 'if model != "coder14":'),
    ('pinned to Qwen2.5-14B', 'pinned to Qwen2.5-Coder-14B'),
    ('"rag-c8-s2": case([1500, 3000], 128, 8, 64, 8, seed=2),',
     '"wave-sync": case([1500, 3000], 128, 8, 40, 8, stagger_ms=0),'),
):
    if source.count(old) != 1:
        raise ValueError(f"driver binding changed: {old}")
    source = source.replace(old, new)
compile(source, str(path), "exec")
path.write_text(source)
PY
sha256sum tools/run_profile_retry.py >driver.sha256

status() {
    local phase=$1 mode=$2
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$phase-$mode.json"
    done
    "$python" -B - "$phase" "$mode" <<'PY'
import json
from pathlib import Path
import sys
phase, mode = sys.argv[1:]
for index in (6, 7):
    def state(point):
        return json.loads(Path(f"gpu{index}-{point}-{mode}.json").read_bytes())
    item = state(phase)
    if (item["physical_index"] != index or item["phase"] != "idle" or item["other_processes"]
            or item["allocation_bytes"] != item["torch_reserved_bytes"] or item["allocation_bytes"] != 20 * 2**30):
        raise RuntimeError(f"GPU {index} is outside its exclusive idle reservation")
    if phase == "after":
        before = state("before")
        if any(item[key] != before[key] for key in ("pid", "start_ticks", "gpu")):
            raise RuntimeError(f"GPU {index} reservation holder changed")
PY
}
for arm in riley-first qkv-first qkv-second riley-second; do
    status before "$arm"
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    rc=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
      "$python" -B "$stage/tools/run_profile_retry.py" coder14 "$arm" >"$arm.controller.log" 2>&1 || rc=$?
    status after "$arm"
    cp -- "gpu7-after-$arm.json" "coder14-$arm/idle-after.json"
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum driver.sha256 gpu{6,7}-{before,after}-*.json \
    coder14-*/{plan,receipt,server-exit,idle-after,wave-sync,wave-50ms}.json \
    coder14-*/server.log >result.sha256
