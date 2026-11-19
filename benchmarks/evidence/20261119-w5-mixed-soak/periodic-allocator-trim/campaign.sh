#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
sha256sum -c harness.sha256
sha256sum -c compiled.sha256
export PYTHONPATH="$stage:$base/src/ci"
export RILEY_W5_PERIODIC_TRIM_LIBRARY="$stage/w5-periodic-trim.so"
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
for mode in cache-off cache-on; do
    mkdir "$mode"
    holder_status >"$mode/idle-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
        >"$mode/gpu-check.log" 2>&1 || rc=$?
    holder_status >"$mode/idle-after.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    printf '%d\n' "$rc" >"$mode/exit"
    "$python" -B - "$mode" <<'PY'
from pathlib import Path
import json
import sys

out = Path(sys.argv[1])
before, after = (json.loads((out / f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
fields = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
if not (all(before[key] == after[key] for key in fields) and after["physical_index"] == 6 and after["phase"] == "idle"
        and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3 and not after["other_processes"]):
    raise ValueError("GPU 6 did not restore its existing 20 GiB reservation")
report = json.loads((out / "gpu-evidence/report.json").read_bytes())
if (report["shutdown_exit"] != 0 or report["mixed_soak"]["mode"] != sys.argv[1]
    or int((out / "exit").read_text()) != int(report["status"] == "failed")
    or report["periodic_trim"]["period_ms"] != 31000):
    raise ValueError("the periodic allocator comparison is incomplete")
PY
done
sha256sum cache-{off,on}/{idle-before,idle-after}.json \
    cache-{off,on}/gpu-evidence/{report.json,soak.json,periodic-trim.jsonl,shutdown.json} >result.sha256
