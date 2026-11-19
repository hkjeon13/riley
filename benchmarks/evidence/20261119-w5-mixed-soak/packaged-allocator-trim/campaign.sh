#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
preceding="$evidence/w4-hf-kv-20261119.VwkutnPO"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
sha256sum -c harness.sha256
sha256sum -c compiled.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 25
done
export PYTHONPATH="$stage:$stage/harness/benchmarks/live:$base/src/ci"
export RILEY_W5_TRIM_LIBRARY="$stage/w5-trim-probe.so"
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
holder_status >idle-before.json
printf 'START %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
    "$base/model" "$checkpoint" "$stage/gpu-evidence" --tools --mixed-soak cache-off >gpu-check.log 2>&1 || rc=$?
holder_status >idle-after.json
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
"$python" -B - "$rc" <<'PY'
import json
from pathlib import Path
import sys

before, after = (json.loads(Path(f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
keys = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
if not (all(before[key] == after[key] for key in keys) and after["physical_index"] == 6 and after["phase"] == "idle"
        and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3 and not after["other_processes"]):
    raise ValueError("GPU 6 did not restore the same 20 GiB idle reservation")
report = json.loads(Path("gpu-evidence/report.json").read_bytes())
if (int(sys.argv[1]) not in (0, 1) or report["shutdown_exit"] != 0
    or report["allocator_trim_diagnostic"]["artifact"] != "allocator-trim.json"):
    raise ValueError("the allocator comparison is incomplete")
PY
sha256sum gpu-evidence/{report,soak,allocator-trim,shutdown}.json gpu-evidence/smaps-*.txt.gz \
    idle-{before,after}.json container-smoke.log >result.sha256
