#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
previous="$evidence/w4-qkv-forward-retry-20261120.QhDayeS4"
reference="$evidence/w4-projection-crossover-20261120.J9REWUkf"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for origin in "$reference" "$previous"; do
    test "$(cat "$origin/campaign.exit")" = 0
    (cd "$origin" && sha256sum -c result.sha256) >"$(basename "$origin").check"
done
mkdir -p tools target/release
cp -- "$previous"/tools/*.py tools/
cp -- capture_decode_rows.py tools/
cp -- "$previous"/holder.py "$previous"/binary.sha256 .
ln -s -- "$previous/fixtures" fixtures
ln -- "$previous/target/release/riley" target/release/riley
sha256sum -c binary.sha256
"$python" -B tools/capture_decode_rows.py --help >preflight.txt
export PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7 CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RILEY_W4_TRACE_DIR="$stage/riley/rows"
reservations() {
    local point=$1
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json"
    done
    "$python" -B - "$point" <<'PY'
import json
from pathlib import Path
import sys
for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (row["physical_index"], row["pid"], row["start_ticks"], row["phase"]) != (index,pid,ticks,"idle") or \
            row["allocation_bytes"] != 20*1024**3 or row["torch_reserved_bytes"] != 20*1024**3 or row["other_processes"]:
        raise ValueError(f"GPU {index} is missing its original reservation")
PY
}
reservations before
rc=0
printf 'START riley %s\n' "$(date -u +%FT%TZ)"
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    "$python" -B "$stage/tools/run_accuracy.py" riley || rc=$?
reservations after
printf 'END riley %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path("riley/receipt.json").read_bytes())
assert receipt["status"] == "passed" and receipt["server_exit"] == 0
report = json.loads(Path("riley/client.log").read_bytes())
assert len(report["cases"]) == 2
Path("verified.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
PY
sha256sum binary.sha256 verified.json riley/{receipt.json,client.log,server.log} \
    riley/rows/*/{rows.bf16,index.tsv} riley/rows/*/qkv-override.bf16 gpu{6,7}-{before,after}.json >result.sha256
