#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source="$evidence/w3-qkv-serving-20261119.MhydzNsX"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
export PYTHONPATH="$stage/tools" CUDA_DEVICE_ORDER=PCI_BUS_ID
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$source/campaign.exit")" = 0
(cd "$source" && sha256sum -c "$stage/source-tools.sha256" binary.sha256) >predecessor-check.log
mkdir tools
cp -- "$source/tools/"{run_profile_retry,profile_workload,load,profile_process,profile_summary,compare_runner}.py tools/
cp -- "$source/holder.py" .
cp -- fixed_w3.py fixed_runner.py tools/
"$python" -B - "$source" <<'PREFLIGHT' >preflight.json
import json
from pathlib import Path
import sys

from fixed_w3 import cell_specs

path = Path("tools/profile_workload.py")
original = path.read_text()
needle = 'if __name__ == "__main__":'
if original.count(needle) != 1:
    raise ValueError("unexpected workload entry point")
path.write_text(original.replace(needle, "from fixed_w3 import load_cell\n\n\n" + needle))
print(json.dumps(cell_specs(Path(sys.argv[1]), "riley-first"), indent=2, allow_nan=False))
PREFLIGHT
reservations() {
    local point=$1
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"$folder/gpu$index-$point.json"
    done
    /usr/bin/python3 -B - "$point" "$folder" <<'PY'
import json
from pathlib import Path
import sys

for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    row = json.loads((Path(sys.argv[2]) / f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (row["physical_index"], row["pid"], row["start_ticks"], row["phase"]) != (index, pid, ticks, "idle") or \
            row["allocation_bytes"] != 20 * 1024**3 or row["torch_reserved_bytes"] != 20 * 1024**3 or row["other_processes"]:
        raise ValueError(f"GPU {index} is missing its original 20 GiB reservation")
PY
}


for mode in riley-first qkv-first qkv-second riley-second; do
    before="$stage/before-$mode"
    mkdir "$before"
    folder=$before
    reservations before
    rc=0
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
         "$python" -B tools/fixed_runner.py "$source" "$mode" || rc=$?
    folder="$stage/qwen14-$mode"
    mkdir -p "$folder"
    reservations after
    mv -- "$before/"gpu{6,7}-before.json "$folder/"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_fixed.py "$stage" "$source" >verified.json
sha256sum verified.json preflight.json tools/profile_workload.py qwen14-*/{receipt,token-id-probe,plan,server-exit,gpu6-before,gpu6-after,gpu7-before,gpu7-after,rag-c8-*}.json >result.sha256
