#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
parent="$evidence/w3-c32-rag-history-20261124.sbjdwoF5"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
printf '%s  %s\n' \
    bd65453c70fb2581d2ee5babc952eeac07059aef2a5c13c8c90417dc2eb39b01 "$parent/dispatch.sha256" \
    7ca9dd9fcc543887f3f69a575fd86b432e532a42700b62bc8e8f063ea8925f80 "$parent/runtime.sha256" \
    f7b6fb56eb4189bd2efd2d30e3e000149c6c74db1418f5ee27b25319bdfaa884 "$parent/result.sha256" |
    sha256sum --strict -c
(cd "$parent" && test "$(cat campaign.exit)" = 0 &&
    sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
mkdir -p tools gpu{6,7}/rag
cp "$parent"/{arm.py,client.py,holdout.py,replay.py,holder.py,schedules.json,cpu-placement.json} .
cp "$parent"/tools/*.py tools/
sha256sum arm.py client.py holdout.py replay.py holder.py schedules.json cpu-placement.json tools/*.py >runtime.sha256
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci" PYTHONDONTWRITEBYTECODE=1

pid=$(cat build.pid)
deadline=$((SECONDS + 1800))
until test -s build.exit; do
    if [[ $(readlink "/proc/$pid/cwd" 2>/dev/null || true) != "$stage" ]] ||
        ! ps -p "$pid" -o args= | grep -Fxq 'bash build.sh' || (( SECONDS >= deadline )); then
        test -s build.exit && break
        exit 75
    fi
    sleep 8
done
test "$(cat build.exit)" = 0
sha256sum --strict -c build-result.sha256 runtime.sha256

reservations() {
    local phase=$1
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$phase.json"
    done
    "$runner" -B - "$phase" "$parent" <<'PY'
import json
from pathlib import Path
import sys

for gpu in (6, 7):
    row = json.loads(Path(f"gpu{gpu}/{sys.argv[1]}.json").read_bytes())
    prior = json.loads((Path(sys.argv[2]) / f"gpu{gpu}/after.json").read_bytes())
    if (row["phase"] != "idle" or row["other_processes"] or
            row["allocation_bytes"] != 20 * 2**30 or row["torch_reserved_bytes"] != 20 * 2**30 or
            any(row[key] != prior[key] for key in ("gpu", "physical_index", "pid", "start_ticks"))):
        raise RuntimeError(f"original reservation is unavailable on GPU {gpu}")
PY
}

arm() {
    local index=$1 history=$2 mode=$3 cpus rc=0
    case "$index" in
        6) cpus=24-33,72-81 ;;
        7) cpus=34-43,82-91 ;;
        *) return 2 ;;
    esac
    printf 'START gpu%s %s %s %s\n' "$index" "$history" "$mode" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" W3_RAG_HISTORY="$history" W3_PREFILL_ORDER="$mode" \
        W3_CLIENT_CPU=$((index + 38)) W3_SERVER_CPUS="$cpus" \
        "${control[@]}" --index "$index" run -- taskset -c "$cpus" "$runner" -B arm.py riley \
        >"gpu$index/$history/$mode.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s %s exit=%s\n' "$index" "$history" "$mode" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}

round() {
    local history=$1 first second rc=0
    arm 6 "$history" "$2" & first=$!
    arm 7 "$history" "$3" & second=$!
    wait "$first" || rc=1
    wait "$second" || rc=1
    return "$rc"
}

rc=0
reservations before
if round rag reserve control; then
    reservations between
    round rag control reserve || rc=1
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
