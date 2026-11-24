#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
parent="$evidence/w3-aged-decode-native-20261124.tNZ3fCMo"
stock="$evidence/w4-pointer-syntax-20261124.HgwYVqCf"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256 campaign-dispatch.sha256
test "$(cat "$parent/campaign.exit")" = 0
test "$(cat "$stock/campaign.exit")" = 0
printf '%s  %s\n' \
    7921e63069ddd4b58736558af62590b5f340083ee129d25aa54eeefbdf78c116 "$parent/schedules.json" \
    1d7f89a4228ed63ab0c6a1b0fc9dd2fe36169e4d8aff1b1047c8de3b6583184d "$parent/cpu-placement.json" \
    882628f9d5a5b813487cfa6dc300d211549ee92bf3fbd91f8c080fb5303d7d56 "$parent/holdout.py" \
    a077c1197de5ab736eeb6d4ba087347223ce867cad3aca318fe1275d5d7e13b7 "$parent/runtime.sha256" \
    874f5af7293eb924741fc2e9d652cd40eeeca1b5d1080c1c33b05b83450ad94b "$parent/result.sha256" \
    8436c243060357615f9ac5499edd98ba404a079e44d55a8df0f76bcf9ce0b6d8 "$stock/cpu-evidence/report.json" |
    sha256sum --strict -c
(cd "$parent" && sha256sum --strict -c runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
mkdir -p gpu6 gpu7 tools
cp -- "$parent"/{arm.py,client.py,holder.py,replay.py,holdout.py,schedules.json,cpu-placement.json} .
cp -- "$parent"/tools/*.py tools/
"$runner" -B prepare.py
sha256sum arm.py client.py holder.py replay.py holdout.py schedules.json cpu-placement.json tools/*.py >runtime.sha256
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci" PYTHONDONTWRITEBYTECODE=1

pid=$(cat build.pid)
printf 'WAIT_FOR_BUILD pid=%s %s\n' "$pid" "$(date -u +%FT%TZ)"
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
    for gpu in 6 7; do
        "${control[@]}" --index "$gpu" status >"gpu$gpu/$phase.json"
    done
    "$runner" -B - "$phase" "$parent" <<'PY'
import json
from pathlib import Path
import sys

expected = {6: (2680326, "6365879881"), 7: (2431237, "6339570875")}
for gpu, (pid, ticks) in expected.items():
    actual = json.loads(Path(f"gpu{gpu}/{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{gpu}/after.json").read_bytes())
    if (actual["physical_index"] != gpu or actual["pid"] != pid or str(actual["start_ticks"]) != ticks
            or actual["gpu"] != original["gpu"] or actual["phase"] != "idle"
            or actual["other_processes"] or actual["allocation_bytes"] != 20 * 2**30
            or actual["torch_reserved_bytes"] != 20 * 2**30):
        raise RuntimeError(f"original hold on GPU {gpu} is unavailable")
PY
}

arm() {
    local gpu=$1 mode=$2 cpus rc=0
    cpus=$(cat "$parent/gpu$gpu/server-cpus.txt")
    printf 'START gpu%s %s %s\n' "$gpu" "$mode" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$gpu" W3_PREFILL_ORDER="$mode" W3_CLIENT_CPU=$((gpu + 38)) W3_SERVER_CPUS="$cpus" \
        "${control[@]}" --index "$gpu" run -- taskset -c "$cpus" "$runner" -B arm.py riley \
        >"gpu$gpu/$mode.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$gpu" "$mode" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}

round() {
    local first second rc=0
    arm 6 "$1" & first=$!
    arm 7 "$2" & second=$!
    wait "$first" || rc=1
    wait "$second" || rc=1
    return "$rc"
}

reservations before
rc=0
if round control candidate; then
    reservations between
    round candidate control || rc=1
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
