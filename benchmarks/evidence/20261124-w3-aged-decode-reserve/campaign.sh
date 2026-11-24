#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
fixed="$evidence/w3-separated-client-20261124.QAxR7GdV"
prior="$evidence/w3-rag-admission-retry-20261124.OaUPAucl"
waves="$evidence/w3-chunk-pair-20261124.Tl1oBWmS"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c build-dispatch.sha256 campaign-dispatch.sha256
printf '%s  %s\n' \
    0546ad7c648fe2574dce01b6a328f01c3f1a354900d721c599cf73da8678c8a1 "$fixed/runtime.sha256" \
    2b6d5568fd03905c0848a5a33392e4c58da5cbac7bd222810db328ee0de41d97 "$prior/runtime.sha256" \
    b06f829006a7980cad22da03fcf231ec4e7946aec1cd3cdb656cd442f994b122 "$waves/result.sha256" \
    3fd07044efe08cd340611ff2276c631a7f03d183b2b4fa3ace4d9b1b6a726350 "$waves/cells.json" \
    b1e4b523d58842b746044d5d4b9b6a91fe7a2470ab69d9842343228fd27428f8 "$checker/harness/benchmarks/live/load.py" |
    sha256sum --strict -c
(cd "$fixed" && sha256sum --strict -c runtime.sha256 >/dev/null)
(cd "$prior" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c runtime.sha256 >/dev/null)
(cd "$waves" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
mkdir tools gpu6 gpu7
cp -- "$prior"/{arm.py,holder.py,replay.py,client.py} .
cp -- "$prior"/tools/*.py tools/
cp -- "$checker/harness/benchmarks/live/load.py" tools/holdout_load.py
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci" PYTHONDONTWRITEBYTECODE=1
"$runner" -B configure.py "$fixed/schedules.json" "$waves"
"$runner" -B - <<'PY'
import os
from pathlib import Path
from configure import CLIENT_CPUS, SERVER_CPUS
for gpu, cpus in SERVER_CPUS.items():
    cpuset = set(cpus)
    assert cpuset <= os.sched_getaffinity(0)
    for cpu in cpus:
        thread_siblings = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        assert set(map(int,thread_siblings.read_text().split(","))) <= cpuset
    assert not {44, 45, 92, 93} & cpuset
    Path(f"gpu{gpu}/server-cpus.txt").write_text(",".join(map(str, cpus)) + "\n")
assert not set(SERVER_CPUS["6"]) & set(SERVER_CPUS["7"])
assert set(CLIENT_CPUS.values()) <= os.sched_getaffinity(0)
PY
sha256sum arm.py client.py holder.py replay.py tools/*.py schedules.json cpu-placement.json >runtime.sha256

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
(cd src && sha256sum --strict -c ../patched-source.sha256)

reservations() {
    local phase=$1
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$phase.json"
    done
    "$runner" -B - "$phase" "$fixed" <<'PY'
import json, sys
from pathlib import Path
for gpu in (6, 7):
    row = json.loads(Path(f"gpu{gpu}/{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{gpu}/after.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
    assert not row["other_processes"] and all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}

arm() {
    local index=$1 mode=$2 cpus rc=0
    cpus=$(cat "gpu$index/server-cpus.txt")
    printf 'START gpu%s %s %s\n' "$index" "$mode" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" W3_PREFILL_ORDER="$mode" W3_CLIENT_CPU=$((index + 38)) W3_SERVER_CPUS="$cpus" \
        "${control[@]}" --index "$index" run -- taskset -c "$cpus" "$runner" -B arm.py riley >"gpu$index/$mode.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$index" "$mode" "$(date -u +%FT%TZ)" "$rc"
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
if round reserve control; then
    reservations between
    round control reserve || rc=1
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
