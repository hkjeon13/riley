#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w3-wide-prefill-order-20261124.cy7XlWYP"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
(cd "$prior"
    test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        92f3c2dbe20a2819ca2b9390300bab755ade2ddfa36d0e2fb51d96d639bc19bb build-result.sha256 \
        6e1d482f164109d85c3993d10b5b9a7c0926b974d8473aa368fa3a4ef7c0f2c1 runtime.sha256 \
        7953b13a5c2fb94e5a6be0804ec0159b691c4dff1878fd1744ee9df09eab26a5 result.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c build-result.sha256 runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
mkdir tools gpu6 gpu7
cp -- "$prior"/{arm.py,client.py,holder.py,replay.py,holdout.py,schedules.json,cpu-placement.json,riley-diagnostic} .
cp -- "$prior"/tools/*.py tools/
cp -- "$prior/gpu6/server-cpus.txt" gpu6/
cp -- "$prior/gpu7/server-cpus.txt" gpu7/
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci" PYTHONDONTWRITEBYTECODE=1
"$runner" -B - <<'PY'
import ast
import json
import os
from pathlib import Path

arm = Path("arm.py")
source = arm.read_text()
anchor = 'out = OUT / os.environ["W3_PREFILL_ORDER"]'
assert source.count(anchor) == 1
source = source.replace(anchor, 'out = OUT / os.environ["W3_PREFILL_LABEL"]')
ast.parse(source)
arm.write_text(source)

placement = json.loads(Path("cpu-placement.json").read_bytes())
for gpu, cpus in placement["server_cpus"].items():
    available = set(cpus)
    assert available <= os.sched_getaffinity(0)
    assert available == set(map(int, Path(f"gpu{gpu}/server-cpus.txt").read_text().strip().split(",")))
    for cpu in cpus:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        assert set(map(int, path.read_text().strip().split(","))) <= available
assert all(cpu in os.sched_getaffinity(0) for cpu in placement["client_cpus"].values())
PY
sha256sum arm.py client.py holder.py replay.py holdout.py tools/*.py schedules.json cpu-placement.json riley-diagnostic >runtime.sha256

reservations() {
    local phase=$1
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$phase.json"
    done
    "$runner" -B - "$phase" "$prior" <<'PY'
import json
from pathlib import Path
import sys

for gpu in (6, 7):
    row = json.loads(Path(f"gpu{gpu}/{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{gpu}/after.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
    assert not row["other_processes"] and all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}

arm() {
    local index=$1 label=$2 mode=control cpus rc=0
    if [[ "$label" == fifo ]]; then mode=fifo; fi
    cpus=$(cat "gpu$index/server-cpus.txt")
    printf 'START gpu%s %s %s\n' "$index" "$label" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" W3_PREFILL_LABEL="$label" W3_PREFILL_ORDER="$mode" W3_CLIENT_CPU=$((index + 38)) \
        W3_SERVER_CPUS="$cpus" "${control[@]}" --index "$index" run -- taskset -c "$cpus" "$runner" -B arm.py riley \
        >"gpu$index/$label.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$index" "$label" "$(date -u +%FT%TZ)" "$rc"
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
if round control-first fifo; then
    reservations between-first
    if round fifo control-first; then
        reservations between-second
        round control-second control-second || rc=1
    else
        rc=1
    fi
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
