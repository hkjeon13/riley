#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
rag="$evidence/w3-coder-arrivals-20261124.ac0dqlbz"
c32="$evidence/w3-coder-c32-20261124.n6DRELdL"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
(cd "$rag"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        6de0f89924fd12ad74ca82f57ffecf44e3f73e4c098a16fff03366cc495007e7 runtime.sha256 \
        30eec1c4ca125c5ca9adf44201a8ab52189c88bd92337ab172ecae3e91cbe3d8 dispatch.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c runtime.sha256 dispatch.sha256 >/dev/null)
(cd "$c32"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' e061b5399da890f08d53dd2e322a0d0f706287d4a10932fab99fd0b7b137dde4 schedules.json |
        sha256sum --strict -c)
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        e0817176330af2fd0de977fc82d54d976adf1348650ef17dc0cc78afa390d841 result.sha256 \
        bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 cpu-evidence/report.json |
        sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)

mkdir gpu6 gpu7 tools
cp -- "$rag"/{arm.py,holder.py,replay.py,vllm-version.txt} .
cp -- "$rag"/tools/*.py tools/
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci"
export PYTHONDONTWRITEBYTECODE=1
"$runner" -B configure.py "$rag/schedules.json" "$c32/schedules.json"
"$runner" -B - <<'PY'
import json, os
from importlib.metadata import version
from pathlib import Path
from replay import verify_payloads

assert version("vllm") == "0.27.1+cu129"
siblings = {cpu: Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
            for cpu in (44, 45)}
assert siblings == {44: "44,92", 45: "45,93"}, siblings
available = os.sched_getaffinity(0)
excluded = {44, 45, 92, 93}
assert excluded <= available
placement = {"clients": {"6": 44, "7": 45}, "siblings": siblings, "server_cpus": sorted(available - excluded)}
Path("cpu-placement.json").write_text(json.dumps(placement, indent=2) + "\n")
Path("server-cpus.txt").write_text(",".join(map(str, placement["server_cpus"])) + "\n")
for cells in json.loads(Path("schedules.json").read_bytes())["schedules"].values():
    for plan in cells.values():
        verify_payloads(plan["cell"], plan["schedule"])
PY
export W3_SERVER_CPUS
W3_SERVER_CPUS=$(cat server-cpus.txt)
sha256sum arm.py holder.py replay.py tools/*.py schedules.json cpu-placement.json server-cpus.txt vllm-version.txt >runtime.sha256

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
    "$runner" -B - "$1" <<'PY'
import json, sys
from pathlib import Path
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}/{sys.argv[1]}.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
PY
}

arm() {
    local index=$1 engine=$2 rc=0
    printf 'START gpu%s %s %s\n' "$index" "$engine" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" W3_CLIENT_CPU="$((index + 38))" "${control[@]}" --index "$index" run -- \
        taskset -c "$W3_SERVER_CPUS" "$runner" -B arm.py "$engine" >"gpu$index/$engine.log" 2>&1 || rc=$?
    printf 'END gpu%s %s %s exit=%s\n' "$index" "$engine" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}

round() {
    local left right rc=0
    arm 6 "$1" &
    left=$!
    arm 7 "$2" &
    right=$!
    wait "$left" || rc=1
    wait "$right" || rc=1
    return "$rc"
}

reservations before
rc=0
if round riley vllm; then
    reservations between
    round vllm riley || rc=1
else
    rc=1
fi
reservations after
find gpu6 gpu7 -type f \( -name '*.json' -o -name '*.log' \) \
    ! -path '*/vllm-cache/*' ! -path '*/torch-cache/*' -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
sha256sum runtime.sha256 schedules.json >>result.sha256
exit "$rc"
