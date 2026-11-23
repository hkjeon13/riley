#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
rag="$evidence/w3-coder-arrivals-20261124.ac0dqlbz"
w4="$evidence/w4-long-answers-20261124.WemA5qyy"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
(cd "$rag"
    printf '%s  %s\n' \
        6de0f89924fd12ad74ca82f57ffecf44e3f73e4c098a16fff03366cc495007e7 runtime.sha256 \
        30eec1c4ca125c5ca9adf44201a8ab52189c88bd92337ab172ecae3e91cbe3d8 dispatch.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c runtime.sha256 dispatch.sha256 >/dev/null)
mkdir gpu6 gpu7 tools
cp -- "$rag"/{arm.py,holder.py,vllm-version.txt} .
cp -- "$rag"/tools/*.py tools/
gzip -cd schedules.json.gz >schedules.json
printf '%s  %s\n' e061b5399da890f08d53dd2e322a0d0f706287d4a10932fab99fd0b7b137dde4 schedules.json |
    sha256sum --strict -c
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf '%s  %s\n' \
        e0817176330af2fd0de977fc82d54d976adf1348650ef17dc0cc78afa390d841 result.sha256 \
        bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 cpu-evidence/report.json |
        sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci"
export PYTHONDONTWRITEBYTECODE=1
"$runner" -B - "$rag/replay.py" <<'PY'
from pathlib import Path
import sys

original = Path(sys.argv[1]).read_text()
before = '"scheduled_write_ns": original["offset_ns"], "write_error_ms": error_ms})'
after = '"scheduled_write_ns": original["offset_ns"], "write_error_ms": error_ms,\n'
after += '                        "content_gaps_s": row["gaps"]})'
assert original.count(before) == 1
Path("replay.py").write_text(original.replace(before, after))
PY
"$runner" -B - <<'PY'
from importlib.metadata import version
from pathlib import Path
import json
from replay import verify_payloads
from profile_workload import load_cell

assert version("vllm") == "0.27.1+cu129"
assert load_cell.__module__ == "replay"
cells = json.loads(Path("schedules.json").read_bytes())["schedules"]
assert set(cells) == {"6", "7"}
for plans in cells.values():
    assert set(plans) == {"c32-riley-arrivals", "c32-vllm-arrivals"}
    for plan in plans.values():
        assert len(plan["schedule"]) == 192
        verify_payloads(plan["cell"], plan["schedule"])
PY
sha256sum arm.py holder.py replay.py tools/*.py schedules.json vllm-version.txt >runtime.sha256

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
    "$runner" -B - "$1" <<'PY'
import json
from pathlib import Path
import sys
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}/{sys.argv[1]}.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
PY
}

arm() {
    local index=$1 engine=$2 rc=0
    printf 'START gpu%s %s %s\n' "$index" "$engine" "$(date -u +%FT%TZ)"
    W3_GPU_INDEX="$index" "${control[@]}" --index "$index" run -- \
        "$runner" -B arm.py "$engine" >"gpu$index/$engine.log" 2>&1 || rc=$?
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

pid=$(cat "$w4/campaign.pid")
printf 'WAIT_FOR_W4 %s pid=%s %s\n' "$w4" "$pid" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 10800))
until test -s "$w4/campaign.exit"; do
    if ! test "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$w4" ||
        ! ps -p "$pid" -o args= | grep -Fxq "bash campaign.sh" || (( SECONDS >= deadline )); then
        test -s "$w4/campaign.exit" && break
        exit 75
    fi
    sleep 20
done
printf 'W4_TERMINAL exit=%s %s\n' "$(cat "$w4/campaign.exit")" "$(date -u +%FT%TZ)"
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
sha256sum schedules.json replay.py runtime.sha256 >>result.sha256
exit "$rc"
