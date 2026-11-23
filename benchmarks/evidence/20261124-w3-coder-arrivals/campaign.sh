#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source="$evidence/w3-package-coder14-20261124.GgTQVni3"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
w5="$evidence/w5-shard-soak-20261124.ZP737OQY"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$source"
    test "$(cat campaign.exit)" = 0
    printf 'fea2613b465f02faefb623f8e421cd384eecc8ab48cdf083b3d57b9ce6e0f2e4  dispatch.sha256\n4c4555e53030ff54d771bddcf100dd249d3f23e59ef30937a7d64a7c238f52bf  result.sha256\n' | sha256sum --strict -c
    sha256sum --strict -c dispatch.sha256 result.sha256 runtime.sha256 first-attempt.sha256 >/dev/null)
(cd "$root" && sha256sum --strict -c "$source/pins.sha256")
(cd "$stock"
    test "$(cat campaign.exit)" = 0
    printf 'e0817176330af2fd0de977fc82d54d976adf1348650ef17dc0cc78afa390d841  result.sha256\nbc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7  cpu-evidence/report.json\n' | sha256sum --strict -c
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
test "$(cat "$source/vllm-version.txt")" = 0.27.1+cu129
"$runner" -B -c 'from importlib.metadata import version; assert version("vllm")=="0.27.1+cu129"; print(version("vllm"))' >vllm-version.txt
mkdir tools gpu6 gpu7
cp -- "$source"/{holder.py,arm.py} .
cp -- "$source"/tools/*.py tools/
"$runner" -B freeze-arrivals.py "$source" >schedules.json
sha256sum --strict -c schedule.sha256
"$runner" -B - <<'PYTHON'
from pathlib import Path

arm = Path("arm.py")
data = arm.read_text()
changes = {
    '"cells.json").read_bytes())': '"schedules.json").read_bytes())["schedules"][os.environ["W3_GPU_INDEX"]]',
    '3278fe2fced2c3edadb2a9348c46a135fdcfd81da5f72ebfc9a55dede0ccfa42':
        '1729f15a677422d552b9a3fd2352dfeb08dd4cccd3558680a016058d27191098',
    'w4-schema-ids-20261123.ZsUYnqkm': 'w5-shard-package-20261124.nhBcrYgP',
    'c5ec7b3918e18428b0c5704b010aa4ee51972a86': '0511b9280f62c3ee87c7a1344444423d8b4623c4',
}
for old, new in changes.items():
    assert data.count(old) == 1
    data = data.replace(old, new)
arm.write_text(data)
workload = Path("tools/profile_workload.py")
data = workload.read_text()
begin = data.index("async def load_cell(")
end = data.index("\ndef run(", begin)
data = data[:begin] + data[end + 1:]
data = data.replace("from load import one, prompt, summarize", "from replay import load_cell")
workload.write_text(data)
PYTHON
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci"
export PYTHONDONTWRITEBYTECODE=1
"$runner" -B - <<'PYTHON'
from pathlib import Path
import asyncio, json
from replay import verify_payloads
from profile_workload import load_cell
import check_distribution_gpu

plans = json.loads(Path("schedules.json").read_bytes())["schedules"]
assert set(plans) == {"6", "7"} and all(len(rows) == 4 for rows in plans.values())
assert asyncio.iscoroutinefunction(load_cell) and load_cell.__module__ == "replay"
for cells in plans.values():
    for case in cells.values():
        verify_payloads(case["cell"], case["schedule"])
PYTHON
sha256sum arm.py holder.py tools/*.py schedules.json vllm-version.txt >runtime.sha256

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index/$1.json"
    done
    "$runner" -B - "$1" <<'PYTHON'
import json, sys
from pathlib import Path
for index in (6, 7):
    path = Path(f"gpu{index}/{sys.argv[1]}.json")
    status = json.loads(path.read_bytes())
    assert status["phase"] == "idle" and status["allocation_bytes"] == 20 * 2**30 and not status["other_processes"]
PYTHON
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

pid=$(cat "$w5/campaign.pid")
printf 'WAIT_FOR_W5 %s pid=%s %s\n' "$w5" "$pid" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 7200))
until test -s "$w5/campaign.exit"; do
    if ! test "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$w5" ||
        ! ps -p "$pid" -o args= | grep -Fxq "bash campaign.sh" ||
        (( SECONDS >= deadline )); then
        test -s "$w5/campaign.exit" && break
        exit 75
    fi
    sleep 20
done
printf 'W5_TERMINAL exit=%s %s\n' "$(cat "$w5/campaign.exit")" "$(date -u +%FT%TZ)"
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
sha256sum schedules.json runtime.sha256 >>result.sha256
exit "$rc"
