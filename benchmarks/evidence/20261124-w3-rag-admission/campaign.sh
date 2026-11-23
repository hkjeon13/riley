#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
baseline="$evidence/w3-separated-client-20261124.QAxR7GdV"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c campaign-dispatch.sha256 build-dispatch.sha256
(cd "$baseline"
    test "$(cat campaign.exit)" = 1
    printf '%s  %s\n' \
        5ddf17228c78283e02a1c03679cb184ea2a5d245a3e1be8e4dbf209ec80c6973 result.sha256 \
        0546ad7c648fe2574dce01b6a328f01c3f1a354900d721c599cf73da8678c8a1 runtime.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
printf '%s  %s\n' bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 "$stock/cpu-evidence/report.json" |
    sha256sum --strict -c

mkdir tools gpu7
cp -- "$baseline"/{arm.py,holder.py,replay.py,client.py,schedules.json,cpu-placement.json,server-cpus.txt} .
cp -- "$baseline"/tools/*.py tools/
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci"
export PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7 W3_CLIENT_CPU=45 W3_SERVER_CPUS
W3_SERVER_CPUS=$(cat server-cpus.txt)
"$runner" -B configure.py
"$runner" -B - <<'PY'
import ast, json, os
from pathlib import Path
from replay import verify_payloads

placement = json.loads(Path("cpu-placement.json").read_bytes())
assert sorted(os.sched_getaffinity(0)-{44,45,92,93}) == placement["server_cpus"]
assert Path("/sys/devices/system/cpu/cpu45/topology/thread_siblings_list").read_text().strip() == "45,93"
for path in [Path("arm.py"), Path("replay.py"), Path("tools/load.py")]:
    ast.parse(path.read_text())
for plan in json.loads(Path("schedules.json").read_bytes())["schedules"]["7"].values():
    assert len(verify_payloads(plan["cell"], plan["schedule"])) == 72
PY
sha256sum arm.py client.py holder.py replay.py tools/*.py schedules.json cpu-placement.json server-cpus.txt >runtime.sha256

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

reservation() {
    "${control[@]}" --index 7 status >"gpu7/$1.json"
    "$runner" -B - "$1" "$baseline" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(f"gpu7/{sys.argv[1]}.json").read_bytes())
original = json.loads((Path(sys.argv[2]) / "gpu7/after.json").read_bytes())
assert row["phase"] == "idle" and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
assert not row["other_processes"] and all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}

reservation before
printf 'START gpu7 riley %s\n' "$(date -u +%FT%TZ)"
rc=0
"${control[@]}" --index 7 run -- taskset -c "$W3_SERVER_CPUS" "$runner" -B arm.py riley >gpu7/riley.log 2>&1 || rc=$?
reservation after
printf 'END gpu7 riley %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
find gpu7 -type f -print0 | sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
