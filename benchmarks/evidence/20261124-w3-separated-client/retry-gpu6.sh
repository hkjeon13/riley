#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
original="$evidence/w3-separated-client-20261124.QAxR7GdV"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
(cd "$original"
    test "$(cat campaign.exit)" = 1
    grep -Fq 'TimeoutError: server readiness' gpu6/coder14-vllm/client.log
    printf '%s  %s\n' \
        5ddf17228c78283e02a1c03679cb184ea2a5d245a3e1be8e4dbf209ec80c6973 result.sha256 \
        0546ad7c648fe2574dce01b6a328f01c3f1a354900d721c599cf73da8678c8a1 runtime.sha256 |
        sha256sum --strict -c
    sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
printf '%s  %s\n' bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 "$stock/cpu-evidence/report.json" |
    sha256sum --strict -c

mkdir tools gpu6
cp -- "$original"/{arm.py,holder.py,replay.py,schedules.json,cpu-placement.json,server-cpus.txt,vllm-version.txt,runtime.sha256} .
cp -- "$original"/tools/*.py tools/
sha256sum --strict -c runtime.sha256 >/dev/null
export PYTHONPATH="$stage:$stage/tools:$checker/harness/ci:$stock/src/ci"
export PYTHONDONTWRITEBYTECODE=1
export W3_SERVER_CPUS
W3_SERVER_CPUS=$(cat server-cpus.txt)

reservation() {
    "${control[@]}" --index 6 status >"gpu6/$1.json"
    "$runner" -B - "$1" "$original" <<'PY'
import json, sys
from pathlib import Path
row = json.loads(Path(f"gpu6/{sys.argv[1]}.json").read_bytes())
original = json.loads((Path(sys.argv[2]) / "gpu6/after.json").read_bytes())
assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
assert all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}

reservation before
printf 'START gpu6 vllm %s\n' "$(date -u +%FT%TZ)"
rc=0
W3_GPU_INDEX=6 W3_CLIENT_CPU=44 "${control[@]}" --index 6 run -- \
    taskset -c "$W3_SERVER_CPUS" "$runner" -B arm.py vllm >gpu6/vllm.log 2>&1 || rc=$?
reservation after
printf 'END gpu6 vllm %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
find gpu6 -type f \( -name '*.json' -o -name '*.log' \) \
    ! -path '*/vllm-cache/*' ! -path '*/torch-cache/*' -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
