#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
first="$evidence/w4-long-answers-20261124.WemA5qyy"
c32="$evidence/w3-coder-c32-20261124.n6DRELdL"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
venv="$root/tools/vllm-venv/bin"
runner="$venv/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
fixture_sha=e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT

sha256sum --strict -c dispatch.sha256
(cd "$first"
    test "$(cat campaign.exit)" = 1
    grep -Fq "No such file or directory: 'ninja'" vllm/server.log
    printf '%s  %s\n' \
        3c5a1572108ed2e74e6500e6083cd1b2085e9b9bbb4c3a7c656c3434cbbcc58b dispatch.sha256 \
        fd1daa0ad4945c225cad455283221329fce1b9a11f7e46a6f856c93ce0b15082 riley/capture.json \
        43529941faaafff0350e1a622d4dad47276cea7d99fd20a5ec4295061212ad79 riley/receipt.json \
        adc690176f8a3a26e3a6a7e25cc1be284ba9e58ec0858aecf6c95ea754f1116c gpu7-after-vllm.json |
        sha256sum --strict -c
    sha256sum --strict -c dispatch.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -- "$first"/{arm.py,facts.json.gz,long_accuracy.py,long_answer_check.py} .
mkdir riley
cp -- "$first/riley"/{capture.json,receipt.json,ready.json,shutdown.json,checkpoint.log} riley/
export PATH="$venv:$PATH"
export PYTHONPATH="$stage:$checker/harness/ci:$stock/src/ci"
test "$(command -v ninja)" = "$venv/ninja"
ninja --version >ninja-version.txt
sha256sum "$venv"/{ninja,vllm} >launchers.sha256
"$runner" -B - "$fixture_sha" <<'PY'
from importlib.metadata import version
from pathlib import Path
import json, sys
from arm import CHECKPOINT, CPU, MODEL, digest
from long_answer_check import load_fixture

fixture = sys.argv[1]
assert version("vllm") == "0.27.1+cu129"
assert digest(CPU) == "bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7"
assert digest(MODEL / "riley-checkpoint.json") == CHECKPOINT
assert len(load_fixture(Path("facts.json.gz"), fixture)) == 33
receipt = json.loads(Path("riley/receipt.json").read_bytes())
assert receipt["shutdown_exit"] == 0 and receipt["fixture_sha256"] == fixture
assert receipt["checkpoint_sha256"] == CHECKPOINT and receipt["capture_sha256"] == digest(Path("riley/capture.json"))
PY

pid=$(cat "$c32/campaign.pid")
printf 'WAIT_FOR_C32 %s pid=%s %s\n' "$c32" "$pid" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 5400))
until test -s "$c32/campaign.exit"; do
    if ! test "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$c32" ||
        ! ps -p "$pid" -o args= | grep -Fxq "bash campaign.sh" || (( SECONDS >= deadline )); then
        test -s "$c32/campaign.exit" && break
        exit 75
    fi
    sleep 20
done
printf 'C32_TERMINAL exit=%s %s\n' "$(cat "$c32/campaign.exit")" "$(date -u +%FT%TZ)"

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index-$1.json"
    done
    "$runner" -B - "$1" "$first" <<'PY'
import json, sys
from pathlib import Path
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{index}-after-vllm.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
    assert all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}
reservations before
printf 'START vllm %s\n' "$(date -u +%FT%TZ)"
rc=0
"${control[@]}" --index 7 run -- "$runner" -B arm.py vllm "$fixture_sha" >vllm.log 2>&1 || rc=$?
reservations after
printf 'END vllm %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$runner" -B long_answer_check.py verify riley/capture.json vllm/capture.json facts.json.gz "$fixture_sha" >report.json
sha256sum report.json facts.json.gz launchers.sha256 ninja-version.txt \
    {riley,vllm}/{capture.json,receipt.json} riley/{ready.json,shutdown.json,checkpoint.log} \
    gpu{6,7}-{before,after}.json >result.sha256
