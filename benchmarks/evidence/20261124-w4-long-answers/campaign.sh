#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
w3="$evidence/w3-coder-arrivals-20261124.ac0dqlbz"
model="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
fixture_sha=e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3

trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
printf '%s  %s\n' \
    "$fixture_sha" facts.json.gz \
    bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 "$stock/cpu-evidence/report.json" \
    c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 "$model/tokenizer.json" \
    7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c "$model/config.json" \
    3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f "$model/generation_config.json" \
    32fd9124908199541eebb45e23c734806ce95f6de8be3d7b681aecd096c9ade1 "$root/tools/vllm-venv/bin/vllm" |
    sha256sum --strict -c
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
export PYTHONPATH="$stage:$checker/harness/ci:$stock/src/ci"
"$runner" -B - "$fixture_sha" <<'PY'
from importlib.metadata import version
from pathlib import Path
import sys
from long_answer_check import load_fixture
from arm import CHECKPOINT, MODEL, digest
assert version("vllm") == "0.27.1+cu129"
assert digest(MODEL / "riley-checkpoint.json") == CHECKPOINT
assert len(load_fixture(Path("facts.json.gz"), sys.argv[1])) == 33
PY

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index-$1.json"
    done
    "$runner" -B - "$1" <<'PY'
import json
from pathlib import Path
import sys
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
PY
}

pid=$(cat "$w3/campaign.pid")
printf 'WAIT_FOR_W3 %s pid=%s %s\n' "$w3" "$pid" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 7200))
until test -s "$w3/campaign.exit"; do
    if ! test "$(readlink "/proc/$pid/cwd" 2>/dev/null || true)" = "$w3" ||
        ! ps -p "$pid" -o args= | grep -Fxq "bash campaign.sh" || (( SECONDS >= deadline )); then
        test -s "$w3/campaign.exit" && break
        exit 75
    fi
    sleep 20
done
printf 'W3_TERMINAL exit=%s %s\n' "$(cat "$w3/campaign.exit")" "$(date -u +%FT%TZ)"
reservations before
for engine in riley vllm; do
    rc=0
    printf 'START %s %s\n' "$engine" "$(date -u +%FT%TZ)"
    "${control[@]}" --index 7 run -- "$runner" -B arm.py "$engine" "$fixture_sha" >"$engine.log" 2>&1 || rc=$?
    reservations "after-$engine"
    printf 'END %s %s exit=%s\n' "$engine" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$runner" -B long_answer_check.py verify riley/capture.json vllm/capture.json facts.json.gz "$fixture_sha" >report.json
sha256sum report.json facts.json.gz {riley,vllm}/{capture.json,receipt.json} \
    riley/{ready.json,shutdown.json,checkpoint.log} gpu{6,7}-{before,after-riley,after-vllm}.json >result.sha256
