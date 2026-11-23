#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
raw="$evidence/w4-long-answers-20261124.WemA5qyy"
paired="$evidence/w4-long-answers-retry-20261124.cy31E4w3"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
venv="$root/tools/vllm-venv/bin"
runner="$venv/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
fixture_sha=e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3

export PATH="$venv:$PATH"
export PYTHONPATH="$stage:$checker/harness/ci:$stock/src/ci"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
sha256sum --strict -c dispatch.sha256

if [[ ${1:-} == prepare ]]; then
    (cd "$raw" && sha256sum --strict -c dispatch.sha256 >/dev/null)
    (cd "$paired"
        test "$(cat campaign.exit)" = 0
        printf '%s  %s\n' 98079692ffb65c4aa7496b9b9afd0806d672264d0fc927cb4ec06a08779b4216 result.sha256 |
            sha256sum --strict -c
        sha256sum --strict -c result.sha256 launchers.sha256 >/dev/null)
    (cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
    printf '%s  %s\n' \
        d8a18b70ff1130c43fb929ec362b2bc884c8eae50a55dacadb4bbafc05059bf5 "$raw/arm.py" \
        "$fixture_sha" "$raw/facts.json.gz" \
        679aec2b87b87957810ebd6984d48a57d8345802287adb6fe127c753f2a47063 "$raw/long_accuracy.py" \
        d922b8d2ffb473807a19c918c8d2a3e534ee3fd6374e0e1c77cc66f0573f89e7 "$raw/long_answer_check.py" \
        bc10cbc7ffdc0164690f1e7ab1401cf26c50476f556ee7286b69b0e4dc382ae7 "$stock/cpu-evidence/report.json" |
        sha256sum --strict -c
    cp -- "$raw"/{facts.json.gz,long_accuracy.py,long_answer_check.py} .
    "$runner" -B - "$raw/arm.py" <<'PY'
import ast
from pathlib import Path
import sys
source = Path(sys.argv[1]).read_text()
anchor = 'str(STAGE / "long_answer_check.py")'
assert source.count(anchor) == 1
updated = source.replace(anchor, 'str(STAGE / "chat_answers.py")')
ast.parse(updated)
with Path("arm.py").open("x") as stream:
    stream.write(updated)
PY
    test "$(command -v ninja)" = "$venv/ninja"
    ninja --version >ninja-version.txt
    sha256sum "$venv"/{ninja,vllm} >launchers.sha256
    "$runner" -B chat_answers.py prepare "$model" facts.json.gz inputs.json.gz >preflight.json
    "$runner" -B - <<'PY'
from importlib.metadata import version
from arm import CHECKPOINT, MODEL, digest
assert version("vllm") == "0.27.1+cu129"
assert digest(MODEL / "riley-checkpoint.json") == CHECKPOINT
PY
    sha256sum arm.py facts.json.gz long_accuracy.py long_answer_check.py inputs.json.gz \
        preflight.json ninja-version.txt launchers.sha256 >runtime.sha256
    exit
fi

test "${1:-}" = run
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c runtime.sha256 launchers.sha256

reservations() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index-$1.json"
    done
    "$runner" -B - "$1" "$paired" <<'PY'
import json
from pathlib import Path
import sys
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{index}-after.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == 20 * 2**30 and not row["other_processes"]
    assert all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}

reservations before
rc=0
for engine in riley vllm; do
    printf 'START %s %s\n' "$engine" "$(date -u +%FT%TZ)"
    "${control[@]}" --index 7 run -- "$runner" -B arm.py "$engine" "$fixture_sha" >"$engine.log" 2>&1 || rc=$?
    reservations "after-$engine"
    printf 'END %s %s exit=%s\n' "$engine" "$(date -u +%FT%TZ)" "$rc"
    if (( rc != 0 )); then
        break
    fi
done
if (( rc == 0 )); then
    "$runner" -B chat_answers.py report "$stage" >report.json
fi
{
    for file in report.json riley.log vllm.log gpu{6,7}-{before,after-riley,after-vllm}.json; do
        if test -f "$file"; then printf '%s\n' "$file"; fi
    done
    for engine in riley vllm; do
        for file in capture.json receipt.json ready.json shutdown.json checkpoint.log client.log server.log; do
            if test -f "$engine/$file"; then printf '%s\n' "$engine/$file"; fi
        done
    done
} | xargs sha256sum >result.sha256
exit "$rc"
