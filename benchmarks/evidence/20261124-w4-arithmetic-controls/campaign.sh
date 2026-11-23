#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w4-chat-answers-20261124.PPoslVc8"
w3="$evidence/w3-rag-prefill-order-retry-20261124.cWPk8nNQ"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
venv="$root/tools/vllm-venv/bin"
runner="$venv/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
printf '%s  %s\n' \
    a7b2dbca08cdbc50477665cd930d4e21c2b8f0bd0f32df4e904599c18a2557de inputs.json \
    a43ac41cda9bd985f9d5669b650d714233aa663a46b14f7549613457c62625c1 "$prior/runtime.sha256" \
    3192bccd62ea251c33e55cfd5423419cde410b9feb5ab0d2cccf09b53689823d "$prior/result.sha256" |
    sha256sum --strict -c
(cd "$prior" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c runtime.sha256 result.sha256 launchers.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -- "$prior"/{arm.py,facts.json.gz,long_answer_check.py,long_accuracy.py,chat_answers.py,launchers.sha256,ninja-version.txt} .
export PATH="$venv:$PATH"
export PYTHONPATH="$stage:$checker/harness/ci:$stock/src/ci"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$runner" -B - <<'PY'
import ast
from pathlib import Path
from arithmetic_capture import workload
path = Path("arm.py")
source = path.read_text()
anchor = 'str(STAGE / "chat_answers.py")'
assert source.count(anchor) == 1
updated = source.replace(anchor, 'str(STAGE / "arithmetic_capture.py")')
ast.parse(updated)
path.write_text(updated)
assert len(workload(Path.cwd())) == 24
PY
sha256sum arm.py facts.json.gz long_answer_check.py long_accuracy.py chat_answers.py inputs.json \
    launchers.sha256 ninja-version.txt >runtime.sha256

pid=$(cat "$w3/campaign.pid")
printf 'WAIT_FOR_W3 pid=%s %s\n' "$pid" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 3600))
until test -s "$w3/campaign.exit"; do
    if [[ $(readlink "/proc/$pid/cwd" 2>/dev/null || true) != "$w3" ]] ||
        ! ps -p "$pid" -o args= | grep -Fxq 'bash campaign.sh' || (( SECONDS >= deadline )); then
        test -s "$w3/campaign.exit" && break
        exit 75
    fi
    sleep 10
done
test "$(cat "$w3/campaign.exit")" = 0
(cd "$w3" && sha256sum --strict -c result.sha256 >/dev/null)
printf 'W3_TERMINAL exit=0 %s\n' "$(date -u +%FT%TZ)"

reservation() {
    for index in 6 7; do
        "${control[@]}" --index "$index" status >"gpu$index-$1.json"
    done
    "$runner" -B - "$1" "$prior" <<'PY'
import json, sys
from pathlib import Path
for index in (6, 7):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    original = json.loads((Path(sys.argv[2]) / f"gpu{index}-after-vllm.json").read_bytes())
    assert row["phase"] == "idle" and row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 2**30
    assert not row["other_processes"] and all(row[k] == original[k] for k in ("pid", "start_ticks", "gpu"))
PY
}
reservation before
rc=0
for engine in riley vllm; do
    printf 'START %s %s\n' "$engine" "$(date -u +%FT%TZ)"
    "${control[@]}" --index 7 run -- "$runner" -B arm.py "$engine" \
        e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3 >"$engine.log" 2>&1 || rc=$?
    reservation "after-$engine"
    printf 'END %s %s exit=%s\n' "$engine" "$(date -u +%FT%TZ)" "$rc"
    if (( rc != 0 )); then break; fi
done
if (( rc == 0 )); then "$runner" -B arithmetic_capture.py report "$stage" >report.json; fi
find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.log' \) ! -name campaign.log -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
