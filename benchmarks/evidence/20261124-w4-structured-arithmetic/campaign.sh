#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w4-arithmetic-controls-20261124.A9VIzLu6"
stock="$evidence/w5-shard-package-20261124.nhBcrYgP"
checker="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
runner="$root/tools/vllm-venv/bin/python"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
printf '%s  %s\n' \
    d32ba0494fce90f0d704e00d0ed3fe7a60b5dbea2c44a9439c0d2a90247be1e6 "$prior/runtime.sha256" \
    b840ce0a3b47cc2ea89921c023a1e0f3a18fde7813b5444869142de8a2b43082 "$prior/result.sha256" |
    sha256sum --strict -c
(cd "$prior" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >/dev/null)
(cd "$checker" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
cp -- "$prior"/{arm.py,facts.json.gz,long_answer_check.py,long_accuracy.py,chat_answers.py,prepare_arithmetic.py} .
export PYTHONPATH="$stage:$checker/harness/ci:$stock/src/ci"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
"$runner" -B - <<'PY'
import ast
from pathlib import Path

def edit(path, replacements):
    source = path.read_text()
    for before, after in replacements:
        assert source.count(before) == 1
        source = source.replace(before, after)
    ast.parse(source)
    path.write_text(source)

edit(Path("arm.py"), [('str(STAGE / "arithmetic_capture.py")', 'str(STAGE / "structured_arithmetic.py")')])
edit(Path("chat_answers.py"), [
    ('def run_chat(address, model, case, variant):',
     'def run_chat(address, model, case, variant, *, response_format=None):'),
    ('    request = Request(f"http://{address}/v1/chat/completions", json.dumps(body).encode(),',
     '    if response_format is not None:\n        body["response_format"] = response_format\n'
     '    request = Request(f"http://{address}/v1/chat/completions", json.dumps(body).encode(),'),
])
PY
"$runner" -B structured_arithmetic.py prepare "$model" facts.json.gz inputs.json >preflight.json
sha256sum arm.py facts.json.gz long_answer_check.py long_accuracy.py chat_answers.py prepare_arithmetic.py \
    inputs.json preflight.json >runtime.sha256

reservations() {
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
    assert not row["other_processes"] and all(row[key] == original[key] for key in ("pid", "start_ticks", "gpu"))
PY
}
reservations before
printf 'START riley %s\n' "$(date -u +%FT%TZ)"
rc=0
"${control[@]}" --index 7 run -- "$runner" -B arm.py riley \
    e68ed38a73caad88ae8de0a0000a439f732331a9b118849de28d92d4366efda3 >riley.log 2>&1 || rc=$?
reservations after
printf 'END riley %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
if (( rc == 0 )); then "$runner" -B structured_arithmetic.py report "$stage" >report.json; fi
find . -maxdepth 2 -type f \( -name '*.json' -o -name '*.log' \) ! -name campaign.log -print0 |
    sort -z | xargs -0 sha256sum >result.sha256
exit "$rc"
