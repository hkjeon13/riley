#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
root=$(cd -- "$run/../.." && pwd -P)
hold="$root/evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
dependencies=(
  w5-mixed-soak-20261119.YJg0Xzhk
  w4-holdout-20261119.cgmtjz4x
  w4-layer-trace-20261119.1gr9Rppl
  w3-rag-profile-20261119.FE1ZUFeE
)
for key in "${dependencies[@]}"; do
    dependency="$root/evidence/$key"
    printf 'WAITING FOR %s %s\n' "$dependency/campaign.exit" "$(date -u +%FT%TZ)"
    for ((attempt=1; attempt<=480; attempt++)); do
        if test -s "$dependency/campaign.exit"; then break; fi
        if ! ps -o args= -p "$(cat "$dependency/campaign.pid")" | grep -Fq -- "$dependency/campaign.sh"; then
            test -s "$dependency/campaign.exit" && break
            printf 'Missing campaign %s\n' "$dependency" >&2
            exit 75
        fi
        sleep 15
    done
    test -s "$dependency/campaign.exit"
done
export PYTHONDONTWRITEBYTECODE=1
export W4_GPU_INDEX=7
failures=0
for label in qwen7-riley qwen7-vllm llama3-vllm llama3-riley qwen4-riley qwen4-vllm qwen3-vllm qwen3-riley; do
    printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
        /usr/bin/python3 -B "$run/tools/run_gsm8k.py" "${label%-*}" "${label##*-}" || rc=$?
    if test -d "$run/$label"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status >"$run/$label/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$label" "$(date -u +%FT%TZ)" "$rc"
    if test "$rc" -ne 0; then failures=1; fi
done
/usr/bin/python3 -B - "$run" <<'PYTHON' || failures=1
from hashlib import sha256
import json
from pathlib import Path
import subprocess,sys

root=Path(sys.argv[1])
failed=False
for model in ("qwen7","llama3","qwen4","qwen3"):
    paths=[root/(model+"-"+role) for role in ("vllm","riley")]
    try:
        for path in paths:
            receipt=json.loads((path/"receipt.json").read_bytes())
            raw=(path/"result.json").read_bytes()
            if receipt["status"]!="passed" or receipt["result_sha256"]!=sha256(raw).hexdigest():
                raise ValueError("invalid arm receipt")
    except (OSError,ValueError,KeyError) as error:
        print(model,error,flush=True)
        failed=True
        continue
    command=[sys.executable,"-B",str(root/"tools/compare_gsm8k.py"),str(root/"dataset.json")]
    command.extend(str(path/"result.json") for path in paths)
    command.append(str(root/f"{model}-comparison.json"))
    print("COMPARE",model,flush=True)
    if subprocess.run(command).returncode:
        failed=True
raise SystemExit(int(failed))
PYTHON
exit "$failures"
