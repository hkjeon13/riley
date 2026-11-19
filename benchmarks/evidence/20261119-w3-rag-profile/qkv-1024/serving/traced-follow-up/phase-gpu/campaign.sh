#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$run"
evidence=$(dirname "$run")
root=$(dirname "$evidence")
original="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
sse="$evidence/w5-chat-sse-20261119.T7Z99RAG"
accuracy="$evidence/w5-chat-sse-accuracy-20261119.BginwHRs"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
report_python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
sha256sum -c used.sha256
test "$(cat "$original/campaign.exit")" = 0

wait_for_terminal() {
    local stage=$1 deadline=$((SECONDS + 14400))
    printf 'WAIT %s %s\n' "$stage" "$(date -u +%FT%TZ)"
    until test -s "$stage/campaign.exit"; do
        if ! ps -o args= -p "$(cat "$stage/campaign.pid")" | grep -Fq -- "$stage/campaign.sh"; then
            test -s "$stage/campaign.exit" && break
            return 75
        fi
        if (( SECONDS >= deadline )); then return 75; fi
        sleep 20
    done
    printf 'TERMINAL %s %s\n' "$stage" "$(cat "$stage/campaign.exit")"
}
wait_for_terminal "$sse"
wait_for_terminal "$accuracy"

holder_status() {
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}

for mode in riley-first qkv-first qkv-second riley-second; do
    holder_status 6 >"gpu6-before-$mode.json"
    holder_status 7 >"idle-before-$mode.json"
    "$report_python" -B - "$mode" <<'PY'
from pathlib import Path
import json
import sys

mode = sys.argv[1]
for index, pid, file in ((6, 2431228, f"gpu6-before-{mode}.json"), (7, 2431237, f"idle-before-{mode}.json")):
    status = json.loads(Path(file).read_bytes())
    if (status["phase"] != "idle" or status["physical_index"] != index or status["pid"] != pid
            or status["other_processes"] or status["allocation_bytes"] != status["torch_reserved_bytes"]
            or status["allocation_bytes"] != 20 * 1024**3):
        raise ValueError(f"GPU {index} does not have its original idle reservation")
PY
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    status=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$run/tools/run_profile_retry.py" qwen14 "$mode" || status=$?
    holder_status 6 >"gpu6-after-$mode.json"
    holder_status 7 >"qwen14-$mode/idle-after.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then exit "$status"; fi
done
