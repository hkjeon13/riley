#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
base=$(dirname "$run")
dependency="$base/w23-matrix-20261119.rJfa4QlD"
holder="$base/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
printf 'WAIT for %s %s\n' "$dependency" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 32400))
until test -s "$dependency/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$dependency/campaign.pid")" | grep -Fq -- "$dependency/campaign.sh"; then
        test -s "$dependency/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
export W5_GPU_INDEX=6 PYTHONDONTWRITEBYTECODE=1
printf 'START slow-stream %s\n' "$(date -u +%FT%TZ)"
result=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    /usr/bin/python3 -B "$run/tools/run_slow_stream.py" slow-stream || result=$?
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$run/idle-after.json"
printf 'END slow-stream %s exit=%d\n' "$(date -u +%FT%TZ)" "$result"
exit "$result"
