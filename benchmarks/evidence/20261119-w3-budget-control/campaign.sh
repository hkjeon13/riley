#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
base=$(dirname "$run")
preceding="$base/w5-slow-stream-20261119.Ewkyq7yF"
holder="$base/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
printf 'WAIT for %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
result=0
export PYTHONDONTWRITEBYTECODE=1
for budget in 1024 2048; do
    printf 'START vllm batch limit %s %s\n' "$budget" "$(date -u +%FT%TZ)"
    status=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        /usr/bin/python3 -B "$run/tools/run_budget.py" "$budget" || status=$?
    if test -d "$run/qwen14-$budget-vllm"; then
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status \
            >"$run/qwen14-$budget-vllm/idle-after.json"
    fi
    printf 'END vllm batch limit %s %s exit=%d\n' "$budget" "$(date -u +%FT%TZ)" "$status"
    if (( status != 0 )); then result=1; fi
done
exit "$result"
