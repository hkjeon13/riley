#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
base=$(dirname "$run")
predecessor="$base/w3-budget-control-20261119.kGQiFg97"
hold="$base/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
printf 'WAIT for %s %s\n' "$predecessor" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 39600))
until test -s "$predecessor/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$predecessor/campaign.pid")" | grep -Fq -- "$predecessor/campaign.sh"; then
        test -s "$predecessor/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
export PYTHONDONTWRITEBYTECODE=1
printf 'START Qwen3-4B profile retry %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_profile_retry.py" qwen4 profile || status=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status >"$run/qwen4-profile/idle-after.json"
printf 'END Qwen3-4B profile retry %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
