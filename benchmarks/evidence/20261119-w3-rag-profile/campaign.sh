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
result=0
for label in qwen14-riley qwen14-vllm qwen14-profile coder14-vllm coder14-riley coder14-profile qwen4-profile; do
    printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
        /usr/bin/python3 -B "$run/tools/run_profile.py" "${label%-*}" "${label##*-}" || rc=$?
    if test -d "$run/$label"; then
        /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status >"$run/$label/idle-after.json"
    fi
    printf 'END %s %s exit=%d\n' "$label" "$(date -u +%FT%TZ)" "$rc"
    if test "$rc" -ne 0; then result=1; fi
done
exit "$result"
