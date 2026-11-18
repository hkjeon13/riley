#!/usr/bin/env bash
set -eu
run="$1"
base="$(dirname "$run")"
w5="$base/w5-mixed-soak-20261119.YJg0Xzhk"
hold="$base/w2-sm17-20261118.6UMfeHh2"
export W5_GPU_INDEX=6 PYTHONDONTWRITEBYTECODE=1
deadline=$((SECONDS + 900))
while [[ ! -s "$w5/campaign.exit" && $SECONDS -lt $deadline ]]; do sleep 5; done
if [[ ! -s "$w5/campaign.exit" ]]; then
    printf 'cancelled: soak GPU reservation snapshots were unavailable\n' >&2
    printf '80\n' >"$run/campaign.exit"
    exit 80
fi
rc=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 run -- /usr/bin/python3 -B "$run/tools/run_operational.py" operational || rc=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status >"$run/idle-after.json" || rc=1
printf '%d\n' "$rc" >"$run/campaign.exit"
exit "$rc"
