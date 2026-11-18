#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
original="$evidence/w5-mixed-soak-20261119.YJg0Xzhk"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
test "$(cat "$original/campaign.exit")" = 0
cd "$run"
mkdir tools
cp -- "$original/holder.py" .
for name in run_soak.py soak.py load.py compare_runner.py; do
    cp -- "$original/tools/$name" "tools/$name"
done
sha256sum holder.py tools/*.py > staged-inputs.sha256
export W5_GPU_INDEX=7 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-before.json
printf 'START cache-off repeat %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_soak.py" cache-off || status=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-after.json
printf 'END cache-off repeat %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
