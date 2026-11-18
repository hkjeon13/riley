#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
repeat="$evidence/w5-cache-off-repeat-20261119.gcvXdzAj"
preceding="$evidence/w3-prefill-continuation-20261119.QjxM6xaM"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$repeat/campaign.exit")" = 0
sha256sum -c dispatch.sha256
mkdir tools
cp -- "$repeat/holder.py" .
for name in run_soak.py soak.py load.py compare_runner.py; do
    cp -- "$repeat/tools/$name" "tools/$name"
done
sha256sum -c "$repeat/staged-inputs.sha256"
gzip -dc soak.patch.gz | git apply --check -
gzip -dc soak.patch.gz | git apply -
sha256sum -c soak.sha256
PYTHONPATH="$run/tools" /usr/bin/python3 -B - <<'PY' > smaps-cpu-check.json
import json
import os
from soak import read_smaps_rollup

sample = read_smaps_rollup(os.getpid())
assert 0 < sample["anonymous_kib"] <= sample["rss_kib"]
print(json.dumps(sample))
PY
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
export W5_GPU_INDEX=7 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-before.json
printf 'START cache-off with smaps rollup %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_soak.py" cache-off || status=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-after.json
printf 'END cache-off with smaps rollup %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
