#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
baseline="$evidence/w5-smaps-rollup-20261119.39So7ib6"
repeat="$evidence/w5-cache-off-repeat-20261119.gcvXdzAj"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$baseline/campaign.exit")" = 0
sha256sum -c dispatch.sha256
mkdir tools
cp -- "$repeat/holder.py" .
for name in run_soak.py soak.py load.py compare_runner.py; do
    cp -- "$repeat/tools/$name" "tools/$name"
done
sha256sum -c "$repeat/staged-inputs.sha256"
git apply --check strict-rss-runner.patch
git apply strict-rss-runner.patch
gzip -dc upstream-soak.py.gz > tools/upstream_soak.py
sha256sum -c upstream-soak.sha256
cp -- attribute_anon_rss.py tools/soak.py
PYTHONPATH="$run/tools" /usr/bin/python3 -B - <<'PY' > smaps-cpu-check.json
import json
import os
from soak import read_mappings, totals
from upstream_soak import read_smaps_rollup

mappings, elapsed_ms = read_mappings(os.getpid())
grouped = totals(mappings)
rollup = read_smaps_rollup(os.getpid())
assert sum(row["Anonymous"] for row in grouped.values()) > 0
assert all(0 <= row["Anonymous"] <= row["Rss"] for row in grouped.values())
assert 0 < rollup["anonymous_kib"] <= rollup["rss_kib"]
print(json.dumps({"mappings": len(mappings), "groups_kib": grouped, "read_ms": elapsed_ms, "rollup": rollup}))
PY
export W5_GPU_INDEX=7 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-before.json
printf 'START cache-off anonymous attribution %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_soak.py" cache-off || status=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-after.json
printf 'END cache-off anonymous attribution %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
