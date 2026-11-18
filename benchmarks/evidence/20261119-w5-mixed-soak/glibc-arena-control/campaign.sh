#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
baseline="$evidence/w5-anon-mappings-20261119.sZulouQC"
preceding="$evidence/w3-prefill-finish-20261119.YeLiU3Fi"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$baseline/campaign.exit")" = 1
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 25
done
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 6 status |
    /usr/bin/python3 -c 'import json,sys; assert json.load(sys.stdin)["phase"]=="idle"'
mkdir tools
cp -- "$baseline/holder.py" .
for name in run_soak.py soak.py upstream_soak.py load.py compare_runner.py; do
    cp -- "$baseline/tools/$name" "tools/$name"
done
sha256sum -c source-tools.sha256
gzip -dc runner.patch.gz | git apply --check -
gzip -dc runner.patch.gz | git apply -
sha256sum -c runner.sha256
export W5_GPU_INDEX=7 PYTHONDONTWRITEBYTECODE=1
for arm in control arena2; do
    mkdir "$arm"
    sha256sum -c <(awk '$2 == "tools/upstream_soak.py"' source-tools.sha256) > "$arm/campaign.log"
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > "$arm/idle-before.json"
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    status=0
    W5_MALLOC_ARENA_ARM="$arm" /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
        /usr/bin/python3 -B "$run/tools/run_soak.py" cache-off || status=$?
    /usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > "$arm/idle-after.json"
    printf '%s\n' "$status" > "$arm/campaign.exit"
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$status"
    /usr/bin/python3 -B - "$arm" <<'PY'
import json
from pathlib import Path
import sys

arm = Path(sys.argv[1])
cache = arm/"cache-off"
receipt = json.loads((cache/"receipt.json").read_bytes())
soak = json.loads((cache/"soak.json").read_bytes())
rss_gate = "rss_growth_second_half_at_most_zero"
failed = {name for name, check in soak["checks"].items() if not check}
expected = None if arm.name == "control" else "2"
assert receipt["server_exit"] == 0 and receipt["server_malloc_arena_max"] == expected
assert type(soak["smaps_rollup_second_half"]["rss_growth_kib"]) is int
assert failed <= {rss_gate} and int((arm/"campaign.exit").read_text()) == bool(failed)
assert (cache/"anon-attribution.json").is_file()
PY
done
