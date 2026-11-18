#!/usr/bin/env bash
set -euo pipefail
umask 077
run=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$run")
repeat="$evidence/w5-cache-off-repeat-20261119.gcvXdzAj"
preceding="$evidence/w3-request-stages-20261119.DWykKwJR"
hold="$evidence/w2-sm17-20261118.6UMfeHh2"
trap 'printf "%d\n" "$?" >"$run/campaign.exit"' EXIT
cd "$run"
test "$(cat "$repeat/campaign.exit")" = 0
sha256sum -c dispatch.sha256
getconf GNU_LIBC_VERSION > libc.txt
cc -std=c11 -Wall -Wextra -Werror -O2 -shared -fPIC -pthread -Wl,--no-undefined \
    sample_allocator.c -o sample_allocator.so
sha256sum -- sample_allocator.so > library.sha256
env RILEY_W5_ALLOCATOR_OUTPUT="$run/cpu-smoke.jsonl" LD_PRELOAD="$run/sample_allocator.so" /bin/sleep 1
/usr/bin/python3 -B - cpu-smoke.jsonl <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
assert len(rows) == 1
row = rows[0]
assert row["rss_kib"] == row["anonymous_kib"] + row["file_kib"] + row["shared_kib"]
assert row["arena_bytes"] == row["arena_in_use_bytes"] + row["arena_free_bytes"] > 0
PY
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 36000))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
mkdir tools
cp -- "$repeat/holder.py" .
for name in run_soak.py soak.py load.py compare_runner.py; do
    cp -- "$repeat/tools/$name" "tools/$name"
done
sha256sum -c "$repeat/staged-inputs.sha256"
sha256sum -c runner-base.sha256
gzip -dc runner.patch.gz | git apply --check -
gzip -dc runner.patch.gz | git apply -
sha256sum -c runner.sha256
export W5_GPU_INDEX=7 PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-before.json
printf 'START cache-off allocator trace %s\n' "$(date -u +%FT%TZ)"
status=0
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 run -- \
    /usr/bin/python3 -B "$run/tools/run_soak.py" cache-off || status=$?
/usr/bin/python3 -B "$hold/control.py" --lease-dir "$hold/leases" --index 7 status > idle-after.json
printf 'END cache-off allocator trace %s exit=%d\n' "$(date -u +%FT%TZ)" "$status"
exit "$status"
