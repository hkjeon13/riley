#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
old="$evidence/w5-allocator-image-perf-20261120.tmZw0Cpo"
w4="$evidence/w4-decode-final-retry-20261120.5bTpAh8k"
driver="$evidence/w5-allocator-packaged-gpu-20261120.nQpLFMUS"
package="$evidence/w5-allocator-image-recheck-20261120.qPXAvyfv"
native="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$old/campaign.exit")" = 0
(cd "$old" && sha256sum -c dispatch.sha256 result.sha256)
(cd "$base" && sha256sum -c "$old/scenario-inputs.sha256")
(cd "$harness" && sha256sum -c "$driver/gpu-soak-inputs.sha256")
export PYTHONPATH="$old:$driver:$package:$accuracy:$native:$harness/ci:$base/src/ci:$harness/benchmarks/live"
export RILEY_W5_PERF_SOURCE="$base/src/benchmarks/live"
export RILEY_W5_PERF_HARNESS="$harness/benchmarks/live"
export RILEY_W5_REPLAY_REFERENCE="$old"
cp -- "$old/gpu-checker-input.json" .
"$python" -B - "$old" <<'PY'
import json
from pathlib import Path
import sys
from fixed_arrival import bodies, read_schedule
from verify_a100_candidate import digest, require

old = Path(sys.argv[1])
data = json.loads(Path("preflight.json").read_bytes())
ref = json.loads((old / "verified-performance.json").read_bytes())
require(ref["all_tested_streams_match"] and ref["all_measured_regression_guards_pass"]
        and digest(old / "verified-performance.json") == "318ddb75a3c2dacfabbecf44acb30a603b34bad433f3d302e5101a8e27c0916c",
        "closed-loop performance reference changed")
require(digest(Path("fixed_arrival.py")) == data["client_sha256"], "fixed-arrival preflight client differs")
for name, spec in data["sources"].items():
    path = old/name/"gpu-evidence/rag-c8-seed2.records.json"
    clock, checksum = read_schedule(path, bodies("m"))
    require(checksum == spec["reference_sha256"] and len(clock) == spec["matched_requests"]
            and max(clock.values()) == spec["scheduled_span_seconds"], "write schedule differs")
PY
printf 'WAIT %s %s\n' "$w4" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$w4/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$w4/campaign.pid")" | grep -Fq -- "$w4/campaign.sh"; then
        test -s "$w4/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'PREDECESSOR_EXIT %s %s\n' "$(cat "$w4/campaign.exit")" "$(date -u +%FT%TZ)"

reservations() {
    local directory=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$directory/idle-$point.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$directory/gpu7-$point.json"
}

for name in parent-first packaged-first packaged-second parent-second; do
    mkdir "$name"
    reservations "$name" before
    "$python" -B -c 'import sys; from pathlib import Path; from qualify_allocator import check_reservations
check_reservations(Path(sys.argv[1]), "before")' "$name"
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    rc=0
    RILEY_W5_REPLAY_ARM=$name /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run --         "$python" -B "$old/run_allocator_performance.py" "$package" "$evidence" "$stage" "$name"         >"$name/gpu-check.log" 2>&1 || rc=$?
    reservations "$name" after
    printf '%s\n' "$rc" >"$name/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_replay.py "$package" "$evidence" "$stage" "$old" >verified-replay.json
sha256sum gpu-checker-input.json verified-replay.json >result.sha256
