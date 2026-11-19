#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
predecessor="$evidence/w3-qkv-fixed-arrival-20261120.RScZSBo3"
old="$evidence/w5-allocator-image-perf-20261120.tmZw0Cpo"
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
export RILEY_W5_DRILL_API="$harness/benchmarks/live/api_check.py"
"$python" -B - "$package" "$evidence" "$old" <<'PY'
from pathlib import Path
import sys
from qualify_packaged_gpu import binding
from verify_a100_candidate import read, require
package, evidence, old = map(Path, sys.argv[1:])
require(read(old / "gpu-checker-input.json") == binding(package, evidence), "packaged GPU binding changed")
PY
"$python" -B route_drill.py --help >preflight.txt
printf 'WAIT %s %s\n' "$(basename "$predecessor")" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 3600))
until [[ -s "$predecessor/campaign.exit" ]]; do
    if ! ps -p 2458506 -o args= | grep -Fq -- "$predecessor/campaign.sh"; then
        [[ -s "$predecessor/campaign.exit" ]] && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$predecessor/campaign.exit")" = 0
(cd "$predecessor" && sha256sum -c dispatch.sha256 result.sha256) >w3-completion-check.log
mkdir route
cp -- "$old/gpu-checker-input.json" route/
cp -- drill_client.py route/performance_client.py
"$python" -B -c 'from route_drill import choose_port; print(choose_port())' >route/port

reservations() {
    local folder=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$folder/idle-$point.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$folder/gpu7-$point.json"
    "$python" -B -c 'from pathlib import Path; import sys; from qualify_allocator import check_reservations
check_reservations(Path(sys.argv[1]), sys.argv[2])' "$folder" "$point"
}
for name in packaged-first parent-second; do
    folder="$stage/route/$name"
    mkdir "$folder"
    reservations "$folder" before
    rc=0
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/route_drill.py" run "$package" "$evidence" "$stage/route" "$name" \
        >"$folder/gpu-check.log" 2>&1 || rc=$?
    reservations "$folder" after
    printf '%d\n' "$rc" >"$folder/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B route_drill.py verify "$package" "$evidence" "$stage/route" >verified.json
sha256sum verified.json route/port route/gpu-checker-input.json \
    route/{packaged-first,parent-second}/{exit,route.json,process-before.json,process-after.json,idle-before.json,idle-after.json,gpu7-before.json,gpu7-after.json,gpu-evidence/report.json} \
    >result.sha256
