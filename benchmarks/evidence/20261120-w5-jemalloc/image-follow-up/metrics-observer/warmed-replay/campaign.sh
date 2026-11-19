#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
reference="$evidence/w5-metrics-observer-20261120.waPD40vq"
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
export PYTHONPATH="$reference:$old:$driver:$package:$accuracy:$native:$harness/ci:$base/src/ci:$harness/benchmarks/live"
export RILEY_W5_PERF_HARNESS="$harness/benchmarks/live"
"$python" -B - "$package" "$evidence" "$old" <<'PY'
from pathlib import Path
import sys
from qualify_packaged_gpu import binding
from verify_a100_candidate import read, require
package, evidence, old = map(Path, sys.argv[1:])
require(read(old / "gpu-checker-input.json") == binding(package, evidence), "packaged GPU binding changed")
PY
"$python" -B warmed_client.py --help >preflight.txt
test "$(cat "$reference/campaign.exit")" = 0
(cd "$reference" && sha256sum -c dispatch.sha256 result.sha256) >reference-check.log
mkdir observation
cp -- "$old/gpu-checker-input.json" observation/
cp -- warmed_client.py observation/performance_client.py

reservations() {
    local folder=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$folder/idle-$point.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$folder/gpu7-$point.json"
    "$python" -B -c 'from pathlib import Path; import sys; from qualify_allocator import check_reservations
check_reservations(Path(sys.argv[1]), sys.argv[2])' "$folder" "$point"
}
name=packaged-first
folder="$stage/observation/$name"
mkdir "$folder"
reservations "$folder" before
rc=0
printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$old/run_allocator_performance.py" "$package" "$evidence" "$stage/observation" "$name" \
    >"$folder/gpu-check.log" 2>&1 || rc=$?
reservations "$folder" after
printf '%d\n' "$rc" >"$folder/exit"
printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B verify_warmed.py "$package" "$evidence" "$stage/observation" "$reference" >verified.json
sha256sum verified.json observation/gpu-checker-input.json \
    "$folder/"{exit,process-before.json,process-after.json,idle-before.json,idle-after.json,gpu7-before.json,gpu7-after.json,gpu-evidence/report.json} \
    >result.sha256
