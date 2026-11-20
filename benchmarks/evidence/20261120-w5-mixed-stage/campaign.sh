#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-serving-smoke-recheck-20261120.PsNfm6KN"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cpu="$base/cpu-evidence/report.json"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf '%s  %s\n' 92f884b05c1101d4bfb6c49df9289483b698e4e64d88fdc1090705998a17de33 "$cpu" | sha256sum -c
mkdir -p ci benchmarks/live
cp -- check_distribution_gpu.py ci/
cp -- mixed_stage_check.py benchmarks/live/
cp -- "$base/src/ci/"{check_distribution_image,check_serving_source_contract}.py ci/
cp -- "$base/src/benchmarks/live/long_accuracy.py" benchmarks/live/
cp -- "$harness/benchmarks/live/api_check.py" benchmarks/live/
sha256sum -c reuse.sha256

reservations() {
    local point=$1
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status \
            >"gpu$index-$point.json"
    done
    "$python" -B - "$point" <<'PY'
from pathlib import Path
import sys
from verify_smoke import reservations
reservations(Path.cwd(), sys.argv[1])
PY
}

reservations before
printf 'START %s\n' "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    "$python" -B "$stage/ci/check_distribution_gpu.py" "$cpu" "$gpu" "$base/model" "$checkpoint" \
    "$stage/gpu-evidence" --tools --mixed-stage >driver.log 2>&1 || rc=$?
reservations after
printf '%d\n' "$rc" >exit
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B verify_mixed.py "$stage" >verified-mixed.json
sha256sum verified-mixed.json >result.sha256
