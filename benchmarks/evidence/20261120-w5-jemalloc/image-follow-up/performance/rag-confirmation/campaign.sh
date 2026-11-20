#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w5-allocator-image-perf-20261120.tmZw0Cpo"
package="$evidence/w5-allocator-image-recheck-20261120.qPXAvyfv"
driver="$evidence/w5-allocator-packaged-gpu-20261120.nQpLFMUS"
native="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$prior/campaign.exit")" = 0
(cd "$prior" && sha256sum -c dispatch.sha256 result.sha256)
(cd "$base" && sha256sum -c "$prior/scenario-inputs.sha256")
(cd "$harness" && sha256sum -c "$driver/gpu-soak-inputs.sha256")
cp -- "$prior/"{run_allocator_performance.py,compare_allocator_performance.py,gpu-checker-input.json} .
sha256sum -c reuse.sha256
export PYTHONPATH="$stage:$prior:$driver:$package:$accuracy:$native:$harness/ci:$base/src/ci"
export RILEY_W5_PERF_SOURCE="$base/src/benchmarks/live"
export RILEY_W5_PERF_HARNESS="$harness/benchmarks/live"
unset RILEY_RAG_PAIR RILEY_W5_REPLAY_ARM

reservations() {
    local directory=$1 point=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$directory/idle-$point.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$directory/gpu7-$point.json"
    "$python" -B -c 'import sys; from pathlib import Path; from qualify_allocator import check_reservations
check_reservations(Path(sys.argv[1]), sys.argv[2])' "$directory" "$point"
}

for name in parent-first packaged-first packaged-second parent-second; do
    mkdir "$name"
    reservations "$name" before
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    rc=0
    RILEY_RAG_PAIR="${name##*-}" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_allocator_performance.py" "$package" "$evidence" "$stage" "$name" \
        >"$name/gpu-check.log" 2>&1 || rc=$?
    reservations "$name" after
    printf '%d\n' "$rc" >"$name/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_confirmation.py "$package" "$evidence" "$stage" "$prior" >verified-confirmation.json
sha256sum gpu-checker-input.json verified-confirmation.json >result.sha256
