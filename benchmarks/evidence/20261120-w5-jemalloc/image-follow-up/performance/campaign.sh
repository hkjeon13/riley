#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-allocator-image-recheck-20261120.qPXAvyfv"
driver="$evidence/w5-allocator-packaged-gpu-20261120.nQpLFMUS"
old="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
(cd "$base" && sha256sum -c "$stage/scenario-inputs.sha256")
(cd "$package" && sha256sum -c dispatch.sha256 result.sha256)
(cd "$driver" && sha256sum -c dispatch.sha256)
(cd "$harness" && sha256sum -c "$driver/gpu-soak-inputs.sha256")
export PYTHONPATH="$driver:$package:$accuracy:$old:$harness/ci:$base/src/ci"
export RILEY_W5_PERF_SOURCE="$base/src/benchmarks/live"
export RILEY_W5_PERF_HARNESS="$harness/benchmarks/live"

printf 'WAIT %s %s\n' "$driver" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$driver/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$driver/campaign.pid")" | grep -Fq -- "$driver/campaign.sh"; then
        test -s "$driver/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$driver/campaign.exit")" = 0
(cd "$driver" && sha256sum -c result.sha256)
if ! test -s "$driver/accuracy-result.sha256"; then
    grep -q '^SKIP_STRICT_SOAK ' "$driver/campaign.log"
    printf 'SKIP_PACKAGED_SOAK %s\n' "$(date -u +%FT%TZ)"
    exit 0
fi
(cd "$driver" && sha256sum -c accuracy-result.sha256)
"$python" -B "$driver/qualify_packaged_gpu.py" verify "$package" "$evidence" "$driver" cache-on cache-off accuracy >gpu-qualification.json
cmp gpu-qualification.json "$driver/verified-all.json"
"$python" -B "$driver/qualify_packaged_gpu.py" bind "$package" "$evidence" "$stage" >gpu-checker-input.json
cmp gpu-checker-input.json "$driver/gpu-checker-input.json"

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
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_allocator_performance.py" "$package" "$evidence" "$stage" "$name" \
        >"$name/gpu-check.log" 2>&1 || rc=$?
    reservations "$name" after
    printf '%d\n' "$rc" >"$name/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B compare_allocator_performance.py "$package" "$evidence" "$stage" >verified-performance.json
sha256sum gpu-qualification.json gpu-checker-input.json verified-performance.json >result.sha256
