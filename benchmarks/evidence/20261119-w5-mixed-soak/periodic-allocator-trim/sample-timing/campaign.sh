#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w5-periodic-trim-20261119.mI9aaaEp"
preceding="$evidence/w3-scheduler-choices-20261119.Hmy0dT5E"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
allocator="$evidence/w5-allocator-trim-20261119.ZEkdRfD9"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c observer.sha256
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
test "$(cat "$prior/campaign.exit")" = 0
test "$(cat "$allocator/campaign.exit")" = 0
(
    cd "$prior"
    sha256sum -c harness.sha256
    printf '%s  %s\n' 5463d8ef947b88f08165c71378ef8dc2910f35989ab5bcf529516c0042ed9edf w5-periodic-trim.so | sha256sum -c -
)
cp -a -- "$prior/harness" harness
sha256sum -c "$prior/harness.sha256"
git apply --check observer.patch
git apply observer.patch
cp -- observe_soak.py harness/benchmarks/live/
sha256sum harness/{ci/check_distribution_gpu.py,benchmarks/live/{observe_soak,soak,load,api_check}.py} >used-harness.sha256
export PYTHONPATH="$stage:$prior:$base/src/ci:$allocator"
export RILEY_W5_PERIODIC_TRIM_LIBRARY="$prior/w5-periodic-trim.so"

holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
mode=cache-off
mkdir "$mode"
holder_status >"$mode/idle-before.json"
printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
    "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
    >"$mode/gpu-check.log" 2>&1 || rc=$?
holder_status >"$mode/idle-after.json"
printf '%d\n' "$rc" >"$mode/exit"
"$python" -B "$prior/verify_periodic_trim.py" "$stage" >"$mode/original-gate.json"
"$python" -B compare_sample_timing.py "$mode/gpu-evidence" >"$mode/timing-summary.json"
sha256sum "$mode/exit" "$mode/"{original-gate,timing-summary,idle-before,idle-after}.json >result.sha256
printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
