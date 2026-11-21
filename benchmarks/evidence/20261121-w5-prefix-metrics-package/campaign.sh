#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
cache="$evidence/w5-prefix-metrics-gpu-20261121.D0Jb5MtZ"
accuracy="$evidence/w4-structured-holdout-20261121.w0hBehiZ"
golden="$base/src/benchmarks/evidence"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
uuid6=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
uuid7=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
cpu="$package/cpu-evidence/report.json"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
while ! test -f "$package/build.exit"; do
    ps -p "$(cat "$package/build.pid")" -o stat= | grep -Eq '^[^Z]+$'
    sleep 5
done
test "$(cat "$package/build.exit")" = 0
(cd "$package" && sha256sum -c build-inputs.sha256 build-result.sha256 >/dev/null)
test "$(cat "$cache/campaign.exit")" = 0
test "$(cat "$accuracy/campaign.exit")" = 0
mkdir -p harness/ci harness/benchmarks/live
cp check_distribution_gpu.py harness/ci/
cp "$prior/harness/benchmarks/live/api_check.py" harness/benchmarks/live/
cp "$cache/"{load,soak}.py harness/benchmarks/live/
cp "$accuracy/"{long_accuracy,structured_holdout_check}.py harness/benchmarks/live/
sha256sum -c runtime-tools.sha256
export PYTHONPATH="$package/src/ci:$rag:$prior/harness/ci:$prior/src/ci"
baseline="$golden/20261119-w4-long/riley/capture.json.gz"
holdout="$golden/20261119-w4-holdout/holdout.json.gz"
holdout_golden="$golden/20261119-w4-holdout/riley/capture.json.gz"
printf '%s  %s\n' b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6 "$baseline" \
    220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11 "$holdout" \
    12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611 "$holdout_golden" \
    "$model_sha" "$base/model/riley-checkpoint.json" | sha256sum --strict -c

reservations() {
    local phase=$1
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$phase.json"
    "$python" -B - "$phase" <<'PY'
import sys
from pathlib import Path
from current_binding import check_reservations
check_reservations(Path.cwd(), sys.argv[1])
PY
}

run_arm() {
    local label=$1 index=$2 uuid=$3 rc=0
    shift 3
    printf 'START %s %s %s\n' "$label" "$index" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$cpu" "$uuid" "$base/model" \
        "$model_sha" "$stage/$label" --tools "$@" >"$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$label.exit"
    printf 'END %s %s %s exit=%d\n' "$label" "$index" "$(date -u +%FT%TZ)" "$rc"
    test -s "$label/report.json"
}

reservations before
run_arm cache-off 6 "$uuid6" --mixed-soak cache-off --require-prefix-cache-metrics &
off=$!
run_arm cache-on 7 "$uuid7" --mixed-soak cache-on --require-prefix-cache-metrics &
on=$!
rc=0
wait "$off" || rc=1
wait "$on" || rc=1
reservations after-soaks
test "$rc" = 0
run_arm accuracy 6 "$uuid6" --accuracy-baseline "$baseline" --accuracy-holdout "$holdout" "$holdout_golden" \
    --structured-holdout "$holdout"
reservations after
test "$(cat accuracy.exit)" = 0
sha256sum {cache-off,cache-on,accuracy}.exit {cache-off,cache-on,accuracy}/{report,api,shutdown}.json \
    cache-{off,on}/soak.json idle-{before,after-soaks,after}.json gpu7-{before,after-soaks,after}.json >result.sha256
