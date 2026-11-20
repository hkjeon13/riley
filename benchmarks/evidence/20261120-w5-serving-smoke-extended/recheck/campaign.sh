#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w5-serving-smoke-extended-20261120.BMmj7efv"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cpu="$base/cpu-evidence/report.json"
python="$root/tools/vllm-venv/bin/python"
model=SmolLM2-1.7B-Instruct
checkpoint=592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753
gpu=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf '%s  %s\n' 92f884b05c1101d4bfb6c49df9289483b698e4e64d88fdc1090705998a17de33 "$cpu" | sha256sum -c
mkdir -p ci benchmarks/live
cp -- check_distribution_gpu.py ci/
cp -- api_check.py soak.py benchmarks/live/
cp -- "$base/src/ci/"{check_distribution_image,check_serving_source_contract}.py ci/
cp -- "$harness/benchmarks/live/load.py" benchmarks/live/
sha256sum -c reuse.sha256
printf 'WAIT %s %s\n' "$prior" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 1200))
until test -f "$prior/campaign.exit"; do
    if ! ps -p "$(cat "$prior/campaign.pid")" -o args= | grep -Fq -- "$prior/campaign.sh"; then
        test -f "$prior/campaign.exit" && break
        exit 75
    fi
    (( SECONDS < deadline )) || exit 75
    sleep 15
done
(cd "$prior" && sha256sum -c dispatch.sha256 result.sha256)
test "$(cat "$prior/$model/exit")" = 1

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
printf 'START %s %s\n' "$model" "$(date -u +%FT%TZ)"
rc=0
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    "$python" -B "$stage/ci/check_distribution_gpu.py" "$cpu" "$gpu" "$prior/model-roots/$model" "$checkpoint" \
    "$stage/gpu-evidence" --serving-smoke --expect-tools-unsupported >driver.log 2>&1 || rc=$?
reservations after
printf '%d\n' "$rc" >exit
printf 'END %s %s exit=%d\n' "$model" "$(date -u +%FT%TZ)" "$rc"
"$python" -B verify_recheck.py "$stage" "$prior" >verified-smoke.json
sha256sum verified-smoke.json >result.sha256
exit "$rc"
