#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-serving-smoke-recheck-20261120.PsNfm6KN"
fixture="$evidence/w4-native-projection-plans-20261120.yvjrWvOj/fixtures/20261119-w4-holdout/holdout.json.gz"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cpu="$base/cpu-evidence/report.json"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf '%s  %s\n' 92f884b05c1101d4bfb6c49df9289483b698e4e64d88fdc1090705998a17de33 "$cpu" | sha256sum -c
mkdir -p ci benchmarks/live
cp -- check_distribution_gpu.py ci/
cp -- tp_holdout_check.py benchmarks/live/
cp -- "$base/src/ci/"{check_distribution_image,check_serving_source_contract}.py ci/
cp -- "$base/src/benchmarks/live/long_accuracy.py" benchmarks/live/
cp -- "$harness/benchmarks/live/api_check.py" benchmarks/live/
cp -- "$fixture" holdout.json.gz
sha256sum -c reuse.sha256
model_stage=$(cat model-stage.txt)
[[ $model_stage =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-tp2-model-20261120\.[A-Za-z0-9]+$ ]]
printf 'WAIT %s %s\n' "$model_stage" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 1200))
until test -f "$model_stage/prepare.exit"; do
    if ! ps -p "$(cat "$model_stage/prepare.pid")" -o args= | grep -Fq -- "$model_stage/prepare.sh"; then
        test -f "$model_stage/prepare.exit" && break
        exit 75
    fi
    (( SECONDS < deadline )) || exit 75
    sleep 15
done
test "$(cat "$model_stage/prepare.exit")" = 0
printf '%s  %s\n' c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539 \
    "$model_stage/model/tokenizer.json" | sha256sum -c
nvidia-smi topo -m | awk '$1 == "GPU6" {print $9}' >peer-link.txt
test "$(cat peer-link.txt)" = NV12

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
    bash "$stage/peer_and_run.sh" "$holder" "$python" "$stage" "$cpu" "$model_stage/model" >driver.log 2>&1 || rc=$?
reservations after
printf '%d\n' "$rc" >exit
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B verify_tp2.py "$stage" >verified-tp2.json
sha256sum verified-tp2.json >result.sha256
