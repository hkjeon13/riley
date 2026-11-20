#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
cpu="$base/cpu-evidence/report.json"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
printf '%s  %s\n' 92f884b05c1101d4bfb6c49df9289483b698e4e64d88fdc1090705998a17de33 "$cpu" | sha256sum -c
mkdir -p ci benchmarks/live
cp -- check_distribution_gpu.py ci/
cp -- "$base/src/ci/"{check_distribution_image,check_serving_source_contract}.py ci/
cp -- "$harness/benchmarks/live/"{api_check,load,soak}.py benchmarks/live/
sha256sum -c reuse.sha256
mkdir model-roots
while read -r model hash kv_blocks; do
    printf 'PREPARE %s %s\n' "$model" "$(date -u +%FT%TZ)"
    "$python" -B prepare_model.py "$root/models/$model" "$stage/model-roots/$model" "$hash"
    docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/sh \
        --mount "type=bind,src=$stage/model-roots/$model,dst=/models,readonly" "$image" -ec '
            test "$(id -u):$(id -g)" = 65532:65532
            for asset in /models/*; do test -r "$asset"; done
        '
    printf 'READABLE %s %s\n' "$model" "$(date -u +%FT%TZ)"
done <models.tsv

reservations() {
    local model=$1 point=$2
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status \
            >"$model/gpu$index-$point.json"
    done
    "$python" -B - "$model" "$point" <<'PY'
from pathlib import Path
import sys
from verify_smoke import reservations
reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

while read -r model hash kv_blocks; do
    mkdir "$model"
    reservations "$model" before
    printf 'START %s %s\n' "$model" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/ci/check_distribution_gpu.py" "$cpu" "$gpu" "$stage/model-roots/$model" "$hash" \
        "$stage/$model/gpu-evidence" --tools --serving-smoke --smoke-kv-blocks "$kv_blocks" >"$model/driver.log" 2>&1 || rc=$?
    reservations "$model" after
    printf '%d\n' "$rc" >"$model/exit"
    printf 'END %s %s exit=%d\n' "$model" "$(date -u +%FT%TZ)" "$rc"
done <models.tsv
"$python" -B verify_extended.py "$stage" >verified-smoke.json
sha256sum verified-smoke.json >result.sha256
"$python" -B - <<'PY'
import json
from pathlib import Path
import sys

models = json.loads(Path("verified-smoke.json").read_bytes())["results"]
sys.exit(int(any(result["status"] != "passed" for result in models.values())))
PY
