#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
baseline="$evidence/w5-dist-off-retry.ORBlgo3W"
combined="$evidence/w5-tcache-control-20261119.liqVpVJ7"
preceding="$evidence/w2-ordered-tables-20261119.e89GGsLO"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
tunable=GLIBC_TUNABLES=glibc.malloc.tcache_count=0
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
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
printf 'PREDECESSOR_EXIT %s %s\n' "$(cat "$preceding/campaign.exit")" "$(date -u +%FT%TZ)"
test "$(cat "$base/campaign.exit")" = 0
test "$(cat "$combined/campaign.exit")" = 0
(cd "$combined" && sha256sum -c result.sha256)

mkdir -p harness/ci harness/benchmarks/live
cp -- "$baseline/harness/ci/check_distribution_gpu.py" harness/ci/
for file in soak.py load.py api_check.py; do
    cp -- "$baseline/harness/benchmarks/live/$file" harness/benchmarks/live/
done
sha256sum -c checker-base.sha256 harness.sha256
gzip -dc tcache-only.patch.gz | git apply --check -
gzip -dc tcache-only.patch.gz | git apply -
sha256sum -c checker-candidate.sha256
for helper in tcache_probe.c compare_tcache.py verify_periodic_trim.py periodic_trim.py; do
    cp -- "$combined/$helper" .
done
sha256sum -c helpers.sha256
export PYTHONPATH="$stage:$base/src/ci"
image=$("$python" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' "$base/cpu-evidence/report.json")
cc -std=c11 -O0 -Wall -Wextra -Werror -o tcache-probe tcache_probe.c
chmod 0755 tcache-probe
probe() {
    docker run --rm --pull=never --network none --entrypoint /tmp/w5-tcache-probe \
        --env NVIDIA_VISIBLE_DEVICES=void \
        --mount "type=bind,source=$stage/tcache-probe,target=/tmp/w5-tcache-probe,readonly" "$@" "$image"
}
probe >probe-default.json
probe --env "$tunable" >probe-disabled.json
"$python" -B compare_tcache_only.py "$stage" "$combined" --cpu-only >cpu-probe-proof.json

holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for index in 6 7; do holder_status "$index" >"initial-reservation-$index.json"; done
"$python" -B - <<'PY'
import json
from pathlib import Path

for index, pid in ((6, 2431228), (7, 2431237)):
    state = json.loads(Path(f"initial-reservation-{index}.json").read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["phase"] != "idle"
            or state["other_processes"] or state["allocation_bytes"] != state["torch_reserved_bytes"]
            or state["allocation_bytes"] != 20 * 1024**3):
        raise ValueError(f"GPU {index} is not in its original 20 GiB idle reservation")
PY
for mode in cache-on cache-off; do
    mkdir "$mode"
    holder_status 6 >"$mode/idle-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    result=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
        >"$mode/gpu-check.log" 2>&1 || result=$?
    holder_status 6 >"$mode/idle-after.json"
    printf '%d\n' "$result" >"$mode/exit"
    test -s "$mode/gpu-evidence/report.json" || exit 74
    "$python" -B compare_tcache_only.py "$stage" "$combined" >"$mode/comparison.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$result"
done
sha256sum probe-{default,disabled}.json cpu-probe-proof.json \
    cache-{off,on}/{exit,idle-before.json,idle-after.json,comparison.json} >result.sha256
