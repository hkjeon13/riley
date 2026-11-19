#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
reference="$evidence/w5-trim-7sec-retry-20261119.fKlbclsw"
preceding="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
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
test "$(cat "$reference/campaign.exit")" = 0
(cd "$reference" && sha256sum -c result.sha256)
for item in periodic_trim.py observe_soak.py verify_periodic_trim.py compare_sample_timing.py allocator_probe.py attribute_anon_rss.py; do
    cp -- "$reference/$item" .
done
sha256sum -c patch-base.sha256
sha256sum -c helpers.sha256
for item in periodic_trim.py observe_soak.py; do
    gzip -dc "$item.patch.gz" | git apply --check -
    gzip -dc "$item.patch.gz" | git apply -
done
sha256sum -c patch-result.sha256
cp -a -- "$reference/harness" .
sha256sum -c harness-base.sha256
install -m 0600 -- observe_soak.py harness/benchmarks/live/observe_soak.py
install -m 0644 -- "$reference/w5-periodic-trim.so" w5-periodic-trim.so
printf '%s  %s\n' 5463d8ef947b88f08165c71378ef8dc2910f35989ab5bcf529516c0042ed9edf w5-periodic-trim.so | sha256sum -c -
export PYTHONPATH="$stage:$base/src/ci"
export RILEY_W5_PERIODIC_TRIM_LIBRARY="$stage/w5-periodic-trim.so"
image=$("$python" -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_id"])' \
    "$base/cpu-evidence/report.json")
tunable=$("$python" -B -c 'from periodic_trim import TCACHE_TUNABLE; print(TCACHE_TUNABLE)')
cc -std=c11 -O0 -Wall -Wextra -Werror -o tcache-probe tcache_probe.c
chmod 0755 tcache-probe
probe() {
    docker run --rm --pull=never --network none --entrypoint /tmp/w5-tcache-probe \
        --env NVIDIA_VISIBLE_DEVICES=void \
        --mount "type=bind,source=$stage/tcache-probe,target=/tmp/w5-tcache-probe,readonly" "$@" "$image"
}
probe >probe-default.json
probe --env "$tunable" >probe-disabled.json
"$python" -B compare_tcache.py "$stage" "$reference" --cpu-only >cpu-probe-proof.json
docker run --rm --pull=never --network none --entrypoint /bin/sh \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --env LD_PRELOAD=/tmp/w5-periodic-trim.so \
    --env "$tunable" \
    --mount "type=bind,source=$stage/w5-periodic-trim.so,target=/tmp/w5-periodic-trim.so,readonly" \
    "$image" -ec '
        test "$(id -u):$(id -g)" = 65532:65532
        test "$GLIBC_TUNABLES" = glibc.malloc.tcache_count=0
        grep -Fq " /tmp/w5-periodic-trim.so" /proc/$$/maps
    '
printf 'CONTAINER_TCACHE_VERIFIED %s\n' "$(date -u +%FT%TZ)"
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
for mode in cache-off cache-on; do
    mkdir "$mode"
    holder_status >"$mode/idle-before.json"
    "$python" -B - "$mode/idle-before.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    lease = json.load(stream)
if (lease["physical_index"] != 6 or lease["pid"] != 2431228 or lease["phase"] != "idle"
        or lease["other_processes"] or lease["allocation_bytes"] != 20 * 1024**3
        or lease["allocation_bytes"] != lease["torch_reserved_bytes"]):
    raise ValueError("GPU 6 is not idle with its original 20 GiB reservation")
PY
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$checkpoint" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
        >"$mode/gpu-check.log" 2>&1 || rc=$?
    holder_status >"$mode/idle-after.json"
    printf '%d\n' "$rc" >"$mode/exit"
    if ! test -s "$mode/gpu-evidence/report.json"; then
        printf 'CHECKER_NO_REPORT %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
        exit 74
    fi
    "$python" -B compare_tcache.py "$stage" "$reference" >"$mode/tcache-comparison.json"
    "$python" -B compare_sample_timing.py "$mode/gpu-evidence" >"$mode/timing-summary.json"
    "$python" -B compare_arena_shape.py "$mode/gpu-evidence" >"$mode/arena-shape.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
done
sha256sum probe-{default,disabled}.json cpu-probe-proof.json \
    cache-{off,on}/{exit,idle-before.json,idle-after.json,tcache-comparison.json,timing-summary.json,arena-shape.json} >result.sha256
