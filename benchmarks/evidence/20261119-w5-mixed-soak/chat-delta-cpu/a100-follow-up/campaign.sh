#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
series="$evidence/w5-rss-series-20261119.DepEsJZ4"
previous="$evidence/w5-http-framing-20261119.IyUaTevw"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
revision=d3d8fe6e5b1c528bff51c0008cdc24544c99f862
base_archive=cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5
patch=01be0e56d60800f1270f555ec15ed10502a424f989c0e21d6305c605c7dd9d7b
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
model=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256

printf 'WAIT %s %s\n' "$series" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 7200))
until test -s "$series/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$series/campaign.pid")" | grep -Fq -- "$series/campaign.sh"; then
        test -s "$series/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$series/campaign.exit")" = 0
(cd "$series" && sha256sum -c result.sha256)
decision=$("$python" -B - "$series/verified.json" <<'PY'
from pathlib import Path
import json
import sys

report = json.loads(Path(sys.argv[1]).read_bytes())
failed = {name for name, passed in report["strict_checks"].items() if not passed}
if report["comparison"]["same_outcome_counts"] is not True:
    raise ValueError("the original diagnostic did not replay the workload")
if failed - {"rss_growth_second_half_at_most_zero"}:
    raise ValueError("the original diagnostic has a failure outside the RSS gate")
print("run" if failed else "skip")
PY
)
printf '%s\n' "$decision" >decision.txt
if [[ "$decision" == skip ]]; then
    printf 'SKIP_RSS_GATE_PASSED %s\n' "$(date -u +%FT%TZ)"
    exit 0
fi

holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
check_holders() {
    local step=$1
    for index in 6 7; do holder_status "$index" >"$step-$index.json"; done
    "$python" -B - "$step" <<'PY'
from pathlib import Path
import json
import sys
for index, pid in ((6, 2431228), (7, 2431237)):
    state = json.loads(Path(f"{sys.argv[1]}-{index}.json").read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["phase"] != "idle"
            or state["other_processes"] or state["allocation_bytes"] != state["torch_reserved_bytes"]
            or state["allocation_bytes"] != 20 * 1024**3):
        raise ValueError(f"GPU {index} does not have its original idle reservation")
PY
}
check_holders initial

mkdir src cpu-evidence harness
printf '%s  %s\n' "$base_archive" "$base/source.tar" "$patch" chat-delta.patch.gz | sha256sum -c -
tar -xf "$base/source.tar" -C src --no-same-owner
gzip -dc chat-delta.patch.gz >chat-delta.patch
(
    cd src
    git apply --check "$stage/chat-delta.patch"
    git apply "$stage/chat-delta.patch"
    "$python" -B ci/check_serving_source_contract.py
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf "$stage/source.tar" .
)
"$python" -B verify_chat_source.py "$base/source.tar" source.tar source-changes.json >source-inventory.json
archive=$(sha256sum source.tar | cut -d ' ' -f 1)
printf '%s\n' "$archive" >source-archive.sha256
cp -r -- "$previous/harness/"{ci,benchmarks} harness/
sha256sum -c harness.sha256
printf '%s  %s\n' 085565c249d4b047acd496321c100b0707456d4be77bfb18cdda44b30571584f \
    "$previous/compare_candidate.py" | sha256sum -c -
"$python" -B - "$previous/compare_candidate.py" "$patch" <<'PY'
from pathlib import Path
import sys

template = Path(sys.argv[1]).read_text()
updated = template.replace("reserve-validated-body.patch.gz", "chat-delta.patch.gz")
updated = updated.replace("28f0ef9ed0e5e2e81b5f82d68f99554977c461ca3cfe91ea929b5becb38fd520", sys.argv[2])
updated = updated.replace("W5 HTTP buffer diagnostic", "W5 chat SSE shaping diagnostic")
Path("compare_candidate.py").write_text(updated)
PY

docker_root=$(docker info --format '{{ .DockerRootDir }}')
available_bytes() {
    df --output=avail -B1 "$docker_root" | tail -n 1
}
(( $(available_bytes) >= 4 * 1024 * 1024 * 1024 ))
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
docker buildx build --load --platform linux/amd64 --progress plain \
    --file "$stage/src/deploy/docker/Dockerfile" \
    --build-arg CARGO_BUILD_JOBS=2 --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2 \
    --label "org.opencontainers.image.revision=$revision" \
    --label "io.riley.source-archive-sha256=$archive" \
    --label "io.riley.diagnostic-patch-sha256=$patch" \
    --iidfile "$stage/image.id" --tag "riley:a100-chat-sse-${stage##*.}" "$stage/src" >build.log 2>&1 &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
    if (( $(available_bytes) < 4 * 1024 * 1024 * 1024 )); then
        printf 'BUILD_STOP %s free space below 4 GiB\n' "$(date -u +%FT%TZ)"
        kill -TERM "$build_pid" 2>/dev/null || true
        wait "$build_pid" || true
        exit 75
    fi
    sleep 5
done
wait "$build_pid"

printf 'CPU_CHECK %s\n' "$(date -u +%FT%TZ)"
"$python" -B src/ci/check_distribution_image.py "$(cat image.id)" "$revision" \
    "$stage/src" "$stage/source.tar" "$base/driver/libcuda.so.1" "$stage/cpu-evidence" >cpu-check.log 2>&1
"$python" -B compare_candidate.py "$stage" "$base" --cpu-only >cpu-comparison.json

for mode in cache-on cache-off; do
    mkdir "$mode"
    check_holders "$mode/idle-before"
    cp -- "$mode/idle-before-6.json" "$mode/idle-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    result=0
    PYTHONPATH="$base/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$stage/cpu-evidence/report.json" "$gpu" \
        "$base/model" "$model" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode" \
        >"$mode/gpu-check.log" 2>&1 || result=$?
    check_holders "$mode/idle-after"
    cp -- "$mode/idle-after-6.json" "$mode/idle-after.json"
    printf '%d\n' "$result" >"$mode/exit"
    test -s "$mode/gpu-evidence/report.json"
    "$python" -B compare_candidate.py "$stage" "$base" >"$mode/comparison.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$result"
done
sha256sum cpu-comparison.json source-archive.sha256 source-inventory.json compare_candidate.py \
    cache-{on,off}/{exit,idle-before.json,idle-after.json,idle-before-7.json,idle-after-7.json,comparison.json} >result.sha256
