#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
baseline="$evidence/w5-dist-off-retry.ORBlgo3W"
allocator="$evidence/w5-tcache-only-20261119.G3IuT0Wh"
accuracy="$evidence/w4-decode-logits-20261119.8VHxJvdK"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
revision=d3d8fe6e5b1c528bff51c0008cdc24544c99f862
base_archive=cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5
patch_digest=28f0ef9ed0e5e2e81b5f82d68f99554977c461ca3cfe91ea929b5becb38fd520
model=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256

wait_for_campaign() {
    local run=$1 deadline=$((SECONDS + 14400))
    printf 'WAIT %s %s\n' "$run" "$(date -u +%FT%TZ)"
    until test -s "$run/campaign.exit"; do
        if ! ps -o args= -p "$(cat "$run/campaign.pid")" | grep -Fq -- "$run/campaign.sh"; then
            test -s "$run/campaign.exit" && break
            return 75
        fi
        if (( SECONDS >= deadline )); then return 75; fi
        sleep 30
    done
    printf 'PREDECESSOR_EXIT %s %s %s\n' "$run" "$(cat "$run/campaign.exit")" "$(date -u +%FT%TZ)"
}

wait_for_campaign "$allocator"
test "$(cat "$allocator/campaign.exit")" = 0
(cd "$allocator" && sha256sum -c result.sha256)
decision=$("$python" -B - "$allocator/cache-off/comparison.json" <<'PY'
import json
from pathlib import Path
import sys

modes = json.loads(Path(sys.argv[1]).read_bytes())["modes"]
if set(modes) != {"cache-on", "cache-off"}:
    raise ValueError("missing allocator control")
verdicts = [value["configurations"]["tcache_only"]["failed_checks"] for value in modes.values()]
if any(set(checks) - {"rss_growth_second_half_at_most_zero"} for checks in verdicts):
    raise ValueError("allocator control has a failure outside the RSS gate")
print("run" if any(verdicts) else "skip")
PY
)
printf '%s\n' "$decision" >allocator-decision.txt
if [[ "$decision" == skip ]]; then exit 0; fi
wait_for_campaign "$accuracy"
test "$(cat "$base/campaign.exit")" = 0

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

mkdir src cpu-evidence harness
printf '%s  %s\n' "$base_archive" "$base/source.tar" "$patch_digest" reserve-validated-body.patch.gz | sha256sum -c
tar -xf "$base/source.tar" -C src --no-same-owner
(
    cd src
    http=crates/riley-server/src/http.rs
    printf '%s  %s\n' c0d1b93d7605ab8d6fd55ea20756fd70c9975cd080144cabbea72a6dc57c7c79 "$http" | sha256sum -c
    gzip -dc "$stage/reserve-validated-body.patch.gz" | git apply --check -
    gzip -dc "$stage/reserve-validated-body.patch.gz" | git apply -
    printf '%s  %s\n' 617f26a7d9333e44f8265666868a032ea539f303cd54ccac9edb408c74e5aef6 "$http" | sha256sum -c
    "$python" -B ci/check_serving_source_contract.py
    tar --sort=name --mtime='@0' --owner=0 --group=0 --numeric-owner -cf "$stage/source.tar" .
)
archive=$(sha256sum source.tar | cut -d ' ' -f 1)
printf '%s\n' "$archive" >source-archive.sha256
cp -r -- "$baseline/harness/"{ci,benchmarks} harness/
sha256sum -c harness.sha256

docker_root=$(docker info --format '{{ .DockerRootDir }}')
available=$(df --output=avail -B1 "$docker_root" | tail -n 1)
(( available >= 4 * 1024 * 1024 * 1024 ))
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
docker buildx build --load --platform linux/amd64 --progress plain   --file "$stage/src/deploy/docker/Dockerfile"   --build-arg CARGO_BUILD_JOBS=2 --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2   --label "org.opencontainers.image.revision=$revision"   --label "io.riley.source-archive-sha256=$archive"   --label "io.riley.diagnostic-patch-sha256=$patch_digest"   --iidfile "$stage/image.id" --tag "riley:a100-http-framing-${stage##*.}" "$stage/src" >build.log 2>&1 &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
    available=$(df --output=avail -B1 "$docker_root" | tail -n 1)
    if (( available < 4 * 1024 * 1024 * 1024 )); then
        printf 'BUILD_STOP %s docker_root_avail_bytes=%s\n' "$(date -u +%FT%TZ)" "$available"
        kill -TERM "$build_pid" 2>/dev/null || true
        wait "$build_pid" || true
        exit 75
    fi
    sleep 5
done
wait "$build_pid"

printf 'CPU_CHECK %s\n' "$(date -u +%FT%TZ)"
"$python" -B src/ci/check_distribution_image.py "$(cat image.id)" "$revision"     "$stage/src" "$stage/source.tar" "$base/driver/libcuda.so.1" "$stage/cpu-evidence" >cpu-check.log 2>&1
"$python" -B compare_candidate.py "$stage" "$base" --cpu-only >cpu-comparison.json

for mode in cache-on cache-off; do
    mkdir "$mode"
    holder_status 6 >"$mode/idle-before.json"
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    result=0
    PYTHONPATH="$base/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run --         "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$stage/cpu-evidence/report.json" "$gpu"         "$base/model" "$model" "$stage/$mode/gpu-evidence" --tools --mixed-soak "$mode"         >"$mode/gpu-check.log" 2>&1 || result=$?
    holder_status 6 >"$mode/idle-after.json"
    printf '%d\n' "$result" >"$mode/exit"
    test -s "$mode/gpu-evidence/report.json"
    "$python" -B compare_candidate.py "$stage" "$base" >"$mode/comparison.json"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$result"
done
sha256sum cpu-comparison.json source-archive.sha256 cache-{on,off}/{exit,idle-before.json,idle-after.json,comparison.json} >result.sha256
