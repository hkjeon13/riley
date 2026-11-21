#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
root=$(dirname "$(dirname "$stage")")
base="$root/evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
smoke="$root/evidence/w5-serving-smoke-recheck-20261120.PsNfm6KN"
tester="$root/evidence/w5-tp2-holdout-20261120.AFFbt3Is"
holder="$root/evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
commit=$(cat source-revision.txt)
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$base/campaign.exit")" = 0
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 "$base/source.tar" | sha256sum -c
printf 'SOURCE %s\n' "$(date -u +%FT%TZ)"
"$python" -B build_inputs.py "$base/source.tar" "$stage"
archive_sha=$(sha256sum source.tar | cut -d ' ' -f 1)
mkdir src cpu-evidence
tar -xf source.tar -C src --no-same-owner
(cd src && "$python" -B ci/check_serving_source_contract.py)
available=$(df --output=avail -B1 /var/lib/docker | tail -n 1)
(( available >= 4 * 1024 * 1024 * 1024 ))
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
docker buildx build --load --platform linux/amd64 --progress plain \
  --file "$stage/src/deploy/docker/Dockerfile" \
  --build-arg CARGO_BUILD_JOBS=2 --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2 \
  --label "org.opencontainers.image.revision=$commit" \
  --label "io.riley.source-archive-sha256=$archive_sha" \
  --iidfile "$stage/image.id" --tag "riley:a100-${commit:0:9}-${stage##*.}" "$stage/src" >build.log 2>&1 &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
    available=$(df --output=avail -B1 /var/lib/docker | tail -n 1)
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
"$python" -B src/ci/check_distribution_image.py "$(cat image.id)" "$commit" \
  "$stage/src" "$stage/source.tar" "$base/driver/libcuda.so.1" "$stage/cpu-evidence" >cpu-check.log 2>&1

mkdir -p harness/ci harness/benchmarks/live
cp -- "$tester/ci/check_distribution_gpu.py" harness/ci/
cp -- "$smoke/benchmarks/live/"{api_check,load,soak}.py harness/benchmarks/live/
sha256sum -c runtime-tools.sha256
printf '%s  %s\n' "$model_sha" "$base/model/riley-checkpoint.json" | sha256sum -c
test -f "$base/model/tokenizer.json"

reservations() {
    for index in 6 7; do
        /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status \
            >"gpu$index-$1.json"
    done
    "$python" -B verify_package.py --reservations "$1"
}
reservations before
printf 'GPU_CHECK %s\n' "$(date -u +%FT%TZ)"
rc=0
PYTHONPATH="$stage/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
  "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$stage/cpu-evidence/report.json" "$gpu" \
  "$base/model" "$model_sha" "$stage/gpu-evidence" --tools --serving-smoke --smoke-kv-blocks 16384 >gpu-check.log 2>&1 || rc=$?
reservations after
printf 'GPU_CHECK_END %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B verify_package.py "$stage" >verified.json
sha256sum verified.json cpu-evidence/report.json gpu-evidence/{report,api,smoke,shutdown}.json gpu{6,7}-{before,after}.json >result.sha256
