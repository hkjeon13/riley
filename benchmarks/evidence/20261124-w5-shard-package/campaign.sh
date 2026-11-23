#!/usr/bin/env bash
set -euo pipefail
umask 077
cd -- "$(dirname -- "$0")"
stage=$(pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
old="$evidence/w4-schema-ids-20261123.ZsUYnqkm"
patch_at="$evidence/w5-parallel-shards-20261124.iMsJcmnF"
native="$evidence/w5-shard-readiness-20261124.Jdlojvu8"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
model_sha=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
holdout="$base/src/benchmarks/evidence/20261119-w4-holdout/holdout.json.gz"
commit=$(cat source-revision.txt)
trap 'printf "%s\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
test "$(cat "$native/campaign.exit")" = 0
(cd "$patch_at" && sha256sum --strict -c dispatch.sha256)
(cd "$prior" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
printf '%s  %s\n' \
    7dc88437ba28ea18ca1c0c11b9b3cd85324110da913ce1b10c521da60ce54dcb "$old/source.tar" \
    220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11 "$holdout" \
    "$model_sha" "$base/model/riley-checkpoint.json" | sha256sum --strict -c
mkdir overlay src cpu-evidence
tar -xf "$old/source.tar" -C overlay --no-same-owner
(cd overlay && git apply --check "$patch_at/overlap.patch" && git apply "$patch_at/overlap.patch" &&
    sha256sum --strict -c "$patch_at/candidate-inputs.sha256")
"$python" -B build_inputs.py "$stage/overlay" "$stage"
archive_sha=$(sha256sum source.tar | cut -d' ' -f1)
tar -xf source.tar -C src --no-same-owner
(cd src && "$python" -B ci/check_serving_source_contract.py)
available=$(df --output=avail -B1 /var/lib/docker | tail -n 1)
(( available >= 4 * 1024 * 1024 * 1024 ))
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
docker buildx build --load --platform linux/amd64 --progress plain --file "$stage/src/deploy/docker/Dockerfile" \
    --build-arg CARGO_BUILD_JOBS=2 --build-arg CMAKE_BUILD_PARALLEL_LEVEL=2 \
    --label "org.opencontainers.image.revision=$commit" --label "io.riley.source-archive-sha256=$archive_sha" \
    --iidfile "$stage/image.id" --tag "riley:a100-${commit:0:9}-${stage##*.}" "$stage/src" >build.log 2>&1 &
build_pid=$!
while kill -0 "$build_pid" 2>/dev/null; do
    available=$(df --output=avail -B1 /var/lib/docker | tail -n 1)
    if (( available < 4 * 1024 * 1024 * 1024 )); then
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
export PYTHONPATH="$prior/harness/ci:$stage/src/ci"
control=(/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases")
run_arm() {
    local index=$1 uuid=$2 name=$3 rc=0
    shift 3
    "${control[@]}" --index "$index" status >"gpu$index-before.json"
    printf 'START %s gpu%s %s\n' "$name" "$index" "$(date -u +%FT%TZ)"
    "${control[@]}" --index "$index" run -- "$python" -B "$prior/harness/ci/check_distribution_gpu.py" \
        "$stage/cpu-evidence/report.json" "$uuid" "$base/model" "$model_sha" "$stage/$name" \
        --tools "$@" >"$name.log" 2>&1 || rc=$?
    "${control[@]}" --index "$index" status >"gpu$index-after.json"
    printf 'END %s gpu%s %s exit=%s\n' "$name" "$index" "$(date -u +%FT%TZ)" "$rc"
    return "$rc"
}
run_arm 6 GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7 structured --structured-holdout "$holdout" &
first=$!
run_arm 7 GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0 smoke --serving-smoke --smoke-kv-blocks 16384 --require-prefix-cache-metrics &
second=$!
rc=0
wait "$first" || rc=1
wait "$second" || rc=1
test "$rc" = 0
sha256sum image.id source.tar cpu-evidence/{report.json,runtime.sha256} \
    structured/{report,shutdown,api}.json smoke/{report,shutdown,api,smoke}.json gpu{6,7}-{before,after}.json >result.sha256
