#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w5-prefix-metrics-build-20261121.5MLLtayu"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
python="$root/tools/vllm-venv/bin/python"
commit=$(cat source-revision.txt)
trap 'printf "%d\n" "$?" >build.exit.tmp; mv -- build.exit.tmp build.exit' EXIT
sha256sum -c build-inputs.sha256
test "$(cat "$previous/build.exit")" = 0
(cd "$previous/src" && sha256sum -c ../runtime-source.sha256 >/dev/null)
"$python" -B build_inputs.py "$previous/src" "$stage"
archive_sha=$(sha256sum source.tar | cut -d' ' -f1)
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
pid=$!
while kill -0 "$pid" 2>/dev/null; do
    available=$(df --output=avail -B1 /var/lib/docker | tail -n 1)
    if (( available < 4 * 1024 * 1024 * 1024 )); then
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || true
        exit 75
    fi
    sleep 5
done
wait "$pid"
"$python" -B src/ci/check_distribution_image.py "$(cat image.id)" "$commit" \
    "$stage/src" "$stage/source.tar" "$base/driver/libcuda.so.1" "$stage/cpu-evidence" >cpu-check.log 2>&1
LC_ALL=C sort hosted-runtime.sha256 >hosted-runtime.sorted
LC_ALL=C sort cpu-evidence/runtime.sha256 >a100-runtime.sorted
cmp hosted-runtime.sorted a100-runtime.sorted
test "$(wc -l <a100-runtime.sorted)" = 21
sha256sum source.tar image.id cpu-evidence/report.json cpu-evidence/runtime.sha256 a100-runtime.sorted >build-result.sha256
printf 'BUILD_COMPLETE %s\n' "$(date -u +%FT%TZ)"
