#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
off="$evidence/w5-dist-off-retry.ORBlgo3W"
cd "$stage"
sha256sum -c dispatch.sha256
test "$(cat "$off/campaign.exit")" = 1
test -s "$off/gpu-evidence/report.json"
mkdir -p harness/ci harness/benchmarks/live
cp -- "$off/harness/ci/check_distribution_gpu.py" harness/ci/
cp -- "$off/harness/benchmarks/live/"{load,soak}.py harness/benchmarks/live/
cp -- "$base/src/benchmarks/live/api_check.py" harness/benchmarks/live/
sha256sum -c harness-base.sha256
git apply --check harness.patch
git apply harness.patch
sha256sum -c harness.sha256
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread trim_probe.c -o w5-trim-probe.so
cc -std=c11 -O2 -Wall -Wextra -Werror trim_smoke.c -o w5-trim-smoke
chmod 0644 w5-trim-probe.so container-smoke.sh
chmod 0755 w5-trim-smoke
sha256sum w5-trim-probe.so w5-trim-smoke >compiled.sha256
image=$(cat "$base/image.id")
timeout 60 docker run --rm --pull=never --network none --entrypoint /bin/sh \
    --env NVIDIA_VISIBLE_DEVICES=void --env LD_PRELOAD=/tmp/w5-trim-probe.so \
    --env RILEY_W5_TRIM_EXE=/tmp/w5-trim-smoke \
    --mount "type=bind,source=$stage/w5-trim-probe.so,target=/tmp/w5-trim-probe.so,readonly" \
    --mount "type=bind,source=$stage/w5-trim-smoke,target=/tmp/w5-trim-smoke,readonly" \
    --mount "type=bind,source=$stage/container-smoke.sh,target=/tmp/container-smoke.sh,readonly" \
    "$image" /tmp/container-smoke.sh >container-smoke.log 2>&1
cat container-smoke.log
