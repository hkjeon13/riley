#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
off="$evidence/w5-dist-off-retry.ORBlgo3W"
cd "$stage"
sha256sum -c dispatch.sha256
test "$(cat "$base/campaign.exit")" = 0
test -s "$off/gpu-evidence/report.json"
mkdir -p harness/ci harness/benchmarks/live
cp -- "$off/harness/ci/check_distribution_gpu.py" harness/ci/
cp -- "$off/harness/benchmarks/live/"{load,soak}.py harness/benchmarks/live/
cp -- "$base/src/benchmarks/live/api_check.py" harness/benchmarks/live/
sha256sum -c harness-base.sha256
git apply --check harness.patch
git apply harness.patch
sha256sum -c harness.sha256
cc -std=c11 -O2 -Wall -Wextra -Werror -fPIC -shared -pthread periodic_trim.c -o w5-periodic-trim.so
cc -std=c11 -O2 -Wall -Wextra -Werror trim_smoke.c -o w5-trim-smoke
chmod 0644 w5-periodic-trim.so container-smoke.sh
chmod 0755 w5-trim-smoke
sha256sum w5-periodic-trim.so w5-trim-smoke >compiled.sha256
image=$(cat "$base/image.id")
timeout 60 docker run --rm --pull=never --network none --entrypoint /bin/sh \
    --env NVIDIA_VISIBLE_DEVICES=void --env LD_PRELOAD=/tmp/w5-periodic-trim.so \
    --env RILEY_W5_PERIODIC_TRIM_EXE=/tmp/w5-trim-smoke --env RILEY_W5_PERIODIC_TRIM_MS=400 \
    --mount "type=bind,source=$stage/w5-periodic-trim.so,target=/tmp/w5-periodic-trim.so,readonly" \
    --mount "type=bind,source=$stage/w5-trim-smoke,target=/tmp/w5-trim-smoke,readonly" \
    --mount "type=bind,source=$stage/container-smoke.sh,target=/tmp/container-smoke.sh,readonly" \
    "$image" /tmp/container-smoke.sh >container-smoke.log 2>&1
cat container-smoke.log
"$root/tools/vllm-venv/bin/python" -B - <<'PY'
from pathlib import Path
import json

from periodic_trim import read_telemetry

lines = Path("container-smoke.log").read_text().splitlines()
scratch = Path("preflight-trims.jsonl")
scratch.write_text("\n".join(lines[2:]) + "\n")
assert lines[0] == "glibc 2.35"
observed = {key: int(value) for word in lines[1].split() for key, value in [word.split("=")]}
trim = read_telemetry(scratch, observed.pop("pid"), 400, 1)
assert trim["calls"] >= 4 and trim["returned_memory"] > 0
assert observed["anonymous_before_kib"] - observed["anonymous_after_kib"] >= 4096
print(json.dumps({**trim, **observed}))
PY
