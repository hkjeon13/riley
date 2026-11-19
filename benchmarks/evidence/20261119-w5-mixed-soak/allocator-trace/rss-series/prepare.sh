#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
baseline="$evidence/w5-dist-off-retry.ORBlgo3W"
python="$root/tools/vllm-venv/bin/python"
sha256sum -c dispatch.sha256
mkdir harness
cp -r -- "$baseline/harness/"{ci,benchmarks} harness/
sha256sum -c harness-base.sha256
"$python" -B - <<'PY'
from pathlib import Path
path = Path("harness/ci/check_distribution_gpu.py")
source = path.read_text()
old = 'SOAK_CHECK = API_CHECK.with_name("soak.py")'
new = 'SOAK_CHECK = API_CHECK.with_name("observe_rss_series.py")'
if source.count(old) != 1:
    raise ValueError("unexpected GPU checker entry point")
path.write_text(source.replace(old, new))
PY
cp -- observe_rss_series.py harness/benchmarks/live/
sha256sum harness/ci/check_distribution_gpu.py harness/benchmarks/live/{soak,load,api_check,observe_rss_series}.py >harness-observer.sha256
export PYTHONPATH="$stage/harness/benchmarks/live"
"$python" -B smoke_rss_series.py >cpu-smoke.log
image=$(cat "$base/image.id")
container=$(docker run -d --pull=never --rm --network none --user 65532:65532 \
    --env NVIDIA_VISIBLE_DEVICES=void --entrypoint /bin/sleep "$image" 45)
trap 'docker stop --time 1 "$container" >/dev/null 2>&1 || true' EXIT
"$python" -B - "$container" "$image" >cpu-preflight.json <<'PY'
from functools import partial
import json
import subprocess
import sys
from observe_rss_series import read_sample
from soak import open_container_proc
import time

container, image = sys.argv[1:]
row, = json.loads(subprocess.check_output(["docker", "container", "inspect", container], text=True))
if row["Image"] != image or row["Config"]["User"] != "65532:65532" or not row["State"]["Running"]:
    raise ValueError("CPU preflight changed image, user or process")
if row["HostConfig"]["DeviceRequests"] or row["HostConfig"]["NetworkMode"] != "none":
    raise ValueError("the CPU read check must be isolated and use no GPU")
pid = row["State"]["Pid"]
reader = partial(open_container_proc, container)
sample = read_sample(pid, reader, 0, time.perf_counter())
if not 0 < sample["anonymous_kib"] <= sample["rss_kib"]:
    raise ValueError("invalid PID 1 memory values")
print(json.dumps({"image_id": image, "container_uid": 65532, "pid": 1, "gpu": False,
                  "pid1_read_ms": sample["read_ms"], "supplementary_probe_passed": True}, indent=2))
PY
sha256sum harness-observer.sha256 cpu-smoke.log cpu-preflight.json >preflight.sha256
