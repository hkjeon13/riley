#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
stage=$1
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
old="$evidence/w5-allocator-image-perf-20261120.tmZw0Cpo"
driver="$evidence/w5-allocator-packaged-gpu-20261120.nQpLFMUS"
package="$evidence/w5-allocator-image-recheck-20261120.qPXAvyfv"
native="$evidence/w5-jemalloc-20261120.hLPizqfG"
accuracy="$evidence/w5-jemalloc-accuracy-20261120.CJrnRg9W"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
harness="$evidence/w5-dist-off-retry.ORBlgo3W/harness"
export PYTHONPATH="$old:$driver:$package:$accuracy:$native:$harness/ci:$base/src/ci:$harness/benchmarks/live"
python="$root/tools/vllm-venv/bin/python"
cmp verified.json <("$python" -B route_drill.py verify "$package" "$evidence" "$stage/route")
"$python" -B - "$stage" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys
from route_drill import ORDER, require

stage=Path(sys.argv[1])
raw=(stage/"verified.json").read_bytes()
lines=[line.split() for line in (stage/"campaign.log").read_text().splitlines()]
chronology=[row for row in lines if row and row[0] in ("START", "END")]
expected=[(event,arm) for arm in ORDER for event in ("START","END")]
require([tuple(row[:2]) for row in chronology]==expected and
        all(row[3:]==["exit=0"] for row in chronology[1::2]) and
        [row[2] for row in chronology]==sorted(row[2] for row in chronology), "incomplete recovery runs")
print(json.dumps({"result":json.loads(raw), "execution":{
    "stage":stage.name, "campaign_exit":0, "chronology":chronology, "physical_gpu":6,
    "remote_verified_json_sha256":sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified":True,
    "remote_independent_comparison_byte_exact":True,
}}, indent=2,allow_nan=False))
PY
