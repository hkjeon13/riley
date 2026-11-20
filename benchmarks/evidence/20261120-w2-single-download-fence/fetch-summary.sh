#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 selected-tools.sha256 candidate.sha256 >/dev/null
prior=$(dirname "$1")/w2-ordered-tables-20261119.e89GGsLO
cmp verified.json <(/usr/bin/python3 -B verify_fence.py "$1" "$prior")
/usr/bin/python3 -B - "$1" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

stage=Path(sys.argv[1])
raw=(stage/"verified.json").read_bytes()
print(json.dumps({"result":json.loads(raw), "execution":{
    "stage":stage.name, "campaign_exit":0, "physical_gpu":7,
    "remote_verified_json_sha256":sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified":True,
    "remote_independent_comparison_byte_exact":True,
}}, indent=2,allow_nan=False))
PY
