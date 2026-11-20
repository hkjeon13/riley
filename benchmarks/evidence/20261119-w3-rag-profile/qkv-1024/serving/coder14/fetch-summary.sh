#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
/usr/bin/python3 -B - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

from compare_coder import compare

root = Path.cwd()
raw = (root / "verified.json").read_bytes()
result = compare(root)
assert json.loads(raw) == json.loads(json.dumps(result, allow_nan=False))
print(json.dumps({"result": result, "execution": {
    "stage": root.name,
    "campaign_exit": 0,
    "remote_verified_json_sha256": sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified": True,
    "remote_independent_comparison_equal": True,
}}, indent=2, allow_nan=False))
PY
