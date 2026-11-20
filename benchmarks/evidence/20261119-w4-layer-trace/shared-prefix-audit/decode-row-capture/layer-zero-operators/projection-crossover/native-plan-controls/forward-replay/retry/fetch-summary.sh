#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
/usr/bin/python3 -B - <<'PY'
from hashlib import sha256
import json
from pathlib import Path

root = Path.cwd()
raw = (root / "verified.json").read_bytes()
report = json.loads(raw)
assert report == json.loads((root / "riley/client.log").read_bytes())
receipt = json.loads((root / "riley/receipt.json").read_bytes())
assert receipt["status"] == "passed" and receipt["server_exit"] == 0
reservations = {}
for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    configurations = []
    for moment in ("before", "after"):
        row = json.loads((root / f"gpu{index}-{moment}.json").read_bytes())
        assert (row["physical_index"], row["phase"], row["pid"], row["start_ticks"]) == (index, "idle", pid, ticks)
        assert row["allocation_bytes"] == row["torch_reserved_bytes"] == 20 * 1024**3 and not row["other_processes"]
        configurations.append({key: row[key] for key in ("physical_index", "pid", "start_ticks", "allocation_bytes")})
    assert configurations[0] == configurations[1]
    reservations[str(index)] = configurations[0]
print(json.dumps({"result":report, "execution": {
    "stage": root.name, "physical_gpu": 7, "campaign_exit": 0,
    "remote_verified_json_sha256": sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified": True,
    "binary_sha256": (root / "binary.sha256").read_text().split()[0],
    "chronology": [line for line in (root / "campaign.log").read_text().splitlines()
                   if line.startswith(("BUILD ", "START ", "END "))],
    "reservations": reservations,
    "prior_attempt": json.loads((root / "provenance.json").read_bytes()),
}}, indent=2, allow_nan=False))
PY
