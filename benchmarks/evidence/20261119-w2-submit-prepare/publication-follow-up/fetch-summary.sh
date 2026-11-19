#!/usr/bin/env bash
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
prior=$(dirname "$1")/w2-ordered-tables-20261119.e89GGsLO
serving=$(dirname "$1")/w2-publication-gaps-20261120.aH6wKcia
for mode in immediate-first-trace after-first-trace after-second-trace immediate-second-trace; do
    cmp "smol-$mode/sync-gaps.json" <(/usr/bin/python3 -B summarize_sync_gaps.py "smol-$mode/trace.sqlite")
done
cmp verified.json <(/usr/bin/python3 -B verify_publication.py "$1" "$prior" "$serving")
/usr/bin/python3 -B - "$1" "$serving" <<'PY'
from hashlib import sha256
import json
import re
from pathlib import Path
import sys

stage=Path(sys.argv[1])
raw=(stage/"verified.json").read_bytes()
old=Path(sys.argv[2])/"smol-immediate-first-trace"
receipt=json.loads((old/"receipt.json").read_bytes())
failed_pid=int(re.search(r"unexpected process on selected GPU pid=(\d+)",receipt["error"]).group(1))
request=json.loads((old/"reason-c32.json").read_bytes())["statistics"]
failure={
    "stage":old.parent.name,
    "failed_pid":failed_pid,
    "matches_own_server_pid":failed_pid==int((old/"server.pid").read_text()),
    "successful_measured_requests":request["ok"], "errors":request["errors"],
    "trace_retained":(old/"trace.nsys-rep").is_file(),
    "shutdown_receipt_seen":(old/"server-exit.json").is_file(),
    "controller_exit":int((old.parent/"campaign.exit").read_text()),
    "receipt_sha256":sha256((old/"receipt.json").read_bytes()).hexdigest(),
}
preflight=json.loads((stage/"owner-preflight.json").read_bytes())
print(json.dumps({"result":json.loads(raw), "initial_trace":failure, "child_binding_preflight":preflight, "execution":{
    "stage":stage.name, "campaign_exit":0, "physical_gpu":7,
    "remote_verified_json_sha256":sha256(raw).hexdigest(),
    "remote_dispatch_and_result_digests_verified":True,
    "remote_independent_comparison_byte_exact":True,
    "remote_independent_sqlite_summaries_byte_exact":True,
}}, indent=2,allow_nan=False))
PY
