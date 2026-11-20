#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-chunk-exact-retry-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute remote stage for this campaign.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/verify_exact_batch.py" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
    set -eu
    cd '$1'
    test \"\$(cat campaign.exit)\" = 0
    sha256sum -c dispatch.sha256 result.sha256 >/dev/null
    printf '%s  verify_exact_batch.py\n' '$expected' | sha256sum -c >/dev/null
    /usr/bin/python3 -B - '$1'
" <<'PY'
import json
from pathlib import Path
import sys
from verify_exact_batch import ARMS, read, request_trace, verify

root = Path(sys.argv[1])
report = verify(root)
report["admission_timeout_ms"] = 90000
for mode, (_, capacity) in ARMS.items():
    stage = root / f"qwen14-{mode}"
    log = (stage / "server.log").read_text()
    serving = log.split("riley startup warm-up: ", 1)[1]
    mapping, traces = request_trace(log, capacity)
    order = [line.split()[1] for line in serving.splitlines()
             if line.startswith("W3-PHASE ") and " front_submit " in line]
    offset = 0
    for level, values in read(stage / "token-id-probe.json")["rows"].items():
        waits = []
        for key in order[offset:offset + len(values)]:
            trace = traces[mapping[key]]
            if trace[0][0] == "submit_queued":
                admitted_at = next(stamp for event, stamp, _, _ in trace if event == "admitted")
                waits.append(admitted_at - trace[0][1])
        offset += len(values)
        group = report["arms"][mode]["scheduler"]["by_client_concurrency"][level]
        assert len(waits) == group["requests_queued"] and all(0 <= ns < 90_000_000_000 for ns in waits)
        group["max_admission_wait_ms_queued_only"] = max(waits) / 1e6 if waits else None
        group["queued_longer_than_default_30s"] = sum(ns > 30_000_000_000 for ns in waits)
    assert offset == len(order)
print(json.dumps(report, indent=2, allow_nan=False))
PY
