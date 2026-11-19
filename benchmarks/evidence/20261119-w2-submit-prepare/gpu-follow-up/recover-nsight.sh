#!/usr/bin/env bash
set -euo pipefail
umask 077
evidence=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence
stage="$evidence/w2-ordered-tables-20261119.e89GGsLO"
cd "$stage"
test "$(cat campaign.exit)" = 1
grep -Eq '^END ordered-trace [^ ]+ exit=1$' campaign.log
candidate=smol-ordered-trace
control=smol-base-trace
pid=$(cat "$candidate/server.pid")
grep -Fq "RuntimeError: unexpected process on selected GPU pid=$pid" campaign.log
test -s "$candidate/trace.nsys-rep"
grep -Fq 'Generated:' "$candidate/client.log"
nsys=${evidence%/evidence}/tools/nsys-2026/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys
python=/usr/bin/python3
"$python" -B - "$nsys" "$control/receipt.json" "$candidate/receipt.json" <<'PY'
from pathlib import Path
import hashlib,json,sys
nsys=Path(sys.argv[1]); digest=hashlib.sha256(nsys.read_bytes()).hexdigest()
for source in sys.argv[2:]:
    receipt=json.loads(Path(source).read_bytes())
    assert receipt['sha256'][str(nsys)]==digest
PY
recovery=$(mktemp -d /var/tmp/riley-w2-ordered-export.XXXXXXXX)
sha256sum "$candidate/trace.nsys-rep" "$control/trace.sqlite" >"$recovery/input.sha256"
nice -n 10 timeout 160 "$nsys" export --type sqlite --output "$recovery/ordered.sqlite" "$candidate/trace.nsys-rep" >"$recovery/export.log" 2>&1
"$python" -B tools/profile_summary.py "$recovery/ordered.sqlite" "$candidate/reason-c32.json" "$pid" "$recovery/ordered-clock.json"
"$python" -B summarize_sync_gaps.py "$recovery/ordered.sqlite" >"$recovery/ordered-sync-gaps.json"
"$python" -B summarize_sync_gaps.py "$control/trace.sqlite" >"$recovery/base-sync-gaps.json"
sha256sum -c "$recovery/input.sha256" >&2
printf '%s\n' "$recovery"
