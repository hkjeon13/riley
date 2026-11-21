#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-phase-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact Coder-14B phase evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch_sha=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
analysis_sha=$(shasum -a 256 "$here/analysis.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$dispatch_sha' '$analysis_sha'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n%s  analysis/analysis.sha256\n' "$2" "$3" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 used.sha256 result.sha256 >/dev/null
(cd analysis && sha256sum -c analysis.sha256 >/dev/null)
evidence=$(dirname "$1")
trace="$evidence/w3-coder-wave-trace-20261121.Z0ZQ8NvS"
history="$evidence/w3-coder14-qkv-20261120.Mx2OpwKm"
(cd "$trace" && sha256sum -c result.sha256 >/dev/null)
(cd "$history" && sha256sum -c result.sha256 >/dev/null)
export PYTHONPATH="$1:$1/analysis:$trace/analysis:$history"
/usr/bin/python3 -B - "$1" "$trace" <<'PYTHON'
import json
from pathlib import Path
import sys

from compare_profile_arms import digest
from verify_phase_gpu import verify

stage, reference = map(Path, sys.argv[1:])
result = verify(stage, reference)
result["remote_stage"] = stage.name
result["source_stage"] = reference.name
result["source_receipt_sha256"] = digest(stage / "analysis/source-trace-report.json")
print(json.dumps(result, indent=2, allow_nan=False))
PYTHON
REMOTE
