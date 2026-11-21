#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-current-package-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact A100 package evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$expected'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 runtime-tools.sha256 result.sha256 >/dev/null
python=$(dirname "$(dirname "$1")")/tools/vllm-venv/bin/python
cmp verified.json <("$python" -B verify_package.py "$1")
"$python" -B - <<'PY'
from pathlib import Path
from base64 import b64encode
import json
parts = ("verified.json", "cpu-evidence/report.json", "cpu-evidence/runtime.sha256")
print(json.dumps({part: b64encode(Path(part).read_bytes()).decode("ascii") for part in parts}))
PY
REMOTE
