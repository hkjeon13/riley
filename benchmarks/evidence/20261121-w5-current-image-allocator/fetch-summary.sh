#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-current-allocator-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact allocator validation directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$expected'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 context.sha256 result.sha256 >/dev/null
evidence=$(dirname "$1")
python=$(dirname "$evidence")/tools/vllm-venv/bin/python
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
cpu="$stock/cpu-evidence/report.json"
export PYTHONPATH="$stock/harness/ci:$stock/src/ci"
cmp verified.json <("$python" -B qualify.py verify "$1" "$cpu")
if "$python" -B -c 'import json; raise SystemExit(json.load(open("verified.json"))["release_gate"]!="passed")'; then
    sha256sum -c accuracy-result.sha256 >/dev/null
    cmp verified-accuracy.json <("$python" -B qualify.py verify-accuracy "$1" "$cpu")
    "$python" -B - "$1" <<'PY'
from pathlib import Path
import sys
from verify_soaks import reservations
p=Path(sys.argv[1])
assert reservations(p,"before")==reservations(p,"after-accuracy")
PY
fi
"$python" -B - <<'PY'
from base64 import b64encode
import json
from pathlib import Path
names=("verified-image.json","verified.json","verified-accuracy.json")
print(json.dumps({name:b64encode(Path(name).read_bytes()).decode("ascii") for name in names if Path(name).is_file()}))
PY
REMOTE
