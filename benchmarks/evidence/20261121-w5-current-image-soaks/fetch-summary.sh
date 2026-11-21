#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-current-soaks-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact A100 validation directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$expected'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
python=$(dirname "$(dirname "$1")")/tools/vllm-venv/bin/python
cmp verified.json <("$python" -B verify_soaks.py "$1")
cat verified.json
REMOTE
