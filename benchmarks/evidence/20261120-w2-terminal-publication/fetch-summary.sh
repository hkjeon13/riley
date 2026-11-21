#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w2-terminal-order-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the full A100 evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$expected'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 reused.sha256 result.sha256 >/dev/null
cmp verified.json <(/usr/bin/python3 -B verify_publication.py "$1")
cat verified.json
REMOTE
