#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-pattern-class(-recovery)?-20261123\.[A-Za-z0-9]+$ || ! -d $2 || ! $3 =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Pass the A100 evidence directory, an existing output directory, and the dispatch hash.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$3" <<'REMOTE' | tar -xf - -C "$2"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
if [[ $1 == */w5-pattern-class-recovery-* ]]; then
    tar -cf - campaign.exit campaign.log result.sha256 verified.json
else
    tar -cf - campaign.exit campaign.log result.sha256 verified.json \
        allocator/{context.sha256,gpu-checker-input.json,verified-image.json,verified.json}
fi
REMOTE
