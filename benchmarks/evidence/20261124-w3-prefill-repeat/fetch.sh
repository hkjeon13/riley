#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-prefill-repeat-20261124.TzItVDGJ
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    ba24e83c118ffb272a1b20b72a4ff994ae701a33a06a7bd5ce9ed0070b74857d dispatch.sha256 \
    d7d73cb614f322713c164c082ffda43776b8c8a1516084c8f49f51eb67535be7 riley-diagnostic |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log campaign.pid campaign.started dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
