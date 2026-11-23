#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-structured-arithmetic-20261124.MnfOAWtz
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" <<'REMOTE' >"$bundle"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' 4b1cc11b3d60d1093abd1842a1fb64a81a1408aad35850e54512d1ec78354cc7 dispatch.sha256 | sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log campaign.pid campaign.started dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256 | sed 's#^\./##'
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
