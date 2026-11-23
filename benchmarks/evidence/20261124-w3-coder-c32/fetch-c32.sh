#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-c32-20261124.n6DRELdL
bundle=${1:?provide the result archive filename}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    4f98597aa8931e1f835287a6aab64c8db2e5297ee2629c82d6511a2efd3de3c0 dispatch.sha256 \
    e061b5399da890f08d53dd2e322a0d0f706287d4a10932fab99fd0b7b137dde4 schedules.json |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
