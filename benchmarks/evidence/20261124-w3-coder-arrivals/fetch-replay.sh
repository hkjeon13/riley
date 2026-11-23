#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-arrivals-20261124.ac0dqlbz
bundle=${1:?provide the result archive filename}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    30eec1c4ca125c5ca9adf44201a8ab52189c88bd92337ab172ecae3e91cbe3d8 dispatch.sha256 \
    6de0f89924fd12ad74ca82f57ffecf44e3f73e4c098a16fff03366cc495007e7 runtime.sha256 |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 schedule.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log dispatch.sha256 runtime.sha256 result.sha256 schedule.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256 schedule.sha256
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
