#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-long-answers-retry-20261124.cy31E4w3
bundle=${1:?provide the result archive filename}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' b391dca31724af297eef01cae351fb2bb76dd10c679a99768b9694eedf783ad0 dispatch.sha256 |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 result.sha256 launchers.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log dispatch.sha256 result.sha256 vllm/server.log vllm/client.log
    cut -c 67- dispatch.sha256 result.sha256
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
