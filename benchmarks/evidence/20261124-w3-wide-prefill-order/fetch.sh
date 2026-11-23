#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-wide-prefill-order-20261124.cy7XlWYP
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    2c4bcbb4136bd68efcfaef4aa4c7b2a88fdbcd916d935299ec676020d92ecbd2 build-dispatch.sha256 \
    92f3c2dbe20a2819ca2b9390300bab755ade2ddfa36d0e2fb51d96d639bc19bb build-result.sha256 \
    363b6b74595a877213c081c1e04e0f2f4636544c95074d091632fe98f6139856 campaign-dispatch.sha256 \
    7921e63069ddd4b58736558af62590b5f340083ee129d25aa54eeefbdf78c116 schedules.json |
    sha256sum --strict -c >&2
sha256sum --strict -c build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' build.exit build.log build.pid build.started build-dispatch.sha256 build-result.sha256 \
        campaign.exit campaign.log campaign.pid campaign.started campaign-dispatch.sha256 compiler.log runtime.sha256 result.sha256
    cut -c 67- build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
