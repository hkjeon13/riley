#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-rag-prefill-order-retry-20261124.cWPk8nNQ
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    b1155d555c934ec563b00544c080dc9da7ef42442cdf44ba34edc9917c2b33a1 build-dispatch.sha256 \
    1d573337167e8df0c3017fc065207ada2df5a97b4df9d07fddafdb6139655b6a build-result.sha256 \
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
