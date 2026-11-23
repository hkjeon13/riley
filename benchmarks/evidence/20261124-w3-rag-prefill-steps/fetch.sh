#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-rag-prefill-steps-20261124.JGRVbNZq
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    b9e7da84d97479243899488b4273431fc15b285c218af2a7b0e2084bc943f452 build-dispatch.sha256 \
    01ea5e0d5513b29d87a29db65b658f3fc97effa7dd0e5869bcd3578911fc9680 campaign-dispatch.sha256 |
    sha256sum --strict -c >&2
sha256sum --strict -c build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' build.exit build.log build.pid build.started build-dispatch.sha256 build-result.sha256 \
        campaign.exit campaign.log campaign.pid campaign.started campaign-dispatch.sha256 compiler.log runtime.sha256 result.sha256
    cut -c 67- build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
