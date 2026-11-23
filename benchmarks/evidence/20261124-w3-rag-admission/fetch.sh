#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-rag-admission-retry-20261124.OaUPAucl
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    430ba293b4b9ee713c04c4b5c6e50239259a2aa318f9ee2cef793acfb0adccd8 build-dispatch.sha256 \
    4deade39fe1cc359448d765015597b360ca2c6ac32eaea44b022b6b7498d1441 campaign-dispatch.sha256 |
    sha256sum --strict -c >&2
sha256sum --strict -c build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' build.exit build.log build.pid build.started build-dispatch.sha256 build-result.sha256 \
        campaign.exit campaign.log campaign.pid campaign.started campaign-dispatch.sha256 compiler.log runtime.sha256 result.sha256
    cut -c 67- build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
