#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-prefill-step-cost-20261124\.[A-Za-z0-9]+$ || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Pass the A100 evidence directory and the build-dispatch SHA-256.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$2" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0 && test "$(cat campaign.exit)" = 0
printf '%s  build-dispatch.sha256\n' "$2" | sha256sum --strict -c >&2
sha256sum --strict -c build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' build.exit build.started build.pid build.log campaign.exit campaign.started campaign.pid campaign.log \
        build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- build-dispatch.sha256 build-result.sha256 campaign-dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
