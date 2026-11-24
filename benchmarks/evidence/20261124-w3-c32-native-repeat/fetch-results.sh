#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 3 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-native-repeat-20261124\.[A-Za-z0-9]+$ ||
      ! $2 =~ ^[a-f0-9]{64}$ || ! $3 =~ ^[0-9]+$ ]]; then
    printf '%s\n' 'Pass the registered stage, dispatch digest and campaign PID.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$2" "$3" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.pid)" = "$3"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.started campaign.pid campaign.log dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256 | grep -Fxv riley-diagnostic
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
