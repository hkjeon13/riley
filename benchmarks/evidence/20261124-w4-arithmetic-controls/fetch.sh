#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-arithmetic-controls-20261124.A9VIzLu6
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    d824c7f4224f8927380248a87372c450e921e35259cc32e6589c3c539316ee66 dispatch.sha256 \
    a7b2dbca08cdbc50477665cd930d4e21c2b8f0bd0f32df4e904599c18a2557de inputs.json |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 launchers.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log campaign.pid campaign.started dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256 | sed 's#^\./##'
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
