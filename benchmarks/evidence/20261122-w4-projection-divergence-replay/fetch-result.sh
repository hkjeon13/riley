#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-projection-divergence-replay-20261122\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact projected-divergence replay evidence stage.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
(cd .. && sha256sum --strict -c \"\$1/captures.sha256\" >/dev/null)
tar -cf - replay.json replay.log checkpoint-verify.log result.sha256 gpu{6,7}-{before,after}.json
' bash '$1' '$dispatch'"
