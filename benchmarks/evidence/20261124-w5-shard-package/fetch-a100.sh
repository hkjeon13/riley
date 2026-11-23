#!/usr/bin/env bash
set -euo pipefail

stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-shard-package-20261124.nhBcrYgP
dest=${1:?provide an existing destination directory}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" <<'REMOTE' | tar -xf - -C "$dest"
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 result.sha256 >&2
tar -cf - campaign.exit campaign.log result.sha256 image.id source-revision.txt \
    cpu-evidence/{report.json,runtime.sha256} structured/{report,shutdown,api}.json \
    smoke/{report,shutdown,api,smoke}.json gpu{6,7}-{before,after}.json
REMOTE
