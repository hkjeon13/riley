#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-uri-package-20261124.73IfvPre
dest=${1:?provide an existing destination directory}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" <<'REMOTE' | tar -xf - -C "$dest"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' \
    e3a322d9c39c1634b9afab6b67bae3137e9630f7eb64cd9a67c7ce9dbba43b3b dispatch.sha256 \
    b1111fd11109c5b70d3381c977d20ea5b09daf6d8f0aa9c6f28bdda52eaaf489 source.tar | sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 result.sha256 >&2
tar -cf - campaign.exit campaign.log campaign.pid campaign.started result.sha256 image.id source-revision.txt \
    cpu-evidence/{report.json,runtime.sha256} structured/{pattern-classes,schema-references,report,shutdown,api}.json \
    smoke/{report,shutdown,api,smoke}.json gpu{6,7}-{before,after}.json
REMOTE
