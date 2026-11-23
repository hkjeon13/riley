#!/usr/bin/env bash
set -euo pipefail

stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-shard-readiness-20261124.Jdlojvu8
bundle=${1:?provide the destination archive path}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 result.sha256 runtime-tools.sha256 server-binaries.sha256 >&2
tar -czf - campaign.log campaign.exit dispatch.sha256 result.sha256 affinity.txt server-binaries.sha256 \
    runtime-tools.sha256 ldd-{baseline,candidate}.txt gpu{6,7}/*.json gpu{6,7}/{baseline,candidate}/*.json
REMOTE
