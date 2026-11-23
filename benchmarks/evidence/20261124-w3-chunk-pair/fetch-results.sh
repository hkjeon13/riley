#!/usr/bin/env bash
set -euo pipefail

stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-chunk-pair-20261124.Tl1oBWmS
bundle=${1:?provide a destination path for the result archive}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 runtime.sha256 arm.sha256 result.sha256 holdout-tool.sha256 first-attempt.sha256 >&2
tar -czf - arm.sha256 campaign.sh cells.json prefill-chunk.patch dispatch.sha256 \
    campaign.log campaign.exit result.sha256 runtime.sha256 checkpoint.log pins.sha256 first-attempt.sha256 holdout-tool.sha256 \
    gpu{6,7}/*.json gpu{6,7}/chunk-{1024,512}/*.json
REMOTE
