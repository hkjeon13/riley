#!/usr/bin/env bash
set -euo pipefail
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-arrivals-20261124.ac0dqlbz
bundle=${1:?provide the result archive filename}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 schedule.sha256 >&2
tar -czf - campaign.exit campaign.log dispatch.sha256 runtime.sha256 result.sha256 schedule.sha256 \
    schedules.json vllm-version.txt gpu{6,7}/*.json gpu{6,7}/coder14-{riley,vllm}/*.json
REMOTE
