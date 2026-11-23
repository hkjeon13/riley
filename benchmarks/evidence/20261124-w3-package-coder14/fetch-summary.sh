#!/usr/bin/env bash
set -euo pipefail

# Run from the repository checkout with the original A100 SSH alias available.
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-package-coder14-20261124.GgTQVni3
bundle=${1:?provide a destination path for the result archive}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "cd '$stage' && bash -s" >"$bundle" <<'REMOTE'
set -euo pipefail
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 >&2
sha256sum --strict -c runtime.sha256 >&2
sha256sum --strict -c result.sha256 >&2
sha256sum --strict -c first-attempt.sha256 >&2
tar -czf - arm.py campaign.sh cells.json dispatch.sha256 pins.sha256 \
    campaign.log campaign.exit result.sha256 runtime.sha256 first-attempt.sha256 \
    checkpoint.log vllm-version.txt gpu{6,7}/*.json gpu{6,7}/coder14-{riley,vllm}/*.json
REMOTE
