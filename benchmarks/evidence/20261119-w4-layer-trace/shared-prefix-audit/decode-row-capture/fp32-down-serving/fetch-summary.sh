#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-fp32-down-serving-20261120\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the absolute FP32 down-projection serving stage.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
expected=$(shasum -a256 "$here/verify_candidate.py" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "
set -eu
cd '$1'
test \"\$(cat campaign.exit)\" = 0
sha256sum -c dispatch.sha256 candidate-source.sha256 result.sha256 >/dev/null
printf '%s  verify_candidate.py\n' '$expected' | sha256sum -c >/dev/null
PYTHONPATH=tools /mnt/efs/nlp/psyche/riley-gap-20261102/tools/vllm-venv/bin/python -B verify_candidate.py '$1'
"
