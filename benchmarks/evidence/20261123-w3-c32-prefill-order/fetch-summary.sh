#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-prefill-order-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact A100 experiment and an existing output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | awk '{print $1}')
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" <<'REMOTE' | tar -xf - -C "$2"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
tar -cf - campaign.exit campaign.log result.sha256 verified.json order-comparison.json previous-build.sha256 \
    gpu{6,7}-{before,after}.json
REMOTE
evidence=$(dirname "$1")
original="$evidence/w3-c32-prefill-only-20261123.sDaHWRWD"
harness="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR/harness/ci"
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I/src/ci"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 \
    "CUDA_VISIBLE_DEVICES= PYTHONPATH='$1:$harness:$stock' '$runner' -B - '$original' '$1'" \
    < "$here/shape-evidence.py" > "$2/batch-shapes.json"
