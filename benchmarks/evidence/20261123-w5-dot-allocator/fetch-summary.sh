#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-dot-allocator-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact A100 allocator campaign and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" <<'REMOTE' | tar -xf - -C "$2"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 context.sha256 result.sha256 >/dev/null
printf '%s  allocator.py\n%s  verify.py\n' \
    6cbece97b900d4d60c512d87355f01672bd549c455621e75986f7697b52f8369 \
    4e2b6d74bd401b6e2a33cff39c174a1dbc5e1b33fb7b1094fd20f68e60a972cc | sha256sum --strict -c >/dev/null
evidence=$(dirname "$1")
stock="$evidence/w4-dot-package-20261123.nKyEd9x9"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
(cd "$stock"
    printf 'd412c16ac878610b3470352b1e6f97a7da408f68aa2d0e13072e8da134e202dc  result.sha256\n' | sha256sum --strict -c >/dev/null
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"
"$runner" -B allocator.py verify "$1" "$stock/cpu-evidence/report.json" | cmp - verified.json
tar -cf - verified-image.json verified.json gpu-checker-input.json context.sha256 result.sha256 campaign.log campaign.exit
REMOTE
