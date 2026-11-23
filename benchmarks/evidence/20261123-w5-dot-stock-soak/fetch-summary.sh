#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-dot-stock-soak-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact A100 stock-image campaign and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" <<'REMOTE' | tar -xf - -C "$2"
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
evidence=$(dirname "$1")
stock="$evidence/w4-dot-package-20261123.nKyEd9x9"
variant="$evidence/w5-dot-allocator-20261123.uoCr9cHQ"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
(cd "$stock"
    printf 'd412c16ac878610b3470352b1e6f97a7da408f68aa2d0e13072e8da134e202dc  result.sha256\n' | sha256sum --strict -c >/dev/null
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$variant"
    printf 'c0af16e2c614fffd56401d758411ba64d0c4c724354270066b29b306214d933a  result.sha256\n6cbece97b900d4d60c512d87355f01672bd549c455621e75986f7697b52f8369  allocator.py\n4e2b6d74bd401b6e2a33cff39c174a1dbc5e1b33fb7b1094fd20f68e60a972cc  verify.py\n' | sha256sum --strict -c >/dev/null
    sha256sum --strict -c result.sha256 >/dev/null)
(cd "$run" && sha256sum --strict -c runtime-tools.sha256 >/dev/null)
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$variant:$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"
"$runner" -B compare-soaks.py "$1" "$variant" "$stock/cpu-evidence/report.json" | cmp - verified.json
tar -cf - verified.json result.sha256 campaign.log campaign.exit
REMOTE
