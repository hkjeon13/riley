#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-dot-recovery-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact A100 recovery evidence and a local output directory.' >&2
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
python="$(dirname "$evidence")/tools/vllm-venv/bin/python"
(cd "$evidence" && sha256sum --strict -c "$1/candidate-inputs.sha256" >/dev/null)
printf '%s  recovery.py\n%s  drain_probe.py\n' \
    5e224c0f5f01f63d3923a96c3f438b1293308c464ab0774b2666e94eb3e8876d \
    ce17cff248098f7ced13d3e223b5178bc36c510c2e6aeeea249dae32baaa1781 | sha256sum --strict -c >/dev/null
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH="$variant:$run/harness/ci:$run/harness/benchmarks/live:$stock/src/ci:$rag:$prior/src/ci"
"$python" -B recovery.py verify "$1" "$variant" "$stock/cpu-evidence/report.json" | cmp - verified.json
tar -cf - verified.json result.sha256 campaign.log campaign.exit
REMOTE
