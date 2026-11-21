#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-prefix-package-recovery-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact A100 recovery evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
evidence=$(dirname "$1")
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
variant="$evidence/w5-prefix-package-allocator-20261121.SNyhv9qr"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
python="$(dirname "$evidence")/tools/vllm-venv/bin/python"
(cd "$evidence" && sha256sum --strict -c "$1/candidate-inputs.sha256" >/dev/null)
export PYTHONPATH="$variant:$run/harness/ci:$run/harness/benchmarks/live:$stock/src/ci:$rag:$prior/src/ci"
cmp verified.json <("$python" -B recovery.py verify "$1" "$variant" "$stock/cpu-evidence/report.json")
tar -cf - verified.json result.sha256
REMOTE
