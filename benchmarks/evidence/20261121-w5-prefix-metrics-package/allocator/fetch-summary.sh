#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-prefix-package-allocator-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact A100 allocator evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/allocator-dispatch.sha256" | cut -d' ' -f1)
wrapper=$(shasum -a 256 "$here/wrapper-source.sha256" | cut -d' ' -f1)
stock_report=$(shasum -a 256 "$here/../verified.json" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" "$wrapper" "$stock_report" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat build.exit)" = 0
test "$(cat campaign.exit)" = 0
printf '%s  allocator-dispatch.sha256\n%s  wrapper-source.sha256\n%s  stock-verified.json\n' "$2" "$3" "$4" \
    | sha256sum --strict -c >/dev/null
sha256sum --strict -c allocator-dispatch.sha256 wrapper-source.sha256 build-result.sha256 result.sha256 >/dev/null
evidence=$(dirname "$1")
stock="$evidence/w5-prefix-metrics-package-20261121.nEFyze5I"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
runner="$(dirname "$evidence")/tools/vllm-venv/bin/python"
export PYTHONPATH="$run/harness/ci:$stock/src/ci:$rag:$prior/src/ci"
cmp verified.json <("$runner" -B allocator.py verify "$1" "$stock/cpu-evidence/report.json")
tar -cf - verified-image.json verified.json context.sha256 result.sha256
REMOTE
