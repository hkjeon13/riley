#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-arrival-phase-gpu-20261123\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact C32 arrival-phase evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
old="$here/../20261123-w3-c32-short-burst-admission"
(cd "$old" && shasum -a 256 -c verifier-inputs.sha256 >/dev/null)
(cd "$here" && shasum -a 256 -c dispatch.sha256 >/dev/null)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
python3.12 -B "$here/compose-verifier.py" |
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 >/dev/null
diff - variant-results.sha256 < <(
    for variant in original late12; do (cd \"\$1-\$variant\" && sha256sum result.sha256); done
)
for variant in original late12; do
    (cd \"\$1-\$variant\" && sha256sum --strict -c result.sha256 runner.sha256 >/dev/null)
done
evidence=\$(dirname \"\$1\")
stock=\"\$evidence/w5-prefix-metrics-package-20261121.nEFyze5I\"
run=\"\$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR\"
warm=\"\$evidence/w3-coder-startup-warmup-20261121.db8Pexwl\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=\"\$1-original:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$warm:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$evidence/w5-current-rag-20261121.hvh3VS67\"
\"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$dispatch'"
