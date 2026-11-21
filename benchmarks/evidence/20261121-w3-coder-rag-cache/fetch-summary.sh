#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-rag-cache-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact Coder-14B cache evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
if ! test -d off-first; then
    initial=\"\$evidence/w3-coder-rag-cache-20261121.kXvzFrJJ\"
    (cd \"\$initial\" && sha256sum -c \"\$1/first-pair-dispatch.sha256\" >/dev/null)
fi
stock=\"\$evidence/w5-current-package-20261121.f6SEbjvA\"
checker=\"\$evidence/w4-structured-holdout-20261121.w0hBehiZ\"
rag=\"\$evidence/w5-current-rag-20261121.hvh3VS67\"
export PYTHONPATH=\"\$1:\$checker:\$rag:\$stock/harness/ci:\$stock/src/ci\"
exec \"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$dispatch'" <"$here/verify.py"
