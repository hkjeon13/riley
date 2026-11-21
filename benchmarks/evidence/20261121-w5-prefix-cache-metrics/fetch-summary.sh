#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-prefix-metrics-gpu-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact prefix metrics evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
gpu_input=$(shasum -a 256 "$here/gpu-inputs.sha256" | cut -d' ' -f1)
build_input=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  gpu-inputs.sha256\\n\" \"\$2\" | sha256sum -c >/dev/null
sha256sum -c gpu-inputs.sha256 result.sha256 binding.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\"\$evidence/w5-prefix-metrics-build-20261121.5MLLtayu\"
prior=\"\$evidence/w5-prefix-metrics-gpu-20261121.8DIRyrNl\"
stock=\"\$evidence/w5-current-package-20261121.f6SEbjvA\"
checker=\"\$evidence/w4-structured-holdout-20261121.w0hBehiZ\"
rag=\"\$evidence/w5-current-rag-20261121.hvh3VS67\"
(
    cd \"\$build\"
    test \"\$(cat build.exit)\" = 0
    printf \"%s  build-inputs.sha256\\n\" \"\$3\" | sha256sum -c >/dev/null
    sha256sum -c build-inputs.sha256 build-result.sha256 >/dev/null
    cd src
    sha256sum -c ../runtime-source.sha256 >/dev/null
)
(cd \"\$prior\" && sha256sum -c gpu-inputs.sha256 result.sha256 >/dev/null)
export PYTHONPATH=\"\$1:\$checker:\$rag:\$stock/harness/ci:\$stock/src/ci\"
exec \"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$gpu_input' '$build_input'" <"$here/verify.py"
