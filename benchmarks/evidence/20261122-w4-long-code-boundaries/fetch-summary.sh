#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-long-code-gpu-20261122\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact long-code boundary evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
build_input=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\"\$(cat build-path.txt)\"
[[ \$build =~ ^\"\$evidence\"/w4-long-code-build-20261122\.[[:alnum:]]+\$ ]]
(cd \"\$evidence\" && sha256sum --strict -c \"\$1/captures.sha256\" \"\$build/archive-inputs.sha256\" >/dev/null)
(cd \"\$build\" && test \"\$(cat build.exit)\" = 0 &&
    printf \"%s  build-inputs.sha256\\n\" \"\$3\" | sha256sum --strict -c >/dev/null &&
    sha256sum --strict -c build-inputs.sha256 build-result.sha256 patched-source.sha256 >/dev/null)
exec \"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$dispatch' '$build_input'" <"$here/verify.py"
