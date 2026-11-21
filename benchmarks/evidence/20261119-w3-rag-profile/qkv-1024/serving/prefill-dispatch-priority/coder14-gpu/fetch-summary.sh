#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-priority-coder-abba-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact Coder-14B priority evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 driver.sha256 >/dev/null
history=../w3-coder14-qkv-20261120.Mx2OpwKm
phase=../w3-coder-phase-20261121.kxmfecoa/analysis
trace=../w3-coder-wave-trace-20261121.Z0ZQ8NvS/analysis
(cd \"\$history\" && sha256sum -c dispatch.sha256 result.sha256 >/dev/null)
(cd \"\$phase\" && sha256sum -c analysis.sha256 >/dev/null)
(cd \"\$trace\" && sha256sum -c analysis.sha256 >/dev/null)
export PYTHONPATH=\"\$phase:\$trace:\$history\"
exec /usr/bin/python3 -B - \"\$1\" \"\$history\"
' bash '$1' '$dispatch'" <"$here/verify_priority_abba.py"
