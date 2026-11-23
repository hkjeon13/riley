#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-steady256-burst-gpu-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact C32 steady-256 burst evidence and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
build_input=8298b116a420802a5bbeba1e0e1558400925dd42b3552fc150a8a1d5c7d53660
(cd "$here" && shasum -a 256 -c verifier-inputs.sha256 >/dev/null)
python3.12 -B "$here/compose-verifier.py" |
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 runner.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\$(cat build-path.txt)
[[ \$build =~ ^\"\$evidence\"/w3-c32-prefill-build-20261123\\.[[:alnum:]]+\$ ]]
(cd \"\$build\"
    printf \"%s  build-inputs.sha256\\n\" \"\$3\" | sha256sum --strict -c >/dev/null
    test \"\$(cat build.exit)\" = 0
    sha256sum --strict -c build-inputs.sha256 build-result.sha256 >/dev/null
    cd src && sha256sum --strict -c ../patched-source.sha256 >/dev/null)
stock=\"\$evidence/w5-prefix-metrics-package-20261121.nEFyze5I\"
run=\"\$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR\"
previous=\"\$evidence/w3-coder-startup-warmup-20261121.db8Pexwl\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=\"\$1:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$previous:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$evidence/w5-current-rag-20261121.hvh3VS67\"
umask 077
temp=\$(mktemp -d \"\$1/verified.XXXXXXXX\")
trap '\''rm -f -- \"\$temp/verified.json\" && rmdir -- \"\$temp\"'\'' EXIT
\"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\" >\"\$temp/verified.json\"
tar -cf - campaign.exit result.sha256 runner.sha256 campaign.log -C \"\$temp\" verified.json
' bash '$1' '$dispatch' '$build_input'" | tar -xf - -C "$2"
