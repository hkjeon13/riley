#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-prefix-min-gpu-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact prefix-minimum evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
gpu_input=$(shasum -a 256 "$here/gpu-inputs.sha256" | cut -d' ' -f1)
build_input=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
python3 -B - "$here/../20261121-w3-coder-rag-cache/verify.py" "$here/verify_probe.py" <<'PY' |
from pathlib import Path
import sys

base, script = [Path(path).read_text() for path in sys.argv[1:]]
print("import sys, types")
print('module = types.ModuleType("comparison_base")')
print("sys.modules[module.__name__] = module")
print(f'exec(compile({base!r}, "comparison_base.py", "exec"), module.__dict__)')
print(script)
PY
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  gpu-inputs.sha256\\n\" \"\$2\" | sha256sum -c >/dev/null
sha256sum -c gpu-inputs.sha256 result.sha256 binding.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\"\$evidence/w3-prefix-min-build-20261121.vdX2vOQA\"
stock=\"\$evidence/w5-current-package-20261121.f6SEbjvA\"
checker=\"\$evidence/w4-structured-holdout-20261121.w0hBehiZ\"
rag=\"\$evidence/w5-current-rag-20261121.hvh3VS67\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
(
    cd \"\$build\"
    test \"\$(cat build.exit)\" = 0
    printf \"%s  build-inputs.sha256\\n\" \"\$3\" | sha256sum -c >/dev/null
    sha256sum -c build-inputs.sha256 build-result.sha256 >/dev/null
    cd src
    sha256sum -c ../patched-source.sha256 >/dev/null
)
(cd \"\$prior\" && sha256sum -c dispatch.sha256 result.sha256 >/dev/null)
export PYTHONPATH=\"\$1:\$prior:\$checker:\$rag:\$stock/harness/ci:\$stock/src/ci\"
exec \"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$gpu_input' '$build_input'"
