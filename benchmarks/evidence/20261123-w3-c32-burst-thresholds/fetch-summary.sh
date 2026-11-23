#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-burst-gpu-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact C32 burst evidence and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
build_input=8298b116a420802a5bbeba1e0e1558400925dd42b3552fc150a8a1d5c7d53660
python3.12 -B - "$here" <<'PY' |
from pathlib import Path
import sys

here = Path(sys.argv[1])
modules = {
    "comparison_base": "20261121-w3-coder-rag-cache/verify.py",
    "verify_probe": "20261121-w3-prefix-cache-min-prompt/verify_probe.py",
    "warm_reference": "20261121-w3-coder-short-warmup/verify_warm.py",
    "private_reference": "20261121-w3-private-startup-warmup/verify.py",
    "replay_verifier": "20261121-w3-c32-arrival-replay/verify.py",
    "observation_reference": "20261122-w3-c32-admission-observations/verify.py",
    "budget_reference": "20261122-w3-c32-steady-budgets/verify.py",
    "http_reference": "20261122-w3-c32-http-workers/verify.py",
}
print("import sys, types")
for name, relative in modules.items():
    path = here.parent / relative
    source = path.read_text()
    if name == "private_reference":
        for anchor, changed in (
            ("stock, parent, build_checks, original, workloads):",
             "stock, parent, build_checks, original, workloads, *, label, budget, decode_cap, prompt_floor, require_stock):"),
            ('    folder = root / f"{case}-gpu{index}"', '    folder = root / label'),
            ('    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]',
             '    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]\n'
             '    argv[argv.index("--prefill-burst-steady-budget") + 1] = str(budget)\n'
             '    argv[argv.index("--prefill-burst-max-decode-rows") + 1] = str(decode_cap)\n'
             '    argv[argv.index("--prefill-burst-min-prompt-tokens") + 1] = str(prompt_floor)'),
            ('    require(exact == REQUEST_COUNTS and', '    require((not require_stock or exact == REQUEST_COUNTS) and'),
        ):
            if source.count(anchor) != 1:
                raise ValueError(f"missing verifier anchor: {anchor}")
            source = source.replace(anchor, changed)
    print(f'module = types.ModuleType({name!r})')
    print("sys.modules[module.__name__] = module")
    print(f'exec(compile({source!r}, {relative!r}, "exec"), module.__dict__)')
print((here / "verify.py").read_text())
PY
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
export PYTHONPATH=\"\$1:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$previous:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$evidence/w5-current-rag-20261121.hvh3VS67\"
umask 077
temp=\$(mktemp -d \"\$1/verified.XXXXXXXX\")
trap '\''rm -f -- \"\$temp/verified.json\" && rmdir -- \"\$temp\"'\'' EXIT
\"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\" >\"\$temp/verified.json\"
tar -cf - result.sha256 runner.sha256 campaign.log -C \"\$temp\" verified.json
' bash '$1' '$dispatch' '$build_input'" | tar -xf - -C "$2"
