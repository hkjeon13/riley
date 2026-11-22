#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-http-workers-gpu-20261122\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact C32 HTTP-worker evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
build_input=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
python3 -B - "$here/../20261121-w3-coder-rag-cache/verify.py" \
    "$here/../20261121-w3-prefix-cache-min-prompt/verify_probe.py" \
    "$here/../20261121-w3-coder-short-warmup/verify_warm.py" \
    "$here/../20261121-w3-private-startup-warmup/verify.py" \
    "$here/../20261121-w3-c32-arrival-replay/verify.py" \
    "$here/../20261122-w3-c32-admission-observations/verify.py" \
    "$here/../20261122-w3-c32-steady-budgets/verify.py" "$here/verify.py" <<'PY' |
from pathlib import Path
import sys

print("import sys, types")
for name, path in zip(("comparison_base", "verify_probe", "warm_reference", "private_reference", "replay_verifier", "observation_reference", "budget_reference"), sys.argv[1:8], strict=True):
    source = Path(path).read_text()
    if name == "private_reference":
        original = "def check_case(root, case, index, owner, stock, parent, build_checks, original, workloads):"
        folder = '    folder = root / f"{case}-gpu{index}"'
        assert source.count(original) == source.count(folder) == 1
        source = source.replace(original, original[:-2] + ", *, label=None, budget=256, require_stock=True):")
        source = source.replace(folder, '    folder = root / (label or f"{case}-gpu{index}")')
        args = '    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]'
        stock_match = "    require(exact == REQUEST_COUNTS and"
        assert source.count(args) == source.count(stock_match) == 1
        source = source.replace(args, args + '\n    argv[argv.index("--prefill-burst-steady-budget") + 1] = str(budget)')
        source = source.replace(stock_match, "    require((not require_stock or exact == REQUEST_COUNTS) and")
    print(f'module = types.ModuleType({name!r})')
    print("sys.modules[module.__name__] = module")
    print(f'exec(compile({source!r}, {Path(path).name!r}, "exec"), module.__dict__)')
print(Path(sys.argv[8]).read_text())
PY
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 runner.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\"\$(cat build-path.txt)\"
[[ \$build =~ ^\"\$evidence\"/w3-c32-http-workers-build-20261122\.[[:alnum:]]+\$ ]]
stock=\"\$evidence/w5-prefix-metrics-package-20261121.nEFyze5I\"
run=\"\$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR\"
previous=\"\$evidence/w3-coder-startup-warmup-20261121.db8Pexwl\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
rag=\"\$evidence/w5-current-rag-20261121.hvh3VS67\"
(cd \"\$evidence\" && sha256sum --strict -c \"\$1/runtime-inputs.sha256\" \"\$1/fixed-inputs.sha256\" >/dev/null)
(cd \"\$build\" && test \"\$(cat build.exit)\" = 0 &&
    printf \"%s  build-inputs.sha256\\n\" \"\$3\" | sha256sum --strict -c >/dev/null &&
    sha256sum --strict -c build-inputs.sha256 build-result.sha256 >/dev/null &&
    cd .. && sha256sum --strict -c \"\$build/external-inputs.sha256\" >/dev/null &&
    cd \"\$build/src\" && sha256sum --strict -c ../patched-source.sha256 >/dev/null)
export PYTHONPATH=\"\$1:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$previous:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$rag\"
exec \"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\"
' bash '$1' '$dispatch' '$build_input'"
