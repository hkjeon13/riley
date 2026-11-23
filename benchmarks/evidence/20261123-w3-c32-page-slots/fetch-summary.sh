#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-(page-slots|bounded-page|row-bounded-page|row-page)-gpu-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact C32 page-slots evidence and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
bounded_dispatch=$(shasum -a 256 "$here/bounded-dispatch.sha256" | cut -d' ' -f1)
workspace_dispatch=$(shasum -a 256 "$here/workspace-dispatch.sha256" | cut -d' ' -f1)
live_dispatch=$(shasum -a 256 "$here/live-dispatch.sha256" | cut -d' ' -f1)
build_input=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
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
             "stock, parent, build_checks, original, workloads, *, label, budget, attention_mode, active_slots, context, max_batch_rows, plateau_override, require_stock):"),
            ('    folder = root / f"{case}-gpu{index}"', '    folder = root / label'),
            ('    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]',
             '    argv = [*SERVE, "--startup-warmup", "step-shapes-v1" if warm else "disabled"]\n'
             '    argv[argv.index("--prefill-burst-steady-budget") + 1] = str(budget)\n'
             '    argv[argv.index("--max-active-sequences") + 1] = str(active_slots)\n'
             '    argv[argv.index("--max-sequence-tokens") + 1] = str(context)\n'
             '    for option in ("--batch-token-budget", "--prefill-chunk-tokens"):\n'
             '        argv[argv.index(option) + 1] = str(max_batch_rows)\n'
             '    bucket_arg = argv.index("--batch-shape-buckets") + 1\n'
             '    argv[bucket_arg] = ",".join(row for row in argv[bucket_arg].split(",") if int(row) <= max_batch_rows)\n'
             '    if plateau_override:\n'
             '        argv += ["--prefill-plateau-rows", "128"]\n'
             '    argv += ["--native-attention-mode", attention_mode]'),
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
[[ \$(cat campaign.exit) =~ ^[01]\$ ]]
validate_campaign() (
    cd \"\$1\"
    printf \"%s  dispatch.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
    sha256sum --strict -c dispatch.sha256 result.sha256 runner.sha256 >/dev/null
    if test -e original-evidence.sha256; then sha256sum --strict -c original-evidence.sha256 >/dev/null; fi
)
evidence=\$(dirname \"\$1\")
initial=\"\$evidence/w3-c32-page-slots-gpu-20261123.wUKuuOzr\"
context_trial=\"\$evidence/w3-c32-bounded-page-gpu-20261123.sYTicdlw\"
row_trial=\"\$evidence/w3-c32-row-bounded-page-gpu-20261123.u1rcSV0z\"
case \"\$1\" in
    \"\$initial\") validate_campaign \"\$1\" \"\$2\" ;;
    \"\$context_trial\")
        validate_campaign \"\$initial\" \"\$2\"
        validate_campaign \"\$1\" \"\$4\"
        ;;
    \"\$row_trial\")
        validate_campaign \"\$initial\" \"\$2\"
        validate_campaign \"\$context_trial\" \"\$4\"
        validate_campaign \"\$1\" \"\$5\"
        ;;
    *)
        validate_campaign \"\$initial\" \"\$2\"
        validate_campaign \"\$context_trial\" \"\$4\"
        validate_campaign \"\$row_trial\" \"\$5\"
        validate_campaign \"\$1\" \"\$6\"
        ;;
esac
sha256sum --strict -c dispatch.sha256 result.sha256 runner.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
build=\$(cat build-path.txt)
[[ \$build =~ ^\"\$evidence\"/w3-c32-page-slots-build-20261123\\.[[:alnum:]]+\$ ]]
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
artifacts=(result.sha256 runner.sha256 campaign.log)
if test -e original-evidence.sha256; then artifacts+=(original-evidence.sha256); fi
tar -cf - \"\${artifacts[@]}\" -C \"\$build\" build-result.sha256 patched-source.sha256 toolchain.txt -C \"\$temp\" verified.json
' bash '$1' '$dispatch' '$build_input' '$bounded_dispatch' '$workspace_dispatch' '$live_dispatch'" | tar -xf - -C "$2"
