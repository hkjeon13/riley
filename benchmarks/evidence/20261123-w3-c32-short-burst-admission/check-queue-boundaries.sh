#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-c32-short-burst-admission-gpu-20261123\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact engine admission timing evidence path.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
(cd "$here" && shasum -a 256 -c verifier-inputs.sha256 >/dev/null)
checksum=$(shasum -a 256 "$here/result.sha256" | cut -d' ' -f1)
{
    python3.12 -B "$here/compose-verifier.py" |
        python3.12 -B -c 'import sys; source = sys.stdin.read(); print(source[:source.rindex("\nif __name__ == \"__main__\":\n")])'
    cat -- "$here/queue-boundaries.py"
    cat <<'PY'

print(json.dumps(queue_boundary_report(Path(sys.argv[1]), sys.argv[2]), indent=2, allow_nan=False))
PY
} |
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -c 'set -euo pipefail
cd \"\$1\"
test \"\$(cat campaign.exit)\" = 0
printf \"%s  result.sha256\\n\" \"\$2\" | sha256sum --strict -c >/dev/null
sha256sum --strict -c result.sha256 >/dev/null
evidence=\$(dirname \"\$1\")
stock=\"\$evidence/w5-prefix-metrics-package-20261121.nEFyze5I\"
run=\"\$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR\"
warm=\"\$evidence/w3-coder-startup-warmup-20261121.db8Pexwl\"
prior=\"\$evidence/w3-coder-rag-cache-20261121.Ngz53FnV\"
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=\"\$1:\$evidence/w3-private-warm-gpu-20261121.Zdv0ggSI:\$prior:\$warm:\$evidence/w3-prefix-min-gpu-20261121.8gBDZBBM:\$evidence/w3-coder-short-warmup-20261121.oQQFSASh:\$run/harness/ci:\$stock/src/ci:\$evidence/w5-current-rag-20261121.hvh3VS67\"
\"\$(dirname \"\$evidence\")/tools/vllm-venv/bin/python\" -B - \"\$1\" \"\$2\"
' bash '$1' '$checksum'"
