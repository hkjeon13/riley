#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-coder-wave-trace-20261121\.[A-Za-z0-9]+$ ]]; then
    printf '%s\n' 'Pass the exact Coder-14B trace evidence directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch_sha=$(shasum -a 256 "$here/dispatch.sha256" | cut -d ' ' -f 1)
analysis_sha=$(shasum -a 256 "$here/analysis.sha256" | cut -d ' ' -f 1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$dispatch_sha' '$analysis_sha'" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n%s  analysis/analysis.sha256\n' "$2" "$3" | sha256sum -c >/dev/null
sha256sum -c dispatch.sha256 result.sha256 driver.sha256 >/dev/null
(cd analysis && sha256sum -c analysis.sha256 >/dev/null)
base="$(dirname "$1")/w3-coder14-qkv-20261120.Mx2OpwKm"
(cd "$base" && sha256sum -c dispatch.sha256 result.sha256 >/dev/null)
export PYTHONPATH="$base"
exec /usr/bin/python3 -B analysis/verify_trace.py "$1" "$base"
REMOTE
