#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-schema-ids-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    exit 2
fi
dir=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$dir/dispatch.sha256" | awk '{print $1}')
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$dispatch" <<'REMOTE' | tar -C "$2" -xf -
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >/dev/null
sha256sum --strict -c dispatch.sha256 result.sha256 >/dev/null
tar -cf - campaign.exit campaign.log result.sha256 image.id cpu-evidence/{report.json,runtime.sha256} \
    structured/{pattern-classes,schema-references,report,shutdown}.json \
    smoke/{report,shutdown,smoke}.json gpu{6,7}-{before,after}.json
REMOTE
cp "$dir/source-revision.txt" "$2/"
python3.12 -B "$dir/verify-a100.py" "$2" >"$2/verified-a100.json"
