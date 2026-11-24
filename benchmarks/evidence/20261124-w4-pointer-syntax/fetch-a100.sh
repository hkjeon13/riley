#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")" && pwd -P)
read -r remote dispatch source pid started < <(python3 -B -c 'import json, sys; r = json.load(open(sys.argv[1])); print(r["stage"], r["dispatch_sha256"], r["a100_source_archive_sha256"], r["campaign_pid"], r["campaign_started_at"])' "$root/registration.json")
[[ $remote =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-pointer-syntax-20261124\.[a-zA-Z0-9]+$ ]]
[[ $dispatch =~ ^[a-f0-9]{64}$ && $source =~ ^[a-f0-9]{64}$ ]]
dest=${1:?provide an existing destination directory}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$remote" "$dispatch" "$source" "$pid" "$started" <<'REMOTE' | tar -xf - -C "$dest"
set -euo pipefail
cd "$1"
test "$(cat campaign.pid)" = "$4"
test "$(cat campaign.started)" = "$5"
test "$(cat campaign.exit)" = 0
printf '%s  %s\n' "$2" dispatch.sha256 "$3" source.tar | sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 result.sha256 >&2
tar -cf - campaign.exit campaign.log campaign.pid campaign.started result.sha256 image.id source-revision.txt \
    cpu-evidence/{report.json,runtime.sha256} structured/{pattern-classes,schema-references,pattern-anchors,pointer-syntax,report,shutdown,api}.json \
    smoke/{report,shutdown,api,smoke}.json gpu{6,7}-{before,after,staged-start}.json
REMOTE
