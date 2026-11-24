#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w5-pattern-anchors-soak-20261124\.[A-Za-z0-9]+$ || ! $2 =~ ^[0-9a-f]{64}$ ]]; then
    printf '%s\n' 'Pass the A100 evidence directory and the dispatch SHA-256.' >&2
    exit 2
fi
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$1" "$2" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = 0
printf '%s  dispatch.sha256\n' "$2" | sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 result.sha256 >&2
test "$(cat recovery/campaign.exit)" = 0
(cd recovery && sha256sum --strict -c result.sha256 >&2)
tar -czf - campaign.sh compare-soaks.py recovery.sh campaign.exit campaign.log campaign.pid campaign.started dispatch.sha256 result.sha256 verified.json \
    allocator/{context.sha256,gpu-checker-input.json,verified-image.json,verified.json} \
    allocator/{cache-off,cache-on,accuracy}.exit allocator/{cache-off,cache-on,accuracy}/{report,api,shutdown}.json \
    allocator/cache-{off,on}/soak.json allocator/{cache-off,cache-on,accuracy}-allocator-{before,after}.json \
    allocator/idle-{before,after-soaks,after}.json allocator/gpu7-{before,after-soaks,after}.json \
    stock/{cache-off,cache-on}.exit stock/{cache-off,cache-on}/{report,api,shutdown,soak}.json \
    stock/idle-{before,between,after-soaks}.json stock/gpu7-{before,between,after-soaks}.json \
    recovery/{campaign.exit,campaign.log,result.sha256,verified.json,ports.json} \
    recovery/{allocator,stock}{.exit,-route.json} recovery/idle-{before,after-allocator,after-stock}.json \
    recovery/gpu7-{before,after-allocator,after-stock}.json recovery/{allocator,stock}/{report,admission,shutdown,process-start}.json
REMOTE
