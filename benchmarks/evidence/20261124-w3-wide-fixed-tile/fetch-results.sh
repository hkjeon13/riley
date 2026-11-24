#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "$0")" && pwd -P)
stage=$(cat "$kit/remote-path")
[[ $stage =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-wide-fixed-tile-20261124\.[A-Za-z0-9]+$ ]]
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" <<'REMOTE'
set -euo pipefail
cd -- "$1"
test "$(cat build-wait.pid)" = 899144
test "$(cat build-wait.started)" = 2026-11-24T05:25:44Z
test "$(cat campaign.pid)" = 1259412
test "$(cat campaign.started)" = 2026-11-24T06:28:30Z
for name in build-wait build campaign-wait campaign; do test "$(cat "$name.exit")" = 0; done
printf '%s  %s\n' \
    7dc18c00d882f65f0905995c7b636b482098277d928cee34aff7508679e6dd20 dispatch.sha256 \
    65f604fe4d827d291291225795472eed3b7e245e43767eb39a840fa822b29c6e campaign-dispatch.sha256 |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 build-result.sha256 build-wait-script.sha256 \
    campaign-dispatch.sha256 campaign-wait-script.sha256 runtime.sha256 result.sha256 >&2
manifests=(dispatch.sha256 build-result.sha256 build-wait-script.sha256 campaign-dispatch.sha256 \
    campaign-wait-script.sha256 runtime.sha256 result.sha256)
{
    printf '%s\n' "${manifests[@]}" build-wait.{pid,started,exit} build.{exit,log} \
        campaign-wait.exit campaign.{pid,started,exit,log} compiler.log gpu-test-build.log
    for manifest in "${manifests[@]}"; do cut -c67- "$manifest"; done
} | LC_ALL=C sort -u | awk '$0 != "riley-diagnostic" && $0 != "flashinfer-attention-test"' |
    tar -czf - --verbatim-files-from -T -
REMOTE
