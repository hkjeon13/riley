#!/usr/bin/env bash
set -euo pipefail
kit=$(cd -- "$(dirname -- "$0")" && pwd -P)
prior=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-wide-fixed-tile-20261124.l71cBGh1
expected=$(shasum -a 256 "$kit/campaign-dispatch.sha256" | cut -c1-64)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$prior" "$expected" <<'REMOTE' | tee "$kit/retry-registration.log"
set -euo pipefail
umask 077
cd -- "$1"
previous=$(pwd -P)
test "$(cat campaign.pid)" = 911285
test "$(cat campaign.started)" = 2026-11-24T05:27:56Z
test "$(cat campaign.exit)" = 1
test "$(cat campaign-wait.exit)" = 1
test "$(cat build.exit)" = 0
test "$(cat build-wait.exit)" = 0
grep -Fq '0 passed; 0 failed; 1 ignored' gpu6/flashinfer-attention-test.log
printf '%s  %s\n' \
    7dc18c00d882f65f0905995c7b636b482098277d928cee34aff7508679e6dd20 dispatch.sha256 \
    2b6018822a235ec6412e8afff5ac8f4cbfdd9495a5156cfe0c225f7f74e42c87 campaign-dispatch.sha256 |
    sha256sum --strict -c
sha256sum --strict -c dispatch.sha256 campaign-dispatch.sha256 build-result.sha256 \
    build-wait-script.sha256 campaign-wait-script.sha256
stage=$(mktemp -d "$(dirname "$previous")/w3-wide-fixed-tile-20261124.retryXXXXXXXX")
cp -p --reflink=auto -- dispatch.sha256 before.sha256 after.sha256 build.sh wide-fixed-tile.patch \
    build-result.sha256 riley-diagnostic flashinfer-attention-test dependencies.txt gpu-test-dependencies.txt \
    toolchain.txt build-wait.{pid,started,exit} build.{exit,log} compiler.log gpu-test-build.log \
    build-wait-script.sha256 wait-for-soak.sh campaign-wait-script.sha256 wait-for-build.sh \
    campaign.sh prepare.py "$stage/"
cd -- "$stage"
python3 -B - <<'PY'
from pathlib import Path
p=Path("campaign.sh")
old='"$stage/flashinfer-attention-test" --test-threads=1'
source=p.read_text()
assert source.count(old)==1
p.write_text(source.replace(old, '"$stage/flashinfer-attention-test" --ignored --test-threads=1'))
PY
printf '%s  campaign-dispatch.sha256\n' "$2" | (
     sha256sum campaign.sh prepare.py >campaign-dispatch.sha256
     sha256sum --strict -c
)
sha256sum --strict -c campaign-dispatch.sha256 build-result.sha256
date -u +%FT%TZ >campaign.started
nohup bash wait-for-build.sh </dev/null >campaign.log 2>&1 &
pid=$!
printf '%s\n' "$pid" >campaign.pid
printf 'STAGE %s\nCAMPAIGN_DISPATCH %s\nCAMPAIGN_PID %s %s\n' "$stage" "$2" "$pid" "$(cat campaign.started)"
ps -o pid,stat,args -p "$pid"
REMOTE
