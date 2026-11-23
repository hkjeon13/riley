#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
stage=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-chat-answers-20261124.PPoslVc8
expected_exit=${2:-0}
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" "$expected_exit" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = "$2"
printf '%s  %s\n' \
    5d5f8db5da3fec9e3bfb06f60fed2d2b40483715b6f8716aaab2104316b53b49 dispatch.sha256 \
    a43ac41cda9bd985f9d5669b650d714233aa663a46b14f7549613457c62625c1 runtime.sha256 \
    e61926f47138ac81af8df165afbc2d374d8ffc63727365d27e7dddbc3cfd13a1 inputs.json.gz |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 launchers.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log campaign.pid campaign.started dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
