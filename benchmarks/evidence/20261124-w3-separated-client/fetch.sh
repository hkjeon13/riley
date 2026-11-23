#!/usr/bin/env bash
set -euo pipefail
bundle=${1:?provide the result archive filename}
root=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence
case ${2:-initial} in
    initial)
        stage="$root/w3-separated-client-20261124.QAxR7GdV"
        exit_code=1
        dispatch=7f9e075263be24f99ea1a37174abefba5253361181303bbc7aca4a5e0c74e727
        ;;
    retry|retry-timeout)
        stage="$root/w3-separated-client-retry-20261124.McqPXWqP"
        if [[ $2 == retry ]]; then
            exit_code=0
        else
            exit_code=1
        fi
        dispatch=6752eefee5b323b1728b6c9479325957e7d6df267bf1a7e2e973eeaf7eb74f1d
        ;;
    *)
        printf '%s\n' 'Select initial, retry, or retry-timeout.' >&2
        exit 2
        ;;
esac
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 bash -s -- "$stage" "$exit_code" "$dispatch" >"$bundle" <<'REMOTE'
set -euo pipefail
cd "$1"
test "$(cat campaign.exit)" = "$2"
printf '%s  %s\n' \
    "$3" dispatch.sha256 \
    0546ad7c648fe2574dce01b6a328f01c3f1a354900d721c599cf73da8678c8a1 runtime.sha256 \
    3f7ea800889776e91cc0fe79dd910f0a005ebae371277417457a1e68a3469b66 schedules.json |
    sha256sum --strict -c >&2
sha256sum --strict -c dispatch.sha256 runtime.sha256 result.sha256 >&2
{
    printf '%s\n' campaign.exit campaign.log dispatch.sha256 runtime.sha256 result.sha256
    cut -c 67- dispatch.sha256 runtime.sha256 result.sha256
} | sort -u | tar -czf - --verbatim-files-from -T -
REMOTE
