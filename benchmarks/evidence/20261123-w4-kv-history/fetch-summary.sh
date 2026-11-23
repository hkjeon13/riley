#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 || ! $1 =~ ^/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w4-kv-history-gpu-20261123\.[A-Za-z0-9]+$ || ! -d $2 ]]; then
    printf '%s\n' 'Pass the exact GPU evidence path and a local output directory.' >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "$0")" && pwd -P)
dispatch=$(shasum -a 256 "$here/dispatch.sha256" | cut -d' ' -f1)
inputs=$(shasum -a 256 "$here/build-inputs.sha256" | cut -d' ' -f1)
capture=$(shasum -a 256 "$here/capture-dispatch.sha256" | cut -d' ' -f1)
checker=$(shasum -a 256 "$here/verification-inputs.sha256" | cut -d' ' -f1)
ssh -o BatchMode=yes -o ConnectTimeout=10 aws-a100 "bash -s -- '$1' '$dispatch' '$inputs' '$capture' '$checker'" <<'REMOTE' | tar -xf - -C "$2"
set -euo pipefail
cd "$1"
printf '%s  dispatch.sha256\n' "$4" | sha256sum --strict -c >/dev/null
(cd verification
    printf '%s  dispatch.sha256\n%s  verification-inputs.sha256\n' "$2" "$5" | sha256sum --strict -c >/dev/null
    sha256sum --strict -c dispatch.sha256 verification-inputs.sha256 >/dev/null)
sha256sum --strict -c dispatch.sha256 verification/result.sha256 >/dev/null
evidence=$(dirname "$1")
build=$(cat build-path.txt)
[[ $build =~ ^"$evidence"/w4-kv-history-build-20261123\.[[:alnum:]]+$ ]]
(cd "$build"
    printf '%s  build-inputs.sha256\n' "$3" | sha256sum --strict -c >/dev/null
    test "$(cat build.exit)" = 0
    sha256sum --strict -c build-inputs.sha256 build-result.sha256 patched-source.sha256 >/dev/null)
prior="$evidence/w4-long-code-gpu-20261122.QAbtnxKV"
prior_build=$(cat "$prior/build-path.txt")
export PYTHONPATH="$1:$prior:$prior_build:$prior_build/tools"
"$(dirname "$evidence")/tools/vllm-venv/bin/python" -B verification/trial.py verify "$1" "$build" | cmp - verification/verified.json
tar -cf - campaign.exit checkpoint-verify.log -C "$1/verification" verified.json result.sha256 -C "$build" build-result.sha256 patched-source.sha256 toolchain.txt
REMOTE
