#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
prior=$1
trap 'echo "$?" > exit-status' EXIT
base=$(realpath ../long-tile)
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
(cd "$base" && sha256sum -c binary.sha256)
cp "$base/binary.sha256" .
sha256sum *.py *.sh *registration.json > inputs.sha256
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_PREFIX_BLOCKS=0 TRIAL_ATTENTION_MODE=page-v1
arm() {
    local gpu=$1
    local ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
    if [[ $gpu == 6 ]]; then
        until [[ -s "$prior/page/lease-after.json" || -s "$prior/exit-status" ]]; do sleep 3; done
    else
        until [[ -s "$prior/exit-status" ]]; do sleep 3; done
    fi
    REGISTER_REDUCTION_GPU="$gpu" "${ctl[@]}" run -- bash run-profile.sh "gpu$gpu" "$base/target/release/riley" 0 control > "gpu$gpu.log" 2>&1 || return 1
    "${ctl[@]}" status > "gpu$gpu/lease-after.json"
}
arm 6 & first=$!
rc=0
arm 7 || rc=1
wait "$first" || rc=1
test "$rc" = 0
touch released
