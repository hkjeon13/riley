#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
trap 'echo "$?" > exit-status' EXIT
base=$(realpath ../long-tile)
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
(cd "$base" && sha256sum -c binary.sha256)
cp "$base/binary.sha256" .
sha256sum *.py *.sh *registration.json > inputs.sha256
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_PREFIX_BLOCKS=0
arm() {
    local gpu=$1 mode=$2 attention=fixed-tile-unsplit-v1
    if [[ $mode == page ]]; then attention=page-v1; fi
    local ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
    FULL_REGISTER_GPU="$gpu" TRIAL_ATTENTION_MODE="$attention" "${ctl[@]}" run -- bash run-profile.sh "gpu$gpu-$mode" "$base/target/release/riley" 0 control > "gpu$gpu-$mode.log" 2>&1 || return 1
    "${ctl[@]}" status > "gpu$gpu-$mode/lease-after.json"
}
round() {
    local first rc=0
    arm 6 "$1" & first=$!
    arm 7 "$2" || rc=1
    wait "$first" || rc=1
    return "$rc"
}
round page tile
round tile page
touch released
