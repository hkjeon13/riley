#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
base=$(realpath "$1")
trap 'echo "$?" > exit-status' EXIT
until [[ -s build.exit ]]; do sleep 3; done
test "$(cat build.exit)" = 0
(cd "$base" && sha256sum -c binary.sha256)
sha256sum -c binary.sha256
sha256sum *.py *.sh *.patch *.json > start-inputs.sha256
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_PREFIX_BLOCKS=0 TRIAL_ATTENTION_MODE=page-v1
arm() {
    local gpu=$1 label=$2 binary=$3
    local ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    local rc=0
    "${ctl[@]}" run -- bash run-profile.sh "$label" "$binary" 0 control > "$label.log" 2>&1 || rc=$?
    "${ctl[@]}" status > "$label/lease-after.json"
    return "$rc"
}
TRIAL_REGISTER_LAYERS=1 arm 6 trace "$PWD/target/release/riley"
until [[ -s hf.exit ]]; do sleep 3; done
test "$(cat hf.exit)" = 0
export TRIAL_REGISTER_SCORE=1
round() {
    local first rc=0
    arm 6 "gpu6-$1" "$3" & first=$!
    arm 7 "gpu7-$2" "$4" || rc=1
    wait "$first" || rc=1
    return "$rc"
}
round control candidate "$base/target/release/riley" "$PWD/target/release/riley"
round candidate control "$PWD/target/release/riley" "$base/target/release/riley"
touch released
