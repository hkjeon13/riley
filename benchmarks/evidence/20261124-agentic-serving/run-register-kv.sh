#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
parent=$(realpath "$1")
trap 'echo "$?" > exit-status' EXIT
test "$(cat "$parent/exit-status")" = 0
(cd "$parent" && sha256sum -c kv-binary.sha256)
cp "$parent"/{*.py,run-profile.sh,register-layer-registration.json,register-reference-registration.json} .
cp "$parent/kv-binary.sha256" binary.sha256
sha256sum *.py *.sh *registration.json > inputs.sha256
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
export TRIAL_REQUEST_TIMEOUT_MS=900000 TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_PREFIX_BLOCKS=0 TRIAL_REGISTER_LAYERS=1
arm() {
    local gpu=$1 mode=$2 attention=fixed-tile-unsplit-v1
    if [[ $mode == page ]]; then attention=page-v1; fi
    local ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    local rc=0
    TRIAL_ATTENTION_MODE="$attention" "${ctl[@]}" run -- bash run-profile.sh "$mode" "$parent/kv-target/release/riley" 0 control > "$mode.log" 2>&1 || rc=$?
    "${ctl[@]}" status > "$mode/lease-after.json"
    return "$rc"
}
arm 6 page & first=$!
rc=0
arm 7 tile || rc=1
wait "$first" || rc=1
test "$rc" = 0
touch released
