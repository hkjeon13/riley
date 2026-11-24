#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
parent=$(realpath ..)
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
test_gpu() {
    local gpu=$1 prior=$2 order=$3
    local ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    for ((i=0; i<180; i++)); do
        [[ -f "$parent/$prior/released" ]] && break
        sleep 2
    done
    test -f "$parent/$prior/released"
    TRIAL_SEQUENCE_LIMIT=32768 TRIAL_OUTPUT_LIMIT=2048 TRIAL_ATTENTION_MODE=page-v1 TRIAL_PREFIX_BLOCKS=0 \
        EXACT_RECORD_TRIAL="$order" "${ctl[@]}" run -- bash run-profile.sh "gpu$gpu" "$parent/named-prefix32/target/release/riley" 0 control > "gpu$gpu.log" 2>&1
    "${ctl[@]}" status > "gpu$gpu/lease-after.json"
}
test_gpu 6 long-tool-records forward &
first=$!
test_gpu 7 long-tile-holdout reverse &
second=$!
a=0
b=0
wait "$first" || a=$?
wait "$second" || b=$?
test "$a" = 0 && test "$b" = 0 && touch released
