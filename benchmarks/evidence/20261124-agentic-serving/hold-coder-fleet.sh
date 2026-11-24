#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
arm() {
    local gpu=$1
    local ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    "${ctl[@]}" run -- bash serve-coder-fleet.sh "$gpu" > "gpu$gpu.log" 2>&1
    "${ctl[@]}" status > "gpu$gpu/lease-after.json"
}
arm 6 &
pid=$!
second=0
arm 7 || second=$?
wait "$pid"
test "$second" = 0 && touch released
