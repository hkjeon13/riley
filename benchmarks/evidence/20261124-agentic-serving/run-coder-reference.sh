#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
prior=../coder-mixed-fleet
for ((i=0; i<900; i++)); do
   [[ -f "$prior/released" ]] && break
   sleep 1
done
test -f "$prior/released"
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
arm() {
    local gpu=$1 cores=$2
    local ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
    "${ctl[@]}" run -- taskset -c "$cores" python3 -B serve_coder_reference.py "$gpu" > "gpu$gpu.log" 2>&1
    "${ctl[@]}" status > "gpu$gpu/lease-after.json"
}
arm 6 24-33,72-81 &
p6=$!
arm 7 34-43,82-91 &
p7=$!
# The engines may compile kernels before exposing the HTTP endpoint.
for ((i=0; i<720; i++)); do
    [[ -f gpu6/address && -f gpu7/address ]] && break
    if ! kill -0 "$p6" || ! kill -0 "$p7"; then break; fi
    sleep 1
done
rc=0
if [[ -f gpu6/address && -f gpu7/address ]]; then
    taskset -c 62 python3 -B coder_mixed_fleet.py > client.log 2>&1 || rc=$?
else
    rc=1
fi
touch finished
wait "$p6" || rc=1
wait "$p7" || rc=1
test "$rc" = 0 && touch released
