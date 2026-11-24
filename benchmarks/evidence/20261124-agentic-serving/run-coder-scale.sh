#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export AGENT_DECODE_MODEL=Qwen2.5-Coder-7B-Instruct CODER_PREFILL_TRACE=1 CODER_SCALE_SERVER=1
holder=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
run() {
    local gpu=$1
    local ctl=(python3 "$holder/control.py" --lease-dir "$holder/leases" --index "$gpu")
    "${ctl[@]}" run -- bash -c 'date +%s%N > "$1.start"; bash run-profile.sh "$1" stock 5091 wide && date +%s%N > "$1.end"' _ "gpu$gpu" > "gpu$gpu.controller.log" 2>&1
    "${ctl[@]}" status > "gpu$gpu/lease-after.json"
}
run 6 & a=$!
run 7 & b=$!
set +e
main() {
    for ((i=0; i<400; i++)); do
        if [[ -s gpu6/address && -s gpu7/address ]]; then
            if [[ -n ${CODER_PLACEMENT:-} ]]; then python3 -B coder_placement_trial.py; else python3 -B coder_scale_trial.py; fi
            return $?
        fi
        kill -0 "$a" && kill -0 "$b" || return 1
        sleep 1
    done
    return 1
}
main
rc=$?
touch finished
wait "$a"; ca=$?
wait "$b"; cb=$?
echo "$ca $cb" > exits.log
echo "$rc" > client.exit
exit $((rc+ca+cb))
