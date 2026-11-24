#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
d=$(realpath ..)
h=$(realpath ../../w2-sm17-20261118.6UMfeHh2)
# When registered, the paired process-replacement drill owns both GPUs first.
if [[ -f $d/operations/operations-registration.json ]]; then
    for ((i=0; i<900; i++)); do
        [[ -f $d/operations/released ]] && break
        sleep 1
    done
    [[ -f $d/operations/released ]] || exit 1
fi
gpu=$1
ctl=(python3 "$h/control.py" --lease-dir "$h/leases" --index "$gpu")
if [[ $gpu == 6 ]]; then
    export FROZEN_INPUT_DIR="$d/reference/riley-a"
    for pair in 1 2 3 4 5; do
        case "$pair" in
            2|3) order=(wide vllm);;
            *) order=(vllm wide);;
        esac
        for engine in "${order[@]}"; do
            out="pair-$pair-$engine"
            if [[ $engine == wide ]]; then
                case "$pair" in
                    1|4) preview=off;;
                    2|3) preview=on;;
                    *) preview=;;
                esac
                WORKFLOW_PREFIX_TRIAL="$preview" "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 wide
            else
                "${ctl[@]}" run -- python3 -B run-reference.py "$out" --seed 761
            fi
            "${ctl[@]}" status > "$out/lease-after.json"
        done
    done
elif [[ $gpu == 7 ]]; then
    export PROTECTED_INDEX=7
    for spec in a:control b:wide c:wide d:control; do
        out="protected-${spec%:*}"
        "${ctl[@]}" run -- bash run-profile.sh "$out" stock 761 "${spec#*:}"
        "${ctl[@]}" status > "$out/lease-after.json"
    done
else
    exit 2
fi
