#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
prior=../image-rollback
for ((i=0; i<600; i++)); do
    [[ -f "$prior/released" ]] && break
    sleep 2
done
test -f "$prior/released"
export CODER_GAP_BUDGET=512
bash hold-coder-fleet.sh > controller.log 2>&1 &
holder=$!
cleanup() { touch finished; wait "$holder" || true; }
trap cleanup EXIT
for ((i=0; i<360; i++)); do
    [[ -s gpu6/address && -s gpu7/address ]] && break
    sleep 1
done
test -s gpu6/address && test -s gpu7/address
# Importing run() omits the standalone driver's automatic release between the tests.
taskset -c 62 python3 -B -c 'import asyncio; from pathlib import Path; from coder_mixed_fleet import run; asyncio.run(run(Path.cwd()))' > client.log 2>&1
archive=$(realpath ../../w3-prefill-cohort-20261124.fC0iBjq3)
runner=/mnt/efs/nlp/psyche/riley-gap-20261102/tools/vllm-venv/bin/python
single() {
    local gpu=$1
    mkdir "single$gpu"
    python3 -B protected-plan.py "$archive" "single$gpu/plan.json" "$gpu"
    W3_CLIENT_CPU=$((gpu+38)) PYTHONPATH="$archive:$archive/tools" "$runner" -B "$archive/client.py" "single$gpu/plan.json" "$(cat "gpu$gpu/address")" > "single$gpu/client.log" 2>&1
}
single 6 &
first=$!
single 7
wait "$first"
touch finished
wait "$holder"
trap - EXIT
touch completed
