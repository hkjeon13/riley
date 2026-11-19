#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w2-ordered-tables-20261119.e89GGsLO"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
(cd "$prior" && sha256sum -c "$stage/reused.sha256")
printf '%s  %s\n' 592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753 \
    "$root/models/SmolLM2-1.7B-Instruct/riley-checkpoint.json" | sha256sum -c
mkdir tools bin bin/immediate bin/after
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior/holder.py" "$prior/summarize_sync_gaps.py" .
ln -- "$prior/bin/base/riley" bin/immediate/riley
ln -- "$prior/bin/base/riley" bin/after/riley
"$python" -B preflight.py tools/run_profile_retry.py
sha256sum tools/run_profile_retry.py >selected-runner.sha256

reservations() {
    local folder=$1 time=$2
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"$folder/gpu$index-$time.json"
    done
    "$python" -B - "$folder" "$time" <<'PY'
from pathlib import Path
import sys
from verify_publication import status
for index in (6, 7):
    status(Path(sys.argv[1]), index, sys.argv[2])
PY
}

for mode in immediate-first after-first after-second immediate-second \
            immediate-first-trace after-first-trace after-second-trace immediate-second-trace; do
    initial="$stage/before-$mode"
    mkdir "$initial"
    reservations "$initial" before
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_profile_retry.py" smol "$mode" >"$initial/control.log" 2>&1 || rc=$?
    folder="$stage/smol-$mode"
    mkdir -p "$folder"
    reservations "$folder" after
    mv -- "$initial/"gpu{6,7}-before.json "$folder/"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
    if [[ $mode == *-trace ]]; then
        "$python" -B summarize_sync_gaps.py "$folder/trace.sqlite" >"$folder/sync-gaps.json"
    fi
done
"$python" -B verify_publication.py "$stage" "$prior" >verified.json
sha256sum verified.json selected-runner.sha256 smol-*/{receipt,plan,server-exit,gpu6-before,gpu6-after,gpu7-before,gpu7-after}.json \
    smol-*/*.sqlite smol-*/sync-gaps.json smol-*/reason-c32.json >result.sha256
