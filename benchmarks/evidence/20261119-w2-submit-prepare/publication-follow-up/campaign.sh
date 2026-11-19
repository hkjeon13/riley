#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
prior="$evidence/w2-ordered-tables-20261119.e89GGsLO"
serving="$evidence/w2-publication-gaps-20261120.aH6wKcia"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$serving/campaign.exit")" = 1
(cd "$prior" && sha256sum -c "$stage/reused.sha256")
printf '%s  %s\n' 592dba98659d43036538251cd53b8f76e0cf6ea292564cf4b3c300f1f6f11753 \
    "$root/models/SmolLM2-1.7B-Instruct/riley-checkpoint.json" | sha256sum -c
mkdir tools bin bin/immediate bin/after
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior/holder.py" "$prior/summarize_sync_gaps.py" .
cp -- server_owner.py tools/
ln -- "$prior/bin/base/riley" bin/immediate/riley
ln -- "$prior/bin/base/riley" bin/after/riley
"$python" -B preflight.py tools/run_profile_retry.py
"$python" -B bind_trace_child.py
sha256sum tools/{run_profile_retry,profile_workload,server_owner}.py >selected-runner.sha256
PYTHONPATH="$stage/tools" "$python" -B - <<'PROBE' >owner-preflight.json
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from server_owner import is_owned, record_child

with TemporaryDirectory() as directory, subprocess.Popen(["/bin/sleep", "10"]) as child:
    try:
        out = Path(directory)
        record_child(out, child)
        current = out / "server-owner.json"
        claim = json.loads(current.read_bytes())
        assert is_owned(current, child.pid, claim["start_ticks"])
        assert not is_owned(current, child.pid + 1, claim["start_ticks"])
        changed = out / "different-start.json"
        changed.write_text(json.dumps({**claim, "start_ticks": claim["start_ticks"] + 1}))
        assert not is_owned(changed, child.pid, claim["start_ticks"])
    finally:
        child.terminate()
print(json.dumps({"known_child": "accepted", "different_pid": "rejected", "different_start": "rejected"}))
PROBE
"$python" -B - "$serving" <<'PYTHON'
from pathlib import Path
import re
import sys
from verify_publication import read, require
folder = Path(sys.argv[1]) / "smol-immediate-first-trace"
failed, _ = read(folder / "receipt.json")
pid = int((folder / "server.pid").read_text())
require(failed["status"] == "failed" and re.search(rf"unexpected process on selected GPU pid={pid}$", failed["error"]),
        "previous failure was not the observed owned-child shutdown race")
PYTHON

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

for mode in immediate-first-trace after-first-trace after-second-trace immediate-second-trace; do
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
"$python" -B verify_publication.py "$stage" "$prior" "$serving" >verified.json
sha256sum verified.json selected-runner.sha256 smol-*/{receipt,plan,server-exit,gpu6-before,gpu6-after,gpu7-before,gpu7-after}.json \
    smol-*/*.sqlite smol-*/sync-gaps.json smol-*/reason-c32.json smol-*/server-owner.json >result.sha256
