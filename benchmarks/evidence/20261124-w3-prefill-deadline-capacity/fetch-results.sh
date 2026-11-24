#!/usr/bin/env bash
set -euo pipefail
output=${1:?pass a new output archive path}
registration=${2:-"$(dirname -- "$0")/registration.json"}
remote_arguments=$(/usr/bin/env python3 -B - "$registration" <<'PY'
import json
from pathlib import Path
import re
import shlex
import sys

row = json.loads(Path(sys.argv[1]).read_bytes())
stage = row["stage"]
if not re.fullmatch(r"/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-[a-z0-9-]+\.[A-Za-z0-9]+", stage):
    raise ValueError("expected a Riley W3 evidence stage")
args = [stage]
for phase in ("build", "campaign"):
    pid, started = row[f"{phase}_pid"], row[f"{phase}_started_at"]
    if type(pid) is not int or pid < 1 or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", started):
        raise ValueError(f"invalid registered {phase} identity")
    args.extend((str(pid), started))
print(shlex.join(args))
PY
)
set -o noclobber
ssh -o BatchMode=yes -o ConnectTimeout=8 aws-a100 "bash -s -- $remote_arguments" >"$output" <<'REMOTE'
set -euo pipefail
cd -- "$1"
test "$(cat build.pid)" = "$2"
test "$(cat build.started)" = "$3"
test "$(cat campaign.pid)" = "$4"
test "$(cat campaign.started)" = "$5"
test "$(cat build.exit)" = 0
test "$(cat campaign.exit)" = 0
sha256sum --strict -c dispatch.sha256 campaign-dispatch.sha256 build-result.sha256 runtime.sha256 result.sha256 >&2
tar -czf - -- \
    build.{pid,started,exit,log,sh} campaign.{pid,started,exit,log,sh} \
    compiler.log cpu-tests.log dependencies.txt toolchain.txt \
    dispatch.sha256 campaign-dispatch.sha256 build-result.sha256 runtime.sha256 result.sha256 \
    prefill-admission-fifo.patch src/crates/riley-scheduler/src/scheduler.rs prepare.py \
    arm.py client.py holder.py replay.py holdout.py schedules.json cpu-placement.json tools/*.py gpu6 gpu7
REMOTE
