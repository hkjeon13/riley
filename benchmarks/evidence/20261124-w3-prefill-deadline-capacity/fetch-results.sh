#!/usr/bin/env bash
set -euo pipefail
output=${1:?pass a new output archive path}
set -o noclobber
ssh -o BatchMode=yes -o ConnectTimeout=8 aws-a100 'bash -s' >"$output" <<'REMOTE'
set -euo pipefail
cd -- /mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-admission-fifo-20261124.hvRFaE7s
test "$(cat build.pid)" = 1506604
test "$(cat build.started)" = 2026-11-24T07:12:28Z
test "$(cat campaign.pid)" = 1520220
test "$(cat campaign.started)" = 2026-11-24T07:14:45Z
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
