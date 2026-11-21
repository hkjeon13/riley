#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
checker="$evidence/w4-structured-holdout-20261121.w0hBehiZ"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for required in "$stock" "$checker"; do test "$(cat "$required/campaign.exit")" = 0; done
(cd "$stock" && sha256sum -c runtime-tools.sha256 >/dev/null)
(cd "$checker" && sha256sum -c dispatch.sha256 >/dev/null)
export PYTHONPATH="$stage:$checker:$rag:$stock/harness/ci:$stock/src/ci"
if [[ ${1-} == reverse ]]; then
    initial="$evidence/w3-coder-rag-cache-20261121.kXvzFrJJ"
    (cd "$initial" && sha256sum -c "$stage/first-pair-dispatch.sha256" >/dev/null)
    test "$(cat "$initial/campaign.exit")" = 1
    test "$(cat "$initial/off-first/exit")" = 0
    test "$(cat "$initial/on-first/exit")" = 0
    grep -qx 'checkpoint verify: verified' "$initial/checkpoint-verify.log"
    cp -- "$initial/checkpoint-verify.log" .
    arms=(on-second off-second)
else
    "$python" -B "$stage/run_arm.py" "$stage" verify
    arms=(off-first on-first on-second off-second)
fi

reservations() {
    local arm=$1 phase=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$arm/idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$arm/gpu7-$phase.json"
    "$python" -B - "$arm" "$phase" <<'PY'
from current_binding import check_reservations
from pathlib import Path
import sys
check_reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

for arm in "${arms[@]}"; do
    mkdir "$arm"
    reservations "$arm" before
    printf 'START %s %s\n' "$arm" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/run_arm.py" "$stage" "$arm" >"$arm/driver.log" 2>&1 || rc=$?
    reservations "$arm" after
    printf '%s\n' "$rc" >"$arm/exit"
    printf 'END %s %s exit=%d\n' "$arm" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum checkpoint-verify.log >result.sha256
for arm in "${arms[@]}"; do
    sha256sum "$arm"/{exit,idle-before.json,idle-after.json,gpu7-before.json,gpu7-after.json,receipt.json} >>result.sha256
done
