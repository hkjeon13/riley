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
build="$evidence/w5-prefix-metrics-build-20261121.5MLLtayu"
prior="$evidence/w5-prefix-metrics-gpu-20261121.8DIRyrNl"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c gpu-inputs.sha256
for source in "$stock" "$checker" "$prior"; do test "$(cat "$source/campaign.exit")" = 0; done
test "$(cat "$build/build.exit")" = 0
(cd "$stock" && sha256sum -c runtime-tools.sha256 >/dev/null)
(cd "$checker" && sha256sum -c dispatch.sha256 >/dev/null)
(cd "$prior" && sha256sum -c gpu-inputs.sha256 result.sha256 >/dev/null)
(cd "$build" && sha256sum -c build-inputs.sha256 build-result.sha256 >/dev/null)
printf '%s  %s\n' "$(sha256sum "$build/build-result.sha256" | cut -d' ' -f1)" "$build/build-result.sha256" >binding.sha256
export PYTHONPATH="$stage:$checker:$rag:$stock/harness/ci:$stock/src/ci"

reservations() {
    local mode=$1 phase=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$mode/idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$mode/gpu7-$phase.json"
    "$python" -B - "$mode" "$phase" <<'PY'
from current_binding import check_reservations
from pathlib import Path
import sys
check_reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

for mode in cache-off cache-on; do
    mkdir "$mode"
    reservations "$mode" before
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_probe.py" "$stage" "$build" "$mode" >"$mode/driver.log" 2>&1 || rc=$?
    reservations "$mode" after
    printf '%s\n' "$rc" >"$mode/exit"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum binding.sha256 {cache-off,cache-on}/{exit,idle-before.json,idle-after.json,gpu7-before.json,gpu7-after.json,receipt.json} >result.sha256
