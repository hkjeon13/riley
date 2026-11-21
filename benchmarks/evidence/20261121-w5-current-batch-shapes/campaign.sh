#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
package="$evidence/w5-current-allocator-20261121.ZVgY3v8o"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for required in "$stock" "$package" "$rag"; do test "$(cat "$required/campaign.exit")" = 0; done
(cd "$stock" && sha256sum -c runtime-tools.sha256 >/dev/null)
(cd "$package" && sha256sum -c result.sha256 accuracy-result.sha256 >/dev/null)
(cd "$rag" && sha256sum -c dispatch.sha256 result.sha256 >/dev/null)
printf '%s  %s\n' 5e506712f4f1be0b955874179a40f4af490b07847630fd756472ad7e29376a33 "$rag/verify_allocator_image.py" | sha256sum -c
export PYTHONPATH="$stage:$rag:$stock/harness/ci:$stock/src/ci"

reservations() {
    local name=$1 phase=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$name/idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$name/gpu7-$phase.json"
    "$python" -B - "$name" "$phase" <<'PY'
from current_binding import check_reservations
from pathlib import Path
import sys
check_reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

for name in stock jemalloc; do
    mkdir "$name"
    reservations "$name" before
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    rc=0
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_shapes.py" "$stage" "$name" >"$name/driver.log" 2>&1 || rc=$?
    reservations "$name" after
    printf '%s\n' "$rc" >"$name/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test -s "$name/gpu-evidence/report.json"
done
sha256sum gpu-checker-input.json {stock,jemalloc}/{exit,idle-before.json,idle-after.json,gpu7-before.json,gpu7-after.json,process-before.json,process-after.json} \
    {stock,jemalloc}/gpu-evidence/{report,smoke,shutdown,api}.json >result.sha256
