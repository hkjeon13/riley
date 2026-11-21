#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w5-current-allocator-20261121.ZVgY3v8o"
stock="$evidence/w5-current-package-20261121.f6SEbjvA"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w5-rag-confirmation-20261120.7zBzLHNW"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$package/campaign.exit")" = 0
test "$(cat "$stock/campaign.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
(cd "$package" && sha256sum -c result.sha256 accuracy-result.sha256 >/dev/null)
(cd "$stock" && sha256sum -c runtime-tools.sha256 >/dev/null)
(cd "$base" && sha256sum -c "$stage/scenario-inputs.sha256")
(cd "$prior" && sha256sum -c result.sha256 >/dev/null)
printf '%s  %s\n' 9aba66838fb889e749e9a9235e0d633408d79917bc71a5924abb4cdc5396ddf3 "$stock/cpu-evidence/report.json" | sha256sum -c
printf '%s  %s\n' 655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f "$base/model/riley-checkpoint.json" | sha256sum -c
cp -- "$package/verify_allocator_image.py" .
printf '%s  verify_allocator_image.py\n' 5e506712f4f1be0b955874179a40f4af490b07847630fd756472ad7e29376a33 | sha256sum -c

export PYTHONPATH="$stage:$stock/harness/ci:$stock/src/ci"
export RILEY_W5_PERF_SOURCE="$base/src/benchmarks/live"
export RILEY_W5_PERF_HARNESS="$stock/harness/benchmarks/live"
unset RILEY_RAG_PAIR RILEY_W5_REPLAY_ARM
"$python" -B - "$package" "$evidence" <<'PY' >gpu-checker-input.json
from pathlib import Path
import json
import sys
from current_binding import binding
print(json.dumps(binding(*map(Path, sys.argv[1:])), indent=2))
PY

reservations() {
    local directory=$1 phase=$2
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"$directory/idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"$directory/gpu7-$phase.json"
    "$python" -B - "$directory" "$phase" <<'PY'
from current_binding import check_reservations
from pathlib import Path
import sys
check_reservations(Path(sys.argv[1]), sys.argv[2])
PY
}

for name in parent-first packaged-first packaged-second parent-second; do
    mkdir "$name"
    reservations "$name" before
    printf 'START %s %s\n' "$name" "$(date -u +%FT%TZ)"
    rc=0
    RILEY_RAG_PAIR="${name##*-}" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B "$stage/run_allocator_performance.py" "$package" "$evidence" "$stage" "$name" \
        >"$name/gpu-check.log" 2>&1 || rc=$?
    reservations "$name" after
    printf '%d\n' "$rc" >"$name/exit"
    printf 'END %s %s exit=%d\n' "$name" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B verify_confirmation.py "$package" "$evidence" "$stage" "$prior" >verified.json
sha256sum gpu-checker-input.json verified.json >result.sha256
