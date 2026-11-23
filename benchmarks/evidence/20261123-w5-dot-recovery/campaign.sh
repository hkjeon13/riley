#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
stock="$evidence/w4-dot-package-20261123.nKyEd9x9"
run="$evidence/w5-prefix-package-gpu-20261121.jBCeUWfR"
variant="$evidence/w5-dot-allocator-20261123.uoCr9cHQ"
prior="$evidence/w5-current-package-20261121.f6SEbjvA"
rag="$evidence/w5-current-rag-20261121.hvh3VS67"
model="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l/model"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$(dirname "$evidence")/tools/vllm-venv/bin/python"
parent="$stock/cpu-evidence/report.json"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum --strict -c dispatch.sha256
(cd "$evidence" && sha256sum --strict -c "$stage/candidate-inputs.sha256")
(cd "$stock" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c result.sha256 >/dev/null)
(cd "$run" && test "$(cat campaign.exit)" = 0 && sha256sum --strict -c runtime-tools.sha256 result.sha256 >/dev/null)
(cd "$variant" && test "$(cat campaign.exit)" = 0 &&
    sha256sum --strict -c dispatch.sha256 context.sha256 result.sha256 >/dev/null)
fixture="$evidence/w5-prefix-package-recovery-20261121.aiBt7Oxk"
cp -- "$fixture/"{recovery,drain_probe}.py .
export PYTHONPATH="$variant:$run/harness/ci:$run/harness/benchmarks/live:$stock/src/ci:$rag:$prior/src/ci"

reservations() {
    local phase=$1
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"idle-$phase.json"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$phase.json"
    "$python" -B - "$phase" <<'PY'
import sys
from pathlib import Path
from current_binding import check_reservations
check_reservations(Path.cwd(), sys.argv[1])
PY
}

"$python" -B recovery.py ports >ports.json
reservations before
for label in allocator stock; do
    rc=0
    printf 'START %s %s\n' "$label" "$(date -u +%FT%TZ)"
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
        "$python" -B recovery.py run "$stage" "$variant" "$parent" "$model" "$label" >"$label.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" >"$label.exit"
    reservations "after-$label"
    printf 'END %s %s exit=%s\n' "$label" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
"$python" -B recovery.py verify "$stage" "$variant" "$parent" >verified.json
sha256sum verified.json ports.json {allocator,stock}{.exit,-route.json} idle-{before,after-allocator,after-stock}.json \
    gpu7-{before,after-allocator,after-stock}.json {allocator,stock}/{report,admission,shutdown,process-start}.json \
    >result.sha256
