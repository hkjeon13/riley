#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
framing="$evidence/w5-http-framing-20261119.IyUaTevw"
accuracy="$evidence/w5-framing-accuracy-20261119.jv9EuHuM"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
sha256sum -c preflight.sha256
sha256sum -c harness-observer.sha256

printf 'WAIT %s %s\n' "$accuracy" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 7200))
until test -s "$accuracy/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$accuracy/campaign.pid")" | grep -Fq -- "$accuracy/campaign.sh"; then
        test -s "$accuracy/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
test "$(cat "$accuracy/campaign.exit")" = 0
test "$(cat "$framing/campaign.exit")" = 0
(cd "$framing" && sha256sum -c result.sha256)
"$python" -B - "$accuracy/decision.json" "$framing/cache-off/comparison.json" <<'PY'
from pathlib import Path
import json
import sys

decision, pair = (json.loads(Path(path).read_bytes()) for path in sys.argv[1:])
modes = decision["soak_modes"]
checks = [value["failed_checks"] for value in modes.values()]
if (set(modes) != {"cache-on", "cache-off"} or decision["run_accuracy"] is not False or not any(checks)
        or any(set(check) - {"rss_growth_second_half_at_most_zero"} for check in checks)):
    raise ValueError("supplementary analysis requires the completed RSS-only framing failure")
for mode, result in modes.items():
    paired = pair["modes"][mode]
    if result != {"status": paired["candidate_status"], "failed_checks": paired["candidate_failed_checks"],
                  "report_sha256": paired["candidate_report_sha256"]}:
        raise ValueError("the framing accuracy gate differs from the stored pair")
PY

holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
for index in 6 7; do holder_status "$index" >"idle-before-$index.json"; done
"$python" -B - <<'PY'
from pathlib import Path
import json
for index, pid in ((6, 2431228), (7, 2431237)):
    row = json.loads(Path(f"idle-before-{index}.json").read_bytes())
    if (row["physical_index"] != index or row["pid"] != pid or row["phase"] != "idle"
            or row["other_processes"] or row["allocation_bytes"] != 20 * 1024**3
            or row["torch_reserved_bytes"] != row["allocation_bytes"]):
        raise ValueError(f"GPU {index} is not in its original reservation")
PY
printf 'START cache-off precise RSS series %s\n' "$(date -u +%FT%TZ)"
result=0
PYTHONPATH="$base/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$base/cpu-evidence/report.json" \
    "$gpu" "$base/model" "$checkpoint" "$stage/gpu-evidence" --tools --mixed-soak cache-off \
    >gpu-check.log 2>&1 || result=$?
for index in 6 7; do holder_status "$index" >"idle-after-$index.json"; done
printf '%d\n' "$result" >checker.exit
PYTHONPATH="$stage/harness/benchmarks/live" "$python" -B verify_rss_series.py "$stage" >verified.json
sha256sum checker.exit idle-{before,after}-{6,7}.json verified.json >result.sha256
printf 'END cache-off precise RSS series %s strict_exit=%d\n' "$(date -u +%FT%TZ)" "$result"
