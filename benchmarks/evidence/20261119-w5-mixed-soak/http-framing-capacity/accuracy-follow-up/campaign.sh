#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
framing="$evidence/w5-http-framing-20261119.IyUaTevw"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
printf 'WAIT %s %s\n' "$framing" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$framing/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$framing/campaign.pid")" | grep -Fq -- "$framing/campaign.sh"; then
        test -s "$framing/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
test "$(cat "$framing/campaign.exit")" = 0
(cd "$framing" && sha256sum -c result.sha256)
"$python" -B accuracy_gate.py "$framing" >decision.json
if ! "$python" -B -c 'import json; raise SystemExit(not json.load(open("decision.json"))["run_accuracy"])'; then
    printf 'SKIP_STRICT_RSS %s\n' "$(date -u +%FT%TZ)"
    exit 0
fi
test "$(cat "$prior/campaign.exit")" = 0
mkdir harness
cp -r -- "$prior/harness/"{ci,benchmarks} harness/
sha256sum -c harness.sha256
baselines="$base/src/benchmarks/evidence"
legacy="$baselines/20261119-w4-long/riley/capture.json.gz"
workload="$baselines/20261119-w4-holdout/holdout.json.gz"
golden="$baselines/20261119-w4-holdout/riley/capture.json.gz"
printf '%s  %s\n' \
    b32bd641dab4e9516a8d3b45de2d77fbde91ad64acad893153f4b6c2090b72a6 "$legacy" \
    220f13a7306518f996480c608733623475471ff3721ea3aeb895a2d5f91abd11 "$workload" \
    12d508ed51f03509432e6302b8d10855f56125f4f6b7f31a559cff03eb10b611 "$golden" | sha256sum -c
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
holder_status >idle-before.json
"$python" -B - <<'PY'
from pathlib import Path
import json
state = json.loads(Path("idle-before.json").read_bytes())
if (state["physical_index"] != 6 or state["pid"] != 2431228 or state["phase"] != "idle" or state["other_processes"]
        or state["allocation_bytes"] != state["torch_reserved_bytes"] or state["allocation_bytes"] != 20 * 1024**3):
    raise ValueError("the original A100 6 reservation is unavailable")
PY
printf 'START accuracy %s\n' "$(date -u +%FT%TZ)"
rc=0
PYTHONPATH="$base/src/ci" /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$framing/cpu-evidence/report.json" "$gpu" \
    "$base/model" "$checkpoint" "$stage/gpu-evidence" --tools \
    --accuracy-baseline "$legacy" --accuracy-holdout "$workload" "$golden" >gpu-check.log 2>&1 || rc=$?
holder_status >idle-after.json
printf 'END accuracy %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B accuracy_gate.py "$framing" --accuracy "$stage" >verified-accuracy.json
sha256sum decision.json idle-{before,after}.json verified-accuracy.json >result.sha256
