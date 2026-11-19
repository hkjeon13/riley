#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
candidate="$evidence/w5-chat-sse-20261119.T7Z99RAG"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-dist-d3d8fe6e5.FE5PU6Y4"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
checkpoint=655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f
gpu=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
helpers="$evidence/w5-framing-accuracy-20261119.jv9EuHuM"
printf '%s  %s\n' 0fe7b279320c5b585868c0664f78c071940ac7d9115976c788fbf084fa228ffb \
    "$helpers/dispatch.sha256" | sha256sum -c -
(cd "$helpers" && sha256sum -c dispatch.sha256)
export PYTHONPATH="$helpers"
printf 'WAIT %s %s\n' "$candidate" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$candidate/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$candidate/campaign.pid")" | grep -Fq -- "$candidate/campaign.sh"; then
        test -s "$candidate/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
test "$(cat "$candidate/campaign.exit")" = 0
(cd "$candidate" && sha256sum -c result.sha256)
"$python" -B verify_chat_accuracy.py "$candidate" >decision.json
if ! "$python" -B -c 'import json; raise SystemExit(not json.load(open("decision.json"))["run_accuracy"])'; then
    printf 'SKIP_STRICT_SOAK %s\n' "$(date -u +%FT%TZ)"
    sha256sum decision.json >result.sha256
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
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index "$1" status
}
holder_status 6 >idle-before.json
holder_status 7 >idle-before-7.json
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
    "$python" -B "$stage/harness/ci/check_distribution_gpu.py" "$candidate/cpu-evidence/report.json" "$gpu" \
    "$base/model" "$checkpoint" "$stage/gpu-evidence" --tools \
    --accuracy-baseline "$legacy" --accuracy-holdout "$workload" "$golden" >gpu-check.log 2>&1 || rc=$?
holder_status 6 >idle-after.json
holder_status 7 >idle-after-7.json
printf 'END accuracy %s exit=%s\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B verify_chat_accuracy.py "$candidate" --accuracy "$stage" >verified-accuracy.json
sha256sum decision.json idle-{before,after}{,-7}.json verified-accuracy.json >result.sha256
