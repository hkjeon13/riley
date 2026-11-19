#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
trace="$evidence/w4-layer-trace-r3-20261119.qpoSLydn"
previous="$evidence/w4-hf-bias-partition-20261119.hj2XRENK"
preceding="$evidence/w5-trim-sample-timing-20261119.UtiF64Me"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
model="$root/models/Qwen2.5-7B-Instruct"
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
test "$(cat "$previous/campaign.exit")" = 0
test "$(cat "$trace/campaign.exit")" = 0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$stage:$previous:$trace/tools"
unset LD_LIBRARY_PATH
"$python" -B hf_mlp_input.py "$model" "$trace" "$previous/result.json" preflight.json --preflight
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 30
done
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
holder_status >idle-before.json
rc=0
printf 'START %s\n' "$(date -u +%FT%TZ)"
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/hf_mlp_input.py" "$model" "$trace" "$previous/result.json" "$stage/result.json" \
    >gpu.log 2>&1 || rc=$?
holder_status >idle-after.json
"$python" -B - <<'PY'
import json
from pathlib import Path

before, after = (json.loads(Path(f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
keys = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
if not (all(before[key] == after[key] for key in keys) and after["physical_index"] == 6 and after["phase"] == "idle"
        and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3 and not after["other_processes"]):
    raise ValueError("GPU 6 did not restore the same 20 GiB reservation")
PY
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
if (( rc != 0 )); then exit "$rc"; fi
sha256sum preflight.json result.json gpu.log idle-{before,after}.json >result.sha256
