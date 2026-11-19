#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
trace="$evidence/w4-layer-trace-r3-20261119.qpoSLydn"
cross="$evidence/w4-hf-bias-partition-20261119.hj2XRENK"
prior="$evidence/w4-hf-mlp-input-20261119.rOd2VDeo"
preceding="$evidence/w5-trim-7sec-retry-20261119.fKlbclsw"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
model="$root/models/Qwen2.5-7B-Instruct"
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
sha256sum -c imports.sha256
test "$(cat "$trace/campaign.exit")" = 0
test "$(cat "$cross/campaign.exit")" = 0
test "$(cat "$prior/campaign.exit")" = 0
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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$stage:$prior:$cross"
unset LD_LIBRARY_PATH
gzip -dc result.json.gz >frozen-result.json
sha256sum -c frozen.sha256
"$python" -B hf_residual_replacement.py "$model" "$trace" "$cross/result.json" frozen-result.json preflight.json --preflight
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
holder_status >idle-before.json
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path

before=json.loads(Path("idle-before.json").read_bytes())
if not (before["physical_index"] == 6 and before["pid"] == 2431228 and before["phase"] == "idle"
        and before["allocation_bytes"] == before["torch_reserved_bytes"] == 20 * 1024**3
        and not before["other_processes"]):
    raise ValueError("GPU 6 is not idle with the expected 20 GiB reservation")
PY
rc=0
printf 'START %s\n' "$(date -u +%FT%TZ)"
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/hf_residual_replacement.py" "$model" "$trace" "$cross/result.json" \
    "$stage/frozen-result.json" "$stage/gpu-result.json" >gpu.log 2>&1 || rc=$?
holder_status >idle-after.json
/usr/bin/python3 -B - <<'PY'
import json
from pathlib import Path

before, after = (json.loads(Path(f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
if not (all(before[key] == after[key] for key in before if key != "updated_ns")
        and not after["other_processes"] and after["phase"] == "idle"):
    raise ValueError("GPU 6 did not restore its original reservation")
PY
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
if (( rc != 0 )); then exit "$rc"; fi
sha256sum preflight.json gpu-result.json gpu.log idle-{before,after}.json >result.sha256
