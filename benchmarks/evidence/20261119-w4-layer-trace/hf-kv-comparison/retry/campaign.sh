#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
reference="$evidence/w4-layer-trace-r3-20261119.qpoSLydn"
native="$evidence/w4-kv-mask-20261119.DEXef1dj"
preceding="$evidence/w5-allocator-trim-20261119.ZEkdRfD9"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >"$stage/campaign.exit.tmp"; mv -- "$stage/campaign.exit.tmp" "$stage/campaign.exit"' EXIT
cd "$stage"
sha256sum -c dispatch.sha256
test "$(cat "$reference/campaign.exit")" = 0
test "$(cat "$native/campaign.exit")" = 0
printf 'WAIT %s %s\n' "$preceding" "$(date -u +%FT%TZ)"
deadline=$((SECONDS + 14400))
until test -s "$preceding/campaign.exit"; do
    if ! ps -o args= -p "$(cat "$preceding/campaign.pid")" | grep -Fq -- "$preceding/campaign.sh"; then
        test -s "$preceding/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 25
done
sha256sum -c replay.sha256
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset LD_LIBRARY_PATH
holder_status() {
    /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status
}
holder_status >idle-before.json
rc=0
printf 'START %s\n' "$(date -u +%FT%TZ)"
/usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/hf_kv_comparison.py" "$root/models/Qwen2.5-7B-Instruct" "$reference" "$native" "$stage" \
    >gpu.log 2>&1 || rc=$?
holder_status >idle-after.json
"$python" -B - <<'PY'
import json
from pathlib import Path

before, after = (json.loads(Path(f"idle-{point}.json").read_bytes()) for point in ("before", "after"))
fields = ("pid", "start_ticks", "gpu", "physical_index", "phase", "allocation_bytes", "torch_reserved_bytes")
if not (all(before[key] == after[key] for key in fields) and after["physical_index"] == 6
        and after["phase"] == "idle" and not after["other_processes"]
        and after["allocation_bytes"] == after["torch_reserved_bytes"] == 20 * 1024**3):
    raise ValueError("same A100 6 holder did not restore its 20 GiB reservation")
PY
printf 'END %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
if (( rc != 0 )); then exit "$rc"; fi
sha256sum result.json hf-case{3,9}/*.bf16 gpu.log idle-{before,after}.json >result.sha256
