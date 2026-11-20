#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
source="$evidence/w4-native-projection-replay-20261120.9XxkzP0C"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
cuda=/usr/local/cuda-12.4
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$source/campaign.exit")" = 0
printf '%s  %s\n' bec6273c272a0af78e9b8e04cbe53b54b6d945c0abb38085ef035bf5f3599457 \
    "$source/result.sha256" | sha256sum -c
(cd "$source" && sha256sum -c dispatch.sha256 result.sha256) >input-check.log
"$python" -B - "$source/replay_projection_plans.cu" <<'PYBUILD'
from pathlib import Path
import sys
old = Path(sys.argv[1]).read_text()
preamble = old[:old.index("int main(")]
Path("bucket_probe.cu").write_text(preamble + Path("bucket-probe-main.cpp").read_text())
PYBUILD
"$cuda/bin/nvcc" -arch=sm_80 -std=c++17 -O2 bucket_probe.cu -lcublasLt -o bucket_probe

reservations() {
    local point=$1
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json" || return 1
    done
    "$python" -B - "$point" <<'PY' || return 1
import json
from pathlib import Path
import sys
for gpu, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    state = json.loads(Path(f"gpu{gpu}-{sys.argv[1]}.json").read_bytes())
    assert (state["physical_index"], state["pid"], state["start_ticks"], state["phase"]) == (gpu, pid, ticks, "idle")
    assert state["allocation_bytes"] == state["torch_reserved_bytes"] == 20 * 1024**3
    assert not state["other_processes"]
PY
}

deadline=$((SECONDS+1800))
until reservations before; do
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'START GPU %s\n' "$(date -u +%FT%TZ)"
rc=0
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    env LD_LIBRARY_PATH="$cuda/lib64" CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout --signal=TERM --kill-after=20s 520s \
    "$python" -B "$stage/run_buckets.py" "$stage" "$source" >gpu.log 2>&1 || rc=$?
reservations after
printf 'END GPU %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
sha256sum bucket_probe.cu bucket_probe gpu.log {attention-31,down-13-unsplit,down-13-split64,fp32-reduction}.log gpu{6,7}-{before,after}.json >result.sha256
