#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
package="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
qkv="$evidence/w3-qkv-serving-20261119.MhydzNsX"
trace="$evidence/w3-request-stages-20261119.DWykKwJR"
prior="$evidence/w3-qkv-tracing-20261119.s9zZEnqW"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for input in "$package" "$qkv" "$trace" "$prior"; do test "$(cat "$input/campaign.exit")" = 0; done
(cd "$prior" && sha256sum -c "$stage/reused.sha256") >prior-tools-check.log
failed="$evidence/w3-kv-pressure-20261120.3kvYNUwX"
test "$(cat "$failed/campaign.exit")" = 1
test ! -e "$failed/binary.sha256"
(cd "$failed" && sha256sum -c dispatch.sha256 source-base.sha256) >prior-attempt-check.log
cp -a -- "$failed/src" src
mkdir tools
cp -- "$prior/tools/"{run_profile_retry,profile_workload,load,profile_summary,profile_process,compare_runner}.py tools/
cp -- "$prior/holder.py" .
sha256sum -c source-base.sha256
"$python" -B instrument.py "$stage"
"$python" -B fix_terminal_observation.py tools/profile_process.py
sha256sum src/crates/riley-{scheduler/src/scheduler,server/src/engine}.rs   tools/{run_profile_retry,profile_workload,profile_process}.py >patched.sha256

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe) >build.log 2>&1
sha256sum target/release/riley >binary.sha256
reservations() {
    local mode=$1 point=$2 name=$3
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point-$mode.json"
    done
    "$python" -B - "$mode" "$point" <<'PY'
import json
from pathlib import Path
import sys
mode,point=sys.argv[1:]
for index,pid,ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
    state=json.loads(Path(f"gpu{index}-{point}-{mode}.json").read_bytes())
    if (state["physical_index"],state["pid"],state["start_ticks"],state["phase"]) != (index,pid,ticks,"idle") or \
            state["allocation_bytes"] != 20*1024**3 or state["torch_reserved_bytes"] != 20*1024**3 or state["other_processes"]:
        raise ValueError(f"GPU {index} lacks the original exclusive idle reservation")
PY
    if test "$point" = after && test -d "$name"; then
        cp "gpu7-after-$mode.json" "$name/idle-after.json"
    fi
}
export PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
for mode in riley-first riley-second; do
    name="qwen14-$mode"
    reservations "$mode" before "$name"
    rc=0
    printf 'START %s %s\n' "$mode" "$(date -u +%FT%TZ)"
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
        "$python" -B "$stage/tools/run_profile_retry.py" qwen14 "$mode" >"$mode.controller.log" 2>&1 || rc=$?
    reservations "$mode" after "$name"
    printf 'END %s %s exit=%d\n' "$mode" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
done
sha256sum patched.sha256 binary.sha256 gpu{6,7}-{before,after}-*.json \
  qwen14-*/{plan,receipt,server-exit,idle-after,idle-before,drain,wave-50ms,rag-c8-s2}.json \
  qwen14-*/{server.log,server.pid} >result.sha256
