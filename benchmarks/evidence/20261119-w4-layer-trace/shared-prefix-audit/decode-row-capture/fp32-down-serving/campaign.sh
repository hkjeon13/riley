#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
native="$evidence/w4-native-projection-plans-20261120.yvjrWvOj"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$native/campaign.exit")" = 0
printf '%s  %s\n' b5de696ebdda0025cdd4d3180baac4f134aa96e884714cc5c31eaea196554212 \
    "$native/result.sha256" | sha256sum -c
(cd "$native" && sha256sum -c result.sha256) >native-check.log

reservations() {
    local point=$1
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 status >"gpu6-$point.json" || return 1
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 status >"gpu7-$point.json" || return 1
    "$python" -B - "$point" <<'PY' || return 1
import json
from pathlib import Path
import sys

for index, pid, ticks in ((6,2431228,"6339570869"),(7,2431237,"6339570875")):
    state=json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (state["physical_index"] != index or state["pid"] != pid or state["start_ticks"] != ticks
            or state["phase"] != "idle" or state["other_processes"]
            or state["allocation_bytes"] != state["torch_reserved_bytes"] or state["allocation_bytes"] != 20*1024**3):
        raise ValueError(f"GPU {index} is missing its original idle reservation")
PY
}

mkdir tools src
cp -- "$native/holder.py" .
cp -- "$native/tools/"{run_accuracy,long_accuracy,compare_runner}.py tools/
cp -- capture_decode_rows.py tools/
ln -s -- "$native/fixtures" fixtures
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$base/source.tar" | sha256sum -c
tar -xf "$base/source.tar" -C src --no-same-owner
sha256sum -c source-inputs.sha256
"$python" -B prepare_candidate.py "$stage"
sha256sum src/kernels/src/gemm.cu src/crates/riley-cuda/src/{gemm,ffi}.rs \
    src/crates/riley-runtime/src/llama/forward.rs tools/run_accuracy.py >candidate-source.sha256
image=sha256:0a907408ec73fc8b9b00f440037305ce1bfb00e05bd8d3ef3096fcee59849ea1
test "$(docker image inspect --format '{{.Id}}' "$image")" = "$image"
docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void --user "$(id -u):$(id -g)" \
    --mount "type=bind,src=$model,dst=/models,readonly" "$image" \
    riley-checkpoint verify --checkpoint /models >checkpoint-verify.log

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
unset RILEY_W4_TRACE_DIR RILEY_W4_LAYER_REFERENCE RILEY_W4_ATTENTION_O31 RILEY_W4_DOWN_FP32
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe) \
    >build.log 2>&1
sha256sum target/release/riley >binary.sha256

run_gpu() {
    local phase=$1 rc=0
    shift
    local deadline=$((SECONDS+1800))
    until reservations "$phase-before"; do
        if (( SECONDS >= deadline )); then return 75; fi
        sleep 20
    done
    printf 'START %s %s\n' "$phase" "$(date -u +%FT%TZ)"
    "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- "$@" || rc=$?
    reservations "$phase-after"
    printf 'END %s %s exit=%d\n' "$phase" "$(date -u +%FT%TZ)" "$rc"
    test "$rc" = 0
}
for arm in baseline candidate; do
    run_gpu "$arm" "$python" -B "$stage/tools/run_accuracy.py" "$arm"
done
sha256sum binary.sha256 candidate-source.sha256 checkpoint-verify.log \
    {baseline,candidate}/{client.log,receipt.json,server.log,rows/{long,holdout}.json} \
    gpu{6,7}-{baseline,candidate}-{before,after}.json >result.sha256
