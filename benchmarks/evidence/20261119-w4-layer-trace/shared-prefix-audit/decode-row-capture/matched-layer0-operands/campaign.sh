#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
previous="$evidence/w4-decode-final-retry-20261120.5bTpAh8k"
performance="$evidence/w4-decode-layer-trace-20261120.kl10bRNz"
operator="$evidence/w4-decode-layer0-ops-20261120.hYAOtc0n"
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
python=/usr/bin/python3
venv="$root/tools/vllm-venv/bin/python"
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
test "$(cat "$previous/campaign.exit")" = 0
test "$(cat "$performance/campaign.exit")" = 0
test "$(cat "$operator/campaign.exit")" = 0
printf '%s  %s\n' 2d6add51a82943e12a1d2ecf432b8e209f7547942e2ae7ac12d7bc97523386de \
    "$operator/result.sha256" | sha256sum -c
(cd "$operator" && sha256sum -c dispatch.sha256 result.sha256) >operator-check.log
export RILEY_W4_LAYER_PROGRESSION="$performance"

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
cp -- "$previous"/holder.py .
cp -- "$previous/tools/"{run_accuracy.py,long_accuracy.py,compare_runner.py,capture_reference.py,capture_final_rows.py,replay_final_head.py} tools/
cp -- "$operator/tools/trace_decode_layers.py" tools/
cp -- "$performance/tools/trace_decode_layers.py" tools/progression_trace.py
cp -- "$evidence/w4-decode-residual-boundaries-20261120.iOM87n62/tools/replay_decode_boundaries.py" tools/
cp -- replay_mlp.py tools/
sha256sum -c study-tools.sha256
ln -s -- "$previous/fixtures" fixtures
sha256sum -c previous-tools.sha256
cat >tools/capture_decode_rows.py <<'PY'
import os
import sys

from trace_decode_layers import main

sys.argv.insert(1,"riley")
sys.argv.append(os.environ["RILEY_W4_LAYER_REFERENCE"])
main()
PY

printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5 \
    "$base/source.tar" | sha256sum -c
tar -xf "$base/source.tar" -C src --no-same-owner
(cd src && sha256sum -c "$stage/source-base.sha256" &&
    gzip -dc "$stage/decode-layer-trace.patch.gz" | git apply --check - &&
    gzip -dc "$stage/decode-layer-trace.patch.gz" | git apply - &&
    sha256sum -c "$stage/source-patched.sha256")
"$python" -B enable_mlp_capture.py "$stage"
sha256sum src/crates/riley-runtime/src/llama/executor/dispatch.rs tools/trace_decode_layers.py >capture-source.sha256
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
export RILEY_W4_TRACE_DIR="$stage/riley/rows" RILEY_W4_LAYER_REFERENCE="$previous/verified.json"
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
run_gpu riley "$python" -B "$stage/tools/run_accuracy.py" riley
mkdir hf
run_gpu hf bash -c '"$@" >hf/verified.json' hf timeout --signal=TERM --kill-after=20s 1000s \
    "$venv" -B "$stage/tools/replay_mlp.py" "$stage" "$operator" "$performance" "$previous/verified.json" "$model"
sha256sum binary.sha256 capture-source.sha256 checkpoint-verify.log riley/{client.log,receipt.json,server.log} \
    riley/rows/*/{rows.bf16,index.tsv} hf/verified.json gpu{6,7}-{riley,hf}-{before,after}.json >result.sha256
