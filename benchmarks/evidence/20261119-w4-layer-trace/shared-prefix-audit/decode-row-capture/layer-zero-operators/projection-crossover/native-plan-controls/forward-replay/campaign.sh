#!/usr/bin/env bash
set -euo pipefail
umask 077
stage=$(cd -- "$(dirname -- "$0")" && pwd -P)
cd "$stage"
evidence=$(dirname "$stage")
root=$(dirname "$evidence")
base="$evidence/w1-dist-d3d8fe6e5.YnZWdh5l"
prior="$evidence/w4-qkv-plan-20261120.hpF6Lg0W"
operator="$evidence/w4-native-plan-controls-20261120.n1qiR9dQ"
coder="$evidence/w3-coder14-qkv-20261120.Mx2OpwKm"
holder="$evidence/w2-sm17-20261118.6UMfeHh2"
model="$root/models/Qwen2.5-7B-Instruct"
python=/usr/bin/python3
trap 'printf "%d\n" "$?" >campaign.exit.tmp; mv -- campaign.exit.tmp campaign.exit' EXIT
sha256sum -c dispatch.sha256
for reference in "$prior" "$operator"; do
    test "$(cat "$reference/campaign.exit")" = 0
    (cd "$reference" && sha256sum -c result.sha256) >"$(basename "$reference").check"
done

printf 'WAIT %s %s\n' "$(basename "$coder")" "$(date -u +%FT%TZ)"
deadline=$((SECONDS+14400))
until test -s "$coder/campaign.exit"; do
    if ! ps -p 2896545 -o args= | grep -Fq -- "$coder/campaign.sh"; then
        test -s "$coder/campaign.exit" && break
        exit 75
    fi
    if (( SECONDS >= deadline )); then exit 75; fi
    sleep 20
done
printf 'PREDECESSOR_TERMINAL %s %s\n' "$(basename "$coder")" "$(cat "$coder/campaign.exit")"

mkdir -p tools src
cp -- "$prior/holder.py" .
cp -- "$prior/tools/"{run_accuracy,long_accuracy,compare_runner,capture_reference,capture_final_rows,replay_final_head,trace_decode_layers}.py tools/
cp -- capture_decode_rows.py tools/
test "$(sha256sum tools/run_accuracy.py | cut -d ' ' -f 1)" = 79afa94d08bc2b41abd72f91b8c88eb211f77e97dfb2648bcd2ddd008ebc698d
test "$(sha256sum tools/capture_reference.py | cut -d ' ' -f 1)" = fd91fa83c0761dc9a0095f62c1f592aaf9b4bce708898e2549d4eca04a8d3150
test "$(sha256sum tools/trace_decode_layers.py | cut -d ' ' -f 1)" = 550c6f54a0a6c00e595cc0b0de3c601e07e90a0d992d963910de10c7e5eb4179
ln -s -- "$prior/fixtures" fixtures
"$python" -B tools/capture_decode_rows.py --help >preflight.txt
printf '%s  %s\n' cbf8f5cd29ac5d4fa942d1e2792ea3c4559cf0a6a011700d8d16ba3e90ca5dd5     "$base/source.tar" | sha256sum -c
tar -xf "$base/source.tar" -C src --no-same-owner
(cd src && sha256sum -c "$prior/source-base.sha256" &&
    gzip -dc "$prior/decode-layer-trace.patch.gz" | git apply --check - &&
    gzip -dc "$prior/decode-layer-trace.patch.gz" | git apply - &&
    sha256sum -c "$prior/source-patched.sha256" &&
    git apply --check "$stage/qkv-row-replay.patch" && git apply "$stage/qkv-row-replay.patch" &&
    printf '%s  %s\n' 6b9b3e41f8efd9f013fa7c3f2b17919b47820380077db131493a2d62f9d0b70c         crates/riley-runtime/src/llama/executor/dispatch.rs | sha256sum -c)
printf '%s  %s\n' 655b7c01c2f3548a6aa35e0dbc9e7d97068f6d1f7e8e1ffa2bcb931e67f9e96f     "$model/riley-checkpoint.json" | sha256sum -c

toolchain="$root/tools/rustup/toolchains/1.85.0-x86_64-unknown-linux-gnu/bin"
export PATH="$toolchain:/usr/local/cuda-12.4/bin:$PATH"
export CARGO_HOME="$root/tools/cargo" CUDA_HOME=/usr/local/cuda-12.4 CUDAToolkit_ROOT=/usr/local/cuda-12.4
export CARGO_TARGET_DIR="$stage/target" RILEY_CUDA_ARCHITECTURES=80
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 W3_GPU_INDEX=7
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export RILEY_W4_TRACE_DIR="$stage/riley/rows"
printf 'BUILD %s\n' "$(date -u +%FT%TZ)"
(cd src && "$toolchain/cargo" build --locked --offline --release -p riley-server --features server,hf-exact-probe) >build.log 2>&1
sha256sum target/release/riley >binary.sha256

reservations() {
    local point=$1
    for index in 6 7; do
        "$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index "$index" status >"gpu$index-$point.json"
    done
    "$python" -B - "$point" <<'PY'
import json
from pathlib import Path
import sys
for index, pid, ticks in ((6, 2431228, "6339570869"), (7, 2431237, "6339570875")):
    row = json.loads(Path(f"gpu{index}-{sys.argv[1]}.json").read_bytes())
    if (row["physical_index"], row["pid"], row["start_ticks"], row["phase"]) != (index,pid,ticks,"idle") or \
            row["allocation_bytes"] != 20*1024**3 or row["torch_reserved_bytes"] != 20*1024**3 or row["other_processes"]:
        raise ValueError(f"GPU {index} is missing its original reservation")
PY
}
reservations before
rc=0
printf 'START riley %s\n' "$(date -u +%FT%TZ)"
"$python" -B "$holder/control.py" --lease-dir "$holder/leases" --index 7 run -- \
    "$python" -B "$stage/tools/run_accuracy.py" riley || rc=$?
reservations after
printf 'END riley %s exit=%d\n' "$(date -u +%FT%TZ)" "$rc"
test "$rc" = 0
"$python" -B - <<'PY'
import json
from pathlib import Path
receipt = json.loads(Path("riley/receipt.json").read_bytes())
if receipt["status"] != "passed" or receipt["server_exit"] != 0:
    raise ValueError("server or model diagnostic did not finish cleanly")
report = json.loads(Path("riley/client.log").read_bytes())
if not report["baseline_and_sham_bit_equal"] or len(report["cases"]) != 2:
    raise ValueError("model diagnostic controls failed")
Path("verified.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
PY
sha256sum binary.sha256 verified.json riley/{receipt.json,client.log,server.log} \
    riley/rows/*/{rows.bf16,index.tsv} riley/rows/*/qkv-override.bf16 gpu{6,7}-{before,after}.json >result.sha256
