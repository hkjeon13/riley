#!/usr/bin/env bash
set -euo pipefail
holder=$1
python=$2
stage=$3
cpu=$4
model=$5
gpu6=GPU-031a7f9d-5789-04b8-c9d6-63f55ee1e6a7
gpu7=GPU-d9e89dc4-e4e2-00d2-8a22-9d696d5b9ff0
test "$CUDA_VISIBLE_DEVICES" = "$gpu7"
export RILEY_GPU_PEER_HOLDER_GPU="$CUDA_VISIBLE_DEVICES"
export RILEY_GPU_PEER_HOLDER_PID="$RILEY_GPU_HOLDER_PID"
export RILEY_GPU_PEER_HOLDER_START_TICKS="$RILEY_GPU_HOLDER_START_TICKS"
exec /usr/bin/python3 -B "$holder/control.py" --lease-dir "$holder/leases" --index 6 run -- \
    "$python" -B "$stage/ci/check_distribution_gpu.py" "$cpu" "$gpu6" "$model" \
    c535a624b44249e514a8facd9c4cb7ccdfdf4a17f3b556e27c2c981fd16c9365 "$stage/gpu-evidence" \
    --tools --tp-holdout "$gpu7" "$stage/holdout.json.gz"
