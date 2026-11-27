#!/usr/bin/env bash
set -euo pipefail

binary=$1
model=$2
: "${RILEY_GPU_HOLDER_LEASE_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

sha256sum "$binary" "$model/riley-checkpoint.json" > identity.sha256
read -r binary_sha _ < identity.sha256
[[ $binary_sha == c179249ff85526546c220b0606051bdfba152eb81133600f91b856831314643c ]]
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,driver_version --format=csv > device.csv

run() (
    mode=$1
    attention=$2
    shift 2
    mkdir "$mode"
    export RILEY_SHUTDOWN_METRICS_PATH="$PWD/$mode/shutdown.json"
    args=(serve --model "$model" --model-id m --inference-profile native-bf16-varlen-v1
        --max-sequence-tokens 32768 --max-output-tokens 2048 --max-active-sequences 8
        --max-weight-bytes 68719476736
        --kv-blocks 4096 --batch-token-budget 256 --prefill-chunk-tokens 256
        --request-timeout-ms 120000 --bind 127.0.0.1:0 --native-attention-mode "$attention")
    printf '%s\n' "${args[@]}" > "$mode/arguments.txt"
    env -u RILEY_API_KEY "$binary" "${args[@]}" > "$mode/server.log" 2>&1 &
    pid=$!
    observer=
    cleanup() {
        local result=0
        if [[ -n $observer ]]; then
            kill "$observer" 2>/dev/null || true
            wait "$observer" || true
        fi
        kill -TERM "$pid" 2>/dev/null || true
        wait "$pid" || result=$?
        printf '%s\n' "$result" > "$mode/exit.txt"
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader > "$mode/compute-after.csv"
    }
    trap cleanup EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM
    printf '%s\n' "$pid" > "$mode/server-pid.txt"
    (
        while kill -0 "$pid" 2>/dev/null; do
            date +%s%N
            nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
            sleep 2
        done
    ) > "$mode/compute-during.txt" &
    observer=$!
    deadline=$((SECONDS + 360))
    until address=$(sed -nE 's/.*listening on http:\/\/(127\.0\.0\.1:[0-9]+).*/\1/p' "$mode/server.log") && [[ -n $address ]] && curl -fsS --max-time 2 "http://$address/readyz" > "$mode/ready.json"; do
        kill -0 "$pid" && ((SECONDS < deadline)) || exit 1
        sleep .25
    done
    printf '%s ready\n' "$mode"
    python3 -B -m benchmarks.live.long_search_trial "$address" "$mode/results.json" "$@"
    curl -fsS --max-time 5 "http://$address/metrics" > "$mode/metrics.json"
)

run page page-v1
[[ $(cat page/exit.txt) == 0 ]]
test -s page/shutdown.json
run fixed fixed-tile-unsplit-v1 --reverse
[[ $(cat fixed/exit.txt) == 0 ]]
test -s fixed/shutdown.json
sha256sum -c identity.sha256
