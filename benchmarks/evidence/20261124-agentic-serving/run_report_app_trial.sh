#!/usr/bin/env bash
set -euo pipefail

binary=$1
model=$2
: "${RILEY_GPU_HOLDER_LEASE_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"

sha256sum "$binary" "$model/riley-checkpoint.json" > identity.sha256
read -r binary_sha _ < identity.sha256
[[ $binary_sha == 13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f ]]
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,uuid,name,driver_version --format=csv > device.csv
python3 -B -m benchmarks.live.report_app_trial prepare "$PWD/reports.sqlite"
sha256sum reports.sqlite > catalog.sha256

run() (
    name=$1
    mode=$2
    mkdir "$name"
    export RILEY_SHUTDOWN_METRICS_PATH="$PWD/$name/shutdown.json"
    args=(serve --model "$model" --model-id m --inference-profile native-bf16-varlen-v1
        --max-sequence-tokens 8192 --max-output-tokens 384 --max-active-sequences 8
        --kv-blocks 4096 --prefix-cache-blocks 1536 --batch-token-budget 1024
        --prefill-chunk-tokens 1024
        --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024
        --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 0
        --prefill-burst-min-prompt-tokens 512 --bind 127.0.0.1:0 --max-weight-bytes 68719476736)
    printf '%s\n' "${args[@]}" > "$name/server-arguments.txt"
    env -u RILEY_API_KEY "$binary" "${args[@]}" > "$name/server.log" 2>&1 &
    server_pid=$!
    app_pid=
    observer_pid=
    cleanup() {
        local cleanup_status=0
        if [[ -n $observer_pid ]]; then
            kill "$observer_pid" 2>/dev/null || true
            wait "$observer_pid" || true
        fi
        if [[ -n $app_pid ]]; then
            kill "$app_pid" 2>/dev/null || true
            wait "$app_pid" || true
        fi
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" || cleanup_status=$?
        printf '%s\n' "$cleanup_status" > "$name/server-exit.txt"
        nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader > "$name/compute-after.csv"
    }
    trap cleanup EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap 'exit 129' HUP
    printf '%s\n' "$server_pid" > "$name/server-pid.txt"
    (
        while kill -0 "$server_pid" 2>/dev/null; do
            date +%s%N
            nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
            sleep 2
        done
    ) > "$name/compute-during.txt" &
    observer_pid=$!
    deadline=$((SECONDS + 360))
    until address=$(sed -nE 's/.*listening on http:\/\/(127\.0\.0\.1:[0-9]+).*/\1/p' "$name/server.log") && [[ -n $address ]] && curl -fsS --max-time 2 "http://$address/readyz" > "$name/ready.json"; do
        kill -0 "$server_pid" && ((SECONDS < deadline)) || exit 1
        sleep .25
    done
    app_options=()
    measure_options=()
    if [[ $mode == baseline ]]; then
        app_options+=(--model-tools --full-catalog)
    else
        measure_options+=(--optimized)
    fi
    env -u RILEY_API_KEY python3 -B -m examples.report_agent.app "$PWD/reports.sqlite" \
        --riley "$address" --model m --port 0 --tool-delay-ms 500 "${app_options[@]}" > "$name/app.log" 2>&1 &
    app_pid=$!
    deadline=$((SECONDS + 20))
    until app_address=$(sed -n '/^http:/p' "$name/app.log") && [[ -n $app_address ]]; do
        kill -0 "$app_pid" && ((SECONDS < deadline)) || exit 1
        sleep .1
    done
    echo "$name app ready"
    python3 -B -m benchmarks.live.report_app_trial measure "$app_address" "$name/workflows.json" "${measure_options[@]}"
    curl -fsS --max-time 5 "http://$address/metrics" > "$name/metrics.json"
)

for name in a1 b1 b2 a2; do
    mode=optimized
    [[ $name != a* ]] || mode=baseline
    echo "$name $mode starting"
    run "$name" "$mode"
    [[ $(cat "$name/server-exit.txt") == 0 ]]
    test -s "$name/shutdown.json"
done
