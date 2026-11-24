#!/usr/bin/env bash
set -euo pipefail
: "${RILEY_GPU_HOLDER_LEASE_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
out=$1
pair=$2
mkdir "$out"
model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-7B-Instruct
here=$(realpath "$(dirname "$0")")
binary=$(realpath "$here/../chat-owned-render-20261125.XCBDIb4g/server-control")
expected=3bc3132a4a2be81b24603d002286424b0351a9df08873cd6f58cc78034358cc2
if [[ $(basename "$pair") == prefix-recency-[01] ]]; then
    case "$(basename "$pair")/$(basename "$out")" in
        prefix-recency-0/gpu6|prefix-recency-1/gpu7) arm=control ;;
        prefix-recency-0/gpu7|prefix-recency-1/gpu6) arm=candidate ;;
        *) exit 1 ;;
    esac
    binary="$here/server-$arm"
    expected=$(awk -v name="server-$arm" '$2 == name { print $1 }' "$here/artifacts.sha256")
elif [[ ${CROSS_ROUTER_HEADROOM:-0} == 1 ]]; then
    binary=$here/server
    (cd "$here" && sha256sum -c server.sha256)
    expected=$(cut -d' ' -f1 "$here/server.sha256")
fi
sha256sum "$binary" > "$out/binary.sha256"
test "$(cut -d' ' -f1 "$out/binary.sha256")" = "$expected"
RILEY_TRIAL_KEY=$(cat "$3")
export RILEY_TRIAL_KEY RILEY_SHUTDOWN_METRICS_PATH="$out/shutdown.json"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
args=(serve --bind 127.0.0.1:0 --model "$model" --model-id m --inference-profile native-bf16-varlen-v1
    --max-weight-bytes 68719476736 --max-sequence-tokens 8192 --max-output-tokens 384 --max-active-sequences 8 --max-waiting-requests 2
    --kv-blocks 4096 --prefix-cache-blocks 1536 --batch-token-budget 256
    --prefill-chunk-tokens 256 --api-key-env RILEY_TRIAL_KEY)
printf '%s\n' "${args[@]}" > "$out/arguments.txt"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version --format=csv > "$out/device.csv"
"$binary" "${args[@]}" > "$out/server.log" 2>&1 &
pid=$!
cleanup() {
    if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; fi
    wait "$pid"
}
trap cleanup EXIT
deadline=$((SECONDS+300))
until address=$(sed -nE 's/.*listening on http:\/\/(127\.0\.0\.1:[0-9]+).*/\1/p' "$out/server.log") && [[ -n $address ]]; do
    kill -0 "$pid" && ((SECONDS < deadline)) || exit 1
    sleep 1
done
curl -fsS "http://$address/readyz" > "$out/ready.json"
printf '%s\n' "$address" > "$out/address.tmp"
mv "$out/address.tmp" "$out/address"
deadline=$((SECONDS+900))
until [[ -e $pair/finished ]]; do
    kill -0 "$pid" && ((SECONDS < deadline)) || exit 1
    sleep 1
done
curl -fsS "http://$address/metrics" > "$out/metrics.json"
