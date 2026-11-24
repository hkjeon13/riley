#!/usr/bin/env bash
set -euo pipefail
: "${RILEY_GPU_HOLDER_LEASE_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
out=$1
pair=$2
keyfile=$3
mkdir "$out"
model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-7B-Instruct
image=sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24
RILEY_TRIAL_KEY=$(cat "$keyfile")
export RILEY_TRIAL_KEY
args=(serve --model /models --model-id m --inference-profile native-bf16-varlen-v1
    --max-sequence-tokens 8192 --max-output-tokens 384 --max-active-sequences 8 --max-waiting-requests 2
    --kv-blocks 4096 --prefix-cache-blocks 1536 --batch-token-budget 256
    --prefill-chunk-tokens 256 --api-key-env RILEY_TRIAL_KEY)
if [[ -n ${IMAGE_ROLLBACK_DIR:-} && ${out##*/} == survivor ]]; then
    image=$(cat "$IMAGE_ROLLBACK_DIR/candidate-image.id")
    args+=(--request-timeout-ms 120000)
fi
printf '%s\n' "${args[@]}" > "$out/arguments.txt"
container=$(docker create --pull=never --user "$(id -u):$(id -g)" --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --publish 127.0.0.1::8080 --mount "type=bind,source=$model,target=/models,readonly" \
    --env RILEY_TRIAL_KEY --env RILEY_SHUTDOWN_METRICS_PATH=/tmp/exit.json "$image" "${args[@]}")
cleanup() {
    docker stop -t 30 "$container" > /dev/null
    docker logs "$container" > "$out/server.log" 2>&1
    docker cp "$container:/tmp/exit.json" "$out/shutdown.json" || echo 'shutdown receipt unavailable' >&2
    docker rm "$container" > /dev/null
}
trap cleanup EXIT
if [[ -n ${IMAGE_ROLLBACK_DIR:-} ]]; then
    docker inspect -f '{{.Image}}' "$container" > "$out/image.id"
    docker inspect -f '{{range .Mounts}}{{println .Destination}}{{end}}' "$container" > "$out/mount-destinations"
fi
docker start "$container" > /dev/null
address=$(docker port "$container" 8080/tcp)
for ((i=0; i<300; i++)); do
    if curl -fsS --max-time 1 "http://$address/readyz" > "$out/ready.json" 2>/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || exit 1
    sleep 1
done
curl -fsS "http://$address/readyz" > "$out/ready.json"
if [[ -n ${IMAGE_ROLLBACK_DIR:-} ]]; then
    docker exec "$container" sha256sum /proc/1/exe > "$out/binary.sha256"
fi
docker exec "$container" nvidia-smi --query-gpu=uuid,name,driver_version --format=csv > "$out/device.csv"
printf '%s\n' "$address" > "$out/address.tmp"
mv "$out/address.tmp" "$out/address"
deadline=$((SECONDS + 900))
while ((SECONDS < deadline)); do
    if [[ -f $out/inject-process-failure ]]; then
        docker kill --signal KILL "$container" > /dev/null
        docker wait "$container" > "$out/fault-exit-code"
        exit 0
    fi
    if [[ -f $pair/finished || -f $out/retire ]]; then
        curl -fsS "http://$address/metrics" > "$out/metrics.json"
        exit 0
    fi
    if [[ -n ${OPERATIONAL_TRIAL:-} ]]; then sleep 0.05; else sleep 1; fi
done
exit 1
