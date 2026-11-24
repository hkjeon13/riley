#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gpu=$1
out="gpu$gpu"
mkdir "$out"
case "$gpu" in
    6) cores=24-33,72-81 ;;
    7) cores=34-43,82-91 ;;
    *) exit 2 ;;
esac
model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-Coder-14B-Instruct
image=sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24
args=(serve --model /models --model-id m --inference-profile native-bf16-varlen-v1
    --max-weight-bytes 68719476736 --max-sequence-tokens 8192 --max-output-tokens 256
    --max-active-sequences 32 --kv-blocks 2576)
if [[ ${CODER_GAP_BUDGET:-} == 512 ]]; then
    args+=(--batch-token-budget 512 --prefill-chunk-tokens 512
        --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512
        --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 8 --prefill-burst-min-prompt-tokens 512)
else
    args+=(--batch-token-budget 256 --prefill-chunk-tokens 256)
fi
printf '%s\n' "${args[@]}" > "$out/args"
c=$(docker create --pull=never --user "$(id -u):$(id -g)" --gpus "device=$CUDA_VISIBLE_DEVICES" --cpuset-cpus "$cores" \
    --publish 127.0.0.1::8080 --mount "type=bind,source=$model,target=/models,readonly" \
    --env RILEY_SHUTDOWN_METRICS_PATH=/tmp/exit.json "$image" "${args[@]}")
cleanup() {
    docker stop -t 30 "$c" >/dev/null
    docker logs "$c" > "$out/server.log" 2>&1
    docker cp "$c:/tmp/exit.json" "$out/shutdown.json" || true
    docker rm "$c" >/dev/null
}
trap cleanup EXIT
docker start "$c" >/dev/null
addr=$(docker port "$c" 8080/tcp)
for ((i=0; i<360; i++)); do
    if curl -fsS --max-time 1 "http://$addr/readyz" > "$out/ready.json" 2>/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$c") == true ]] || exit 1
    sleep 1
done
curl -fsS "http://$addr/readyz" > "$out/ready.json"
docker exec "$c" sha256sum /proc/1/exe > "$out/binary.sha256"
printf '%s\n' "$addr" > "$out/address.tmp"
mv "$out/address.tmp" "$out/address"
for ((i=0; i<1200; i++)); do
    if [[ -f finished ]]; then
        curl -fsS "http://$addr/metrics" > "$out/metrics.json"
        exit 0
    fi
    sleep 1
done
exit 2
