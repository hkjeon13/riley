#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
cache=$1
out=$2
mkdir "$out"
model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-7B-Instruct
image=sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24
sha256sum "$model/riley-checkpoint.json" agent_workflow.py registration.json > "$out/inputs.sha256"
cache_args=()
if ((cache > 0)); then cache_args=(--prefix-cache-blocks "$cache"); fi
container=$(docker create --pull=never --user "$(id -u):$(id -g)" --gpus "device=$CUDA_VISIBLE_DEVICES"     --publish 127.0.0.1::8080 --mount "type=bind,source=$model,target=/models,readonly"     "$image" serve --model /models --model-id m --inference-profile native-bf16-varlen-v1     --max-sequence-tokens 8192 --max-output-tokens 384 --max-active-sequences 8     --kv-blocks 4096 "${cache_args[@]}" --batch-token-budget 256 --prefill-chunk-tokens 256)
cleanup() {
    docker stop -t 30 "$container" > /dev/null
    docker logs "$container" > "$out/server.log" 2>&1
    docker rm "$container" > /dev/null
}
trap cleanup EXIT
docker start "$container" >/dev/null
address=$(docker port "$container" 8080/tcp)
for ((i=0; i<180; i++)); do
    if curl -fsS --max-time 1 "http://$address/readyz" > "$out/ready.json" 2>/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || exit 1
    sleep 1
done
curl -fsS "http://$address/readyz" > "$out/ready.json"
docker exec "$container" sha256sum /proc/1/exe > "$out/binary.sha256"
docker exec "$container" nvidia-smi --query-gpu=uuid,name,driver_version --format=csv > "$out/device.csv"
curl -fsS "http://$address/metrics" > "$out/before.json"
python3 -B agent_workflow.py "$address" m --out "$out/workflows.json"
curl -fsS "http://$address/metrics" > "$out/after.json"
