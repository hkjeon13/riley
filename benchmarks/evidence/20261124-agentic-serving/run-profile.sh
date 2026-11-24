#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=$1
binary=$2
seed=$3
policy=${4:-control}
mkdir "$out"
model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-7B-Instruct
if [[ -n ${AGENT_DECODE_MODEL:-} ]]; then
    model="/mnt/efs/nlp/psyche/riley-gap-20261102/models/$AGENT_DECODE_MODEL"
elif [[ ${LONG_CONTEXT_MODEL:-} == llama31-8b ]]; then
    model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Meta-Llama-3.1-8B-Instruct
elif [[ -n ${PROTECTED_INDEX:-} ]]; then
    model=/mnt/efs/nlp/psyche/riley-gap-20261102/models/Qwen2.5-Coder-14B-Instruct
fi
image=sha256:67efe6fbdaebcc8a088eaa9626f1adb43cf1af3e673ccae2d9af076fa99bdd24
opts=(--user "$(id -u):$(id -g)" --gpus "device=$CUDA_VISIBLE_DEVICES" --publish 127.0.0.1::8080)
opts+=(--mount "type=bind,source=$model,target=/models,readonly")
if [[ $binary != stock ]]; then
    opts+=(--mount "type=bind,source=$binary,target=/opt/riley/bin/riley,readonly")
fi
opts+=(--env RILEY_SHUTDOWN_METRICS_PATH=/tmp/exit.json)
server=(serve --model /models --model-id m --inference-profile native-bf16-varlen-v1)
if [[ -n ${TRIAL_REGISTER_LAYERS:-} ]]; then
    trace_dir=$(realpath "$out")/trace
    mkdir -p "$trace_dir"/{lookup-541,lookup-862}-{prefill,decode}
    opts+=(--mount "type=bind,source=$trace_dir,target=/trace" --env RILEY_REGISTER_LAYER_TRACE=/trace)
    server+=(--metadata-transport synchronous --execution-completion iteration-batch --decode-step-graph disabled)
fi
if [[ -n ${TRIAL_REQUEST_TIMEOUT_MS:-} ]]; then
    server+=(--request-timeout-ms "$TRIAL_REQUEST_TIMEOUT_MS")
fi
if [[ ${TRIAL_ATTENTION_MODE:-} == page-v1 ]]; then
    server+=(--native-attention-mode page-v1)
fi
if [[ -n ${PROTECTED_INDEX:-} ]]; then
    case "$PROTECTED_INDEX" in
        6) opts+=(--cpuset-cpus 24-33,72-81);;
        7) opts+=(--cpuset-cpus 34-43,82-91);;
        *) exit 2;;
    esac
    server+=(--max-weight-bytes 68719476736 --max-sequence-tokens 8192 --max-output-tokens 256
        --max-active-sequences 32 --kv-blocks 2576)
else
    server+=(--max-sequence-tokens "${TRIAL_SEQUENCE_LIMIT:-8192}" --max-output-tokens "${TRIAL_OUTPUT_LIMIT:-384}" --max-active-sequences "${TRIAL_ACTIVE_LIMIT:-8}"
        --kv-blocks "${TRIAL_KV_BLOCKS:-4096}")
    if [[ ${TRIAL_PREFIX_BLOCKS:-1536} != 0 ]]; then
        server+=(--prefix-cache-blocks "${TRIAL_PREFIX_BLOCKS:-1536}")
    fi
fi
case "$policy" in
    named-prefix32) server=(serve --model /models --model-id m --inference-profile native-bf16-prefix-32k-v1);;
    long-page) server+=(--batch-token-budget 32 --prefill-chunk-tokens 32 --batch-shape-buckets 1,2,4,8,16,32
        --prefill-plateau-rows 8,16 --decode-step-graph disabled --max-weight-bytes 68719476736);;
    control) server+=(--batch-token-budget 256 --prefill-chunk-tokens 256);;
    wide) server+=(--batch-token-budget 1024 --prefill-chunk-tokens 1024
        --batch-shape-buckets 1,2,4,8,16,24,32,48,64,80,96,112,128,144,160,176,192,208,224,240,256,512,1024
        --prefill-burst-steady-budget 256 --prefill-burst-max-decode-rows 0 --prefill-burst-min-prompt-tokens 512);;
    profile-128|profile-256|profile-512|profile-1024)
        width=${policy#profile-}
        buckets=()
        for rows in 1 2 4 8 16 24 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256 512 1024; do
            if ((rows <= width)); then buckets+=("$rows"); fi
        done
        if ((width == 128)); then server+=(--prefill-plateau-rows 64,96 --prefill-plateau-min-decode-rows 12); fi
        joined=$(IFS=,; echo "${buckets[*]}")
        server+=(--batch-token-budget "$width" --prefill-chunk-tokens "$width" --batch-shape-buckets "$joined");;
    *) exit 2;;
esac
if [[ -n ${AGENT_DECODE_MODEL:-} ]]; then
    : "${RILEY_GPU_HOLDER_LEASE_DIR:?}" "${CUDA_VISIBLE_DEVICES:?}"
    if [[ -n ${RILEY_TRIAL_BINARY:-} ]]; then
        executable=$(realpath "$RILEY_TRIAL_BINARY")
        expected=${RILEY_TRIAL_BINARY_SHA256:?}
    else
        executable=$(realpath ../prefix-range-20261125.bn73vRNe/server-candidate)
        expected=13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f
    fi
    workload=model_decode_workload.py
    if [[ -n ${CODER_PREFILL_TRACE:-} ]]; then
        executable=$(realpath trace-server)
        expected=$(cut -d' ' -f1 trace-server.sha256)
        workload=coder_prefill_trace.py
        if [[ ${CODER_PHASE_METRICS:-0} == 1 ]]; then workload=iteration_phase_trial.py; fi
        if [[ -n ${CODER_WAIT_MODE:-} ]]; then workload=coder_wait_trial.py; fi
    fi
    sha256sum "$executable" > "$out/binary.sha256"
    test "$(cut -d' ' -f1 "$out/binary.sha256")" = "$expected"
    if [[ -n ${TRIAL_HTTP_WORKERS:-} ]]; then server+=(--http-workers "$TRIAL_HTTP_WORKERS"); fi
    server[2]="$model"
    server+=(--bind 127.0.0.1:0)
    if [[ $AGENT_DECODE_MODEL == Qwen2.5-Coder-7B-Instruct ]]; then
        server+=(--max-weight-bytes 68719476736)
    fi
    printf '%s\n' "${server[@]}" > "$out/server-arguments.txt"
    export RILEY_SHUTDOWN_METRICS_PATH="$(realpath "$out")/shutdown.json"
    export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}"
    nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,name,driver_version --format=csv > "$out/device.csv"
    "$executable" "${server[@]}" > "$out/server.log" 2>&1 &
    pid=$!
    stop_server() { if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid"; fi; wait "$pid"; }
    trap stop_server EXIT
    deadline=$((SECONDS+360))
    until address=$(sed -nE 's/.*listening on http:\/\/(127\.0\.0\.1:[0-9]+).*/\1/p' "$out/server.log") && [[ -n $address ]]; do
        kill -0 "$pid" && ((SECONDS < deadline)) || exit 1
        sleep 1
    done
    curl -fsS "http://$address/readyz" > "$out/ready.json"
    if [[ -n ${CODER_SCALE_SERVER:-} ]]; then
        printf '%s\n' "$address" > "$out/address.tmp"
        mv "$out/address.tmp" "$out/address"
        for ((i=0; i<600; i++)); do
            [[ ! -f finished ]] || break
            sleep 1
        done
        test -f finished || exit 2
    elif [[ -n ${CODER_WINDOW_GPU:-} ]]; then
        python3 -B coder_window_trial.py "$address" "$out" "$CODER_WINDOW_GPU"
    elif [[ -n ${CODER_RELEASE_GPU:-} ]]; then
        python3 -B coder_release_trial.py "$address" "$out" "$CODER_RELEASE_GPU"
    elif [[ -n ${CODER_NAMED_MODE:-} ]]; then
        python3 -B coder_named_burst.py "$address" "$out" "$CODER_NAMED_MODE"
    elif [[ -n ${CODER_CATALOG_COMPARE:-} ]]; then
        python3 -B coder_catalog_compare.py "$address" "$out"
    elif [[ -n ${CODER_CATALOG_MODE:-} ]]; then
        python3 -B coder_catalog_trial.py "$address" "$out" "$CODER_CATALOG_MODE"
    elif [[ -n ${CODER_PRESSURE_MODE:-} ]]; then
        python3 -B coder_pressure.py "$address" "$out" "$CODER_PRESSURE_MODE"
    elif [[ -n ${CODER_WAIT_CLIENT:-} ]]; then
        python3 -B qualified_tool_wait.py "$address" "$out" coder-wait-client-registration.json --include-completed-call
    else
        python3 -B "$workload" "$address" "$out" "$FROZEN_INPUT_DIR"
    fi
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
printf '%s\n' "${server[@]}" > "$out/server-arguments.txt"
container=$(docker create --pull=never "${opts[@]}" "$image" "${server[@]}")
cleanup() {
    docker stop -t 30 "$container" >/dev/null
    docker logs "$container" > "$out/server.log" 2>&1
    docker cp "$container:/tmp/exit.json" "$out/shutdown.json" || echo 'shutdown receipt unavailable' >&2
    docker rm "$container" > /dev/null
}
trap cleanup EXIT
docker start "$container" >/dev/null
address=$(docker port "$container" 8080/tcp)
for ((i=0; i<300; i++)); do
    if curl -fsS --max-time 1 "http://$address/readyz" > "$out/ready.json" 2>/dev/null; then break; fi
    [[ $(docker inspect -f '{{.State.Running}}' "$container") == true ]] || exit 1
    sleep 1
done
curl -fsS "http://$address/readyz" > "$out/ready.json"
docker exec "$container" sha256sum /proc/1/exe > "$out/binary.sha256"
docker exec "$container" nvidia-smi --query-gpu=uuid,name,driver_version --format=csv > "$out/device.csv"
if [[ -n ${PROTECTED_INDEX:-} ]]; then
    archive=/mnt/efs/nlp/psyche/riley-gap-20261102/evidence/w3-prefill-cohort-20261124.fC0iBjq3
    runner=/mnt/efs/nlp/psyche/riley-gap-20261102/tools/vllm-venv/bin/python
    python3 -B protected-plan.py "$archive" "$out/plan.json" "$PROTECTED_INDEX"
    W3_CLIENT_CPU=$((PROTECTED_INDEX + 38)) PYTHONPATH="$archive:$archive/tools" \
        "$runner" -B "$archive/client.py" "$out/plan.json" "$address"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${CHAT_RENDER_TRIAL:-} ]]; then
    python3 -B chat_render_trial.py serve "$address" "$out/requests.json"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${FINAL_LAYOUT:-} ]]; then
    python3 -B final_layout_trial.py "$address" "$out/workflows.json" riley
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${TRIAL_REGISTER_LAYERS:-} ]]; then
    python3 -B register_layers.py "$out/witness.json" --address "$address"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${TRIAL_REGISTER_SCORE:-} ]]; then
    python3 -B register_score_workload.py "$out/score.json" --address "$address"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${REGISTER_REDUCTION_GPU:-} ]]; then
    if [[ -f register-provenance-registration.json ]]; then
        python3 -B register_provenance.py "$address" "$out/provenance.json" "$REGISTER_REDUCTION_GPU"
    else
        python3 -B register_reduction.py "$address" "$out/reduction.json" "$REGISTER_REDUCTION_GPU"
    fi
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${REGISTER_TOKEN_WITNESS:-} ]]; then
    python3 -B register_reference.py "$out/witness.json" --address "$address"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${FULL_REGISTER_GPU:-} ]]; then
    order=()
    if [[ $FULL_REGISTER_GPU == 7 ]]; then order+=(--reverse); fi
    python3 -B full_register.py "$address" "$out/full-register.json" "${order[@]}"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${EXACT_RECORD_TRIAL:-} ]]; then
    order=()
    if [[ $EXACT_RECORD_TRIAL == reverse ]]; then order+=(--reverse); fi
    python3 -B validate_projection.py "$address" "$out/projection.json" "${order[@]}"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${LONG_TOOL_RECORDS:-} ]]; then
    python3 -B long_tool_records.py "$address" "$out/tool-records.json"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${LONG_TILE_HOLDOUT:-} ]]; then
    python3 -B long_tile_holdout.py "$address" "$out/holdout.json"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${PUBLIC_CONTINUATION_TRIAL:-} ]]; then
    python3 -B public_cached_continuations.py run "$address" "$PUBLIC_CONTINUATION_TRIAL" "$out/continuations.json"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${PUBLIC_SHAPE_TRIAL:-} ]]; then
    if [[ ${LONG_CONTEXT_MODEL:-} == llama31-8b || -n ${TRIAL_RETRIEVAL_FILLER:-} ]]; then
        python3 -B probe_long_retrieval.py "$address" "$out/retrieval.json" --filler "${TRIAL_RETRIEVAL_FILLER:-45200}"
    fi
    python3 -B public_agent_shapes.py run "$address" "$PUBLIC_SHAPE_TRIAL" "$out/shapes.json"
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
if [[ -n ${AGENT_SOAK_SECONDS:-} ]]; then
    python3 -B agent_soak.py "$address" "$out" --seconds "$AGENT_SOAK_SECONDS"
    exit 0
fi
if [[ $policy == profile-* ]]; then
    for trial_seed in "$seed" "$((seed+10))" "$((seed+660))"; do
        python3 -B agent_workflow.py "$address" m --cohort --seed "$trial_seed" --history-entries 160 \
            --out "$out/cohort-$trial_seed.json" || test -s "$out/cohort-$trial_seed.json"
    done
    curl -fsS "http://$address/metrics" > "$out/metrics.json"
    exit 0
fi
for history in 0 200; do
    curl -fsS "http://$address/metrics" > "$out/before-$history.json"
    python3 -B agent_workflow.py "$address" m --seed "$seed" --history-entries "$history" --out "$out/workflows-$history.json"
    curl -fsS "http://$address/metrics" > "$out/after-$history.json"
done
if [[ -n ${FROZEN_INPUT_DIR:-} ]]; then
    python3 -B fixed_agent_decode.py "$address" "$FROZEN_INPUT_DIR" "$out/fixed.json"
fi
if [[ -n ${WORKFLOW_PREFIX_TRIAL:-} ]]; then
    python3 -B tool_wait_trial.py "$address" "$WORKFLOW_PREFIX_TRIAL" "$out/prefix-trial.json"
fi
if [[ -f citation-registration.json && -f verify_agent_citations.py ]]; then
    python3 -B verify_agent_citations.py "$address" "$out" citation-registration.json
fi
if [[ -f wait-policy-registration.json && -f qualified_tool_wait.py ]]; then
    python3 -B qualified_tool_wait.py "$address" "$out" wait-policy-registration.json
fi
