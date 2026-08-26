#!/usr/bin/env bash
# Remote-only PR16 release gate. The host may use Python/jq/docker, but the
# immutable production container is network-isolated and must contain none of
# those Python-family runtime dependencies.
set -euo pipefail

export LC_ALL=C
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
checker="$repo_root/benchmarks/scripts/check_python_free_release_e2e.py"
packager="$repo_root/ci/release/package_python_free_e2e_evidence.py"

: "${RUSTINFER_E2E_OUTPUT:?set a new absolute evidence directory}"
: "${RUSTINFER_E2E_IMAGE_ID:?set the immutable sha256: release image ID}"
: "${RUSTINFER_E2E_SOURCE_REVISION:?set the exact clean 40-character revision}"
: "${RUSTINFER_E2E_SOURCE_ARCHIVE:?set the source archive path}"
: "${RUSTINFER_E2E_SOURCE_ARCHIVE_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_RELEASE_BINARY:?set the standalone release binary path}"
: "${RUSTINFER_E2E_RELEASE_BINARY_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_RELEASE_BUNDLE:?set the release bundle path}"
: "${RUSTINFER_E2E_RELEASE_BUNDLE_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_MODEL_DIR:?set the real checkpoint directory}"
: "${RUSTINFER_E2E_MODEL_TREE_SHA256:?set its reviewed canonical tree SHA-256}"
: "${RUSTINFER_E2E_MODEL_REVISION:?set the immutable model revision}"
: "${RUSTINFER_E2E_CONFIG_SHA256:?set the reviewed config.json SHA-256}"
: "${RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH:?set the primary weights path below the model directory}"
: "${RUSTINFER_E2E_WEIGHTS_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH:?set the tokenizer path below the model directory}"
: "${RUSTINFER_E2E_TOKENIZER_JSON_SHA256:?set the reviewed tokenizer.json SHA-256}"
: "${RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256:?set the native correctness tokenizer aggregate SHA-256}"
: "${RUSTINFER_E2E_CORRECTNESS_GOLDEN:?set the approved E2E golden JSON path}"
: "${RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_CORRECTNESS_REPORT:?set the passing native E0 correctness report}"
: "${RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256:?set its reviewed SHA-256}"
: "${RUSTINFER_E2E_DEVICE:=0}"
: "${RUSTINFER_E2E_BIND:=127.0.0.1:8080}"
: "${RUSTINFER_E2E_CANCEL_TOKENS:=512}"

for tool in bash docker jq python3 sha256sum find sort awk sed grep tail date mktemp readelf wc; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "required host tool is unavailable: $tool" >&2
        exit 2
    }
done

sha_re='^[0-9a-f]{64}$'
git_re='^[0-9a-f]{40}$'
[[ $RUSTINFER_E2E_SOURCE_REVISION =~ $git_re ]] || { echo "invalid source revision" >&2; exit 2; }
[[ $RUSTINFER_E2E_IMAGE_ID =~ ^sha256:[0-9a-f]{64}$ ]] || { echo "image must be immutable sha256:<digest>" >&2; exit 2; }
for digest in \
    "$RUSTINFER_E2E_SOURCE_ARCHIVE_SHA256" \
    "$RUSTINFER_E2E_RELEASE_BINARY_SHA256" \
    "$RUSTINFER_E2E_RELEASE_BUNDLE_SHA256" \
    "$RUSTINFER_E2E_MODEL_TREE_SHA256" \
    "$RUSTINFER_E2E_CONFIG_SHA256" \
    "$RUSTINFER_E2E_WEIGHTS_SHA256" \
    "$RUSTINFER_E2E_TOKENIZER_JSON_SHA256" \
    "$RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256" \
    "$RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256" \
    "$RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256"
do
    [[ $digest =~ $sha_re ]] || { echo "invalid reviewed SHA-256 binding" >&2; exit 2; }
done
[[ $RUSTINFER_E2E_DEVICE =~ ^[0-9]+$ ]] || { echo "device must be an ordinal" >&2; exit 2; }
test "$RUSTINFER_E2E_BIND" = 127.0.0.1:8080 || { echo "network-none probe bind must be 127.0.0.1:8080" >&2; exit 2; }
[[ $RUSTINFER_E2E_CANCEL_TOKENS =~ ^[0-9]+$ ]] || { echo "cancel token bound must be numeric" >&2; exit 2; }
if [ "$RUSTINFER_E2E_CANCEL_TOKENS" -lt 32 ] || [ "$RUSTINFER_E2E_CANCEL_TOKENS" -gt 1024 ]; then
    echo "cancel token bound must be from 32 through 1024" >&2
    exit 2
fi
case "$RUSTINFER_E2E_OUTPUT" in /*) ;; *) echo "output must be absolute" >&2; exit 2 ;; esac
case "$RUSTINFER_E2E_MODEL_DIR" in /*) ;; *) echo "model directory must be absolute" >&2; exit 2 ;; esac
for relative_artifact in "$RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH" "$RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH"; do
case "$relative_artifact" in
    /*|..|../*|*/../*|*/..) echo "model artifact paths must remain below the model directory" >&2; exit 2 ;;
esac
done
for path in "$RUSTINFER_E2E_OUTPUT" "$RUSTINFER_E2E_MODEL_DIR"; do
    [[ $path != *$'\n'* && $path != *','* ]] || { echo "paths must not contain newlines or commas" >&2; exit 2; }
done
test ! -e "$RUSTINFER_E2E_OUTPUT"
test -d "$RUSTINFER_E2E_MODEL_DIR"
test -f "$RUSTINFER_E2E_MODEL_DIR/config.json" && test ! -L "$RUSTINFER_E2E_MODEL_DIR/config.json"
test -f "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH"
test ! -L "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH"
test -f "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH"
test ! -L "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH"
for artifact in \
    "$RUSTINFER_E2E_SOURCE_ARCHIVE" \
    "$RUSTINFER_E2E_RELEASE_BINARY" \
    "$RUSTINFER_E2E_RELEASE_BUNDLE" \
    "$RUSTINFER_E2E_CORRECTNESS_GOLDEN" \
    "$RUSTINFER_E2E_CORRECTNESS_REPORT"
do
    test -f "$artifact" && test ! -L "$artifact"
done

sha_file() { sha256sum "$1" | awk '{print $1}'; }
test "$(sha_file "$RUSTINFER_E2E_SOURCE_ARCHIVE")" = "$RUSTINFER_E2E_SOURCE_ARCHIVE_SHA256"
test "$(sha_file "$RUSTINFER_E2E_RELEASE_BINARY")" = "$RUSTINFER_E2E_RELEASE_BINARY_SHA256"
test "$(sha_file "$RUSTINFER_E2E_RELEASE_BUNDLE")" = "$RUSTINFER_E2E_RELEASE_BUNDLE_SHA256"
test "$(sha_file "$RUSTINFER_E2E_MODEL_DIR/config.json")" = "$RUSTINFER_E2E_CONFIG_SHA256"
test "$(sha_file "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH")" = "$RUSTINFER_E2E_WEIGHTS_SHA256"
test "$(sha_file "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH")" = "$RUSTINFER_E2E_TOKENIZER_JSON_SHA256"
test "$(sha_file "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256"
test "$(sha_file "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256"
python3 "$repo_root/ci/release/verify_release_bundle.py" "$RUSTINFER_E2E_RELEASE_BUNDLE" >/dev/null

scratch=$(mktemp -d)
container_id=
container_ids=()
cleanup() {
    local cleanup_id
    for cleanup_id in "${container_ids[@]}"; do
        docker rm -f "$cleanup_id" >/dev/null 2>&1 || true
    done
    rm -rf -- "$scratch"
}
trap cleanup EXIT

model_manifest="$scratch/model.SHA256SUMS"
: >"$model_manifest"
if find "$RUSTINFER_E2E_MODEL_DIR" -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo "model tree contains a symlink or non-regular entry" >&2
    exit 2
fi
model_file_count=0
while IFS= read -r -d '' model_file; do
    relative=${model_file#"$RUSTINFER_E2E_MODEL_DIR"/}
    [[ $relative =~ ^[A-Za-z0-9._/+@=-]+$ ]] || {
        echo "model paths must use the safe ASCII path alphabet" >&2
        exit 2
    }
    printf '%s  %s\n' "$(sha_file "$model_file")" "$relative" >>"$model_manifest"
    model_file_count=$((model_file_count + 1))
done < <(find "$RUSTINFER_E2E_MODEL_DIR" -type f -print0 | sort -z)
test "$model_file_count" -gt 0
test "$(sha_file "$model_manifest")" = "$RUSTINFER_E2E_MODEL_TREE_SHA256"

golden_schema=$(jq -er '.schema_version' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")
test "$golden_schema" = rustinfer.python-free-release-e2e-golden.v1
jq -e 'keys == ["config_sha256","correctness_gate_id","correctness_report_sha256","expected_greedy_text_sha256","max_tokens","model_id","model_revision","prompt","schema_version","source_revision","tokenizer_aggregate_sha256","tokenizer_json_sha256","weights_sha256"]' \
    "$RUSTINFER_E2E_CORRECTNESS_GOLDEN" >/dev/null
test "$(jq -er '.correctness_gate_id' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = smollm2-fp32-bf16-native-e0-v3
test "$(jq -er '.correctness_report_sha256' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256"
test "$(jq -er '.source_revision' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_SOURCE_REVISION"
test "$(jq -er '.model_revision' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_MODEL_REVISION"
test "$(jq -er '.config_sha256' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_CONFIG_SHA256"
test "$(jq -er '.weights_sha256' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_WEIGHTS_SHA256"
test "$(jq -er '.tokenizer_json_sha256' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_TOKENIZER_JSON_SHA256"
test "$(jq -er '.tokenizer_aggregate_sha256' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")" = "$RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256"
test "$(jq -er '.gate_id' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = smollm2-fp32-bf16-native-e0-v3
test "$(jq -er '.status' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = pass
test "$(jq -er '.bindings.candidate_git_revision' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_SOURCE_REVISION"
test "$(jq -er '.bindings.candidate_git_status_sha256' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$(printf '' | sha256sum | awk '{print $1}')"
test "$(jq -er '.bindings.model_revision' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_MODEL_REVISION"
test "$(jq -er '.bindings.config_sha256' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_CONFIG_SHA256"
test "$(jq -er '.bindings.weights_sha256' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_WEIGHTS_SHA256"
test "$(jq -er '.bindings.tokenizer_sha256' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256"
model_id=$(jq -er '.model_id | select(test("^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"))' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")
test "$(jq -er '.bindings.model_id' "$RUSTINFER_E2E_CORRECTNESS_REPORT")" = "$model_id"
golden_prompt=$(jq -er '.prompt | select(type == "string" and length > 0)' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")
[[ $golden_prompt != *$'\n'* && $golden_prompt != *$'\r'* ]] || { echo "golden prompt must be one line" >&2; exit 2; }
golden_max_tokens=$(jq -er '.max_tokens | select(type == "number" and floor == . and . >= 2 and . <= 1024)' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")
approved_text_sha256=$(jq -er '.expected_greedy_text_sha256 | select(test("^[0-9a-f]{64}$"))' "$RUSTINFER_E2E_CORRECTNESS_GOLDEN")
prompt_sha256=$(printf '%s' "$golden_prompt" | sha256sum | awk '{print $1}')

resolved_image=$(docker image inspect --format '{{.Id}}' "$RUSTINFER_E2E_IMAGE_ID")
test "$resolved_image" = "$RUSTINFER_E2E_IMAGE_ID"
mkdir -m 0700 "$RUSTINFER_E2E_OUTPUT"
mkdir -m 0777 "$RUSTINFER_E2E_OUTPUT/container-evidence"
shutdown_metrics="$RUSTINFER_E2E_OUTPUT/container-evidence/shutdown-metrics.json"
repeat_shutdown_metrics="$RUSTINFER_E2E_OUTPUT/container-evidence/repeat-shutdown-metrics.json"
image_inspect="$scratch/image-inspect.json"
docker image inspect "$RUSTINFER_E2E_IMAGE_ID" >"$image_inspect"

launch_container() {
    local metrics_name=$1
    docker run --detach \
        --network none \
        --gpus "device=$RUSTINFER_E2E_DEVICE" \
        --read-only \
        --tmpfs /tmp:rw,nosuid,nodev,noexec,size=16m \
        --mount "type=bind,src=$RUSTINFER_E2E_MODEL_DIR,dst=/models/checkpoint,readonly" \
        --mount "type=bind,src=$RUSTINFER_E2E_OUTPUT/container-evidence,dst=/evidence" \
        --env "RUSTINFER_SHUTDOWN_METRICS_PATH=/evidence/$metrics_name" \
        "$RUSTINFER_E2E_IMAGE_ID" \
        serve --model /models/checkpoint --model-id "$model_id" --bind "$RUSTINFER_E2E_BIND" \
        --max-output-tokens 1024
}

container_id=$(launch_container shutdown-metrics.json)
container_ids+=("$container_id")
[[ $container_id =~ ^[0-9a-f]{64}$ ]]
network_mode=$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$container_id")
test "$network_mode" = none
container_first_pre="$scratch/container-first-pre.json"
process_first_pre="$scratch/process-first-pre.txt"
docker inspect "$container_id" >"$container_first_pre"
docker top "$container_id" -eo pid,ppid,comm,args >"$process_first_pre"

container_http() {
    local method=$1 target=$2 body=$3 output=$4
    docker exec \
        --env "RUSTINFER_HTTP_METHOD=$method" \
        --env "RUSTINFER_HTTP_TARGET=$target" \
        --env "RUSTINFER_HTTP_BODY=$body" \
        "$container_id" /bin/bash -c '
            set -euo pipefail
            export LC_ALL=C
            exec 3<>/dev/tcp/127.0.0.1/8080
            if [ "$RUSTINFER_HTTP_METHOD" = POST ]; then
                printf "%s %s HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s" \
                    "$RUSTINFER_HTTP_METHOD" "$RUSTINFER_HTTP_TARGET" "${#RUSTINFER_HTTP_BODY}" "$RUSTINFER_HTTP_BODY" >&3
            else
                printf "%s %s HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n" \
                    "$RUSTINFER_HTTP_METHOD" "$RUSTINFER_HTTP_TARGET" >&3
            fi
            cat <&3
        ' >"$output"
}

http_status() { awk 'NR == 1 {print $2}' "$1"; }
http_body() { sed '1,/^\r$/d' "$1"; }

wait_ready() {
    local ready_raw=$1 ready_body=$2 ready_deadline
    ready_deadline=$((SECONDS + 180))
    while :; do
        if container_http GET /readyz '' "$ready_raw" 2>/dev/null \
            && [ "$(http_status "$ready_raw")" = 200 ]; then
            http_body "$ready_raw" >"$ready_body"
            if jq -e '.ready == true and .accepting == true' "$ready_body" >/dev/null; then return; fi
        fi
        docker inspect --format '{{.State.Running}}' "$container_id" | grep -qx true
        [ "$SECONDS" -lt "$ready_deadline" ] || { echo "readiness deadline exceeded" >&2; exit 1; }
        sleep 0.2
    done
}

ready_raw="$scratch/ready.raw"
ready_body="$scratch/ready.body"
wait_ready "$ready_raw" "$ready_body"

models_raw="$scratch/models.raw"
models_body="$scratch/models.body"
container_http GET /v1/models '' "$models_raw"
test "$(http_status "$models_raw")" = 200
http_body "$models_raw" >"$models_body"
models_json=$(jq -c '[.data[].id]' "$models_body")
test "$models_json" = "[\"$model_id\"]"

# The request ID participates in Philox stream derivation. Sampling is the
# first admitted request in each of two clean-start containers so both the
# explicit seed and the stable request stream are reproduced exactly.
sampling_request=$(jq -cn --arg model "$model_id" --arg prompt "$golden_prompt" \
    '{model:$model,prompt:$prompt,max_tokens:16,temperature:0.8,top_p:0.95,seed:424242,stream:false}')
request_sampling="$scratch/request-sampling.json"
printf '%s\n' "$sampling_request" >"$request_sampling"
sample_first_raw="$scratch/sample-first.raw"
sample_first_body="$scratch/sample-first.body"
sample_first_text="$scratch/sample-first.text"
container_http POST /v1/completions "$sampling_request" "$sample_first_raw"
sample_first_http_status=$(http_status "$sample_first_raw")
test "$sample_first_http_status" = 200
http_body "$sample_first_raw" >"$sample_first_body"
jq -ejr '.choices | select(length == 1) | .[0].text' "$sample_first_body" >"$sample_first_text"
test -s "$sample_first_text"
sample_first_sha256=$(sha_file "$sample_first_text")
sample_first_completion_tokens=$(jq -er '.usage.completion_tokens | select(type == "number" and floor == . and . >= 1)' "$sample_first_body")
sample_first_finish_reason=$(jq -er '.choices[0].finish_reason | select(. == "length" or . == "stop")' "$sample_first_body")

greedy_request=$(jq -cn --arg model "$model_id" --arg prompt "$golden_prompt" \
    --argjson max_tokens "$golden_max_tokens" \
    '{model:$model,prompt:$prompt,max_tokens:$max_tokens,temperature:0,top_p:1,stream:false}')
greedy_stream_request=$(jq -c '.stream=true' <<<"$greedy_request")
request_greedy="$scratch/request-greedy.json"
request_greedy_stream="$scratch/request-greedy-stream.json"
printf '%s\n' "$greedy_request" >"$request_greedy"
printf '%s\n' "$greedy_stream_request" >"$request_greedy_stream"
greedy_raw="$scratch/greedy.raw"
greedy_body="$scratch/greedy.body"
container_http POST /v1/completions "$greedy_request" "$greedy_raw"
test "$(http_status "$greedy_raw")" = 200
http_body "$greedy_raw" >"$greedy_body"
greedy_text_file="$scratch/greedy.text"
jq -ejr '.choices | select(length == 1) | .[0].text' "$greedy_body" >"$greedy_text_file"
greedy_text_sha256=$(sha_file "$greedy_text_file")
completion_tokens=$(jq -er '.usage.completion_tokens | select(type == "number" and floor == . and . >= 2)' "$greedy_body")
finish_reason=$(jq -er '.choices[0].finish_reason | select(. == "length" or . == "stop")' "$greedy_body")
test "$greedy_text_sha256" = "$approved_text_sha256"

stream_raw="$scratch/stream.raw"
stream_body="$scratch/stream.body"
stream_frames="$scratch/stream.frames.jsonl"
container_http POST /v1/completions "$greedy_stream_request" "$stream_raw"
test "$(http_status "$stream_raw")" = 200
http_body "$stream_raw" >"$stream_body"
test "$(grep -Fxc 'data: [DONE]' "$stream_body")" -eq 1
sed -n 's/^data: \({.*}\)$/\1/p' "$stream_body" >"$stream_frames"
stream_text_file="$scratch/stream.text"
jq -sjr '[.[] | .choices[0].text] | join("")' "$stream_frames" >"$stream_text_file"
stream_text_sha256=$(sha_file "$stream_text_file")
stream_token_events=$(jq -sr '[.[] | select(.choices[0].finish_reason == null)] | length' "$stream_frames")
test "$stream_text_sha256" = "$greedy_text_sha256"
test "$stream_token_events" -eq "$completion_tokens"

metrics_before_raw="$scratch/metrics-before.raw"
metrics_before_body="$scratch/metrics-before.body"
container_http GET /metrics '' "$metrics_before_raw"
test "$(http_status "$metrics_before_raw")" = 200
http_body "$metrics_before_raw" >"$metrics_before_body"
cancellations_before=$(jq -er '.counters.cancellations' "$metrics_before_body")
disconnects_before=$(jq -er '.counters.disconnects' "$metrics_before_body")
cancel_request=$(jq -cn --arg model "$model_id" --arg prompt "$golden_prompt" \
    --argjson max_tokens "$RUSTINFER_E2E_CANCEL_TOKENS" \
    '{model:$model,prompt:$prompt,max_tokens:$max_tokens,temperature:0,top_p:1,stream:true}')
cancellation_request="$scratch/cancellation-request.raw"
cancellation_response_prefix="$scratch/cancellation-response-prefix.raw"
cancel_length=$(printf '%s' "$cancel_request" | wc -c | awk '{print $1}')
printf 'POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: %s\r\nConnection: close\r\n\r\n%s' \
    "$cancel_length" "$cancel_request" >"$cancellation_request"
docker exec --interactive "$container_id" /bin/bash -c '
    set -euo pipefail
    export LC_ALL=C
    exec 3<>/dev/tcp/127.0.0.1/8080
    cat >&3
    dd bs=1 count=12 status=none <&3
' <"$cancellation_request" >"$cancellation_response_prefix"
test "$(sha_file "$cancellation_response_prefix")" = "$(printf 'HTTP/1.1 200' | sha256sum | awk '{print $1}')"

metrics_after_raw="$scratch/metrics-after.raw"
metrics_after_body="$scratch/metrics-after.body"
cancel_deadline=$((SECONDS + 120))
while :; do
    container_http GET /metrics '' "$metrics_after_raw"
    test "$(http_status "$metrics_after_raw")" = 200
    http_body "$metrics_after_raw" >"$metrics_after_body"
    if jq -e --argjson cb "$cancellations_before" --argjson db "$disconnects_before" \
        '.active_requests == 0 and .waiting_requests == 0 and .counters.cancellations > $cb and .counters.disconnects > $db' \
        "$metrics_after_body" >/dev/null; then break; fi
    [ "$SECONDS" -lt "$cancel_deadline" ] || { echo "disconnect cancellation deadline exceeded" >&2; exit 1; }
    sleep 0.2
done
cancellations_after=$(jq -er '.counters.cancellations' "$metrics_after_body")
disconnects_after=$(jq -er '.counters.disconnects' "$metrics_after_body")
active_after=$(jq -er '.active_requests' "$metrics_after_body")
waiting_after=$(jq -er '.waiting_requests' "$metrics_after_body")

image_binary="$scratch/image-binary"
image_native_dependencies="$scratch/image-native-dependencies.txt"
image_ldd="$scratch/image-ldd.txt"
image_readelf="$scratch/image-readelf.txt"
image_python_scan="$scratch/image-python-scan.txt"
docker cp "$container_id:/opt/rustinfer/bin/rustinfer" "$image_binary"
docker cp "$container_id:/opt/rustinfer/manifest/native-dependencies.txt" "$image_native_dependencies"
image_binary_sha256=$(sha_file "$image_binary")
test "$image_binary_sha256" = "$RUSTINFER_E2E_RELEASE_BINARY_SHA256"
readelf --file-header --program-headers --dynamic "$image_binary" >"$image_readelf"
docker exec "$container_id" ldd /opt/rustinfer/bin/rustinfer >"$image_ldd"
{
    printf '[forbidden-executables]\n'
    docker exec --user 0 "$container_id" /bin/bash -c '
    set -euo pipefail
    for command_name in python python3 pip pip3; do
        if command -v "$command_name" >/dev/null 2>&1; then printf "%s\n" "$command_name"; fi
    done
    find / -xdev -type f -perm /111 \( -iname "python*" -o -iname "pypy*" -o -iname "pip*" \) -print | sort -u
'
    printf '[forbidden-artifacts]\n'
    docker exec --user 0 "$container_id" /bin/bash -c '
    set -euo pipefail
    find / -xdev -type f \( \
        -name "*.py" -o -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \
        -o -name "*.whl" -o -name "*.pkl" -o -name "*.pickle" \
        -o -iname "*pytorch*" -o -iname "*torch*" -o -iname "*transformers*" -o -iname "*triton*" \
    \) -print | sort -u
'
} >"$image_python_scan"
test "$(sha_file "$image_python_scan")" = "$(printf '[forbidden-executables]\n[forbidden-artifacts]\n' | sha256sum | awk '{print $1}')"
forbidden_executables_json='[]'
forbidden_artifact_count=0
manifest_dependencies_json=$(awk -F= '/^dependency=/{print $2}' "$image_native_dependencies" | jq -Rsc 'split("\n") | map(select(length > 0))')
loader_output=$(<"$image_ldd")
loader_dependencies_json=$(printf '%s\n' "$loader_output" | jq -Rsc 'split("\n") | map(select(length > 0))')
unresolved_dependencies_json=$(printf '%s\n' "$loader_output" | grep -E '=>[[:space:]]+not found' || true)
unresolved_dependencies_json=$(printf '%s' "$unresolved_dependencies_json" | jq -Rsc 'split("\n") | map(select(length > 0))')
test "$unresolved_dependencies_json" = '[]'
forbidden_dependency_matches_json=$(printf '%s\n%s\n' "$manifest_dependencies_json" "$loader_output" \
    | grep -Ei 'python|pip|pytorch|torch|transformers|triton|pickle' || true)
forbidden_dependency_matches_json=$(printf '%s' "$forbidden_dependency_matches_json" | jq -Rsc 'split("\n") | map(select(length > 0))')
test "$forbidden_dependency_matches_json" = '[]'

container_first_runtime="$scratch/container-first-runtime.json"
process_first_runtime="$scratch/process-first-runtime.txt"
docker inspect "$container_id" >"$container_first_runtime"
docker top "$container_id" -eo pid,ppid,comm,args >"$process_first_runtime"
processes_json=$(tail -n +2 "$process_first_runtime" | jq -Rsc '
    split("\n") | map(select(length > 0) | capture("^\\s*(?<pid>[0-9]+)\\s+(?<ppid>[0-9]+)\\s+(?<comm>[^ ]+)\\s+(?<args>.+)$") |
    {pid:(.pid|tonumber),ppid:(.ppid|tonumber),comm:.comm,args:.args})')
jq -e 'length >= 1 and any(.comm == "rustinfer") and all((.comm + " " + .args) | test("python|pip|pytorch|torch|transformers|triton|pickle"; "i") | not)' \
    <<<"$processes_json" >/dev/null

docker kill --signal TERM "$container_id" >/dev/null
container_exit_code=$(docker wait "$container_id")
test "$container_exit_code" -eq 0
container_first_post="$scratch/container-first-post.json"
docker inspect "$container_id" >"$container_first_post"
test -f "$shutdown_metrics" && test ! -L "$shutdown_metrics"
jq -e '
    .active_requests == 0 and .waiting_requests == 0 and .kv_allocated_blocks == 0 and
    .allocation.device_live_count == 0 and .allocation.device_live_bytes == 0 and
    .allocation.pinned_live_count == 0 and .allocation.pinned_live_bytes == 0
' "$shutdown_metrics" >/dev/null
shutdown_metrics_json=$(jq -c . "$shutdown_metrics")
shutdown_metrics_sha256=$(sha_file "$shutdown_metrics")
first_container_id=$container_id

container_id=$(launch_container repeat-shutdown-metrics.json)
container_ids+=("$container_id")
[[ $container_id =~ ^[0-9a-f]{64}$ ]]
test "$container_id" != "$first_container_id"
test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "$container_id")" = none
container_second_pre="$scratch/container-second-pre.json"
process_second_pre="$scratch/process-second-pre.txt"
docker inspect "$container_id" >"$container_second_pre"
docker top "$container_id" -eo pid,ppid,comm,args >"$process_second_pre"
repeat_ready_raw="$scratch/repeat-ready.raw"
repeat_ready_body="$scratch/repeat-ready.body"
wait_ready "$repeat_ready_raw" "$repeat_ready_body"
sample_second_raw="$scratch/sample-second.raw"
sample_second_body="$scratch/sample-second.body"
sample_second_text="$scratch/sample-second.text"
container_http POST /v1/completions "$sampling_request" "$sample_second_raw"
sample_second_http_status=$(http_status "$sample_second_raw")
test "$sample_second_http_status" = 200
http_body "$sample_second_raw" >"$sample_second_body"
jq -ejr '.choices | select(length == 1) | .[0].text' "$sample_second_body" >"$sample_second_text"
test -s "$sample_second_text"
sample_second_sha256=$(sha_file "$sample_second_text")
sample_second_completion_tokens=$(jq -er '.usage.completion_tokens | select(type == "number" and floor == . and . >= 1)' "$sample_second_body")
sample_second_finish_reason=$(jq -er '.choices[0].finish_reason | select(. == "length" or . == "stop")' "$sample_second_body")
test "$sample_first_sha256" = "$sample_second_sha256"
test "$sample_first_completion_tokens" -eq "$sample_second_completion_tokens"
test "$sample_first_finish_reason" = "$sample_second_finish_reason"
container_second_runtime="$scratch/container-second-runtime.json"
process_second_runtime="$scratch/process-second-runtime.txt"
docker inspect "$container_id" >"$container_second_runtime"
docker top "$container_id" -eo pid,ppid,comm,args >"$process_second_runtime"
docker kill --signal TERM "$container_id" >/dev/null
repeat_container_exit_code=$(docker wait "$container_id")
test "$repeat_container_exit_code" -eq 0
container_second_post="$scratch/container-second-post.json"
docker inspect "$container_id" >"$container_second_post"
test -f "$repeat_shutdown_metrics" && test ! -L "$repeat_shutdown_metrics"
jq -e '
    .active_requests == 0 and .waiting_requests == 0 and .kv_allocated_blocks == 0 and
    .allocation.device_live_count == 0 and .allocation.device_live_bytes == 0 and
    .allocation.pinned_live_count == 0 and .allocation.pinned_live_bytes == 0
' "$repeat_shutdown_metrics" >/dev/null
repeat_shutdown_metrics_json=$(jq -c . "$repeat_shutdown_metrics")
repeat_shutdown_metrics_sha256=$(sha_file "$repeat_shutdown_metrics")
second_container_id=$container_id

raw_evidence="$RUSTINFER_E2E_OUTPUT/raw-evidence.json"
( set -o noclobber
  jq -nS \
    --arg run_id "python-free-e2e-$(date -u +%Y%m%dT%H%M%SZ)-${RUSTINFER_E2E_SOURCE_REVISION:0:12}" \
    --arg recorded_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg git_revision "$RUSTINFER_E2E_SOURCE_REVISION" \
    --arg source_archive_sha256 "$RUSTINFER_E2E_SOURCE_ARCHIVE_SHA256" \
    --arg binary_sha256 "$RUSTINFER_E2E_RELEASE_BINARY_SHA256" \
    --arg bundle_sha256 "$RUSTINFER_E2E_RELEASE_BUNDLE_SHA256" \
    --arg image_sha256 "${RUSTINFER_E2E_IMAGE_ID#sha256:}" \
    --arg model_id "$model_id" \
    --arg model_revision "$RUSTINFER_E2E_MODEL_REVISION" \
    --arg model_tree_sha256 "$RUSTINFER_E2E_MODEL_TREE_SHA256" \
    --arg config_sha256 "$RUSTINFER_E2E_CONFIG_SHA256" \
    --arg weights_sha256 "$RUSTINFER_E2E_WEIGHTS_SHA256" \
    --arg tokenizer_aggregate_sha256 "$RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256" \
    --arg tokenizer_json_sha256 "$RUSTINFER_E2E_TOKENIZER_JSON_SHA256" \
    --arg correctness_gate_id smollm2-fp32-bf16-native-e0-v3 \
    --arg correctness_report_sha256 "$RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256" \
    --arg correctness_golden_sha256 "$RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256" \
    --arg first_container_id "$first_container_id" --arg second_container_id "$second_container_id" \
    --arg image_id "$RUSTINFER_E2E_IMAGE_ID" \
    --arg image_binary_sha256 "$image_binary_sha256" \
    --arg model_ids "$models_json" \
    --arg prompt_sha256 "$prompt_sha256" --argjson max_tokens "$golden_max_tokens" \
    --arg greedy_text_sha256 "$greedy_text_sha256" --arg stream_text_sha256 "$stream_text_sha256" \
    --arg approved_text_sha256 "$approved_text_sha256" \
    --argjson completion_tokens "$completion_tokens" --argjson stream_token_events "$stream_token_events" \
    --arg finish_reason "$finish_reason" \
    --arg sample_first "$sample_first_sha256" --arg sample_second "$sample_second_sha256" \
    --argjson sample_first_http_status "$sample_first_http_status" \
    --argjson sample_second_http_status "$sample_second_http_status" \
    --argjson sample_first_completion_tokens "$sample_first_completion_tokens" \
    --argjson sample_second_completion_tokens "$sample_second_completion_tokens" \
    --arg sample_first_finish_reason "$sample_first_finish_reason" \
    --arg sample_second_finish_reason "$sample_second_finish_reason" \
    --argjson cancellations_before "$cancellations_before" --argjson cancellations_after "$cancellations_after" \
    --argjson disconnects_before "$disconnects_before" --argjson disconnects_after "$disconnects_after" \
    --argjson active_after "$active_after" --argjson waiting_after "$waiting_after" \
    --arg forbidden_executables "$forbidden_executables_json" \
    --argjson forbidden_artifact_count "$forbidden_artifact_count" \
    --arg manifest_dependencies "$manifest_dependencies_json" \
    --arg loader_dependencies "$loader_dependencies_json" \
    --arg unresolved_dependencies "$unresolved_dependencies_json" \
    --arg forbidden_dependency_matches "$forbidden_dependency_matches_json" \
    --arg processes "$processes_json" --arg shutdown_metrics "$shutdown_metrics_json" \
    --arg repeat_shutdown_metrics "$repeat_shutdown_metrics_json" \
    --arg shutdown_metrics_sha256 "$shutdown_metrics_sha256" \
    --arg repeat_shutdown_metrics_sha256 "$repeat_shutdown_metrics_sha256" \
    '{
      schema_version:"rustinfer.python-free-release-e2e-raw.v2",run_id:$run_id,
      recorded_at_utc:$recorded_at_utc,status:"success",
      source:{git_revision:$git_revision,git_dirty:false,source_archive_sha256:$source_archive_sha256},
      release:{binary_sha256:$binary_sha256,bundle_sha256:$bundle_sha256,image_sha256:$image_sha256},
      model:{model_id:$model_id,model_revision:$model_revision,model_tree_sha256:$model_tree_sha256,config_sha256:$config_sha256,weights_sha256:$weights_sha256,tokenizer_aggregate_sha256:$tokenizer_aggregate_sha256,tokenizer_json_sha256:$tokenizer_json_sha256,correctness_gate_id:$correctness_gate_id,correctness_report_sha256:$correctness_report_sha256,correctness_golden_sha256:$correctness_golden_sha256},
      runtime:{container_ids:[$first_container_id,$second_container_id],network_mode:"none",image_id:$image_id,image_binary_sha256:$image_binary_sha256},
      observations:{
        readyz:{http_status:200,ready:true,accepting:true},models:{http_status:200,model_ids:($model_ids|fromjson)},
        greedy:{non_stream_http_status:200,stream_http_status:200,non_stream_text_sha256:$greedy_text_sha256,stream_text_sha256:$stream_text_sha256,approved_text_sha256:$approved_text_sha256,completion_tokens:$completion_tokens,stream_token_events:$stream_token_events,finish_reason:$finish_reason,stream_done:true,prompt_sha256:$prompt_sha256,max_tokens:$max_tokens},
        sampling:{seed:424242,temperature:0.8,top_p:0.95,first_http_status:$sample_first_http_status,second_http_status:$sample_second_http_status,first_completion_tokens:$sample_first_completion_tokens,second_completion_tokens:$sample_second_completion_tokens,first_finish_reason:$sample_first_finish_reason,second_finish_reason:$sample_second_finish_reason,first_text_sha256:$sample_first,second_text_sha256:$sample_second},
        cancellation:{disconnect_probe_sent:true,cancellations_before:$cancellations_before,cancellations_after:$cancellations_after,disconnects_before:$disconnects_before,disconnects_after:$disconnects_after,active_requests_after:$active_after,waiting_requests_after:$waiting_after},
        shutdown:{signal:"SIGTERM",exit_code:0,metrics:($shutdown_metrics|fromjson),metrics_sha256:$shutdown_metrics_sha256,repeat_exit_code:0,repeat_metrics:($repeat_shutdown_metrics|fromjson),repeat_metrics_sha256:$repeat_shutdown_metrics_sha256},
        python_free:{forbidden_executables:($forbidden_executables|fromjson),forbidden_artifact_count:$forbidden_artifact_count,processes:($processes|fromjson),manifest_dependencies:($manifest_dependencies|fromjson),loader_dependencies:($loader_dependencies|fromjson),unresolved_dependencies:($unresolved_dependencies|fromjson),forbidden_dependency_matches:($forbidden_dependency_matches|fromjson)}
      }
    }' >"$raw_evidence"
)

raw_archive="$RUSTINFER_E2E_OUTPUT/python-free-evidence.tar"
python3 "$packager" \
    --output "$raw_archive" \
    --model-dir "$RUSTINFER_E2E_MODEL_DIR" \
    --raw-evidence "$raw_evidence" \
    --correctness-golden "$RUSTINFER_E2E_CORRECTNESS_GOLDEN" \
    --model-manifest "$model_manifest" \
    --shutdown-metrics "$shutdown_metrics" \
    --repeat-shutdown-metrics "$repeat_shutdown_metrics" \
    --image-inspect "$image_inspect" \
    --image-binary "$image_binary" \
    --image-native-dependencies "$image_native_dependencies" \
    --image-ldd "$image_ldd" \
    --image-readelf "$image_readelf" \
    --image-python-scan "$image_python_scan" \
    --container-first-pre "$container_first_pre" \
    --container-first-runtime "$container_first_runtime" \
    --container-first-post "$container_first_post" \
    --container-second-pre "$container_second_pre" \
    --container-second-runtime "$container_second_runtime" \
    --container-second-post "$container_second_post" \
    --process-first-pre "$process_first_pre" \
    --process-first-runtime "$process_first_runtime" \
    --process-second-pre "$process_second_pre" \
    --process-second-runtime "$process_second_runtime" \
    --request-greedy "$request_greedy" \
    --request-greedy-stream "$request_greedy_stream" \
    --request-sampling "$request_sampling" \
    --http-readyz "$ready_raw" \
    --http-models "$models_raw" \
    --http-greedy "$greedy_raw" \
    --http-greedy-stream "$stream_raw" \
    --http-sampling-first "$sample_first_raw" \
    --http-sampling-second "$sample_second_raw" \
    --http-metrics-before "$metrics_before_raw" \
    --http-metrics-after "$metrics_after_raw" \
    --cancellation-request "$cancellation_request" \
    --cancellation-response-prefix "$cancellation_response_prefix"

python3 "$checker" \
    --evidence "$raw_evidence" \
    --raw-archive "$raw_archive" \
    --source-revision "$RUSTINFER_E2E_SOURCE_REVISION" \
    --source-archive "$RUSTINFER_E2E_SOURCE_ARCHIVE" \
    --release-binary "$RUSTINFER_E2E_RELEASE_BINARY" \
    --release-bundle "$RUSTINFER_E2E_RELEASE_BUNDLE" \
    --image-id "$RUSTINFER_E2E_IMAGE_ID" \
    --model-dir "$RUSTINFER_E2E_MODEL_DIR" \
    --model-tree-sha256 "$RUSTINFER_E2E_MODEL_TREE_SHA256" \
    --weights "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_WEIGHTS_RELATIVE_PATH" \
    --weights-sha256 "$RUSTINFER_E2E_WEIGHTS_SHA256" \
    --tokenizer "$RUSTINFER_E2E_MODEL_DIR/$RUSTINFER_E2E_TOKENIZER_RELATIVE_PATH" \
    --tokenizer-json-sha256 "$RUSTINFER_E2E_TOKENIZER_JSON_SHA256" \
    --tokenizer-aggregate-sha256 "$RUSTINFER_E2E_TOKENIZER_AGGREGATE_SHA256" \
    --correctness-golden "$RUSTINFER_E2E_CORRECTNESS_GOLDEN" \
    --correctness-golden-sha256 "$RUSTINFER_E2E_CORRECTNESS_GOLDEN_SHA256" \
    --correctness-report "$RUSTINFER_E2E_CORRECTNESS_REPORT" \
    --correctness-report-sha256 "$RUSTINFER_E2E_CORRECTNESS_REPORT_SHA256" \
    --shutdown-metrics "$shutdown_metrics" \
    --repeat-shutdown-metrics "$repeat_shutdown_metrics" \
    --report "$RUSTINFER_E2E_OUTPUT/attestation.json"

( set -o noclobber
  cd "$RUSTINFER_E2E_OUTPUT"
  sha256sum attestation.json python-free-evidence.tar >SHA256SUMS
)
echo "Python-free real-model release E2E gate passed: $RUSTINFER_E2E_OUTPUT"
