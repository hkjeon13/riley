#!/usr/bin/env bash
# Run the PR-16 soak against the production rustinfer CLI.  This driver uses
# host tools only; it never installs Python or a reference runtime in the
# production image.  Evidence files are create-only or append-only.
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
: "${RUSTINFER_SOAK_MANIFEST:=$repo_root/benchmarks/soak/reliability-soak-v1.json}"
: "${RUSTINFER_SOAK_OUTPUT:?set a new absolute evidence directory}"
: "${RUSTINFER_SOURCE_REVISION:?set the full clean source revision}"
: "${RUSTINFER_SOURCE_ARCHIVE_SHA256:?set the git archive SHA-256}"
: "${RUSTINFER_BINARY_SHA256:?set the production binary SHA-256}"
: "${RUSTINFER_IMAGE_SHA256:?set the immutable release image SHA-256}"
: "${RUSTINFER_MODEL_SHA256:?set the immutable model artifact SHA-256}"
: "${RUSTINFER_MODEL_ID:?set the model identifier}"
: "${RUSTINFER_MODEL_REVISION:?set the immutable model revision}"
: "${RUSTINFER_SOAK_FINAL_METRICS_JSON:?set the shutdown metrics artifact path}"

for tool in bash jq curl sha256sum awk ps flock nvidia-smi readlink find wc date env; do
    command -v "$tool" >/dev/null 2>&1 || { echo "required host tool is unavailable: $tool" >&2; exit 2; }
done
case "$RUSTINFER_SOAK_OUTPUT" in /*) ;; *) echo "RUSTINFER_SOAK_OUTPUT must be absolute" >&2; exit 2 ;; esac
case "$RUSTINFER_SOAK_FINAL_METRICS_JSON" in /*) ;; *) echo "RUSTINFER_SOAK_FINAL_METRICS_JSON must be absolute" >&2; exit 2 ;; esac
test ! -e "$RUSTINFER_SOAK_OUTPUT"
test ! -e "$RUSTINFER_SOAK_FINAL_METRICS_JSON"
mkdir -m 0700 "$RUSTINFER_SOAK_OUTPUT"
events="$RUSTINFER_SOAK_OUTPUT/events.jsonl"
sequence_file="$RUSTINFER_SOAK_OUTPUT/.sequence"
monotonic_file="$RUSTINFER_SOAK_OUTPUT/.monotonic"
lock_file="$RUSTINFER_SOAK_OUTPUT/.append.lock"
run_file="$RUSTINFER_SOAK_OUTPUT/run.json"
: >"$events"
printf '0\n' >"$sequence_file"
printf '0\n' >"$monotonic_file"
: >"$lock_file"

sha_re='^[0-9a-f]{64}$'
git_re='^[0-9a-f]{40}([0-9a-f]{24})?$'
[[ $RUSTINFER_SOURCE_REVISION =~ $git_re ]] || { echo "invalid source revision" >&2; exit 2; }
for value in "$RUSTINFER_SOURCE_ARCHIVE_SHA256" "$RUSTINFER_BINARY_SHA256" "$RUSTINFER_IMAGE_SHA256" "$RUSTINFER_MODEL_SHA256"; do
    [[ $value =~ $sha_re ]] || { echo "invalid SHA-256 binding" >&2; exit 2; }
done
jq -e '.schema_version == "rustinfer.reliability-soak-manifest.v1"' "$RUSTINFER_SOAK_MANIFEST" >/dev/null
golden_generated_sha256=$(jq -er '.golden.generated_sha256 | select(test("^[0-9a-f]{64}$") and . != ("0" * 64))' "$RUSTINFER_SOAK_MANIFEST")
golden_provenance_sha256=$(jq -er '.golden.provenance_sha256 | select(test("^[0-9a-f]{64}$") and . != ("0" * 64))' "$RUSTINFER_SOAK_MANIFEST")
test -n "$golden_generated_sha256" && test -n "$golden_provenance_sha256"

binary=${RUSTINFER_SOAK_BINARY:-$(jq -er '.target.binary' "$RUSTINFER_SOAK_MANIFEST")}
model_path=${RUSTINFER_SOAK_MODEL_PATH:-$(jq -er '.target.model_path' "$RUSTINFER_SOAK_MANIFEST")}
bind=${RUSTINFER_SOAK_BIND:-$(jq -er '.target.bind' "$RUSTINFER_SOAK_MANIFEST")}
target_kind=$(jq -er '.target.kind' "$RUSTINFER_SOAK_MANIFEST")
test -x "$binary"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$RUSTINFER_BINARY_SHA256"
test -e "$model_path"
manifest_sha=$(sha256sum "$RUSTINFER_SOAK_MANIFEST" | awk '{print $1}')
source_json=$(jq -cnS \
    --arg git_commit "$RUSTINFER_SOURCE_REVISION" \
    --arg source_archive_sha256 "$RUSTINFER_SOURCE_ARCHIVE_SHA256" \
    --arg binary_sha256 "$RUSTINFER_BINARY_SHA256" \
    --arg image_sha256 "$RUSTINFER_IMAGE_SHA256" \
    --arg model_sha256 "$RUSTINFER_MODEL_SHA256" \
    --arg model_id "$RUSTINFER_MODEL_ID" \
    --arg model_revision "$RUSTINFER_MODEL_REVISION" \
    '{git_commit:$git_commit,git_dirty:false,source_archive_sha256:$source_archive_sha256,binary_sha256:$binary_sha256,image_sha256:$image_sha256,model_sha256:$model_sha256,model_id:$model_id,model_revision:$model_revision}')
binding_sha=$(printf '%s' "$source_json" | sha256sum | awk '{print $1}')
base_url="http://$bind"
health_path=$(jq -er '.target.health_path' "$RUSTINFER_SOAK_MANIFEST")
completion_path=$(jq -er '.target.completion_path' "$RUSTINFER_SOAK_MANIFEST")
metrics_path=$(jq -er '.target.metrics_path' "$RUSTINFER_SOAK_MANIFEST")
sample_interval_ms=$(jq -er '.thresholds.sample_interval_ms' "$RUSTINFER_SOAK_MANIFEST")
shutdown_deadline_ms=$(jq -er '.thresholds.graceful_shutdown_deadline_ms' "$RUSTINFER_SOAK_MANIFEST")
run_id="soak-$(date -u +%Y%m%dT%H%M%SZ)-${RUSTINFER_SOURCE_REVISION:0:12}"
target_pid=0
sampler_pid=
emit_shutdown_metrics=0

cleanup() {
    local attempt
    if [ -n "${sampler_pid:-}" ]; then kill "$sampler_pid" 2>/dev/null || true; fi
    if [ "${target_pid:-0}" -gt 0 ] && kill -0 "$target_pid" 2>/dev/null; then
        kill -TERM "$target_pid" 2>/dev/null || true
        for attempt in {1..50}; do
            kill -0 "$target_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$target_pid" 2>/dev/null; then kill -KILL "$target_pid" 2>/dev/null || true; fi
        wait "$target_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

monotonic_ns() {
    awk '{printf "%.0f\n", $1 * 1000000000}' /proc/uptime
}

append_event() {
    local payload=$1 sequence now previous
    exec 8>>"$lock_file"
    flock 8
    sequence=$(<"$sequence_file")
    sequence=$((sequence + 1))
    printf '%s\n' "$sequence" >"$sequence_file"
    now=$(monotonic_ns)
    previous=$(<"$monotonic_file")
    if (( now <= previous )); then now=$((previous + 1)); fi
    printf '%s\n' "$now" >"$monotonic_file"
    jq -cn \
        --argjson payload "$payload" \
        --argjson sequence "$sequence" \
        --argjson monotonic_ns "$now" \
        --arg binding_sha256 "$binding_sha" \
        '$payload + {schema_version:"rustinfer.reliability-soak-event.v1",sequence:$sequence,monotonic_ns:$monotonic_ns,binding_sha256:$binding_sha256}' >>"$events"
    flock -u 8
    exec 8>&-
}

launch_target() {
    local mode=$1 argument replaced command_sha ready_deadline
    local -a arguments=()
    while IFS= read -r argument; do
        replaced=${argument//\{model_path\}/$model_path}
        replaced=${replaced//\{bind\}/$bind}
        replaced=${replaced//\{execution_completion\}/$mode}
        arguments+=("$replaced")
    done < <(jq -er '.target.launch_arguments[]' "$RUSTINFER_SOAK_MANIFEST")
    command_sha=$( { printf '%s\0' "$binary"; printf '%s\0' "${arguments[@]}"; } | sha256sum | awk '{print $1}')
    if [ "$emit_shutdown_metrics" -eq 1 ]; then
        RUSTINFER_SHUTDOWN_METRICS_PATH="$RUSTINFER_SOAK_FINAL_METRICS_JSON" \
            "$binary" "${arguments[@]}" </dev/null >>"$RUSTINFER_SOAK_OUTPUT/server.stdout.log" 2>>"$RUSTINFER_SOAK_OUTPUT/server.stderr.log" &
    else
        env -u RUSTINFER_SHUTDOWN_METRICS_PATH \
            "$binary" "${arguments[@]}" </dev/null >>"$RUSTINFER_SOAK_OUTPUT/server.stdout.log" 2>>"$RUSTINFER_SOAK_OUTPUT/server.stderr.log" &
    fi
    target_pid=$!
    if [ ! -e "$run_file" ]; then
        jq -nS \
            --arg run_id "$run_id" --arg manifest_sha256 "$manifest_sha" \
            --arg binding_sha256 "$binding_sha" --argjson source "$source_json" \
            --arg kind "$target_kind" --argjson pid "$target_pid" \
            --arg image_id "sha256:$RUSTINFER_IMAGE_SHA256" --arg command_sha256 "$command_sha" \
            --arg started_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            '{schema_version:"rustinfer.reliability-soak-run.v1",run_id:$run_id,manifest_sha256:$manifest_sha256,binding_sha256:$binding_sha256,source:$source,target:{kind:$kind,pid:$pid,image_id:$image_id,command_sha256:$command_sha256},started_at_utc:$started_at_utc}' >"$run_file"
    fi
    ready_deadline=$(( $(monotonic_ns) + 120000000000 ))
    until curl --fail --silent --show-error --max-time 2 "$base_url$health_path" >/dev/null; do
        kill -0 "$target_pid" 2>/dev/null || { echo "production server exited during startup" >&2; return 1; }
        (( $(monotonic_ns) < ready_deadline )) || { echo "readiness deadline exceeded" >&2; return 1; }
        sleep 0.2
    done
}

stop_target() {
    local start deadline exit_code=0 state
    start=$(monotonic_ns)
    kill -TERM "$target_pid"
    deadline=$((start + shutdown_deadline_ms * 1000000))
    while kill -0 "$target_pid" 2>/dev/null; do
        state=$(ps -o stat= -p "$target_pid" 2>/dev/null | awk '{$1=$1;print}')
        [[ $state == Z* ]] && break
        if (( $(monotonic_ns) >= deadline )); then
            kill -TERM "$target_pid" 2>/dev/null || true
            return 1
        fi
        sleep 0.1
    done
    wait "$target_pid" || exit_code=$?
    target_pid=0
    [ "$exit_code" -eq 0 ]
}

descendants_json() {
    local frontier="$target_pid" next pid child rows='[]'
    while [ -n "$frontier" ]; do
        next=
        for pid in $frontier; do
            while read -r child; do
                [ -n "$child" ] || continue
                comm=$(ps -o comm= -p "$child" 2>/dev/null | awk '{$1=$1;print}' || true)
                executable=$(readlink -f "/proc/$child/exe" 2>/dev/null || printf 'unavailable')
                rows=$(jq -cn --argjson rows "$rows" --argjson pid "$child" --arg comm "${comm:-unavailable}" --arg executable "$executable" '$rows + [{pid:$pid,comm:$comm,executable:$executable}]')
                next="$next $child"
            done < <(ps -o pid= --ppid "$pid" 2>/dev/null | awk '{$1=$1;print}')
        done
        frontier=$next
    done
    printf '%s\n' "$rows"
}

sample_once() {
    local scenario_id=$1 status children metrics vram pids child_pid gpu_rows matched=0
    if [ ! -r "/proc/$target_pid/status" ]; then
        append_event "$(jq -cn --arg scenario_id "$scenario_id" '{kind:"failure",scenario_id:$scenario_id,stage:"sample",message:"target /proc status disappeared"}')"
        return 1
    fi
    status=$(<"/proc/$target_pid/status")
    rss_kib=$(awk '/^VmRSS:/ {print $2}' <<<"$status")
    hwm_kib=$(awk '/^VmHWM:/ {print $2}' <<<"$status")
    threads=$(awk '/^Threads:/ {print $2}' <<<"$status")
    fd_count=$(find "/proc/$target_pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | awk '{$1=$1;print}')
    children=$(descendants_json)
    pids=" $target_pid "
    while read -r child_pid; do pids="$pids$child_pid "; done < <(jq -r '.[].pid' <<<"$children")
    vram=0
    gpu_rows=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null) || {
        append_event "$(jq -cn --arg scenario_id "$scenario_id" '{kind:"failure",scenario_id:$scenario_id,stage:"nvidia-smi",message:"per-process VRAM query failed"}')"
        return 1
    }
    while IFS=, read -r gpu_pid used_mib; do
        gpu_pid=$(awk '{$1=$1;print}' <<<"$gpu_pid")
        used_mib=$(awk '{$1=$1;print}' <<<"$used_mib")
        if [[ $pids == *" $gpu_pid "* ]]; then
            if [[ ! $used_mib =~ ^[0-9]+$ ]]; then
                append_event "$(jq -cn --arg scenario_id "$scenario_id" '{kind:"failure",scenario_id:$scenario_id,stage:"nvidia-smi",message:"target VRAM value is not numeric"}')"
                return 1
            fi
            matched=1
            vram=$((vram + used_mib * 1024 * 1024))
        fi
    done <<<"$gpu_rows"
    if [ "$matched" -ne 1 ]; then
        append_event "$(jq -cn --arg scenario_id "$scenario_id" '{kind:"failure",scenario_id:$scenario_id,stage:"nvidia-smi",message:"target process is absent from the compute-app inventory"}')"
        return 1
    fi
    metrics=$(curl --fail --silent --show-error --max-time 2 "$base_url$metrics_path") || {
        append_event "$(jq -cn --arg scenario_id "$scenario_id" '{kind:"failure",scenario_id:$scenario_id,stage:"metrics",message:"metrics snapshot unavailable"}')"
        return 1
    }
    jq -e 'type == "object" and (.active_requests|type)=="number" and (.waiting_requests|type)=="number" and (.kv_allocated_blocks|type)=="number" and (.allocation|type)=="object" and (.counters|type)=="object"' <<<"$metrics" >/dev/null
    append_event "$(jq -cn --arg scenario_id "$scenario_id" --argjson pid "$target_pid" --argjson rss_bytes "$((rss_kib * 1024))" --argjson hwm_bytes "$((hwm_kib * 1024))" --argjson fd_count "$fd_count" --argjson thread_count "$threads" --argjson children "$children" --argjson vram_bytes "$vram" --argjson metrics "$metrics" '{kind:"sample",scenario_id:$scenario_id,process:{pid:$pid,rss_bytes:$rss_bytes,hwm_bytes:$hwm_bytes,fd_count:$fd_count,thread_count:$thread_count,children:$children},gpu:{vram_bytes:$vram_bytes},metrics:$metrics,sample_dropped:false}')"
}

sampler_loop() {
    local scenario_id=$1
    while kill -0 "$target_pid" 2>/dev/null; do
        sample_once "$scenario_id" || return 1
        sleep "$(awk -v ms="$sample_interval_ms" 'BEGIN {printf "%.3f", ms / 1000}')"
    done
}

run_request() {
    local scenario_id=$1 profile=$2 action=${3:-normal} request_id body output start end curl_code http_status latency outcome generated
    request_id="$scenario_id-$BASHPID-$RANDOM-$(monotonic_ns)"
    body=$(jq -c --arg profile "$profile" '.requests[$profile] | if has("prompt_repeat") then .prompt = (.prompt * .prompt_repeat) | del(.prompt_repeat) else . end' "$RUSTINFER_SOAK_MANIFEST")
    output="$RUSTINFER_SOAK_OUTPUT/request-$request_id.body"
    start=$(monotonic_ns)
    curl_code=0
    if [ "$action" = normal ]; then
        http_status=$(curl --silent --show-error --max-time 300 -o "$output" -w '%{http_code}' -H 'content-type: application/json' --data-binary "$body" "$base_url$completion_path") || curl_code=$?
    else
        http_status=$(curl --silent --show-error --max-time 0.05 -o "$output" -w '%{http_code}' -H 'content-type: application/json' --data-binary "$body" "$base_url$completion_path") || curl_code=$?
    fi
    end=$(monotonic_ns)
    latency=$(awk -v ns="$((end-start))" 'BEGIN {printf "%.6f", ns / 1000000}')
    http_status=$((10#${http_status:-0}))
    generated=null
    if [ "$action" = cancel ] && [ "$curl_code" -ne 0 ]; then outcome=cancelled
    elif [ "$action" = disconnect ] && [ "$curl_code" -ne 0 ]; then outcome=disconnected
    elif [ "$curl_code" -ne 0 ]; then outcome=failure
    elif [ "$http_status" = 429 ]; then outcome=overload
    elif [ "$http_status" -ge 400 ] && [ "$http_status" -lt 500 ]; then outcome=invalid
    elif [ "$http_status" -ge 200 ] && [ "$http_status" -lt 300 ]; then
        outcome=success
        generated=$(jq -jr '[.choices[].text] | join("")' "$output" | sha256sum | awk '{print $1}')
    else outcome=failure
    fi
    rm -f "$output"
    append_event "$(jq -cn --arg scenario_id "$scenario_id" --arg request_id "$request_id" --arg outcome "$outcome" --argjson http_status "${http_status:-0}" --argjson latency_ms "$latency" --argjson generated_sha256 "$(if [ "$generated" = null ]; then printf null; else jq -cn --arg value "$generated" '$value'; fi)" '{kind:"request",scenario_id:$scenario_id,request_id:$request_id,outcome:$outcome,http_status:$http_status,latency_ms:$latency_ms,generated_sha256:$generated_sha256}')"
}

probe_hash() {
    local profile=$1 body output
    body=$(jq -c --arg profile "$profile" '.requests[$profile] | if has("prompt_repeat") then .prompt = (.prompt * .prompt_repeat) | del(.prompt_repeat) else . end' "$RUSTINFER_SOAK_MANIFEST")
    output=$(curl --fail --silent --show-error --max-time 300 -H 'content-type: application/json' --data-binary "$body" "$base_url$completion_path")
    jq -jr '[.choices[].text] | join("")' <<<"$output" | sha256sum | awk '{print $1}'
}

run_scenario() {
    local encoded=$1 id kind duration concurrency cycle_interval_ms primary secondary mode deadline iteration=0 before after restart_start restart_elapsed worker profile action worker_pid
    local -a worker_pids=()
    id=$(jq -r '.id' <<<"$encoded"); kind=$(jq -r '.kind' <<<"$encoded")
    duration=$(jq -r '.duration_seconds' <<<"$encoded"); concurrency=$(jq -r '.concurrency' <<<"$encoded")
    cycle_interval_ms=$(jq -r '.cycle_interval_ms' <<<"$encoded")
    primary=$(jq -r '.request_profile' <<<"$encoded"); secondary=$(jq -r '.secondary_request_profile // .request_profile' <<<"$encoded")
    mode=$(jq -r '.execution_completion' <<<"$encoded")
    launch_target "$mode"
    append_event "$(jq -cn --arg id "$id" --arg mode "$mode" '{kind:"scenario_start",scenario_id:$id,execution_completion:$mode}')"
    sampler_loop "$id" & sampler_pid=$!
    deadline=$(( $(monotonic_ns) + duration * 1000000000 ))
    while (( $(monotonic_ns) < deadline )); do
        iteration=$((iteration + 1))
        for ((worker=0; worker<concurrency; worker++)); do
            profile=$primary; action=normal
            [ "$kind" = mixed ] && (( worker % 2 == 1 )) && profile=$secondary
            if [ "$kind" = cancellation-disconnect ]; then
                if (( worker % 2 == 0 )); then action=cancel; else action=disconnect; fi
            fi
            run_request "$id" "$profile" "$action" &
            worker_pids+=("$!")
        done
        for worker_pid in "${worker_pids[@]}"; do wait "$worker_pid"; done
        worker_pids=()
        if [ "$cycle_interval_ms" -gt 0 ]; then
            sleep "$(awk -v ms="$cycle_interval_ms" 'BEGIN {printf "%.3f", ms / 1000}')"
        fi
    done
    if [ "$kind" = graceful-restart ]; then
        before=$(probe_hash "$primary")
        restart_start=$(monotonic_ns)
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        stop_target
        launch_target "$mode"
        after=$(probe_hash "$primary")
        restart_elapsed=$(( ($(monotonic_ns) - restart_start) / 1000000 ))
        append_event "$(jq -cn --arg id "$id" --argjson elapsed_ms "$restart_elapsed" --arg before "$before" --arg after "$after" '{kind:"restart",scenario_id:$id,graceful:true,exit_code:0,elapsed_ms:$elapsed_ms,before_generated_sha256:$before,after_generated_sha256:$after}')"
        sampler_loop "$id" & sampler_pid=$!
    fi
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    sample_once "$id"
    append_event "$(jq -cn --arg id "$id" '{kind:"scenario_end",scenario_id:$id,status:"success"}')"
    stop_target
}

append_event '{"kind":"run_start","scenario_id":null}'
scenario_count=$(jq -er '.scenarios | length' "$RUSTINFER_SOAK_MANIFEST")
scenario_index=0
while IFS= read -r scenario; do
    scenario_index=$((scenario_index + 1))
    if [ "$scenario_index" -eq "$scenario_count" ]; then emit_shutdown_metrics=1; fi
    run_scenario "$scenario"
done < <(jq -c '.scenarios[]' "$RUSTINFER_SOAK_MANIFEST")

# The server writes this post-shutdown snapshot only after native allocation
# counters have observed all close operations.  Synthesizing zeros is forbidden.
jq -e '.active_requests == 0 and .waiting_requests == 0 and .kv_allocated_blocks == 0 and ([.allocation[]] | all(. == 0))' "$RUSTINFER_SOAK_FINAL_METRICS_JSON" >/dev/null
final_metrics=$(jq -c '.' "$RUSTINFER_SOAK_FINAL_METRICS_JSON")
append_event "$(jq -cn --argjson metrics "$final_metrics" '{kind:"sample",scenario_id:null,process:{pid:0,rss_bytes:0,hwm_bytes:0,fd_count:0,thread_count:0,children:[]},gpu:{vram_bytes:0},metrics:$metrics,sample_dropped:false}')"
append_event '{"kind":"run_end","scenario_id":null,"status":"success"}'
rm -f "$sequence_file" "$monotonic_file" "$lock_file"
trap - EXIT
echo "soak evidence complete: $RUSTINFER_SOAK_OUTPUT"
