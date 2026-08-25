#!/usr/bin/env bash
# Execute inside the network-isolated GPU build container on server-4096.

set -euo pipefail
umask 022

: "${RUSTINFER_SOURCE_REVISION:?missing source revision}"
: "${RUSTINFER_SOURCE_ARCHIVE_SHA256:?missing source archive digest}"
: "${RUSTINFER_BUILD_IMAGE_ID:?missing immutable build image ID}"
: "${RUSTINFER_MODEL_TREE_SHA256:?missing model tree digest}"

test "$(pwd -P)" = /workspace
test -d /evidence
test -d /runner-output
test -d /model
test -z "$(find /evidence -mindepth 1 -print -quit)"
test -z "$(find /runner-output -mindepth 1 -print -quit)"
test "$(uname -m)" = x86_64

for forbidden in python python3 pip pip3; do
    if command -v "${forbidden}" >/dev/null 2>&1; then
        echo "optimizer runner build image contains forbidden ${forbidden}" >&2
        exit 1
    fi
done

export CARGO_NET_OFFLINE=true
export CARGO_TERM_COLOR=never
export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export RUSTINFER_CUDA_ARCHITECTURES=89
export RUSTUP_TOOLCHAIN=1.85.0-x86_64-unknown-linux-gnu

command_records=/runner-output/commands.v2
subject_records=/runner-output/subjects.v2
printf '%s\n' 'rustinfer.optimizer-command-log.v2' >"${command_records}"
printf '%s\n' 'rustinfer.optimizer-subjects.v2' >"${subject_records}"

encode_record() {
    printf '%s' "$1" | base64 -w0
}

record_environment() {
    local model_environment=$1
    local key value
    for key in \
        CARGO_NET_OFFLINE \
        CARGO_TERM_COLOR \
        CUDA_HOME \
        CUDAToolkit_ROOT \
        RUSTINFER_CUDA_ARCHITECTURES \
        RUSTUP_TOOLCHAIN
    do
        value=${!key}
        printf 'ENV %s %s\n' "$(encode_record "${key}")" "$(encode_record "${value}")" \
            >>"${command_records}"
    done
    if [[ "${model_environment}" == model ]]; then
        printf 'ENV %s %s\n' \
            "$(encode_record RUSTINFER_REAL_CHECKPOINT)" \
            "$(encode_record /model)" >>"${command_records}"
    fi
}

run_recorded() {
    local command_id=$1
    local log_name=$2
    local subject=$3
    local model_environment=$4
    local status argument
    shift 4

    printf 'BEGIN %s\n' "$(encode_record "${command_id}")" >>"${command_records}"
    printf 'LOG %s\n' "$(encode_record "${log_name}")" >>"${command_records}"
    printf 'SUBJECT %s\n' "$(encode_record "${subject}")" >>"${command_records}"
    record_environment "${model_environment}"
    for argument in "$@"; do
        printf 'ARG %s\n' "$(encode_record "${argument}")" >>"${command_records}"
    done

    set +e
    if [[ "${model_environment}" == model ]]; then
        RUSTINFER_REAL_CHECKPOINT=/model "$@" >"/evidence/${log_name}" 2>&1
        status=$?
    else
        env -u RUSTINFER_REAL_CHECKPOINT "$@" >"/evidence/${log_name}" 2>&1
        status=$?
    fi
    set -e

    printf 'EXIT %s\n' "$(encode_record "${status}")" >>"${command_records}"
    printf 'END\n' >>"${command_records}"
    if ((status != 0)); then
        cat "/evidence/${log_name}" >&2
        return "${status}"
    fi
}

discover_and_copy() {
    local log_name=$1
    local cargo_target=$2
    local target_dir=$3
    local evidence_name=$4
    local compile_id=$5
    local execute_id=$6
    local candidates=()
    local cargo_executable cargo_sha copied_sha size

    mapfile -t candidates < <(
        sed -n 's/.*"executable":"\([^"]*\)".*/\1/p' "/evidence/${log_name}"
    )
    if ((${#candidates[@]} != 1)); then
        echo "expected one Cargo compiler-artifact executable for ${cargo_target}, found ${#candidates[@]}" >&2
        return 1
    fi
    cargo_executable=${candidates[0]}
    case "${cargo_executable}" in
        "${target_dir}"/debug/deps/"${cargo_target}"-*) ;;
        *)
            echo "Cargo executable escaped the reviewed target directory: ${cargo_executable}" >&2
            return 1
            ;;
    esac
    test -f "${cargo_executable}"
    test -x "${cargo_executable}"
    install -m 0755 -- "${cargo_executable}" "/evidence/${evidence_name}"
    cargo_sha=$(sha256sum "${cargo_executable}" | awk '{print $1}')
    copied_sha=$(sha256sum "/evidence/${evidence_name}" | awk '{print $1}')
    test "${cargo_sha}" = "${copied_sha}"
    size=$(stat -c '%s' "/evidence/${evidence_name}")
    test "${size}" -gt 0
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${evidence_name}" \
        "${cargo_target}" \
        "${cargo_executable}" \
        "${cargo_sha}" \
        "/evidence/${evidence_name}" \
        "${copied_sha}" \
        "${size}" \
        "${compile_id}" \
        "${execute_id}" >>"${subject_records}"
}

# Hash the same closed model-file manifest used by the Python-free release E2E
# gate.  This reads model bytes but does not load a checkpoint or initialize an
# inference runtime.
if find /model -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo "model tree contains a link or non-regular entry" >&2
    exit 1
fi
model_manifest=/runner-output/model-SHA256SUMS
: >"${model_manifest}"
model_count=0
while IFS= read -r -d '' model_file; do
    relative=${model_file#/model/}
    [[ ${relative} =~ ^[A-Za-z0-9._/+@=-]+$ ]] || {
        echo "model path uses an unsafe alphabet: ${relative}" >&2
        exit 1
    }
    printf '%s  %s\n' "$(sha256sum "${model_file}" | awk '{print $1}')" "${relative}" \
        >>"${model_manifest}"
    model_count=$((model_count + 1))
done < <(find /model -type f -print0 | sort -z)
test "${model_count}" -gt 0
test "$(sha256sum "${model_manifest}" | awk '{print $1}')" = "${RUSTINFER_MODEL_TREE_SHA256}"
test "$(sha256sum /model/model.safetensors | awk '{print $1}')" = \
    80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1
test "$(sha256sum /model/tokenizer.json | awk '{print $1}')" = \
    9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c

nvidia-smi \
    --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,compute_cap \
    --format=csv,noheader,nounits > /runner-output/gpu.csv
test "$(wc -l < /runner-output/gpu.csv)" -eq 1
grep -F 'GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0' /runner-output/gpu.csv >/dev/null

run_recorded \
    cuda-compile-only \
    cuda-compile-only.log \
    - \
    none \
    /bin/sh ci/verify_python_free_cuda.sh
install -m 0755 -- target/release/rustinfer-profile /runner-output/rustinfer-profile

run_recorded \
    workspace-all-features-all-targets \
    workspace-all-features-all-targets.log \
    - \
    none \
    cargo test --workspace --all-features --all-targets --locked --offline --color never

lifecycle_target=/workspace/target/optimizer-evidence/command-batch-lifecycle
run_recorded \
    compile-command-batch-lifecycle \
    command-batch-lifecycle-build.log \
    host-runtime-gpu-test \
    none \
    cargo test --locked --offline --package rustinfer-cuda --no-default-features \
        --features cuda --test host_runtime_gpu --no-run \
        --message-format=json-render-diagnostics --color never \
        --target-dir "${lifecycle_target}"
discover_and_copy \
    command-batch-lifecycle-build.log \
    host_runtime_gpu \
    "${lifecycle_target}" \
    host-runtime-gpu-test \
    compile-command-batch-lifecycle \
    command-batch-lifecycle
run_recorded \
    command-batch-lifecycle \
    command-batch-lifecycle-gpu.log \
    host-runtime-gpu-test \
    none \
    /evidence/host-runtime-gpu-test \
        command_batch_proxy_is_one_shot_and_drop_restores_stream_use \
        --ignored --exact --nocapture --test-threads=1 --color never

ledger_target=/workspace/target/optimizer-evidence/command-batch-resource-ledger
run_recorded \
    compile-command-batch-resource-ledger \
    command-batch-resource-ledger-build.log \
    primitives-gpu-test \
    none \
    cargo test --locked --offline --package rustinfer-cuda --no-default-features \
        --features cuda --test primitives_gpu --no-run \
        --message-format=json-render-diagnostics --color never \
        --target-dir "${ledger_target}"
discover_and_copy \
    command-batch-resource-ledger-build.log \
    primitives_gpu \
    "${ledger_target}" \
    primitives-gpu-test \
    compile-command-batch-resource-ledger \
    command-batch-resource-ledger
run_recorded \
    command-batch-resource-ledger \
    command-batch-primitives-gpu.log \
    primitives-gpu-test \
    none \
    /evidence/primitives-gpu-test \
        command_batch_releases_multi_primitive_resource_ledger_after_validation_error \
        --ignored --exact --nocapture --test-threads=1 --color never

parity_target=/workspace/target/optimizer-evidence/smollm2-multi-step-greedy-exact
run_recorded \
    compile-smollm2-multi-step-greedy-exact \
    smollm2-multi-step-greedy-exact-build.log \
    llama-batch-gpu-test \
    none \
    cargo test --locked --offline --package rustinfer-runtime --no-default-features \
        --features cuda --test llama_batch_gpu --no-run \
        --message-format=json-render-diagnostics --color never \
        --target-dir "${parity_target}"
discover_and_copy \
    smollm2-multi-step-greedy-exact-build.log \
    llama_batch_gpu \
    "${parity_target}" \
    llama-batch-gpu-test \
    compile-smollm2-multi-step-greedy-exact \
    smollm2-multi-step-greedy-exact
run_recorded \
    smollm2-multi-step-greedy-exact \
    iteration-command-batch-model-parity-gpu.log \
    llama-batch-gpu-test \
    model \
    /evidence/llama-batch-gpu-test \
        iteration_batch_completion_matches_per_operation_multi_step_greedy_exactly \
        --ignored --exact --nocapture --test-threads=1 --color never

printf '%s\n' rustinfer.optimizer-remote-run.completed.v2 > /runner-output/completed
