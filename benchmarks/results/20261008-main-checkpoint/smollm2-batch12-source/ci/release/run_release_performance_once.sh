#!/usr/bin/env bash
# Execute one native release-performance candidate run inside the immutable
# optimizer image.  The host runner creates a fresh container for every call.

set -euo pipefail
umask 022

: "${RILEY_PERF_PAIR_INDEX:?missing pair index}"
: "${RILEY_PERF_CAPTURE_ID:?missing capture ID}"
: "${RILEY_PERF_SOURCE_REVISION:?missing source revision}"
: "${RILEY_PERF_SOURCE_ARCHIVE_SHA256:?missing source archive SHA-256}"
: "${RILEY_PERF_PROFILE_BINARY_SHA256:?missing profile binary SHA-256}"
: "${RILEY_PERF_OPTIMIZER_REPORT_SHA256:?missing optimizer report SHA-256}"
: "${RILEY_PERF_OPTIMIZER_IMAGE_SHA256:?missing optimizer image SHA-256}"
: "${RILEY_PERF_MODEL_TREE_SHA256:?missing model tree SHA-256}"

readonly expected_gpu_name='NVIDIA GeForce RTX 4090'
readonly expected_gpu_uuid='GPU-9087e425-6aca-b722-b8c9-cc0423b39fb0'
readonly expected_gpu_pci_bus_id='00000000:01:00.0'
readonly expected_gpu_compute_capability='8.9'
readonly expected_gpu_memory_mib='24564'
readonly expected_driver_version='580.173.02'
readonly expected_kernel_release='6.8.0-138-generic'
readonly expected_os_pretty_name='Ubuntu 22.04.5 LTS'
readonly expected_cpu_model='13th Gen Intel(R) Core(TM) i7-13700K'
readonly expected_physical_cores='16'
readonly expected_logical_cores='24'
readonly expected_ram_bytes='67185598464'
readonly expected_cuda_runtime='12.8.1'
readonly expected_cuda_toolkit='12.8.93'
readonly expected_cublas='12.8.4.1'
readonly expected_weights_sha256='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
readonly expected_tokenizer_sha256='9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'

sha_re='^[0-9a-f]{64}$'
revision_re='^[0-9a-f]{40}$'
[[ ${RILEY_PERF_PAIR_INDEX} =~ ^[1-5]$ ]] || {
    echo 'release performance: pair index must be in 1..5' >&2
    exit 2
}
[[ ${RILEY_PERF_SOURCE_REVISION} =~ ${revision_re} ]] || {
    echo 'release performance: invalid source revision' >&2
    exit 2
}
for digest in \
    "${RILEY_PERF_SOURCE_ARCHIVE_SHA256}" \
    "${RILEY_PERF_PROFILE_BINARY_SHA256}" \
    "${RILEY_PERF_OPTIMIZER_REPORT_SHA256}" \
    "${RILEY_PERF_OPTIMIZER_IMAGE_SHA256}" \
    "${RILEY_PERF_MODEL_TREE_SHA256}"
do
    [[ ${digest} =~ ${sha_re} ]] || {
        echo 'release performance: invalid SHA-256 binding' >&2
        exit 2
    }
done

for tool in awk chmod date find grep head nvidia-smi sed sha256sum sort tar tr uname wc; do
    command -v "${tool}" >/dev/null 2>&1 || {
        echo "release performance: missing container tool ${tool}" >&2
        exit 2
    }
done

test "$(pwd -P)" = /workspace
test -r /input/source.tar
test -x /input/riley-profile
test -r /input/optimizer-correctness-report.json
test -d /model
test ! -L /model
test -d /evidence
test ! -e "/evidence/candidate-${RILEY_PERF_PAIR_INDEX}.json"
test "$(sha256sum /input/source.tar | awk '{print $1}')" = \
    "${RILEY_PERF_SOURCE_ARCHIVE_SHA256}"
test "$(sha256sum /input/riley-profile | awk '{print $1}')" = \
    "${RILEY_PERF_PROFILE_BINARY_SHA256}"
test "$(sha256sum /input/optimizer-correctness-report.json | awk '{print $1}')" = \
    "${RILEY_PERF_OPTIMIZER_REPORT_SHA256}"
test -r /workspace/benchmarks/prompts.jsonl
test -r /workspace/ci/release/run_release_performance_once.sh

if find /model -mindepth 1 ! -type d ! -type f -print -quit | grep -q .; then
    echo 'release performance: model tree contains a link or special entry' >&2
    exit 2
fi
model_manifest=/tmp/riley-release-performance-model-SHA256SUMS
test ! -e "${model_manifest}"
: >"${model_manifest}"
model_file_count=0
while IFS= read -r -d '' model_file; do
    relative=${model_file#/model/}
    [[ ${relative} =~ ^[A-Za-z0-9._/+@=-]+$ ]] || {
        echo "release performance: unsafe model path ${relative}" >&2
        exit 2
    }
    printf '%s  %s\n' "$(sha256sum "${model_file}" | awk '{print $1}')" "${relative}" \
        >>"${model_manifest}"
    model_file_count=$((model_file_count + 1))
done < <(find /model -type f -print0 | sort -z)
test "${model_file_count}" -gt 0
test "$(sha256sum "${model_manifest}" | awk '{print $1}')" = \
    "${RILEY_PERF_MODEL_TREE_SHA256}"
test "$(sha256sum /model/model.safetensors | awk '{print $1}')" = \
    "${expected_weights_sha256}"
test "$(sha256sum /model/tokenizer.json | awk '{print $1}')" = \
    "${expected_tokenizer_sha256}"

trim() {
    local value=$1
    value=${value#"${value%%[![:space:]]*}"}
    value=${value%"${value##*[![:space:]]}"}
    printf '%s' "${value}"
}

test "$(nvidia-smi --list-gpus | wc -l | tr -d '[:space:]')" = 1
gpu_row=$(nvidia-smi --id="${expected_gpu_uuid}" \
    --query-gpu=name,uuid,pci.bus_id,memory.total,driver_version,compute_cap \
    --format=csv,noheader,nounits)
IFS=',' read -r gpu_name gpu_uuid gpu_pci_bus_id gpu_memory_mib driver_version compute_capability \
    <<<"${gpu_row}"
gpu_name=$(trim "${gpu_name}")
gpu_uuid=$(trim "${gpu_uuid}")
gpu_pci_bus_id=$(trim "${gpu_pci_bus_id}")
gpu_memory_mib=$(trim "${gpu_memory_mib}")
driver_version=$(trim "${driver_version}")
compute_capability=$(trim "${compute_capability}")
test "${gpu_name}" = "${expected_gpu_name}"
test "${gpu_uuid}" = "${expected_gpu_uuid}"
test "${gpu_pci_bus_id}" = "${expected_gpu_pci_bus_id}"
test "${gpu_memory_mib}" = "${expected_gpu_memory_mib}"
test "${driver_version}" = "${expected_driver_version}"
test "${compute_capability}" = "${expected_gpu_compute_capability}"

test "$(uname -r)" = "${expected_kernel_release}"
test "$(uname -m)" = x86_64
os_id=$(sed -n 's/^ID=//p' /etc/os-release | tr -d '"' | head -n 1)
os_version=$(sed -n 's/^VERSION_ID=//p' /etc/os-release | tr -d '"' | head -n 1)
os_pretty_name=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | head -n 1)
os_pretty_name=${os_pretty_name#\"}
os_pretty_name=${os_pretty_name%\"}
test "${os_id}" = ubuntu
test "${os_version}" = 22.04
test "${os_pretty_name}" = "${expected_os_pretty_name}"
cpu_model=$(awk -F: '/^model name[[:space:]]*:/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}' /proc/cpuinfo)
test "${cpu_model}" = "${expected_cpu_model}"
physical_cores=$(awk -F: '
    /^physical id[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); package=$2}
    /^core id[[:space:]]*:/ {gsub(/[[:space:]]/, "", $2); seen[package ":" $2]=1}
    END {for (key in seen) count++; print count+0}
' /proc/cpuinfo)
logical_cores=$(awk -F: '/^processor[[:space:]]*:/ {count++} END {print count+0}' /proc/cpuinfo)
ram_kib=$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo)
[[ ${ram_kib} =~ ^[0-9]+$ ]]
ram_bytes=$((ram_kib * 1024))
test "${physical_cores}" = "${expected_physical_cores}"
test "${logical_cores}" = "${expected_logical_cores}"
test "${ram_bytes}" = "${expected_ram_bytes}"
[[ ${RILEY_PERF_CAPTURE_ID:-} =~ ^[0-9a-f]{64}$ ]]

test "${CUDA_VERSION:-}" = "${expected_cuda_runtime}"
nvcc_output=$(/usr/local/cuda/bin/nvcc --version)
grep -F 'Cuda compilation tools, release 12.8, V12.8.93' <<<"${nvcc_output}" >/dev/null
cublas_header=/usr/local/cuda/include/cublas_api.h
test -r "${cublas_header}"
cublas_major=$(awk '$1 == "#define" && $2 == "CUBLAS_VER_MAJOR" {print $3; exit}' "${cublas_header}")
cublas_minor=$(awk '$1 == "#define" && $2 == "CUBLAS_VER_MINOR" {print $3; exit}' "${cublas_header}")
cublas_patch=$(awk '$1 == "#define" && $2 == "CUBLAS_VER_PATCH" {print $3; exit}' "${cublas_header}")
cublas_build=$(awk '$1 == "#define" && $2 == "CUBLAS_VER_BUILD" {print $3; exit}' "${cublas_header}")
cublas_version="${cublas_major}.${cublas_minor}.${cublas_patch}.${cublas_build}"
test "${cublas_version}" = "${expected_cublas}"

recorded_at_utc=$(/usr/bin/date -u +%Y-%m-%dT%H:%M:%S.%NZ)
run_id="pr16-iteration-batch-${RILEY_PERF_SOURCE_REVISION:0:12}-${RILEY_PERF_CAPTURE_ID}-pair${RILEY_PERF_PAIR_INDEX}"
gpu_vram_bytes=$((expected_gpu_memory_mib * 1024 * 1024))
output="/evidence/candidate-${RILEY_PERF_PAIR_INDEX}.json"

/input/riley-profile \
    --model /model \
    --prompts /workspace/benchmarks/prompts.jsonl \
    --output "${output}" \
    --role candidate \
    --pair-index "${RILEY_PERF_PAIR_INDEX}" \
    --run-id "${run_id}" \
    --recorded-at-utc "${recorded_at_utc}" \
    --git-commit "${RILEY_PERF_SOURCE_REVISION}" \
    --git-dirty false \
    --executable-sha256 "${RILEY_PERF_PROFILE_BINARY_SHA256}" \
    --implementation-id native-iteration-command-batch \
    --runtime-flag-name execution_completion \
    --runtime-flag-value iteration-batch \
    --semantic-class E0 \
    --correctness-gate-id pr15-iteration-command-batch-exact-v1 \
    --correctness-report-sha256 "${RILEY_PERF_OPTIMIZER_REPORT_SHA256}" \
    --gpu-model "${gpu_name}" \
    --gpu-uuid "${gpu_uuid}" \
    --device-index 0 \
    --gpu-pci-bus-id "${gpu_pci_bus_id}" \
    --gpu-compute-capability "${compute_capability}" \
    --gpu-vram-bytes "${gpu_vram_bytes}" \
    --environment-id server-4096-rtx4090-pr15-v1 \
    --cpu-model "${cpu_model}" \
    --physical-core-count "${physical_cores}" \
    --logical-core-count "${logical_cores}" \
    --ram-bytes "${ram_bytes}" \
    --os-release "${os_pretty_name}" \
    --kernel-release "${expected_kernel_release}" \
    --architecture x86_64 \
    --nvidia-driver-version "${driver_version}" \
    --cuda-runtime-version "${expected_cuda_runtime}" \
    --cuda-toolkit-version "${expected_cuda_toolkit}" \
    --cublas-version "${cublas_version}" \
    --container-image-sha256 "${RILEY_PERF_OPTIMIZER_IMAGE_SHA256}" \
    --workload-id smollm2-c1-p128-o32-greedy-v1 \
    --model-id HuggingFaceTB/SmolLM2-135M \
    --model-revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 \
    --weights-sha256 "${expected_weights_sha256}" \
    --tokenizer-sha256 "${expected_tokenizer_sha256}" \
    --dtype bf16 \
    --concurrency 1 \
    --prompt-tokens 128 \
    --output-tokens 32 \
    --warmups 5 \
    --measured-iterations 30 \
    --sampling-id greedy \
    --seed none

test -s "${output}"
/usr/bin/chmod 0444 "${output}"
