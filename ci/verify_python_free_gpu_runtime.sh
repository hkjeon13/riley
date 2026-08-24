#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

for command_name in python python3 pip pip3; do
    if command -v "$command_name" >/dev/null 2>&1; then
        echo "Python-free GPU runtime smoke found forbidden executable: $command_name" >&2
        exit 1
    fi
done

: "${RUSTINFER_GPU_EVIDENCE_DIR:?mount a writable evidence directory and set RUSTINFER_GPU_EVIDENCE_DIR}"
: "${RUSTINFER_SOURCE_REVISION:?set the exact source revision used to build the image}"
: "${RUSTINFER_SOURCE_ARCHIVE_SHA256:?set the SHA-256 of git archive --format=tar HEAD}"
: "${RUSTINFER_GPU_IMAGE_ID:?set the immutable Docker image ID under test}"
: "${RUSTINFER_CUDA_LEAK_ITERATIONS:=128}"
: "${RUSTINFER_CUDA_COMPUTE_SANITIZER:=0}"

if ! printf '%s\n' "$RUSTINFER_SOURCE_REVISION" \
    | grep -Eq '^[0-9a-f]{40,64}$'
then
    echo "RUSTINFER_SOURCE_REVISION must be a full lowercase Git object ID" >&2
    exit 1
fi
if ! printf '%s\n' "$RUSTINFER_SOURCE_ARCHIVE_SHA256" \
    | grep -Eq '^[0-9a-f]{64}$'
then
    echo "RUSTINFER_SOURCE_ARCHIVE_SHA256 must be a lowercase SHA-256" >&2
    exit 1
fi
if ! printf '%s\n' "$RUSTINFER_GPU_IMAGE_ID" \
    | grep -Eq '^sha256:[0-9a-f]{64}$'
then
    echo "RUSTINFER_GPU_IMAGE_ID must be an immutable sha256 Docker image ID" >&2
    exit 1
fi

case "$RUSTINFER_CUDA_COMPUTE_SANITIZER" in
    0|false|no) sanitizer_enabled=0 ;;
    1|true|yes) sanitizer_enabled=1 ;;
    *)
        echo "RUSTINFER_CUDA_COMPUTE_SANITIZER must be a boolean" >&2
        exit 1
        ;;
esac

case "$RUSTINFER_GPU_EVIDENCE_DIR" in
    /*) ;;
    *) echo "RUSTINFER_GPU_EVIDENCE_DIR must be an absolute container path" >&2; exit 1 ;;
esac
if [ "$RUSTINFER_GPU_EVIDENCE_DIR" = / ]; then
    echo "refusing to write GPU evidence at the filesystem root" >&2
    exit 1
fi
test -d "$RUSTINFER_GPU_EVIDENCE_DIR"
test -w "$RUSTINFER_GPU_EVIDENCE_DIR"

case "$RUSTINFER_CUDA_LEAK_ITERATIONS" in
    ''|*[!0-9]*)
        echo "RUSTINFER_CUDA_LEAK_ITERATIONS must be an integer from 32 through 4096" >&2
        exit 1
        ;;
esac
if [ "$RUSTINFER_CUDA_LEAK_ITERATIONS" -lt 32 ] \
    || [ "$RUSTINFER_CUDA_LEAK_ITERATIONS" -gt 4096 ]
then
    echo "RUSTINFER_CUDA_LEAK_ITERATIONS must be from 32 through 4096" >&2
    exit 1
fi
export RUSTINFER_CUDA_LEAK_ITERATIONS
export RUST_TEST_THREADS=1

environment_log="$RUSTINFER_GPU_EVIDENCE_DIR/environment.txt"
device_list_log="$RUSTINFER_GPU_EVIDENCE_DIR/nvidia-smi-list.txt"
device_csv="$RUSTINFER_GPU_EVIDENCE_DIR/nvidia-smi-device-metadata.csv"
test_list_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-test-list.txt"
test_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-tests.log"
checksum_file="$RUSTINFER_GPU_EVIDENCE_DIR/SHA256SUMS"
sanitizer_log="$RUSTINFER_GPU_EVIDENCE_DIR/compute-sanitizer-memcheck.log"
ldd_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-ldd.txt"
readelf_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-readelf.txt"
nm_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-nm.txt"
driver_libraries_log="$RUSTINFER_GPU_EVIDENCE_DIR/cuda-driver-libraries.txt"

for output in \
    "$environment_log" \
    "$device_list_log" \
    "$device_csv" \
    "$test_list_log" \
    "$test_log" \
    "$ldd_log" \
    "$readelf_log" \
    "$nm_log" \
    "$driver_libraries_log" \
    "$checksum_file"
do
    if [ -e "$output" ]; then
        echo "refusing to replace existing GPU evidence: $output" >&2
        exit 1
    fi
done
if [ "$sanitizer_enabled" -eq 1 ] && [ -e "$sanitizer_log" ]; then
    echo "refusing to replace existing GPU evidence: $sanitizer_log" >&2
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; run the image with NVIDIA utility capability" >&2
    exit 1
fi
for device_node in /dev/nvidiactl /dev/nvidia-uvm; do
    if [ ! -c "$device_node" ]; then
        echo "required NVIDIA device node is unavailable: $device_node" >&2
        exit 1
    fi
done

if ! {
    printf 'source_revision=%s\n' "$RUSTINFER_SOURCE_REVISION"
    printf 'source_archive_command=git archive --format=tar HEAD\n'
    printf 'source_archive_sha256=%s\n' "$RUSTINFER_SOURCE_ARCHIVE_SHA256"
    printf 'gpu_image_id=%s\n' "$RUSTINFER_GPU_IMAGE_ID"
    printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-all}"
    printf 'nvidia_visible_devices=%s\n' "${NVIDIA_VISIBLE_DEVICES:-all}"
    printf 'leak_iterations=%s\n' "$RUSTINFER_CUDA_LEAK_ITERATIONS"
    printf 'compute_sanitizer=%s\n' "$sanitizer_enabled"
    uname -a
    rustc --version --verbose
    cargo --version --verbose
    "${CUDA_HOME:?CUDA_HOME is required}/bin/nvcc" --version
    cmake --version
    target/release/rustinfer --version
    cat /etc/os-release
} >"$environment_log" 2>&1
then
    cat "$environment_log"
    exit 1
fi
cat "$environment_log"

if ! nvidia-smi -L >"$device_list_log" 2>&1; then
    cat "$device_list_log"
    exit 1
fi
cat "$device_list_log"

if ! nvidia-smi \
    --query-gpu=index,uuid,name,compute_cap,memory.total,driver_version \
    --format=csv,noheader,nounits >"$device_csv" 2>&1
then
    cat "$device_csv"
    exit 1
fi
cat "$device_csv"
test -s "$device_csv"

if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test host_runtime_gpu \
    -- --list --format terse --color never >"$test_list_log" 2>&1
then
    cat "$test_list_log"
    exit 1
fi
cat "$test_list_log"

expected_tests='device_metadata_is_reported
invalid_device_is_rejected
two_stream_event_ordering_is_explicit
async_fill_is_correct_after_sync
invalid_launch_reports_launch_stage
events_report_positive_elapsed_time
repeated_create_drop_has_no_resource_leak'

for test_name in $expected_tests; do
    if ! grep -Fqx "$test_name: test" "$test_list_log"; then
        echo "GPU integration target is missing required test: $test_name" >&2
        exit 1
    fi
done
test_list_count=$(grep -Ec ': test$' "$test_list_log")
if [ "$test_list_count" -ne 7 ]; then
    echo "expected exactly 7 GPU integration tests, found $test_list_count" >&2
    exit 1
fi

test_binary=
binary_count=0
for candidate in target/debug/deps/host_runtime_gpu-*; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        test_binary=$candidate
        binary_count=$((binary_count + 1))
    fi
done
if [ "$binary_count" -ne 1 ]; then
    echo "expected one host_runtime_gpu test executable, found $binary_count" >&2
    exit 1
fi

if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test host_runtime_gpu \
    -- --ignored --test-threads=1 --nocapture --color never >"$test_log" 2>&1
then
    cat "$test_log"
    exit 1
fi
cat "$test_log"

grep -Eq \
    'rustinfer-cuda-device-metadata ordinal=[0-9]+ name=.+ compute_capability=[0-9]+\.[0-9]+ total_memory_bytes=[1-9][0-9]* multiprocessor_count=[1-9][0-9]* driver_version=[0-9]+ runtime_version=[0-9]+' \
    "$test_log"
grep -Eq \
    "rustinfer-cuda-leak-smoke iterations=${RUSTINFER_CUDA_LEAK_ITERATIONS}( |$)" \
    "$test_log"
grep -Eq 'test result: ok\. 7 passed; 0 failed; 0 ignored;' "$test_log"

ldd "$test_binary" >"$ldd_log"
cat "$ldd_log"
if grep -Eq '=>[[:space:]]+not found' "$ldd_log"; then
    echo "host-runtime test has an unresolved dynamic dependency" >&2
    exit 1
fi
grep -Eq 'libcudart\.so' "$ldd_log"
grep -Eq 'libcuda\.so\.1' "$ldd_log"

readelf -d "$test_binary" >"$readelf_log"
cat "$readelf_log"
grep -Eq 'NEEDED.*libcudart\.so' "$readelf_log"
grep -Eq 'NEEDED.*libcuda\.so\.1' "$readelf_log"
if grep -Eq '(RPATH|RUNPATH).*stubs' "$readelf_log"; then
    echo "host-runtime test embeds a CUDA driver stubs runtime path" >&2
    exit 1
fi

nm -D --undefined-only "$test_binary" >"$nm_log"
cat "$nm_log"

ldconfig -p >"$driver_libraries_log"
cat "$driver_libraries_log"
grep -Eq 'libcuda\.so\.1' "$driver_libraries_log"
grep -Eq 'libcudart\.so' "$driver_libraries_log"

if grep -Eiq 'python|pytorch|torch|transformers|triton' \
    "$ldd_log" "$readelf_log" "$nm_log"
then
    echo "host-runtime test contains a forbidden runtime dependency" >&2
    exit 1
fi

if [ "$sanitizer_enabled" -eq 1 ]; then
    command -v compute-sanitizer >/dev/null 2>&1
    if ! compute-sanitizer \
        --tool memcheck \
        --leak-check full \
        --error-exitcode 86 \
        "$test_binary" \
        --ignored \
        --test-threads=1 \
        --nocapture \
        --color never >"$sanitizer_log" 2>&1
    then
        cat "$sanitizer_log"
        exit 1
    fi
    cat "$sanitizer_log"
    grep -Eq 'ERROR SUMMARY: 0 errors' "$sanitizer_log"
fi

(
    cd "$RUSTINFER_GPU_EVIDENCE_DIR"
    evidence_files='environment.txt
nvidia-smi-list.txt
nvidia-smi-device-metadata.csv
host-runtime-test-list.txt
host-runtime-tests.log
host-runtime-ldd.txt
host-runtime-readelf.txt
host-runtime-nm.txt
cuda-driver-libraries.txt'
    if [ "$sanitizer_enabled" -eq 1 ]; then
        evidence_files="$evidence_files
compute-sanitizer-memcheck.log"
    fi
    # File names are fixed above and intentionally contain no whitespace.
    # shellcheck disable=SC2086
    sha256sum $evidence_files >SHA256SUMS
)
cat "$checksum_file"

echo "Python-free CUDA host-runtime GPU verification passed"
