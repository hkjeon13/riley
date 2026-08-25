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
memory_test_list_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-test-list.txt"
memory_test_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-tests.log"
memory_fault_test_list_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-fault-test-list.txt"
memory_fault_test_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-fault-tests.log"
memory_fault_test_binary_checksum_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-fault-test-binary.sha256"
checksum_file="$RUSTINFER_GPU_EVIDENCE_DIR/SHA256SUMS"
sanitizer_log="$RUSTINFER_GPU_EVIDENCE_DIR/compute-sanitizer-memcheck.log"
memory_sanitizer_log="$RUSTINFER_GPU_EVIDENCE_DIR/compute-sanitizer-memory-memcheck.log"
ldd_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-ldd.txt"
readelf_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-readelf.txt"
nm_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-nm.txt"
memory_ldd_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-ldd.txt"
memory_readelf_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-readelf.txt"
memory_nm_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-nm.txt"
test_binary_checksum_log="$RUSTINFER_GPU_EVIDENCE_DIR/host-runtime-test-binary.sha256"
memory_test_binary_checksum_log="$RUSTINFER_GPU_EVIDENCE_DIR/memory-test-binary.sha256"
release_binary_checksum_log="$RUSTINFER_GPU_EVIDENCE_DIR/release-binary.sha256"
release_ldd_log="$RUSTINFER_GPU_EVIDENCE_DIR/release-ldd.txt"
release_readelf_log="$RUSTINFER_GPU_EVIDENCE_DIR/release-readelf.txt"
release_nm_log="$RUSTINFER_GPU_EVIDENCE_DIR/release-nm.txt"
driver_libraries_log="$RUSTINFER_GPU_EVIDENCE_DIR/cuda-driver-libraries.txt"

for output in \
    "$environment_log" \
    "$device_list_log" \
    "$device_csv" \
    "$test_list_log" \
    "$test_log" \
    "$memory_test_list_log" \
    "$memory_test_log" \
    "$memory_fault_test_list_log" \
    "$memory_fault_test_log" \
    "$memory_fault_test_binary_checksum_log" \
    "$ldd_log" \
    "$readelf_log" \
    "$nm_log" \
    "$memory_ldd_log" \
    "$memory_readelf_log" \
    "$memory_nm_log" \
    "$test_binary_checksum_log" \
    "$memory_test_binary_checksum_log" \
    "$release_binary_checksum_log" \
    "$release_ldd_log" \
    "$release_readelf_log" \
    "$release_nm_log" \
    "$driver_libraries_log" \
    "$sanitizer_log" \
    "$memory_sanitizer_log" \
    "$checksum_file"
do
    if [ -e "$output" ]; then
        echo "refusing to replace existing GPU evidence: $output" >&2
        exit 1
    fi
done

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
command_batch_proxy_is_one_shot_and_drop_restores_stream_use
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
test_list_count=$(grep -Ec ': test$' "$test_list_log" || true)
if [ "$test_list_count" -ne 8 ]; then
    echo "expected exactly 8 GPU integration tests, found $test_list_count" >&2
    exit 1
fi

if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test memory_gpu \
    -- --list --format terse --color never >"$memory_test_list_log" 2>&1
then
    cat "$memory_test_list_log"
    exit 1
fi
cat "$memory_test_list_log"

expected_memory_tests='allocation_accounting_returns_to_zero
zero_byte_allocations_and_copies_are_logical_noops
pinned_host_device_round_trip_is_exact
two_stream_copy_handoff_prevents_early_reuse
copy_ranges_and_context_ownership_are_validated'

for test_name in $expected_memory_tests; do
    if ! grep -Fqx "$test_name: test" "$memory_test_list_log"; then
        echo "GPU memory target is missing required test: $test_name" >&2
        exit 1
    fi
done
memory_test_list_count=$(grep -Ec ': test$' "$memory_test_list_log" || true)
if [ "$memory_test_list_count" -ne 5 ]; then
    echo "expected exactly 5 GPU memory tests, found $memory_test_list_count" >&2
    exit 1
fi

if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda-test-fault-injection \
    --test memory_fault_injection_gpu \
    -- --list --format terse --color never >"$memory_fault_test_list_log" 2>&1
then
    cat "$memory_fault_test_list_log"
    exit 1
fi
cat "$memory_fault_test_list_log"
grep -Fqx 'memory_fault_cases_are_subprocess_isolated: test' "$memory_fault_test_list_log"
grep -Fqx 'memory_fault_subprocess: test' "$memory_fault_test_list_log"
memory_fault_test_count=$(grep -Ec ': test$' "$memory_fault_test_list_log" || true)
if [ "$memory_fault_test_count" -ne 2 ]; then
    echo "expected exactly 2 GPU memory fault harness tests, found $memory_fault_test_count" >&2
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

memory_test_binary=
memory_binary_count=0
for candidate in target/debug/deps/memory_gpu-*; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        memory_test_binary=$candidate
        memory_binary_count=$((memory_binary_count + 1))
    fi
done
if [ "$memory_binary_count" -ne 1 ]; then
    echo "expected one memory_gpu test executable, found $memory_binary_count" >&2
    exit 1
fi

memory_fault_test_binary=
memory_fault_binary_count=0
for candidate in target/debug/deps/memory_fault_injection_gpu-*; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        memory_fault_test_binary=$candidate
        memory_fault_binary_count=$((memory_fault_binary_count + 1))
    fi
done
if [ "$memory_fault_binary_count" -ne 1 ]; then
    echo "expected one memory_fault_injection_gpu test executable, found $memory_fault_binary_count" >&2
    exit 1
fi

sha256sum "$test_binary" >"$test_binary_checksum_log"
sha256sum "$memory_test_binary" >"$memory_test_binary_checksum_log"
sha256sum "$memory_fault_test_binary" >"$memory_fault_test_binary_checksum_log"
cat "$test_binary_checksum_log"
cat "$memory_test_binary_checksum_log"
cat "$memory_fault_test_binary_checksum_log"

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
grep -Eq 'test result: ok\. 8 passed; 0 failed; 0 ignored;' "$test_log"

if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test memory_gpu \
    -- --ignored --test-threads=1 --nocapture --color never >"$memory_test_log" 2>&1
then
    cat "$memory_test_log"
    exit 1
fi
cat "$memory_test_log"

memory_accounting_marker='rustinfer-cuda-memory-accounting device_live_bytes=0 device_live_allocations=0 pinned_host_live_bytes=0 pinned_host_live_allocations=0'
memory_marker_count=$(grep -Fxc "$memory_accounting_marker" "$memory_test_log" || true)
if [ "$memory_marker_count" -ne 1 ]; then
    echo "expected exactly one all-zero GPU memory accounting marker, found $memory_marker_count" >&2
    exit 1
fi
grep -Eq 'test result: ok\. 5 passed; 0 failed; 0 ignored;' "$memory_test_log"

# Only the parent harness is selected. It starts each destructive ambiguity case
# in a fresh child process so a poisoned primary context is never reused.
if ! CARGO_TERM_COLOR=never cargo test \
    --color never \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda-test-fault-injection \
    --test memory_fault_injection_gpu \
    -- --ignored --exact memory_fault_cases_are_subprocess_isolated \
    --test-threads=1 --nocapture --color never >"$memory_fault_test_log" 2>&1
then
    cat "$memory_fault_test_log"
    exit 1
fi
cat "$memory_fault_test_log"
grep -Eq 'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;' "$memory_fault_test_log"

for fault_case in \
    create-rollback-ambiguous \
    explicit-close-ambiguous \
    deferred-submission-error \
    completion-restore-ambiguous
do
    spawn_marker_count=$(grep -Ec "rustinfer-cuda-memory-fault-case case=${fault_case} event=spawn parent_pid=[1-9][0-9]* child_pid=[1-9][0-9]*" "$memory_fault_test_log" || true)
    start_marker_count=$(grep -Ec "rustinfer-cuda-memory-fault-case case=${fault_case} event=start child_pid=[1-9][0-9]*" "$memory_fault_test_log" || true)
    passed_marker_count=$(grep -Ec "rustinfer-cuda-memory-fault-case case=${fault_case} event=passed child_pid=[1-9][0-9]*" "$memory_fault_test_log" || true)
    joined_marker_count=$(grep -Ec "rustinfer-cuda-memory-fault-case case=${fault_case} event=joined parent_pid=[1-9][0-9]* child_pid=[1-9][0-9]* exit_code=0" "$memory_fault_test_log" || true)
    if [ "$spawn_marker_count" -ne 1 ] \
        || [ "$start_marker_count" -ne 1 ] \
        || [ "$passed_marker_count" -ne 1 ] \
        || [ "$joined_marker_count" -ne 1 ]
    then
        echo "fault case did not emit one complete subprocess marker sequence: $fault_case" >&2
        exit 1
    fi
done

fault_summary_count=$(grep -Ec 'test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out;' "$memory_fault_test_log" || true)
if [ "$fault_summary_count" -ne 5 ]; then
    echo "expected four child and one parent fault-test success summaries, found $fault_summary_count" >&2
    exit 1
fi

release_binary=target/release/rustinfer
test -f "$release_binary"
test -x "$release_binary"
sha256sum "$release_binary" >"$release_binary_checksum_log"
{
    printf 'artifact=%s\n' "$release_binary"
    ldd "$release_binary"
} >"$release_ldd_log"
{
    printf 'artifact=%s\n' "$release_binary"
    readelf -d "$release_binary"
} >"$release_readelf_log"
{
    printf 'artifact=%s\n' "$release_binary"
    nm -a --defined-only "$release_binary"
} >"$release_nm_log"
cat "$release_binary_checksum_log"
cat "$release_ldd_log"
cat "$release_readelf_log"
cat "$release_nm_log"

if grep -Eq '=>[[:space:]]+not found' "$release_ldd_log"; then
    echo "production release binary has an unresolved dynamic dependency" >&2
    exit 1
fi
grep -Eq 'libcudart\.so' "$release_ldd_log"
grep -Eq 'libcuda\.so\.1' "$release_ldd_log"
grep -Eq 'NEEDED.*libcudart\.so' "$release_readelf_log"
grep -Eq 'NEEDED.*libcuda\.so\.1' "$release_readelf_log"
if grep -Eq '(RPATH|RUNPATH)' "$release_readelf_log"; then
    echo "production release binary embeds an unreviewed runtime search path" >&2
    exit 1
fi
if grep -aFq 'rustinfer_cuda_test_memory_fault_' "$release_binary" \
    || grep -Fq 'rustinfer_cuda_test_memory_fault_' "$release_nm_log"
then
    echo "production release binary unexpectedly contains CUDA test fault injection" >&2
    exit 1
fi
if grep -Eiq 'libpython|pytorch|torch|transformers|triton' \
    "$release_ldd_log" "$release_readelf_log"
then
    echo "production release binary contains a forbidden runtime dependency" >&2
    exit 1
fi

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

ldd "$memory_test_binary" >"$memory_ldd_log"
cat "$memory_ldd_log"
if grep -Eq '=>[[:space:]]+not found' "$memory_ldd_log"; then
    echo "memory GPU test has an unresolved dynamic dependency" >&2
    exit 1
fi
grep -Eq 'libcudart\.so' "$memory_ldd_log"
grep -Eq 'libcuda\.so\.1' "$memory_ldd_log"

readelf -d "$memory_test_binary" >"$memory_readelf_log"
cat "$memory_readelf_log"
grep -Eq 'NEEDED.*libcudart\.so' "$memory_readelf_log"
grep -Eq 'NEEDED.*libcuda\.so\.1' "$memory_readelf_log"
if grep -Eq '(RPATH|RUNPATH).*stubs' "$memory_readelf_log"; then
    echo "memory GPU test embeds a CUDA driver stubs runtime path" >&2
    exit 1
fi

nm -D --undefined-only "$memory_test_binary" >"$memory_nm_log"
cat "$memory_nm_log"

ldconfig -p >"$driver_libraries_log"
cat "$driver_libraries_log"
grep -Eq 'libcuda\.so\.1' "$driver_libraries_log"
grep -Eq 'libcudart\.so' "$driver_libraries_log"

if grep -Eiq 'python|pytorch|torch|transformers|triton' \
    "$ldd_log" "$readelf_log" "$nm_log" \
    "$memory_ldd_log" "$memory_readelf_log" "$memory_nm_log"
then
    echo "GPU integration test contains a forbidden runtime dependency" >&2
    exit 1
fi

if [ "$sanitizer_enabled" -eq 1 ]; then
    command -v compute-sanitizer >/dev/null 2>&1
    if ! compute-sanitizer \
        --tool memcheck \
        --leak-check full \
        --report-api-errors no \
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
    grep -Eq 'LEAK SUMMARY:[[:space:]]+0 bytes leaked' "$sanitizer_log"
    grep -Eq 'test result: ok\. 8 passed; 0 failed; 0 ignored;' "$sanitizer_log"

    if ! compute-sanitizer \
        --tool memcheck \
        --leak-check full \
        --report-api-errors no \
        --error-exitcode 86 \
        "$memory_test_binary" \
        --ignored \
        --test-threads=1 \
        --nocapture \
        --color never >"$memory_sanitizer_log" 2>&1
    then
        cat "$memory_sanitizer_log"
        exit 1
    fi
    cat "$memory_sanitizer_log"
    grep -Eq 'ERROR SUMMARY: 0 errors' "$memory_sanitizer_log"
    grep -Eq 'LEAK SUMMARY:[[:space:]]+0 bytes leaked' "$memory_sanitizer_log"
    memory_sanitizer_marker_count=$(grep -Fxc "$memory_accounting_marker" "$memory_sanitizer_log" || true)
    if [ "$memory_sanitizer_marker_count" -ne 1 ]; then
        echo "expected exactly one sanitizer all-zero memory marker, found $memory_sanitizer_marker_count" >&2
        exit 1
    fi
    grep -Eq 'test result: ok\. 5 passed; 0 failed; 0 ignored;' "$memory_sanitizer_log"
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
host-runtime-test-binary.sha256
memory-test-list.txt
memory-tests.log
memory-fault-test-list.txt
memory-fault-tests.log
memory-fault-test-binary.sha256
memory-ldd.txt
memory-readelf.txt
memory-nm.txt
memory-test-binary.sha256
release-binary.sha256
release-ldd.txt
release-readelf.txt
release-nm.txt
cuda-driver-libraries.txt'
    if [ "$sanitizer_enabled" -eq 1 ]; then
        evidence_files="$evidence_files
compute-sanitizer-memcheck.log
compute-sanitizer-memory-memcheck.log"
    fi
    # File names are fixed above and intentionally contain no whitespace.
    # shellcheck disable=SC2086
    sha256sum $evidence_files | LC_ALL=C sort -k2 >SHA256SUMS
)
cat "$checksum_file"

echo "Python-free CUDA host-runtime and memory GPU verification passed"
