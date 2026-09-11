#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

for command_name in python python3 pip pip3; do
    if command -v "$command_name" >/dev/null 2>&1; then
        echo "Python-free CUDA smoke found forbidden executable: $command_name" >&2
        exit 1
    fi
done

: "${CUDA_HOME:?CUDA_HOME must identify the pinned CUDA toolkit}"
: "${CUDAToolkit_ROOT:?CUDAToolkit_ROOT must identify the pinned CUDA toolkit}"
: "${RILEY_CUDA_ARCHITECTURES:?RILEY_CUDA_ARCHITECTURES must be explicit}"
: "${RILEY_SOURCE_REVISION:?RILEY_SOURCE_REVISION must identify the exact source revision}"

if ! printf '%s\n' "$RILEY_SOURCE_REVISION" | grep -Eq '^[0-9a-f]{40}$'; then
    echo "RILEY_SOURCE_REVISION must be a full lowercase Git object ID" >&2
    exit 1
fi

test "$CUDA_HOME" = "$CUDAToolkit_ROOT"
test -x "$CUDA_HOME/bin/nvcc"

rustc_version=$(rustc --version)
cargo_version=$(cargo --version)
printf '%s\n' "$rustc_version" "$cargo_version"
printf '%s\n' "$rustc_version" | grep -Eq '^rustc 1\.85\.0 '
printf '%s\n' "$cargo_version" | grep -Eq '^cargo 1\.85\.0 '
uname -m | grep -Eq '^x86_64$'
"$CUDA_HOME/bin/nvcc" --version

build_log=$(mktemp)
invalid_log=$(mktemp)
ldd_log=$(mktemp)
readelf_log=$(mktemp)
nm_log=$(mktemp)
driver_stub_runtime=$(mktemp -d)
cleanup() {
    rm -f -- "$build_log" "$invalid_log" "$ldd_log" "$readelf_log" "$nm_log"
    rm -rf -- "$driver_stub_runtime"
    rm -rf -- /tmp/riley-invalid-cuda-target
}
trap cleanup EXIT HUP INT TERM

# PR 03 links the Driver API, while this compile-only image deliberately has no
# host driver injection. The CUDA-gated calibration producer additionally
# links NVML but is never executed here. Supply both toolkit stubs under their
# SONAMEs only for dependency inspection and the two metadata-only executables
# below. Device/context/NVML calls remain forbidden; the separate GPU verifier
# requires the real injected driver libraries.
driver_stub_linker="$CUDA_HOME/lib64/stubs/libcuda.so"
nvml_stub_linker="$CUDA_HOME/lib64/stubs/libnvidia-ml.so"
test -f "$driver_stub_linker"
test -f "$nvml_stub_linker"
ln -s "$driver_stub_linker" "$driver_stub_runtime/libcuda.so.1"
ln -s "$nvml_stub_linker" "$driver_stub_runtime/libnvidia-ml.so.1"
run_host_metadata() {
    LD_LIBRARY_PATH="$driver_stub_runtime:$CUDA_HOME/lib64" "$@"
}

if ! cargo build --locked --release --features cuda,server >"$build_log" 2>&1; then
    cat "$build_log"
    exit 1
fi
cat "$build_log"

# Execute the plan's exact root command spelling as a second, cached smoke.
# The first command proves that its dependency resolution is lockfile-closed.
cargo build --locked --offline --release --features cuda,server

# Compile and link the native benchmark evidence producer without executing a
# model or initializing a device. Its bench+CUDA-only surface belongs to the
# same Python-free, lockfile-closed AOT lane as the production binary.
cargo build \
    --locked \
    --release \
    --package riley-server \
    --no-default-features \
    --features bench,cuda \
    --bin riley-profile

cargo test \
    --locked \
    --package riley-server \
    --no-default-features \
    --features bench,cuda \
    --bin riley-profile \
    --no-run

# Build the Python-free calibration producer exactly once as a locked release
# artifact. This command only compiles and links it; executing the producer or
# allowing it to initialize NVML, CUDA, or a model is forbidden in this lane.
cargo build \
    --locked \
    --release \
    --package riley-native \
    --no-default-features \
    --features cuda \
    --bin riley-native

grep -Eiq \
    "CUDA.*architectures[^0-9]*${RILEY_CUDA_ARCHITECTURES}|CUDA arch[^0-9]*${RILEY_CUDA_ARCHITECTURES}" \
    "$build_log"

run_host_metadata cargo test \
    --locked \
    --package riley-cuda \
    --no-default-features \
    --features cuda \
    --test abi_link \
    -- --nocapture

abi_link_binary=
abi_link_binary_count=0
for candidate in target/debug/deps/abi_link-*; do
    if [ -f "$candidate" ] && [ -x "$candidate" ]; then
        abi_link_binary=$candidate
        abi_link_binary_count=$((abi_link_binary_count + 1))
    fi
done
if [ "$abi_link_binary_count" -ne 1 ]; then
    echo "expected one abi_link test executable, found $abi_link_binary_count" >&2
    exit 1
fi

# Compile the cumulative PR 03 host-runtime, PR 04 memory, and C05 graph GPU
# integration targets into the reusable image without executing them. Device
# access is reserved for verify_python_free_gpu_runtime.sh.
cargo test \
    --locked \
    --package riley-cuda \
    --no-default-features \
    --features cuda \
    --test host_runtime_gpu \
    --no-run

cargo test \
    --locked \
    --package riley-cuda \
    --no-default-features \
    --features cuda \
    --test memory_gpu \
    --no-run

cargo test \
    --locked \
    --package riley-cuda \
    --no-default-features \
    --features cuda \
    --test graph_gpu \
    --no-run

# Compile, but never execute, the subprocess-only destructive fault gate while
# building the reusable image. Its native symbols exist only for this feature.
cargo test \
    --locked \
    --package riley-cuda \
    --no-default-features \
    --features cuda-test-fault-injection \
    --test memory_fault_injection_gpu \
    --no-run

# Compile the tensor metadata and CUDA-backed ownership adapters without
# executing any GPU target or model operation.
cargo test \
    --locked \
    --package riley-tensor \
    --no-default-features \
    --features cuda \
    --no-run

# Lint the complete CUDA-enabled Rust surface, including all ignored GPU test
# targets, without executing a device operation in this compile-only image.
cargo clippy \
    --locked \
    --package riley-cuda \
    --all-targets \
    --no-default-features \
    --features cuda \
    -- -D warnings

cargo clippy \
    --locked \
    --package riley-tensor \
    --all-targets \
    --no-default-features \
    --features cuda \
    -- -D warnings

cargo clippy \
    --locked \
    --package riley-server \
    --all-targets \
    --no-default-features \
    --features server,bench,cuda \
    -- -D warnings

cargo clippy \
    --locked \
    --package riley-native \
    --all-targets \
    --no-default-features \
    --features cuda \
    -- -D warnings

version_output=$(run_host_metadata target/release/riley --version)
printf '%s\n' "$version_output"
printf '%s\n' "$version_output" | grep -Eiq 'riley.*0\.1\.0'
printf '%s\n' "$version_output" | grep -Eiq 'cuda.*abi.*1'

# An explicit invalid root must fail clearly even when a valid nvcc is on PATH.
if env \
    CARGO_TARGET_DIR=/tmp/riley-invalid-cuda-target \
    CUDA_HOME=/definitely/missing/riley-cuda \
    CUDAToolkit_ROOT=/definitely/missing/riley-cuda \
    RILEY_CUDA_ARCHITECTURES="$RILEY_CUDA_ARCHITECTURES" \
    cargo check --locked --package riley-cuda --no-default-features --features cuda \
    >"$invalid_log" 2>&1
then
    echo "invalid CUDA toolkit root unexpectedly succeeded" >&2
    exit 1
fi
cat "$invalid_log"
grep -Eiq \
    'CUDA.*(toolkit|root|path|nvcc).*(invalid|missing|not found|does not exist|not a directory)|invalid.*CUDA' \
    "$invalid_log"

{
    printf 'artifact=%s\n' target/release/riley
    run_host_metadata ldd target/release/riley
    printf 'artifact=%s\n' target/release/riley-profile
    run_host_metadata ldd target/release/riley-profile
    printf 'artifact=%s\n' target/release/riley-native
    run_host_metadata ldd target/release/riley-native
    printf 'artifact=%s\n' "$abi_link_binary"
    run_host_metadata ldd "$abi_link_binary"
} >"$ldd_log"
cat "$ldd_log"
{
    printf 'artifact=%s\n' target/release/riley
    readelf -d target/release/riley
    printf 'artifact=%s\n' target/release/riley-profile
    readelf -d target/release/riley-profile
    printf 'artifact=%s\n' target/release/riley-native
    readelf -d target/release/riley-native
    printf 'artifact=%s\n' "$abi_link_binary"
    readelf -d "$abi_link_binary"
} >"$readelf_log"
cat "$readelf_log"
{
    printf 'artifact=%s\n' target/release/riley
    nm -D --undefined-only target/release/riley
    printf 'artifact=%s\n' target/release/riley-profile
    nm -D --undefined-only target/release/riley-profile
    printf 'artifact=%s\n' target/release/riley-native
    nm -D --undefined-only target/release/riley-native
    printf 'artifact=%s\n' "$abi_link_binary"
    nm -D --undefined-only "$abi_link_binary"
} >"$nm_log"
cat "$nm_log"
if grep -Fq 'riley_cuda_test_memory_fault_' "$nm_log"; then
    echo "production artifact unexpectedly exposes CUDA test fault injection" >&2
    exit 1
fi

if grep -Eiq '=>[[:space:]]+not found' "$ldd_log"; then
    echo "production artifact has an unresolved shared-library dependency" >&2
    exit 1
fi
grep -Eiq 'libcudart\.so' "$ldd_log" "$readelf_log"
grep -F "libcuda.so.1 => $driver_stub_runtime/libcuda.so.1" "$ldd_log"
grep -Eq 'NEEDED.*libcuda\.so\.1' "$readelf_log"
if grep -Eq '(RPATH|RUNPATH).*stubs' "$readelf_log"; then
    echo "production artifact embeds a CUDA driver stubs runtime path" >&2
    exit 1
fi

# NVML is a development-only calibration dependency. The native producer must
# declare and use it, while both shipped production/profile binaries must stay
# free of an NVML DT_NEEDED entry and undefined NVML symbols.
run_host_metadata ldd target/release/riley-native \
    | grep -F "libnvidia-ml.so.1 => $driver_stub_runtime/libnvidia-ml.so.1"
readelf -d target/release/riley-native \
    | grep -Eq 'NEEDED.*libnvidia-ml\.so\.1'
nm -D --undefined-only target/release/riley-native \
    | grep -Eq '[[:space:]]U[[:space:]]+nvml[A-Za-z0-9_]+'
for production_artifact in target/release/riley target/release/riley-profile; do
    if readelf -d "$production_artifact" | grep -Eq 'NEEDED.*libnvidia-ml\.so\.1'; then
        echo "$production_artifact unexpectedly depends on NVML" >&2
        exit 1
    fi
    if nm -D --undefined-only "$production_artifact" \
        | grep -Eq '[[:space:]]U[[:space:]]+nvml[A-Za-z0-9_]+'
    then
        echo "$production_artifact unexpectedly imports NVML symbols" >&2
        exit 1
    fi
done

if grep -Eiq 'python|pytorch|torch|transformers|triton' \
    "$ldd_log" "$readelf_log" "$nm_log" Cargo.lock
then
    echo "production artifact or Cargo lock contains a forbidden runtime dependency" >&2
    exit 1
fi

echo "Python-free CUDA production/profile/native compile, C ABI link, tensor memory, version, and dependency smoke passed"
