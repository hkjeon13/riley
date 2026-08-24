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
: "${RUSTINFER_CUDA_ARCHITECTURES:?RUSTINFER_CUDA_ARCHITECTURES must be explicit}"

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
    rm -rf -- /tmp/rustinfer-invalid-cuda-target
}
trap cleanup EXIT HUP INT TERM

# PR 03 links the Driver API, while this compile-only image deliberately has no
# host driver injection. Supply the toolkit stub under its SONAME only for the
# two metadata-only executables below. Device/context calls remain forbidden;
# the separate GPU verifier requires the real injected libcuda.so.1.
driver_stub_linker="$CUDA_HOME/lib64/stubs/libcuda.so"
test -f "$driver_stub_linker"
ln -s "$driver_stub_linker" "$driver_stub_runtime/libcuda.so.1"
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
cargo build --release --features cuda,server

grep -Eiq \
    "CUDA.*architectures[^0-9]*${RUSTINFER_CUDA_ARCHITECTURES}|CUDA arch[^0-9]*${RUSTINFER_CUDA_ARCHITECTURES}" \
    "$build_log"

run_host_metadata cargo test \
    --locked \
    --package rustinfer-cuda \
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

# Compile the PR 03 GPU integration target into the reusable image without
# executing it. Device access is reserved for verify_python_free_gpu_runtime.sh.
cargo test \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test host_runtime_gpu \
    --no-run

version_output=$(run_host_metadata target/release/rustinfer --version)
printf '%s\n' "$version_output"
printf '%s\n' "$version_output" | grep -Eiq 'rustinfer.*0\.1\.0'
printf '%s\n' "$version_output" | grep -Eiq 'cuda.*abi.*1'

# An explicit invalid root must fail clearly even when a valid nvcc is on PATH.
if env \
    CARGO_TARGET_DIR=/tmp/rustinfer-invalid-cuda-target \
    CUDA_HOME=/definitely/missing/rustinfer-cuda \
    CUDAToolkit_ROOT=/definitely/missing/rustinfer-cuda \
    RUSTINFER_CUDA_ARCHITECTURES="$RUSTINFER_CUDA_ARCHITECTURES" \
    cargo check --locked --package rustinfer-cuda --no-default-features --features cuda \
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
    printf 'artifact=%s\n' target/release/rustinfer
    run_host_metadata ldd target/release/rustinfer
    printf 'artifact=%s\n' "$abi_link_binary"
    run_host_metadata ldd "$abi_link_binary"
} >"$ldd_log"
cat "$ldd_log"
{
    printf 'artifact=%s\n' target/release/rustinfer
    readelf -d target/release/rustinfer
    printf 'artifact=%s\n' "$abi_link_binary"
    readelf -d "$abi_link_binary"
} >"$readelf_log"
cat "$readelf_log"
{
    printf 'artifact=%s\n' target/release/rustinfer
    nm -D --undefined-only target/release/rustinfer
    printf 'artifact=%s\n' "$abi_link_binary"
    nm -D --undefined-only "$abi_link_binary"
} >"$nm_log"
cat "$nm_log"

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

if grep -Eiq 'python|pytorch|torch|transformers|triton' \
    "$ldd_log" "$readelf_log" "$nm_log" Cargo.lock
then
    echo "production artifact or Cargo lock contains a forbidden runtime dependency" >&2
    exit 1
fi

echo "Python-free CUDA compile, C ABI link, version, and dependency smoke passed"
