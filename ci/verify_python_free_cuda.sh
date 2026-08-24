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

rustc --version | grep -Eq '^rustc 1\.85\.0 '
cargo --version | grep -Eq '^cargo 1\.85\.0 '
uname -m | grep -Eq '^x86_64$'
"$CUDA_HOME/bin/nvcc" --version

build_log=$(mktemp)
invalid_log=$(mktemp)
ldd_log=$(mktemp)
readelf_log=$(mktemp)
nm_log=$(mktemp)
cleanup() {
    rm -f -- "$build_log" "$invalid_log" "$ldd_log" "$readelf_log" "$nm_log"
    rm -rf -- /tmp/rustinfer-invalid-cuda-target
}
trap cleanup EXIT HUP INT TERM

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

cargo test \
    --locked \
    --package rustinfer-cuda \
    --no-default-features \
    --features cuda \
    --test abi_link \
    -- --nocapture

version_output=$(target/release/rustinfer --version)
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

ldd target/release/rustinfer >"$ldd_log"
cat "$ldd_log"
readelf -d target/release/rustinfer >"$readelf_log"
cat "$readelf_log"
nm -D --undefined-only target/release/rustinfer >"$nm_log"
cat "$nm_log"

if grep -Eiq 'python|pytorch|torch|transformers|triton' \
    "$ldd_log" "$readelf_log" "$nm_log" Cargo.lock
then
    echo "production artifact or Cargo lock contains a forbidden runtime dependency" >&2
    exit 1
fi

echo "Python-free CUDA compile, C ABI link, version, and dependency smoke passed"
