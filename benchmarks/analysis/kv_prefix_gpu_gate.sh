#!/usr/bin/env bash
# Run on the existing remote recovery workspace. Offline harness only; the
# serving/runtime path remains Rust -> C ABI -> CUDA.
set -euo pipefail
run_root=${1:?recovery workspace root required}
evidence=${2:?new absolute evidence directory required}
mkdir "$evidence"
export PATH="/home/psyche/.cargo/bin:$run_root/toolchain130/nvidia/cu13/bin:/usr/bin:/bin"
export CUDA_HOME="$run_root/toolchain130/nvidia/cu13"
export CUDAToolkit_ROOT="$CUDA_HOME"
export CMAKE=/data/cmake-3.31.12/bin/cmake
export CMAKE_BUILD_PARALLEL_LEVEL=4 CARGO_BUILD_JOBS=4
export CARGO_TARGET_DIR="$run_root/target"
export RILEY_FA3_SOURCE="$run_root/deps/flash-attention"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/data/riley-vllm-interim.CfrT9T/venv/lib/python3.13/site-packages/nvidia/cu13/lib"
cd "$run_root/source"
nvidia-smi > "$evidence/gpu-before.txt"
timeout 1800 cargo test -p riley-runtime --features cuda-test-fault-injection \
  --test kv_prefix_copy_gpu --no-run --message-format=json \
  > "$evidence/build.jsonl" 2> "$evidence/build.log"
test_binary=$(/usr/bin/python3 - "$evidence/build.jsonl" <<'PY'
import json, sys
paths = set()
for line in open(sys.argv[1]):
    event = json.loads(line)
    if event.get('reason') == 'compiler-artifact' and event.get('target', {}).get('name') == 'kv_prefix_copy_gpu' and event.get('executable'):
        paths.add(event['executable'])
assert len(paths) == 1, paths
print(paths.pop())
PY
)
printf '%s\n' "$test_binary" > "$evidence/test-binary.txt"
sha256sum "$test_binary" > "$evidence/test-binary.sha256"
timeout 90 "$test_binary" --ignored --nocapture --test-threads=1 > "$evidence/gpu.log" 2>&1
# The ambiguity subprocess intentionally retains allocations. It runs above;
# the zero-leak sanitizer scope covers publish/cancel/orphan/partial failure.
timeout 90 /data/cuda-12.8.1/bin/compute-sanitizer --tool memcheck \
  --leak-check full --error-exitcode 97 "$test_binary" --ignored --nocapture \
  --test-threads=1 --skip kv_page_ambiguous_completion_retains_all_holds \
  > "$evidence/memcheck.log" 2>&1
# Nsight originals can contain environment credentials. Keep them outside the
# export directory, profile with a minimal environment, and publish only a
# freshly constructed numeric/API-name SQLite subset.
trace_private=$(mktemp -d "$run_root/kv-prefix-trace-private.XXXXXX")
timeout 90 env -i PATH="$PATH" HOME="$HOME" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  /data/cuda-12.8.1/bin/nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --output="$trace_private/copy" "$test_binary" --ignored --nocapture --test-threads=1 \
  --skip kv_page_ambiguous_completion_retains_all_holds > "$evidence/trace.log" 2>&1
/data/cuda-12.8.1/bin/nsys export --type sqlite --output="$trace_private/copy.sqlite" \
  "$trace_private/copy.nsys-rep" > "$evidence/export.log" 2>&1
/usr/bin/python3 benchmarks/analysis/sanitize_kv_trace.py "$trace_private/copy.sqlite" \
  "$evidence/copy.sqlite" > "$evidence/sanitization.log"
# Existing H2D/D2H owners use the same native completion/close functions.
timeout 1800 cargo test -p riley-cuda --features cuda-test-fault-injection \
  --test memory_gpu -- --ignored --nocapture --test-threads=1 \
  > "$evidence/memory-regression.log" 2>&1
sha256sum kernels/src/memory.cu kernels/src/ffi_internal.hpp kernels/include/riley_cuda.h \
  crates/riley-cuda/src/ffi.rs crates/riley-cuda/src/lib.rs crates/riley-cuda/src/memory.rs \
  crates/riley-cuda/src/memory/kv_copy.rs crates/riley-runtime/Cargo.toml \
  crates/riley-runtime/src/paged_kv.rs crates/riley-runtime/src/paged_kv/prefix.rs \
  crates/riley-runtime/src/paged_kv/prefix/cuda.rs \
  crates/riley-runtime/tests/kv_prefix_copy_gpu.rs benchmarks/analysis/kv_prefix_gpu_gate.sh \
  benchmarks/analysis/sanitize_kv_trace.py \
  > "$evidence/source.sha256"
nvidia-smi > "$evidence/gpu-after.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
  > "$evidence/compute-after.txt"
printf 'complete\n' > "$evidence/completion.txt"
tail -6 "$evidence/gpu.log"
tail -5 "$evidence/memcheck.log"
tail -4 "$evidence/memory-regression.log"
