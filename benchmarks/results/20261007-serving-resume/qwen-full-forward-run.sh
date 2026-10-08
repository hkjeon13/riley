#!/bin/sh
set -eu

artifact_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo=/data/riley-serving-260915-n01
checkpoint=/data/riley-models/Qwen2.5-3B-Instruct-aa8e72537993ba99e69dfaafa59ed015b17504d1
workload=/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/inputs/qwen3b-c8-p2048-o128.json
teacher_dir=/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-hf-eager-teacher-forced-r1-20260916T002940Z
stage_dir=/data/riley-benchmarks/20260915T134348Z-n06a-shared-host/qwen3b-p2048-cacheon-layer-stage-r7-20260917T141511Z
output="$artifact_dir/qwen3b-p2048-cache-on-m1-cublas-qk-av-full-forward-trace.json"
expected_revision=1011691e699bb0051c95878504f67ad20ed6d637

test -d "$repo"
test -d "$checkpoint"
test -f "$workload"
test -f "$teacher_dir/teacher-forced-oracle.json"
test -f "$teacher_dir/cache-off-logits.safetensors"
test -f "$teacher_dir/cache-on-logits.safetensors"
test -f "$stage_dir/qwen3b-p2048-cache-on-layer-stage.json"
test -f "$stage_dir/qwen3b-p2048-cache-on-layer-stage.safetensors"
test ! -e "$output"
test "$(git -C "$repo" branch --show-current)" = main
test "$(git -C "$repo" rev-parse HEAD)" = "$expected_revision"
test -z "$(git -C "$repo" status --porcelain=v1 --untracked-files=all -- kernels/include/riley_cuda.h kernels/src/decode_attention.cu crates/riley-cuda/src/ffi.rs crates/riley-cuda/src/decode.rs crates/riley-runtime/src/llama/decode.rs crates/riley-runtime/tests/qwen3b_p2051_full_forward_gpu.rs)"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$artifact_dir/gpu-before.csv"
for resource in cpu io memory; do
  cat "/proc/pressure/$resource" > "$artifact_dir/pressure-before-$resource.txt"
done

set +e
(
  cd "$repo"
  export PATH=/home/psyche/.cargo/bin:/data/cuda-12.8.1/bin:/data/cmake-3.31.12/bin:$PATH
  export CUDA_HOME=/data/cuda-12.8.1
  export CUDAToolkit_ROOT=/data/cuda-12.8.1
  export CUDACXX=/data/cuda-12.8.1/bin/nvcc
  export CMAKE=/data/cmake-3.31.12/bin/cmake
  export CARGO_BUILD_JOBS=1
  export RUSTUP_TOOLCHAIN=1.85.0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export RILEY_QWEN_SERVING_WORKLOAD="$workload"
  export RILEY_QWEN_HF_TEACHER_FORCED_ORACLE="$teacher_dir/teacher-forced-oracle.json"
  export RILEY_QWEN_HF_CACHE_OFF_SIDECAR="$teacher_dir/cache-off-logits.safetensors"
  export RILEY_QWEN_HF_CACHE_ON_SIDECAR="$teacher_dir/cache-on-logits.safetensors"
  export RILEY_QWEN3B_P2048_CACHE_ON_STAGE_MANIFEST="$stage_dir/qwen3b-p2048-cache-on-layer-stage.json"
  export RILEY_QWEN3B_P2048_CACHE_ON_STAGE_SIDECAR="$stage_dir/qwen3b-p2048-cache-on-layer-stage.safetensors"
  export RILEY_QWEN3B_CHECKPOINT="$checkpoint"
  export RILEY_QWEN3B_P2048_CACHE_ON_M1_CUBLAS_QK_AV_STAGE_OUTPUT="$output"
  cargo test -p riley-runtime --features cuda,cuda-cublas-gemm-probe --test qwen3b_p2051_full_forward_gpu -- --ignored --exact qwen3b_p2048_hf_compatible_cache_on_m1_cublas_qk_av_full_forward_candidate_trace --nocapture
) > "$artifact_dir/m1-cublas-qk-av-full-forward.log" 2>&1
test_exit=$?
set -e

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$artifact_dir/gpu-after.csv"
for resource in cpu io memory; do
  cat "/proc/pressure/$resource" > "$artifact_dir/pressure-after-$resource.txt"
done

TEST_EXIT="$test_exit" ARTIFACT_DIR="$artifact_dir" SOURCE_REVISION="$expected_revision" python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ARTIFACT_DIR"])
files = sorted(path for path in root.iterdir() if path.is_file() and path.name not in {"run.json", "SHA256SUMS"})
record = {
    "schema_version": "riley.qwen3b-p2048-cache-on-m1-cublas-qk-av-full-forward-run.v1",
    "performance_claim_eligible": False,
    "source_revision": os.environ["SOURCE_REVISION"],
    "test_exit_code": int(os.environ["TEST_EXIT"]),
    "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
}
payload = (json.dumps(record, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
fd = os.open(root / "run.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "wb") as stream:
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
PY
(
  cd "$artifact_dir"
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | LC_ALL=C sort | while IFS= read -r file; do sha256sum "$file"; done > SHA256SUMS
  sha256sum -c SHA256SUMS
)
exit "$test_exit"
