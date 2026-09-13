#!/usr/bin/env bash
set -euo pipefail
root=${1:?prepared root}; evidence=${2:?new output directory}
mkdir "$evidence"
cd "$root/source"
export LD_LIBRARY_PATH="$root/toolchain130/nvidia/cu13/lib"
nvcc="$root/toolchain130/nvidia/cu13/bin/nvcc"
args=(-std=c++17 -O3 -lineinfo -Xptxas=-v)
timeout 180 "$nvcc" "${args[@]}" -arch=sm_89 benchmarks/analysis/query_reuse_mixed_probe.cu -o "$evidence/probe" > "$evidence/build.log" 2>&1
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]
timeout 180 "$evidence/probe" --check-only > "$evidence/check.log" 2>&1
timeout 180 "$evidence/probe" > "$evidence/timing.log" 2>&1
timeout 180 "$evidence/probe" --serving-shapes > "$evidence/serving-shapes.log" 2>&1
for tool in memcheck racecheck; do
 timeout 180 /data/cuda-12.8.1/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 "$evidence/probe" --sanitizer-small > "$evidence/$tool.log" 2>&1
done
for arch in sm_90a sm_100a; do
 timeout 180 "$nvcc" "${args[@]}" -arch="$arch" -c benchmarks/analysis/query_reuse_mixed_probe.cu -o "$evidence/$arch.o" > "$evidence/$arch-build.log" 2>&1
done
sha256sum benchmarks/analysis/query_reuse_mixed_probe.cu kernels/optional/query_reuse_mixed_attention.cuh kernels/optional/compact_mixed_attention.cuh kernels/src/mixed_attention_v49.cuh > "$evidence/source-sha256.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$evidence/compute-after.csv"
echo complete > "$evidence/complete.txt"
