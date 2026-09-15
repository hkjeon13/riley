#!/usr/bin/env bash
# Caller retains exclusive test-GPU ownership. Blender remains stopped under
# the leave_stopped_no_restore lifecycle policy. Do not stop the static public web viewers.
set -euo pipefail
root=${1:?prepared root}; evidence=${2:?new evidence directory}
mkdir "$evidence"
cd "$root/source"
export LD_LIBRARY_PATH="$root/toolchain130/nvidia/cu13/lib"
nvcc="$root/toolchain130/nvidia/cu13/bin/nvcc"
args=(-std=c++17 -O3 -lineinfo -Xptxas=-v)
timeout 180 "$nvcc" "${args[@]}" -arch=sm_89 benchmarks/analysis/gqa_staged_mixed_probe.cu -o "$evidence/probe" > "$evidence/build.log" 2>&1
if pgrep -x blender >/dev/null; then echo "Blender must remain stopped for this lifecycle" >&2; exit 1; fi
for mode in check-only serving-shapes; do
 name=$mode; [[ $mode != check-only ]] || name=check
 timeout 180 "$evidence/probe" "--$mode" > "$evidence/$name.log" 2>&1
done
for tool in memcheck racecheck; do
 timeout 180 /data/cuda-12.8.1/bin/compute-sanitizer --tool "$tool" --error-exitcode 99 "$evidence/probe" --sanitizer-small > "$evidence/$tool.log" 2>&1
done
for arch in sm_90a sm_100a; do
 timeout 180 "$nvcc" "${args[@]}" -arch="$arch" -c benchmarks/analysis/gqa_staged_mixed_probe.cu -o "$evidence/$arch.o" > "$evidence/$arch-build.log" 2>&1
done
sha256sum benchmarks/analysis/gqa_staged_mixed_probe.cu kernels/optional/gqa_staged_mixed_attention.cuh kernels/optional/compact_mixed_attention.cuh kernels/src/mixed_attention_v49.cuh > "$evidence/source-sha256.txt"
echo complete > "$evidence/complete.txt"
