#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/prefill_shape_projection.cuh"

// Experimental FFN transport backend. MMA order and BF16 boundaries match V7.
// All threads participate in stage commit/wait/barriers, including inactive rows.
namespace riley_ffn_pipeline {
template<bool Gate> struct alignas(16) Stage {
  __nv_bfloat16 x[32 * 64];
  __nv_bfloat16 weight[Gate ? 1024 : 512];
};
static_assert(sizeof(Stage<true>) == 6144);
static_assert(sizeof(Stage<false>) == 5120);

__device__ __forceinline__ void copy16(void* dst, const void* src, bool valid) {
  auto address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
               :: "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}
__device__ __forceinline__ void wait() {
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}

template<int K, bool Gate>
__device__ __forceinline__ void load(Stage<Gate>& stage, const __nv_bfloat16* x,
    const __nv_bfloat16* weights, const __nv_bfloat16* up, unsigned rows, int depth) {
  for (int vector = threadIdx.x; vector < 256; vector += blockDim.x) {
    int row = vector / 8, col = (vector % 8) * 8;
    bool valid = row < rows && depth + col < K;
    // Even predicated zero-fill uses an in-bounds source address.
    const auto* src = valid ? x + row * K + depth + col : x;
    copy16(stage.x + vector * 8, src, valid);
  }
  for (int vector = threadIdx.x; vector < (Gate ? 128 : 64); vector += blockDim.x) {
    int matrix = vector / 64, offset = (vector % 64) * 8;
    const auto* base = matrix ? up : weights;
    bool valid = depth + (offset / 128) * 16 < K;
    const auto* src = valid ? base + blockIdx.x * (K / 16) * 128 + (depth / 16) * 128 + offset : base;
    copy16(stage.weight + vector * 8, src, valid);
  }
  commit();
}

__device__ __forceinline__ unsigned pair(const __nv_bfloat16* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

__global__ void gate_up(const __nv_bfloat16* x, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* out, const unsigned* active) {
  unsigned rows = *active;
  if (!rows || rows > 32) return;
  __shared__ Stage<true> stages[2];
  const int lane = threadIdx.x % 32, row = (threadIdx.x / 32) * 16 + lane / 4;
  const int t = lane % 4;
  float gd[4] = {}, ud[4] = {};
  load<576, true>(stages[0], x, gate, up, rows, 0);
  wait(); __syncthreads();
  for (int depth = 0, index = 0; depth < 576; depth += 64, index ^= 1) {
    if (depth + 64 < 576) load<576, true>(stages[index ^ 1], x, gate, up, rows, depth + 64);
    const auto& stage = stages[index];
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
      const auto* a = stage.x + row * 64 + step * 16 + 2 * t;
      const auto* b = stage.weight + step * 128 + lane * 2;
      unsigned a0 = pair(a), a1 = pair(a + 8 * 64), a2 = pair(a + 8), a3 = pair(a + 8 * 64 + 8);
      riley_prefill_shape::mma(gd, a0, a1, a2, a3, pair(b), pair(b + 64));
      riley_prefill_shape::mma(ud, a0, a1, a2, a3, pair(b + 512), pair(b + 576));
    }
    wait(); __syncthreads(); // both next-stage completion and previous-stage readers
  }
  for (int half = 0; half < 2; ++half) if (row + half * 8 < rows) {
    for (int j = 0; j < 2; ++j) {
      float g = __bfloat162float(__float2bfloat16_rn(gd[half * 2 + j]));
      float u = __bfloat162float(__float2bfloat16_rn(ud[half * 2 + j]));
      out[(row + half * 8) * 1536 + blockIdx.x * 8 + 2 * t + j] =
          __float2bfloat16_rn((g / (1.F + expf(-g))) * u);
    }
  }
}

__global__ void down_parts(const __nv_bfloat16* x, const __nv_bfloat16* weights,
    float* parts, const unsigned* active) {
  unsigned rows = *active;
  if (!rows || rows > 32) return;
  __shared__ Stage<false> stages[2];
  const int lane = threadIdx.x, g = lane / 4, t = lane % 4;
  const int begin = blockIdx.y * 320, end = min(begin + 320, 1536);
  float d[2][4] = {};
  load<1536, false>(stages[0], x, weights, weights, rows, begin);
  wait(); __syncthreads();
  for (int depth = begin, index = 0; depth < end; depth += 64, index ^= 1) {
    if (depth + 64 < end) load<1536, false>(stages[index ^ 1], x, weights, weights, rows, depth + 64);
    const auto& stage = stages[index];
    #pragma unroll
    for (int step = 0; step < 4; ++step) if (depth + step * 16 < end) {
      const auto* b = stage.weight + step * 128 + lane * 2;
      #pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
        const auto* a = stage.x + (g + tile * 16) * 64 + step * 16 + 2 * t;
        riley_prefill_shape::mma(d[tile], pair(a), pair(a + 8 * 64), pair(a + 8),
            pair(a + 8 * 64 + 8), pair(b), pair(b + 64));
      }
    }
    wait(); __syncthreads();
  }
  #pragma unroll
  for (int tile = 0; tile < 2; ++tile) for (int half = 0; half < 2; ++half) {
    int row = tile * 16 + g + half * 8;
    if (row < rows) for (int j = 0; j < 2; ++j) {
      auto value = __float2bfloat16_rn(d[tile][half * 2 + j]);
      parts[(blockIdx.y * 32 + row) * 576 + blockIdx.x * 8 + 2 * t + j] = __bfloat162float(value);
    }
  }
}
} // namespace riley_ffn_pipeline
