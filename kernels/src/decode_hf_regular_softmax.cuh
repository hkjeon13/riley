// Preserved prefill regular reduction for the bounded Qwen M1 diagnostic.
#pragma once
#include <cfloat>
#include <cuda_bf16.h>
namespace {
template <bool Maximum>
__device__ __forceinline__ float decode_hf_block_reduce(float value,
                                                 float* shared) {
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t warp = threadIdx.x / 32;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffffU, value, offset);
    if constexpr (Maximum) {
      value = value < other ? other : value;
    } else {
      value += other;
    }
  }
  __syncthreads();
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32
              ? shared[lane]
              : (Maximum ? -FLT_MAX : 0.0F);
  if (warp == 0) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffU, value, offset);
      if constexpr (Maximum) {
        value = value < other ? other : value;
      } else {
        value += other;
      }
    }
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

template <int RegisterCount>
__global__ void decode_hf_regular_softmax_kernel(__nv_bfloat16* scores,
                                          uint64_t token_count) {
  __shared__ float reduction[32];
  __nv_bfloat16 values[RegisterCount];
  const uint64_t row_offset = static_cast<uint64_t>(blockIdx.x) * token_count;
  float thread_maximum = -FLT_MAX;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      values[register_index] = scores[row_offset + column];
      const float value = __bfloat162float(values[register_index]);
      thread_maximum = thread_maximum < value ? value : thread_maximum;
    }
  }
  const float maximum = decode_hf_block_reduce<true>(thread_maximum, reduction);
  float thread_sum = 0.0F;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      thread_sum +=
          expf(__bfloat162float(values[register_index]) - maximum);
    }
  }
  const float sum = decode_hf_block_reduce<false>(thread_sum, reduction);
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      scores[row_offset + column] = __float2bfloat16_rn(
          expf(__bfloat162float(values[register_index]) - maximum) / sum);
    }
  }
}

} // namespace
