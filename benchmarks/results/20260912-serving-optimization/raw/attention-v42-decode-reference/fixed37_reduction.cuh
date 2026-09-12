#ifndef RILEY_CUDA_FIXED37_REDUCTION_CUH_
#define RILEY_CUDA_FIXED37_REDUCTION_CUH_

#include "riley_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace riley_cuda_fixed37 {

constexpr uint64_t kChunkElements =
    RILEY_CUDA_FIXED37_CHUNK_ELEMENTS;
constexpr uint64_t kMaximumChunkCount =
    RILEY_CUDA_FIXED37_MAX_CHUNK_COUNT;
constexpr uint32_t kThreadsPerBlock = 256;
constexpr uint32_t kMaximumBlocks = 65535;

constexpr uint64_t chunk_count(uint64_t element_count) noexcept {
  return element_count == 0
             ? 0
             : ((element_count - 1) / kChunkElements) + 1;
}

constexpr uint64_t shared_bytes(uint64_t element_count) noexcept {
  return chunk_count(element_count) * 2 * sizeof(float);
}

constexpr uint32_t block_count(uint64_t work_items) noexcept {
  if (work_items == 0) {
    return 0;
  }
  return static_cast<uint32_t>(work_items < kMaximumBlocks
                                   ? work_items
                                   : kMaximumBlocks);
}

// Every caller initializes `first[0..partial_count)` before entering. The two
// arrays alternate so no merge destination can race a source still needed by
// an adjacent pair at the same level. All threads in the block must call this
// function with the same arguments. If a caller reuses either partial array
// after reading the result, it must synchronize the block once more first.
__device__ __forceinline__ float balanced_sum(float* first, float* second,
                                               uint64_t partial_count) {
  float* source = first;
  float* destination = second;
  uint64_t active = partial_count;
  while (active > 1) {
    const uint64_t pair_count = active / 2;
    for (uint64_t pair = threadIdx.x; pair < pair_count;
         pair += blockDim.x) {
      destination[pair] =
          __fadd_rn(source[pair * 2], source[pair * 2 + 1]);
    }
    if ((active & 1U) != 0U && threadIdx.x == 0) {
      destination[pair_count] = source[active - 1];
    }
    __syncthreads();
    float* temporary = source;
    source = destination;
    destination = temporary;
    active = pair_count + (active & 1U);
  }
  return source[0];
}

__device__ __forceinline__ float balanced_max(float* first, float* second,
                                               uint64_t partial_count) {
  float* source = first;
  float* destination = second;
  uint64_t active = partial_count;
  while (active > 1) {
    const uint64_t pair_count = active / 2;
    for (uint64_t pair = threadIdx.x; pair < pair_count;
         pair += blockDim.x) {
      destination[pair] = fmaxf(source[pair * 2], source[pair * 2 + 1]);
    }
    if ((active & 1U) != 0U && threadIdx.x == 0) {
      destination[pair_count] = source[active - 1];
    }
    __syncthreads();
    float* temporary = source;
    source = destination;
    destination = temporary;
    active = pair_count + (active & 1U);
  }
  return source[0];
}

}  // namespace riley_cuda_fixed37

#endif  // RILEY_CUDA_FIXED37_REDUCTION_CUH_
