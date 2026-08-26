#include "attention_online.hpp"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kWarpsPerBlock = 8;
constexpr uint32_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr uint32_t kHeadSize = 64;
constexpr uint32_t kKeyTileSize = 32;
constexpr uint32_t kKeyTileElements = kKeyTileSize * kHeadSize;
constexpr uint32_t kFullWarpMask = 0xffffffffU;

static_assert(kThreadsPerBlock == 256,
              "online prefill launch geometry changed");
static_assert(kHeadSize == 2 * kWarpSize,
              "each lane must own exactly two output elements");

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ void update_online_state(
    float score, float* maximum, float* denominator) {
  if (isnan(score) || isnan(*maximum)) {
    *maximum = CUDART_NAN_F;
    *denominator = CUDART_NAN_F;
    return;
  }

  const bool score_is_positive_infinity = isinf(score) && score > 0.0F;
  const bool maximum_is_positive_infinity =
      isinf(*maximum) && *maximum > 0.0F;
  if (score_is_positive_infinity) {
    if (maximum_is_positive_infinity) {
      *denominator += 1.0F;
    } else {
      *maximum = CUDART_INF_F;
      *denominator = 1.0F;
    }
    return;
  }
  if (maximum_is_positive_infinity ||
      (isinf(score) && score < 0.0F)) {
    return;
  }

  const float next_maximum = fmaxf(*maximum, score);
  const float alpha = *denominator == 0.0F
                          ? 0.0F
                          : expf(*maximum - next_maximum);
  const float beta = expf(score - next_maximum);
  *denominator = fmaf(alpha, *denominator, beta);
  *maximum = next_maximum;
}

__device__ __forceinline__ float stage_bf16_scaled_score(float dot_product,
                                                          float scale) {
  // Preserve the established BF16 attention-score contract while keeping the
  // score matrix virtual: the materialized backend rounds once after QK and
  // once after scaling. No staged value is written to global memory here.
  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(staged_dot) * scale));
}

__device__ __forceinline__ float staged_warp_tree_score(
    float query_low, float query_high, float key_low, float key_high,
    float scale) {
  float score = fmaf(query_low, key_low, query_high * key_high);
  score = warp_sum(score);
  return stage_bf16_scaled_score(
      __shfl_sync(kFullWarpMask, score, 0), scale);
}

__device__ __forceinline__ float stage_bf16_probability(
    float score, float maximum, float denominator) {
  float probability = 0.0F;
  if (isnan(score) || isnan(maximum) || isnan(denominator)) {
    probability = CUDART_NAN_F;
  } else if (isinf(maximum) && maximum > 0.0F) {
    // Preserve the online state's equal weighting of every +Inf maximum.
    probability = isinf(score) && score > 0.0F
                      ? 1.0F / denominator
                      : 0.0F;
  } else if (denominator > 0.0F) {
    probability = expf(score - maximum) / denominator;
  }
  return __bfloat162float(__float2bfloat16_rn(probability));
}

__device__ __forceinline__ float accumulate_staged_probability(
    float accumulator, float probability, float value) {
  // Preserve the prior online path's zero-weight handling for infinite values.
  return probability == 0.0F ? accumulator
                             : fmaf(probability, value, accumulator);
}

__global__ __launch_bounds__(kThreadsPerBlock) void online_bf16_gqa_prefill(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* output,
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale,
    bool causal_local, uint64_t local_window_size) {
  __shared__ __nv_bfloat16 key_tile[kKeyTileElements];
  __shared__ __nv_bfloat16 value_tile[kKeyTileElements];

  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint64_t batch = blockIdx.z;
  const uint64_t query_head = blockIdx.y;
  const uint64_t query_tile_start =
      static_cast<uint64_t>(blockIdx.x) * kWarpsPerBlock;
  const uint64_t query_token = query_tile_start + warp;
  const bool active = query_token < token_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;

  float query_low = 0.0F;
  float query_high = 0.0F;
  if (active) {
    const uint64_t query_base =
        ((batch * token_count + query_token) * query_head_count +
         query_head) *
        kHeadSize;
    query_low = __bfloat162float(query[query_base + lane]);
    query_high =
        __bfloat162float(query[query_base + lane + kWarpSize]);
  }

  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;

  const uint64_t active_query_count =
      token_count - query_tile_start < kWarpsPerBlock
          ? token_count - query_tile_start
          : kWarpsPerBlock;
  const uint64_t maximum_query =
      query_tile_start + active_query_count - 1;

  // Pass one preserves the existing warp-tree QK staging and F32 online
  // maximum/denominator recurrence; values are deliberately not consumed.
  for (uint64_t key_tile_start = 0; key_tile_start <= maximum_query;
       key_tile_start += kKeyTileSize) {
    const uint64_t remaining_keys = token_count - key_tile_start;
    const uint64_t key_tile_count =
        remaining_keys < kKeyTileSize ? remaining_keys : kKeyTileSize;
    const uint64_t tile_elements = key_tile_count * kHeadSize;
    for (uint64_t tile_index = threadIdx.x; tile_index < tile_elements;
         tile_index += kThreadsPerBlock) {
      const uint64_t key_offset = tile_index / kHeadSize;
      const uint64_t depth = tile_index % kHeadSize;
      const uint64_t key_token = key_tile_start + key_offset;
      const uint64_t key_value_index =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
              kHeadSize +
          depth;
      key_tile[tile_index] = key[key_value_index];
    }
    __syncthreads();

    if (active && (!causal_local || local_window_size != 0)) {
      const uint64_t minimum_key =
          causal_local && query_token + 1 > local_window_size
              ? query_token + 1 - local_window_size
              : 0;
      const uint64_t tile_end = key_tile_start + key_tile_count;
      const uint64_t key_begin =
          key_tile_start > minimum_key ? key_tile_start : minimum_key;
      const uint64_t causal_end = query_token + 1;
      const uint64_t key_end = tile_end < causal_end ? tile_end : causal_end;

      for (uint64_t key_token = key_begin; key_token < key_end;
           ++key_token) {
        const uint64_t tile_base =
            (key_token - key_tile_start) * kHeadSize;
        const float key_low =
            __bfloat162float(key_tile[tile_base + lane]);
        const float key_high =
            __bfloat162float(key_tile[tile_base + lane + kWarpSize]);
        const float score = staged_warp_tree_score(
            query_low, query_high, key_low, key_high, scale);

        if (lane == 0) {
          update_online_state(score, &maximum, &denominator);
        }
      }
    }
    __syncthreads();
  }

  maximum = __shfl_sync(kFullWarpMask, maximum, 0);
  denominator = __shfl_sync(kFullWarpMask, denominator, 0);
  float accumulator_low = 0.0F;
  float accumulator_high = 0.0F;

  // Recompute staged scores after the final online normalizer is known. Each
  // normalized probability is narrowed to BF16 before logical key-order AV,
  // without writing an HBM score/probability matrix.
  for (uint64_t key_tile_start = 0; key_tile_start <= maximum_query;
       key_tile_start += kKeyTileSize) {
    const uint64_t remaining_keys = token_count - key_tile_start;
    const uint64_t key_tile_count =
        remaining_keys < kKeyTileSize ? remaining_keys : kKeyTileSize;
    const uint64_t tile_elements = key_tile_count * kHeadSize;
    for (uint64_t tile_index = threadIdx.x; tile_index < tile_elements;
         tile_index += kThreadsPerBlock) {
      const uint64_t key_offset = tile_index / kHeadSize;
      const uint64_t depth = tile_index % kHeadSize;
      const uint64_t key_token = key_tile_start + key_offset;
      const uint64_t key_value_index =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
              kHeadSize +
          depth;
      key_tile[tile_index] = key[key_value_index];
      value_tile[tile_index] = value[key_value_index];
    }
    __syncthreads();

    if (active && (!causal_local || local_window_size != 0)) {
      const uint64_t minimum_key =
          causal_local && query_token + 1 > local_window_size
              ? query_token + 1 - local_window_size
              : 0;
      const uint64_t tile_end = key_tile_start + key_tile_count;
      const uint64_t key_begin =
          key_tile_start > minimum_key ? key_tile_start : minimum_key;
      const uint64_t causal_end = query_token + 1;
      const uint64_t key_end = tile_end < causal_end ? tile_end : causal_end;

      for (uint64_t key_token = key_begin; key_token < key_end;
           ++key_token) {
        const uint64_t tile_base =
            (key_token - key_tile_start) * kHeadSize;
        const float key_low =
            __bfloat162float(key_tile[tile_base + lane]);
        const float key_high =
            __bfloat162float(key_tile[tile_base + lane + kWarpSize]);
        const float score = staged_warp_tree_score(
            query_low, query_high, key_low, key_high, scale);

        float probability = 0.0F;
        if (lane == 0) {
          probability =
              stage_bf16_probability(score, maximum, denominator);
        }
        probability = __shfl_sync(kFullWarpMask, probability, 0);
        const float value_low =
            __bfloat162float(value_tile[tile_base + lane]);
        const float value_high =
            __bfloat162float(value_tile[tile_base + lane + kWarpSize]);
        accumulator_low = accumulate_staged_probability(
            accumulator_low, probability, value_low);
        accumulator_high = accumulate_staged_probability(
            accumulator_high, probability, value_high);
      }
    }
    __syncthreads();
  }

  if (active) {
    const uint64_t output_base =
        ((batch * token_count + query_token) * query_head_count +
         query_head) *
        kHeadSize;
    output[output_base + lane] =
        __float2bfloat16_rn(accumulator_low);
    output[output_base + lane + kWarpSize] =
        __float2bfloat16_rn(accumulator_high);
  }
}

}  // namespace

cudaError_t rustinfer_cuda_attention_online::launch_bf16_gqa_prefill(
    const void* query, const void* key, const void* value, void* output,
    uint64_t batch_count, uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale, bool causal_local,
    uint64_t local_window_size, cudaStream_t stream) noexcept {
  const uint64_t query_tile_count =
      ((token_count - 1) / kWarpsPerBlock) + 1;
  const dim3 grid(static_cast<uint32_t>(query_tile_count),
                  static_cast<uint32_t>(query_head_count),
                  static_cast<uint32_t>(batch_count));
  online_bf16_gqa_prefill<<<grid, kThreadsPerBlock, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(output), token_count, query_head_count,
      key_value_head_count, scale, causal_local, local_window_size);
  return cudaGetLastError();
}
