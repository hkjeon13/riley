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
    float score, float* maximum, float* denominator, float* alpha,
    float* beta) {
  if (isnan(score) || isnan(*maximum)) {
    *maximum = CUDART_NAN_F;
    *denominator = CUDART_NAN_F;
    *alpha = CUDART_NAN_F;
    *beta = CUDART_NAN_F;
    return;
  }

  const bool score_is_positive_infinity = isinf(score) && score > 0.0F;
  const bool maximum_is_positive_infinity =
      isinf(*maximum) && *maximum > 0.0F;
  if (score_is_positive_infinity) {
    if (maximum_is_positive_infinity) {
      *alpha = 1.0F;
      *beta = 1.0F;
      *denominator += 1.0F;
    } else {
      *maximum = CUDART_INF_F;
      *denominator = 1.0F;
      *alpha = 0.0F;
      *beta = 1.0F;
    }
    return;
  }
  if (maximum_is_positive_infinity ||
      (isinf(score) && score < 0.0F)) {
    *alpha = 1.0F;
    *beta = 0.0F;
    return;
  }

  const float next_maximum = fmaxf(*maximum, score);
  *alpha = *denominator == 0.0F
               ? 0.0F
               : expf(*maximum - next_maximum);
  *beta = expf(score - next_maximum);
  *denominator = fmaf(*alpha, *denominator, *beta);
  *maximum = next_maximum;
}

__device__ __forceinline__ float update_numerator(float numerator,
                                                   float value, float alpha,
                                                   float beta) {
  // Avoid 0*Inf producing NaN for entries whose online weight is exactly zero.
  if (beta == 0.0F) {
    return alpha * numerator;
  }
  if (alpha == 0.0F) {
    return beta * value;
  }
  return fmaf(beta, value, alpha * numerator);
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
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;

  const uint64_t active_query_count =
      token_count - query_tile_start < kWarpsPerBlock
          ? token_count - query_tile_start
          : kWarpsPerBlock;
  const uint64_t maximum_query =
      query_tile_start + active_query_count - 1;

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
        float score = fmaf(query_low, key_low, query_high * key_high);
        score = warp_sum(score);
        score = stage_bf16_scaled_score(
            __shfl_sync(kFullWarpMask, score, 0), scale);

        float alpha = 0.0F;
        float beta = 0.0F;
        if (lane == 0) {
          update_online_state(score, &maximum, &denominator, &alpha, &beta);
        }
        alpha = __shfl_sync(kFullWarpMask, alpha, 0);
        beta = __shfl_sync(kFullWarpMask, beta, 0);

        const float value_low =
            __bfloat162float(value_tile[tile_base + lane]);
        const float value_high =
            __bfloat162float(value_tile[tile_base + lane + kWarpSize]);
        numerator_low =
            update_numerator(numerator_low, value_low, alpha, beta);
        numerator_high =
            update_numerator(numerator_high, value_high, alpha, beta);
      }
    }
    __syncthreads();
  }

  if (active) {
    float inverse_denominator = 0.0F;
    if (lane == 0) {
      inverse_denominator = isnan(denominator)
                                ? CUDART_NAN_F
                                : (denominator > 0.0F
                                       ? 1.0F / denominator
                                       : 0.0F);
    }
    inverse_denominator =
        __shfl_sync(kFullWarpMask, inverse_denominator, 0);
    const uint64_t output_base =
        ((batch * token_count + query_token) * query_head_count +
         query_head) *
        kHeadSize;
    output[output_base + lane] =
        __float2bfloat16_rn(numerator_low * inverse_denominator);
    output[output_base + lane + kWarpSize] =
        __float2bfloat16_rn(numerator_high * inverse_denominator);
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
