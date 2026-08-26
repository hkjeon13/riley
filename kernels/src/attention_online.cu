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
constexpr uint32_t kQueryTileElements = kWarpsPerBlock * kHeadSize;
constexpr uint32_t kScoreTileElements = kWarpsPerBlock * kKeyTileSize;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
constexpr uint32_t kCausalMaskBf16AsF32Bits = 0xff7f0000U;

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

__device__ __forceinline__ __nv_bfloat16 reference_staged_causal_score(
    const __nv_bfloat16* query_row,
    const __nv_bfloat16* transposed_key_tile, uint32_t key_offset,
    float scale, uint64_t query_token, uint64_t key_token) {
  float dot_product = 0.0F;
#pragma unroll 1
  for (uint32_t depth = 0; depth < kHeadSize; ++depth) {
    dot_product = fmaf(
        __bfloat162float(query_row[depth]),
        __bfloat162float(
            transposed_key_tile[depth * kKeyTileSize + key_offset]),
        dot_product);
  }
  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  const __nv_bfloat16 scaled = __float2bfloat16_rn(
      __bfloat162float(staged_dot) * scale);
  const float mask = key_token > query_token
                         ? __uint_as_float(kCausalMaskBf16AsF32Bits)
                         : 0.0F;
  return __float2bfloat16_rn(__bfloat162float(scaled) + mask);
}

__global__ __launch_bounds__(kThreadsPerBlock)
void reference_exact_causal_bf16_gqa_prefill(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* output,
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  __shared__ __nv_bfloat16 query_tile[kQueryTileElements];
  // QK lanes own keys, so transpose K to make each depth load contiguous
  // across a warp. AV lanes own depths, so V remains row-major by key.
  __shared__ __nv_bfloat16 key_tile[kKeyTileElements];
  __shared__ __nv_bfloat16 value_tile[kKeyTileElements];
  __shared__ __nv_bfloat16 score_tile[kScoreTileElements];

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
  const uint32_t query_row = warp * kHeadSize;
  const uint32_t score_row = warp * kKeyTileSize;

  if (active) {
    const uint64_t query_base =
        ((batch * token_count + query_token) * query_head_count +
         query_head) *
        kHeadSize;
    query_tile[query_row + lane] = query[query_base + lane];
    query_tile[query_row + lane + kWarpSize] =
        query[query_base + lane + kWarpSize];
  } else {
    query_tile[query_row + lane] = __float2bfloat16_rn(0.0F);
    query_tile[query_row + lane + kWarpSize] =
        __float2bfloat16_rn(0.0F);
  }
  __syncthreads();

  float maximum = -CUDART_INF_F;
  bool has_nan = false;

  // Pass one reproduces the materialized backend's serial D=64 QK fold,
  // staged BF16 scale/mask, and logical-key-order maximum/NaN scan. Future
  // masked scores are evaluated because the reference evaluates them too.
  for (uint64_t key_tile_start = 0; key_tile_start < token_count;
       key_tile_start += kKeyTileSize) {
    const uint64_t remaining_keys = token_count - key_tile_start;
    const uint32_t key_tile_count = static_cast<uint32_t>(
        remaining_keys < kKeyTileSize ? remaining_keys : kKeyTileSize);
    const uint64_t tile_elements =
        static_cast<uint64_t>(key_tile_count) * kHeadSize;
    for (uint64_t tile_index = threadIdx.x; tile_index < tile_elements;
         tile_index += kThreadsPerBlock) {
      const uint32_t key_offset =
          static_cast<uint32_t>(tile_index / kHeadSize);
      const uint32_t depth =
          static_cast<uint32_t>(tile_index % kHeadSize);
      const uint64_t key_token = key_tile_start + key_offset;
      const uint64_t key_index =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
              kHeadSize +
          depth;
      key_tile[depth * kKeyTileSize + key_offset] = key[key_index];
    }
    __syncthreads();

    if (active) {
      if (lane < key_tile_count) {
        const uint64_t key_token = key_tile_start + lane;
        score_tile[score_row + lane] = reference_staged_causal_score(
            &query_tile[query_row], key_tile, lane, scale, query_token,
            key_token);
      }
      __syncwarp();
      if (lane == 0) {
        for (uint32_t key_offset = 0; key_offset < key_tile_count;
             ++key_offset) {
          const float score =
              __bfloat162float(score_tile[score_row + key_offset]);
          has_nan = has_nan || isnan(score);
          maximum = fmaxf(maximum, score);
        }
      }
    }
    __syncthreads();
  }

  maximum = __shfl_sync(kFullWarpMask, maximum, 0);
  has_nan = __shfl_sync(kFullWarpMask, has_nan ? 1U : 0U, 0) != 0;
  float denominator = 0.0F;

  // Pass two repeats the exact staged scores and then performs the reference
  // denominator's left fold in ascending logical-key order.
  for (uint64_t key_tile_start = 0; key_tile_start < token_count;
       key_tile_start += kKeyTileSize) {
    const uint64_t remaining_keys = token_count - key_tile_start;
    const uint32_t key_tile_count = static_cast<uint32_t>(
        remaining_keys < kKeyTileSize ? remaining_keys : kKeyTileSize);
    const uint64_t tile_elements =
        static_cast<uint64_t>(key_tile_count) * kHeadSize;
    for (uint64_t tile_index = threadIdx.x; tile_index < tile_elements;
         tile_index += kThreadsPerBlock) {
      const uint32_t key_offset =
          static_cast<uint32_t>(tile_index / kHeadSize);
      const uint32_t depth =
          static_cast<uint32_t>(tile_index % kHeadSize);
      const uint64_t key_token = key_tile_start + key_offset;
      const uint64_t key_index =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
              kHeadSize +
          depth;
      key_tile[depth * kKeyTileSize + key_offset] = key[key_index];
    }
    __syncthreads();

    if (active && !has_nan) {
      if (lane < key_tile_count) {
        const uint64_t key_token = key_tile_start + lane;
        score_tile[score_row + lane] = reference_staged_causal_score(
            &query_tile[query_row], key_tile, lane, scale, query_token,
            key_token);
      }
      __syncwarp();
      if (lane == 0) {
        for (uint32_t key_offset = 0; key_offset < key_tile_count;
             ++key_offset) {
          const float score =
              __bfloat162float(score_tile[score_row + key_offset]);
          denominator += expf(score - maximum);
        }
      }
    }
    __syncthreads();
  }

  denominator = __shfl_sync(kFullWarpMask, denominator, 0);
  float accumulator_low = 0.0F;
  float accumulator_high = 0.0F;

  // Pass three stages the normalized probabilities to BF16, then performs
  // unconditional logical-key-order AV FMAs just like the materialized path.
  for (uint64_t key_tile_start = 0; key_tile_start < token_count;
       key_tile_start += kKeyTileSize) {
    const uint64_t remaining_keys = token_count - key_tile_start;
    const uint32_t key_tile_count = static_cast<uint32_t>(
        remaining_keys < kKeyTileSize ? remaining_keys : kKeyTileSize);
    const uint64_t tile_elements =
        static_cast<uint64_t>(key_tile_count) * kHeadSize;
    for (uint64_t tile_index = threadIdx.x; tile_index < tile_elements;
         tile_index += kThreadsPerBlock) {
      const uint32_t key_offset =
          static_cast<uint32_t>(tile_index / kHeadSize);
      const uint32_t depth =
          static_cast<uint32_t>(tile_index % kHeadSize);
      const uint64_t key_token = key_tile_start + key_offset;
      const uint64_t key_value_index =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
              kHeadSize +
          depth;
      key_tile[depth * kKeyTileSize + key_offset] = key[key_value_index];
      value_tile[key_offset * kHeadSize + depth] =
          value[key_value_index];
    }
    __syncthreads();

    if (active) {
      if (lane < key_tile_count) {
        __nv_bfloat16 probability;
        if (has_nan) {
          probability = __float2bfloat16_rn(CUDART_NAN_F);
        } else {
          const uint64_t key_token = key_tile_start + lane;
          const float score = __bfloat162float(
              reference_staged_causal_score(
                  &query_tile[query_row], key_tile, lane, scale,
                  query_token, key_token));
          probability = __float2bfloat16_rn(
              expf(score - maximum) / denominator);
        }
        score_tile[score_row + lane] = probability;
      }
      __syncwarp();
      for (uint32_t key_offset = 0; key_offset < key_tile_count;
           ++key_offset) {
        const float probability =
            __bfloat162float(score_tile[score_row + key_offset]);
        accumulator_low = fmaf(
            probability,
            __bfloat162float(
                value_tile[key_offset * kHeadSize + lane]),
            accumulator_low);
        accumulator_high = fmaf(
            probability,
            __bfloat162float(value_tile[
                key_offset * kHeadSize + lane + kWarpSize]),
            accumulator_high);
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
  if (causal_local) {
    online_bf16_gqa_prefill<<<grid, kThreadsPerBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(query),
        static_cast<const __nv_bfloat16*>(key),
        static_cast<const __nv_bfloat16*>(value),
        static_cast<__nv_bfloat16*>(output), token_count,
        query_head_count, key_value_head_count, scale, true,
        local_window_size);
  } else {
    reference_exact_causal_bf16_gqa_prefill
        <<<grid, kThreadsPerBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(query),
            static_cast<const __nv_bfloat16*>(key),
            static_cast<const __nv_bfloat16*>(value),
            static_cast<__nv_bfloat16*>(output), token_count,
            query_head_count, key_value_head_count, scale);
  }
  return cudaGetLastError();
}
