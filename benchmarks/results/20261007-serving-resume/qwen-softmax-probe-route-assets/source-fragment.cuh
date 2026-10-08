__global__ void decode_softmax_reference_kernel(
    __nv_bfloat16* scores, uint64_t logical_token_count,
    uint64_t query_head_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t query_head = first; query_head < query_head_count;
       query_head += stride) {
    const uint64_t base = query_head * logical_token_count;
    float maximum = -CUDART_INF_F;
    bool has_nan = false;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      const float score = __bfloat162float(scores[base + token]);
      has_nan = has_nan || isnan(score);
      maximum = fmaxf(maximum, score);
    }
    if (has_nan) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t token = 0; token < logical_token_count; ++token) {
        scores[base + token] = nan;
      }
      continue;
    }
    if (isinf(maximum)) {
      if (maximum > 0.0F) {
        uint64_t positive_infinity_count = 0;
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          const float score = __bfloat162float(scores[base + token]);
          positive_infinity_count +=
              static_cast<uint64_t>(isinf(score) && score > 0.0F);
        }
        const float tied_probability =
            1.0F / static_cast<float>(positive_infinity_count);
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          const float score = __bfloat162float(scores[base + token]);
          const float probability = isinf(score) && score > 0.0F
                                        ? tied_probability
                                        : 0.0F;
          scores[base + token] = __float2bfloat16_rn(probability);
        }
      } else {
        const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          scores[base + token] = zero;
        }
      }
      continue;
    }
    float denominator = 0.0F;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      denominator +=
          expf(__bfloat162float(scores[base + token]) - maximum);
    }
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      const float numerator =
          expf(__bfloat162float(scores[base + token]) - maximum);
      scores[base + token] =
          __float2bfloat16_rn(numerator / denominator);
    }
  }
}

