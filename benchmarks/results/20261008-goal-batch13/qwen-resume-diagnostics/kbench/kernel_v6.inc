constexpr uint32_t kNativeBf16RaggedD128RowwiseWarps = 8;
constexpr uint32_t kNativeBf16RaggedD128RowwiseThreads =
    kNativeBf16RaggedD128RowwiseWarps * kWarpSize;
constexpr uint32_t kNativeBf16RaggedD128RowwiseGroupHeads = 8;
constexpr uint32_t kNativeBf16RaggedD128RowwiseSavedPages = 8;

__global__ __launch_bounds__(kNativeBf16RaggedD128RowwiseThreads)
void native_bf16_ragged_d128_rowwise_mma_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output, DeviceBatch batch,
    uint64_t page_capacity, float scale) {
  constexpr uint32_t kHead = 128;
  constexpr uint32_t kPage = 16;
  constexpr uint32_t kWarps = kNativeBf16RaggedD128RowwiseWarps;
  constexpr uint32_t kHeads = kNativeBf16RaggedD128RowwiseGroupHeads;
  constexpr uint32_t kSaved = kNativeBf16RaggedD128RowwiseSavedPages;
  __shared__ float warp_maximum[kWarps][kHeads];
  __shared__ float warp_sum[kWarps][kHeads];
  __shared__ float warp_numerator[kWarps][kHeads][kHead];
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint32_t group = lane / 4;
  const uint32_t pair = lane % 4;
  const uint64_t row = blockIdx.x;
  const uint64_t key_value_head = blockIdx.y;
  __nv_bfloat16* const row_output =
      output + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
                key_value_head * kHeads) *
                   kHead;
  // Every exit before the barriers below depends only on CTA-uniform values.
  const bool inactive = row >= active_rows(batch);
  const uint64_t page_count =
      inactive ? 0 : static_cast<uint64_t>(batch.row_positions[row]) / kPage + 1;
  if (inactive || page_count > page_capacity) {
    const float fill = inactive ? 0.0F : CUDART_NAN_F;
#pragma unroll
    for (uint32_t part = 0;
         part < kHeads * kHead / kNativeBf16RaggedD128RowwiseThreads; ++part) {
      row_output[threadIdx.x + part * kNativeBf16RaggedD128RowwiseThreads] =
          __float2bfloat16_rn(fill);
    }
    return;
  }

  // Query operand B[k = depth][n = head]: head `group` of this KV head.
  const __nv_bfloat16* const query_head =
      query +
      (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
       key_value_head * kHeads + group) *
          kHead +
      2 * pair;
  uint32_t query_operand[8][2];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) {
    query_operand[step][0] =
        *reinterpret_cast<const uint32_t*>(query_head + 16 * step);
    query_operand[step][1] =
        *reinterpret_cast<const uint32_t*>(query_head + 16 * step + 8);
  }
  // This lane's four scores of a page: [token group, head 2 pair],
  // [token group, head 2 pair + 1], [token group + 8, head 2 pair],
  // [token group + 8, head 2 pair + 1]. An unresolvable page scores NaN.
  const auto page_scores = [&](uint64_t page, float* score) {
    uint64_t block_head_base = 0;
    uint64_t token_count = 0;
    if (!resolve_ragged_partial_block(batch, row, page, key_value_head,
                                      &block_head_base, &token_count)) {
      score[0] = CUDART_NAN_F;
      score[1] = CUDART_NAN_F;
      score[2] = CUDART_NAN_F;
      score[3] = CUDART_NAN_F;
      return;
    }
    // Key operand A[m = token][k = depth]: tokens `group` and `group + 8`.
    const __nv_bfloat16* const key_low =
        key_pool + block_head_base + group * kHead + 2 * pair;
    const __nv_bfloat16* const key_high = key_low + 8 * kHead;
    float partial[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) {
      partial[step][0] = 0.0F;
      partial[step][1] = 0.0F;
      partial[step][2] = 0.0F;
      partial[step][3] = 0.0F;
      mma_m16n8k16_bf16(
          partial[step],
          *reinterpret_cast<const uint32_t*>(key_low + 16 * step),
          *reinterpret_cast<const uint32_t*>(key_high + 16 * step),
          *reinterpret_cast<const uint32_t*>(key_low + 16 * step + 8),
          *reinterpret_cast<const uint32_t*>(key_high + 16 * step + 8),
          query_operand[step][0], query_operand[step][1]);
    }
#pragma unroll
    for (uint32_t element = 0; element < 4; ++element) {
      const float dot =
          ((partial[0][element] + partial[1][element]) +
           (partial[2][element] + partial[3][element])) +
          ((partial[4][element] + partial[5][element]) +
           (partial[6][element] + partial[7][element]));
      const bool present = (element < 2 ? group : group + 8) < token_count;
      score[element] =
          present ? staged_attention_score(dot, scale) : -CUDART_INF_F;
    }
  };

  // 1. Scores and the row maximum of each head.
  uint16_t saved_score[kSaved][4];
  float maximum0 = -CUDART_INF_F;
  float maximum1 = -CUDART_INF_F;
  uint32_t slot = 0;
  for (uint64_t page = warp; page < page_count; page += kWarps, ++slot) {
    float score[4];
    page_scores(page, score);
    if (slot < kSaved) {
      // A staged score is a BF16 value, -Inf or NaN: its bits are exact.
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) {
        saved_score[slot][element] =
            __bfloat16_as_ushort(__float2bfloat16_rn(score[element]));
      }
    }
    maximum0 = fmaxf(maximum0, fmaxf(score[0], score[2]));
    maximum1 = fmaxf(maximum1, fmaxf(score[1], score[3]));
  }
#pragma unroll
  for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) {
    maximum0 =
        fmaxf(maximum0, __shfl_xor_sync(kFullWarpMask, maximum0, offset));
    maximum1 =
        fmaxf(maximum1, __shfl_xor_sync(kFullWarpMask, maximum1, offset));
  }
  if (group == 0) {
    warp_maximum[warp][2 * pair] = maximum0;
    warp_maximum[warp][2 * pair + 1] = maximum1;
  }
  __syncthreads();
  maximum0 = -CUDART_INF_F;
  maximum1 = -CUDART_INF_F;
#pragma unroll
  for (uint32_t other = 0; other < kWarps; ++other) {
    maximum0 = fmaxf(maximum0, warp_maximum[other][2 * pair]);
    maximum1 = fmaxf(maximum1, warp_maximum[other][2 * pair + 1]);
  }

  // 2 and 3. Weights exp(score - M), their sum, and the weighted values; the
  // weight operand is e in three exact BF16 pieces.
  float sum0 = 0.0F;
  float sum1 = 0.0F;
  float numerator[8][4];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) {
    numerator[step][0] = 0.0F;
    numerator[step][1] = 0.0F;
    numerator[step][2] = 0.0F;
    numerator[step][3] = 0.0F;
  }
  // The probability operand of lane `(group, pair)` is head `group` at
  // tokens `2 * pair (+1, +8, +9)`; those weights sit in lanes
  // `(2 * pair, group / 2)` and `(2 * pair + 1, group / 2)`.
  const uint32_t source_even = 8 * pair + group / 2;
  const uint32_t source_odd = source_even + 4;
  const uint32_t source_shift = 16 * (group % 2);
  slot = 0;
  for (uint64_t page = warp; page < page_count; page += kWarps, ++slot) {
    float score[4];
    if (slot < kSaved) {
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) {
        score[element] = __bfloat162float(
            __ushort_as_bfloat16(saved_score[slot][element]));
      }
    } else {
      page_scores(page, score);
    }
    float weight[4];
    weight[0] = expf(score[0] - maximum0);
    weight[1] = expf(score[1] - maximum1);
    weight[2] = expf(score[2] - maximum0);
    weight[3] = expf(score[3] - maximum1);
    sum0 += weight[0] + weight[2];
    sum1 += weight[1] + weight[3];
    // Weight operands B[k = token][n = head group], one per BF16 piece. Each
    // subtraction below is exact, and the third piece holds what is left.
    uint32_t weight_operand[3][2];
#pragma unroll
    for (uint32_t piece = 0; piece < 3; ++piece) {
      float rounded[4];
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) {
        rounded[element] =
            __bfloat162float(__float2bfloat16_rn(weight[element]));
        weight[element] -= rounded[element];
      }
      const uint32_t low_tokens = pack_bf16_pair(rounded[0], rounded[1]);
      const uint32_t high_tokens = pack_bf16_pair(rounded[2], rounded[3]);
      const uint32_t even_low =
          __shfl_sync(kFullWarpMask, low_tokens, source_even);
      const uint32_t even_high =
          __shfl_sync(kFullWarpMask, high_tokens, source_even);
      const uint32_t odd_low =
          __shfl_sync(kFullWarpMask, low_tokens, source_odd);
      const uint32_t odd_high =
          __shfl_sync(kFullWarpMask, high_tokens, source_odd);
      weight_operand[piece][0] =
          ((even_low >> source_shift) & 0xffffU) |
          (((odd_low >> source_shift) & 0xffffU) << 16);
      weight_operand[piece][1] =
          ((even_high >> source_shift) & 0xffffU) |
          (((odd_high >> source_shift) & 0xffffU) << 16);
    }
    uint64_t block_head_base = 0;
    uint64_t token_count = 0;
    // An unresolvable page scored NaN, so the sum is NaN; its values are not
    // read.
    if (!resolve_ragged_partial_block(batch, row, page, key_value_head,
                                      &block_head_base, &token_count)) {
      continue;
    }
    // Value operand A[m = depth][k = token]: depths `16 step + group (+8)`;
    // tokens past the page's valid count are staged as zero.
    const uint16_t* const value_token =
        reinterpret_cast<const uint16_t*>(value_pool + block_head_base +
                                          2 * pair * kHead + group);
    const bool present0 = 2 * pair < token_count;
    const bool present1 = 2 * pair + 1 < token_count;
    const bool present2 = 2 * pair + 8 < token_count;
    const bool present3 = 2 * pair + 9 < token_count;
    uint32_t value_operand[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) {
      const uint16_t* const token0 = value_token + 16 * step;
      const uint16_t* const token1 = token0 + kHead;
      const uint16_t* const token2 = token0 + 8 * kHead;
      const uint16_t* const token3 = token2 + kHead;
      value_operand[step][0] =
          (present0 ? static_cast<uint32_t>(token0[0]) : 0U) |
          ((present1 ? static_cast<uint32_t>(token1[0]) : 0U) << 16);
      value_operand[step][1] =
          (present0 ? static_cast<uint32_t>(token0[8]) : 0U) |
          ((present1 ? static_cast<uint32_t>(token1[8]) : 0U) << 16);
      value_operand[step][2] =
          (present2 ? static_cast<uint32_t>(token2[0]) : 0U) |
          ((present3 ? static_cast<uint32_t>(token3[0]) : 0U) << 16);
      value_operand[step][3] =
          (present2 ? static_cast<uint32_t>(token2[8]) : 0U) |
          ((present3 ? static_cast<uint32_t>(token3[8]) : 0U) << 16);
    }
    // Piece by piece across the eight depth steps: consecutive MMAs then
    // write different accumulators, and each accumulator still receives its
    // pieces in the order high, middle, low.
#pragma unroll
    for (uint32_t piece = 0; piece < 3; ++piece) {
#pragma unroll
      for (uint32_t step = 0; step < 8; ++step) {
        mma_m16n8k16_bf16(numerator[step], value_operand[step][0],
                          value_operand[step][1], value_operand[step][2],
                          value_operand[step][3], weight_operand[piece][0],
                          weight_operand[piece][1]);
      }
    }
  }
#pragma unroll
  for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) {
    sum0 += __shfl_xor_sync(kFullWarpMask, sum0, offset);
    sum1 += __shfl_xor_sync(kFullWarpMask, sum1, offset);
  }
  if (group == 0) {
    warp_sum[warp][2 * pair] = sum0;
    warp_sum[warp][2 * pair + 1] = sum1;
  }
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) {
    warp_numerator[warp][2 * pair][16 * step + group] = numerator[step][0];
    warp_numerator[warp][2 * pair + 1][16 * step + group] = numerator[step][1];
    warp_numerator[warp][2 * pair][16 * step + group + 8] = numerator[step][2];
    warp_numerator[warp][2 * pair + 1][16 * step + group + 8] =
        numerator[step][3];
  }
  __syncthreads();
#pragma unroll
  for (uint32_t part = 0;
       part < kHeads * kHead / kNativeBf16RaggedD128RowwiseThreads; ++part) {
    const uint32_t index =
        threadIdx.x + part * kNativeBf16RaggedD128RowwiseThreads;
    const uint32_t head = index / kHead;
    const uint32_t depth = index % kHead;
    float row_sum = 0.0F;
    float merged = 0.0F;
#pragma unroll
    for (uint32_t state_warp = 0; state_warp < kWarps; ++state_warp) {
      row_sum += warp_sum[state_warp][head];
      merged += warp_numerator[state_warp][head][depth];
    }
    // A NaN sum marks a non-finite score or an unresolvable page.
    const float inverse_sum =
        isnan(row_sum) ? CUDART_NAN_F
                       : (row_sum > 0.0F ? 1.0F / row_sum : 0.0F);
    row_output[index] = __float2bfloat16_rn(merged * inverse_sum);
  }
}

