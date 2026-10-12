// Tensor-core page-state producer (harness prototype). One single-warp CTA folds the pages `x, x + stride, ...` of one
// (KV head, row), each page on its own, and writes the page's partial state `[m, l, n[128]]` for the KV head's eight
// query heads in the layout of the paged producers. Every page state depends only on its own page and row, so any
// stride gives the same bytes. The existing global-max merge then combines the page states.
//   scores = BF16(BF16(q . k) * scale): eight MMAs (tokens x heads over depth), fixed pairwise tree; -Inf past the valid tokens
//   m      = max of the page's scores;   e = exp(score - m);   l = sum of e (two tokens per lane, then a fixed lane tree)
//   n      = sum of e * v with e carried exactly as three BF16 pieces through MMAs (depth x heads over tokens)
namespace page {
__global__ __launch_bounds__(32) void producer_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, float* partial_states, DeviceBatch batch, uint64_t partial_state_capacity, uint32_t stride, float scale) {
  constexpr uint32_t kHead = 128, kStride = 130, kHeads = 8;
  const uint32_t lane = threadIdx.x, group = lane / 4, pair = lane % 4; const uint64_t key_value_head = blockIdx.y, row = blockIdx.z;
  if (row >= active_rows(batch)) return;
  const uint64_t page_count = static_cast<uint64_t>(batch.row_positions[row]) / 16 + 1;
  if (page_count > partial_state_capacity) return;
  const __nv_bfloat16* const query_head = query + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kHeads + group) * kHead + 2 * pair;
  uint32_t query_operand[8][2];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { query_operand[step][0] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step); query_operand[step][1] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step + 8); }
  const uint32_t source_even = 8 * pair + group / 2, source_odd = source_even + 4, source_shift = 16 * (group % 2);
  for (uint64_t page = blockIdx.x; page < page_count; page += stride) {
    float* const state0 = partial_states + ((row * partial_state_capacity + page) * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kHeads + 2 * pair) * kStride;
    float* const state1 = state0 + kStride;
    uint64_t block_head_base = 0, token_count = 0;
    const bool valid = resolve_ragged_partial_block(batch, row, page, key_value_head, &block_head_base, &token_count);
    float maximum0 = CUDART_NAN_F, maximum1 = CUDART_NAN_F, sum0 = CUDART_NAN_F, sum1 = CUDART_NAN_F; float numerator[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) { numerator[step][0] = numerator[step][1] = numerator[step][2] = numerator[step][3] = valid ? 0.0F : CUDART_NAN_F; }
    if (valid) {
      const __nv_bfloat16* const key_low = key_pool + block_head_base + group * kHead + 2 * pair; const __nv_bfloat16* const key_high = key_low + 8 * kHead;
      float partial[8][4];
#pragma unroll
      for (uint32_t step = 0; step < 8; ++step) { partial[step][0] = partial[step][1] = partial[step][2] = partial[step][3] = 0.0F;
        mma_m16n8k16_bf16(partial[step], *reinterpret_cast<const uint32_t*>(key_low + 16 * step), *reinterpret_cast<const uint32_t*>(key_high + 16 * step),
                          *reinterpret_cast<const uint32_t*>(key_low + 16 * step + 8), *reinterpret_cast<const uint32_t*>(key_high + 16 * step + 8), query_operand[step][0], query_operand[step][1]); }
      float score[4];
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) {
        const float dot = ((partial[0][element] + partial[1][element]) + (partial[2][element] + partial[3][element])) + ((partial[4][element] + partial[5][element]) + (partial[6][element] + partial[7][element]));
        const bool present = (element < 2 ? group : group + 8) < token_count;
        score[element] = present ? staged_attention_score(dot, scale) : -CUDART_INF_F; }
      maximum0 = fmaxf(score[0], score[2]); maximum1 = fmaxf(score[1], score[3]);
#pragma unroll
      for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) { maximum0 = fmaxf(maximum0, __shfl_xor_sync(kFullWarpMask, maximum0, offset)); maximum1 = fmaxf(maximum1, __shfl_xor_sync(kFullWarpMask, maximum1, offset)); }
      float weight[4]; weight[0] = expf(score[0] - maximum0); weight[1] = expf(score[1] - maximum1); weight[2] = expf(score[2] - maximum0); weight[3] = expf(score[3] - maximum1);
      sum0 = weight[0] + weight[2]; sum1 = weight[1] + weight[3];
#pragma unroll
      for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) { sum0 += __shfl_xor_sync(kFullWarpMask, sum0, offset); sum1 += __shfl_xor_sync(kFullWarpMask, sum1, offset); }
      uint32_t weight_operand[3][2];
#pragma unroll
      for (uint32_t piece = 0; piece < 3; ++piece) { float rounded[4];
#pragma unroll
        for (uint32_t element = 0; element < 4; ++element) { rounded[element] = __bfloat162float(__float2bfloat16_rn(weight[element])); weight[element] -= rounded[element]; }
        const uint32_t low_tokens = pack_bf16_pair(rounded[0], rounded[1]), high_tokens = pack_bf16_pair(rounded[2], rounded[3]);
        const uint32_t even_low = __shfl_sync(kFullWarpMask, low_tokens, source_even), even_high = __shfl_sync(kFullWarpMask, high_tokens, source_even);
        const uint32_t odd_low = __shfl_sync(kFullWarpMask, low_tokens, source_odd), odd_high = __shfl_sync(kFullWarpMask, high_tokens, source_odd);
        weight_operand[piece][0] = ((even_low >> source_shift) & 0xffffU) | (((odd_low >> source_shift) & 0xffffU) << 16);
        weight_operand[piece][1] = ((even_high >> source_shift) & 0xffffU) | (((odd_high >> source_shift) & 0xffffU) << 16); }
      const uint16_t* const value_token = reinterpret_cast<const uint16_t*>(value_pool + block_head_base + 2 * pair * kHead + group);
      const bool present0 = 2 * pair < token_count, present1 = 2 * pair + 1 < token_count, present2 = 2 * pair + 8 < token_count, present3 = 2 * pair + 9 < token_count;
#pragma unroll
      for (uint32_t step = 0; step < 8; ++step) {
        const uint16_t* const token0 = value_token + 16 * step; const uint16_t* const token1 = token0 + kHead; const uint16_t* const token2 = token0 + 8 * kHead; const uint16_t* const token3 = token2 + kHead;
        const uint32_t value0 = (present0 ? static_cast<uint32_t>(token0[0]) : 0U) | ((present1 ? static_cast<uint32_t>(token1[0]) : 0U) << 16);
        const uint32_t value1 = (present0 ? static_cast<uint32_t>(token0[8]) : 0U) | ((present1 ? static_cast<uint32_t>(token1[8]) : 0U) << 16);
        const uint32_t value2 = (present2 ? static_cast<uint32_t>(token2[0]) : 0U) | ((present3 ? static_cast<uint32_t>(token3[0]) : 0U) << 16);
        const uint32_t value3 = (present2 ? static_cast<uint32_t>(token2[8]) : 0U) | ((present3 ? static_cast<uint32_t>(token3[8]) : 0U) << 16);
#pragma unroll
        for (uint32_t piece = 0; piece < 3; ++piece) mma_m16n8k16_bf16(numerator[step], value0, value1, value2, value3, weight_operand[piece][0], weight_operand[piece][1]); } }
    if (group == 0) { state0[0] = maximum0; state0[1] = sum0; state1[0] = maximum1; state1[1] = sum1; }
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) { state0[2 + 16 * step + group] = numerator[step][0]; state1[2 + 16 * step + group] = numerator[step][1]; state0[2 + 16 * step + group + 8] = numerator[step][2]; state1[2 + 16 * step + group + 8] = numerator[step][3]; } } }
}  // namespace page
