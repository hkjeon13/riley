// Chain-wise tensor-core attention (prototype for the harness).
//
// A row's logical pages are dealt to eight chains: chain `c` owns pages `c, c + 8, ...`. One warp folds one chain:
//   scores  = BF16(BF16(q . k) * scale) through eight MMAs per page (fixed pairwise tree), -Inf past the valid tokens;
//   m_c     = maximum of the chain's scores; e = exp(score - m_c); S_c = sum of e (lane order, then a fixed lane tree);
//   N_c     = sum of e * v with e carried exactly as three BF16 pieces through MMAs, pages in ascending order.
// The chains are merged per head: M = max m_c; w_c = exp(m_c - M); S = fma chain of w_c * S_c and N = fma chain of
// w_c * N_c in ascending chain order over the chains that own a page; output = BF16(N * (1 / S)).
// Two launch geometries run exactly these operations: one CTA per (row, KV head) with a warp per chain and the merge
// inside the CTA, or one single-warp CTA per (chain, KV head, row) followed by a merge kernel.
constexpr uint32_t kChainCount = 8;
constexpr uint32_t kChainHeads = 8;
constexpr uint32_t kChainSavedPages = 8;
constexpr uint32_t kChainStateFloats = 2 * kChainHeads + kChainHeads * 128;   // m[8], S[8], N[8][128]

struct ChainLaneState { float maximum0, maximum1, sum0, sum1; float numerator[8][4]; };

__device__ __forceinline__ void fold_chain(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, const DeviceBatch& batch,
                                           uint64_t row, uint64_t key_value_head, uint32_t chain, uint64_t page_count, float scale, uint32_t lane, ChainLaneState* state) {
  constexpr uint32_t kHead = 128;
  const uint32_t group = lane / 4, pair = lane % 4;
  const __nv_bfloat16* const query_head = query + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kChainHeads + group) * kHead + 2 * pair;
  uint32_t query_operand[8][2];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { query_operand[step][0] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step); query_operand[step][1] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step + 8); }
  const auto page_scores = [&](uint64_t page, float* score) {
    uint64_t block_head_base = 0, token_count = 0;
    if (!resolve_ragged_partial_block(batch, row, page, key_value_head, &block_head_base, &token_count)) { score[0] = score[1] = score[2] = score[3] = CUDART_NAN_F; return; }
    const __nv_bfloat16* const key_low = key_pool + block_head_base + group * kHead + 2 * pair; const __nv_bfloat16* const key_high = key_low + 8 * kHead;
    float partial[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) { partial[step][0] = partial[step][1] = partial[step][2] = partial[step][3] = 0.0F;
      mma_m16n8k16_bf16(partial[step], *reinterpret_cast<const uint32_t*>(key_low + 16 * step), *reinterpret_cast<const uint32_t*>(key_high + 16 * step),
                        *reinterpret_cast<const uint32_t*>(key_low + 16 * step + 8), *reinterpret_cast<const uint32_t*>(key_high + 16 * step + 8), query_operand[step][0], query_operand[step][1]); }
#pragma unroll
    for (uint32_t element = 0; element < 4; ++element) {
      const float dot = ((partial[0][element] + partial[1][element]) + (partial[2][element] + partial[3][element])) + ((partial[4][element] + partial[5][element]) + (partial[6][element] + partial[7][element]));
      const bool present = (element < 2 ? group : group + 8) < token_count;
      score[element] = present ? staged_attention_score(dot, scale) : -CUDART_INF_F; } };
  uint16_t saved_score[kChainSavedPages][4];
  float maximum0 = -CUDART_INF_F, maximum1 = -CUDART_INF_F; uint32_t slot = 0;
  for (uint64_t page = chain; page < page_count; page += kChainCount, ++slot) { float score[4]; page_scores(page, score);
    if (slot < kChainSavedPages) {
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) saved_score[slot][element] = __bfloat16_as_ushort(__float2bfloat16_rn(score[element])); }
    maximum0 = fmaxf(maximum0, fmaxf(score[0], score[2])); maximum1 = fmaxf(maximum1, fmaxf(score[1], score[3])); }
#pragma unroll
  for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) { maximum0 = fmaxf(maximum0, __shfl_xor_sync(kFullWarpMask, maximum0, offset)); maximum1 = fmaxf(maximum1, __shfl_xor_sync(kFullWarpMask, maximum1, offset)); }
  float sum0 = 0.0F, sum1 = 0.0F;
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { state->numerator[step][0] = state->numerator[step][1] = state->numerator[step][2] = state->numerator[step][3] = 0.0F; }
  const uint32_t source_even = 8 * pair + group / 2, source_odd = source_even + 4, source_shift = 16 * (group % 2);
  slot = 0;
  for (uint64_t page = chain; page < page_count; page += kChainCount, ++slot) {
    float score[4];
    if (slot < kChainSavedPages) {
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) score[element] = __bfloat162float(__ushort_as_bfloat16(saved_score[slot][element]));
    } else { page_scores(page, score); }
    float weight[4]; weight[0] = expf(score[0] - maximum0); weight[1] = expf(score[1] - maximum1); weight[2] = expf(score[2] - maximum0); weight[3] = expf(score[3] - maximum1);
    sum0 += weight[0] + weight[2]; sum1 += weight[1] + weight[3];
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
    uint64_t block_head_base = 0, token_count = 0;
    if (!resolve_ragged_partial_block(batch, row, page, key_value_head, &block_head_base, &token_count)) continue;
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
      for (uint32_t piece = 0; piece < 3; ++piece) mma_m16n8k16_bf16(state->numerator[step], value0, value1, value2, value3, weight_operand[piece][0], weight_operand[piece][1]); } }
#pragma unroll
  for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) { sum0 += __shfl_xor_sync(kFullWarpMask, sum0, offset); sum1 += __shfl_xor_sync(kFullWarpMask, sum1, offset); }
  state->maximum0 = maximum0; state->maximum1 = maximum1; state->sum0 = sum0; state->sum1 = sum1; }

// Stores this lane's part of a chain state `[m[8], S[8], N[8][128]]`.
__device__ __forceinline__ void store_chain(float* chain_state, const ChainLaneState& state, uint32_t lane) {
  const uint32_t group = lane / 4, pair = lane % 4;
  if (group == 0) { chain_state[2 * pair] = state.maximum0; chain_state[2 * pair + 1] = state.maximum1; chain_state[kChainHeads + 2 * pair] = state.sum0; chain_state[kChainHeads + 2 * pair + 1] = state.sum1; }
  float* const numerator = chain_state + 2 * kChainHeads;
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { numerator[(2 * pair) * 128 + 16 * step + group] = state.numerator[step][0]; numerator[(2 * pair + 1) * 128 + 16 * step + group] = state.numerator[step][1];
    numerator[(2 * pair) * 128 + 16 * step + group + 8] = state.numerator[step][2]; numerator[(2 * pair + 1) * 128 + 16 * step + group + 8] = state.numerator[step][3]; } }

// One merged output element from the states of the chains that own a page; `stride` floats separate two chains.
__device__ __forceinline__ float merge_chains(const float* chain_states, size_t stride, uint32_t chains, uint32_t head, uint32_t depth) {
  float maximum = -CUDART_INF_F;
  for (uint32_t chain = 0; chain < chains; ++chain) maximum = fmaxf(maximum, chain_states[chain * stride + head]);
  float sum = 0.0F, merged = 0.0F;
  for (uint32_t chain = 0; chain < chains; ++chain) { const float* const state = chain_states + chain * stride; const float weight = expf(state[head] - maximum);
    sum = fmaf(weight, state[kChainHeads + head], sum); merged = fmaf(weight, state[2 * kChainHeads + head * 128 + depth], merged); }
  const float inverse_sum = isnan(sum) ? CUDART_NAN_F : (sum > 0.0F ? 1.0F / sum : 0.0F);
  return merged * inverse_sum; }

// Geometry 1: one CTA per (row, KV head), a warp per chain, merge inside the CTA.
__global__ __launch_bounds__(256) void chain_attention_single_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, __nv_bfloat16* output, DeviceBatch batch, uint64_t page_capacity, float scale) {
  __shared__ float states[kChainCount][kChainStateFloats];
  const uint32_t lane = threadIdx.x % kWarpSize, warp = threadIdx.x / kWarpSize; const uint64_t row = blockIdx.x, key_value_head = blockIdx.y;
  __nv_bfloat16* const row_output = output + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kChainHeads) * 128;
  const bool inactive = row >= active_rows(batch); const uint64_t page_count = inactive ? 0 : static_cast<uint64_t>(batch.row_positions[row]) / 16 + 1;
  if (inactive || page_count > page_capacity) { const float fill = inactive ? 0.0F : CUDART_NAN_F;
#pragma unroll
    for (uint32_t part = 0; part < 4; ++part) row_output[threadIdx.x + part * 256] = __float2bfloat16_rn(fill);
    return; }
  if (warp < page_count) { ChainLaneState state; fold_chain(query, key_pool, value_pool, batch, row, key_value_head, warp, page_count, scale, lane, &state); store_chain(states[warp], state, lane); }
  __syncthreads();
  const uint32_t chains = page_count < kChainCount ? static_cast<uint32_t>(page_count) : kChainCount;
#pragma unroll
  for (uint32_t part = 0; part < 4; ++part) { const uint32_t index = threadIdx.x + part * 256; row_output[index] = __float2bfloat16_rn(merge_chains(&states[0][0], kChainStateFloats, chains, index / 128, index % 128)); } }

// Geometry 2: one single-warp CTA per (chain, KV head, row) writes its chain state; a merge kernel finishes the row.
__global__ __launch_bounds__(32) void chain_attention_producer_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, float* chain_states, DeviceBatch batch, uint64_t page_capacity, float scale) {
  const uint32_t lane = threadIdx.x, chain = blockIdx.x; const uint64_t key_value_head = blockIdx.y, row = blockIdx.z;
  if (row >= active_rows(batch)) return;
  const uint64_t page_count = static_cast<uint64_t>(batch.row_positions[row]) / 16 + 1;
  if (page_count > page_capacity || chain >= page_count) return;
  ChainLaneState state; fold_chain(query, key_pool, value_pool, batch, row, key_value_head, chain, page_count, scale, lane, &state);
  store_chain(chain_states + ((row * 2 + key_value_head) * kChainCount + chain) * kChainStateFloats, state, lane); }
__global__ __launch_bounds__(256) void chain_attention_merge_kernel(const float* chain_states, __nv_bfloat16* output, DeviceBatch batch, uint64_t page_capacity) {
  const uint64_t row = blockIdx.x, key_value_head = blockIdx.y;
  __nv_bfloat16* const row_output = output + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kChainHeads) * 128;
  const bool inactive = row >= active_rows(batch); const uint64_t page_count = inactive ? 0 : static_cast<uint64_t>(batch.row_positions[row]) / 16 + 1;
  if (inactive || page_count > page_capacity) { const float fill = inactive ? 0.0F : CUDART_NAN_F;
#pragma unroll
    for (uint32_t part = 0; part < 4; ++part) row_output[threadIdx.x + part * 256] = __float2bfloat16_rn(fill);
    return; }
  const uint32_t chains = page_count < kChainCount ? static_cast<uint32_t>(page_count) : kChainCount;
  const float* const states = chain_states + (row * 2 + key_value_head) * kChainCount * kChainStateFloats;
#pragma unroll
  for (uint32_t part = 0; part < 4; ++part) { const uint32_t index = threadIdx.x + part * 256; row_output[index] = __float2bfloat16_rn(merge_chains(states, kChainStateFloats, chains, index / 128, index % 128)); } }
