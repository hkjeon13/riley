// Chain-wise tensor-core attention, launch form: one single-warp CTA per (chain, KV head, row) stages each page tile in
// shared memory with 16-byte coalesced loads and reads the MMA fragments with ldmatrix; a merge kernel finishes the row.
// Same arithmetic as kernel_chain.inc.
namespace chain2 {
__device__ __forceinline__ void ldmatrix_x4(uint32_t* fragment, uint32_t shared_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3]) : "r"(shared_row) : "memory"); }
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* fragment, uint32_t shared_row) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(fragment[0]), "=r"(fragment[1]), "=r"(fragment[2]), "=r"(fragment[3]) : "r"(shared_row) : "memory"); }

__global__ __launch_bounds__(32) void producer_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, float* chain_states, DeviceBatch batch, uint64_t page_capacity, float scale) {
  constexpr uint32_t kHead = 128, kPage = 16;
  __shared__ __align__(16) uint32_t tile[kPage * kHead / 2];
  const uint32_t lane = threadIdx.x, chain = blockIdx.x, group = lane / 4, pair = lane % 4; const uint64_t key_value_head = blockIdx.y, row = blockIdx.z;
  if (row >= active_rows(batch)) return;
  const uint64_t page_count = static_cast<uint64_t>(batch.row_positions[row]) / kPage + 1;
  if (page_count > page_capacity || chain >= page_count) return;
  const uint32_t tile_address = static_cast<uint32_t>(__cvta_generic_to_shared(tile));
  // Sixteen bytes per lane and instruction: 512 consecutive bytes, two tokens, per instruction. Tokens at or past
  // `present_tokens` are staged as zero.
  const auto stage_tile = [&](const __nv_bfloat16* page_tile, uint64_t present_tokens) {
    const uint4* const words = reinterpret_cast<const uint4*>(page_tile) + lane; uint4* const staged = reinterpret_cast<uint4*>(tile) + lane;
#pragma unroll
    for (uint32_t part = 0; part < kPage * kHead / (8 * kWarpSize); ++part) { const uint4 loaded = words[part * kWarpSize]; const bool present = 2 * part + lane / 16 < present_tokens;
      staged[part * kWarpSize] = present ? loaded : make_uint4(0, 0, 0, 0); } };
  const __nv_bfloat16* const query_head = query + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kChainHeads + group) * kHead + 2 * pair;
  uint32_t query_operand[8][2];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { query_operand[step][0] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step); query_operand[step][1] = *reinterpret_cast<const uint32_t*>(query_head + 16 * step + 8); }
  const uint32_t key_row = tile_address + ((lane % 8 + 8 * ((lane / 8) % 2)) * kHead + 8 * (lane / 16)) * 2;
  const uint32_t value_row = tile_address + ((lane % 8 + 8 * (lane / 16)) * kHead + 8 * ((lane / 8) % 2)) * 2;
  const auto page_scores = [&](uint64_t page, float* score) {
    uint64_t block_head_base = 0, token_count = 0;
    if (!resolve_ragged_partial_block(batch, row, page, key_value_head, &block_head_base, &token_count)) { score[0] = score[1] = score[2] = score[3] = CUDART_NAN_F; return; }
    stage_tile(key_pool + block_head_base, kPage);
    float partial[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) { uint32_t key_operand[4]; ldmatrix_x4(key_operand, key_row + 32 * step);
      partial[step][0] = partial[step][1] = partial[step][2] = partial[step][3] = 0.0F;
      mma_m16n8k16_bf16(partial[step], key_operand[0], key_operand[1], key_operand[2], key_operand[3], query_operand[step][0], query_operand[step][1]); }
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
  ChainLaneState state; float sum0 = 0.0F, sum1 = 0.0F;
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) { state.numerator[step][0] = state.numerator[step][1] = state.numerator[step][2] = state.numerator[step][3] = 0.0F; }
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
    stage_tile(value_pool + block_head_base, token_count);
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) { uint32_t value_operand[4]; ldmatrix_x4_trans(value_operand, value_row + 32 * step);
#pragma unroll
      for (uint32_t piece = 0; piece < 3; ++piece) mma_m16n8k16_bf16(state.numerator[step], value_operand[0], value_operand[1], value_operand[2], value_operand[3], weight_operand[piece][0], weight_operand[piece][1]); } }
#pragma unroll
  for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) { sum0 += __shfl_xor_sync(kFullWarpMask, sum0, offset); sum1 += __shfl_xor_sync(kFullWarpMask, sum1, offset); }
  state.maximum0 = maximum0; state.maximum1 = maximum1; state.sum0 = sum0; state.sum1 = sum1;
  store_chain(chain_states + ((row * 2 + key_value_head) * kChainCount + chain) * kChainStateFloats, state, lane); }

// Merge: per head, M = max m_c, w_c = exp(m_c - M), S and N as fma chains over the chains in ascending order.
__global__ __launch_bounds__(256) void merge_kernel(const float* chain_states, __nv_bfloat16* output, DeviceBatch batch, uint64_t page_capacity) {
  __shared__ float weights[kChainCount][kChainHeads]; __shared__ float inverse_sums[kChainHeads];
  const uint64_t row = blockIdx.x, key_value_head = blockIdx.y;
  __nv_bfloat16* const row_output = output + (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + key_value_head * kChainHeads) * 128;
  const bool inactive = row >= active_rows(batch); const uint64_t page_count = inactive ? 0 : static_cast<uint64_t>(batch.row_positions[row]) / 16 + 1;
  if (inactive || page_count > page_capacity) { const float fill = inactive ? 0.0F : CUDART_NAN_F;
#pragma unroll
    for (uint32_t part = 0; part < 4; ++part) row_output[threadIdx.x + part * 256] = __float2bfloat16_rn(fill);
    return; }
  const uint32_t chains = page_count < kChainCount ? static_cast<uint32_t>(page_count) : kChainCount;
  const float* const states = chain_states + (row * 2 + key_value_head) * kChainCount * kChainStateFloats;
  if (threadIdx.x < kChainHeads) { const uint32_t head = threadIdx.x; float maximum = -CUDART_INF_F;
    for (uint32_t chain = 0; chain < chains; ++chain) maximum = fmaxf(maximum, states[chain * kChainStateFloats + head]);
    float sum = 0.0F;
    for (uint32_t chain = 0; chain < chains; ++chain) { const float* const state = states + chain * kChainStateFloats; const float weight = expf(state[head] - maximum); weights[chain][head] = weight; sum = fmaf(weight, state[kChainHeads + head], sum); }
    inverse_sums[head] = isnan(sum) ? CUDART_NAN_F : (sum > 0.0F ? 1.0F / sum : 0.0F); }
  __syncthreads();
#pragma unroll
  for (uint32_t part = 0; part < 4; ++part) { const uint32_t index = threadIdx.x + part * 256, head = index / 128, depth = index % 128; float merged = 0.0F;
    for (uint32_t chain = 0; chain < chains; ++chain) merged = fmaf(weights[chain][head], states[chain * kChainStateFloats + 2 * kChainHeads + head * 128 + depth], merged);
    row_output[index] = __float2bfloat16_rn(merged * inverse_sums[head]); } }
}  // namespace chain2
