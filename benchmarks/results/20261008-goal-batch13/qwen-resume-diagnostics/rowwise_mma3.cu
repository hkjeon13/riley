// Diagnostic: row-wise tensor-core (MMA) attention for the Qwen2.5-3B D128 route (16 query heads, 2 KV heads, 16-token
// pages) against the current pair "shared-KV partial-state producer + global-max merge".
//
// New kernel: one CTA per (row, KV head). Each of its eight warps folds the pages `warp, warp + 8, ...` of the row's
// context in ascending order with an online softmax: scores for the KV head's eight query heads come from eight
// m16n8k16 MMAs per page (tokens x heads), probabilities are rounded to BF16 and weight the page's values through eight
// more MMAs (depth x heads). The eight warp states are then merged inside the CTA (global maximum, fixed warp order) and
// the BF16 output is written directly: no partial-state buffer and no merge kernel.
// Every output depends only on its own row's query and its own pages, so it cannot depend on the batch composition.
//
// Timing: CUDA graph replay of 36 layers with distinct K/V pools (cold reads). Values are checked against a CPU reference.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, G = 8, PAGE = 16, STRIDE = 130;
constexpr unsigned FULL = 0xffffffffu;

// ---------------- current design: shared-KV producer + global-max merge (ported from kernels/src/batch_primitives.cu) -------------
__device__ __forceinline__ float staged(float dot, float scale) { return __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(dot)) * scale)); }
__device__ __forceinline__ float warp_sum_all(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v; }
__device__ __forceinline__ void online(float score, float* m, float* l, float* alpha, float* beta) {
  if (isnan(score) || isnan(*m)) { *m = *l = *alpha = *beta = CUDART_NAN_F; return; }
  const float nm = fmaxf(*m, score); *alpha = *l == 0.0F ? 0.0F : expf(*m - nm); *beta = expf(score - nm); *l = fmaf(*alpha, *l, *beta); *m = nm; }
__global__ __launch_bounds__(256) void base_producer(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, float* states, int cap, int active, float scale) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z;
  if (row >= active) return;
  const int T = positions[row] + 1, pages = (T - 1) / PAGE + 1;
  if (page >= pages || page >= cap) return;
  const int head = kvh * G + warp, count = min(PAGE, T - page * PAGE);
  const size_t qb = (size_t(row) * QH + head) * D, base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D;
  __shared__ float sk[PAGE * D]; __shared__ float sv[PAGE * D];
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = __bfloat162float(key_pool[base + e]); sv[e] = __bfloat162float(value_pool[base + e]); }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.F, n0 = 0.F, n1 = 0.F, n2 = 0.F, n3 = 0.F;
  for (int t = 0; t < count; ++t) { const int b = t * D + lane;
    float s = fmaf(q0, sk[b], q1 * sk[b + 32]); s = fmaf(q2, sk[b + 64], s); s = fmaf(q3, sk[b + 96], s); s = staged(warp_sum_all(s), scale);
    float alpha = 0.F, beta = 0.F; online(s, &m, &l, &alpha, &beta);
    const float v0 = sv[b], v1 = sv[b + 32], v2 = sv[b + 64], v3 = sv[b + 96];
    if (beta == 0.F) { n0 *= alpha; n1 *= alpha; n2 *= alpha; n3 *= alpha; }
    else if (alpha == 0.F) { n0 = beta * v0; n1 = beta * v1; n2 = beta * v2; n3 = beta * v3; }
    else { n0 = fmaf(beta, v0, alpha * n0); n1 = fmaf(beta, v1, alpha * n1); n2 = fmaf(beta, v2, alpha * n2); n3 = fmaf(beta, v3, alpha * n3); } }
  float* st = states + ((size_t(row) * cap + page) * QH + head) * STRIDE;
  if (lane == 0) { st[0] = m; st[1] = l; }
  st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3; }
__global__ __launch_bounds__(128) void base_merge(const float* states, const uint32_t* positions, __nv_bfloat16* out, int cap, int active) {
  __shared__ float weights[128]; __shared__ float red[128];
  const int head = blockIdx.x, row = blockIdx.y, depth = threadIdx.x; const size_t ob = (size_t(row) * QH + head) * D;
  if (row >= active) { out[ob + depth] = __float2bfloat16_rn(0.F); return; }
  const int pages = positions[row] / PAGE + 1; const size_t ps = size_t(QH) * STRIDE; const float* st = states + (size_t(row) * cap * QH + head) * STRIDE;
  float lm = -CUDART_INF_F; for (int p = depth; p < pages; p += 128) lm = fmaxf(lm, st[p * ps]);
  red[depth] = lm; __syncthreads();
  for (int o = 64; o; o >>= 1) { if (depth < o) red[depth] = fmaxf(red[depth], red[depth + o]); __syncthreads(); }
  const float M = red[0]; __syncthreads(); float ld = 0.F;
  for (int p = depth; p < pages; p += 128) { const float w = expf(st[p * ps] - M); weights[p] = w; ld = fmaf(w, st[p * ps + 1], ld); }
  red[depth] = ld; __syncthreads();
  for (int o = 64; o; o >>= 1) { if (depth < o) red[depth] += red[depth + o]; __syncthreads(); }
  const float den = red[0], inv = den > 0.F ? 1.F / den : 0.F; const float* nu = st + 2 + depth; float n = 0.F; int p = 0;
  for (; p + 4 <= pages; p += 4) { const float a = nu[p * ps], b = nu[(p + 1) * ps], c = nu[(p + 2) * ps], d = nu[(p + 3) * ps];
    n = fmaf(weights[p], a, n); n = fmaf(weights[p + 1], b, n); n = fmaf(weights[p + 2], c, n); n = fmaf(weights[p + 3], d, n); }
  for (; p < pages; ++p) n = fmaf(weights[p], nu[p * ps], n);
  out[ob + depth] = __float2bfloat16_rn(n * inv); }

// ---------------- new design: row-wise MMA, single stage -------------
__device__ __forceinline__ uint32_t pack(float a, float b) { return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) | (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16); }
__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
// One page of one row for the eight query heads of a KV head: scores (tokens x heads), online-softmax update of the warp's
// running state, probability-weighted values (depth x heads). MASKED = the page holds fewer than sixteen tokens.
// CH = number of independent score accumulators (depth steps dealt round-robin, summed in a fixed tree).
// STAGED = round the score to BF16 after the dot product and again after scaling, as the shared-KV producer does.
template <int CH, bool STAGED, bool MASKED>
__device__ __forceinline__ void page_fold(const __nv_bfloat16* key_page, const __nv_bfloat16* value_page, const uint32_t (*qb)[2], int count, float scale, int g, int t, int src_a, int src_b, int shift,
                                          float& mx0, float& mx1, float& den0, float& den1, float (*num)[4]) {
  const __nv_bfloat16* kp = key_page + 2 * t; float sc[CH][4];
#pragma unroll
  for (int c = 0; c < CH; ++c) { sc[c][0] = sc[c][1] = sc[c][2] = sc[c][3] = 0.F; }
#pragma unroll
  for (int s = 0; s < 8; ++s) { const __nv_bfloat16* k0 = kp + g * D + 16 * s; const __nv_bfloat16* k1 = kp + (g + 8) * D + 16 * s;
    mma(sc[s % CH], *reinterpret_cast<const uint32_t*>(k0), *reinterpret_cast<const uint32_t*>(k1), *reinterpret_cast<const uint32_t*>(k0 + 8), *reinterpret_cast<const uint32_t*>(k1 + 8), qb[s][0], qb[s][1]); }
#pragma unroll
  for (int step = 1; step < CH; step *= 2)
#pragma unroll
    for (int c = 0; c + step < CH; c += 2 * step) { sc[c][0] += sc[c + step][0]; sc[c][1] += sc[c + step][1]; sc[c][2] += sc[c + step][2]; sc[c][3] += sc[c + step][3]; }
  // sc[0]: [token g, head 2t], [token g, head 2t+1], [token g+8, head 2t], [token g+8, head 2t+1].
  float s00 = STAGED ? staged(sc[0][0], scale) : sc[0][0] * scale, s01 = STAGED ? staged(sc[0][1], scale) : sc[0][1] * scale, s10 = STAGED ? staged(sc[0][2], scale) : sc[0][2] * scale, s11 = STAGED ? staged(sc[0][3], scale) : sc[0][3] * scale;
  if (MASKED) { const bool lo = g < count, hi = g + 8 < count; if (!lo) { s00 = -CUDART_INF_F; s01 = -CUDART_INF_F; } if (!hi) { s10 = -CUDART_INF_F; s11 = -CUDART_INF_F; } }
  float pm0 = fmaxf(s00, s10), pm1 = fmaxf(s01, s11);
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) { pm0 = fmaxf(pm0, __shfl_xor_sync(FULL, pm0, o)); pm1 = fmaxf(pm1, __shfl_xor_sync(FULL, pm1, o)); }
  const float nm0 = fmaxf(mx0, pm0), nm1 = fmaxf(mx1, pm1), alpha0 = expf(mx0 - nm0), alpha1 = expf(mx1 - nm1); mx0 = nm0; mx1 = nm1;
  const float p00 = round_bf16(expf(s00 - nm0)), p01 = round_bf16(expf(s01 - nm1)), p10 = round_bf16(expf(s10 - nm0)), p11 = round_bf16(expf(s11 - nm1));
  float ps0 = p00 + p10, ps1 = p01 + p11;
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) { ps0 += __shfl_xor_sync(FULL, ps0, o); ps1 += __shfl_xor_sync(FULL, ps1, o); }
  den0 = fmaf(den0, alpha0, ps0); den1 = fmaf(den1, alpha1, ps1);
  // Probability operand B[k = token][n = head g]: tokens 2t, 2t+1 | 2t+8, 2t+9.
  const uint32_t w_lo = pack(p00, p01), w_hi = pack(p10, p11);
  const uint32_t a_lo = __shfl_sync(FULL, w_lo, src_a), a_hi = __shfl_sync(FULL, w_hi, src_a), b_lo = __shfl_sync(FULL, w_lo, src_b), b_hi = __shfl_sync(FULL, w_hi, src_b);
  const uint32_t pb0 = ((a_lo >> shift) & 0xffffu) | (((b_lo >> shift) & 0xffffu) << 16), pb1 = ((a_hi >> shift) & 0xffffu) | (((b_hi >> shift) & 0xffffu) << 16);
  // Value operand A[m = depth][k = token]: depths 16s+g and 16s+g+8; tokens past the page's count are zero.
  const uint16_t* vp = reinterpret_cast<const uint16_t*>(value_page + g + (2 * t) * D);
  const bool v0 = !MASKED || 2 * t < count, v1 = !MASKED || 2 * t + 1 < count, v2 = !MASKED || 2 * t + 8 < count, v3 = !MASKED || 2 * t + 9 < count;
#pragma unroll
  for (int s = 0; s < 8; ++s) { const uint16_t* e0 = vp + 16 * s; const uint16_t* e1 = e0 + D; const uint16_t* e2 = e0 + 8 * D; const uint16_t* e3 = e2 + D;
    const uint32_t a0 = (v0 ? uint32_t(e0[0]) : 0u) | ((v1 ? uint32_t(e1[0]) : 0u) << 16), a1 = (v0 ? uint32_t(e0[8]) : 0u) | ((v1 ? uint32_t(e1[8]) : 0u) << 16);
    const uint32_t a2 = (v2 ? uint32_t(e2[0]) : 0u) | ((v3 ? uint32_t(e3[0]) : 0u) << 16), a3 = (v2 ? uint32_t(e2[8]) : 0u) | ((v3 ? uint32_t(e3[8]) : 0u) << 16);
    num[s][0] *= alpha0; num[s][1] *= alpha1; num[s][2] *= alpha0; num[s][3] *= alpha1;
    mma(num[s], a0, a1, a2, a3, pb0, pb1); }
}
// SKIP = warps without a page neither store a state nor take part in the merge (same bits as merging their empty state).
template <int CH, int WARPS, bool STAGED, bool SKIP>
__global__ __launch_bounds__(WARPS * 32) void rowwise_mma(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, __nv_bfloat16* out, int cap, int active, float scale) {
  __shared__ float s_max[WARPS][G]; __shared__ float s_den[WARPS][G]; __shared__ float s_w[WARPS][G]; __shared__ float s_num[WARPS][G][D];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3, row = blockIdx.x, kvh = blockIdx.y;
  if (row >= active) return;
  const int T = positions[row] + 1, pages = (T - 1) / PAGE + 1, merge_warps = SKIP ? min(WARPS, pages) : WARPS;
  if (warp < merge_warps) {
    // Query operand B[k = depth][n = head]: head g of this KV head.
    const __nv_bfloat16* qp = query + (size_t(row) * QH + kvh * G + g) * D + 2 * t; uint32_t qb[8][2];
#pragma unroll
    for (int s = 0; s < 8; ++s) { qb[s][0] = *reinterpret_cast<const uint32_t*>(qp + 16 * s); qb[s][1] = *reinterpret_cast<const uint32_t*>(qp + 16 * s + 8); }
    float mx0 = -CUDART_INF_F, mx1 = -CUDART_INF_F, den0 = 0.F, den1 = 0.F; float num[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) { num[s][0] = num[s][1] = num[s][2] = num[s][3] = 0.F; }
    const int src_a = 8 * t + (g >> 1), src_b = src_a + 4, shift = 16 * (g & 1);
    for (int page = warp; page < pages; page += WARPS) {
      const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D; const int count = min(PAGE, T - page * PAGE);
      if (count == PAGE) page_fold<CH, STAGED, false>(key_pool + base, value_pool + base, qb, count, scale, g, t, src_a, src_b, shift, mx0, mx1, den0, den1, num);
      else page_fold<CH, STAGED, true>(key_pool + base, value_pool + base, qb, count, scale, g, t, src_a, src_b, shift, mx0, mx1, den0, den1, num); }
    if (g == 0) { s_max[warp][2 * t] = mx0; s_max[warp][2 * t + 1] = mx1; s_den[warp][2 * t] = den0; s_den[warp][2 * t + 1] = den1; }
#pragma unroll
    for (int s = 0; s < 8; ++s) { s_num[warp][2 * t][16 * s + g] = num[s][0]; s_num[warp][2 * t + 1][16 * s + g] = num[s][1]; s_num[warp][2 * t][16 * s + g + 8] = num[s][2]; s_num[warp][2 * t + 1][16 * s + g + 8] = num[s][3]; }
  }
  __syncthreads();
  if (threadIdx.x < merge_warps * G) { const int w = threadIdx.x / G, h = threadIdx.x % G; float M = -CUDART_INF_F;
    for (int i = 0; i < merge_warps; ++i) M = fmaxf(M, s_max[i][h]);
    s_w[w][h] = expf(s_max[w][h] - M); }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < (G * D) / (WARPS * 32); ++j) { const int idx = threadIdx.x + WARPS * 32 * j, h = idx / D, d = idx % D; float L = 0.F, N = 0.F;
    for (int w = 0; w < merge_warps; ++w) { L = fmaf(s_w[w][h], s_den[w][h], L); N = fmaf(s_w[w][h], s_num[w][h][d], N); }
    out[(size_t(row) * QH + kvh * G + h) * D + d] = __float2bfloat16_rn(L > 0.F ? N / L : 0.F); }
}

static uint16_t bf(float v) { uint32_t b; std::memcpy(&b, &v, 4); b += 0x7fff + ((b >> 16) & 1); return uint16_t(b >> 16); }
static float fb(uint16_t h) { uint32_t b = uint32_t(h) << 16; float v; std::memcpy(&v, &b, 4); return v; }
struct Row { int seq, pos; };
struct Case { const char* name; int cap; int dense; std::vector<int> seq_len; std::vector<Row> rows; };
static int bucket(int n) { static const int b[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256}; for (int x : b) if (x >= n) return x; return n; }
static Case decode_case(const char* name, int cap, const std::vector<int>& ctx) { Case c{name, cap, bucket(int(ctx.size())), ctx, {}}; for (int i = 0; i < int(ctx.size()); ++i) c.rows.push_back({i, ctx[i] - 1}); return c; }
// decode rows (one per context) followed by prefill chunks {first position, token count}, each its own sequence
static Case mixed_case(const char* name, int cap, const std::vector<int>& ctx, const std::vector<std::pair<int, int>>& chunks) { Case c = decode_case(name, cap, ctx); std::vector<Row> rows;
  for (auto& ch : chunks) { const int seq = int(c.seq_len.size()); c.seq_len.push_back(ch.first + ch.second); for (int i = 0; i < ch.second; ++i) rows.push_back({seq, ch.first + i}); }
  rows.insert(rows.end(), c.rows.begin(), c.rows.end()); c.rows = rows; c.dense = bucket(int(rows.size())); return c; }
static std::vector<int> cycle(std::initializer_list<int> v, int n) { std::vector<int> o; std::vector<int> s(v); for (int i = 0; i < n; ++i) o.push_back(s[i % s.size()] + (i * 7) % 23); return o; }

template <typename Launch> static double time_graph(cudaStream_t s, int L, Launch launch) {
  cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); for (int l = 0; l < L; ++l) launch(l); cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&e, g, 0);
  for (int i = 0; i < 5; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); double best = 1e18;
  for (int rep = 0; rep < 9; ++rep) { const double t0 = now(); for (int i = 0; i < 10; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); best = std::min(best, (now() - t0) / (10 * L)); }
  cudaGraphExecDestroy(e); cudaGraphDestroy(g); return best; }

int main() {
  const int L = 36; cudaStream_t s; cudaStreamCreate(&s); const float scale = 1.F / std::sqrt(float(D));
  std::vector<Case> cases;
  cases.push_back(decode_case("C1 fixed decode ctx 140", 16, {140}));
  cases.push_back(decode_case("C1 natural decode ctx 270", 64, {270}));
  cases.push_back(decode_case("C1 natural decode ctx 500", 64, {500}));
  cases.push_back(decode_case("C8 fixed decode", 16, cycle({135}, 8)));
  cases.push_back(decode_case("C8 natural decode", 64, cycle({30, 160, 450}, 8)));
  cases.push_back(decode_case("C16 fixed decode", 16, cycle({135}, 16)));
  cases.push_back(decode_case("C16 natural decode", 64, cycle({30, 160, 450}, 16)));
  cases.push_back(decode_case("C32 fixed decode", 16, cycle({135}, 32)));
  cases.push_back(decode_case("C32 natural decode", 64, cycle({30, 160, 450}, 32)));
  cases.push_back(mixed_case("c16-fixed mixed: 15 decode + 126 prompt", 16, cycle({135}, 15), {{0, 126}}));
  cases.push_back(mixed_case("c32-fixed mixed: 30 decode + 126 + 100 prompt", 16, cycle({135}, 30), {{0, 126}, {0, 100}}));
  cases.push_back(mixed_case("c32-natural mixed: 30 decode + 128 + 16 + 82(0..) prompt", 64, cycle({30, 160, 450}, 30), {{0, 128}, {0, 16}, {0, 82}}));
  cases.push_back(mixed_case("c32-natural mixed: 30 decode + 174(82..255) + 52(0..) prompt", 64, cycle({30, 160, 450}, 30), {{82, 174}, {0, 52}}));
  cases.push_back(mixed_case("prompt only 126 rows at 0", 16, {}, {{0, 126}}));
  cases.push_back(mixed_case("prompt only 16 rows at 0", 64, {}, {{0, 16}}));
  cases.push_back(mixed_case("prompt only 256 rows at 0", 64, {}, {{0, 256}}));
  cases.push_back(mixed_case("prompt only 128 rows at 200", 64, {}, {{200, 128}}));
  cases.push_back(mixed_case("prompt only 224 rows at 174", 64, {}, {{174, 224}}));
  for (const Case& c : cases) {
    const int R = int(c.rows.size()), S = int(c.seq_len.size()); std::vector<int> first_block(S + 1, 0); for (int i = 0; i < S; ++i) first_block[i + 1] = first_block[i] + (c.seq_len[i] + PAGE - 1) / PAGE;
    const int blocks = first_block[S]; const size_t pool = size_t(blocks) * KVH * PAGE * D, qn = size_t(c.dense) * QH * D;
    std::vector<uint16_t> hq(qn), hk(pool * L), hv(pool * L); uint32_t seed = 777u + R * 31u + blocks; auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float((seed >> 8) & 0xffff) / 65536.F - .5F); };
    for (auto& x : hq) x = bf(rnd() * 2.F); for (auto& x : hk) x = bf(rnd() * 2.F); for (auto& x : hv) x = bf(rnd());
    std::vector<uint32_t> hb(size_t(c.dense) * c.cap, 0), hp(c.dense, 0); long page_evals = 0;
    for (int r = 0; r < R; ++r) { hp[r] = c.rows[r].pos; const int pages = c.rows[r].pos / PAGE + 1; page_evals += pages; for (int p = 0; p < pages; ++p) hb[size_t(r) * c.cap + p] = first_block[c.rows[r].seq] + p; }
    __nv_bfloat16 *q, *k, *v, *o_base, *o_new; uint32_t *b, *p; float* st;
    cudaMalloc(&q, qn * 2); cudaMalloc(&k, pool * L * 2); cudaMalloc(&v, pool * L * 2); cudaMalloc(&o_base, qn * 2); cudaMalloc(&o_new, qn * 2); cudaMalloc(&b, hb.size() * 4); cudaMalloc(&p, hp.size() * 4);
    cudaMalloc(&st, size_t(c.dense) * c.cap * QH * STRIDE * 4);
    cudaMemcpy(q, hq.data(), qn * 2, cudaMemcpyHostToDevice); cudaMemcpy(k, hk.data(), pool * L * 2, cudaMemcpyHostToDevice); cudaMemcpy(v, hv.data(), pool * L * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(b, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(p, hp.data(), hp.size() * 4, cudaMemcpyHostToDevice);
    const dim3 pgrid(c.cap, KVH, c.dense), mgrid(QH, c.dense), ngrid(c.dense, KVH);
    auto base = [&](int l) { base_producer<<<pgrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, b, p, st, c.cap, R, scale); base_merge<<<mgrid, 128, 0, s>>>(st, p, o_base, c.cap, R); };
    auto base_prod_only = [&](int l) { base_producer<<<pgrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, b, p, st, c.cap, R, scale); };
#define VARIANT(name, CH, ST, SK) auto name = [&](int l) { rowwise_mma<CH, 8, ST, SK><<<ngrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, b, p, o_new, c.cap, R, scale); };
    VARIANT(v_4_plain_full, 4, false, false) VARIANT(v_4_plain_skip, 4, false, true) VARIANT(v_8_plain_skip, 8, false, true) VARIANT(v_4_staged_skip, 4, true, true) VARIANT(v_8_staged_skip, 8, true, true)
    // skipping empty warps must not change a bit
    std::vector<uint16_t> hob(qn), hon(qn), hfull(qn);
    v_4_plain_full(0); cudaStreamSynchronize(s); cudaMemcpy(hfull.data(), o_new, qn * 2, cudaMemcpyDeviceToHost);
    v_4_plain_skip(0); cudaStreamSynchronize(s); cudaMemcpy(hon.data(), o_new, qn * 2, cudaMemcpyDeviceToHost);
    long skip_diff = 0; for (size_t i = 0; i < size_t(R) * QH * D; ++i) skip_diff += hfull[i] != hon[i];
    base(0); v_8_staged_skip(0); cudaStreamSynchronize(s);
    cudaMemcpy(hob.data(), o_base, qn * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hon.data(), o_new, qn * 2, cudaMemcpyDeviceToHost);
    double worst_new = 0, worst_base = 0, diff = 0; long same = 0, total = 0;
    for (int r = 0; r < R; ++r) for (int h = 0; h < QH; ++h) for (int e = 0; e < D; ++e) { const size_t i = (size_t(r) * QH + h) * D + e; ++total; same += hob[i] == hon[i]; diff = std::max(diff, double(std::fabs(fb(hob[i]) - fb(hon[i])))); }
    for (int r = 0; r < R; r += std::max(1, R / 6)) for (int h = 0; h < QH; h += 5) { const int kvh = h / G, n = c.rows[r].pos + 1, fbk = first_block[c.rows[r].seq]; std::vector<double> sc(n); double mx = -1e30;
        auto at = [&](const std::vector<uint16_t>& pl, int tok, int e) { return double(fb(pl[(size_t(fbk + tok / PAGE) * KVH + kvh) * PAGE * D + size_t(tok % PAGE) * D + e])); };
        for (int j = 0; j < n; ++j) { double d = 0; for (int e = 0; e < D; ++e) d += double(fb(hq[(size_t(r) * QH + h) * D + e])) * at(hk, j, e); sc[j] = d * scale; mx = std::max(mx, sc[j]); }
        double den = 0; for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
        for (int e = 0; e < D; e += 17) { double acc = 0; for (int j = 0; j < n; ++j) acc += sc[j] * at(hv, j, e); const size_t i = (size_t(r) * QH + h) * D + e;
          worst_new = std::max(worst_new, std::fabs(acc / den - fb(hon[i]))); worst_base = std::max(worst_base, std::fabs(acc / den - fb(hob[i]))); } }
    const double tb = time_graph(s, L, base), tp = time_graph(s, L, base_prod_only), t1 = time_graph(s, L, v_4_plain_full), t2 = time_graph(s, L, v_4_plain_skip), t3 = time_graph(s, L, v_8_plain_skip), t4 = time_graph(s, L, v_4_staged_skip), t5 = time_graph(s, L, v_8_staged_skip);
    std::printf("%-58s rows %3d page-evals %5ld | current %6.2f (producer %6.2f) | ch4 full %6.2f  ch4 skip %6.2f  ch8 skip %6.2f  ch4 staged skip %6.2f  ch8 staged skip %6.2f | skip-vs-full differing elements %ld | staged ch8: err vs CPU %.4f (current %.4f), vs current max |diff| %.4f bit-same %.1f%%\n",
                c.name, R, page_evals, tb, tp, t1, t2, t3, t4, t5, skip_diff, worst_new, worst_base, diff, 100.0 * same / total);
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_base); cudaFree(o_new); cudaFree(b); cudaFree(p); cudaFree(st); }
  return 0; }
