// Diagnostic harness: the runtime's row-wise attention kernel, compiled verbatim from extracted source variants
// (kernel_*.inc), timed against the paged pair (shared-KV producer + global-max merge) and compared bit for bit
// between variants. Timing: CUDA graph replay of 36 layers with distinct K/V pools.
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
constexpr uint32_t kWarpSize = 32; constexpr uint32_t kFullWarpMask = 0xffffffffu;
#define RILEY_CUDA_PAGED_KV_BLOCK_SIZE 16
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount = 16, kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount = 2, kNativeBf16RaggedPagedSplitGqaD128HeadSize = 128;
struct DeviceBatch { const uint32_t* sequence_block_offsets; const uint32_t* block_ids; const uint16_t* valid_tokens; const uint32_t* row_sequence_slots; const uint32_t* row_positions;
  uint64_t sequence_count; uint64_t block_count; uint64_t active_row_count; uint64_t physical_block_count; const uint32_t* dynamic_counts; };
__device__ __forceinline__ uint64_t active_rows(const DeviceBatch& batch) { return batch.dynamic_counts != nullptr ? static_cast<uint64_t>(batch.dynamic_counts[0]) : batch.active_row_count; }
__device__ __forceinline__ bool resolve_ragged_partial_block(const DeviceBatch& batch, uint64_t row, uint64_t logical_block, uint64_t key_value_head, uint64_t* block_head_base, uint64_t* token_count) {
  if (block_head_base == nullptr || token_count == nullptr || row >= active_rows(batch)) return false;
  const uint64_t sequence = batch.row_sequence_slots[row]; if (sequence >= batch.sequence_count) return false;
  const uint64_t block_begin = batch.sequence_block_offsets[sequence]; const uint64_t block_end = batch.sequence_block_offsets[sequence + 1];
  if (block_begin > block_end || block_end > batch.block_count || logical_block >= block_end - block_begin) return false;
  const uint64_t logical_token_count = static_cast<uint64_t>(batch.row_positions[row]) + 1; const uint64_t block_token_begin = logical_block * RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (block_token_begin >= logical_token_count) return false;
  const uint64_t block_index = block_begin + logical_block; const uint64_t physical_block = batch.block_ids[block_index]; const uint64_t valid = batch.valid_tokens[block_index];
  uint64_t required = logical_token_count - block_token_begin; if (required > RILEY_CUDA_PAGED_KV_BLOCK_SIZE) required = RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (physical_block >= batch.physical_block_count || valid < required || valid == 0 || valid > RILEY_CUDA_PAGED_KV_BLOCK_SIZE) return false;
  *block_head_base = (physical_block * kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount + key_value_head) * RILEY_CUDA_PAGED_KV_BLOCK_SIZE * kNativeBf16RaggedPagedSplitGqaD128HeadSize; *token_count = required; return true; }
__device__ __forceinline__ float staged_attention_score(float dot_product, float scale) { const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product); return __bfloat162float(__float2bfloat16_rn(__bfloat162float(staged_dot) * scale)); }
__device__ __forceinline__ uint32_t pack_bf16_pair(float first, float second) { return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(first))) | (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(second))) << 16); }
__device__ __forceinline__ void mma_m16n8k16_bf16(float* accumulator, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
#ifndef VARIANT_A
#define VARIANT_A "kernel_v4.inc"
#endif
#ifndef VARIANT_B
#define VARIANT_B "kernel_v5.inc"
#endif
namespace variant_a {
#include VARIANT_A
}
namespace variant_b {
#include VARIANT_B
}
#include "kernel_chain.inc"
#include "kernel_chain2.inc"

// ---- paged pair (as in rowwise_mma.cu) ----
constexpr int D = 128, QH = 16, KVH = 2, G = 8, PAGE = 16, STRIDE = 130;
__device__ __forceinline__ float warp_sum_all(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(kFullWarpMask, v, o);
  return v; }
__device__ __forceinline__ void online(float score, float* m, float* l, float* alpha, float* beta) { const float nm = fmaxf(*m, score); *alpha = *l == 0.0F ? 0.0F : expf(*m - nm); *beta = expf(score - nm); *l = fmaf(*alpha, *l, *beta); *m = nm; }
__global__ __launch_bounds__(256) void base_producer(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, DeviceBatch batch, float* states, int cap, float scale) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z;
  if (row >= (int)active_rows(batch)) return;
  const int T = batch.row_positions[row] + 1, pages = (T - 1) / PAGE + 1; if (page >= pages || page >= cap) return;
  const int head = kvh * G + warp; uint64_t base = 0, count64 = 0; const bool ok = resolve_ragged_partial_block(batch, row, page, kvh, &base, &count64); const int count = ok ? int(count64) : 0;
  const size_t qb = (size_t(row) * QH + head) * D;
  __shared__ float sk[PAGE * D]; __shared__ float sv[PAGE * D];
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = __bfloat162float(key_pool[base + e]); sv[e] = __bfloat162float(value_pool[base + e]); }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.F, n0 = 0.F, n1 = 0.F, n2 = 0.F, n3 = 0.F;
  for (int t = 0; t < count; ++t) { const int b = t * D + lane;
    float s = fmaf(q0, sk[b], q1 * sk[b + 32]); s = fmaf(q2, sk[b + 64], s); s = fmaf(q3, sk[b + 96], s); s = staged_attention_score(warp_sum_all(s), scale);
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

static uint16_t bf(float v) { uint32_t b; std::memcpy(&b, &v, 4); b += 0x7fff + ((b >> 16) & 1); return uint16_t(b >> 16); }
struct Row { int seq, pos; };
struct Case { const char* name; int cap; int dense; std::vector<int> seq_len; std::vector<Row> rows; };
static int bucket(int n) { static const int b[] = {1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256}; for (int x : b) if (x >= n) return x; return n; }
static Case decode_case(const char* name, int cap, const std::vector<int>& ctx) { Case c{name, cap, bucket(int(ctx.size())), ctx, {}}; for (int i = 0; i < int(ctx.size()); ++i) c.rows.push_back({i, ctx[i] - 1}); return c; }
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
  cases.push_back(decode_case("C1 decode ctx 40", 64, {40}));
  cases.push_back(decode_case("C1 decode ctx 140", 16, {140}));
  cases.push_back(decode_case("C1 decode ctx 270", 64, {270}));
  cases.push_back(decode_case("C1 decode ctx 400", 64, {400}));
  cases.push_back(decode_case("C1 decode ctx 500", 64, {500}));
  cases.push_back(decode_case("C1 decode ctx 1000", 64, {1000}));
  cases.push_back(decode_case("C8 fixed decode", 16, cycle({135}, 8)));
  cases.push_back(decode_case("C8 natural decode", 64, cycle({30, 160, 450}, 8)));
  cases.push_back(decode_case("C16 natural decode", 64, cycle({30, 160, 450}, 16)));
  cases.push_back(decode_case("C32 fixed decode", 16, cycle({135}, 32)));
  cases.push_back(decode_case("C32 natural decode", 64, cycle({30, 160, 450}, 32)));
  cases.push_back(mixed_case("c16-fixed mixed: 15 decode + 126 prompt", 16, cycle({135}, 15), {{0, 126}}));
  cases.push_back(mixed_case("c32-fixed mixed: 30 decode + 126 + 100 prompt", 16, cycle({135}, 30), {{0, 126}, {0, 100}}));
  cases.push_back(mixed_case("c32-natural mixed: 30 decode + 128 + 16 + 82 prompt", 64, cycle({30, 160, 450}, 30), {{0, 128}, {0, 16}, {0, 82}}));
  cases.push_back(mixed_case("c32-natural mixed: 30 decode + 174(82..255) + 52 prompt", 64, cycle({30, 160, 450}, 30), {{82, 174}, {0, 52}}));
  cases.push_back(mixed_case("c1-natural prefill 256 rows at 0", 64, {}, {{0, 256}}));
  cases.push_back(mixed_case("c1-natural prefill 142 rows at 256", 64, {}, {{256, 142}}));
  for (const Case& c : cases) {
    const int R = int(c.rows.size()), S = int(c.seq_len.size()); std::vector<uint32_t> offsets(S + 1, 0); for (int i = 0; i < S; ++i) offsets[i + 1] = offsets[i] + (c.seq_len[i] + PAGE - 1) / PAGE;
    const int blocks = offsets[S]; const size_t pool = size_t(blocks) * KVH * PAGE * D, qn = size_t(c.dense) * QH * D;
    std::vector<uint16_t> hq(qn), hk(pool * L), hv(pool * L); uint32_t seed = 777u + R * 31u + blocks; auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float((seed >> 8) & 0xffff) / 65536.F - .5F); };
    for (auto& x : hq) x = bf(rnd() * 2.F); for (auto& x : hk) x = bf(rnd() * 2.F); for (auto& x : hv) x = bf(rnd());
    // shuffled physical pages, as a real pool would hold them
    std::vector<uint32_t> ids(blocks); for (int i = 0; i < blocks; ++i) ids[i] = uint32_t((size_t(i) * 29 + 5) % blocks); if (blocks % 29 == 0) for (int i = 0; i < blocks; ++i) ids[i] = i;
    std::vector<uint16_t> valid(blocks, PAGE); std::vector<uint32_t> slots(c.dense, 0), pos(c.dense, 0); long page_evals = 0;
    for (int r = 0; r < R; ++r) { slots[r] = c.rows[r].seq; pos[r] = c.rows[r].pos; page_evals += c.rows[r].pos / PAGE + 1; }
    __nv_bfloat16 *o_cs, *o_cc; float* chain_states; cudaMalloc(&o_cs, qn * 2); cudaMalloc(&o_cc, qn * 2); cudaMalloc(&chain_states, size_t(c.dense) * 2 * kChainCount * kChainStateFloats * 4);
    __nv_bfloat16 *q, *k, *v, *o_base, *o_a, *o_b; uint32_t *d_off, *d_ids, *d_slots, *d_pos; uint16_t* d_valid; float* st;
    cudaMalloc(&q, qn * 2); cudaMalloc(&k, pool * L * 2); cudaMalloc(&v, pool * L * 2); cudaMalloc(&o_base, qn * 2); cudaMalloc(&o_a, qn * 2); cudaMalloc(&o_b, qn * 2);
    cudaMalloc(&d_off, offsets.size() * 4); cudaMalloc(&d_ids, ids.size() * 4); cudaMalloc(&d_valid, valid.size() * 2); cudaMalloc(&d_slots, slots.size() * 4); cudaMalloc(&d_pos, pos.size() * 4);
    cudaMalloc(&st, size_t(c.dense) * c.cap * QH * STRIDE * 4);
    cudaMemcpy(q, hq.data(), qn * 2, cudaMemcpyHostToDevice); cudaMemcpy(k, hk.data(), pool * L * 2, cudaMemcpyHostToDevice); cudaMemcpy(v, hv.data(), pool * L * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, offsets.data(), offsets.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_ids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_valid, valid.data(), valid.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_pos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice);
    const DeviceBatch batch{d_off, d_ids, d_valid, d_slots, d_pos, uint64_t(S), uint64_t(blocks), uint64_t(R), uint64_t(blocks), nullptr};
    const dim3 pgrid(c.cap, KVH, c.dense), mgrid(QH, c.dense), ngrid(c.dense, KVH);
    auto base = [&](int l) { base_producer<<<pgrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, batch, st, c.cap, scale); base_merge<<<mgrid, 128, 0, s>>>(st, d_pos, o_base, c.cap, R); };
    auto ka = [&](int l) { variant_a::native_bf16_ragged_d128_rowwise_mma_kernel<<<ngrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, o_a, batch, uint64_t(c.cap), scale); };
    auto kb = [&](int l) { variant_b::native_bf16_ragged_d128_rowwise_mma_kernel<<<ngrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, o_b, batch, uint64_t(c.cap), scale); };
    const dim3 cgrid(kChainCount, KVH, c.dense);
    auto kcs = [&](int l) { chain_attention_single_kernel<<<ngrid, 256, 0, s>>>(q, k + pool * l, v + pool * l, o_cs, batch, uint64_t(c.cap), scale); };
    auto kcc = [&](int l) { chain_attention_producer_kernel<<<cgrid, 32, 0, s>>>(q, k + pool * l, v + pool * l, chain_states, batch, uint64_t(c.cap), scale); chain_attention_merge_kernel<<<ngrid, 256, 0, s>>>(chain_states, o_cc, batch, uint64_t(c.cap)); };
    auto kcp = [&](int l) { chain_attention_producer_kernel<<<cgrid, 32, 0, s>>>(q, k + pool * l, v + pool * l, chain_states, batch, uint64_t(c.cap), scale); };
    __nv_bfloat16* o_c2; cudaMalloc(&o_c2, qn * 2);
    auto kc2 = [&](int l) { chain2::producer_kernel<<<cgrid, 32, 0, s>>>(q, k + pool * l, v + pool * l, chain_states, batch, uint64_t(c.cap), scale); chain2::merge_kernel<<<ngrid, 256, 0, s>>>(chain_states, o_c2, batch, uint64_t(c.cap)); };
    auto kc2p = [&](int l) { chain2::producer_kernel<<<cgrid, 32, 0, s>>>(q, k + pool * l, v + pool * l, chain_states, batch, uint64_t(c.cap), scale); };
    kc2(0); cudaStreamSynchronize(s); std::vector<uint16_t> hc2(qn); cudaMemcpy(hc2.data(), o_c2, qn * 2, cudaMemcpyDeviceToHost);
    kcs(0); kcc(0);
    base(0); ka(0); kb(0); cudaStreamSynchronize(s);
    std::vector<uint16_t> hcs(qn), hcc(qn); cudaMemcpy(hcs.data(), o_cs, qn * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hcc.data(), o_cc, qn * 2, cudaMemcpyDeviceToHost);
    std::vector<uint16_t> hob(qn), hoa(qn), hobb(qn); cudaMemcpy(hob.data(), o_base, qn * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hoa.data(), o_a, qn * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hobb.data(), o_b, qn * 2, cudaMemcpyDeviceToHost);
    long ab = 0, apaged = 0; for (size_t i = 0; i < size_t(R) * QH * D; ++i) { ab += hoa[i] != hobb[i]; apaged += hoa[i] != hob[i]; }
    long geo = 0, cpaged = 0, ca = 0, c2 = 0; for (size_t i = 0; i < size_t(R) * QH * D; ++i) { geo += hcs[i] != hcc[i]; cpaged += hcs[i] != hob[i]; ca += hcs[i] != hoa[i]; c2 += hc2[i] != hcc[i]; }
    const double tc2 = time_graph(s, L, kc2), tc2p = time_graph(s, L, kc2p);
    const double tb = time_graph(s, L, base), ta = time_graph(s, L, ka), tbb = time_graph(s, L, kb), tcs = time_graph(s, L, kcs), tcc = time_graph(s, L, kcc), tcp = time_graph(s, L, kcp);
    std::printf("%-56s rows %3d | paged %6.2f | v4 %6.2f | chain: one-CTA %6.2f, CTAs+merge %6.2f (prod %6.2f) | staged CTAs+merge %6.2f (prod %6.2f) us | differ: geometries %ld, staged %ld | chain vs paged %ld, vs v4 %ld of %zu\n", c.name, R, tb, ta, tcs, tcc, tcp, tc2, tc2p, geo, c2, cpaged, ca, size_t(R) * QH * D);
    cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o_base); cudaFree(o_a); cudaFree(o_b); cudaFree(d_off); cudaFree(d_ids); cudaFree(d_valid); cudaFree(d_slots); cudaFree(d_pos); cudaFree(st); }
  return 0; }
