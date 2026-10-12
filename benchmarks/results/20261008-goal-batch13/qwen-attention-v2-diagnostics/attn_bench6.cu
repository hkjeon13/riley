// Diagnostic microbenchmark, revision 2: page-parallel two-pass softmax attention (Qwen D128, QH=16, KVH=2, 16-token pages).
// K1 scores per (page, kv head, row) CTA with a shared K tile; K2 softmax statistics + probabilities per (head, row);
// K3 probability-weighted V per (page, kv head, row) CTA; K4 ordered page sum per (head, row).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, G = 8, PAGE = 16, STRIDE = 130;
__device__ __forceinline__ float staged(float dot, float scale) { return __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(dot)) * scale)); }
__device__ __forceinline__ float nan_max(float a, float b) { return (isnan(a) || isnan(b)) ? CUDART_NAN_F : fmaxf(a, b); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o); return v; }
// state[(row * cap + page) * QH + head][STRIDE]: slots 2..17 hold scores, then probabilities; slots 2..129 hold numerators after K3.
__global__ __launch_bounds__(256) void k1_scores(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ float sk[PAGE * D];
  const int page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z, lane = threadIdx.x & 31, hg = threadIdx.x >> 5, head = kvh * G + hg;
  const int T = positions[row] + 1; if (page * PAGE >= T) return;
  const int count = min(PAGE, T - page * PAGE);
  const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D;
  for (int e = threadIdx.x; e < count * D; e += 256) sk[e] = __bfloat162float(key_pool[base + e]);
  const size_t qb = (size_t(row) * QH + head) * D;
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  __syncthreads();
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE;
  for (int t = 0; t < count; ++t) { const float* k = sk + t * D + lane;
    float s = fmaf(q0, k[0], q1 * k[32]); s = fmaf(q2, k[64], s); s = fmaf(q3, k[96], s); s = warp_sum(s);
    if (lane == 0) st[2 + t] = staged(s, scale); } }
__global__ void k2_stats(const uint32_t* positions, float* state, float* stats, int cap) {
  extern __shared__ float red[];
  const int head = blockIdx.x, row = blockIdx.y, T = positions[row] + 1, pages = (T - 1) / PAGE + 1;
  float local = -CUDART_INF_F;
  for (int page = threadIdx.x; page < pages; page += blockDim.x) { const float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE + 2; const int count = min(PAGE, T - page * PAGE);
    for (int t = 0; t < count; ++t) local = nan_max(local, st[t]); }
  red[threadIdx.x] = local; __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) { if (threadIdx.x < off) red[threadIdx.x] = nan_max(red[threadIdx.x], red[threadIdx.x + off]); __syncthreads(); }
  const float m = red[0]; __syncthreads(); float partial = 0.0F;
  for (int page = threadIdx.x; page < pages; page += blockDim.x) { float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE + 2; const int count = min(PAGE, T - page * PAGE);
    for (int t = 0; t < count; ++t) { const float w = expf(st[t] - m); st[t] = w; partial += w; } }
  red[threadIdx.x] = partial; __syncthreads();
  for (int off = blockDim.x / 2; off; off >>= 1) { if (threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off]; __syncthreads(); }
  const float inv = 1.0F / red[0];
  for (int page = threadIdx.x; page < pages; page += blockDim.x) { float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE + 2; const int count = min(PAGE, T - page * PAGE);
    for (int t = 0; t < count; ++t) st[t] = __bfloat162float(__float2bfloat16_rn(st[t] * inv)); }
  if (threadIdx.x == 0) { stats[(size_t(row) * QH + head) * 2] = m; stats[(size_t(row) * QH + head) * 2 + 1] = red[0]; } }
__global__ __launch_bounds__(256) void k3_values(const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap) {
  __shared__ float sv[PAGE * D];
  const int page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z, lane = threadIdx.x & 31, hg = threadIdx.x >> 5, head = kvh * G + hg;
  const int T = positions[row] + 1; if (page * PAGE >= T) return;
  const int count = min(PAGE, T - page * PAGE);
  const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D;
  for (int e = threadIdx.x; e < count * D; e += 256) sv[e] = __bfloat162float(value_pool[base + e]);
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE;
  const float mine = lane < count ? st[2 + lane] : 0.0F;
  __syncthreads();
  float n0 = 0.0F, n1 = 0.0F, n2 = 0.0F, n3 = 0.0F;
  for (int t = 0; t < count; ++t) { const float p = __shfl_sync(0xffffffffu, mine, t); const float* v = sv + t * D + lane;
    n0 = fmaf(p, v[0], n0); n1 = fmaf(p, v[32], n1); n2 = fmaf(p, v[64], n2); n3 = fmaf(p, v[96], n3); }
  st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3; }
__global__ __launch_bounds__(128) void k4_combine(const uint32_t* positions, const float* state, __nv_bfloat16* output, int cap) {
  const int head = blockIdx.x, row = blockIdx.y, dim = threadIdx.x, T = positions[row] + 1, pages = (T - 1) / PAGE + 1;
  const float* st = state + (size_t(row) * cap * QH + head) * STRIDE + 2 + dim; float acc = 0.0F; int page = 0;
  for (; page + 4 <= pages; page += 4) { const float a = st[size_t(page) * QH * STRIDE], b = st[size_t(page + 1) * QH * STRIDE], c = st[size_t(page + 2) * QH * STRIDE], d = st[size_t(page + 3) * QH * STRIDE]; acc += a; acc += b; acc += c; acc += d; }
  for (; page < pages; ++page) acc += st[size_t(page) * QH * STRIDE];
  output[(size_t(row) * QH + head) * D + dim] = __float2bfloat16_rn(acc); }

// ---- current design (shared-KV producer + transition + reduce), same data layout, for a like-for-like baseline ----
__device__ __forceinline__ void online(float score, float* m, float* l, float* alpha, float* beta) {
  const float nm = fmaxf(*m, score); *alpha = *l == 0.0F ? 0.0F : expf(*m - nm); *beta = expf(score - nm); *l = fmaf(*alpha, *l, *beta); *m = nm; }
__device__ __forceinline__ float upd(float n, float v, float a, float b) { if (b == 0.0F) return a * n; if (a == 0.0F) return b * v; return fmaf(b, v, a * n); }
__global__ __launch_bounds__(256) void c1_producer(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ __nv_bfloat16 sk[PAGE * D]; __shared__ __nv_bfloat16 sv[PAGE * D];
  const int page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z, lane = threadIdx.x & 31, hg = threadIdx.x >> 5, head = kvh * G + hg;
  const int T = positions[row] + 1; if (page * PAGE >= T) return;
  const int count = min(PAGE, T - page * PAGE);
  const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D;
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = key_pool[base + e]; sv[e] = value_pool[base + e]; }
  __syncthreads();
  const size_t qb = (size_t(row) * QH + head) * D;
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0, n1 = 0, n2 = 0, n3 = 0;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float s = fmaf(q0, __bfloat162float(sk[tb]), q1 * __bfloat162float(sk[tb + 32])); s = fmaf(q2, __bfloat162float(sk[tb + 64]), s); s = fmaf(q3, __bfloat162float(sk[tb + 96]), s);
    s = warp_sum(s); s = staged(__shfl_sync(0xffffffffu, s, 0), scale);
    float a = 0, b = 0; if (lane == 0) online(s, &m, &l, &a, &b); a = __shfl_sync(0xffffffffu, a, 0); b = __shfl_sync(0xffffffffu, b, 0);
    n0 = upd(n0, __bfloat162float(sv[tb]), a, b); n1 = upd(n1, __bfloat162float(sv[tb + 32]), a, b); n2 = upd(n2, __bfloat162float(sv[tb + 64]), a, b); n3 = upd(n3, __bfloat162float(sv[tb + 96]), a, b); }
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE;
  if (lane == 0) { st[0] = m; st[1] = l; } st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3; }
__global__ void c2_transition(const uint32_t* positions, const float* state, float* steps, float* norm, int cap) {
  if (threadIdx.x) return; const int head = blockIdx.x, row = blockIdx.y, pages = positions[row] / PAGE + 1; float m = -CUDART_INF_F, l = 0.0F;
  for (int p = 0; p < pages; ++p) { const size_t i = (size_t(row) * cap + p) * QH + head; const float om = state[i * STRIDE], ol = state[i * STRIDE + 1]; float ls, rs;
    if (l == 0.0F) { m = om; l = ol; ls = 0; rs = 1; } else { const float nm = fmaxf(m, om); ls = expf(m - nm); rs = expf(om - nm); l = fmaf(ls, l, rs * ol); m = nm; }
    steps[i * 2] = ls; steps[i * 2 + 1] = rs; }
  norm[row * QH + head] = l; }
__global__ __launch_bounds__(128) void c3_reduce(const uint32_t* positions, const float* state, const float* steps, const float* norm, __nv_bfloat16* output, int cap) {
  const int head = blockIdx.x, row = blockIdx.y, dim = threadIdx.x, pages = positions[row] / PAGE + 1; float n = 0.0F;
  for (int p = 0; p < pages; ++p) { const size_t i = (size_t(row) * cap + p) * QH + head; const float ls = steps[i * 2], rs = steps[i * 2 + 1], o = state[i * STRIDE + 2 + dim];
    n = rs == 0.0F ? (ls == 1.0F ? n : ls * n) : (ls == 0.0F ? (rs == 1.0F ? o : rs * o) : fmaf(rs, o, ls * n)); }
  output[(size_t(row) * QH + head) * D + dim] = __float2bfloat16_rn(n / norm[row * QH + head]); }

// ---- candidate: keep the page producer, replace transition+reduce with one global-max merge kernel ----
__global__ __launch_bounds__(128) void m1_merge(const uint32_t* positions, const float* state, __nv_bfloat16* output, int cap) {
  __shared__ float sw[128]; __shared__ float red[128];
  const int head = blockIdx.x, row = blockIdx.y, tid = threadIdx.x, pages = positions[row] / PAGE + 1;
  const float* base = state + (size_t(row) * cap * QH + head) * STRIDE; const size_t step = size_t(QH) * STRIDE;
  float local = -CUDART_INF_F; for (int p = tid; p < pages; p += 128) local = nan_max(local, base[p * step]);
  red[tid] = local; __syncthreads();
  for (int off = 64; off; off >>= 1) { if (tid < off) red[tid] = nan_max(red[tid], red[tid + off]); __syncthreads(); }
  const float M = red[0]; __syncthreads();
  float part = 0.0F; for (int p = tid; p < pages; p += 128) { const float w = expf(base[p * step] - M); if (p < 128) sw[p] = w; part = fmaf(w, base[p * step + 1], part); }
  red[tid] = part; __syncthreads();
  for (int off = 64; off; off >>= 1) { if (tid < off) red[tid] += red[tid + off]; __syncthreads(); }
  const float inv = 1.0F / red[0]; float acc = 0.0F; int p = 0; const float* n = base + 2 + tid;
  for (; p + 4 <= pages; p += 4) { const float a = n[p * step], b = n[(p + 1) * step], c = n[(p + 2) * step], d = n[(p + 3) * step];
    acc = fmaf(sw[p], a, acc); acc = fmaf(sw[p + 1], b, acc); acc = fmaf(sw[p + 2], c, acc); acc = fmaf(sw[p + 3], d, acc); }
  for (; p < pages; ++p) acc = fmaf(sw[p], n[p * step], acc);
  output[(size_t(row) * QH + head) * D + tid] = __float2bfloat16_rn(acc * inv); }

// ---- candidate: four consecutive rows of one sequence share a K/V tile (prefill chunks) ----
__global__ __launch_bounds__(1024) void g1_producer4(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, int rows, float scale) {
  __shared__ __nv_bfloat16 sk[PAGE * D]; __shared__ __nv_bfloat16 sv[PAGE * D];
  const int page = blockIdx.x, kvh = blockIdx.y, group = blockIdx.z, rs = threadIdx.x >> 8, tid = threadIdx.x & 255, lane = tid & 31, hg = tid >> 5, head = kvh * G + hg;
  const int row0 = group * 4, row = row0 + rs;
  int maxcount = 0;
  #pragma unroll
  for (int i = 0; i < 4; ++i) { const int r = row0 + i; if (r < rows) { const int Tr = positions[r] + 1; if (page * PAGE < Tr) maxcount = max(maxcount, min(PAGE, Tr - page * PAGE)); } }
  if (maxcount == 0) return;                      // CTA-uniform
  const size_t base = (size_t(block_ids[row0 * cap + page]) * KVH + kvh) * PAGE * D;   // bench: the group shares its pages
  for (int e = threadIdx.x; e < maxcount * D; e += 1024) { sk[e] = key_pool[base + e]; sv[e] = value_pool[base + e]; }
  __syncthreads();
  if (row >= rows) return; const int T = positions[row] + 1; if (page * PAGE >= T) return;   // warp-uniform, after the only barrier
  const int count = min(PAGE, T - page * PAGE);
  const size_t qb = (size_t(row) * QH + head) * D;
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0, n1 = 0, n2 = 0, n3 = 0;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float sc = fmaf(q0, __bfloat162float(sk[tb]), q1 * __bfloat162float(sk[tb + 32])); sc = fmaf(q2, __bfloat162float(sk[tb + 64]), sc); sc = fmaf(q3, __bfloat162float(sk[tb + 96]), sc);
    sc = warp_sum(sc); sc = staged(__shfl_sync(0xffffffffu, sc, 0), scale);
    float a = 0, b = 0; if (lane == 0) online(sc, &m, &l, &a, &b); a = __shfl_sync(0xffffffffu, a, 0); b = __shfl_sync(0xffffffffu, b, 0);
    n0 = upd(n0, __bfloat162float(sv[tb]), a, b); n1 = upd(n1, __bfloat162float(sv[tb + 32]), a, b); n2 = upd(n2, __bfloat162float(sv[tb + 64]), a, b); n3 = upd(n3, __bfloat162float(sv[tb + 96]), a, b); }
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE;
  if (lane == 0) { st[0] = m; st[1] = l; } st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3; }
int main() {
  const int cfg[][3] = {{224, 2, 18}, {224, 4, 100}, {128, 1, 200}, {224, 8, 4}, {64, 2, 300}};
  for (auto& c : cfg) { const int R = c[0], S = c[1], START = c[2], per = R / S, T = START + per, cap = (T + PAGE - 1) / PAGE; const size_t pool = size_t(S) * cap * KVH * PAGE * D;
    __nv_bfloat16 *q, *kp, *vp, *out; uint32_t *ids, *pos; float *state, *state2;
    cudaMalloc(&q, size_t(R) * QH * D * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&out, size_t(R) * QH * D * 2); cudaMalloc(&ids, size_t(R) * cap * 4); cudaMalloc(&pos, R * 4);
    cudaMalloc(&state, size_t(R) * cap * QH * STRIDE * 4); cudaMalloc(&state2, size_t(R) * cap * QH * STRIDE * 4); cudaMemset(state, 0, size_t(R) * cap * QH * STRIDE * 4); cudaMemset(state2, 0, size_t(R) * cap * QH * STRIDE * 4);
    std::vector<__nv_bfloat16> h(pool), hq(size_t(R) * QH * D); for (size_t i = 0; i < pool; ++i) h[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F); for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2bfloat16(float((i * 40503u) % 1999) / 1000.0F - 1.0F);
    cudaMemcpy(kp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    std::vector<uint32_t> hi(size_t(R) * cap), hp(R);
    for (int r = 0; r < R; ++r) { const int seq = r / per; hp[r] = START + (r % per); for (int p = 0; p < cap; ++p) hi[size_t(r) * cap + p] = uint32_t((size_t(seq) * cap + p) * 7 % (size_t(S) * cap) == 0 ? seq * cap + p : seq * cap + p); }
    cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(pos, hp.data(), R * 4, cudaMemcpyHostToDevice);
    const float scale = 1.0F / std::sqrt(float(D)); const dim3 pg(cap, KVH, R), pg4(cap, KVH, (R + 3) / 4);
    auto timeit0 = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 200; };
    const double a = timeit0([&] { c1_producer<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); });
    const double b = timeit0([&] { g1_producer4<<<pg4, 1024>>>(q, kp, vp, ids, pos, state2, cap, R, scale); });
    cudaDeviceSynchronize(); const size_t n = size_t(R) * cap * QH * STRIDE; std::vector<float> s1(n), s2(n); cudaMemcpy(s1.data(), state, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s2.data(), state2, n * 4, cudaMemcpyDeviceToHost);
    size_t diff = 0; for (size_t i = 0; i < n; ++i) if (std::memcmp(&s1[i], &s2[i], 4) != 0) ++diff;
    std::printf("prefill rows=%3d over %d sequences (positions %d..%d, %d pages): per-row CTA %6.1f us, four-row shared tile %6.1f us | state words differing %zu of %zu (err %d)\n", R, S, START, T - 1, cap, a, b, diff, n, int(cudaGetLastError()));
    cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(out); cudaFree(ids); cudaFree(pos); cudaFree(state); cudaFree(state2); }
  return 0; }
