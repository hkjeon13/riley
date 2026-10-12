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
#include <cstdint>
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
constexpr unsigned FULL = 0xffffffffu;
__device__ __forceinline__ float staged10(float dot, float scale) { const __nv_bfloat16 s = __float2bfloat16_rn(dot); return __bfloat162float(__float2bfloat16_rn(__bfloat162float(s) * scale)); }
__device__ __forceinline__ float warp_sum_all(float v) { for (int o = 16; o != 0; o /= 2) v += __shfl_xor_sync(FULL, v, o); return v; }
__device__ __forceinline__ void online10(float score, float* maximum, float* denominator, float* alpha, float* beta) {
  if (isnan(score) || isnan(*maximum)) { *maximum = CUDART_NAN_F; *denominator = CUDART_NAN_F; *alpha = CUDART_NAN_F; *beta = CUDART_NAN_F; return; }
  const bool sp = isinf(score) && score > 0.0F; const bool mp = isinf(*maximum) && *maximum > 0.0F;
  if (sp) { if (mp) { *alpha = 1.0F; *beta = 1.0F; *denominator += 1.0F; } else { *maximum = CUDART_INF_F; *denominator = 1.0F; *alpha = 0.0F; *beta = 1.0F; } return; }
  if (mp || (isinf(score) && score < 0.0F)) { *alpha = 1.0F; *beta = 0.0F; return; }
  const float next_maximum = fmaxf(*maximum, score); *alpha = *denominator == 0.0F ? 0.0F : expf(*maximum - next_maximum); *beta = expf(score - next_maximum);
  *denominator = fmaf(*alpha, *denominator, *beta); *maximum = next_maximum; }
__device__ __forceinline__ float upd10(float numerator, float value, float alpha, float beta) { if (beta == 0.0F) return alpha * numerator; if (alpha == 0.0F) return beta * value; return fmaf(beta, value, alpha * numerator); }
#define HEADER \
  const int page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, head = kvh * G + warp; \
  const int T = positions[row] + 1; if (page * PAGE >= T) return; const int count = min(PAGE, T - page * PAGE); \
  const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D; const size_t qb = (size_t(row) * QH + head) * D;
#define FOOTER \
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE; if (lane == 0) { st[0] = m; st[1] = l; } st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3;
__global__ __launch_bounds__(256) void p_bfly(const __nv_bfloat16* query, const __nv_bfloat16* kp, const __nv_bfloat16* vp, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ float sk[PAGE * D]; __shared__ float sv[PAGE * D]; HEADER
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = __bfloat162float(kp[base + e]); sv[e] = __bfloat162float(vp[base + e]); }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0.0F, n1 = 0.0F, n2 = 0.0F, n3 = 0.0F;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float s = fmaf(q0, sk[tb], q1 * sk[tb + 32]); s = fmaf(q2, sk[tb + 64], s); s = fmaf(q3, sk[tb + 96], s);
    s = staged10(warp_sum_all(s), scale);
    float a = 0.0F, b = 0.0F; online10(s, &m, &l, &a, &b);
    const float v0 = sv[tb], v1 = sv[tb + 32], v2 = sv[tb + 64], v3 = sv[tb + 96];
    if (b == 0.0F) { n0 = a * n0; n1 = a * n1; n2 = a * n2; n3 = a * n3; }
    else if (a == 0.0F) { n0 = b * v0; n1 = b * v1; n2 = b * v2; n3 = b * v3; }
    else { n0 = fmaf(b, v0, a * n0); n1 = fmaf(b, v1, a * n1); n2 = fmaf(b, v2, a * n2); n3 = fmaf(b, v3, a * n3); } }
  FOOTER }
int main() {
  const int cfg[][4] = {{32, 400, 32, 400}, {256, 400, 32, 130}, {160, 400, 32, 200}, {96, 330, 32, 330}, {256, 130, 256, 130}, {224, 398, 0, 300}, {128, 328, 0, 264}};
  for (auto& c : cfg) { const int R = c[0], T = c[1], LONG = c[2], SHORT = c[3], cap = (T + PAGE - 1) / PAGE; const size_t pool = size_t(R) * cap * KVH * PAGE * D;
    __nv_bfloat16 *q, *kp, *vp, *out; uint32_t *ids, *pos; float *state, *stats;
    cudaMalloc(&q, size_t(R) * QH * D * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&out, size_t(R) * QH * D * 2); cudaMalloc(&ids, size_t(R) * cap * 4); cudaMalloc(&pos, R * 4);
    cudaMalloc(&state, size_t(R) * cap * QH * STRIDE * 4); cudaMalloc(&stats, size_t(R) * QH * 8);
    std::vector<__nv_bfloat16> h(pool); for (size_t i = 0; i < pool; ++i) h[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F);
    cudaMemcpy(kp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, h.data(), size_t(R) * QH * D * 2, cudaMemcpyHostToDevice);
    std::vector<uint32_t> hi(size_t(R) * cap), hp(R, T - 1); for (int r = LONG; r < R; ++r) hp[r] = SHORT - 1; const size_t n = hi.size(); size_t mult = 7919; while (n % mult == 0 || std::fmod(double(mult), 2.0) == 0) ++mult;
    { std::vector<char> seen(n); bool ok = true; for (size_t i = 0; i < n; ++i) { hi[i] = uint32_t((i * mult) % n); if (seen[hi[i]]) ok = false; seen[hi[i]] = 1; } if (!ok) for (size_t i = 0; i < n; ++i) hi[i] = uint32_t(i); }
    cudaMemcpy(ids, hi.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(pos, hp.data(), R * 4, cudaMemcpyHostToDevice);
    auto timeit0 = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 200; };
    const float scale = 1.0F / std::sqrt(float(D)); const dim3 pg(cap, KVH, R), hr(QH, R); int st_threads = 1; while (st_threads < cap && st_threads < 128) st_threads <<= 1;
    auto k1 = [&] { k1_scores<<<pg, 256>>>(q, kp, ids, pos, state, cap, scale); };
    auto k2 = [&] { k2_stats<<<hr, st_threads, st_threads * 4>>>(pos, state, stats, cap); };
    auto k3 = [&] { k3_values<<<pg, 256>>>(vp, ids, pos, state, cap); };
    auto k4 = [&] { k4_combine<<<hr, 128>>>(pos, state, out, cap); };
    float *steps, *norm; cudaMalloc(&steps, size_t(R) * cap * QH * 8); cudaMalloc(&norm, size_t(R) * QH * 4);
    auto cur = [&] { c1_producer<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); c2_transition<<<hr, 32>>>(pos, state, steps, norm, cap); c3_reduce<<<hr, 128>>>(pos, state, steps, norm, out, cap); };
    auto curp = [&] { c1_producer<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); };
    auto bf = [&] { p_bfly<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); m1_merge<<<hr, 128>>>(pos, state, out, cap); }; const double base_all = timeit0(bf), base_prod = timeit0([&] { p_bfly<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); });
    bf(); cudaDeviceSynchronize(); std::vector<__nv_bfloat16> hb(size_t(R) * QH * D); cudaMemcpy(hb.data(), out, hb.size() * 2, cudaMemcpyDeviceToHost);
    auto mer = [&] { m1_merge<<<hr, 128>>>(pos, state, out, cap); }; auto curm = [&] { c1_producer<<<pg, 256>>>(q, kp, vp, ids, pos, state, cap, scale); m1_merge<<<hr, 128>>>(pos, state, out, cap); };
    auto tr = [&] { c2_transition<<<hr, 32>>>(pos, state, steps, norm, cap); c3_reduce<<<hr, 128>>>(pos, state, steps, norm, out, cap); };

    auto all = [&] { k1(); k2(); k3(); k4(); };
    auto timeit = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 200; };
    const double a = timeit(k1), b = timeit(k2), cc = timeit(k3), d = timeit(k4), tot = timeit(all);
    all(); cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> ho(size_t(R) * QH * D); cudaMemcpy(ho.data(), out, ho.size() * 2, cudaMemcpyDeviceToHost); double sum = 0; int nan = 0; for (auto v : ho) { float f = __bfloat162float(v); if (std::isnan(f)) ++nan; else sum += f; }
    double maxdiff = 0; int ulp_gt1 = 0; for (size_t i = 0; i < ho.size(); ++i) { const double df = std::fabs(double(__bfloat162float(ho[i])) - double(__bfloat162float(hb[i]))); if (df > maxdiff) maxdiff = df; const double step = std::fabs(double(__bfloat162float(hb[i]))) / 128.0; if (df > step) ++ulp_gt1; }
    std::printf("rows=%3d T=%4d: butterfly producer + global merge %6.1f (producer %6.1f) | TWO-PASS scores %5.1f stats %5.1f values %5.1f combine %4.1f all %6.1f us | max|new-cur| %.5f, elems differing by more than 2^-7 relative: %d of %zu, nan %d\n", R, T, base_all, base_prod, a, b, cc, d, tot, maxdiff, ulp_gt1, ho.size(), nan);
    cudaFree(steps); cudaFree(norm);
    cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(out); cudaFree(ids); cudaFree(pos); cudaFree(state); cudaFree(stats); }
  return 0; }
