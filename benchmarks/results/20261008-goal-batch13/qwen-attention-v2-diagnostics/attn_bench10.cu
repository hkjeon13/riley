// Diagnostic microbenchmark 10: instruction-count variants of the shared-KV D128 partial-state producer.
// base  = serving kernel (BF16 tiles in shared memory, lane-0 online update, three broadcasts per token).
// ftile = K/V tiles converted to float once when staged.
// bfly  = ftile + xor-butterfly score reduction (every lane ends with the same bits as lane 0 of the shfl_down tree,
//         because each level adds the same two subtree sums), so the score/alpha/beta broadcasts disappear and every
//         lane runs the scalar online update; numerator branches are taken once per token instead of per component.
// All variants must write bit-identical [m, l, n[128]] states.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, G = 8, PAGE = 16, STRIDE = 130; constexpr unsigned FULL = 0xffffffffu;
__device__ __forceinline__ float staged(float dot, float scale) { const __nv_bfloat16 s = __float2bfloat16_rn(dot); return __bfloat162float(__float2bfloat16_rn(__bfloat162float(s) * scale)); }
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o != 0; o /= 2) v += __shfl_down_sync(FULL, v, o); return v; }
__device__ __forceinline__ float warp_sum_all(float v) { for (int o = 16; o != 0; o /= 2) v += __shfl_xor_sync(FULL, v, o); return v; }
__device__ __forceinline__ void update_online_state(float score, float* maximum, float* denominator, float* alpha, float* beta) {
  if (isnan(score) || isnan(*maximum)) { *maximum = CUDART_NAN_F; *denominator = CUDART_NAN_F; *alpha = CUDART_NAN_F; *beta = CUDART_NAN_F; return; }
  const bool sp = isinf(score) && score > 0.0F; const bool mp = isinf(*maximum) && *maximum > 0.0F;
  if (sp) { if (mp) { *alpha = 1.0F; *beta = 1.0F; *denominator += 1.0F; } else { *maximum = CUDART_INF_F; *denominator = 1.0F; *alpha = 0.0F; *beta = 1.0F; } return; }
  if (mp || (isinf(score) && score < 0.0F)) { *alpha = 1.0F; *beta = 0.0F; return; }
  const float next_maximum = fmaxf(*maximum, score); *alpha = *denominator == 0.0F ? 0.0F : expf(*maximum - next_maximum); *beta = expf(score - next_maximum);
  *denominator = fmaf(*alpha, *denominator, *beta); *maximum = next_maximum; }
__device__ __forceinline__ float update_numerator(float numerator, float value, float alpha, float beta) { if (beta == 0.0F) return alpha * numerator; if (alpha == 0.0F) return beta * value; return fmaf(beta, value, alpha * numerator); }
#define HEADER \
  const int page = blockIdx.x, kvh = blockIdx.y, row = blockIdx.z, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, head = kvh * G + warp; \
  const int T = positions[row] + 1; if (page * PAGE >= T) return; const int count = min(PAGE, T - page * PAGE); \
  const size_t base = (size_t(block_ids[row * cap + page]) * KVH + kvh) * PAGE * D; const size_t qb = (size_t(row) * QH + head) * D;
#define FOOTER \
  float* st = state + ((size_t(row) * cap + page) * QH + head) * STRIDE; if (lane == 0) { st[0] = m; st[1] = l; } st[2 + lane] = n0; st[2 + lane + 32] = n1; st[2 + lane + 64] = n2; st[2 + lane + 96] = n3;
__global__ __launch_bounds__(256) void p_base(const __nv_bfloat16* query, const __nv_bfloat16* kp, const __nv_bfloat16* vp, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ __nv_bfloat16 sk[PAGE * D]; __shared__ __nv_bfloat16 sv[PAGE * D]; HEADER
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = kp[base + e]; sv[e] = vp[base + e]; }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0.0F, n1 = 0.0F, n2 = 0.0F, n3 = 0.0F;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float s = fmaf(q0, __bfloat162float(sk[tb]), q1 * __bfloat162float(sk[tb + 32])); s = fmaf(q2, __bfloat162float(sk[tb + 64]), s); s = fmaf(q3, __bfloat162float(sk[tb + 96]), s);
    s = warp_sum(s); s = staged(__shfl_sync(FULL, s, 0), scale);
    float a = 0.0F, b = 0.0F; if (lane == 0) update_online_state(s, &m, &l, &a, &b); a = __shfl_sync(FULL, a, 0); b = __shfl_sync(FULL, b, 0);
    n0 = update_numerator(n0, __bfloat162float(sv[tb]), a, b); n1 = update_numerator(n1, __bfloat162float(sv[tb + 32]), a, b); n2 = update_numerator(n2, __bfloat162float(sv[tb + 64]), a, b); n3 = update_numerator(n3, __bfloat162float(sv[tb + 96]), a, b); }
  FOOTER }
__global__ __launch_bounds__(256) void p_ftile(const __nv_bfloat16* query, const __nv_bfloat16* kp, const __nv_bfloat16* vp, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ float sk[PAGE * D]; __shared__ float sv[PAGE * D]; HEADER
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = __bfloat162float(kp[base + e]); sv[e] = __bfloat162float(vp[base + e]); }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0.0F, n1 = 0.0F, n2 = 0.0F, n3 = 0.0F;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float s = fmaf(q0, sk[tb], q1 * sk[tb + 32]); s = fmaf(q2, sk[tb + 64], s); s = fmaf(q3, sk[tb + 96], s);
    s = warp_sum(s); s = staged(__shfl_sync(FULL, s, 0), scale);
    float a = 0.0F, b = 0.0F; if (lane == 0) update_online_state(s, &m, &l, &a, &b); a = __shfl_sync(FULL, a, 0); b = __shfl_sync(FULL, b, 0);
    n0 = update_numerator(n0, sv[tb], a, b); n1 = update_numerator(n1, sv[tb + 32], a, b); n2 = update_numerator(n2, sv[tb + 64], a, b); n3 = update_numerator(n3, sv[tb + 96], a, b); }
  FOOTER }
__global__ __launch_bounds__(256) void p_bfly(const __nv_bfloat16* query, const __nv_bfloat16* kp, const __nv_bfloat16* vp, const uint32_t* block_ids, const uint32_t* positions, float* state, int cap, float scale) {
  __shared__ float sk[PAGE * D]; __shared__ float sv[PAGE * D]; HEADER
  for (int e = threadIdx.x; e < count * D; e += 256) { sk[e] = __bfloat162float(kp[base + e]); sv[e] = __bfloat162float(vp[base + e]); }
  __syncthreads();
  const float q0 = __bfloat162float(query[qb + lane]), q1 = __bfloat162float(query[qb + lane + 32]), q2 = __bfloat162float(query[qb + lane + 64]), q3 = __bfloat162float(query[qb + lane + 96]);
  float m = -CUDART_INF_F, l = 0.0F, n0 = 0.0F, n1 = 0.0F, n2 = 0.0F, n3 = 0.0F;
  for (int t = 0; t < count; ++t) { const int tb = t * D + lane;
    float s = fmaf(q0, sk[tb], q1 * sk[tb + 32]); s = fmaf(q2, sk[tb + 64], s); s = fmaf(q3, sk[tb + 96], s);
    s = staged(warp_sum_all(s), scale);
    float a = 0.0F, b = 0.0F; update_online_state(s, &m, &l, &a, &b);
    const float v0 = sv[tb], v1 = sv[tb + 32], v2 = sv[tb + 64], v3 = sv[tb + 96];
    if (b == 0.0F) { n0 = a * n0; n1 = a * n1; n2 = a * n2; n3 = a * n3; }
    else if (a == 0.0F) { n0 = b * v0; n1 = b * v1; n2 = b * v2; n3 = b * v3; }
    else { n0 = fmaf(b, v0, a * n0); n1 = fmaf(b, v1, a * n1); n2 = fmaf(b, v2, a * n2); n3 = fmaf(b, v3, a * n3); } }
  FOOTER }
int main() {
  // {rows, tokens of long rows, long rows, tokens of the rest}: decode-like and prefill-mixed shapes
  const int cfg[][4] = {{1, 300, 1, 300}, {8, 300, 8, 300}, {32, 400, 32, 400}, {32, 150, 32, 150}, {256, 400, 32, 130}, {160, 400, 32, 200}, {96, 330, 32, 330}, {256, 130, 256, 130}};
  for (auto& c : cfg) { const int R = c[0], T = c[1], LONG = c[2], SHORT = c[3], cap = (T + PAGE - 1) / PAGE; const size_t pool = size_t(R) * cap * KVH * PAGE * D;
    __nv_bfloat16 *q, *kp, *vp; uint32_t *ids, *pos; float *s0, *s1;
    cudaMalloc(&q, size_t(R) * QH * D * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&ids, size_t(R) * cap * 4); cudaMalloc(&pos, R * 4);
    const size_t sn = size_t(R) * cap * QH * STRIDE; cudaMalloc(&s0, sn * 4); cudaMalloc(&s1, sn * 4);
    std::vector<__nv_bfloat16> hk(pool), hv(pool), hq(size_t(R) * QH * D);
    for (size_t i = 0; i < pool; ++i) { hk[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F); hv[i] = __float2bfloat16(float((i * 40503u + 17) % 1999) / 999.0F - 1.0F); }
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = __float2bfloat16(float((i * 2246822519u) % 2003) / 500.0F - 2.0F);
    // a few special values to exercise the non-finite branches of the online update
    hk[5] = __float2bfloat16(INFINITY); if (pool > 5000) hk[4100] = __float2bfloat16(-INFINITY);
    cudaMemcpy(kp, hk.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, hv.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice);
    std::vector<uint32_t> hi(size_t(R) * cap), hp(R, T - 1); for (int r = LONG; r < R; ++r) hp[r] = SHORT - 1; const size_t n = hi.size(); size_t mult = 7919; while (n % mult == 0 || mult % 2 == 0) ++mult;
    { std::vector<char> seen(n); bool ok = true; for (size_t i = 0; i < n; ++i) { hi[i] = uint32_t((i * mult) % n); if (seen[hi[i]]) ok = false; seen[hi[i]] = 1; } if (!ok) for (size_t i = 0; i < n; ++i) hi[i] = uint32_t(i); }
    cudaMemcpy(ids, hi.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(pos, hp.data(), R * 4, cudaMemcpyHostToDevice);
    const float scale = 1.0F / std::sqrt(float(D)); const dim3 pg(cap, KVH, R);
    auto timeit = [&](auto f) { for (int i = 0; i < 30; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 400; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 400; };
    auto diff = [&](auto f) { cudaMemset(s0, 0, sn * 4); cudaMemset(s1, 0, sn * 4); p_base<<<pg, 256>>>(q, kp, vp, ids, pos, s0, cap, scale); f(); cudaDeviceSynchronize();
      std::vector<uint32_t> a(sn), b(sn); cudaMemcpy(a.data(), s0, sn * 4, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), s1, sn * 4, cudaMemcpyDeviceToHost); long d = 0; for (size_t i = 0; i < sn; ++i) d += a[i] != b[i]; return d; };
    const double tb = timeit([&] { p_base<<<pg, 256>>>(q, kp, vp, ids, pos, s0, cap, scale); });
    const double tf = timeit([&] { p_ftile<<<pg, 256>>>(q, kp, vp, ids, pos, s1, cap, scale); }); const long df = diff([&] { p_ftile<<<pg, 256>>>(q, kp, vp, ids, pos, s1, cap, scale); });
    const double ty = timeit([&] { p_bfly<<<pg, 256>>>(q, kp, vp, ids, pos, s1, cap, scale); }); const long dy = diff([&] { p_bfly<<<pg, 256>>>(q, kp, vp, ids, pos, s1, cap, scale); });
    std::printf("rows=%3d (%d x %d tokens, %d x %d): base %6.1f us | float tiles %6.1f (state words differing %ld) | + butterfly %6.1f (differing %ld)\n", R, LONG, T, R - LONG, SHORT, tb, tf, df, ty, dy);
    cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(ids); cudaFree(pos); cudaFree(s0); cudaFree(s1); }
  return 0; }
