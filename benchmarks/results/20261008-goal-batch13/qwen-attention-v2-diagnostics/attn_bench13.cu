// Diagnostic microbenchmark 13: cost of launching the butterfly producer over a page extent larger than the rows need
// (a recorded step launches over the workspace capacity; CTAs past a row's page count exit immediately).
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
  // {rows, tokens per row, launch page extents...}
  const int cfg[][2] = {{1, 300}, {8, 300}, {16, 300}, {32, 300}, {32, 150}, {32, 100}, {256, 130}, {160, 200}};
  const int extents[] = {0, 16, 32, 64};
  for (auto& c : cfg) { const int R = c[0], T = c[1], need = (T + PAGE - 1) / PAGE, cap = 64; const size_t pool = size_t(R) * cap * KVH * PAGE * D;
    __nv_bfloat16 *q, *kp, *vp; uint32_t *ids, *pos; float* s0;
    cudaMalloc(&q, size_t(R) * QH * D * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&ids, size_t(R) * cap * 4); cudaMalloc(&pos, R * 4);
    const size_t sn = size_t(R) * cap * QH * STRIDE; cudaMalloc(&s0, sn * 4);
    cudaMemset(kp, 0x3c, pool * 2); cudaMemset(vp, 0x3c, pool * 2); cudaMemset(q, 0x3c, size_t(R) * QH * D * 2);
    std::vector<uint32_t> hi(size_t(R) * cap), hp(R, T - 1); for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint32_t(i);
    cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(pos, hp.data(), R * 4, cudaMemcpyHostToDevice);
    const float scale = 1.0F / std::sqrt(float(D));
    auto timeit = [&](int extent) { auto f = [&] { p_bfly<<<dim3(extent, KVH, R), 256>>>(q, kp, vp, ids, pos, s0, cap, scale); };
      for (int i = 0; i < 30; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 400; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 400; };
    std::printf("rows=%3d tokens=%3d (needs %2d pages):", R, T, need);
    for (int e : extents) { const int extent = e == 0 ? need : e; if (extent < need) continue; std::printf(" extent %2d -> %6.1f us |", extent, timeit(extent)); }
    std::printf("\n");
    cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(ids); cudaFree(pos); cudaFree(s0); }
  return 0; }
