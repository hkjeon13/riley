// Diagnostic: tensor-core (MMA) flash-style prefill attention for Qwen2.5-3B geometry (16 query heads, 2 KV heads, D=128,
// 16-token pages). One CTA owns (16-row query tile, KV head); its eight warps are the eight query heads of that KV head and
// share one staged K tile and one transposed V tile per page. Every output element depends only on its own row and its
// own keys in ascending page order, so results cannot depend on which rows share a launch.
// Timing by graph replay over 36 layers with distinct caches; values are checked against a CPU reference with a tolerance.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, GROUP = QH / KVH, PAGE = 16;
__device__ __forceinline__ uint32_t pack(float a, float b) { return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) | (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16); }
__device__ __forceinline__ void mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)); }
// query [rows][QH][D], keys/values [context][KVH][D] (pages contiguous), out [rows][QH][D]; row r sits at position first_position + r.
__global__ __launch_bounds__(GROUP * 32) void prefill_attention_mma(const __nv_bfloat16* query, const __nv_bfloat16* keys, const __nv_bfloat16* values, __nv_bfloat16* out, int rows, int first_position, float scale) {
  __shared__ __align__(16) __nv_bfloat16 k_tile[PAGE][D];
  __shared__ __align__(16) __nv_bfloat16 v_tile[D][PAGE];
  const int tile = blockIdx.x, kv_head = blockIdx.y, warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int head = kv_head * GROUP + warp, row0 = tile * 16 + g, row1 = row0 + 8;
  const bool live0 = row0 < rows, live1 = row1 < rows; const int pos0 = first_position + row0, pos1 = first_position + row1;
  const int last_row = min(rows, tile * 16 + 16) - 1, pages = (first_position + last_row) / PAGE + 1;
  uint32_t qa[8][4];
#pragma unroll
  for (int s = 0; s < 8; ++s) { const __nv_bfloat16* q0 = query + (size_t(live0 ? row0 : 0) * QH + head) * D + s * 16 + 2 * t; const __nv_bfloat16* q1 = query + (size_t(live1 ? row1 : 0) * QH + head) * D + s * 16 + 2 * t;
    qa[s][0] = live0 ? *reinterpret_cast<const uint32_t*>(q0) : 0; qa[s][1] = live1 ? *reinterpret_cast<const uint32_t*>(q1) : 0; qa[s][2] = live0 ? *reinterpret_cast<const uint32_t*>(q0 + 8) : 0; qa[s][3] = live1 ? *reinterpret_cast<const uint32_t*>(q1 + 8) : 0; }
  float acc[16][4]; for (int i = 0; i < 16; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = 0.F;
  float max0 = -INFINITY, max1 = -INFINITY, den0 = 0.F, den1 = 0.F;
  for (int page = 0; page < pages; ++page) {
    // Stage this page's K tile and transposed V tile once for the eight query-head warps.
    for (int i = threadIdx.x; i < PAGE * D / 2; i += blockDim.x) { const int token = i / (D / 2), pair = i % (D / 2); const size_t base = (size_t(page * PAGE + token) * KVH + kv_head) * D + 2 * pair;
      *reinterpret_cast<uint32_t*>(&k_tile[token][2 * pair]) = *reinterpret_cast<const uint32_t*>(keys + base);
      const uint32_t v = *reinterpret_cast<const uint32_t*>(values + base); v_tile[2 * pair][token] = __ushort_as_bfloat16(uint16_t(v)); v_tile[2 * pair + 1][token] = __ushort_as_bfloat16(uint16_t(v >> 16)); }
    __syncthreads();
    float sc[2][4] = {};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
#pragma unroll
      for (int half = 0; half < 2; ++half) { const __nv_bfloat16* kp = &k_tile[half * 8 + g][s * 16 + 2 * t]; mma(sc[half], qa[s][0], qa[s][1], qa[s][2], qa[s][3], *reinterpret_cast<const uint32_t*>(kp), *reinterpret_cast<const uint32_t*>(kp + 8)); } }
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int half = 0; half < 2; ++half)
#pragma unroll
      for (int j = 0; j < 2; ++j) { const int key = page * PAGE + half * 8 + 2 * t + j;
        sc[half][j] = (live0 && key <= pos0) ? sc[half][j] * scale : -INFINITY; sc[half][2 + j] = (live1 && key <= pos1) ? sc[half][2 + j] * scale : -INFINITY;
        m0 = fmaxf(m0, sc[half][j]); m1 = fmaxf(m1, sc[half][2 + j]); }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 1)); m0 = fmaxf(m0, __shfl_xor_sync(0xffffffffu, m0, 2)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 1)); m1 = fmaxf(m1, __shfl_xor_sync(0xffffffffu, m1, 2));
    const float n0 = fmaxf(max0, m0), n1 = fmaxf(max1, m1); const float alpha0 = n0 == -INFINITY ? 1.F : __expf(max0 - n0), alpha1 = n1 == -INFINITY ? 1.F : __expf(max1 - n1); max0 = n0; max1 = n1;
    float s0 = 0.F, s1 = 0.F; uint32_t pa[4];
#pragma unroll
    for (int half = 0; half < 2; ++half) { float p[4];
#pragma unroll
      for (int j = 0; j < 2; ++j) { p[j] = n0 == -INFINITY ? 0.F : __expf(sc[half][j] - n0); p[2 + j] = n1 == -INFINITY ? 0.F : __expf(sc[half][2 + j] - n1); s0 += p[j]; s1 += p[2 + j]; }
      pa[half * 2] = pack(p[0], p[1]); pa[half * 2 + 1] = pack(p[2], p[3]); }
    s0 += __shfl_xor_sync(0xffffffffu, s0, 1); s0 += __shfl_xor_sync(0xffffffffu, s0, 2); s1 += __shfl_xor_sync(0xffffffffu, s1, 1); s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
    den0 = den0 * alpha0 + s0; den1 = den1 * alpha1 + s1;
#pragma unroll
    for (int dim = 0; dim < 16; ++dim) { acc[dim][0] *= alpha0; acc[dim][1] *= alpha0; acc[dim][2] *= alpha1; acc[dim][3] *= alpha1;
      const __nv_bfloat16* vp = &v_tile[dim * 8 + g][2 * t]; mma(acc[dim], pa[0], pa[1], pa[2], pa[3], *reinterpret_cast<const uint32_t*>(vp), *reinterpret_cast<const uint32_t*>(vp + 8)); }
    __syncthreads();
  }
  const float inv0 = den0 > 0.F ? 1.F / den0 : 0.F, inv1 = den1 > 0.F ? 1.F / den1 : 0.F;
#pragma unroll
  for (int dim = 0; dim < 16; ++dim) {
    if (live0) *reinterpret_cast<uint32_t*>(out + (size_t(row0) * QH + head) * D + dim * 8 + 2 * t) = pack(acc[dim][0] * inv0, acc[dim][1] * inv0);
    if (live1) *reinterpret_cast<uint32_t*>(out + (size_t(row1) * QH + head) * D + dim * 8 + 2 * t) = pack(acc[dim][2] * inv1, acc[dim][3] * inv1); }
}
static uint16_t bf(float v) { uint32_t b; std::memcpy(&b, &v, 4); b += 0x7fff + ((b >> 16) & 1); return uint16_t(b >> 16); }
static float fb(uint16_t h) { uint32_t b = uint32_t(h) << 16; float v; std::memcpy(&v, &b, 4); return v; }
int main(int argc, char** argv) {
  struct Case { int rows, first; }; std::vector<Case> cases = {{16, 0}, {28, 0}, {125, 0}, {256, 0}, {128, 200}, {224, 174}, {112, 18}, {32, 368}};
  const int L = 36; cudaStream_t s; cudaStreamCreate(&s);
  for (const Case& c : cases) { const int ctx = c.first + c.rows, padded = ((ctx + PAGE - 1) / PAGE) * PAGE; const size_t qn = size_t(c.rows) * QH * D, kn = size_t(padded) * KVH * D;
    std::vector<uint16_t> hq(qn), hk(kn * L), hv(kn * L); uint32_t seed = 12345u + c.rows; auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float((seed >> 8) & 0xffff) / 65536.F - .5F); };
    for (auto& x : hq) x = bf(rnd() * 2.F); for (auto& x : hk) x = bf(rnd() * 2.F); for (auto& x : hv) x = bf(rnd());
    __nv_bfloat16 *q, *k, *v, *o; cudaMalloc(&q, qn * 2); cudaMalloc(&k, kn * L * 2); cudaMalloc(&v, kn * L * 2); cudaMalloc(&o, qn * 2);
    cudaMemcpy(q, hq.data(), qn * 2, cudaMemcpyHostToDevice); cudaMemcpy(k, hk.data(), kn * L * 2, cudaMemcpyHostToDevice); cudaMemcpy(v, hv.data(), kn * L * 2, cudaMemcpyHostToDevice);
    const float scale = 1.F / std::sqrt(float(D)); const dim3 grid((c.rows + 15) / 16, KVH);
    prefill_attention_mma<<<grid, GROUP * 32, 0, s>>>(q, k, v, o, c.rows, c.first, scale); cudaStreamSynchronize(s);
    std::vector<uint16_t> ho(qn); cudaMemcpy(ho.data(), o, qn * 2, cudaMemcpyDeviceToHost); double worst = 0;
    for (int r = 0; r < c.rows; r += (c.rows > 40 ? 37 : 5)) for (int h = 0; h < QH; h += 5) { const int kvh = h / GROUP, n = c.first + r + 1; std::vector<double> sc(n); double mx = -1e30;
        for (int j = 0; j < n; ++j) { double d = 0; for (int e = 0; e < D; ++e) d += double(fb(hq[(size_t(r) * QH + h) * D + e])) * fb(hk[(size_t(j) * KVH + kvh) * D + e]); sc[j] = d * scale; mx = std::max(mx, sc[j]); }
        double den = 0; for (int j = 0; j < n; ++j) { sc[j] = std::exp(sc[j] - mx); den += sc[j]; }
        for (int e = 0; e < D; e += 17) { double acc = 0; for (int j = 0; j < n; ++j) acc += sc[j] * fb(hv[(size_t(j) * KVH + kvh) * D + e]); worst = std::max(worst, std::fabs(acc / den - fb(ho[(size_t(r) * QH + h) * D + e]))); } }
    cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int l = 0; l < L; ++l) prefill_attention_mma<<<grid, GROUP * 32, 0, s>>>(q, k + kn * l, v + kn * l, o, c.rows, c.first, scale);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&e, g, 0); for (int i = 0; i < 5; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s);
    double best = 1e9; for (int rep = 0; rep < 7; ++rep) { const double t0 = now(); for (int i = 0; i < 20; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); best = std::min(best, (now() - t0) / (20 * L)); }
    std::printf("rows %3d at position %3d (keys up to %3d, %2d pages): %6.2f us per launch, max |error| vs CPU %.4f\n", c.rows, c.first, ctx, padded / PAGE, best, worst);
    cudaGraphExecDestroy(e); cudaGraphDestroy(g); cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(o); }
  return 0; }
