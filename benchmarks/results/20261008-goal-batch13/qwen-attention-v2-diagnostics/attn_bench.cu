// Diagnostic microbenchmark for a two-pass softmax decode attention (Qwen D128, QH=16, KVH=2, 16-token pages).
// Pass 1: staged BF16 scores into an F32 buffer + per-(row,head) max and sum. Pass 2: BF16-rounded probabilities times V.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, G = 8, PAGE = 16, TILE = 32;
__device__ __forceinline__ float staged(float dot, float scale) { return __bfloat162float(__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(dot)) * scale)); }
__device__ __forceinline__ float nan_max(float a, float b) { return (isnan(a) || isnan(b)) ? CUDART_NAN_F : fmaxf(a, b); }
// pool layout: [page][kvh][token16][D]; block_ids[row * pages_per_row + logical_page]
__global__ __launch_bounds__(256) void score_kernel(const __nv_bfloat16* query, const __nv_bfloat16* key_pool, const uint32_t* block_ids, const uint32_t* positions,
    float* scores, float* stats, int pages_per_row, int tcap, float scale) {
  __shared__ float sq[G * D]; __shared__ float sk[TILE * D]; __shared__ float red[256]; __shared__ uint64_t page_base[2];
  const int kvh = blockIdx.x, row = blockIdx.y, slot = threadIdx.x & 31, hg = threadIdx.x >> 5, head = kvh * G + hg;
  const int T = positions[row] + 1;
  for (int i = threadIdx.x; i < G * D; i += 256) sq[i] = __bfloat162float(query[(size_t(row) * QH + kvh * G) * D + i]);
  float run_max = -CUDART_INF_F; float* srow = scores + (size_t(row) * QH + head) * tcap;
  for (int base = 0; base < T; base += TILE) {
    __syncthreads();
    const int count = min(TILE, T - base);
    if (threadIdx.x < 2) page_base[threadIdx.x] = (size_t(block_ids[row * pages_per_row + base / PAGE + threadIdx.x]) * KVH + kvh) * PAGE * D;
    __syncthreads();
    for (int e = threadIdx.x; e < count * D; e += 256) { const int te = e >> 7; sk[e] = __bfloat162float(key_pool[page_base[te >> 4] + (te & 15) * D + (e & 127)]); }
    __syncthreads();
    if (slot < count) { float acc = 0.0F; const float* q = sq + hg * D; const float* k = sk + slot * D;
      #pragma unroll 8
      for (int d = 0; d < D; ++d) acc = fmaf(q[d], k[d], acc);
      const float s = staged(acc, scale); srow[base + slot] = s; run_max = nan_max(run_max, s); } }
  __syncthreads(); red[threadIdx.x] = run_max; __syncthreads();
  for (int off = 16; off; off >>= 1) { if (slot < off) red[threadIdx.x] = nan_max(red[threadIdx.x], red[threadIdx.x + off]); __syncthreads(); }
  const float m = red[hg * 32]; float partial = 0.0F;
  for (int t = slot; t < T; t += TILE) partial += expf(srow[t] - m);
  __syncthreads(); red[threadIdx.x] = partial; __syncthreads();
  for (int off = 16; off; off >>= 1) { if (slot < off) red[threadIdx.x] += red[threadIdx.x + off]; __syncthreads(); }
  if (slot == 0) { stats[(size_t(row) * QH + head) * 2] = m; stats[(size_t(row) * QH + head) * 2 + 1] = red[threadIdx.x]; } }
__global__ __launch_bounds__(1024) void value_kernel(const __nv_bfloat16* value_pool, const uint32_t* block_ids, const uint32_t* positions, const float* scores, const float* stats,
    __nv_bfloat16* output, int pages_per_row, int tcap) {
  __shared__ float sv[TILE * D]; __shared__ float sp[G * TILE]; __shared__ uint64_t page_base[2];
  const int kvh = blockIdx.x, row = blockIdx.y, dim = threadIdx.x & 127, hg = threadIdx.x >> 7, head = kvh * G + hg;
  const int T = positions[row] + 1; const float m = stats[(size_t(row) * QH + head) * 2]; const float inv = 1.0F / stats[(size_t(row) * QH + head) * 2 + 1];
  const float* srow = scores + (size_t(row) * QH + head) * tcap; float acc = 0.0F;
  for (int base = 0; base < T; base += TILE) {
    __syncthreads();
    const int count = min(TILE, T - base);
    if (threadIdx.x < 2) page_base[threadIdx.x] = (size_t(block_ids[row * pages_per_row + base / PAGE + threadIdx.x]) * KVH + kvh) * PAGE * D;
    __syncthreads();
    for (int e = threadIdx.x; e < count * D; e += 1024) { const int te = e >> 7; sv[e] = __bfloat162float(value_pool[page_base[te >> 4] + (te & 15) * D + (e & 127)]); }
    if (dim < count) sp[hg * TILE + dim] = __bfloat162float(__float2bfloat16_rn(expf(srow[base + dim] - m) * inv));
    __syncthreads();
    const float* p = sp + hg * TILE;
    #pragma unroll 8
    for (int t = 0; t < count; ++t) acc = fmaf(p[t], sv[t * D + dim], acc); }
  output[(size_t(row) * QH + head) * D + dim] = __float2bfloat16_rn(acc); }
int main() {
  const int cfg[][2] = {{1, 1000}, {1, 150}, {8, 1000}, {8, 150}, {32, 1000}, {32, 500}, {32, 150}, {256, 300}, {256, 130}};
  for (auto& c : cfg) { const int R = c[0], T = c[1], pages = (T + PAGE - 1) / PAGE + 1, tcap = pages * PAGE; const size_t pool = size_t(R) * pages * KVH * PAGE * D;
    __nv_bfloat16 *q, *kp, *vp, *out; uint32_t *ids, *pos; float *scores, *stats;
    cudaMalloc(&q, size_t(R) * QH * D * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&out, size_t(R) * QH * D * 2); cudaMalloc(&ids, size_t(R) * pages * 4); cudaMalloc(&pos, R * 4);
    cudaMalloc(&scores, size_t(R) * QH * tcap * 4); cudaMalloc(&stats, size_t(R) * QH * 8);
    std::vector<__nv_bfloat16> h(pool); for (size_t i = 0; i < pool; ++i) h[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F);
    cudaMemcpy(kp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, h.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, h.data(), size_t(R) * QH * D * 2, cudaMemcpyHostToDevice);
    std::vector<uint32_t> hi(size_t(R) * pages), hp(R, T - 1); for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint32_t((i * 7919u) % hi.size());  // scattered, unique pages
    { std::vector<char> seen(hi.size()); bool ok = true; for (auto v : hi) { if (seen[v]) ok = false; seen[v] = 1; } if (!ok) for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint32_t(i); }
    cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(pos, hp.data(), R * 4, cudaMemcpyHostToDevice);
    const float scale = 1.0F / std::sqrt(float(D)); const dim3 grid(KVH, R);
    auto k1 = [&] { score_kernel<<<grid, 256>>>(q, kp, ids, pos, scores, stats, pages, tcap, scale); };
    auto k2 = [&] { value_kernel<<<grid, 1024>>>(vp, ids, pos, scores, stats, out, pages, tcap); };
    auto timeit = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 200; };
    const double a = timeit(k1), b = timeit(k2), both = timeit([&] { k1(); k2(); });
    std::vector<__nv_bfloat16> ho(size_t(R) * QH * D); cudaMemcpy(ho.data(), out, ho.size() * 2, cudaMemcpyDeviceToHost); double sum = 0; int nan = 0; for (auto v : ho) { float f = __bfloat162float(v); if (std::isnan(f)) ++nan; else sum += f; }
    std::printf("rows=%3d T=%4d: scores %6.1f us, values %6.1f us, both %6.1f us | checksum %.4f nan %d (err %d)\n", R, T, a, b, both, sum, nan, int(cudaGetLastError()));
    cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(out); cudaFree(ids); cudaFree(pos); cudaFree(scores); cudaFree(stats); }
  return 0; }
