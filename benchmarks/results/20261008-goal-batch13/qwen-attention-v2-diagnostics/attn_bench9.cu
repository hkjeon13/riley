// Diagnostic microbenchmark 9: does the batched-GEMM prefill attention keep each row's bits when the launch shape changes?
// One fixed 1024-token sequence (values depend only on absolute position). A chunk [P0, P0+R) is computed with exact
// dims, then with padded key count, padded row count, and replicated batch entries; every real row must stay bit-identical.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdio>
#include <vector>
constexpr int D = 128, QH = 16, KVH = 2, G = 8, MAXT = 1024;
static __nv_bfloat16 *g_k, *g_v, *g_q;  // [KVH][MAXT][D], [KVH][MAXT][D], [MAXT][QH][D]
__global__ void fill_q(const __nv_bfloat16* q, __nv_bfloat16* qm, int P0, int R, int RP, int copies) {
  const size_t n = size_t(copies) * KVH * RP * G * D;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = i % D; const size_t c = i / D; const int g = c % G; const size_t r = c / G; const int local = r % RP; const int batch = r / RP; const int kvh = batch % KVH;
    qm[i] = local < R ? q[(size_t(P0 + local) * QH + kvh * G + g) * D + d] : __float2bfloat16_rn(0.0F); } }
__global__ void fill_kv(const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* km, __nv_bfloat16* vm, int keys, int KP, int copies) {
  const size_t n = size_t(copies) * KVH * KP * D;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = i % D; const size_t c = i / D; const int key = c % KP; const int batch = c / KP; const int kvh = batch % KVH;
    const bool real = key < keys; const size_t src = (size_t(kvh) * MAXT + key) * D + d;
    km[i] = real ? k[src] : __float2bfloat16_rn(0.0F); vm[i] = real ? v[src] : __float2bfloat16_rn(0.0F); } }
__global__ void causal_softmax(__nv_bfloat16* scores, int KP, int RP, int R, int P0, float scale) {
  const int column = blockIdx.x, batch = blockIdx.y; const int local = column / G; const int visible = local < R ? P0 + local + 1 : 0;
  __nv_bfloat16* s = scores + (size_t(batch) * RP * G + column) * KP;
  __shared__ float red[256];
  float m = -CUDART_INF_F;
  for (int k = threadIdx.x; k < visible; k += blockDim.x) m = fmaxf(m, __bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)));
  red[threadIdx.x] = m; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
  m = red[0]; __syncthreads(); float sum = 0.0F;
  for (int k = threadIdx.x; k < visible; k += blockDim.x) sum += expf(__bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)) - m);
  red[threadIdx.x] = sum; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float inv = visible > 0 ? 1.0F / red[0] : 0.0F;
  for (int k = threadIdx.x; k < KP; k += blockDim.x)
    s[k] = k < visible ? __float2bfloat16_rn(expf(__bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)) - m) * inv) : __float2bfloat16_rn(0.0F); }
// Returns om of the LAST batch copy pair: [KVH][RP*G][D]
static std::vector<__nv_bfloat16> run(cublasHandle_t h, int P0, int R, int KP, int RP, int copies) {
  const int keys = P0 + R, B = copies * KVH, R8 = RP * G; __nv_bfloat16 *qm, *km, *vm, *sc, *om;
  cudaMalloc(&qm, size_t(B) * R8 * D * 2); cudaMalloc(&om, size_t(B) * R8 * D * 2); cudaMalloc(&km, size_t(B) * KP * D * 2); cudaMalloc(&vm, size_t(B) * KP * D * 2); cudaMalloc(&sc, size_t(B) * R8 * KP * 2);
  const float one = 1.0F, zero = 0.0F, scale = 1.0F / std::sqrt(float(D));
  // The softmax uses a fixed 256-thread block so its own reduction tree does not depend on the launch shape.
  fill_q<<<2048, 256>>>(g_q, qm, P0, R, RP, copies); fill_kv<<<2048, 256>>>(g_k, g_v, km, vm, keys, KP, copies);
  cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, KP, R8, D, &one, km, CUDA_R_16BF, D, (long long)KP * D, qm, CUDA_R_16BF, D, (long long)R8 * D, &zero, sc, CUDA_R_16BF, KP, (long long)R8 * KP, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  causal_softmax<<<dim3(R8, B), 256>>>(sc, KP, RP, R, P0, scale);
  cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, R8, KP, &one, vm, CUDA_R_16BF, D, (long long)KP * D, sc, CUDA_R_16BF, KP, (long long)R8 * KP, &zero, om, CUDA_R_16BF, D, (long long)R8 * D, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  std::vector<__nv_bfloat16> ho(size_t(KVH) * R8 * D); cudaMemcpy(ho.data(), om + size_t(B - KVH) * R8 * D, ho.size() * 2, cudaMemcpyDeviceToHost);
  cudaFree(qm); cudaFree(km); cudaFree(vm); cudaFree(sc); cudaFree(om); return ho; }
// element of row (absolute position), kv head, group head, dim from a run with (P0, RP)
static inline uint16_t at(const std::vector<__nv_bfloat16>& o, int P0, int RP, int pos, int kvh, int g, int d) {
  uint16_t bits; const __nv_bfloat16 v = o[((size_t(kvh) * RP + (pos - P0)) * G + g) * D + d]; memcpy(&bits, &v, 2); return bits; }
int main() {
  cublasHandle_t h; cublasCreate(&h);
  { const size_t kn = size_t(KVH) * MAXT * D, qn = size_t(MAXT) * QH * D; std::vector<__nv_bfloat16> hk(kn), hv(kn), hq(qn);
    for (size_t i = 0; i < kn; ++i) { hk[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F); hv[i] = __float2bfloat16(float((i * 40503u + 17) % 1999) / 999.0F - 1.0F); }
    for (size_t i = 0; i < qn; ++i) hq[i] = __float2bfloat16(float((i * 2246822519u) % 2003) / 500.0F - 2.0F);
    cudaMalloc(&g_k, kn * 2); cudaMalloc(&g_v, kn * 2); cudaMalloc(&g_q, qn * 2);
    cudaMemcpy(g_k, hk.data(), kn * 2, cudaMemcpyHostToDevice); cudaMemcpy(g_v, hv.data(), kn * 2, cudaMemcpyHostToDevice); cudaMemcpy(g_q, hq.data(), qn * 2, cudaMemcpyHostToDevice); }
  // Canonical value of every position: single-row chunks with exact dims.
  std::vector<std::vector<uint16_t>> canon(MAXT);
  const int probe[] = {0, 1, 2, 15, 16, 17, 31, 63, 64, 100, 127, 128, 199, 255, 256, 300, 397, 511, 512, 700, 767, 900, 1023};
  for (int p : probe) { auto o = run(h, p, 1, p + 1, 1, 1); canon[p].resize(QH * D); for (int kvh = 0; kvh < KVH; ++kvh) for (int g = 0; g < G; ++g) for (int d = 0; d < D; ++d) canon[p][(kvh * G + g) * D + d] = at(o, p, 1, p, kvh, g, d); }
  long cases = 0, bad_cases = 0; long elements = 0, differing = 0;
  const int P0s[] = {0, 5, 100, 250, 500, 768}, Rs[] = {1, 3, 16, 60, 128, 256}, copiesList[] = {1, 4, 16};
  for (int P0 : P0s) for (int R : Rs) { if (P0 + R > MAXT) continue; const int keys = P0 + R;
    const int KPs[] = {keys, (keys + 15) / 16 * 16, 256, 512, 1024}; const int RPs[] = {R, 64, 256};
    for (int KP : KPs) { if (KP < keys) continue; for (int RP : RPs) { if (RP < R) continue; for (int copies : copiesList) {
      if (size_t(copies) * KVH * RP * G * KP * 2 > (size_t(3) << 30)) continue;
      auto o = run(h, P0, R, KP, RP, copies); long diff = 0, n = 0;
      for (int p : probe) { if (p < P0 || p >= P0 + R) continue;
        for (int kvh = 0; kvh < KVH; ++kvh) for (int g = 0; g < G; ++g) for (int d = 0; d < D; ++d) { ++n; if (at(o, P0, RP, p, kvh, g, d) != canon[p][(kvh * G + g) * D + d]) ++diff; } }
      if (n == 0) continue; ++cases; elements += n; differing += diff;
      if (diff) { ++bad_cases; if (bad_cases <= 25) std::printf("DIFF P0=%d R=%d keysPad=%d rowsPad=%d batch=%d: %ld of %ld elements\n", P0, R, KP, RP, copies * KVH, diff, n); } } } } }
  std::printf("cases %ld, cases with any difference %ld, elements compared %ld, differing %ld\n", cases, bad_cases, elements, differing);
  cublasDestroy(h); return 0; }
