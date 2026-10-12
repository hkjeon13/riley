// Diagnostic: can a hand-written small-M GEMV-style kernel beat cuBLAS on cold BF16 weights? Compares a pure read
// (bandwidth floor), cublasGemmEx, the cuBLASLt first heuristic, and a custom warp-per-output kernel.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// W: N columns of K contiguous BF16 (ld K). X: M columns of K. Y: M columns of N.
template <int M> __global__ void gemv_kernel(const __nv_bfloat16* W, const __nv_bfloat16* X, __nv_bfloat16* Y, int K, int N) {
  extern __shared__ __nv_bfloat16 sx[];
  for (int i = threadIdx.x; i < K * M; i += blockDim.x) sx[i] = X[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5, total = gridDim.x * warps;
  for (int n = blockIdx.x * warps + warp; n < N; n += total) {
    const __nv_bfloat16* w = W + size_t(n) * K; float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0F;
    for (int k = lane * 8; k < K; k += 256) {
      const uint4 raw = *reinterpret_cast<const uint4*>(w + k); const __nv_bfloat16* v = reinterpret_cast<const __nv_bfloat16*>(&raw);
      #pragma unroll
      for (int i = 0; i < 8; ++i) { const float wf = __bfloat162float(v[i]);
        #pragma unroll
        for (int m = 0; m < M; ++m) acc[m] = fmaf(wf, __bfloat162float(sx[m * K + k + i]), acc[m]); } }
    #pragma unroll
    for (int m = 0; m < M; ++m) { float a = acc[m];
      #pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_down_sync(0xffffffffu, a, o);
      if (lane == 0) Y[size_t(m) * N + n] = __float2bfloat16_rn(a); } } }
__global__ void read_kernel(const __nv_bfloat16* W, float* out, size_t count) {
  float acc = 0; const size_t total = size_t(gridDim.x) * blockDim.x, id = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (size_t i = id * 8; i < count; i += total * 8) { const uint4 raw = *reinterpret_cast<const uint4*>(W + i); acc += float(raw.x ^ raw.y ^ raw.z ^ raw.w); }
  if (acc == 12345.678F) out[0] = acc; }
int main() {
  cublasLtHandle_t lt; cublasLtCreate(&lt); cublasHandle_t cb; cublasCreate(&cb); size_t ws = 16u << 20; void* work; cudaMalloc(&work, ws); float* sink; cudaMalloc(&sink, 64);
  const int KN[][2] = {{2048, 11008}, {11008, 2048}, {2048, 2048}, {2048, 151936}};
  for (auto& kn : KN) { const int K = kn[0], N = kn[1]; const size_t bytes = size_t(K) * N * 2; const int R = int((700u << 20) / bytes) + 2;
    __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, bytes * R); cudaMalloc(&X, size_t(K) * 32 * 2); cudaMalloc(&Y, size_t(N) * 32 * 2); cudaMemset(Wall, 0x11, bytes * R); cudaMemset(X, 0x11, size_t(K) * 32 * 2); int rot = 0;
    auto timeit = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 150; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 150; };
    auto Wn = [&] { return Wall + size_t(K) * N * (rot++ % R); };
    std::printf("K=%d N=%d weight MB %.1f\n", K, N, bytes / 1e6);
    for (int blocks : {288, 576, 1152, 2304}) std::printf("  read-only blocks=%4d x256: %7.1f us\n", blocks, timeit([&] { read_kernel<<<blocks, 256>>>(Wn(), sink, size_t(K) * N); }));
    for (int M : {1, 8, 32}) {
      const float alpha = 1, beta = 0;
      double ex = timeit([&] { cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); });
      cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
      cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
      cublasLtMatrixLayout_t la, lb, lc; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
      cublasLtMatmulHeuristicResult_t r[1]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, p, 1, r, &n);
      double ltus = n ? timeit([&] { cublasLtMatmul(lt, d, &alpha, Wn(), la, X, lb, &beta, Y, lc, Y, lc, &r[0].algo, work, ws, 0); }) : -1;
      std::printf("  M=%2d cublasGemmEx %7.1f us | cuBLASLt heuristic@M %7.1f us | custom:", M, ex, ltus);
      if (M <= 8 && K * M * 2 <= 44000) for (int blocks : {288, 576, 1152}) for (int threads : {128, 256}) {
        double us = M == 1 ? timeit([&] { gemv_kernel<1><<<blocks, threads, K * 2>>>(Wn(), X, Y, K, N); }) : timeit([&] { gemv_kernel<8><<<blocks, threads, K * 16>>>(Wn(), X, Y, K, N); });
        std::printf(" [%dx%d] %.1f", blocks, threads, us); }
      std::printf("\n"); }
    cudaFree(Wall); cudaFree(X); cudaFree(Y); }
  return 0; }
