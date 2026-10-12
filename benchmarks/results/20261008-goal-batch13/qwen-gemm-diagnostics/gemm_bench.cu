// Diagnostic microbenchmark: cuBLAS/cuBLASLt BF16 GEMM forms for Qwen2.5-3B projection shapes.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  cublasHandle_t h; cublasCreate(&h); cublasLtHandle_t lt; cublasLtCreate(&lt);
  const int Ms[] = {1, 8, 32}; const int KN[][2] = {{2048, 11008}, {11008, 2048}, {2048, 2048}};
  size_t ws = 64u << 20; void* work; cudaMalloc(&work, ws);
  for (auto& kn : KN) for (int M : Ms) {
    const int K = kn[0], N = kn[1];
    __nv_bfloat16 *W, *X, *Y; cudaMalloc(&W, size_t(K) * N * 2); cudaMalloc(&X, size_t(K) * M * 2); cudaMalloc(&Y, size_t(N) * M * 2);
    std::vector<__nv_bfloat16> hw(size_t(K) * N), hx(size_t(K) * M);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.f - 1.f);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2bfloat16(float((i * 40503u) % 2001) / 1000.f - 1.f);
    cudaMemcpy(W, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0; const int iters = 200;
    auto timeit = [&](auto&& f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < iters; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / iters; };
    // (a) the HF/Riley-probe form: Yc[N,M] = Wc[K,N]^T * Xc[K,M]
    double a = timeit([&] { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    // (b) transposed problem: Yr[M,N] = Xc^T[M,K] * Wc[K,N]
    double b = timeit([&] { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, X, CUDA_R_16BF, K, W, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    // (c) cuBLASLt heuristic candidates for form (a)
    cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatrixLayout_t la, lb, lc; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
    cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t r[16]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, p, 16, r, &n);
    double best = 1e30, first = -1; int besti = -1;
    for (int i = 0; i < n; ++i) { if (r[i].state != CUBLAS_STATUS_SUCCESS) continue;
      double c = timeit([&] { cublasLtMatmul(lt, d, &alpha, W, la, X, lb, &beta, Y, lc, Y, lc, &r[i].algo, work, ws, 0); });
      if (i == 0) first = c; if (c < best) { best = c; besti = i; } }
    double bytes = double(K) * N * 2; std::printf("K=%5d N=%5d M=%2d  gemmEx(T,N) %7.1f us  transposed %7.1f us  Lt first %7.1f us  Lt best[%d/%d] %7.1f us  weight-read bound@900GB/s %5.1f us\n", K, N, M, a, b, first, besti, n, best, bytes / 900e9 * 1e6);
    cudaFree(W); cudaFree(X); cudaFree(Y);
  }
  return 0;
}
