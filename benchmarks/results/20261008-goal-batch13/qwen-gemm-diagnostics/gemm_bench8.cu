// Diagnostic: does a different cuBLAS entry point or compute type explain vLLM's faster projection kernels at M=32?
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  cublasLtHandle_t lt; cublasLtCreate(&lt); cublasHandle_t cb; cublasCreate(&cb); size_t ws = 32u << 20; void* work; cudaMalloc(&work, ws);
  const int KN[][2] = {{2048, 2560}, {2048, 2048}, {2048, 22016}, {11008, 2048}};
  for (auto& kn : KN) { const int K = kn[0], N = kn[1]; const size_t bytes = size_t(K) * N * 2; const int R = int((700u << 20) / bytes) + 2;
    __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, bytes * R); cudaMalloc(&X, size_t(K) * 256 * 2); cudaMalloc(&Y, size_t(N) * 256 * 2); cudaMemset(Wall, 0x11, bytes * R); cudaMemset(X, 0x11, size_t(K) * 256 * 2); int rot = 0;
    auto timeit = [&](auto f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 200; };
    auto Wn = [&] { return Wall + size_t(K) * N * (rot++ % R); };
    std::printf("K=%d N=%d\n", K, N);
    for (int M : {1, 8, 32}) { const float alpha = 1, beta = 0;
      std::printf("  M=%2d gemmEx:", M);
      const cublasComputeType_t cts[] = {CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_COMPUTE_32F_PEDANTIC}; const char* cn[] = {"32F", "FAST_16BF", "FAST_TF32", "PEDANTIC"};
      for (int c = 0; c < 4; ++c) for (cublasGemmAlgo_t ga : {CUBLAS_GEMM_DEFAULT, CUBLAS_GEMM_DEFAULT_TENSOR_OP}) {
        const cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, cts[c], ga);
        if (st != CUBLAS_STATUS_SUCCESS) { std::printf(" %s/%d n/a", cn[c], int(ga)); continue; }
        std::printf(" %s/%s %.1f", cn[c], ga == CUBLAS_GEMM_DEFAULT ? "def" : "tensor", timeit([&] { cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, cts[c], ga); })); }
      // row-major style call as a framework would issue it: C[M,N]^T ... equivalently swap operands: C' (M x N, ld M) = X^T (M x K) * W (K x N)
      { const cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, X, CUDA_R_16BF, K, Wn(), CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) std::printf(" | swapped-operands(out MxN) %.1f", timeit([&] { cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, X, CUDA_R_16BF, K, Wn(), CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); })); }
      // weights stored row-major-by-output as [N][K] is what we have; alternative storage [K][N] (ld N): C (N x M) = W2 (N x K) * X (K x M)
      { const cublasStatus_t st = cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, N, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st == CUBLAS_STATUS_SUCCESS) std::printf(" | weight-as-NxK-colmajor %.1f", timeit([&] { cublasGemmEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, N, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); })); }
      std::printf("\n"); }
    cudaFree(Wall); cudaFree(X); cudaFree(Y); }
  return 0; }
