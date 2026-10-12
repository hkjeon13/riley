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
  const int Ms[] = {1, 32}; const int R = 14; const int KN[][2] = {{2048, 11008}, {11008, 2048}, {2048, 2048}};
  size_t ws = 64u << 20; void* work; cudaMalloc(&work, ws);
  for (auto& kn : KN) for (int M : Ms) {
    const int K = kn[0], N = kn[1];
    __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, size_t(K) * N * 2 * R); int rot = 0; auto Wn = [&]() { return Wall + size_t(K) * N * (rot++ % R); }; cudaMalloc(&X, size_t(K) * M * 2); cudaMalloc(&Y, size_t(N) * M * 2);
    std::vector<__nv_bfloat16> hw(size_t(K) * N), hx(size_t(K) * M);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.f - 1.f);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2bfloat16(float((i * 40503u) % 2001) / 1000.f - 1.f);
    for (int c = 0; c < R; ++c) cudaMemcpy(Wall + size_t(K) * N * c, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0; const int iters = 200;
    auto timeit = [&](auto&& f) { for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < iters; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / iters; };
    // (a) the HF/Riley-probe form: Yc[N,M] = Wc[K,N]^T * Xc[K,M]
    double a = timeit([&] { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, Wn(), CUDA_R_16BF, K, X, CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    // (b) transposed problem: Yr[M,N] = Xc^T[M,K] * Wc[K,N]
    double b = timeit([&] { cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, X, CUDA_R_16BF, K, Wn(), CUDA_R_16BF, K, &beta, Y, CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP); });
    // (c) cuBLASLt heuristic candidates for form (a)
    cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatrixLayout_t la, lb, lc; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
    cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t r[16]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, p, 16, r, &n);
    double best = 1e30, first = -1; int besti = -1; std::printf("   Lt algos:");
    for (int i = 0; i < n; ++i) { if (r[i].state != CUBLAS_STATUS_SUCCESS) continue;
      uint32_t sk = 0, rs = 0, id = 0, tile = 0, st = 0; size_t w;
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, 4, &w);
      cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_ID, &id, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, 4, &w);
      double c = timeit([&] { cublasLtMatmul(lt, d, &alpha, Wn(), la, X, lb, &beta, Y, lc, Y, lc, &r[i].algo, work, ws, 0); });
      cublasLtMatmulAlgo_t forced = r[i].algo; uint32_t one = 1, none = CUBLASLT_REDUCTION_SCHEME_NONE; double f = -1; size_t fw = 0;
      cublasLtMatmulAlgoConfigSetAttribute(&forced, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, 4); cublasLtMatmulAlgoConfigSetAttribute(&forced, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, 4);
      cublasLtMatmulHeuristicResult_t chk{}; if (cublasLtMatmulAlgoCheck(lt, d, la, lb, lc, lc, &forced, &chk) == CUBLAS_STATUS_SUCCESS && chk.state == CUBLAS_STATUS_SUCCESS) { fw = chk.workspaceSize; f = timeit([&] { cublasLtMatmul(lt, d, &alpha, Wn(), la, X, lb, &beta, Y, lc, Y, lc, &forced, work, ws, 0); }); }
      std::printf("\n     [%d] id=%u tile=%u stages=%u splitk=%u red=%u ws=%zu : %.1f us | forced splitk1/none ws=%zu : %.1f us", i, id, tile, st, sk, rs, r[i].workspaceSize, c, fw, f);
      if (i == 0) first = c; if (c < best) { best = c; besti = i; } } std::printf("\n");
    double bytes = double(K) * N * 2; std::printf("K=%5d N=%5d M=%2d  gemmEx(T,N) %7.1f us  transposed %7.1f us  Lt first %7.1f us  Lt best[%d/%d] %7.1f us  weights MB %5.1f; @900GB/s %5.1f us\n", K, N, M, a, b, first, besti, n, best, bytes / 1e6, bytes / 900e9 * 1e6);
    cudaFree(Wall); cudaFree(X); cudaFree(Y);
  }
  return 0;
}
