// Diagnostic: time the first-heuristic algorithm chosen at an anchor row count when reused at other row counts (cold weights).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  cublasLtHandle_t lt; cublasLtCreate(&lt); size_t ws = 16u << 20; void* work; cudaMalloc(&work, ws);
  const int KN[][2] = {{2048, 2560}, {2048, 2048}, {2048, 256}, {2048, 22016}, {2048, 11008}}; const int Ms[] = {1, 8, 32, 256}; const int As[] = {1, 8, 32, 64, 128, 256}; const int R = 40;
  for (auto& kn : KN) { const int K = kn[0], N = kn[1];
    __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, size_t(K) * N * 2 * R); cudaMalloc(&X, size_t(K) * 256 * 2); cudaMalloc(&Y, size_t(N) * 256 * 2);
    cudaMemset(Wall, 0x11, size_t(K) * N * 2 * R); cudaMemset(X, 0x11, size_t(K) * 256 * 2); int rot = 0;
    cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatrixLayout_t la; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K);
    std::printf("K=%d N=%d  (rows: time us at M=1, 8, 32, 256; -1 = algorithm rejected for that M)\n", K, N);
    for (int A : As) {
      cublasLtMatrixLayout_t lb, lc; cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, A, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, A, N);
      cublasLtMatmulHeuristicResult_t r[8]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, p, 8, r, &n);
      for (int cand = 0; cand < n && cand < 1; ++cand) {
        uint32_t id = 0, tile = 0, st = 0, sk = 0, rs = 0; size_t w;
        cublasLtMatmulAlgoConfigGetAttribute(&r[cand].algo, CUBLASLT_ALGO_CONFIG_ID, &id, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[cand].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[cand].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &st, 4, &w);
        cublasLtMatmulAlgoConfigGetAttribute(&r[cand].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4, &w); cublasLtMatmulAlgoConfigGetAttribute(&r[cand].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, 4, &w);
        std::printf("  anchor M=%3d cand %d id=%2u tile=%2u stages=%2u splitk=%2u red=%u :", A, cand, id, tile, st, sk, rs);
        for (int M : Ms) {
          cublasLtMatrixLayout_t mb, mc; cublasLtMatrixLayoutCreate(&mb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&mc, CUDA_R_16BF, N, M, N);
          cublasLtMatmulHeuristicResult_t chk{}; double us = -1;
          if (cublasLtMatmulAlgoCheck(lt, d, la, mb, mc, mc, &r[cand].algo, &chk) == CUBLAS_STATUS_SUCCESS && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= ws) {
            const float alpha = 1, beta = 0; auto f = [&] { cublasLtMatmul(lt, d, &alpha, Wall + size_t(K) * N * (rot++ % R), la, X, mb, &beta, Y, mc, Y, mc, &r[cand].algo, work, ws, 0); };
            for (int i = 0; i < 20; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 150; ++i) f(); cudaDeviceSynchronize(); us = (now() - t0) / 150; }
          std::printf(" %7.1f", us); }
        std::printf("\n"); } }
    cudaFree(Wall); cudaFree(X); cudaFree(Y); }
  return 0;
}
