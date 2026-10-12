// Diagnostic: exhaustive cuBLASLt algorithm search for one BF16 projection shape. For every algorithm id and every
// supported (tile, stages, split-K) configuration that is valid at M = 1, 32 and 256, time cold-weight GEMMs and report the
// configurations with the lowest weighted cost (decode-heavy mix), next to the first-heuristic-at-128 baseline.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Cfg { int id; uint32_t tile, stages, splitk, red; double t[3]; double cost; };
int main(int argc, char** argv) {
  const int K = std::atoi(argv[1]), N = std::atoi(argv[2]); const int Ms[3] = {1, 32, 256}; const double W[3] = {0.0, 0.67, 0.33};
  cublasLtHandle_t lt; cublasLtCreate(&lt); size_t ws = 32u << 20; void* work; cudaMalloc(&work, ws);
  const size_t bytes = size_t(K) * N * 2; const int R = int((700u << 20) / bytes) + 2;
  __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, bytes * R); cudaMalloc(&X, size_t(K) * 256 * 2); cudaMalloc(&Y, size_t(N) * 256 * 2); cudaMemset(Wall, 0x11, bytes * R); cudaMemset(X, 0x11, size_t(K) * 256 * 2); int rot = 0;
  cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t tr = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof(tr));
  cublasLtMatrixLayout_t la, lb[3], lc[3]; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K);
  for (int i = 0; i < 3; ++i) { cublasLtMatrixLayoutCreate(&lb[i], CUDA_R_16BF, K, Ms[i], K); cublasLtMatrixLayoutCreate(&lc[i], CUDA_R_16BF, N, Ms[i], N); }
  const float alpha = 1, beta = 0;
  auto timeit = [&](const cublasLtMatmulAlgo_t& algo, int i, int warm, int reps) { auto f = [&] { cublasLtMatmul(lt, d, &alpha, Wall + size_t(K) * N * (rot++ % R), la, X, lb[i], &beta, Y, lc[i], Y, lc[i], &algo, work, ws, 0); };
    for (int j = 0; j < warm; ++j) f(); cudaDeviceSynchronize(); double t0 = now(); for (int j = 0; j < reps; ++j) f(); cudaDeviceSynchronize(); return (now() - t0) / reps; };
  // baseline: first heuristic at 128 rows reused
  { cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatrixLayout_t b128, c128; cublasLtMatrixLayoutCreate(&b128, CUDA_R_16BF, K, 128, K); cublasLtMatrixLayoutCreate(&c128, CUDA_R_16BF, N, 128, N);
    cublasLtMatmulHeuristicResult_t r[1]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, b128, c128, c128, p, 1, r, &n);
    std::printf("K=%d N=%d baseline first-heuristic@128:", K, N); double c = 0; for (int i = 0; i < 3; ++i) { double t = timeit(r[0].algo, i, 20, 150); c += W[i] * t; std::printf(" M=%d %.1f", Ms[i], t); } std::printf(" | weighted %.1f us\n", c); }
  int ids[256]; int nid = 0; cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 256, ids, &nid);
  std::vector<Cfg> ok; long tried = 0;
  for (int a = 0; a < nid; ++a) { cublasLtMatmulAlgo_t algo;
    if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[a], &algo) != CUBLAS_STATUS_SUCCESS) continue;
    size_t sz = 0; uint32_t tiles[128] = {0}, stages[128] = {0}; int nt = 1, ns = 1; int32_t splitk_ok = 0; uint32_t redmask = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { nt = int(sz / 4); if (nt > 128) nt = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, nt * 4, &sz); }
    sz = 0; if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { ns = int(sz / 4); if (ns > 128) ns = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, ns * 4, &sz); }
    cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, 4, &sz); cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redmask, 4, &sz);
    const uint32_t splits[] = {1, 2, 3, 4, 5, 8}; const uint32_t reds[] = {CUBLASLT_REDUCTION_SCHEME_NONE, CUBLASLT_REDUCTION_SCHEME_INPLACE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};
    for (int ti = 0; ti < nt; ++ti) for (int si = 0; si < ns; ++si) for (uint32_t sk : splits) { if (sk > 1 && !splitk_ok) break;
      for (uint32_t rs : reds) { if ((sk == 1) != (rs == CUBLASLT_REDUCTION_SCHEME_NONE)) continue; if (rs != CUBLASLT_REDUCTION_SCHEME_NONE && !(redmask & rs)) continue;
        cublasLtMatmulAlgo_t c = algo; cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[ti], 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages[si], 4);
        cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, 4);
        bool valid = true; for (int i = 0; i < 3 && valid; ++i) { cublasLtMatmulHeuristicResult_t chk{}; valid = cublasLtMatmulAlgoCheck(lt, d, la, lb[i], lc[i], lc[i], &c, &chk) == CUBLAS_STATUS_SUCCESS && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= ws; }
        if (!valid) continue; ++tried;
        Cfg g{ids[a], tiles[ti], stages[si], sk, rs, {0, 0, 0}, 0}; g.t[1] = timeit(c, 1, 3, 12); if (g.t[1] > 400) continue; g.t[2] = timeit(c, 2, 3, 12); g.cost = W[1] * g.t[1] + W[2] * g.t[2]; ok.push_back(g); } } }
  std::sort(ok.begin(), ok.end(), [](const Cfg& x, const Cfg& y) { return x.cost < y.cost; });
  std::printf("algorithm ids %d, valid-at-all-M configurations timed %ld, kept %zu\n", nid, tried, ok.size());
  for (size_t i = 0; i < ok.size() && i < 8; ++i) { Cfg g = ok[i]; cublasLtMatmulAlgo_t c; cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, g.id, &c);
    cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_TILE_ID, &g.tile, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_STAGES_ID, &g.stages, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &g.splitk, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &g.red, 4);
    double t[3], cost = 0; for (int j = 0; j < 3; ++j) { t[j] = timeit(c, j, 20, 150); cost += W[j] * t[j]; }
    std::printf("  #%zu id=%d tile=%u stages=%u splitk=%u red=%u : M=1 %.1f M=32 %.1f M=256 %.1f | weighted %.1f us\n", i, g.id, g.tile, g.stages, g.splitk, g.red, t[0], t[1], t[2], cost); }
  return 0; }
