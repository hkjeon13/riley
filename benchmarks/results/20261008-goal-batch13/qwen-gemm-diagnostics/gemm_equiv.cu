// Diagnostic: bit-equivalence classes of cuBLASLt configurations for one BF16 projection shape.
// Reference = the first heuristic at 128 rows (Riley's anchor), which is reused for every row count. For each row count M,
// every valid (algorithm id, tile, stages, split-K, reduction) configuration is run on the same varied input; configurations
// whose whole output is bit-identical to the reference are "in class" and could replace it at that M without changing any
// output. Reports the fastest in-class configuration per M next to the reference and the fastest configuration overall.
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
struct Cfg { int id; uint32_t tile, stages, splitk, red; double t; bool same; };
int main(int argc, char** argv) {
  const int K = std::atoi(argv[1]), N = std::atoi(argv[2]); const int Ms[] = {1, 2, 4, 8, 16, 32, 64, 128, 256}; const int NM = 9, MAXM = 256;
  cublasLtHandle_t lt; cublasLtCreate(&lt); size_t ws = 32u << 20; void* work; cudaMalloc(&work, ws);
  const size_t elems = size_t(K) * N, bytes = elems * 2; const int R = int((700u << 20) / bytes) + 2;
  __nv_bfloat16 *Wall, *X, *Y; cudaMalloc(&Wall, bytes * R); cudaMalloc(&X, size_t(K) * MAXM * 2); cudaMalloc(&Y, size_t(N) * MAXM * 2);
  { std::vector<__nv_bfloat16> hw(elems), hx(size_t(K) * MAXM); uint64_t s = 0x9e3779b97f4a7c15ull;
    auto rnd = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return float(double(s >> 11) / double(1ull << 53)) * 2.0F - 1.0F; };
    for (auto& v : hw) v = __float2bfloat16(rnd() * 0.05F); for (auto& v : hx) v = __float2bfloat16(rnd() * 3.0F);
    cudaMemcpy(Wall, hw.data(), bytes, cudaMemcpyHostToDevice); for (int r = 1; r < R; ++r) cudaMemcpy(Wall + elems * r, Wall, bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(X, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice); }
  int rot = 0;
  cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t tr = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof(tr));
  cublasLtMatrixLayout_t la, lb[NM], lc[NM]; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K);
  for (int i = 0; i < NM; ++i) { cublasLtMatrixLayoutCreate(&lb[i], CUDA_R_16BF, K, Ms[i], K); cublasLtMatrixLayoutCreate(&lc[i], CUDA_R_16BF, N, Ms[i], N); }
  const float alpha = 1, beta = 0;
  auto call = [&](const cublasLtMatmulAlgo_t& algo, int i) { return cublasLtMatmul(lt, d, &alpha, Wall + elems * (rot++ % R), la, X, lb[i], &beta, Y, lc[i], Y, lc[i], &algo, work, ws, 0); };
  auto timeit = [&](const cublasLtMatmulAlgo_t& algo, int i, int warm, int reps) { for (int j = 0; j < warm; ++j) call(algo, i); cudaDeviceSynchronize(); double t0 = now(); for (int j = 0; j < reps; ++j) call(algo, i); cudaDeviceSynchronize(); return (now() - t0) / reps; };
  auto output = [&](const cublasLtMatmulAlgo_t& algo, int i, std::vector<uint16_t>& out) { cudaMemset(Y, 0, size_t(N) * Ms[i] * 2); if (call(algo, i) != CUBLAS_STATUS_SUCCESS) return false; if (cudaDeviceSynchronize() != cudaSuccess) return false; out.resize(size_t(N) * Ms[i]); cudaMemcpy(out.data(), Y, out.size() * 2, cudaMemcpyDeviceToHost); return true; };
  cublasLtMatmulAlgo_t anchor;
  { cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t r[1]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb[7], lc[7], lc[7], p, 1, r, &n); anchor = r[0].algo; }
  // Reference outputs per M and the anchor's own row invariance (row j of every M must equal row j at M=256).
  std::vector<std::vector<uint16_t>> ref(NM); for (int i = 0; i < NM; ++i) output(anchor, i, ref[i]);
  { long bad = 0; for (int i = 0; i < NM - 1; ++i) if (std::memcmp(ref[i].data(), ref[NM - 1].data(), ref[i].size() * 2) != 0) ++bad; std::printf("K=%d N=%d anchor row-invariance across M: %s\n", K, N, bad ? "BROKEN" : "holds"); }
  int ids[256]; int nid = 0; cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 256, ids, &nid);
  for (int i = 0; i < NM; ++i) { std::vector<Cfg> cfgs; long tried = 0, same = 0; std::vector<uint16_t> out;
    for (int a = 0; a < nid; ++a) { cublasLtMatmulAlgo_t algo;
      if (cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[a], &algo) != CUBLAS_STATUS_SUCCESS) continue;
      size_t sz = 0; uint32_t tiles[128] = {0}, stages[128] = {0}; int nt = 1, ns = 1; int32_t splitk_ok = 0; uint32_t redmask = 0;
      if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { nt = int(sz / 4); if (nt > 128) nt = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, nt * 4, &sz); }
      sz = 0; if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { ns = int(sz / 4); if (ns > 128) ns = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, ns * 4, &sz); }
      cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &splitk_ok, 4, &sz); cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &redmask, 4, &sz);
      const uint32_t splits[] = {1, 2, 3, 4, 5, 6, 8, 12, 16}; const uint32_t reds[] = {CUBLASLT_REDUCTION_SCHEME_NONE, CUBLASLT_REDUCTION_SCHEME_INPLACE, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE, CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE};
      for (int ti = 0; ti < nt; ++ti) for (int si = 0; si < ns; ++si) for (uint32_t sk : splits) { if (sk > 1 && !splitk_ok) break;
        for (uint32_t rs : reds) { if ((sk == 1) != (rs == CUBLASLT_REDUCTION_SCHEME_NONE)) continue; if (rs != CUBLASLT_REDUCTION_SCHEME_NONE && !(redmask & rs)) continue;
          cublasLtMatmulAlgo_t c = algo; cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[ti], 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages[si], 4);
          cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &rs, 4);
          cublasLtMatmulHeuristicResult_t chk{}; if (!(cublasLtMatmulAlgoCheck(lt, d, la, lb[i], lc[i], lc[i], &c, &chk) == CUBLAS_STATUS_SUCCESS && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= ws)) continue;
          if (!output(c, i, out)) { cudaGetLastError(); continue; } ++tried;
          Cfg g{ids[a], tiles[ti], stages[si], sk, rs, 0, std::memcmp(out.data(), ref[i].data(), out.size() * 2) == 0}; if (g.same) ++same;
          g.t = timeit(c, i, 2, 8); cfgs.push_back(g); } } }
    auto rebuild = [&](const Cfg& g) { cublasLtMatmulAlgo_t c; cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, g.id, &c);
      cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_TILE_ID, &g.tile, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_STAGES_ID, &g.stages, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &g.splitk, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &g.red, 4); return c; };
    std::sort(cfgs.begin(), cfgs.end(), [](const Cfg& x, const Cfg& y) { return x.t < y.t; });
    const double tref = timeit(anchor, i, 20, 200); double best_same = 1e9, best_any = 1e9; Cfg bs{}, ba{}; int ns_ = 0, na_ = 0;
    for (const Cfg& g : cfgs) { if (g.same && ns_ < 6) { ++ns_; const double t = timeit(rebuild(g), i, 20, 200); if (t < best_same) { best_same = t; bs = g; } } if (na_ < 6) { ++na_; const double t = timeit(rebuild(g), i, 20, 200); if (t < best_any) { best_any = t; ba = g; } } }
    std::printf("M=%3d configs %4ld in-class %4ld | anchor %6.1f us | best in-class %6.1f us (id=%d tile=%u stages=%u splitk=%u red=%u) | best any %6.1f us (id=%d tile=%u stages=%u splitk=%u red=%u)\n",
                Ms[i], tried, same, tref, best_same, bs.id, bs.tile, bs.stages, bs.splitk, bs.red, best_any, ba.id, ba.tile, ba.stages, ba.splitk, ba.red); std::fflush(stdout); }
  return 0; }
