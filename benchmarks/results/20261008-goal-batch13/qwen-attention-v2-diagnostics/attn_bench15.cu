// Diagnostic microbenchmark 15 (enumerated pinned configurations; shapes padded to >= 8 rows and key multiples of 16):
// derived from microbenchmark 14: benchmark 9's shape sweep with cuBLASLt algorithms pinned at one reference shape
// (the way Riley anchors its projection GEMMs), plus timings of padded batched launches.
// One fixed 1024-token sequence (values depend only on absolute position). A chunk [P0, P0+R) is computed with exact
// dims, then with padded key count, padded row count, and replicated batch entries; every real row must stay bit-identical.
#include <cublasLt.h>
#include <chrono>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>
#include <cstdint>
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

static cublasLtHandle_t g_lt; static void* g_work; static size_t g_ws = 32u << 20; static cublasLtMatmulAlgo_t g_qk_algo, g_pv_algo; static bool g_pinned = false; static long g_check_fail = 0;
struct Mm { cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t a, b, c; };
static Mm make(bool qk, int KP, int R8, int B) { Mm m; cublasLtMatmulDescCreate(&m.d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (qk) { cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(m.d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatrixLayoutCreate(&m.a, CUDA_R_16BF, D, KP, D); cublasLtMatrixLayoutCreate(&m.b, CUDA_R_16BF, D, R8, D); cublasLtMatrixLayoutCreate(&m.c, CUDA_R_16BF, KP, R8, KP); }
  else { cublasLtMatrixLayoutCreate(&m.a, CUDA_R_16BF, D, KP, D); cublasLtMatrixLayoutCreate(&m.b, CUDA_R_16BF, KP, R8, KP); cublasLtMatrixLayoutCreate(&m.c, CUDA_R_16BF, D, R8, D); }
  const int32_t batch = B; const int64_t sa = int64_t(KP) * D, sb = qk ? int64_t(R8) * D : int64_t(R8) * KP, sc = qk ? int64_t(R8) * KP : int64_t(R8) * D;
  const cublasLtMatrixLayout_t layouts[3] = {m.a, m.b, m.c}; const int64_t strides[3] = {sa, sb, sc};
  for (int i = 0; i < 3; ++i) { cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)); cublasLtMatrixLayoutSetAttribute(layouts[i], CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])); }
  return m; }
static void destroy(Mm& m) { cublasLtMatmulDescDestroy(m.d); cublasLtMatrixLayoutDestroy(m.a); cublasLtMatrixLayoutDestroy(m.b); cublasLtMatrixLayoutDestroy(m.c); }
static cublasLtMatmulAlgo_t heuristic(Mm& m) { cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_ws, sizeof(g_ws));
  cublasLtMatmulHeuristicResult_t r[1]; int n = 0; cublasLtMatmulAlgoGetHeuristic(g_lt, m.d, m.a, m.b, m.c, m.c, p, 1, r, &n); cublasLtMatmulPreferenceDestroy(p); return r[0].algo; }
static bool matmul(Mm& m, const cublasLtMatmulAlgo_t& pinned, const void* A, const void* Bp, void* C) { const float one = 1.0F, zero = 0.0F; cublasLtMatmulAlgo_t algo = pinned;
  if (g_pinned) { cublasLtMatmulHeuristicResult_t chk{}; if (cublasLtMatmulAlgoCheck(g_lt, m.d, m.a, m.b, m.c, m.c, &algo, &chk) != CUBLAS_STATUS_SUCCESS || chk.state != CUBLAS_STATUS_SUCCESS || chk.workspaceSize > g_ws) { ++g_check_fail; algo = heuristic(m); } }
  else algo = heuristic(m);
  return cublasLtMatmul(g_lt, m.d, &one, A, m.a, Bp, m.b, &zero, C, m.c, C, m.c, &algo, g_work, g_ws, 0) == CUBLAS_STATUS_SUCCESS; }
// Returns om of the LAST batch copy pair: [KVH][RP*G][D]
static std::vector<__nv_bfloat16> run(void*, int P0, int R, int KP, int RP, int copies, double* micros = nullptr) {
  const int keys = P0 + R, B = copies * KVH, R8 = RP * G; __nv_bfloat16 *qm, *km, *vm, *sc, *om;
  cudaMalloc(&qm, size_t(B) * R8 * D * 2); cudaMalloc(&om, size_t(B) * R8 * D * 2); cudaMalloc(&km, size_t(B) * KP * D * 2); cudaMalloc(&vm, size_t(B) * KP * D * 2); cudaMalloc(&sc, size_t(B) * R8 * KP * 2);
  const float scale = 1.0F / std::sqrt(float(D)); Mm qk = make(true, KP, R8, B), pv = make(false, KP, R8, B);
  auto all = [&] { fill_q<<<2048, 256>>>(g_q, qm, P0, R, RP, copies); fill_kv<<<2048, 256>>>(g_k, g_v, km, vm, keys, KP, copies);
    matmul(qk, g_qk_algo, km, qm, sc); causal_softmax<<<dim3(R8, B), 256>>>(sc, KP, RP, R, P0, scale); matmul(pv, g_pv_algo, vm, sc, om); };
  all(); cudaDeviceSynchronize();
  if (micros) { for (int i = 0; i < 20; ++i) all(); cudaDeviceSynchronize(); const auto t0 = std::chrono::steady_clock::now(); for (int i = 0; i < 200; ++i) all(); cudaDeviceSynchronize();
    *micros = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 200; }
  std::vector<__nv_bfloat16> ho(size_t(KVH) * R8 * D); cudaMemcpy(ho.data(), om + size_t(B - KVH) * R8 * D, ho.size() * 2, cudaMemcpyDeviceToHost);
  destroy(qk); destroy(pv); cudaFree(qm); cudaFree(km); cudaFree(vm); cudaFree(sc); cudaFree(om); return ho; }
// element of row (absolute position), kv head, group head, dim from a run with (P0, RP)
static inline uint16_t at(const std::vector<__nv_bfloat16>& o, int P0, int RP, int pos, int kvh, int g, int d) {
  uint16_t bits; const __nv_bfloat16 v = o[((size_t(kvh) * RP + (pos - P0)) * G + g) * D + d]; memcpy(&bits, &v, 2); return bits; }
int main() {
  void* h = nullptr; cublasLtCreate(&g_lt); cudaMalloc(&g_work, g_ws);
  { const size_t kn = size_t(KVH) * MAXT * D, qn = size_t(MAXT) * QH * D; std::vector<__nv_bfloat16> hk(kn), hv(kn), hq(qn);
    for (size_t i = 0; i < kn; ++i) { hk[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F); hv[i] = __float2bfloat16(float((i * 40503u + 17) % 1999) / 999.0F - 1.0F); }
    for (size_t i = 0; i < qn; ++i) hq[i] = __float2bfloat16(float((i * 2246822519u) % 2003) / 500.0F - 2.0F);
    cudaMalloc(&g_k, kn * 2); cudaMalloc(&g_v, kn * 2); cudaMalloc(&g_q, qn * 2);
    cudaMemcpy(g_k, hk.data(), kn * 2, cudaMemcpyHostToDevice); cudaMemcpy(g_v, hv.data(), kn * 2, cudaMemcpyHostToDevice); cudaMemcpy(g_q, hq.data(), qn * 2, cudaMemcpyHostToDevice); }
  for (int pass = 0; pass < 2; ++pass) { g_pinned = pass == 1; g_check_fail = 0;
  if (g_pinned) {
    // Enumerate (id, tile, stages) with no split-K and keep those AlgoCheck accepts for every padded probe shape.
    const int KPs[] = {16, 32, 64, 128, 256, 400, 528, 1024}, R8s[] = {64, 128, 512, 1024, 2048}, Bs[] = {2, 4, 8, 16, 32, 64};
    int ids[256]; int nid = 0; cublasLtMatmulAlgoGetIds(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 256, ids, &nid);
    for (int op = 0; op < 2; ++op) { int valid = 0; bool have = false; cublasLtMatmulAlgo_t first{}; int fid = -1; uint32_t ftile = 0, fstages = 0;
      for (int a = 0; a < nid; ++a) { cublasLtMatmulAlgo_t algo; if (cublasLtMatmulAlgoInit(g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[a], &algo) != CUBLAS_STATUS_SUCCESS) continue;
        size_t sz = 0; uint32_t tiles[128] = {0}, stages[128] = {0}; int nt = 1, ns = 1;
        if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { nt = int(sz / 4); if (nt > 128) nt = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS, tiles, nt * 4, &sz); }
        sz = 0; if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &sz) == CUBLAS_STATUS_SUCCESS && sz) { ns = int(sz / 4); if (ns > 128) ns = 128; cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_STAGES_IDS, stages, ns * 4, &sz); }
        for (int ti = 0; ti < nt; ++ti) for (int si = 0; si < ns; ++si) { cublasLtMatmulAlgo_t c = algo; const uint32_t one = 1, none = CUBLASLT_REDUCTION_SCHEME_NONE;
          cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_TILE_ID, &tiles[ti], 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages[si], 4);
          cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &one, 4); cublasLtMatmulAlgoConfigSetAttribute(&c, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &none, 4);
          bool ok = true;
          for (int kp : KPs) { if (!ok) break; for (int r8 : R8s) { if (!ok) break; for (int bb : Bs) { if (size_t(bb) * r8 * kp * 2 > (size_t(2) << 30)) continue; Mm m = make(op == 0, kp, r8, bb); cublasLtMatmulHeuristicResult_t chk{};
            ok = cublasLtMatmulAlgoCheck(g_lt, m.d, m.a, m.b, m.c, m.c, &c, &chk) == CUBLAS_STATUS_SUCCESS && chk.state == CUBLAS_STATUS_SUCCESS && chk.workspaceSize <= g_ws; destroy(m); if (!ok) break; } } }
          if (ok) { ++valid; if (!have) { have = true; first = c; fid = ids[a]; ftile = tiles[ti]; fstages = stages[si]; } } } }
      std::printf("%s: configurations valid on every padded probe shape: %d; pinned id=%d tile=%u stages=%u\n", op == 0 ? "QK" : "PV", valid, fid, ftile, fstages);
      if (!have) { std::printf("no pinnable configuration\n"); return 1; }
      (op == 0 ? g_qk_algo : g_pv_algo) = first; } }
  // Canonical value of every position: single-row chunks with exact dims.
  std::vector<std::vector<uint16_t>> canon(MAXT);
  const int probe[] = {0, 1, 2, 15, 16, 17, 31, 63, 64, 100, 127, 128, 199, 255, 256, 300, 397, 511, 512, 700, 767, 900, 1023};
  for (int p : probe) { auto o = run(h, p, 1, (p + 1 + 15) / 16 * 16, 8, 1); canon[p].resize(QH * D); for (int kvh = 0; kvh < KVH; ++kvh) for (int g = 0; g < G; ++g) for (int d = 0; d < D; ++d) canon[p][(kvh * G + g) * D + d] = at(o, p, 8, p, kvh, g, d); }
  long cases = 0, bad_cases = 0; long elements = 0, differing = 0;
  const int P0s[] = {0, 5, 100, 250, 500, 768}, Rs[] = {1, 3, 16, 60, 128, 256}, copiesList[] = {1, 4, 16};
  for (int P0 : P0s) for (int R : Rs) { if (P0 + R > MAXT) continue; const int keys = P0 + R;
    const int KPs[] = {(keys + 15) / 16 * 16, (keys + 63) / 64 * 64, 256, 512, 1024}; const int RPs[] = {(R + 7) / 8 * 8, 64, 256};
    for (int KP : KPs) { if (KP < keys) continue; for (int RP : RPs) { if (RP < R) continue; for (int copies : copiesList) {
      if (size_t(copies) * KVH * RP * G * KP * 2 > (size_t(3) << 30)) continue;
      auto o = run(h, P0, R, KP, RP, copies); long diff = 0, n = 0;
      for (int p : probe) { if (p < P0 || p >= P0 + R) continue;
        for (int kvh = 0; kvh < KVH; ++kvh) for (int g = 0; g < G; ++g) for (int d = 0; d < D; ++d) { ++n; if (at(o, P0, RP, p, kvh, g, d) != canon[p][(kvh * G + g) * D + d]) ++diff; } }
      if (n == 0) continue; ++cases; elements += n; differing += diff;
      if (diff) { ++bad_cases; if (bad_cases <= 12) std::printf("DIFF P0=%d R=%d keysPad=%d rowsPad=%d batch=%d: %ld of %ld elements\n", P0, R, KP, RP, copies * KVH, diff, n); } } } } }
  std::printf("%s: cases %ld, cases with any difference %ld, elements compared %ld, differing %ld, algo-check fallbacks %ld\n", g_pinned ? "PINNED cuBLASLt algorithms" : "per-shape heuristic", cases, bad_cases, elements, differing, g_check_fail);
  // timings: {sequences, rows per sequence (padded launch covers them), first position}
  const int tc[][3] = {{1, 128, 200}, {1, 224, 0}, {1, 174, 224}, {2, 112, 18}, {4, 56, 100}, {8, 28, 4}, {3, 60, 60}};
  for (auto& c : tc) { double us = 0; run(h, c[2], c[1], ((c[2] + c[1]) + 15) / 16 * 16, (c[1] + 7) / 8 * 8, c[0], &us); std::printf("  timing %d sequence(s) x %d rows, keys %d: %6.1f us per launch set\n", c[0], c[1], c[2] + c[1], us); }
  }
  return 0; }
