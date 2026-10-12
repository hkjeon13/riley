// Diagnostic: how much does CUDA-graph replay save over eager launches for a Qwen2.5-3B-shaped decode step?
// One step = 36 layers x (5 cuBLASLt projections + 11 small kernels) + final norm + LM head + argmax-sized kernel.
// The small kernels are placeholders with realistic element counts: the question is the per-kernel boundary cost,
// which does not depend on what the small kernels compute. Weights are distinct per layer (cold, ~6 GB total).
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void ew(__nv_bfloat16* x, size_t n) { for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) * 1.0F); }
struct Gemm { cublasLtMatmulDesc_t d; cublasLtMatrixLayout_t la, lb, lc; cublasLtMatmulAlgo_t algo; int K, N; };
int main(int argc, char** argv) {
  const int M = argc > 1 ? std::atoi(argv[1]) : 32, L = 36, H = 2048, QKV = 2560, I = 11008, V = 151936;
  cublasLtHandle_t lt; cublasLtCreate(&lt); const size_t ws = 32u << 20; void* work; cudaMalloc(&work, ws); cudaStream_t s; cudaStreamCreate(&s);
  auto mk = [&](int K, int N) { Gemm g; g.K = K; g.N = N; cublasLtMatmulDescCreate(&g.d, CUBLAS_COMPUTE_32F, CUDA_R_32F); cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(g.d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t));
    cublasLtMatrixLayoutCreate(&g.la, CUDA_R_16BF, K, N, K); cublasLtMatrixLayoutCreate(&g.lb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&g.lc, CUDA_R_16BF, N, M, N);
    cublasLtMatrixLayout_t b128, c128; cublasLtMatrixLayoutCreate(&b128, CUDA_R_16BF, K, 128, K); cublasLtMatrixLayoutCreate(&c128, CUDA_R_16BF, N, 128, N);
    cublasLtMatmulPreference_t p; cublasLtMatmulPreferenceCreate(&p); size_t w = ws; cublasLtMatmulPreferenceSetAttribute(p, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &w, sizeof(w));
    cublasLtMatmulHeuristicResult_t r[1]; int n = 0; cublasLtMatmulAlgoGetHeuristic(lt, g.d, g.la, b128, c128, c128, p, 1, r, &n); g.algo = r[0].algo; return g; };
  Gemm gq = mk(H, QKV), go = mk(H, H), gg = mk(H, I), gd = mk(I, H), gh = mk(H, V);
  auto alloc = [&](size_t elems) { __nv_bfloat16* p; if (cudaMalloc(&p, elems * 2) != cudaSuccess) { std::printf("alloc failed\n"); std::exit(1); } cudaMemset(p, 0x11, elems * 2); return p; };
  std::vector<__nv_bfloat16*> wq(L), wo(L), wg(L), wu(L), wd(L);
  for (int l = 0; l < L; ++l) { wq[l] = alloc(size_t(H) * QKV); wo[l] = alloc(size_t(H) * H); wg[l] = alloc(size_t(H) * I); wu[l] = alloc(size_t(H) * I); wd[l] = alloc(size_t(I) * H); }
  __nv_bfloat16* wh = alloc(size_t(H) * V);
  __nv_bfloat16 *hid = alloc(size_t(M) * H), *nrm = alloc(size_t(M) * H), *qkv = alloc(size_t(M) * QKV), *att = alloc(size_t(M) * H), *gate = alloc(size_t(M) * I), *up = alloc(size_t(M) * I), *ctx = alloc(size_t(M) * H), *logits = alloc(size_t(M) * V);
  const float alpha = 1, beta = 0;
  auto gemm = [&](Gemm& g, const __nv_bfloat16* W, const __nv_bfloat16* X, __nv_bfloat16* Y) { cublasLtMatmul(lt, g.d, &alpha, W, g.la, X, g.lb, &beta, Y, g.lc, Y, g.lc, &g.algo, work, ws, s); };
  auto small = [&](__nv_bfloat16* p, size_t n) { ew<<<unsigned((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256), 256, 0, s>>>(p, n); };
  int kernels = 0;
  auto step = [&](bool count) { int k = 0;
    for (int l = 0; l < L; ++l) {
      small(nrm, size_t(M) * H); gemm(gq, wq[l], nrm, qkv); small(qkv, size_t(M) * QKV); small(qkv, size_t(M) * H); small(qkv, size_t(M) * 256); small(qkv, size_t(M) * 512);
      small(att, size_t(M) * H); small(att, size_t(M) * H); gemm(go, wo[l], att, ctx); small(ctx, size_t(M) * H);
      gemm(gg, wg[l], ctx, gate); gemm(gg, wu[l], ctx, up); small(gate, size_t(M) * I); small(up, size_t(M) * I); gemm(gd, wd[l], up, hid); small(hid, size_t(M) * H); k += 16; }
    small(nrm, size_t(M) * H); gemm(gh, wh, nrm, logits); small(logits, size_t(M) * V); k += 3; if (count) kernels = k; };
  for (int i = 0; i < 5; ++i) step(true); cudaStreamSynchronize(s);
  const int reps = 60; double t0 = now(); for (int i = 0; i < reps; ++i) { step(false); cudaStreamSynchronize(s); } const double eager = (now() - t0) / reps;
  cudaGraph_t graph; cudaGraphExec_t exec; cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal); step(false); const cudaError_t ce = cudaStreamEndCapture(s, &graph);
  size_t nodes = 0; cudaGraphGetNodes(graph, nullptr, &nodes); const cudaError_t ci = cudaGraphInstantiate(&exec, graph, 0);
  for (int i = 0; i < 5; ++i) cudaGraphLaunch(exec, s); cudaStreamSynchronize(s);
  t0 = now(); for (int i = 0; i < reps; ++i) { cudaGraphLaunch(exec, s); cudaStreamSynchronize(s); } const double graphed = (now() - t0) / reps;
  // eager again after the graph, to rule out ordering/thermal effects
  t0 = now(); for (int i = 0; i < reps; ++i) { step(false); cudaStreamSynchronize(s); } const double eager2 = (now() - t0) / reps;
  std::printf("M=%d launches/step=%d graph nodes=%zu (capture %d instantiate %d): eager %.1f us, graph %.1f us, eager again %.1f us | graph saves %.1f us (%.1f%%), %.2f us per launch\n",
              M, kernels, nodes, int(ce), int(ci), eager, graphed, eager2, (eager + eager2) / 2 - graphed, ((eager + eager2) / 2 - graphed) / ((eager + eager2) / 2) * 100, ((eager + eager2) / 2 - graphed) / double(nodes));
  return 0; }
