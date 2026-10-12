// Diagnostic (Qwen2.5-3B shapes): does prefetching a projection's weights on a parallel graph branch, while latency-bound
// kernels run on the main branch, shorten the following cuBLASLt GEMM? Layers rotate over distinct weight buffers so L2 is cold.
// argv: N K (weight N x K, BF16), gap kernels count, prefetch megabytes (0 = whole weight).
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void filler(const __nv_bfloat16* x, __nv_bfloat16* out, int iters) { float d = 0.F; for (int i = 0; i < iters; ++i) d = fmaf(d, 1.0001F, __bfloat162float(x[(threadIdx.x + i * 7) % 2048])); if (threadIdx.x == 0) out[blockIdx.x] = __float2bfloat16_rn(d); }
// Gap kernel whose blocks past `work` touch one word per cache line of a slice of the next weights.
__global__ void filler_prefetch(const __nv_bfloat16* x, __nv_bfloat16* out, const uint8_t* bytes, uint64_t count, int work, int iters) {
  if (int(blockIdx.x) < work) { float d = 0.F; for (int i = 0; i < iters; ++i) d = fmaf(d, 1.0001F, __bfloat162float(x[(threadIdx.x + i * 7) % 2048])); if (threadIdx.x == 0) out[blockIdx.x] = __float2bfloat16_rn(d); return; }
  const uint64_t slots = uint64_t(gridDim.x - work) * blockDim.x, lines = count / 64, per = (lines + slots - 1) / slots, first = (uint64_t(blockIdx.x - work) * blockDim.x + threadIdx.x) * per; uint32_t acc = 0;
  for (uint64_t i = first; i < first + per && i < lines; ++i) acc ^= *reinterpret_cast<const volatile uint32_t*>(bytes + i * 64); (void)acc; }
__global__ void prefetch(const uint8_t* bytes, uint64_t count) {
  const uint64_t slots = uint64_t(gridDim.x) * blockDim.x, lines = count / 64, per = (lines + slots - 1) / slots, first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * per; uint32_t acc = 0;
  for (uint64_t i = first; i < first + per && i < lines; ++i) acc ^= *reinterpret_cast<const volatile uint32_t*>(bytes + i * 64); (void)acc; }
int main(int argc, char** argv) {
  const int N = std::atoi(argv[1]), K = std::atoi(argv[2]), gaps = std::atoi(argv[3]); const double mb = std::atof(argv[4]); const int M = argc > 5 ? std::atoi(argv[5]) : 1; const int iters = argc > 6 ? std::atoi(argv[6]) : 96; const int extra = argc > 7 ? std::atoi(argv[7]) : 256;
  const size_t wbytes = size_t(N) * K * 2; const int L = int((2000ull << 20) / wbytes) < 48 ? int((2000ull << 20) / wbytes) : 48; const uint64_t pbytes = mb > 0 ? uint64_t(mb * 1048576) & ~63ull : wbytes;
  __nv_bfloat16 *w, *x, *y, *out; cudaMalloc(&w, wbytes * L); cudaMalloc(&x, size_t(128) * K * 2); cudaMalloc(&y, size_t(128) * N * 2); cudaMalloc(&out, 4096); cudaMemset(w, 1, wbytes * L); cudaMemset(x, 1, size_t(128) * K * 2);
  cublasLtHandle_t lt; cublasLtCreate(&lt); cublasLtMatmulDesc_t d; cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F); const cublasOperation_t t = CUBLAS_OP_T; cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
  // Row-major W[N][K] as column-major K x N, transposed: Y(N x M) = W^T(N x K) * X(K x M).
  cublasLtMatrixLayout_t la, lb, lc; cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, N, K); cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, M, K); cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N);
  cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref); size_t ws = 64 << 20; cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws); void* work; cudaMalloc(&work, ws);
  cublasLtMatmulHeuristicResult_t h; int found = 0; cublasLtMatmulAlgoGetHeuristic(lt, d, la, lb, lc, lc, pref, 1, &h, &found); if (!found) { std::printf("no algo\n"); return 1; }
  const float alpha = 1, beta = 0; cudaStream_t s, side; cudaStreamCreate(&s); cudaStreamCreate(&side); cudaEvent_t fork, join; cudaEventCreateWithFlags(&fork, cudaEventDisableTiming); cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
  // mode 0: gaps + GEMM. 1: side-branch prefetch during gaps, joined before GEMM. 2: gaps only. 3: side-branch prefetch, joined only after the GEMM.
  auto run = [&](int mode) { cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < L; ++l) { const __nv_bfloat16* wl = w + size_t(l) * N * K;
      if (mode == 1 || mode == 3) { cudaEventRecord(fork, s); cudaStreamWaitEvent(side, fork, 0); prefetch<<<384, 256, 0, side>>>(reinterpret_cast<const uint8_t*>(wl), pbytes); cudaEventRecord(join, side); }
      for (int f = 0; f < gaps; ++f) { if (mode == 4 || mode == 5) { const uint64_t slice = (pbytes / gaps) & ~63ull; filler_prefetch<<<20 + extra, 256, 0, s>>>(x, out, reinterpret_cast<const uint8_t*>(wl) + slice * f, slice, 20, iters); } else filler<<<20, 256, 0, s>>>(x, out, iters); }
      if (mode == 1) cudaStreamWaitEvent(s, join, 0);
      if (mode != 2 && mode != 5) cublasLtMatmul(lt, d, &alpha, wl, la, x, lb, &beta, y, lc, y, lc, &h.algo, work, ws, s);
      if (mode == 3) cudaStreamWaitEvent(s, join, 0); }
    if (cudaStreamEndCapture(s, &g) != cudaSuccess) { std::printf("capture failed\n"); return -1.0; } cudaGraphInstantiate(&e, g, 0); for (int i = 0; i < 3; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s);
    double best = 1e9; for (int rep = 0; rep < 5; ++rep) { const double t0 = now(); for (int i = 0; i < 4; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); const double tt = (now() - t0) / (4 * L); if (tt < best) best = tt; }
    cudaGraphExecDestroy(e); cudaGraphDestroy(g); return best; };
  const double only = run(2), base = run(0), joined = run(1), late = run(3), inside = run(4), gapsp = run(5);
  std::printf("N=%d K=%d M=%d weight %.1f MB, %d layers, %d gap kernels, prefetch %.1f MB | gaps only %.2f us | gaps+GEMM %.2f us (GEMM %.2f) | side prefetch joined before GEMM %.2f us (%+.2f) | joined after GEMM %.2f us (%+.2f) | prefetch blocks inside the gap kernels %.2f us (%+.2f); gaps alone with prefetch %.2f us (%+.2f) so GEMM after prefetch %.2f\n", N, K, M, wbytes / 1048576.0, L, gaps, pbytes / 1048576.0, only, base, base - only, joined, joined - base, late, late - base, inside, inside - base, gapsp, gapsp - only, inside - gapsp);
  return 0; }
