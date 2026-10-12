// Diagnostic: does touching a layer's weights in an earlier kernel (so they sit in the GPU L2 cache) shorten a later
// latency-bound projection kernel? Model-like pattern: 30 layers of 3.5 MB (gate+up of SmolLM2), each read once per pass.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
// Consumer: 192 blocks x 2 warps, each warp reads 8 columns x 576 of one matrix with a dependent MMA-like chain (fma accumulate).
__global__ void consumer(const __nv_bfloat16* gate, const __nv_bfloat16* up, const __nv_bfloat16* x, __nv_bfloat16* out) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4; const __nv_bfloat16* w = (warp ? up : gate) + (blockIdx.x * 8 + g) * 576;
  float d = 0.F; for (int depth = 0; depth < 576; depth += 16) { const uint32_t b = *reinterpret_cast<const uint32_t*>(w + depth + 2 * t), bb = *reinterpret_cast<const uint32_t*>(w + depth + 2 * t + 8);
    d = fmaf(float(b & 0xffff), __bfloat162float(x[depth + 2 * t]), d) + float(bb >> 16); }
  if (t == 0) out[warp * 1536 + blockIdx.x * 8 + g] = __float2bfloat16_rn(d); }
// Prefetch: blocks stride over the weight bytes with 64-byte steps (one load per cache line) and keep nothing.
__global__ void prefetch(const uint8_t* bytes, uint64_t count, uint32_t* sink) {
  const uint64_t lines = count / 64, per = (lines + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x); const uint64_t first = (uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * per; uint32_t acc = 0;
  for (uint64_t i = first; i < first + per && i < lines; ++i) acc ^= *reinterpret_cast<const uint32_t*>(bytes + i * 64);
  if (acc == 0x12345678u) sink[0] = acc; }
// Small latency-bound kernel standing in for attention/norm work that precedes the consumer.
__global__ void filler(const __nv_bfloat16* x, __nv_bfloat16* out) { float d = 0.F; for (int i = 0; i < 64; ++i) d = fmaf(d, 1.0001F, __bfloat162float(x[(threadIdx.x + i * 7) % 576])); if (threadIdx.x == 0) out[blockIdx.x] = __float2bfloat16_rn(d); }
int main() {
  const int L = 60; const size_t M = 1536 * 576; __nv_bfloat16 *w, *x, *out; uint32_t* sink; cudaMalloc(&w, L * 2 * M * 2); cudaMalloc(&x, 576 * 2); cudaMalloc(&out, 2 * 1536 * 2); cudaMalloc(&sink, 4);
  cudaMemset(w, 1, L * 2 * M * 2); cudaMemset(x, 1, 576 * 2); cudaStream_t s, side; cudaStreamCreate(&s); cudaStreamCreate(&side); cudaEvent_t fork, join; cudaEventCreateWithFlags(&fork, cudaEventDisableTiming); cudaEventCreateWithFlags(&join, cudaEventDisableTiming);
  // mode 0: fillers + consumer. mode 1: prefetch on a parallel graph branch during the fillers, joined before the consumer. mode 2: fillers only. mode 3: prefetch serial before fillers.
  auto run = [&](int mode, int fillers) { cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int l = 0; l < L; ++l) { const __nv_bfloat16* gate = w + size_t(l) * 2 * M;
      if (mode == 1) { cudaEventRecord(fork, s); cudaStreamWaitEvent(side, fork, 0); prefetch<<<192, 256, 0, side>>>(reinterpret_cast<const uint8_t*>(gate), 2 * M * 2, sink); cudaEventRecord(join, side); }
      if (mode == 3) prefetch<<<192, 256, 0, s>>>(reinterpret_cast<const uint8_t*>(gate), 2 * M * 2, sink);
      for (int f = 0; f < fillers; ++f) filler<<<24, 96, 0, s>>>(x, out);
      if (mode == 1) cudaStreamWaitEvent(s, join, 0);
      if (mode != 2) consumer<<<192, 64, 0, s>>>(gate, gate + M, x, out); }
    if (cudaStreamEndCapture(s, &g) != cudaSuccess) { std::printf("capture failed\n"); return -1.0; } cudaGraphInstantiate(&e, g, 0); for (int i = 0; i < 5; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s);
    double best = 1e9; for (int rep = 0; rep < 7; ++rep) { const double t0 = now(); for (int i = 0; i < 10; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); const double t = (now() - t0) / (10 * L); if (t < best) best = t; }
    cudaGraphExecDestroy(e); cudaGraphDestroy(g); return best; };
  for (int fillers : {1, 2, 3, 4}) { const double base = run(0, fillers), only = run(2, fillers), par = run(1, fillers), ser = run(3, fillers);
    std::printf("%d fillers: fillers only %.2f us | + consumer %.2f us | serial prefetch + fillers + consumer %.2f us | parallel-branch prefetch %.2f us (delta vs no prefetch %+.2f us)\n", fillers, only, base, ser, par, par - base); }
  return 0; }
