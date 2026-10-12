// Diagnostic: cost of early-exit CTAs in the D128 producer's capacity-sized grid (pages x KV heads x rows, 256 threads each).
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void gated(const uint32_t* pages, float* out) { const uint32_t row = blockIdx.z; if (blockIdx.x >= pages[row]) return; out[(blockIdx.z * gridDim.y + blockIdx.y) * 64 + blockIdx.x] = float(threadIdx.x); }
int main() { uint32_t* pages; float* out; cudaMalloc(&pages, 256 * 4); cudaMalloc(&out, 256 * 2 * 64 * 4); uint32_t host[256]; cudaStream_t s; cudaStreamCreate(&s);
  struct Case { const char* name; int extent, rows, live_pages; };
  const Case cases[] = {{"natural decode C32: extent 64, 32 rows, 10 live pages", 64, 32, 10}, {"same with extent 16", 16, 32, 10}, {"natural mixed: extent 64, 256 rows, 10 live pages", 64, 256, 10}, {"same with extent 32", 32, 256, 10}, {"same with extent 16", 16, 256, 10},
                        {"natural mixed: extent 64, 256 rows, 25 live pages", 64, 256, 25}, {"same with extent 32", 32, 256, 25}, {"fixed mixed: extent 16, 256 rows, 8 live pages", 16, 256, 8}, {"fixed decode C32: extent 16, 32 rows, 10 live pages", 16, 32, 10}, {"C1 decode fixed: extent 16, 1 row, 10 pages", 16, 1, 10}, {"C1 decode natural: extent 64, 1 row, 10 pages", 64, 1, 10}};
  for (const Case& c : cases) { for (int i = 0; i < 256; ++i) host[i] = c.live_pages; cudaMemcpy(pages, host, sizeof host, cudaMemcpyHostToDevice);
    cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal); for (int i = 0; i < 36; ++i) gated<<<dim3(c.extent, 2, c.rows), 256, 0, s>>>(pages, out); cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&e, g, 0);
    for (int i = 0; i < 5; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); double best = 1e9; for (int rep = 0; rep < 7; ++rep) { const double t0 = now(); for (int i = 0; i < 20; ++i) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); const double t = (now() - t0) / (20 * 36); if (t < best) best = t; }
    std::printf("%-58s grid %6d CTAs: %6.2f us per launch\n", c.name, c.extent * 2 * c.rows, best); cudaGraphExecDestroy(e); cudaGraphDestroy(g); }
  return 0; }
