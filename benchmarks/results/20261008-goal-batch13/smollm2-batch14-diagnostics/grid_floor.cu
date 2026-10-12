// Diagnostic: cost of inactive blocks in a capacity-sized launch grid (SmolLM2 shared32 decode kernels launch grids for 32 rows
// even when one row is live). Each kernel reads the live-row word and returns unless its block row is live; timed by graph replay.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
__global__ void gated(const uint32_t* live, float* out, int rows_per_y) {
  const uint32_t rows = *live; if (blockIdx.y / rows_per_y >= rows) return;
  out[(blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x] = 1.0F; }
int main() {
  uint32_t* live; float* out; cudaMalloc(&live, 4); cudaMalloc(&out, 4096 * 256 * 4); const uint32_t one = 1; cudaMemcpy(live, &one, 4, cudaMemcpyHostToDevice);
  cudaStream_t s; cudaStreamCreate(&s);
  struct Case { const char* name; int gx, gy, block, rows_per_y; };
  const Case cases[] = {{"scores now (32,96)x32", 32, 96, 32, 3}, {"scores live-only (32,3)x32", 32, 3, 32, 3}, {"scores (4,96)x256", 4, 96, 256, 3}, {"scores (4,32)x768", 4, 32, 768, 1}, {"scores (2,32)x768", 2, 32, 768, 1}, {"scores (1,32)x1024", 1, 32, 1024, 1},
                        {"values now (8,96)x96", 8, 96, 96, 3}, {"values live-only (8,3)x96", 8, 3, 96, 3}, {"values (8,32)x288", 8, 32, 288, 1}, {"values (4,32)x576", 4, 32, 576, 1}, {"values (2,32)x576", 2, 32, 576, 1}};
  for (const Case& c : cases) { const int calls = 256; cudaGraph_t g; cudaGraphExec_t e; cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
    for (int i = 0; i < calls; ++i) gated<<<dim3(c.gx, c.gy), c.block, 0, s>>>(live, out, c.rows_per_y);
    cudaStreamEndCapture(s, &g); cudaGraphInstantiate(&e, g, 0); for (int w = 0; w < 5; ++w) cudaGraphLaunch(e, s); cudaStreamSynchronize(s);
    double best = 1e9; for (int rep = 0; rep < 5; ++rep) { const double t0 = now(); for (int w = 0; w < 20; ++w) cudaGraphLaunch(e, s); cudaStreamSynchronize(s); const double t = (now() - t0) / (20 * calls); if (t < best) best = t; }
    std::printf("%-30s %5.2f us/kernel\n", c.name, best); cudaGraphExecDestroy(e); cudaGraphDestroy(g); }
  return 0; }
