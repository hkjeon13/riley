// Diagnostic microbenchmark 8 (chunk-split invariance of microbenchmark 7's pipeline): prefill-row attention through batched tensor-core GEMMs (Qwen D128, QH=16, KVH=2).
// Per (sequence, kv head): gather Q/K/V into contiguous matrices, scores = K^T Q (BF16 GEMM), causal softmax kernel,
// output = V P (BF16 GEMM). Timed against the per-row paged producer numbers of attn_bench6 for the same shapes.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int D = 128, QH = 16, KVH = 2, G = 8;
// query[row][head][D] (engine layout) -> qm[batch = seq * KVH + kvh][(local_row * G + g)][D]
__global__ void gather_q(const __nv_bfloat16* query, __nv_bfloat16* qm, int rows_per_seq, int nseq) {
  const size_t n = size_t(nseq) * rows_per_seq * QH * D;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = i % D; const size_t rh = i / D; const int head = rh % QH; const size_t row = rh / QH;
    const int seq = row / rows_per_seq, local = row % rows_per_seq, kvh = head / G, g = head % G;
    qm[((size_t(seq) * KVH + kvh) * rows_per_seq * G + size_t(local) * G + g) * D + d] = query[i]; } }
// paged pools pool[block][kvh][16][D] with block ids per (seq, page) -> km[batch][key][D]
__global__ void gather_kv(const __nv_bfloat16* kpool, const __nv_bfloat16* vpool, const uint32_t* ids, __nv_bfloat16* km, __nv_bfloat16* vm, int keys, int pages, int nseq) {
  const size_t n = size_t(nseq) * KVH * keys * D;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = i % D; const size_t bk = i / D; const int key = bk % keys; const int batch = bk / keys; const int seq = batch / KVH, kvh = batch % KVH;
    const size_t src = ((size_t(ids[seq * pages + key / 16]) * KVH + kvh) * 16 + key % 16) * D + d;
    km[i] = kpool[src]; vm[i] = vpool[src]; } }
// scores[batch][column = local_row * G + g][keys] (column-major keys x R8): causal softmax in place, BF16 probabilities.
__global__ void causal_softmax(__nv_bfloat16* scores, int keys, int rows_per_seq, int first_position, float scale) {
  const int column = blockIdx.x, batch = blockIdx.y; const int local = column / G; const int visible = first_position + local + 1;
  __nv_bfloat16* s = scores + (size_t(batch) * rows_per_seq * G + column) * keys;
  __shared__ float red[256];
  float m = -CUDART_INF_F;
  for (int k = threadIdx.x; k < visible; k += blockDim.x) m = fmaxf(m, __bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)));
  red[threadIdx.x] = m; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]); __syncthreads(); }
  m = red[0]; __syncthreads(); float sum = 0.0F;
  for (int k = threadIdx.x; k < visible; k += blockDim.x) sum += expf(__bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)) - m);
  red[threadIdx.x] = sum; __syncthreads();
  for (int o = blockDim.x / 2; o; o >>= 1) { if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o]; __syncthreads(); }
  const float inv = 1.0F / red[0];
  for (int k = threadIdx.x; k < keys; k += blockDim.x)
    s[k] = k < visible ? __float2bfloat16_rn(expf(__bfloat162float(__float2bfloat16_rn(__bfloat162float(s[k]) * scale)) - m) * inv) : __float2bfloat16_rn(0.0F); }
// om[batch][local_row * G + g][D] -> output[row][head][D]
__global__ void scatter_o(const __nv_bfloat16* om, __nv_bfloat16* output, int rows_per_seq, int nseq) {
  const size_t n = size_t(nseq) * rows_per_seq * QH * D;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    const int d = i % D; const size_t rh = i / D; const int head = rh % QH; const size_t row = rh / QH;
    const int seq = row / rows_per_seq, local = row % rows_per_seq, kvh = head / G, g = head % G;
    output[i] = om[((size_t(seq) * KVH + kvh) * rows_per_seq * G + size_t(local) * G + g) * D + d]; } }

static std::vector<__nv_bfloat16> run(cublasHandle_t h, int R, int P0) {
  const int S = 1, keys = P0 + R, pages = (keys + 15) / 16, B = S * KVH, R8 = R * G;
  const size_t pool = size_t(pages) * KVH * 16 * D, qn = size_t(R) * QH * D;
  __nv_bfloat16 *q, *kp, *vp, *qm, *km, *vm, *sc, *om, *out; uint32_t* ids;
  cudaMalloc(&q, qn * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&qm, qn * 2); cudaMalloc(&om, qn * 2); cudaMalloc(&out, qn * 2);
  cudaMalloc(&km, size_t(B) * keys * D * 2); cudaMalloc(&vm, size_t(B) * keys * D * 2); cudaMalloc(&sc, size_t(B) * R8 * keys * 2); cudaMalloc(&ids, size_t(pages) * 4);
  // Values depend only on the absolute (position, head, dim) so every chunking sees the same sequence.
  std::vector<__nv_bfloat16> hk(pool), hv(pool), hq(qn);
  for (size_t i = 0; i < pool; ++i) { hk[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F); hv[i] = __float2bfloat16(float((i * 40503u + 17) % 1999) / 999.0F - 1.0F); }
  for (size_t i = 0; i < qn; ++i) { const size_t absolute = size_t(P0) * QH * D + i; hq[i] = __float2bfloat16(float((absolute * 2246822519u) % 2003) / 500.0F - 2.0F); }
  cudaMemcpy(kp, hk.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, hv.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, hq.data(), qn * 2, cudaMemcpyHostToDevice);
  std::vector<uint32_t> hi(pages); for (int i = 0; i < pages; ++i) hi[i] = i; cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
  const float one = 1.0F, zero = 0.0F, scale = 1.0F / std::sqrt(float(D));
  auto blocks = [](size_t n) { return unsigned((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256); };
  int sm_threads = 32; while (sm_threads < keys && sm_threads < 256) sm_threads <<= 1;
  gather_q<<<blocks(qn), 256>>>(q, qm, R, S); gather_kv<<<blocks(size_t(B) * keys * D), 256>>>(kp, vp, ids, km, vm, keys, pages, S);
  cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, R8, D, &one, km, CUDA_R_16BF, D, (long long)keys * D, qm, CUDA_R_16BF, D, (long long)R8 * D, &zero, sc, CUDA_R_16BF, keys, (long long)R8 * keys, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  causal_softmax<<<dim3(R8, B), sm_threads>>>(sc, keys, R, P0, scale);
  cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, R8, keys, &one, vm, CUDA_R_16BF, D, (long long)keys * D, sc, CUDA_R_16BF, keys, (long long)R8 * keys, &zero, om, CUDA_R_16BF, D, (long long)R8 * D, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  scatter_o<<<blocks(qn), 256>>>(om, out, R, S); cudaDeviceSynchronize();
  std::vector<__nv_bfloat16> ho(qn); cudaMemcpy(ho.data(), out, qn * 2, cudaMemcpyDeviceToHost);
  cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(qm); cudaFree(km); cudaFree(vm); cudaFree(sc); cudaFree(om); cudaFree(out); cudaFree(ids); return ho; }
static void compare(const char* label, const std::vector<__nv_bfloat16>& whole, size_t offset_rows, const std::vector<__nv_bfloat16>& part) {
  size_t diff = 0; double maxrel = 0; const size_t off = offset_rows * QH * D;
  for (size_t i = 0; i < part.size(); ++i) { const float a = __bfloat162float(whole[off + i]), b = __bfloat162float(part[i]);
    if (a != b) { ++diff; const double r = std::fabs(double(a) - b) / (std::fabs(double(a)) + 1e-9); if (r > maxrel) maxrel = r; } }
  std::printf("%s: elements %zu differing %zu (%.3f%%) max relative difference %.5f\n", label, part.size(), diff, 100.0 * diff / part.size(), maxrel); }
int main() {
  cublasHandle_t h; cublasCreate(&h);
  const auto whole = run(h, 128, 200), again = run(h, 128, 200);
  compare("same chunk run twice (rows 200..327)", whole, 0, again);
  compare("one 128-row chunk vs first 64-row chunk (rows 200..263)", whole, 0, run(h, 64, 200));
  compare("one 128-row chunk vs second 64-row chunk (rows 264..327)", whole, 64, run(h, 64, 264));
  compare("one 128-row chunk vs 32-row chunk (rows 296..327)", whole, 96, run(h, 32, 296));
  compare("one 128-row chunk vs single row (row 327)", whole, 127, run(h, 1, 327));
  cublasDestroy(h); return 0; }
