// Diagnostic microbenchmark 7: prefill-row attention through batched tensor-core GEMMs (Qwen D128, QH=16, KVH=2).
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
int main() {
  cublasHandle_t h; cublasCreate(&h);
  // {sequences, rows per sequence, first position of the chunk}; comparison numbers from attn_bench6.log (per-row CTA producer only).
  const int cfg[][3] = {{2, 112, 18}, {4, 56, 100}, {1, 128, 200}, {8, 28, 4}, {2, 32, 300}, {1, 256, 142}};
  const double ref[] = {36.7, 58.6, 67.7, 14.4, 42.4, -1};
  int ci = 0;
  for (auto& c : cfg) { const int S = c[0], R = c[1], P0 = c[2], keys = P0 + R, pages = (keys + 15) / 16, B = S * KVH, R8 = R * G;
    const size_t pool = size_t(S) * pages * KVH * 16 * D, qn = size_t(S) * R * QH * D;
    __nv_bfloat16 *q, *kp, *vp, *qm, *km, *vm, *sc, *om, *out; uint32_t* ids;
    cudaMalloc(&q, qn * 2); cudaMalloc(&kp, pool * 2); cudaMalloc(&vp, pool * 2); cudaMalloc(&qm, qn * 2); cudaMalloc(&om, qn * 2); cudaMalloc(&out, qn * 2);
    cudaMalloc(&km, size_t(B) * keys * D * 2); cudaMalloc(&vm, size_t(B) * keys * D * 2); cudaMalloc(&sc, size_t(B) * R8 * keys * 2); cudaMalloc(&ids, size_t(S) * pages * 4);
    std::vector<__nv_bfloat16> hb(pool > qn ? pool : qn); for (size_t i = 0; i < hb.size(); ++i) hb[i] = __float2bfloat16(float((i * 2654435761u) % 2001) / 1000.0F - 1.0F);
    cudaMemcpy(kp, hb.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(vp, hb.data(), pool * 2, cudaMemcpyHostToDevice); cudaMemcpy(q, hb.data(), qn * 2, cudaMemcpyHostToDevice);
    std::vector<uint32_t> hi(size_t(S) * pages); for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint32_t((i * 7 + 3) % hi.size());
    { std::vector<char> seen(hi.size()); bool ok = true; for (auto v : hi) { if (seen[v]) ok = false; seen[v] = 1; } if (!ok) for (size_t i = 0; i < hi.size(); ++i) hi[i] = uint32_t(i); }
    cudaMemcpy(ids, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice);
    const float one = 1.0F, zero = 0.0F, scale = 1.0F / std::sqrt(float(D));
    auto blocks = [](size_t n) { return unsigned((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256); };
    int sm_threads = 32; while (sm_threads < keys && sm_threads < 256) sm_threads <<= 1;
    auto gather = [&] { gather_q<<<blocks(qn), 256>>>(q, qm, R, S); gather_kv<<<blocks(size_t(B) * keys * D), 256>>>(kp, vp, ids, km, vm, keys, pages, S); };
    auto qk = [&] { cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, keys, R8, D, &one, km, CUDA_R_16BF, D, (long long)keys * D, qm, CUDA_R_16BF, D, (long long)R8 * D, &zero, sc, CUDA_R_16BF, keys, (long long)R8 * keys, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
    auto sm = [&] { causal_softmax<<<dim3(R8, B), sm_threads>>>(sc, keys, R, P0, scale); };
    auto pv = [&] { cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, R8, keys, &one, vm, CUDA_R_16BF, D, (long long)keys * D, sc, CUDA_R_16BF, keys, (long long)R8 * keys, &zero, om, CUDA_R_16BF, D, (long long)R8 * D, B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
    auto sca = [&] { scatter_o<<<blocks(qn), 256>>>(om, out, R, S); };
    auto all = [&] { gather(); qk(); sm(); pv(); sca(); };
    auto timeit = [&](auto f) { for (int i = 0; i < 30; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 300; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 300; };
    all(); cudaDeviceSynchronize(); const cudaError_t e = cudaGetLastError();
    const double tg = timeit(gather), tq = timeit(qk), ts = timeit(sm), tp = timeit(pv), tc = timeit(sca), ta = timeit(all);
    std::vector<__nv_bfloat16> ho(qn); cudaMemcpy(ho.data(), out, qn * 2, cudaMemcpyDeviceToHost); double sum = 0; int nan = 0; for (auto v : ho) { const float f = __bfloat162float(v); if (std::isnan(f)) ++nan; else sum += f; }
    std::printf("prefill rows=%3d over %d sequences (positions %d..%d, keys %d): gather %5.1f qk %5.1f softmax %5.1f pv %5.1f scatter %5.1f | all %6.1f us | per-row paged producer (attn_bench6) %5.1f us | cuda %d nan %d checksum %.3f\n",
                S * R, S, P0, P0 + R - 1, keys, tg, tq, ts, tp, tc, ta, ref[ci], int(e), nan, sum);
    ++ci; cudaFree(q); cudaFree(kp); cudaFree(vp); cudaFree(qm); cudaFree(km); cudaFree(vm); cudaFree(sc); cudaFree(om); cudaFree(out); cudaFree(ids); }
  cublasDestroy(h); return 0; }
