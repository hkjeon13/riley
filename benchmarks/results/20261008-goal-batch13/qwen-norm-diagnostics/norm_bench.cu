// Diagnostic microbenchmark: BF16 RMSNorm and residual+RMSNorm kernels (hidden 2048, 256 threads per row).
// "old" copies the serving kernels; "reg" keeps every arithmetic statement and the reduction tree but loads each
// thread's columns into registers once (no second load pass); "regshfl" also folds the last five tree levels with
// warp shuffles in the same pairing. Outputs must be bit-identical to "old".
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
static double now() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int T = 256, MAXC = 8;
__device__ __forceinline__ float rs(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__global__ void norm_old(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint64_t row_count, uint64_t hidden_size, float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) { const uint64_t base = row * hidden_size; float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float value = __bfloat162float(input[base + column]); sum += value * value; }
    partial_sums[threadIdx.x] = sum; __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset != 0; offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); }
    const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float normalized = __bfloat162float(input[base + column]) * inverse_rms; const float nw = rs(normalized);
      output[base + column] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); }
    __syncthreads(); } }
template <bool SHFL>
__global__ void norm_new(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint64_t row_count, uint64_t hidden_size, float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) { const uint64_t base = row * hidden_size; float values[MAXC]; uint32_t count = 0;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) values[count++] = __bfloat162float(input[base + column]);
    float sum = 0.0F;
    for (uint32_t i = 0; i < count; ++i) { const float value = values[i]; sum += value * value; }
    partial_sums[threadIdx.x] = sum; __syncthreads();
    uint32_t offset = blockDim.x / 2;
    for (; offset != 0 && (!SHFL || offset >= 32); offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); }
    if (SHFL) { if (threadIdx.x < 32) { float v = partial_sums[threadIdx.x]; for (; offset != 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset); if (threadIdx.x == 0) partial_sums[0] = v; } __syncthreads(); }
    const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
    uint32_t i = 0;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x, ++i) { const float normalized = values[i] * inverse_rms; const float nw = rs(normalized);
      output[base + column] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); }
    __syncthreads(); } }
__global__ void res_old(const __nv_bfloat16* left, const __nv_bfloat16* right, const __nv_bfloat16* weight, __nv_bfloat16* residual_output, __nv_bfloat16* normalized_output, uint64_t row_count, uint64_t hidden_size, float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) { const uint64_t base = row * hidden_size; float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const uint64_t index = base + column;
      const float residual = rs(__bfloat162float(left[index]) + __bfloat162float(right[index])); residual_output[index] = __float2bfloat16_rn(residual); sum += residual * residual; }
    partial_sums[threadIdx.x] = sum; __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset != 0; offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); }
    const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const uint64_t index = base + column; const float normalized = __bfloat162float(residual_output[index]) * inverse_rms; const float nw = rs(normalized);
      normalized_output[index] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); }
    __syncthreads(); } }
template <bool SHFL>
__global__ void res_new(const __nv_bfloat16* left, const __nv_bfloat16* right, const __nv_bfloat16* weight, __nv_bfloat16* residual_output, __nv_bfloat16* normalized_output, uint64_t row_count, uint64_t hidden_size, float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) { const uint64_t base = row * hidden_size; float l[MAXC], r[MAXC], values[MAXC]; uint32_t count = 0;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x, ++count) { l[count] = __bfloat162float(left[base + column]); r[count] = __bfloat162float(right[base + column]); }
    float sum = 0.0F; uint32_t i = 0;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x, ++i) { const float residual = rs(l[i] + r[i]); residual_output[base + column] = __float2bfloat16_rn(residual); values[i] = residual; sum += residual * residual; }
    partial_sums[threadIdx.x] = sum; __syncthreads();
    uint32_t offset = blockDim.x / 2;
    for (; offset != 0 && (!SHFL || offset >= 32); offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); }
    if (SHFL) { if (threadIdx.x < 32) { float v = partial_sums[threadIdx.x]; for (; offset != 0; offset /= 2) v += __shfl_down_sync(0xffffffffu, v, offset); if (threadIdx.x == 0) partial_sums[0] = v; } __syncthreads(); }
    const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
    i = 0;
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x, ++i) { const float normalized = values[i] * inverse_rms; const float nw = rs(normalized);
      normalized_output[base + column] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); }
    __syncthreads(); } }
int main() {
  const int H = 2048, MAXR = 256; const size_t n = size_t(MAXR) * H;
  __nv_bfloat16 *a, *b, *w, *o1, *o2, *r1, *r2; cudaMalloc(&a, n * 2); cudaMalloc(&b, n * 2); cudaMalloc(&w, H * 2); cudaMalloc(&o1, n * 2); cudaMalloc(&o2, n * 2); cudaMalloc(&r1, n * 2); cudaMalloc(&r2, n * 2);
  { std::vector<uint16_t> ha(n), hb(n), hw(H); uint64_t s = 0x9e3779b97f4a7c15ull; auto next = [&](uint32_t eb, uint32_t span) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; const uint32_t bits = uint32_t(s >> 24); return uint16_t(((bits >> 31) << 15) | ((eb + (bits >> 8) % span) << 7) | (bits & 0x7f)); };
    for (auto& v : ha) v = next(0x78, 12); for (auto& v : hb) v = next(0x78, 12); for (auto& v : hw) v = next(0x7c, 4);
    cudaMemcpy(a, ha.data(), n * 2, cudaMemcpyHostToDevice); cudaMemcpy(b, hb.data(), n * 2, cudaMemcpyHostToDevice); cudaMemcpy(w, hw.data(), H * 2, cudaMemcpyHostToDevice); }
  auto timeit = [&](auto f) { for (int i = 0; i < 200; ++i) f(); cudaDeviceSynchronize(); double t0 = now(); for (int i = 0; i < 4000; ++i) f(); cudaDeviceSynchronize(); return (now() - t0) / 4000; };
  auto diff = [&](const __nv_bfloat16* x, const __nv_bfloat16* y, int rows) { std::vector<uint16_t> hx(size_t(rows) * H), hy(size_t(rows) * H); cudaMemcpy(hx.data(), x, hx.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(hy.data(), y, hy.size() * 2, cudaMemcpyDeviceToHost); long d = 0; for (size_t i = 0; i < hx.size(); ++i) d += hx[i] != hy[i]; return d; };
  const float eps = 1e-6F; const size_t sh = T * sizeof(float);
  for (int rows : {1, 8, 32, 128, 256}) {
    const double t_old = timeit([&] { norm_old<<<rows, T, sh>>>(a, w, o1, rows, H, eps); });
    long d[2]; double t_new[2];
    t_new[0] = timeit([&] { norm_new<false><<<rows, T, sh>>>(a, w, o2, rows, H, eps); }); cudaDeviceSynchronize(); norm_old<<<rows, T, sh>>>(a, w, o1, rows, H, eps); norm_new<false><<<rows, T, sh>>>(a, w, o2, rows, H, eps); cudaDeviceSynchronize(); d[0] = diff(o1, o2, rows);
    t_new[1] = timeit([&] { norm_new<true><<<rows, T, sh>>>(a, w, o2, rows, H, eps); }); norm_new<true><<<rows, T, sh>>>(a, w, o2, rows, H, eps); cudaDeviceSynchronize(); d[1] = diff(o1, o2, rows);
    const double r_old = timeit([&] { res_old<<<rows, T, sh>>>(a, b, w, r1, o1, rows, H, eps); });
    long e[2][2]; double r_new[2];
    r_new[0] = timeit([&] { res_new<false><<<rows, T, sh>>>(a, b, w, r2, o2, rows, H, eps); }); res_old<<<rows, T, sh>>>(a, b, w, r1, o1, rows, H, eps); res_new<false><<<rows, T, sh>>>(a, b, w, r2, o2, rows, H, eps); cudaDeviceSynchronize(); e[0][0] = diff(r1, r2, rows); e[0][1] = diff(o1, o2, rows);
    r_new[1] = timeit([&] { res_new<true><<<rows, T, sh>>>(a, b, w, r2, o2, rows, H, eps); }); res_new<true><<<rows, T, sh>>>(a, b, w, r2, o2, rows, H, eps); cudaDeviceSynchronize(); e[1][0] = diff(r1, r2, rows); e[1][1] = diff(o1, o2, rows);
    std::printf("rows=%3d rms_norm old %5.2f us | reg %5.2f (diff %ld) | reg+shfl %5.2f (diff %ld) || residual_rms_norm old %5.2f | reg %5.2f (diff %ld/%ld) | reg+shfl %5.2f (diff %ld/%ld)\n", rows, t_old, t_new[0], d[0], t_new[1], d[1], r_old, r_new[0], e[0][0], e[0][1], r_new[1], e[1][0], e[1][1]); }
  return 0; }
