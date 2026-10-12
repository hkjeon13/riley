// Diagnostic: where does the ~3.8 us of the 256-thread RMSNorm kernel go? Variants are timed on the GPU side (nsys).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <vector>
__device__ __forceinline__ float rs(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
#define BODY(SHARED_DECL) \
  SHARED_DECL; \
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) { const uint64_t base = row * hidden_size; float sum = 0.0F; \
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float value = __bfloat162float(input[base + column]); sum += value * value; } \
    partial_sums[threadIdx.x] = sum; __syncthreads(); \
    for (uint32_t offset = blockDim.x / 2; offset != 0; offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); } \
    const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon); \
    for (uint64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float normalized = __bfloat162float(input[base + column]) * inverse_rms; const float nw = rs(normalized); \
      output[base + column] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); } \
    __syncthreads(); }
__global__ void v_dynamic(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint64_t row_count, uint64_t hidden_size, float epsilon) { BODY(extern __shared__ float partial_sums[]) }
__global__ void v_static(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint64_t row_count, uint64_t hidden_size, float epsilon) { BODY(__shared__ float partial_sums[256]) }
// same arithmetic, 32-bit indices and a single row per block (no row loop, no trailing barrier)
__global__ void v_lean(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint32_t row_count, uint32_t hidden_size, float epsilon) {
  __shared__ float partial_sums[256]; const uint32_t row = blockIdx.x; if (row >= row_count) return; const __nv_bfloat16* in = input + size_t(row) * hidden_size; __nv_bfloat16* out = output + size_t(row) * hidden_size;
  float sum = 0.0F;
  for (uint32_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float value = __bfloat162float(in[column]); sum += value * value; }
  partial_sums[threadIdx.x] = sum; __syncthreads();
  for (uint32_t offset = blockDim.x / 2; offset != 0; offset /= 2) { if (threadIdx.x < offset) partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset]; __syncthreads(); }
  const float inverse_rms = rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
  for (uint32_t column = threadIdx.x; column < hidden_size; column += blockDim.x) { const float normalized = __bfloat162float(in[column]) * inverse_rms; const float nw = rs(normalized);
    out[column] = __float2bfloat16_rn(nw * __bfloat162float(weight[column])); } }
// reference points: elementwise kernel over the same data, and a barrier-only kernel
__global__ void v_elementwise(const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output, uint32_t row_count, uint32_t hidden_size) {
  const uint32_t n = row_count * hidden_size; for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) output[i] = __float2bfloat16_rn(__bfloat162float(input[i]) * __bfloat162float(weight[i % hidden_size])); }
__global__ void v_barriers(__nv_bfloat16* output, int count) { __shared__ float s[256]; s[threadIdx.x] = threadIdx.x; for (int i = 0; i < count; ++i) { __syncthreads(); if (threadIdx.x < 128) s[threadIdx.x] += s[threadIdx.x + 128]; } __syncthreads(); if (threadIdx.x == 0) output[blockIdx.x] = __float2bfloat16_rn(s[0]); }
int main() {
  const int H = 2048, MAXR = 256; const size_t n = size_t(MAXR) * H; __nv_bfloat16 *a, *w, *o1, *o2; cudaMalloc(&a, n * 2); cudaMalloc(&w, H * 2); cudaMalloc(&o1, n * 2); cudaMalloc(&o2, n * 2);
  { std::vector<uint16_t> ha(n), hw(H); uint64_t s = 0x9e3779b97f4a7c15ull; auto next = [&](uint32_t eb, uint32_t span) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; const uint32_t bits = uint32_t(s >> 24); return uint16_t(((bits >> 31) << 15) | ((eb + (bits >> 8) % span) << 7) | (bits & 0x7f)); };
    for (auto& v : ha) v = next(0x78, 12); for (auto& v : hw) v = next(0x7c, 4); cudaMemcpy(a, ha.data(), n * 2, cudaMemcpyHostToDevice); cudaMemcpy(w, hw.data(), H * 2, cudaMemcpyHostToDevice); }
  const float eps = 1e-6F;
  for (int rows : {1, 32}) { for (int i = 0; i < 3000; ++i) {
      v_dynamic<<<rows, 256, 256 * 4>>>(a, w, o1, rows, H, eps); v_static<<<rows, 256>>>(a, w, o2, rows, H, eps); v_lean<<<rows, 256>>>(a, w, o2, rows, H, eps);
      v_elementwise<<<(rows * H + 255) / 256, 256>>>(a, w, o2, rows, H); v_barriers<<<rows, 256>>>(o2, 1); v_barriers<<<rows, 256>>>(o2, 9); cudaDeviceSynchronize(); }
    v_dynamic<<<rows, 256, 256 * 4>>>(a, w, o1, rows, H, eps); v_lean<<<rows, 256>>>(a, w, o2, rows, H, eps); cudaDeviceSynchronize();
    std::vector<uint16_t> x(size_t(rows) * H), y(size_t(rows) * H); cudaMemcpy(x.data(), o1, x.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(y.data(), o2, y.size() * 2, cudaMemcpyDeviceToHost); long d = 0; for (size_t i = 0; i < x.size(); ++i) d += x[i] != y[i];
    std::printf("rows=%d lean vs dynamic differing %ld\n", rows, d); }
  return 0; }
