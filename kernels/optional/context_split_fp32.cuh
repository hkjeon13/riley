#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

// Experimental numerical control. BF16 KV, FP32 probabilities and partials.
// Not the legacy exact profile. Scores use the existing QK producer/layout.
namespace riley_split_fp32 {
struct Partial { float maximum, denominator, value[64]; };
constexpr unsigned max_splits = 32;
constexpr unsigned max_rows = 32;
constexpr unsigned heads = 9;
constexpr size_t partial_count = max_rows * heads * max_splits;

__global__ void partials(const float* scores, const __nv_bfloat16* values,
                        Partial* scratch, const uint32_t* shape,
                        const uint32_t* pages, const uint32_t* live,
                        unsigned splits, unsigned capacity) {
    unsigned row = blockIdx.y / heads, head = blockIdx.y % heads;
    unsigned active = *live, split = blockIdx.x, lane = threadIdx.x;
    if (active > max_rows || row >= active) return;
    unsigned position = shape[row * 416 + 1];
    if (position >= capacity) return;
    unsigned count = position + 1;
    unsigned width = (capacity + splits - 1) / splits;
    unsigned begin = split * width, end = min(begin + width, count);
    Partial& part = scratch[(row * heads + head) * max_splits + split];
    if (begin >= count) {
        if (lane == 0) { part.maximum = -CUDART_INF_F; part.denominator = 0; }
        part.value[lane] = 0; part.value[lane + 32] = 0;
        return;
    }
    const float* input = scores + (row * heads + head) * 4096;
    float maximum = -CUDART_INF_F;
    for (unsigned t = begin + lane; t < end; t += 32) maximum = fmaxf(maximum, input[t]);
    for (unsigned delta = 16; delta; delta >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    float denominator = 0, low = 0, high = 0;
    // Warp broadcast avoids rounding probabilities to BF16 before P*V.
    // This intentionally simple control exposes the scalar arithmetic cost.
    for (unsigned t = begin; t < end; ++t) {
        float probability = lane == 0 ? exp2f((input[t] - maximum) * 1.4426950408889634F) : 0;
        probability = __shfl_sync(0xffffffff, probability, 0);
        denominator += probability;
        unsigned page = pages[row * 416 + t / 16];
        unsigned offset = ((page * 3 + head / 3) * 16 + t % 16) * 64;
        low = fmaf(probability, __bfloat162float(values[offset + lane]), low);
        high = fmaf(probability, __bfloat162float(values[offset + lane + 32]), high);
    }
    if (lane == 0) { part.maximum = maximum; part.denominator = denominator; }
    part.value[lane] = low; part.value[lane + 32] = high;
}

__global__ void merge(const Partial* scratch, __nv_bfloat16* output,
                      const uint32_t* shape, const uint32_t* live,
                      unsigned splits, unsigned capacity) {
    unsigned row = blockIdx.x / heads, head = blockIdx.x % heads, lane = threadIdx.x;
    unsigned active = *live;
    if (active > max_rows || row >= active || shape[row * 416 + 1] >= capacity) return;
    const Partial* parts = scratch + (row * heads + head) * max_splits;
    float maximum = -CUDART_INF_F;
    for (unsigned s = 0; s < splits; ++s)
        if (parts[s].denominator > 0) maximum = fmaxf(maximum, parts[s].maximum);
    float denominator = 0, low = 0, high = 0;
    for (unsigned s = 0; s < splits; ++s) {
        if (parts[s].denominator == 0) continue;
        float scale = exp2f((parts[s].maximum - maximum) * 1.4426950408889634F);
        denominator = fmaf(parts[s].denominator, scale, denominator);
        low = fmaf(parts[s].value[lane], scale, low);
        high = fmaf(parts[s].value[lane + 32], scale, high);
    }
    output[row * 576 + head * 64 + lane] = __float2bfloat16_rn(low / denominator);
    output[row * 576 + head * 64 + lane + 32] = __float2bfloat16_rn(high / denominator);
}

// Caller owns all extents/page validity and keeps scratch alive through replay.
// Host launch extents fail before enqueue; device invalid rows remain untouched.
inline cudaError_t enqueue(cudaStream_t stream, const float* scores,
                           const __nv_bfloat16* values, Partial* scratch,
                           __nv_bfloat16* output, const uint32_t* shape,
                           const uint32_t* pages, const uint32_t* live,
                           unsigned splits, unsigned capacity) {
    if (!scores || !values || !scratch || !output || !shape || !pages || !live ||
        splits < 1 || splits > max_splits || capacity < 1 || capacity > 4096)
        return cudaErrorInvalidValue;
    partials<<<dim3(splits, max_rows * heads), 32, 0, stream>>>(
        scores, values, scratch, shape, pages, live, splits, capacity);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    merge<<<max_rows * heads, 32, 0, stream>>>(scratch, output, shape, live, splits, capacity);
    return cudaGetLastError();
}
}
