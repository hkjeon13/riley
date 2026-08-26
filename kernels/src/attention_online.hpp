#ifndef RUSTINFER_CUDA_ATTENTION_ONLINE_HPP_
#define RUSTINFER_CUDA_ATTENTION_ONLINE_HPP_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace rustinfer_cuda_attention_online {

// The public C ABI validates every argument before calling this internal
// allocation-free, two-score-pass launcher. The first pass retains the online
// F32 maximum/denominator; the second stages normalized probabilities to BF16
// before logical-key-order F32 AV. causal_local distinguishes a zero-width
// all-masked local window from full causal attention, whose window is also zero.
cudaError_t launch_bf16_gqa_prefill(
    const void* query, const void* key, const void* value, void* output,
    uint64_t batch_count, uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale, bool causal_local,
    uint64_t local_window_size, cudaStream_t stream) noexcept;

}  // namespace rustinfer_cuda_attention_online

#endif  // RUSTINFER_CUDA_ATTENTION_ONLINE_HPP_
