#ifndef RILEY_CUDA_ATTENTION_ONLINE_HPP_
#define RILEY_CUDA_ATTENTION_ONLINE_HPP_

#include <cuda_runtime_api.h>

#include <cstdint>

namespace riley_cuda_attention_online {

// The public C ABI validates every argument before calling this internal
// allocation-free launcher. Full causal attention uses three score passes to
// reproduce the staged-BF16 materialized reduction order without writing an
// HBM score matrix. Causal-local attention retains the two-pass online
// normalizer, including its zero-width all-masked behavior. causal_local
// distinguishes that zero-width local window from full causal attention,
// whose window field is also zero.
cudaError_t launch_bf16_gqa_prefill(
    const void* query, const void* key, const void* value, void* output,
    uint64_t batch_count, uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale, bool causal_local,
    uint64_t local_window_size, cudaStream_t stream) noexcept;

}  // namespace riley_cuda_attention_online

#endif  // RILEY_CUDA_ATTENTION_ONLINE_HPP_
