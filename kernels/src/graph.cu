#include "ffi_internal.hpp"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <new>

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_thread_graph_capture_owner;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::drain_capture_deferred_closes;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::native_thread_token;
using riley_cuda_internal::next_graph_capture_id;
using riley_cuda_internal::next_graph_exec_id;
using riley_cuda_internal::release_child;
using riley_cuda_internal::release_capture_domain_capture;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::same_context;
using riley_cuda_internal::set_error;
using riley_cuda_internal::thread_graph_capture_is_owner;
using riley_cuda_internal::thread_has_active_command_batch;
using riley_cuda_internal::thread_has_active_graph_capture;
using riley_cuda_internal::try_publish_thread_graph_capture;
using riley_cuda_internal::try_acquire_exclusive_use;
using riley_cuda_internal::try_begin_capture_domain;
using riley_cuda_internal::validation_error;

constexpr const char* kBeginOperation = "begin CUDA Graph capture";
constexpr const char* kAbortOperation = "abort CUDA Graph capture";
constexpr const char* kBeginFillOperation = "begin CUDA Graph fill capture";
constexpr const char* kEnqueueFillOperation = "enqueue CUDA Graph fill";
constexpr const char* kBeginH2DOperation = "begin CUDA Graph H2D capture";
constexpr const char* kEnqueueH2DOperation = "enqueue CUDA Graph H2D";
constexpr const char* kBeginSiluBf16Operation =
    "begin CUDA Graph BF16 SiLU capture";
constexpr const char* kEnqueueSiluBf16Operation =
    "enqueue CUDA Graph BF16 SiLU";
constexpr const char* kBeginGatedMultiplyBf16Operation =
    "begin CUDA Graph BF16 gated-multiply capture";
constexpr const char* kEnqueueGatedMultiplyBf16Operation =
    "enqueue CUDA Graph BF16 gated multiply";
constexpr const char* kBeginResidualAddBf16Operation =
    "begin CUDA Graph BF16 residual-add capture";
constexpr const char* kEnqueueResidualAddBf16Operation =
    "enqueue CUDA Graph BF16 residual add";
constexpr const char* kBeginCanonicalRmsNormBf16Operation =
    "begin CUDA Graph canonical BF16 RMSNorm capture";
constexpr const char* kEnqueueCanonicalRmsNormBf16Operation =
    "enqueue CUDA Graph canonical BF16 RMSNorm";
constexpr const char* kBeginBf16ArgmaxOperation =
    "begin CUDA Graph deterministic BF16 argmax capture";
constexpr const char* kEnqueueBf16ArgmaxOperation =
    "enqueue CUDA Graph deterministic BF16 argmax";
constexpr const char* kQueryCaptureCapabilityOperation =
    "query CUDA Graph capture capability";
constexpr const char* kStageH2DOperation = "stage CUDA Graph H2D source";
constexpr const char* kEndOperation = "end CUDA Graph capture";
constexpr const char* kInstantiateOperation = "instantiate CUDA Graph";
constexpr const char* kLaunchOperation = "launch CUDA Graph exec";
constexpr const char* kCompleteOperation = "complete CUDA Graph launch";
constexpr const char* kCloseGraphOperation = "close CUDA Graph";
constexpr const char* kCloseExecOperation = "close CUDA Graph exec";
constexpr uint32_t kGraphFillThreads = 256;
constexpr uint64_t kMaximumGraphFillGridX = static_cast<uint64_t>(INT_MAX);
constexpr uint32_t kGraphSiluThreads = 256;
constexpr uint32_t kMaximumGraphSiluBlocks = 65535;
constexpr uint32_t kGraphCanonicalRmsNormThreads = 256;
constexpr uint32_t kMaximumGraphCanonicalRmsNormBlocks = 65535;
constexpr uint32_t kGraphBf16ArgmaxThreads = 256;
constexpr uint32_t kMaximumGraphBf16ArgmaxBlocks = 65535;
constexpr uint32_t kGraphBf16ArgmaxWarpSize = 32;
constexpr uint32_t kGraphBf16ArgmaxFullWarpMask = 0xffffffffU;
constexpr const char* kBeginBf16RowGatherOperation =
    "begin CUDA Graph BF16 row-gather capture";
constexpr const char* kEnqueueBf16RowGatherOperation =
    "enqueue CUDA Graph BF16 row gather";
constexpr uint32_t kGraphBf16RowGatherThreads = 256;
constexpr uint32_t kMaximumGraphBf16RowGatherBlocks = 65535;
constexpr const char* kBeginBf16RowGatherArgmaxOperation =
    "begin CUDA Graph BF16 row-gather argmax capture";
constexpr const char* kEnqueueBf16RowGatherArgmaxOperation =
    "enqueue CUDA Graph BF16 row-gather argmax";
constexpr const char* kBeginBf16RowGatherArgmaxD2HOperation =
    "begin CUDA Graph BF16 row-gather argmax D2H capture";
constexpr const char* kEnqueueBf16RowGatherArgmaxD2HOperation =
    "enqueue CUDA Graph BF16 row-gather argmax D2H";
constexpr const char* kReadBf16RowGatherArgmaxD2HResultsOperation =
    "read completed CUDA Graph BF16 row-gather argmax D2H results";
constexpr const char* kBeginIndexedRopeBf16Operation =
    "begin CUDA Graph BF16 indexed-RoPE capture";
constexpr const char* kEnqueueIndexedRopeBf16Operation =
    "enqueue CUDA Graph BF16 indexed-RoPE";
constexpr const char* kBeginRaggedPagedKvCacheWriteBf16Operation =
    "begin CUDA Graph BF16 ragged paged-KV write capture";
constexpr const char* kEnqueueRaggedPagedKvCacheWriteBf16Operation =
    "enqueue CUDA Graph BF16 ragged paged-KV write";
constexpr const char* kBeginGroupedRaggedPagedAttentionBf16Operation =
    "begin CUDA Graph grouped BF16 ragged paged-attention capture";
constexpr const char* kEnqueueGroupedRaggedPagedAttentionBf16Operation =
    "enqueue CUDA Graph grouped BF16 ragged paged attention";
constexpr const char* kBeginBf16EmbeddingStatusD2HOperation =
    "begin CUDA Graph BF16 embedding validation-status D2H capture";
constexpr const char* kEnqueueBf16EmbeddingStatusD2HOperation =
    "enqueue CUDA Graph BF16 embedding validation-status D2H";
constexpr const char* kReadBf16EmbeddingStatusD2HReportOperation =
    "read completed CUDA Graph BF16 embedding validation-status D2H report";
constexpr const char* kBeginCanonicalGemmBf16Operation =
    "begin CUDA Graph canonical cuBLASLt BF16 GEMM capture";
constexpr const char* kEnqueueCanonicalGemmBf16Operation =
    "enqueue CUDA Graph canonical cuBLASLt BF16 GEMM";
// A failed node launch during a multi-node capture cannot safely be retried:
// CUDA may have invalidated the capture after accepting a prefix. Keep this
// terminal marker in the capture-only enqueue state so abort remains the sole
// admissible transition while the exact fixed-address leases stay held.
constexpr uint32_t kBf16RowGatherArgmaxEnqueueTerminal = UINT32_MAX;
constexpr uint32_t kBf16RowGatherArgmaxD2HEnqueueTerminal = UINT32_MAX;
constexpr uint32_t kIndexedRopeBf16EnqueueTerminal = UINT32_MAX;
constexpr uint32_t kRaggedPagedKvCacheWriteBf16EnqueueTerminal = UINT32_MAX;
constexpr uint32_t kGroupedRaggedPagedAttentionBf16EnqueueTerminal =
    UINT32_MAX;
constexpr uint32_t kBf16EmbeddingStatusD2HEnqueueTerminal = UINT32_MAX;
constexpr uint32_t kCanonicalGemmBf16EnqueueTerminal = UINT32_MAX;
constexpr uint32_t kGraphBf16EmbeddingThreads = 256;
constexpr uint32_t kMaximumGraphBf16EmbeddingBlocks = 65535;
constexpr uint32_t kGraphRaggedAttentionWarpSize = 32;
constexpr uint32_t kGraphRaggedAttentionFullWarpMask = 0xffffffffU;
constexpr uint32_t kGraphRaggedAttentionMaximumWarpsPerBlock = 16;
constexpr uint32_t kGraphRaggedAttentionThreads =
    kGraphRaggedAttentionWarpSize *
    kGraphRaggedAttentionMaximumWarpsPerBlock;
constexpr uint64_t kGraphRaggedAttentionHeadSize = 64;
constexpr uint32_t kGraphRaggedAttentionSharedTileTokens =
    RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
constexpr uint64_t kGraphMaximumGridX = 2147483647;
constexpr uint64_t kGraphMaximumGridYOrZ = 65535;

// C05-20 reuses C05-16's existing five-resource ownership ledger but never
// its row-gather/argmax arithmetic. Keeping this discriminator centralized
// makes generic capture/graph/exec lifecycle transitions require an exact
// operation-specific predicate before touching those shared slots.
bool is_bf16_d2h_operation(RileyCudaGraphCaptureOperation operation) noexcept {
  return operation == RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H ||
         operation == RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H;
}

bool is_canonical_gemm_bf16_operation(
    RileyCudaGraphCaptureOperation operation) noexcept {
  return operation == RileyCudaGraphCaptureOperation::kCanonicalGemmBf16;
}

__global__ void graph_fill_f32(float* output, uint64_t element_count,
                               float value) {
  const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  if (index < element_count) {
    output[index] = value;
  }
}

// This is deliberately capture-local rather than riley_cuda_silu_execute:
// eager SiLU owns transient ExclusiveUses and synchronizes completion, neither
// of which is admissible while a graph capture owns permanent input/output
// leases. Keep the arithmetic and grid-stride topology equal to the eager
// BF16 primitive so graph parity includes its exact storage-rounding boundary.
__global__ void graph_silu_bf16(const __nv_bfloat16* input,
                                 __nv_bfloat16* output,
                                 uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const float value = __bfloat162float(input[index]);
    output[index] = __float2bfloat16_rn(value / (1.0F + expf(-value)));
  }
}

// This is deliberately capture-local rather than riley_cuda_gated_multiply_execute:
// eager multiply owns transient ExclusiveUses and synchronizes completion,
// neither of which is admissible while graph capture owns permanent leases for
// three fixed device allocations. Keep this BF16 conversion and grid-stride
// topology equal to the eager primitive so graph parity includes its exact
// storage-rounding boundary.
__global__ void graph_gated_multiply_bf16(const __nv_bfloat16* activated_gate,
                                          const __nv_bfloat16* up,
                                          __nv_bfloat16* output,
                                          uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    output[index] = __float2bfloat16_rn(__bfloat162float(activated_gate[index]) *
                                        __bfloat162float(up[index]));
  }
}

// This is deliberately capture-local rather than riley_cuda_residual_add_execute:
// eager residual add owns transient ExclusiveUses and synchronizes completion,
// neither of which is admissible while graph capture owns permanent leases for
// three fixed device allocations. Keep the conversion and grid-stride topology
// equal to the eager BF16 primitive so graph parity includes its exact
// storage-rounding boundary.
__global__ void graph_residual_add_bf16(const __nv_bfloat16* left,
                                        const __nv_bfloat16* right,
                                        __nv_bfloat16* output,
                                        uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    output[index] = __float2bfloat16_rn(__bfloat162float(left[index]) +
                                        __bfloat162float(right[index]));
  }
}

// This is deliberately capture-local rather than riley_cuda_rms_norm_execute:
// eager RMSNorm owns transient ExclusiveUses and synchronizes completion,
// neither of which is admissible while graph capture owns permanent input,
// weight, and output leases. Keep the canonical BF16 reduction topology and
// normalized-storage rounding boundary equal to the generic eager primitive.
// It intentionally excludes the distinct Hugging Face SmolLM2 and Fixed37
// reduction profiles.
__global__ void graph_canonical_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, uint64_t row_count, uint64_t hidden_size,
    float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t base = row * hidden_size;
    float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float value = __bfloat162float(input[base + column]);
      sum += value * value;
    }
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset != 0; offset /= 2) {
      if (threadIdx.x < offset) {
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
      }
      __syncthreads();
    }
    const float inverse_rms =
        rsqrtf(partial_sums[0] / static_cast<float>(hidden_size) + epsilon);
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float normalized =
          __bfloat162float(input[base + column]) * inverse_rms;
      const __nv_bfloat16 normalized_for_weight =
          __float2bfloat16_rn(normalized);
      output[base + column] = __float2bfloat16_rn(
          __bfloat162float(normalized_for_weight) *
          __bfloat162float(weight[column]));
    }
    __syncthreads();
  }
}

// This is deliberately capture-local rather than riley_cuda_bf16_argmax_execute:
// eager argmax owns transient ExclusiveUses and synchronizes completion,
// neither of which is admissible while capture owns permanent fixed-address
// logits/result leases. Keep the 256-thread, two-stage warp reduction exactly
// equal to the eager primitive so ties and non-finite rows remain byte-stable.
__device__ __forceinline__ void graph_bf16_argmax_select_candidate(
    float candidate_value, uint32_t candidate_token, float* selected_value,
    uint32_t* selected_token) {
  if (candidate_token != RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID &&
      (*selected_token == RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID ||
       candidate_value > *selected_value ||
       (candidate_value == *selected_value &&
        candidate_token < *selected_token))) {
    *selected_value = candidate_value;
    *selected_token = candidate_token;
  }
}

__device__ __forceinline__ void graph_bf16_argmax_reduce_warp(
    float* selected_value, uint32_t* selected_token, uint32_t* non_finite) {
  const uint32_t lane = threadIdx.x % kGraphBf16ArgmaxWarpSize;
  for (uint32_t offset = kGraphBf16ArgmaxWarpSize / 2; offset != 0;
       offset /= 2) {
    const float candidate_value = __shfl_down_sync(
        kGraphBf16ArgmaxFullWarpMask, *selected_value, offset);
    const uint32_t candidate_token = __shfl_down_sync(
        kGraphBf16ArgmaxFullWarpMask, *selected_token, offset);
    const uint32_t candidate_non_finite = __shfl_down_sync(
        kGraphBf16ArgmaxFullWarpMask, *non_finite, offset);
    if (lane + offset < kGraphBf16ArgmaxWarpSize) {
      *non_finite |= candidate_non_finite;
      graph_bf16_argmax_select_candidate(candidate_value, candidate_token,
                                         selected_value, selected_token);
    }
  }
}

__global__ void graph_bf16_argmax_bf16(
    const __nv_bfloat16* logits, RileyCudaBf16ArgmaxResult* results,
    uint64_t row_count, uint64_t vocabulary_size) {
  constexpr uint32_t kWarpCount =
      kGraphBf16ArgmaxThreads / kGraphBf16ArgmaxWarpSize;
  __shared__ float warp_values[kWarpCount];
  __shared__ uint32_t warp_tokens[kWarpCount];
  __shared__ uint32_t warp_non_finite[kWarpCount];

  const uint32_t lane = threadIdx.x % kGraphBf16ArgmaxWarpSize;
  const uint32_t warp = threadIdx.x / kGraphBf16ArgmaxWarpSize;
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    float selected_value = -CUDART_INF_F;
    uint32_t selected_token = RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID;
    uint32_t non_finite = 0;
    const uint64_t row_base = row * vocabulary_size;
    for (uint64_t column = threadIdx.x; column < vocabulary_size;
         column += blockDim.x) {
      const float value = __bfloat162float(logits[row_base + column]);
      if (!isfinite(value)) {
        non_finite = 1;
        continue;
      }
      graph_bf16_argmax_select_candidate(
          value, static_cast<uint32_t>(column), &selected_value,
          &selected_token);
    }

    graph_bf16_argmax_reduce_warp(&selected_value, &selected_token,
                                  &non_finite);
    if (lane == 0) {
      warp_values[warp] = selected_value;
      warp_tokens[warp] = selected_token;
      warp_non_finite[warp] = non_finite;
    }
    __syncthreads();

    if (warp == 0) {
      if (lane < kWarpCount) {
        selected_value = warp_values[lane];
        selected_token = warp_tokens[lane];
        non_finite = warp_non_finite[lane];
      } else {
        selected_value = -CUDART_INF_F;
        selected_token = RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID;
        non_finite = 0;
      }
      graph_bf16_argmax_reduce_warp(&selected_value, &selected_token,
                                    &non_finite);
      if (lane == 0) {
        if (non_finite != 0 ||
            selected_token == RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID) {
          results[row].token_id = RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID;
          results[row].status = RILEY_CUDA_BF16_ARGMAX_STATUS_NON_FINITE;
        } else {
          results[row].token_id = selected_token;
          results[row].status = RILEY_CUDA_BF16_ARGMAX_STATUS_SUCCESS;
        }
      }
    }
    __syncthreads();
  }
}

// This is deliberately capture-local rather than riley_cuda_row_gather_execute:
// eager row gather owns transient ExclusiveUses and synchronizes completion,
// neither of which is admissible while graph capture owns permanent leases for
// its fixed input, index, and output addresses. Preserve its BF16 grid-stride
// topology and raw out-of-range index rule exactly: every output element in an
// invalid row is rounded from CUDART_NAN_F to BF16.
__global__ void graph_bf16_row_gather_bf16(
    const __nv_bfloat16* input, const uint32_t* row_indices,
    __nv_bfloat16* output, uint64_t input_row_count, uint64_t column_count,
    uint64_t output_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < output_element_count; index += stride) {
    const uint64_t output_row = index / column_count;
    const uint64_t column = index - output_row * column_count;
    const uint64_t input_row = row_indices[output_row];
    if (input_row >= input_row_count) {
      output[index] = __float2bfloat16_rn(CUDART_NAN_F);
    } else {
      output[index] = input[input_row * column_count + column];
    }
  }
}

// This C05-20 chain is deliberately capture-local rather than
// riley_cuda_embedding_execute: eager embedding owns transient leases and a
// stack report, neither of which is valid once a graph permanently retains
// fixed raw addresses. Keep the node order, 256-thread grid-stride topology,
// atomic earliest-error selection, BF16 storage move, and fail-before-write
// rule byte-for-byte equivalent to primitives.cu's BF16 path.
__global__ void graph_reset_embedding_error(
    RileyCudaEmbeddingErrorReport* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    report->struct_size = sizeof(RileyCudaEmbeddingErrorReport);
    report->code = RILEY_CUDA_EMBEDDING_ERROR_NONE;
    report->token_position = UINT64_MAX;
    report->token_id = 0;
    report->reserved = 0;
  }
}

__global__ void graph_validate_embedding_tokens(
    const uint32_t* token_ids, uint64_t token_count,
    uint64_t vocabulary_size, RileyCudaEmbeddingErrorReport* report) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t position = first; position < token_count; position += stride) {
    if (token_ids[position] >= vocabulary_size) {
      atomicMin(reinterpret_cast<unsigned long long*>(&report->token_position),
                static_cast<unsigned long long>(position));
    }
  }
}

__global__ void graph_bf16_embedding(
    const __nv_bfloat16* table, const uint32_t* token_ids,
    __nv_bfloat16* output, const RileyCudaEmbeddingErrorReport* report,
    uint64_t hidden_size, uint64_t output_elements) {
  // The validation node is ordered before this node on the captured stream.
  // If any raw device token is invalid, retain eager's all-or-nothing output
  // behavior instead of exposing a partial gather.
  if (report->token_position != UINT64_MAX) {
    return;
  }
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < output_elements; index += stride) {
    const uint64_t token_position = index / hidden_size;
    const uint64_t hidden_index = index - token_position * hidden_size;
    const uint64_t token_id = token_ids[token_position];
    // This is a storage move. Preserve BF16 NaN payloads by avoiding an FP32
    // round trip exactly as eager embedding_kernel<__nv_bfloat16> does.
    output[index] = table[token_id * hidden_size + hidden_index];
  }
}

__global__ void graph_finalize_embedding_error(
    const uint32_t* token_ids, uint64_t token_count,
    RileyCudaEmbeddingErrorReport* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (report->token_position == UINT64_MAX) {
      report->token_position = 0;
      report->token_id = 0;
      report->code = RILEY_CUDA_EMBEDDING_ERROR_NONE;
    } else if (report->token_position < token_count) {
      report->token_id = token_ids[report->token_position];
      report->code = RILEY_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE;
    }
  }
}

// This is deliberately capture-local rather than riley_cuda_indexed_rope_execute:
// eager indexed RoPE acquires transient ExclusiveUses and synchronizes
// completion, neither of which is admissible once the graph owns permanent
// leases for its five fixed device allocations. Keep every BF16
// storage-rounding boundary and the raw device-position OOB -> NaN behavior
// identical to indexed_rope_kernel<__nv_bfloat16>.
__global__ void graph_indexed_rope_bf16(
    const __nv_bfloat16* input, const float* cos, const float* sin,
    const uint32_t* positions, __nv_bfloat16* output,
    uint64_t table_position_count, uint64_t head_count,
    uint64_t head_size, uint64_t rotary_dimension,
    uint64_t work_item_count) {
  const uint64_t half = rotary_dimension / 2;
  const uint64_t tail = head_size - rotary_dimension;
  const uint64_t units_per_head = half + tail;
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t work = first; work < work_item_count; work += stride) {
    const uint64_t head_linear = work / units_per_head;
    const uint64_t unit = work - head_linear * units_per_head;
    const uint64_t row = head_linear / head_count;
    const uint64_t base = head_linear * head_size;
    if (unit < half) {
      const uint64_t first_index = base + unit;
      const uint64_t second_index = first_index + half;
      const uint64_t position = positions[row];
      if (position >= table_position_count) {
        output[first_index] = __float2bfloat16_rn(CUDART_NAN_F);
        output[second_index] = __float2bfloat16_rn(CUDART_NAN_F);
        continue;
      }
      const uint64_t table_index = position * half + unit;
      const float first_value = __bfloat162float(input[first_index]);
      const float second_value = __bfloat162float(input[second_index]);
      const float cosine =
          __bfloat162float(__float2bfloat16_rn(cos[table_index]));
      const float sine =
          __bfloat162float(__float2bfloat16_rn(sin[table_index]));
      const float first_cosine = __bfloat162float(
          __float2bfloat16_rn(first_value * cosine));
      const float second_sine = __bfloat162float(
          __float2bfloat16_rn(second_value * sine));
      const float second_cosine = __bfloat162float(
          __float2bfloat16_rn(second_value * cosine));
      const float first_sine = __bfloat162float(
          __float2bfloat16_rn(first_value * sine));
      output[first_index] = __float2bfloat16_rn(first_cosine - second_sine);
      output[second_index] = __float2bfloat16_rn(second_cosine + first_sine);
    } else {
      const uint64_t dimension = rotary_dimension + (unit - half);
      output[base + dimension] = input[base + dimension];
    }
  }
}

// This capture-local copy mirrors ragged_paged_kv_cache_write_kernel. Eager
// execution owns transient ExclusiveUses and synchronizes completion, while a
// graph keeps all nine raw device allocations leased for its lifetime. Raw
// metadata must remain device-resident: bounds-invalid rows are ignored just
// as in eager execution, and duplicate valid destinations remain unspecified.
struct GraphRaggedPagedKvDeviceBatch {
  const uint32_t* sequence_block_offsets;
  const uint32_t* block_ids;
  const uint16_t* valid_tokens;
  const uint32_t* row_sequence_slots;
  const uint32_t* row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
};

__device__ __forceinline__ bool graph_resolve_ragged_paged_kv_cache_base(
    const GraphRaggedPagedKvDeviceBatch& batch, uint64_t row,
    uint64_t logical_position, uint64_t key_value_head,
    uint64_t key_value_head_count, uint64_t head_size, uint64_t* output) {
  if (output == nullptr || row >= batch.active_row_count) {
    return false;
  }
  const uint64_t sequence = batch.row_sequence_slots[row];
  if (sequence >= batch.sequence_count) {
    return false;
  }
  const uint64_t block_begin = batch.sequence_block_offsets[sequence];
  const uint64_t block_end = batch.sequence_block_offsets[sequence + 1];
  if (block_begin > block_end || block_end > batch.block_count) {
    return false;
  }
  const uint64_t logical_block =
      logical_position / RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (logical_block >= block_end - block_begin) {
    return false;
  }
  const uint64_t block_index = block_begin + logical_block;
  const uint64_t token_in_block =
      logical_position % RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  const uint64_t physical_block = batch.block_ids[block_index];
  const uint64_t valid = batch.valid_tokens[block_index];
  if (physical_block >= batch.physical_block_count || valid == 0 ||
      valid > RILEY_CUDA_PAGED_KV_BLOCK_SIZE || token_in_block >= valid) {
    return false;
  }
  *output =
      ((physical_block * key_value_head_count + key_value_head) *
           RILEY_CUDA_PAGED_KV_BLOCK_SIZE +
       token_in_block) *
      head_size;
  return true;
}

__global__ void graph_ragged_paged_kv_cache_write_bf16(
    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    GraphRaggedPagedKvDeviceBatch batch, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         static_cast<uint64_t>(threadIdx.x);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t packed_row = index / head_size;
    const uint64_t key_value_head = packed_row % key_value_head_count;
    const uint64_t row = packed_row / key_value_head_count;
    const uint64_t logical_position = batch.row_positions[row];
    uint64_t cache_base = 0;
    if (graph_resolve_ragged_paged_kv_cache_base(
            batch, row, logical_position, key_value_head,
            key_value_head_count, head_size, &cache_base)) {
      key_pool[cache_base + depth] = key_source[index];
      value_pool[cache_base + depth] = value_source[index];
    }
  }
}

// C05-19 keeps this capture-local copy of the eager grouped attention
// arithmetic instead of calling the eager entry point: eager ownership uses
// transient ExclusiveUses and synchronous completion, whereas graph replay
// permanently owns all nine fixed addresses. Keep the per-warp reductions,
// staged BF16 score boundary, online-softmax token order, and raw metadata
// recovery exactly aligned with batch_primitives.cu.
__device__ __forceinline__ float graph_ragged_attention_warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kGraphRaggedAttentionWarpSize / 2; offset != 0;
       offset /= 2) {
    value += __shfl_down_sync(kGraphRaggedAttentionFullWarpMask, value,
                              offset);
  }
  return value;
}

__device__ __forceinline__ float graph_ragged_attention_staged_score(
    float dot_product, float scale) {
  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(staged_dot) * scale));
}

__device__ __forceinline__ void graph_ragged_attention_update_online_state(
    float score, float* maximum, float* denominator, float* alpha,
    float* beta) {
  if (isnan(score) || isnan(*maximum)) {
    *maximum = CUDART_NAN_F;
    *denominator = CUDART_NAN_F;
    *alpha = CUDART_NAN_F;
    *beta = CUDART_NAN_F;
    return;
  }
  const bool score_is_positive_infinity = isinf(score) && score > 0.0F;
  const bool maximum_is_positive_infinity =
      isinf(*maximum) && *maximum > 0.0F;
  if (score_is_positive_infinity) {
    if (maximum_is_positive_infinity) {
      *alpha = 1.0F;
      *beta = 1.0F;
      *denominator += 1.0F;
    } else {
      *maximum = CUDART_INF_F;
      *denominator = 1.0F;
      *alpha = 0.0F;
      *beta = 1.0F;
    }
    return;
  }
  if (maximum_is_positive_infinity || (isinf(score) && score < 0.0F)) {
    *alpha = 1.0F;
    *beta = 0.0F;
    return;
  }
  const float next_maximum = fmaxf(*maximum, score);
  *alpha = *denominator == 0.0F ? 0.0F : expf(*maximum - next_maximum);
  *beta = expf(score - next_maximum);
  *denominator = fmaf(*alpha, *denominator, *beta);
  *maximum = next_maximum;
}

__device__ __forceinline__ float graph_ragged_attention_update_numerator(
    float numerator, float value, float alpha, float beta) {
  if (beta == 0.0F) {
    return alpha * numerator;
  }
  if (alpha == 0.0F) {
    return beta * value;
  }
  return fmaf(beta, value, alpha * numerator);
}

__global__ __launch_bounds__(kGraphRaggedAttentionThreads)
void graph_grouped_ragged_paged_attention_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    GraphRaggedPagedKvDeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x % kGraphRaggedAttentionWarpSize;
  const uint32_t warp = threadIdx.x / kGraphRaggedAttentionWarpSize;
  const uint64_t row = blockIdx.x;
  const uint64_t warps_per_block =
      blockDim.x / kGraphRaggedAttentionWarpSize;
  const uint64_t query_head =
      static_cast<uint64_t>(blockIdx.y) * warps_per_block + warp;
  if (query_head >= query_head_count) {
    return;
  }
  const uint64_t output_base =
      (row * query_head_count + query_head) * kGraphRaggedAttentionHeadSize;
  if (row >= batch.active_row_count) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    output[output_base + lane] = zero;
    output[output_base + lane + kGraphRaggedAttentionWarpSize] = zero;
    return;
  }
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base =
      (row * query_head_count + query_head) * kGraphRaggedAttentionHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high = __bfloat162float(
      query[query_base + lane + kGraphRaggedAttentionWarpSize]);
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  bool valid = true;
  for (uint64_t logical_token = 0; logical_token < logical_token_count;
       ++logical_token) {
    uint64_t cache_base = 0;
    if (!graph_resolve_ragged_paged_kv_cache_base(
            batch, row, logical_token, key_value_head, key_value_head_count,
            kGraphRaggedAttentionHeadSize, &cache_base)) {
      valid = false;
      break;
    }
    const float key_low = __bfloat162float(key_pool[cache_base + lane]);
    const float key_high = __bfloat162float(
        key_pool[cache_base + lane + kGraphRaggedAttentionWarpSize]);
    float score = fmaf(query_low, key_low, query_high * key_high);
    score = graph_ragged_attention_warp_sum(score);
    score = graph_ragged_attention_staged_score(
        __shfl_sync(kGraphRaggedAttentionFullWarpMask, score, 0), scale);
    float alpha = 0.0F;
    float beta = 0.0F;
    if (lane == 0) {
      graph_ragged_attention_update_online_state(score, &maximum,
                                                 &denominator, &alpha, &beta);
    }
    alpha = __shfl_sync(kGraphRaggedAttentionFullWarpMask, alpha, 0);
    beta = __shfl_sync(kGraphRaggedAttentionFullWarpMask, beta, 0);
    numerator_low = graph_ragged_attention_update_numerator(
        numerator_low, __bfloat162float(value_pool[cache_base + lane]), alpha,
        beta);
    numerator_high = graph_ragged_attention_update_numerator(
        numerator_high,
        __bfloat162float(value_pool[cache_base + lane +
                                    kGraphRaggedAttentionWarpSize]),
        alpha, beta);
  }
  denominator =
      __shfl_sync(kGraphRaggedAttentionFullWarpMask, denominator, 0);
  const float inverse_denominator =
      !valid ? CUDART_NAN_F
             : (isnan(denominator)
                    ? CUDART_NAN_F
                    : (denominator > 0.0F ? 1.0F / denominator : 0.0F));
  output[output_base + lane] =
      __float2bfloat16_rn(numerator_low * inverse_denominator);
  output[output_base + lane + kGraphRaggedAttentionWarpSize] =
      __float2bfloat16_rn(numerator_high * inverse_denominator);
}

__global__ __launch_bounds__(kGraphRaggedAttentionThreads)
void graph_grouped_ragged_paged_attention_gqa_shared_kv_bf16(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    GraphRaggedPagedKvDeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x % kGraphRaggedAttentionWarpSize;
  const uint32_t warp = threadIdx.x / kGraphRaggedAttentionWarpSize;
  const uint64_t row = blockIdx.x;
  const uint64_t warps_per_block =
      blockDim.x / kGraphRaggedAttentionWarpSize;
  const uint64_t key_value_head = blockIdx.y;
  const uint64_t query_head = key_value_head * warps_per_block + warp;
  const uint64_t output_base =
      (row * query_head_count + query_head) * kGraphRaggedAttentionHeadSize;
  if (row >= batch.active_row_count) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    output[output_base + lane] = zero;
    output[output_base + lane + kGraphRaggedAttentionWarpSize] = zero;
    return;
  }
  const uint64_t query_base =
      (row * query_head_count + query_head) * kGraphRaggedAttentionHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high = __bfloat162float(
      query[query_base + lane + kGraphRaggedAttentionWarpSize]);
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;
  __shared__ __nv_bfloat16
      shared_keys[kGraphRaggedAttentionSharedTileTokens]
                 [kGraphRaggedAttentionHeadSize];
  __shared__ __nv_bfloat16
      shared_values[kGraphRaggedAttentionSharedTileTokens]
                   [kGraphRaggedAttentionHeadSize];
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  bool valid = true;
  for (uint64_t tile_start = 0; tile_start < logical_token_count;
       tile_start += kGraphRaggedAttentionSharedTileTokens) {
    if (tile_start != 0) {
      __syncthreads();
    }
    const uint64_t remaining = logical_token_count - tile_start;
    const uint32_t tile_count = static_cast<uint32_t>(
        remaining < kGraphRaggedAttentionSharedTileTokens
            ? remaining
            : kGraphRaggedAttentionSharedTileTokens);
    bool tile_valid = true;
    for (uint32_t tile_offset = warp; tile_offset < tile_count;
         tile_offset += static_cast<uint32_t>(warps_per_block)) {
      uint32_t cache_base_low = 0;
      uint32_t cache_base_high = 0;
      int token_valid = 0;
      if (lane == 0) {
        uint64_t cache_base = 0;
        token_valid = graph_resolve_ragged_paged_kv_cache_base(
                          batch, row, tile_start + tile_offset,
                          key_value_head, key_value_head_count,
                          kGraphRaggedAttentionHeadSize, &cache_base)
                          ? 1
                          : 0;
        cache_base_low = static_cast<uint32_t>(cache_base);
        cache_base_high = static_cast<uint32_t>(cache_base >> 32);
      }
      token_valid =
          __shfl_sync(kGraphRaggedAttentionFullWarpMask, token_valid, 0);
      cache_base_low =
          __shfl_sync(kGraphRaggedAttentionFullWarpMask, cache_base_low, 0);
      cache_base_high =
          __shfl_sync(kGraphRaggedAttentionFullWarpMask, cache_base_high, 0);
      if (token_valid == 0) {
        tile_valid = false;
        continue;
      }
      const uint64_t cache_base =
          (static_cast<uint64_t>(cache_base_high) << 32) | cache_base_low;
      shared_keys[tile_offset][lane] = key_pool[cache_base + lane];
      shared_keys[tile_offset][lane + kGraphRaggedAttentionWarpSize] =
          key_pool[cache_base + lane + kGraphRaggedAttentionWarpSize];
      shared_values[tile_offset][lane] = value_pool[cache_base + lane];
      shared_values[tile_offset][lane + kGraphRaggedAttentionWarpSize] =
          value_pool[cache_base + lane + kGraphRaggedAttentionWarpSize];
    }
    if (__syncthreads_or(tile_valid ? 0 : 1) != 0) {
      valid = false;
      break;
    }
    for (uint32_t tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
      const float key_low =
          __bfloat162float(shared_keys[tile_offset][lane]);
      const float key_high = __bfloat162float(
          shared_keys[tile_offset][lane + kGraphRaggedAttentionWarpSize]);
      float score = fmaf(query_low, key_low, query_high * key_high);
      score = graph_ragged_attention_warp_sum(score);
      score = graph_ragged_attention_staged_score(
          __shfl_sync(kGraphRaggedAttentionFullWarpMask, score, 0), scale);
      float alpha = 0.0F;
      float beta = 0.0F;
      if (lane == 0) {
        graph_ragged_attention_update_online_state(score, &maximum,
                                                   &denominator, &alpha, &beta);
      }
      alpha = __shfl_sync(kGraphRaggedAttentionFullWarpMask, alpha, 0);
      beta = __shfl_sync(kGraphRaggedAttentionFullWarpMask, beta, 0);
      numerator_low = graph_ragged_attention_update_numerator(
          numerator_low,
          __bfloat162float(shared_values[tile_offset][lane]), alpha, beta);
      numerator_high = graph_ragged_attention_update_numerator(
          numerator_high,
          __bfloat162float(shared_values[tile_offset]
                                         [lane +
                                          kGraphRaggedAttentionWarpSize]),
          alpha, beta);
    }
  }
  denominator =
      __shfl_sync(kGraphRaggedAttentionFullWarpMask, denominator, 0);
  const float inverse_denominator =
      !valid ? CUDART_NAN_F
             : (isnan(denominator)
                    ? CUDART_NAN_F
                    : (denominator > 0.0F ? 1.0F / denominator : 0.0F));
  output[output_base + lane] =
      __float2bfloat16_rn(numerator_low * inverse_denominator);
  output[output_base + lane + kGraphRaggedAttentionWarpSize] =
      __float2bfloat16_rn(numerator_high * inverse_denominator);
}

bool residual_add_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && capture->residual_add_left == nullptr &&
         capture->residual_add_right == nullptr &&
         capture->residual_add_element_count == 0 &&
         capture->residual_add_enqueue_count == 0 &&
         !capture->residual_add_left_lease_held &&
         !capture->residual_add_right_lease_held;
}

bool residual_add_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->residual_add_left == nullptr &&
         graph->residual_add_right == nullptr &&
         graph->residual_add_element_count == 0;
}

bool residual_add_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->residual_add_left == nullptr &&
         exec->residual_add_right == nullptr &&
         exec->residual_add_element_count == 0;
}

bool canonical_rms_norm_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && capture->canonical_rms_norm_input == nullptr &&
         capture->canonical_rms_norm_weight == nullptr &&
         capture->canonical_rms_norm_row_count == 0 &&
         capture->canonical_rms_norm_hidden_size == 0 &&
         capture->canonical_rms_norm_epsilon == 0.0F &&
         capture->canonical_rms_norm_enqueue_count == 0 &&
         !capture->canonical_rms_norm_input_lease_held &&
         !capture->canonical_rms_norm_weight_lease_held;
}

bool canonical_rms_norm_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->canonical_rms_norm_input == nullptr &&
         graph->canonical_rms_norm_weight == nullptr &&
         graph->canonical_rms_norm_row_count == 0 &&
         graph->canonical_rms_norm_hidden_size == 0 &&
         graph->canonical_rms_norm_epsilon == 0.0F;
}

bool canonical_rms_norm_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->canonical_rms_norm_input == nullptr &&
         exec->canonical_rms_norm_weight == nullptr &&
         exec->canonical_rms_norm_row_count == 0 &&
         exec->canonical_rms_norm_hidden_size == 0 &&
         exec->canonical_rms_norm_epsilon == 0.0F;
}

bool bf16_argmax_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && capture->bf16_argmax_logits == nullptr &&
         capture->bf16_argmax_row_count == 0 &&
         capture->bf16_argmax_vocabulary_size == 0 &&
         capture->bf16_argmax_enqueue_count == 0 &&
         !capture->bf16_argmax_logits_lease_held;
}

bool bf16_argmax_graph_fields_are_clear(const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->bf16_argmax_logits == nullptr &&
         graph->bf16_argmax_row_count == 0 &&
         graph->bf16_argmax_vocabulary_size == 0;
}

bool bf16_argmax_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->bf16_argmax_logits == nullptr &&
         exec->bf16_argmax_row_count == 0 &&
         exec->bf16_argmax_vocabulary_size == 0;
}

bool bf16_row_gather_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && capture->bf16_row_gather_input == nullptr &&
         capture->bf16_row_gather_indices == nullptr &&
         capture->bf16_row_gather_input_row_count == 0 &&
         capture->bf16_row_gather_output_row_count == 0 &&
         capture->bf16_row_gather_column_count == 0 &&
         capture->bf16_row_gather_enqueue_count == 0 &&
         !capture->bf16_row_gather_input_lease_held &&
         !capture->bf16_row_gather_indices_lease_held;
}

bool bf16_row_gather_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->bf16_row_gather_input == nullptr &&
         graph->bf16_row_gather_indices == nullptr &&
         graph->bf16_row_gather_input_row_count == 0 &&
         graph->bf16_row_gather_output_row_count == 0 &&
         graph->bf16_row_gather_column_count == 0;
}

bool bf16_row_gather_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->bf16_row_gather_input == nullptr &&
         exec->bf16_row_gather_indices == nullptr &&
         exec->bf16_row_gather_input_row_count == 0 &&
         exec->bf16_row_gather_output_row_count == 0 &&
         exec->bf16_row_gather_column_count == 0;
}

bool bf16_row_gather_argmax_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         capture->bf16_row_gather_argmax_input == nullptr &&
         capture->bf16_row_gather_argmax_indices == nullptr &&
         capture->bf16_row_gather_argmax_gathered_logits == nullptr &&
         capture->bf16_row_gather_argmax_input_row_count == 0 &&
         capture->bf16_row_gather_argmax_output_row_count == 0 &&
         capture->bf16_row_gather_argmax_vocabulary_size == 0 &&
         capture->bf16_row_gather_argmax_enqueue_count == 0 &&
         !capture->bf16_row_gather_argmax_input_lease_held &&
         !capture->bf16_row_gather_argmax_indices_lease_held &&
         !capture->bf16_row_gather_argmax_gathered_logits_lease_held;
}

bool bf16_row_gather_argmax_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->bf16_row_gather_argmax_input == nullptr &&
         graph->bf16_row_gather_argmax_indices == nullptr &&
         graph->bf16_row_gather_argmax_gathered_logits == nullptr &&
         graph->bf16_row_gather_argmax_input_row_count == 0 &&
         graph->bf16_row_gather_argmax_output_row_count == 0 &&
         graph->bf16_row_gather_argmax_vocabulary_size == 0;
}

bool bf16_row_gather_argmax_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->bf16_row_gather_argmax_input == nullptr &&
         exec->bf16_row_gather_argmax_indices == nullptr &&
         exec->bf16_row_gather_argmax_gathered_logits == nullptr &&
         exec->bf16_row_gather_argmax_input_row_count == 0 &&
         exec->bf16_row_gather_argmax_output_row_count == 0 &&
         exec->bf16_row_gather_argmax_vocabulary_size == 0;
}

bool bf16_row_gather_argmax_d2h_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         capture->bf16_row_gather_argmax_d2h_input == nullptr &&
         capture->bf16_row_gather_argmax_d2h_indices == nullptr &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits == nullptr &&
         capture->bf16_row_gather_argmax_d2h_pinned_results == nullptr &&
         capture->bf16_row_gather_argmax_d2h_input_row_count == 0 &&
         capture->bf16_row_gather_argmax_d2h_output_row_count == 0 &&
         capture->bf16_row_gather_argmax_d2h_vocabulary_size == 0 &&
         capture->bf16_row_gather_argmax_d2h_result_byte_len == 0 &&
         capture->bf16_row_gather_argmax_d2h_enqueue_count == 0 &&
         !capture->bf16_row_gather_argmax_d2h_input_lease_held &&
         !capture->bf16_row_gather_argmax_d2h_indices_lease_held &&
         !capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held &&
         !capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held;
}

bool bf16_row_gather_argmax_d2h_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->bf16_row_gather_argmax_d2h_input == nullptr &&
         graph->bf16_row_gather_argmax_d2h_indices == nullptr &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits == nullptr &&
         graph->bf16_row_gather_argmax_d2h_pinned_results == nullptr &&
         graph->bf16_row_gather_argmax_d2h_input_row_count == 0 &&
         graph->bf16_row_gather_argmax_d2h_output_row_count == 0 &&
         graph->bf16_row_gather_argmax_d2h_vocabulary_size == 0 &&
         graph->bf16_row_gather_argmax_d2h_result_byte_len == 0;
}

bool bf16_row_gather_argmax_d2h_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->bf16_row_gather_argmax_d2h_input == nullptr &&
         exec->bf16_row_gather_argmax_d2h_indices == nullptr &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits == nullptr &&
         exec->bf16_row_gather_argmax_d2h_pinned_results == nullptr &&
         exec->bf16_row_gather_argmax_d2h_input_row_count == 0 &&
         exec->bf16_row_gather_argmax_d2h_output_row_count == 0 &&
         exec->bf16_row_gather_argmax_d2h_vocabulary_size == 0 &&
         exec->bf16_row_gather_argmax_d2h_result_byte_len == 0 &&
         !exec->bf16_row_gather_argmax_d2h_completion_visible;
}

bool indexed_rope_bf16_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr && capture->indexed_rope_bf16_input == nullptr &&
         capture->indexed_rope_bf16_cos == nullptr &&
         capture->indexed_rope_bf16_sin == nullptr &&
         capture->indexed_rope_bf16_positions == nullptr &&
         capture->indexed_rope_bf16_active_row_count == 0 &&
         capture->indexed_rope_bf16_head_count == 0 &&
         capture->indexed_rope_bf16_head_size == 0 &&
         capture->indexed_rope_bf16_rotary_dimension == 0 &&
         capture->indexed_rope_bf16_table_position_count == 0 &&
         capture->indexed_rope_bf16_enqueue_count == 0 &&
         !capture->indexed_rope_bf16_input_lease_held &&
         !capture->indexed_rope_bf16_cos_lease_held &&
         !capture->indexed_rope_bf16_sin_lease_held &&
         !capture->indexed_rope_bf16_positions_lease_held;
}

bool indexed_rope_bf16_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && graph->indexed_rope_bf16_input == nullptr &&
         graph->indexed_rope_bf16_cos == nullptr &&
         graph->indexed_rope_bf16_sin == nullptr &&
         graph->indexed_rope_bf16_positions == nullptr &&
         graph->indexed_rope_bf16_active_row_count == 0 &&
         graph->indexed_rope_bf16_head_count == 0 &&
         graph->indexed_rope_bf16_head_size == 0 &&
         graph->indexed_rope_bf16_rotary_dimension == 0 &&
         graph->indexed_rope_bf16_table_position_count == 0;
}

bool indexed_rope_bf16_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && exec->indexed_rope_bf16_input == nullptr &&
         exec->indexed_rope_bf16_cos == nullptr &&
         exec->indexed_rope_bf16_sin == nullptr &&
         exec->indexed_rope_bf16_positions == nullptr &&
         exec->indexed_rope_bf16_active_row_count == 0 &&
         exec->indexed_rope_bf16_head_count == 0 &&
         exec->indexed_rope_bf16_head_size == 0 &&
         exec->indexed_rope_bf16_rotary_dimension == 0 &&
         exec->indexed_rope_bf16_table_position_count == 0;
}

bool canonical_gemm_bf16_state_is_clear(
    const RileyCudaCanonicalGemmBf16GraphState& state) noexcept {
  return state.plan == nullptr && state.input == nullptr &&
         state.weight == nullptr && state.workspace == nullptr &&
         state.input_byte_len == 0 && state.weight_byte_len == 0 &&
         state.output_byte_len == 0 && state.workspace_byte_len == 0 &&
         state.enqueue_count == 0 && !state.plan_lease_held &&
         !state.input_lease_held && !state.weight_lease_held &&
         !state.workspace_lease_held;
}

bool canonical_gemm_bf16_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         canonical_gemm_bf16_state_is_clear(capture->canonical_gemm_bf16);
}

bool canonical_gemm_bf16_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr &&
         canonical_gemm_bf16_state_is_clear(graph->canonical_gemm_bf16);
}

bool canonical_gemm_bf16_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr &&
         canonical_gemm_bf16_state_is_clear(exec->canonical_gemm_bf16);
}

bool ragged_paged_kv_cache_write_bf16_state_is_clear(
    const RileyCudaRaggedPagedKvCacheWriteBf16State& state) noexcept {
  return state.key_source == nullptr && state.value_source == nullptr &&
         state.key_pool == nullptr && state.value_pool == nullptr &&
         state.sequence_block_offsets == nullptr && state.block_ids == nullptr &&
         state.valid_tokens == nullptr && state.row_sequence_slots == nullptr &&
         state.row_positions == nullptr && state.sequence_count == 0 &&
         state.block_count == 0 && state.active_row_count == 0 &&
         state.physical_block_count == 0 && state.key_value_head_count == 0 &&
         state.head_size == 0 && state.query_head_count == 0 &&
         state.output_row_count == 0 && state.attention_scale == 0.0F &&
         !state.grouped_attention && state.enqueue_count == 0 &&
         !state.key_source_lease_held && !state.value_source_lease_held &&
         !state.value_pool_lease_held &&
         !state.sequence_block_offsets_lease_held &&
         !state.block_ids_lease_held && !state.valid_tokens_lease_held &&
         !state.row_sequence_slots_lease_held &&
         !state.row_positions_lease_held;
}

bool ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         ragged_paged_kv_cache_write_bf16_state_is_clear(
             capture->ragged_paged_kv_write_bf16);
}

bool ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && ragged_paged_kv_cache_write_bf16_state_is_clear(
                                 graph->ragged_paged_kv_write_bf16);
}

bool ragged_paged_kv_cache_write_bf16_exec_fields_are_clear(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && ragged_paged_kv_cache_write_bf16_state_is_clear(
                                exec->ragged_paged_kv_write_bf16);
}

bool bf16_argmax_shape_is_valid(uint64_t row_count, uint64_t vocabulary_size,
                                uint64_t* out_logit_element_count) noexcept {
  if (out_logit_element_count == nullptr || row_count == 0 ||
      vocabulary_size == 0 || vocabulary_size > UINT32_MAX ||
      row_count > UINT64_MAX / vocabulary_size) {
    return false;
  }
  *out_logit_element_count = row_count * vocabulary_size;
  return true;
}

bool bf16_row_gather_shape_is_valid(
    uint64_t input_row_count, uint64_t output_row_count,
    uint64_t column_count, uint64_t* out_input_element_count,
    uint64_t* out_output_element_count) noexcept {
  if (out_input_element_count == nullptr || out_output_element_count == nullptr ||
      input_row_count == 0 || output_row_count == 0 || column_count == 0 ||
      input_row_count > UINT64_MAX / column_count ||
      output_row_count > UINT64_MAX / column_count) {
    return false;
  }
  *out_input_element_count = input_row_count * column_count;
  *out_output_element_count = output_row_count * column_count;
  return true;
}

bool bf16_row_gather_argmax_shape_is_valid(
    uint64_t input_row_count, uint64_t output_row_count,
    uint64_t vocabulary_size, uint64_t* out_input_element_count,
    uint64_t* out_gathered_element_count) noexcept {
  if (!bf16_row_gather_shape_is_valid(
          input_row_count, output_row_count, vocabulary_size,
          out_input_element_count, out_gathered_element_count) ||
      vocabulary_size > UINT32_MAX) {
    return false;
  }
  return true;
}

bool bf16_row_gather_argmax_d2h_shape_is_valid(
    uint64_t input_row_count, uint64_t output_row_count,
    uint64_t vocabulary_size, uint64_t* out_input_element_count,
    uint64_t* out_gathered_element_count,
    uint64_t* out_result_byte_len) noexcept {
  if (out_result_byte_len == nullptr ||
      !bf16_row_gather_argmax_shape_is_valid(
          input_row_count, output_row_count, vocabulary_size,
          out_input_element_count, out_gathered_element_count) ||
      output_row_count >
          UINT64_MAX / sizeof(RileyCudaBf16ArgmaxResult)) {
    return false;
  }
  *out_result_byte_len =
      output_row_count * sizeof(RileyCudaBf16ArgmaxResult);
  return true;
}

bool bf16_embedding_status_d2h_shape_is_valid(
    uint64_t vocabulary_size, uint64_t token_count, uint64_t hidden_size,
    uint64_t* out_table_element_count,
    uint64_t* out_output_element_count) noexcept {
  if (out_table_element_count == nullptr || out_output_element_count == nullptr ||
      vocabulary_size == 0 || token_count == 0 || hidden_size == 0 ||
      vocabulary_size > UINT64_MAX / hidden_size ||
      token_count > UINT64_MAX / hidden_size) {
    return false;
  }
  *out_table_element_count = vocabulary_size * hidden_size;
  *out_output_element_count = token_count * hidden_size;
  return true;
}

bool indexed_rope_bf16_shape_is_valid(
    uint64_t active_row_count, uint64_t head_count, uint64_t head_size,
    uint64_t rotary_dimension, uint64_t table_position_count,
    uint64_t* out_tensor_element_count, uint64_t* out_table_element_count,
    uint64_t* out_work_item_count) noexcept {
  if (out_tensor_element_count == nullptr || out_table_element_count == nullptr ||
      out_work_item_count == nullptr || active_row_count == 0 ||
      head_count == 0 || head_size == 0 || rotary_dimension == 0 ||
      rotary_dimension > head_size || rotary_dimension % 2 != 0 ||
      table_position_count == 0) {
    return false;
  }
  const uint64_t half = rotary_dimension / 2;
  const uint64_t tail = head_size - rotary_dimension;
  uint64_t head_rows = 0;
  uint64_t units_per_head = 0;
  if (active_row_count > UINT64_MAX / head_count ||
      (head_rows = active_row_count * head_count) > UINT64_MAX / head_size ||
      table_position_count > UINT64_MAX / half ||
      half > UINT64_MAX - tail ||
      (units_per_head = half + tail) == 0 ||
      head_rows > UINT64_MAX / units_per_head) {
    return false;
  }
  *out_tensor_element_count = head_rows * head_size;
  *out_table_element_count = table_position_count * half;
  *out_work_item_count = head_rows * units_per_head;
  return *out_tensor_element_count != 0 && *out_table_element_count != 0 &&
         *out_work_item_count != 0;
}

bool ragged_paged_kv_cache_write_bf16_shape_is_valid(
    uint64_t sequence_count, uint64_t block_count,
    uint64_t active_row_count, uint64_t physical_block_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t* out_source_element_count,
    uint64_t* out_pool_element_count) noexcept {
  if (out_source_element_count == nullptr || out_pool_element_count == nullptr ||
      sequence_count == 0 || block_count == 0 || active_row_count == 0 ||
      physical_block_count == 0 || key_value_head_count == 0 ||
      head_size == 0 || sequence_count > UINT32_MAX ||
      block_count > UINT32_MAX || active_row_count > UINT32_MAX ||
      physical_block_count > UINT32_MAX || block_count > physical_block_count) {
    return false;
  }
  if (active_row_count > UINT64_MAX / key_value_head_count) {
    return false;
  }
  const uint64_t source_rows = active_row_count * key_value_head_count;
  if (source_rows > UINT64_MAX / head_size) {
    return false;
  }
  if (physical_block_count > UINT64_MAX / key_value_head_count) {
    return false;
  }
  const uint64_t pool_heads = physical_block_count * key_value_head_count;
  if (pool_heads > UINT64_MAX / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) {
    return false;
  }
  const uint64_t pool_tokens =
      pool_heads * RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (pool_tokens > UINT64_MAX / head_size) {
    return false;
  }
  *out_source_element_count = source_rows * head_size;
  *out_pool_element_count = pool_tokens * head_size;
  return *out_source_element_count != 0 && *out_pool_element_count != 0;
}

bool grouped_ragged_paged_attention_bf16_shape_is_valid(
    uint64_t sequence_count, uint64_t block_count,
    uint64_t active_row_count, uint64_t physical_block_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t output_row_count, float scale,
    uint64_t* out_query_element_count, uint64_t* out_pool_element_count,
    uint64_t* out_output_element_count) noexcept {
  if (out_query_element_count == nullptr || out_pool_element_count == nullptr ||
      out_output_element_count == nullptr || sequence_count == 0 ||
      block_count == 0 || active_row_count == 0 ||
      physical_block_count == 0 || query_head_count == 0 ||
      key_value_head_count == 0 || output_row_count < active_row_count ||
      sequence_count > UINT32_MAX || block_count > UINT32_MAX ||
      active_row_count > UINT32_MAX || physical_block_count > UINT32_MAX ||
      block_count > physical_block_count ||
      query_head_count > kGraphMaximumGridYOrZ ||
      output_row_count > kGraphMaximumGridX ||
      query_head_count % key_value_head_count != 0 ||
      !std::isfinite(scale) || scale <= 0.0F) {
    return false;
  }
  if (active_row_count > UINT64_MAX / query_head_count) {
    return false;
  }
  const uint64_t query_rows = active_row_count * query_head_count;
  if (query_rows > UINT64_MAX / kGraphRaggedAttentionHeadSize ||
      output_row_count > UINT64_MAX / query_head_count) {
    return false;
  }
  const uint64_t output_rows = output_row_count * query_head_count;
  if (output_rows > UINT64_MAX / kGraphRaggedAttentionHeadSize ||
      physical_block_count > UINT64_MAX / key_value_head_count) {
    return false;
  }
  const uint64_t pool_heads = physical_block_count * key_value_head_count;
  if (pool_heads > UINT64_MAX / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) {
    return false;
  }
  const uint64_t pool_tokens =
      pool_heads * RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (pool_tokens > UINT64_MAX / kGraphRaggedAttentionHeadSize) {
    return false;
  }
  *out_query_element_count =
      query_rows * kGraphRaggedAttentionHeadSize;
  *out_pool_element_count =
      pool_tokens * kGraphRaggedAttentionHeadSize;
  *out_output_element_count =
      output_rows * kGraphRaggedAttentionHeadSize;
  return *out_query_element_count != 0 && *out_pool_element_count != 0 &&
         *out_output_element_count != 0;
}

bool canonical_rms_norm_element_count(uint64_t row_count, uint64_t hidden_size,
                                      uint64_t* out_element_count) noexcept {
  if (out_element_count == nullptr || row_count == 0 || hidden_size == 0 ||
      row_count > UINT64_MAX / hidden_size) {
    return false;
  }
  *out_element_count = row_count * hidden_size;
  return true;
}

// These predicates are intentionally structural only: callers separately
// establish whether a capture is live, whether its CUDA graph/exec handle is
// present, and which one-shot lifecycle boundary has occurred. Keeping the
// immutable three-buffer contract here lets every transition reject mixed
// operation state without borrowing or repurposing C05-10's gated fields.
bool residual_add_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kResidualAddBf16 &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->residual_add_left != nullptr &&
         capture->residual_add_right != nullptr &&
         capture->fill_buffer != capture->residual_add_left &&
         capture->fill_buffer != capture->residual_add_right &&
         capture->residual_add_left != capture->residual_add_right &&
         capture->fill_lease_held &&
         capture->residual_add_left_lease_held &&
         capture->residual_add_right_lease_held &&
         capture->residual_add_element_count != 0 &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner, capture->residual_add_left->owner) &&
         same_context(capture->owner, capture->residual_add_right->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->residual_add_left->device_data != nullptr &&
         capture->residual_add_right->device_data != nullptr &&
         capture->residual_add_element_count <=
             capture->fill_buffer->byte_len / sizeof(__nv_bfloat16) &&
         capture->residual_add_element_count <=
             capture->residual_add_left->byte_len / sizeof(__nv_bfloat16) &&
         capture->residual_add_element_count <=
             capture->residual_add_right->byte_len / sizeof(__nv_bfloat16) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->residual_add_left->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->residual_add_right->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool residual_add_graph_state_is_valid(const RileyCudaGraph* graph) noexcept {
  return graph != nullptr &&
         graph->operation ==
             RileyCudaGraphCaptureOperation::kResidualAddBf16 &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr && graph->residual_add_left != nullptr &&
         graph->residual_add_right != nullptr &&
         graph->fill_buffer != graph->residual_add_left &&
         graph->fill_buffer != graph->residual_add_right &&
         graph->residual_add_left != graph->residual_add_right &&
         graph->residual_add_element_count != 0 &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner, graph->residual_add_left->owner) &&
         same_context(graph->owner, graph->residual_add_right->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->residual_add_left->device_data != nullptr &&
         graph->residual_add_right->device_data != nullptr &&
         graph->residual_add_element_count <=
             graph->fill_buffer->byte_len / sizeof(__nv_bfloat16) &&
         graph->residual_add_element_count <=
             graph->residual_add_left->byte_len / sizeof(__nv_bfloat16) &&
         graph->residual_add_element_count <=
             graph->residual_add_right->byte_len / sizeof(__nv_bfloat16) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->residual_add_left->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->residual_add_right->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool residual_add_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr &&
         exec->operation ==
             RileyCudaGraphCaptureOperation::kResidualAddBf16 &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr && exec->residual_add_left != nullptr &&
         exec->residual_add_right != nullptr &&
         exec->fill_buffer != exec->residual_add_left &&
         exec->fill_buffer != exec->residual_add_right &&
         exec->residual_add_left != exec->residual_add_right &&
         exec->residual_add_element_count != 0 &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner, exec->residual_add_left->owner) &&
         same_context(exec->owner, exec->residual_add_right->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->residual_add_left->device_data != nullptr &&
         exec->residual_add_right->device_data != nullptr &&
         exec->residual_add_element_count <=
             exec->fill_buffer->byte_len / sizeof(__nv_bfloat16) &&
         exec->residual_add_element_count <=
             exec->residual_add_left->byte_len / sizeof(__nv_bfloat16) &&
         exec->residual_add_element_count <=
             exec->residual_add_right->byte_len / sizeof(__nv_bfloat16) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->residual_add_left->active_uses.load(std::memory_order_acquire) ==
             1 &&
         exec->residual_add_right->active_uses.load(
             std::memory_order_acquire) == 1;
}

// Canonical RMSNorm owns three distinct BF16 allocations. It has the same
// fixed-address lease shape as residual add, but carries the exact generic
// RMSNorm geometry and epsilon so profile-specific reductions cannot enter.
bool canonical_rms_norm_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t element_count = 0;
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16 &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->canonical_rms_norm_input != nullptr &&
         capture->canonical_rms_norm_weight != nullptr &&
         capture->fill_buffer != capture->canonical_rms_norm_input &&
         capture->fill_buffer != capture->canonical_rms_norm_weight &&
         capture->canonical_rms_norm_input !=
             capture->canonical_rms_norm_weight &&
         capture->fill_lease_held &&
         capture->canonical_rms_norm_input_lease_held &&
         capture->canonical_rms_norm_weight_lease_held &&
         canonical_rms_norm_element_count(
             capture->canonical_rms_norm_row_count,
             capture->canonical_rms_norm_hidden_size, &element_count) &&
         std::isfinite(capture->canonical_rms_norm_epsilon) &&
         capture->canonical_rms_norm_epsilon > 0.0F &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner,
                      capture->canonical_rms_norm_input->owner) &&
         same_context(capture->owner,
                      capture->canonical_rms_norm_weight->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->canonical_rms_norm_input->device_data != nullptr &&
         capture->canonical_rms_norm_weight->device_data != nullptr &&
         element_count <= capture->fill_buffer->byte_len /
                              sizeof(__nv_bfloat16) &&
         element_count <= capture->canonical_rms_norm_input->byte_len /
                              sizeof(__nv_bfloat16) &&
         capture->canonical_rms_norm_hidden_size <=
             capture->canonical_rms_norm_weight->byte_len /
                 sizeof(__nv_bfloat16) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->canonical_rms_norm_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->canonical_rms_norm_weight->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool canonical_rms_norm_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t element_count = 0;
  return graph != nullptr &&
         graph->operation ==
             RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16 &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr && graph->canonical_rms_norm_input != nullptr &&
         graph->canonical_rms_norm_weight != nullptr &&
         graph->fill_buffer != graph->canonical_rms_norm_input &&
         graph->fill_buffer != graph->canonical_rms_norm_weight &&
         graph->canonical_rms_norm_input != graph->canonical_rms_norm_weight &&
         canonical_rms_norm_element_count(graph->canonical_rms_norm_row_count,
                                          graph->canonical_rms_norm_hidden_size,
                                          &element_count) &&
         std::isfinite(graph->canonical_rms_norm_epsilon) &&
         graph->canonical_rms_norm_epsilon > 0.0F &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner, graph->canonical_rms_norm_input->owner) &&
         same_context(graph->owner, graph->canonical_rms_norm_weight->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->canonical_rms_norm_input->device_data != nullptr &&
         graph->canonical_rms_norm_weight->device_data != nullptr &&
         element_count <=
             graph->fill_buffer->byte_len / sizeof(__nv_bfloat16) &&
         element_count <= graph->canonical_rms_norm_input->byte_len /
                              sizeof(__nv_bfloat16) &&
         graph->canonical_rms_norm_hidden_size <=
             graph->canonical_rms_norm_weight->byte_len /
                 sizeof(__nv_bfloat16) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->canonical_rms_norm_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->canonical_rms_norm_weight->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool canonical_rms_norm_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t element_count = 0;
  return exec != nullptr &&
         exec->operation ==
             RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16 &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr && exec->canonical_rms_norm_input != nullptr &&
         exec->canonical_rms_norm_weight != nullptr &&
         exec->fill_buffer != exec->canonical_rms_norm_input &&
         exec->fill_buffer != exec->canonical_rms_norm_weight &&
         exec->canonical_rms_norm_input != exec->canonical_rms_norm_weight &&
         canonical_rms_norm_element_count(exec->canonical_rms_norm_row_count,
                                          exec->canonical_rms_norm_hidden_size,
                                          &element_count) &&
         std::isfinite(exec->canonical_rms_norm_epsilon) &&
         exec->canonical_rms_norm_epsilon > 0.0F &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner, exec->canonical_rms_norm_input->owner) &&
         same_context(exec->owner, exec->canonical_rms_norm_weight->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->canonical_rms_norm_input->device_data != nullptr &&
         exec->canonical_rms_norm_weight->device_data != nullptr &&
         element_count <= exec->fill_buffer->byte_len /
                              sizeof(__nv_bfloat16) &&
         element_count <= exec->canonical_rms_norm_input->byte_len /
                              sizeof(__nv_bfloat16) &&
         exec->canonical_rms_norm_hidden_size <=
             exec->canonical_rms_norm_weight->byte_len /
                 sizeof(__nv_bfloat16) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->canonical_rms_norm_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->canonical_rms_norm_weight->active_uses.load(
             std::memory_order_acquire) == 1;
}

// C05-13 owns exactly one BF16 logits allocation and one distinct U32 result
// allocation. The vocabulary bound preserves the eager U32 token-id ABI, and
// capacity checks avoid materializing byte products that could overflow.
bool bf16_argmax_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t logit_element_count = 0;
  return capture != nullptr &&
         capture->operation == RileyCudaGraphCaptureOperation::kBf16Argmax &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->bf16_argmax_logits != nullptr &&
         capture->fill_buffer != capture->bf16_argmax_logits &&
         capture->fill_lease_held &&
         capture->bf16_argmax_logits_lease_held &&
         bf16_argmax_shape_is_valid(capture->bf16_argmax_row_count,
                                    capture->bf16_argmax_vocabulary_size,
                                    &logit_element_count) &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner, capture->bf16_argmax_logits->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->bf16_argmax_logits->device_data != nullptr &&
         capture->bf16_argmax_row_count <=
             capture->fill_buffer->byte_len / sizeof(RileyCudaBf16ArgmaxResult) &&
         logit_element_count <= capture->bf16_argmax_logits->byte_len /
                                    sizeof(__nv_bfloat16) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->bf16_argmax_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_argmax_graph_state_is_valid(const RileyCudaGraph* graph) noexcept {
  uint64_t logit_element_count = 0;
  return graph != nullptr &&
         graph->operation == RileyCudaGraphCaptureOperation::kBf16Argmax &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr && graph->bf16_argmax_logits != nullptr &&
         graph->fill_buffer != graph->bf16_argmax_logits &&
         bf16_argmax_shape_is_valid(graph->bf16_argmax_row_count,
                                    graph->bf16_argmax_vocabulary_size,
                                    &logit_element_count) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner, graph->bf16_argmax_logits->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->bf16_argmax_logits->device_data != nullptr &&
         graph->bf16_argmax_row_count <=
             graph->fill_buffer->byte_len / sizeof(RileyCudaBf16ArgmaxResult) &&
         logit_element_count <= graph->bf16_argmax_logits->byte_len /
                                    sizeof(__nv_bfloat16) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->bf16_argmax_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_argmax_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t logit_element_count = 0;
  return exec != nullptr &&
         exec->operation == RileyCudaGraphCaptureOperation::kBf16Argmax &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr && exec->bf16_argmax_logits != nullptr &&
         exec->fill_buffer != exec->bf16_argmax_logits &&
         bf16_argmax_shape_is_valid(exec->bf16_argmax_row_count,
                                    exec->bf16_argmax_vocabulary_size,
                                    &logit_element_count) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner, exec->bf16_argmax_logits->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->bf16_argmax_logits->device_data != nullptr &&
         exec->bf16_argmax_row_count <=
             exec->fill_buffer->byte_len / sizeof(RileyCudaBf16ArgmaxResult) &&
         logit_element_count <= exec->bf16_argmax_logits->byte_len /
                                    sizeof(__nv_bfloat16) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->bf16_argmax_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

// C05-14 owns three distinct fixed device allocations: BF16 input, U32 row
// indices, and BF16 output. The native raw ABI deliberately does not retain a
// host mirror; malformed device indices remain a kernel-level NaN result.
bool bf16_row_gather_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t input_element_count = 0;
  uint64_t output_element_count = 0;
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGather &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->bf16_row_gather_input != nullptr &&
         capture->bf16_row_gather_indices != nullptr &&
         capture->fill_buffer != capture->bf16_row_gather_input &&
         capture->fill_buffer != capture->bf16_row_gather_indices &&
         capture->bf16_row_gather_input !=
             capture->bf16_row_gather_indices &&
         capture->fill_lease_held &&
         capture->bf16_row_gather_input_lease_held &&
         capture->bf16_row_gather_indices_lease_held &&
         bf16_row_gather_shape_is_valid(
             capture->bf16_row_gather_input_row_count,
             capture->bf16_row_gather_output_row_count,
             capture->bf16_row_gather_column_count, &input_element_count,
             &output_element_count) &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner, capture->bf16_row_gather_input->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_indices->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->bf16_row_gather_input->device_data != nullptr &&
         capture->bf16_row_gather_indices->device_data != nullptr &&
         input_element_count <= capture->bf16_row_gather_input->byte_len /
                                    sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_output_row_count <=
             capture->bf16_row_gather_indices->byte_len / sizeof(uint32_t) &&
         output_element_count <= capture->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->bf16_row_gather_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_indices->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t input_element_count = 0;
  uint64_t output_element_count = 0;
  return graph != nullptr &&
         graph->operation == RileyCudaGraphCaptureOperation::kBf16RowGather &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr && graph->bf16_row_gather_input != nullptr &&
         graph->bf16_row_gather_indices != nullptr &&
         graph->fill_buffer != graph->bf16_row_gather_input &&
         graph->fill_buffer != graph->bf16_row_gather_indices &&
         graph->bf16_row_gather_input != graph->bf16_row_gather_indices &&
         bf16_row_gather_shape_is_valid(
             graph->bf16_row_gather_input_row_count,
             graph->bf16_row_gather_output_row_count,
             graph->bf16_row_gather_column_count, &input_element_count,
             &output_element_count) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner, graph->bf16_row_gather_input->owner) &&
         same_context(graph->owner, graph->bf16_row_gather_indices->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->bf16_row_gather_input->device_data != nullptr &&
         graph->bf16_row_gather_indices->device_data != nullptr &&
         input_element_count <= graph->bf16_row_gather_input->byte_len /
                                    sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_output_row_count <=
             graph->bf16_row_gather_indices->byte_len / sizeof(uint32_t) &&
         output_element_count <= graph->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_indices->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t input_element_count = 0;
  uint64_t output_element_count = 0;
  return exec != nullptr &&
         exec->operation == RileyCudaGraphCaptureOperation::kBf16RowGather &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr && exec->bf16_row_gather_input != nullptr &&
         exec->bf16_row_gather_indices != nullptr &&
         exec->fill_buffer != exec->bf16_row_gather_input &&
         exec->fill_buffer != exec->bf16_row_gather_indices &&
         exec->bf16_row_gather_input != exec->bf16_row_gather_indices &&
         bf16_row_gather_shape_is_valid(
             exec->bf16_row_gather_input_row_count,
             exec->bf16_row_gather_output_row_count,
             exec->bf16_row_gather_column_count, &input_element_count,
             &output_element_count) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner, exec->bf16_row_gather_input->owner) &&
         same_context(exec->owner, exec->bf16_row_gather_indices->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->bf16_row_gather_input->device_data != nullptr &&
         exec->bf16_row_gather_indices->device_data != nullptr &&
         input_element_count <= exec->bf16_row_gather_input->byte_len /
                                    sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_output_row_count <=
             exec->bf16_row_gather_indices->byte_len / sizeof(uint32_t) &&
         output_element_count <= exec->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_indices->active_uses.load(
             std::memory_order_acquire) == 1;
}

// C05-15 records a dependent fixed-address pair: BF16 row gather writes an
// intermediate BF16 logits slab, then the exact C05-13 argmax consumes that
// slab into U32 result records. Keep all four allocation identities and their
// permanent graph leases explicit; neither stage may alias the other.
bool bf16_row_gather_argmax_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->bf16_row_gather_argmax_input != nullptr &&
         capture->bf16_row_gather_argmax_indices != nullptr &&
         capture->bf16_row_gather_argmax_gathered_logits != nullptr &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_input &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_indices &&
         capture->fill_buffer !=
             capture->bf16_row_gather_argmax_gathered_logits &&
         capture->bf16_row_gather_argmax_input !=
             capture->bf16_row_gather_argmax_indices &&
         capture->bf16_row_gather_argmax_input !=
             capture->bf16_row_gather_argmax_gathered_logits &&
         capture->bf16_row_gather_argmax_indices !=
             capture->bf16_row_gather_argmax_gathered_logits &&
         capture->fill_lease_held &&
         capture->bf16_row_gather_argmax_input_lease_held &&
         capture->bf16_row_gather_argmax_indices_lease_held &&
         capture->bf16_row_gather_argmax_gathered_logits_lease_held &&
         bf16_row_gather_argmax_shape_is_valid(
             capture->bf16_row_gather_argmax_input_row_count,
             capture->bf16_row_gather_argmax_output_row_count,
             capture->bf16_row_gather_argmax_vocabulary_size,
             &input_element_count, &gathered_element_count) &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_input->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_indices->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_gathered_logits->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->bf16_row_gather_argmax_input->device_data != nullptr &&
         capture->bf16_row_gather_argmax_indices->device_data != nullptr &&
         capture->bf16_row_gather_argmax_gathered_logits->device_data != nullptr &&
         input_element_count <=
             capture->bf16_row_gather_argmax_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_argmax_output_row_count <=
             capture->bf16_row_gather_argmax_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             capture->bf16_row_gather_argmax_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_argmax_output_row_count <=
             capture->fill_buffer->byte_len /
                 sizeof(RileyCudaBf16ArgmaxResult) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->bf16_row_gather_argmax_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_argmax_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  return graph != nullptr &&
         graph->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr &&
         graph->bf16_row_gather_argmax_input != nullptr &&
         graph->bf16_row_gather_argmax_indices != nullptr &&
         graph->bf16_row_gather_argmax_gathered_logits != nullptr &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_input &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_indices &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_gathered_logits &&
         graph->bf16_row_gather_argmax_input !=
             graph->bf16_row_gather_argmax_indices &&
         graph->bf16_row_gather_argmax_input !=
             graph->bf16_row_gather_argmax_gathered_logits &&
         graph->bf16_row_gather_argmax_indices !=
             graph->bf16_row_gather_argmax_gathered_logits &&
         bf16_row_gather_argmax_shape_is_valid(
             graph->bf16_row_gather_argmax_input_row_count,
             graph->bf16_row_gather_argmax_output_row_count,
             graph->bf16_row_gather_argmax_vocabulary_size,
             &input_element_count, &gathered_element_count) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_input->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_indices->owner) &&
         same_context(
             graph->owner,
             graph->bf16_row_gather_argmax_gathered_logits->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->bf16_row_gather_argmax_input->device_data != nullptr &&
         graph->bf16_row_gather_argmax_indices->device_data != nullptr &&
         graph->bf16_row_gather_argmax_gathered_logits->device_data != nullptr &&
         input_element_count <=
             graph->bf16_row_gather_argmax_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_argmax_output_row_count <=
             graph->bf16_row_gather_argmax_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             graph->bf16_row_gather_argmax_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_argmax_output_row_count <=
             graph->fill_buffer->byte_len /
                 sizeof(RileyCudaBf16ArgmaxResult) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_argmax_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  return exec != nullptr &&
         exec->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr &&
         exec->bf16_row_gather_argmax_input != nullptr &&
         exec->bf16_row_gather_argmax_indices != nullptr &&
         exec->bf16_row_gather_argmax_gathered_logits != nullptr &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_input &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_indices &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_gathered_logits &&
         exec->bf16_row_gather_argmax_input !=
             exec->bf16_row_gather_argmax_indices &&
         exec->bf16_row_gather_argmax_input !=
             exec->bf16_row_gather_argmax_gathered_logits &&
         exec->bf16_row_gather_argmax_indices !=
             exec->bf16_row_gather_argmax_gathered_logits &&
         bf16_row_gather_argmax_shape_is_valid(
             exec->bf16_row_gather_argmax_input_row_count,
             exec->bf16_row_gather_argmax_output_row_count,
             exec->bf16_row_gather_argmax_vocabulary_size,
             &input_element_count, &gathered_element_count) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_input->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_indices->owner) &&
         same_context(
             exec->owner,
             exec->bf16_row_gather_argmax_gathered_logits->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->bf16_row_gather_argmax_input->device_data != nullptr &&
         exec->bf16_row_gather_argmax_indices->device_data != nullptr &&
         exec->bf16_row_gather_argmax_gathered_logits->device_data != nullptr &&
         input_element_count <=
             exec->bf16_row_gather_argmax_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_argmax_output_row_count <=
             exec->bf16_row_gather_argmax_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             exec->bf16_row_gather_argmax_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_argmax_output_row_count <=
             exec->fill_buffer->byte_len /
                 sizeof(RileyCudaBf16ArgmaxResult) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1;
}

// C05-16 adds a fixed pinned result destination as the third dependency node.
// Its raw host pointer must remain exclusively leased while the graph retains
// a replayable D2H node; only the op-specific completion receipt may inspect
// its bytes after a known launch completion.
bool bf16_row_gather_argmax_d2h_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  uint64_t result_byte_len = 0;
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->bf16_row_gather_argmax_d2h_input != nullptr &&
         capture->bf16_row_gather_argmax_d2h_indices != nullptr &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         capture->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_d2h_input &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_d2h_indices &&
         capture->fill_buffer !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->bf16_row_gather_argmax_d2h_input !=
             capture->bf16_row_gather_argmax_d2h_indices &&
         capture->bf16_row_gather_argmax_d2h_input !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->bf16_row_gather_argmax_d2h_indices !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->fill_lease_held &&
         capture->bf16_row_gather_argmax_d2h_input_lease_held &&
         capture->bf16_row_gather_argmax_d2h_indices_lease_held &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held &&
         capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held &&
         bf16_row_gather_argmax_d2h_shape_is_valid(
             capture->bf16_row_gather_argmax_d2h_input_row_count,
             capture->bf16_row_gather_argmax_d2h_output_row_count,
             capture->bf16_row_gather_argmax_d2h_vocabulary_size,
             &input_element_count, &gathered_element_count, &result_byte_len) &&
         capture->bf16_row_gather_argmax_d2h_result_byte_len ==
             result_byte_len &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(
             capture->owner,
             capture->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(
             capture->owner,
             capture->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         capture->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         input_element_count <=
             capture->bf16_row_gather_argmax_d2h_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_argmax_d2h_output_row_count <=
             capture->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             capture->bf16_row_gather_argmax_d2h_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         result_byte_len <= capture->fill_buffer->byte_len &&
         result_byte_len ==
             capture->bf16_row_gather_argmax_d2h_pinned_results->byte_len &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_argmax_d2h_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  uint64_t result_byte_len = 0;
  return graph != nullptr &&
         graph->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr &&
         graph->bf16_row_gather_argmax_d2h_input != nullptr &&
         graph->bf16_row_gather_argmax_d2h_indices != nullptr &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         graph->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_input &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_indices &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         graph->bf16_row_gather_argmax_d2h_input !=
             graph->bf16_row_gather_argmax_d2h_indices &&
         graph->bf16_row_gather_argmax_d2h_input !=
             graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         graph->bf16_row_gather_argmax_d2h_indices !=
             graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         bf16_row_gather_argmax_d2h_shape_is_valid(
             graph->bf16_row_gather_argmax_d2h_input_row_count,
             graph->bf16_row_gather_argmax_d2h_output_row_count,
             graph->bf16_row_gather_argmax_d2h_vocabulary_size,
             &input_element_count, &gathered_element_count, &result_byte_len) &&
         graph->bf16_row_gather_argmax_d2h_result_byte_len == result_byte_len &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(
             graph->owner,
             graph->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(
             graph->owner,
             graph->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         graph->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         input_element_count <=
             graph->bf16_row_gather_argmax_d2h_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_argmax_d2h_output_row_count <=
             graph->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             graph->bf16_row_gather_argmax_d2h_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         result_byte_len <= graph->fill_buffer->byte_len &&
         result_byte_len ==
             graph->bf16_row_gather_argmax_d2h_pinned_results->byte_len &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_row_gather_argmax_d2h_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  uint64_t result_byte_len = 0;
  return exec != nullptr &&
         exec->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr &&
         exec->bf16_row_gather_argmax_d2h_input != nullptr &&
         exec->bf16_row_gather_argmax_d2h_indices != nullptr &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         exec->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_input &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_indices &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         exec->bf16_row_gather_argmax_d2h_input !=
             exec->bf16_row_gather_argmax_d2h_indices &&
         exec->bf16_row_gather_argmax_d2h_input !=
             exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         exec->bf16_row_gather_argmax_d2h_indices !=
             exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         bf16_row_gather_argmax_d2h_shape_is_valid(
             exec->bf16_row_gather_argmax_d2h_input_row_count,
             exec->bf16_row_gather_argmax_d2h_output_row_count,
             exec->bf16_row_gather_argmax_d2h_vocabulary_size,
             &input_element_count, &gathered_element_count, &result_byte_len) &&
         exec->bf16_row_gather_argmax_d2h_result_byte_len == result_byte_len &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(
             exec->owner,
             exec->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(
             exec->owner,
             exec->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         exec->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         input_element_count <=
             exec->bf16_row_gather_argmax_d2h_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_argmax_d2h_output_row_count <=
             exec->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         gathered_element_count <=
             exec->bf16_row_gather_argmax_d2h_gathered_logits->byte_len /
                 sizeof(__nv_bfloat16) &&
         result_byte_len <= exec->fill_buffer->byte_len &&
         result_byte_len ==
             exec->bf16_row_gather_argmax_d2h_pinned_results->byte_len &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

// C05-20 uses C05-16's storage slots only as a five-address ledger. The
// operation tag and this separate geometry predicate prevent any row-gather
// interpretation from leaking into the embedding chain:
//   fill_buffer=output, d2h_input=table, d2h_indices=token IDs,
//   d2h_gathered_logits=device error scratch, d2h_pinned_results=report.
bool bf16_embedding_status_d2h_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t table_element_count = 0;
  uint64_t output_element_count = 0;
  return capture != nullptr &&
         capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->bf16_row_gather_argmax_d2h_input != nullptr &&
         capture->bf16_row_gather_argmax_d2h_indices != nullptr &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         capture->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_d2h_input &&
         capture->fill_buffer != capture->bf16_row_gather_argmax_d2h_indices &&
         capture->fill_buffer !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->bf16_row_gather_argmax_d2h_input !=
             capture->bf16_row_gather_argmax_d2h_indices &&
         capture->bf16_row_gather_argmax_d2h_input !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->bf16_row_gather_argmax_d2h_indices !=
             capture->bf16_row_gather_argmax_d2h_gathered_logits &&
         capture->fill_lease_held &&
         capture->bf16_row_gather_argmax_d2h_input_lease_held &&
         capture->bf16_row_gather_argmax_d2h_indices_lease_held &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held &&
         capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held &&
         bf16_embedding_status_d2h_shape_is_valid(
             capture->bf16_row_gather_argmax_d2h_input_row_count,
             capture->bf16_row_gather_argmax_d2h_output_row_count,
             capture->bf16_row_gather_argmax_d2h_vocabulary_size,
             &table_element_count, &output_element_count) &&
         capture->bf16_row_gather_argmax_d2h_result_byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(capture->owner,
                      capture->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         capture->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         table_element_count <=
             capture->bf16_row_gather_argmax_d2h_input->byte_len /
                 sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_argmax_d2h_output_row_count <=
             capture->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         output_element_count <= capture->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         reinterpret_cast<uintptr_t>(
             capture->bf16_row_gather_argmax_d2h_gathered_logits->device_data) %
                 alignof(RileyCudaEmbeddingErrorReport) ==
             0 &&
         capture->bf16_row_gather_argmax_d2h_pinned_results->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) ==
             1 &&
         capture->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_embedding_status_d2h_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t table_element_count = 0;
  uint64_t output_element_count = 0;
  return graph != nullptr &&
         graph->operation ==
             RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr &&
         graph->bf16_row_gather_argmax_d2h_input != nullptr &&
         graph->bf16_row_gather_argmax_d2h_indices != nullptr &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         graph->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_input &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_indices &&
         graph->fill_buffer != graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         graph->bf16_row_gather_argmax_d2h_input !=
             graph->bf16_row_gather_argmax_d2h_indices &&
         graph->bf16_row_gather_argmax_d2h_input !=
             graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         graph->bf16_row_gather_argmax_d2h_indices !=
             graph->bf16_row_gather_argmax_d2h_gathered_logits &&
         bf16_embedding_status_d2h_shape_is_valid(
             graph->bf16_row_gather_argmax_d2h_input_row_count,
             graph->bf16_row_gather_argmax_d2h_output_row_count,
             graph->bf16_row_gather_argmax_d2h_vocabulary_size,
             &table_element_count, &output_element_count) &&
         graph->bf16_row_gather_argmax_d2h_result_byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(graph->owner,
                      graph->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         graph->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         table_element_count <= graph->bf16_row_gather_argmax_d2h_input->byte_len /
                                    sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_argmax_d2h_output_row_count <=
             graph->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         output_element_count <= graph->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         reinterpret_cast<uintptr_t>(
             graph->bf16_row_gather_argmax_d2h_gathered_logits->device_data) %
                 alignof(RileyCudaEmbeddingErrorReport) ==
             0 &&
         graph->bf16_row_gather_argmax_d2h_pinned_results->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_embedding_status_d2h_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t table_element_count = 0;
  uint64_t output_element_count = 0;
  return exec != nullptr &&
         exec->operation ==
             RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr &&
         exec->bf16_row_gather_argmax_d2h_input != nullptr &&
         exec->bf16_row_gather_argmax_d2h_indices != nullptr &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits != nullptr &&
         exec->bf16_row_gather_argmax_d2h_pinned_results != nullptr &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_input &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_indices &&
         exec->fill_buffer != exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         exec->bf16_row_gather_argmax_d2h_input !=
             exec->bf16_row_gather_argmax_d2h_indices &&
         exec->bf16_row_gather_argmax_d2h_input !=
             exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         exec->bf16_row_gather_argmax_d2h_indices !=
             exec->bf16_row_gather_argmax_d2h_gathered_logits &&
         bf16_embedding_status_d2h_shape_is_valid(
             exec->bf16_row_gather_argmax_d2h_input_row_count,
             exec->bf16_row_gather_argmax_d2h_output_row_count,
             exec->bf16_row_gather_argmax_d2h_vocabulary_size,
             &table_element_count, &output_element_count) &&
         exec->bf16_row_gather_argmax_d2h_result_byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_input->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_indices->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_gathered_logits->owner) &&
         same_context(exec->owner,
                      exec->bf16_row_gather_argmax_d2h_pinned_results->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_input->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_indices->device_data != nullptr &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits->device_data !=
             nullptr &&
         exec->bf16_row_gather_argmax_d2h_pinned_results->host_data !=
             nullptr &&
         table_element_count <= exec->bf16_row_gather_argmax_d2h_input->byte_len /
                                    sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_argmax_d2h_output_row_count <=
             exec->bf16_row_gather_argmax_d2h_indices->byte_len /
                 sizeof(uint32_t) &&
         output_element_count <= exec->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         reinterpret_cast<uintptr_t>(
             exec->bf16_row_gather_argmax_d2h_gathered_logits->device_data) %
                 alignof(RileyCudaEmbeddingErrorReport) ==
             0 &&
         exec->bf16_row_gather_argmax_d2h_pinned_results->byte_len ==
             sizeof(RileyCudaEmbeddingErrorReport) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_indices->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool bf16_d2h_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         (capture->operation ==
                  RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H
              ? bf16_row_gather_argmax_d2h_capture_state_is_valid(capture)
              : capture->operation ==
                        RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
                    bf16_embedding_status_d2h_capture_state_is_valid(capture));
}

bool bf16_d2h_graph_state_is_valid(const RileyCudaGraph* graph) noexcept {
  return graph != nullptr &&
         (graph->operation ==
                  RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H
              ? bf16_row_gather_argmax_d2h_graph_state_is_valid(graph)
              : graph->operation ==
                        RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
                    bf16_embedding_status_d2h_graph_state_is_valid(graph));
}

bool bf16_d2h_exec_state_is_valid(const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr &&
         (exec->operation == RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H
              ? bf16_row_gather_argmax_d2h_exec_state_is_valid(exec)
              : exec->operation ==
                        RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H &&
                    bf16_embedding_status_d2h_exec_state_is_valid(exec));
}

// C05-17 owns five distinct fixed device allocations. The host position
// mirror is intentionally absent from owner state: it has completed its
// begin-time validation before any permanent lease or CUDA capture exists.
bool indexed_rope_bf16_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  uint64_t tensor_element_count = 0;
  uint64_t table_element_count = 0;
  uint64_t work_item_count = 0;
  return capture != nullptr &&
         capture->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16 &&
         capture->owner != nullptr && capture->stream != nullptr &&
         capture->fill_buffer != nullptr &&
         capture->indexed_rope_bf16_input != nullptr &&
         capture->indexed_rope_bf16_cos != nullptr &&
         capture->indexed_rope_bf16_sin != nullptr &&
         capture->indexed_rope_bf16_positions != nullptr &&
         capture->fill_buffer != capture->indexed_rope_bf16_input &&
         capture->fill_buffer != capture->indexed_rope_bf16_cos &&
         capture->fill_buffer != capture->indexed_rope_bf16_sin &&
         capture->fill_buffer != capture->indexed_rope_bf16_positions &&
         capture->indexed_rope_bf16_input != capture->indexed_rope_bf16_cos &&
         capture->indexed_rope_bf16_input != capture->indexed_rope_bf16_sin &&
         capture->indexed_rope_bf16_input != capture->indexed_rope_bf16_positions &&
         capture->indexed_rope_bf16_cos != capture->indexed_rope_bf16_sin &&
         capture->indexed_rope_bf16_cos != capture->indexed_rope_bf16_positions &&
         capture->indexed_rope_bf16_sin != capture->indexed_rope_bf16_positions &&
         capture->fill_lease_held &&
         capture->indexed_rope_bf16_input_lease_held &&
         capture->indexed_rope_bf16_cos_lease_held &&
         capture->indexed_rope_bf16_sin_lease_held &&
         capture->indexed_rope_bf16_positions_lease_held &&
         indexed_rope_bf16_shape_is_valid(
             capture->indexed_rope_bf16_active_row_count,
             capture->indexed_rope_bf16_head_count,
             capture->indexed_rope_bf16_head_size,
             capture->indexed_rope_bf16_rotary_dimension,
             capture->indexed_rope_bf16_table_position_count,
             &tensor_element_count, &table_element_count, &work_item_count) &&
         capture->fill_element_count == 0 && capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 &&
         !capture->h2d_source_lease_held && capture->silu_input == nullptr &&
         capture->silu_element_count == 0 && capture->silu_enqueue_count == 0 &&
         !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         same_context(capture->owner, capture->stream->owner) &&
         same_context(capture->owner, capture->fill_buffer->owner) &&
         same_context(capture->owner, capture->indexed_rope_bf16_input->owner) &&
         same_context(capture->owner, capture->indexed_rope_bf16_cos->owner) &&
         same_context(capture->owner, capture->indexed_rope_bf16_sin->owner) &&
         same_context(capture->owner, capture->indexed_rope_bf16_positions->owner) &&
         capture->fill_buffer->device_data != nullptr &&
         capture->indexed_rope_bf16_input->device_data != nullptr &&
         capture->indexed_rope_bf16_cos->device_data != nullptr &&
         capture->indexed_rope_bf16_sin->device_data != nullptr &&
         capture->indexed_rope_bf16_positions->device_data != nullptr &&
         tensor_element_count <= capture->indexed_rope_bf16_input->byte_len /
                                     sizeof(__nv_bfloat16) &&
         tensor_element_count <= capture->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         table_element_count <= capture->indexed_rope_bf16_cos->byte_len /
                                    sizeof(float) &&
         table_element_count <= capture->indexed_rope_bf16_sin->byte_len /
                                    sizeof(float) &&
         capture->indexed_rope_bf16_active_row_count <=
             capture->indexed_rope_bf16_positions->byte_len / sizeof(uint32_t) &&
         capture->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         capture->indexed_rope_bf16_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->indexed_rope_bf16_cos->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->indexed_rope_bf16_sin->active_uses.load(
             std::memory_order_acquire) == 1 &&
         capture->indexed_rope_bf16_positions->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool indexed_rope_bf16_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  uint64_t tensor_element_count = 0;
  uint64_t table_element_count = 0;
  uint64_t work_item_count = 0;
  return graph != nullptr &&
         graph->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16 &&
         graph->owner != nullptr && graph->stream != nullptr &&
         graph->fill_buffer != nullptr && graph->indexed_rope_bf16_input != nullptr &&
         graph->indexed_rope_bf16_cos != nullptr &&
         graph->indexed_rope_bf16_sin != nullptr &&
         graph->indexed_rope_bf16_positions != nullptr &&
         graph->fill_buffer != graph->indexed_rope_bf16_input &&
         graph->fill_buffer != graph->indexed_rope_bf16_cos &&
         graph->fill_buffer != graph->indexed_rope_bf16_sin &&
         graph->fill_buffer != graph->indexed_rope_bf16_positions &&
         graph->indexed_rope_bf16_input != graph->indexed_rope_bf16_cos &&
         graph->indexed_rope_bf16_input != graph->indexed_rope_bf16_sin &&
         graph->indexed_rope_bf16_input != graph->indexed_rope_bf16_positions &&
         graph->indexed_rope_bf16_cos != graph->indexed_rope_bf16_sin &&
         graph->indexed_rope_bf16_cos != graph->indexed_rope_bf16_positions &&
         graph->indexed_rope_bf16_sin != graph->indexed_rope_bf16_positions &&
         indexed_rope_bf16_shape_is_valid(
             graph->indexed_rope_bf16_active_row_count,
             graph->indexed_rope_bf16_head_count,
             graph->indexed_rope_bf16_head_size,
             graph->indexed_rope_bf16_rotary_dimension,
             graph->indexed_rope_bf16_table_position_count,
             &tensor_element_count, &table_element_count, &work_item_count) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         same_context(graph->owner, graph->stream->owner) &&
         same_context(graph->owner, graph->fill_buffer->owner) &&
         same_context(graph->owner, graph->indexed_rope_bf16_input->owner) &&
         same_context(graph->owner, graph->indexed_rope_bf16_cos->owner) &&
         same_context(graph->owner, graph->indexed_rope_bf16_sin->owner) &&
         same_context(graph->owner, graph->indexed_rope_bf16_positions->owner) &&
         graph->fill_buffer->device_data != nullptr &&
         graph->indexed_rope_bf16_input->device_data != nullptr &&
         graph->indexed_rope_bf16_cos->device_data != nullptr &&
         graph->indexed_rope_bf16_sin->device_data != nullptr &&
         graph->indexed_rope_bf16_positions->device_data != nullptr &&
         tensor_element_count <= graph->indexed_rope_bf16_input->byte_len /
                                     sizeof(__nv_bfloat16) &&
         tensor_element_count <= graph->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         table_element_count <= graph->indexed_rope_bf16_cos->byte_len /
                                    sizeof(float) &&
         table_element_count <= graph->indexed_rope_bf16_sin->byte_len /
                                    sizeof(float) &&
         graph->indexed_rope_bf16_active_row_count <=
             graph->indexed_rope_bf16_positions->byte_len / sizeof(uint32_t) &&
         graph->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         graph->indexed_rope_bf16_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->indexed_rope_bf16_cos->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->indexed_rope_bf16_sin->active_uses.load(
             std::memory_order_acquire) == 1 &&
         graph->indexed_rope_bf16_positions->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool indexed_rope_bf16_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  uint64_t tensor_element_count = 0;
  uint64_t table_element_count = 0;
  uint64_t work_item_count = 0;
  return exec != nullptr &&
         exec->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16 &&
         exec->owner != nullptr && exec->stream != nullptr &&
         exec->fill_buffer != nullptr && exec->indexed_rope_bf16_input != nullptr &&
         exec->indexed_rope_bf16_cos != nullptr &&
         exec->indexed_rope_bf16_sin != nullptr &&
         exec->indexed_rope_bf16_positions != nullptr &&
         exec->fill_buffer != exec->indexed_rope_bf16_input &&
         exec->fill_buffer != exec->indexed_rope_bf16_cos &&
         exec->fill_buffer != exec->indexed_rope_bf16_sin &&
         exec->fill_buffer != exec->indexed_rope_bf16_positions &&
         exec->indexed_rope_bf16_input != exec->indexed_rope_bf16_cos &&
         exec->indexed_rope_bf16_input != exec->indexed_rope_bf16_sin &&
         exec->indexed_rope_bf16_input != exec->indexed_rope_bf16_positions &&
         exec->indexed_rope_bf16_cos != exec->indexed_rope_bf16_sin &&
         exec->indexed_rope_bf16_cos != exec->indexed_rope_bf16_positions &&
         exec->indexed_rope_bf16_sin != exec->indexed_rope_bf16_positions &&
         indexed_rope_bf16_shape_is_valid(
             exec->indexed_rope_bf16_active_row_count,
             exec->indexed_rope_bf16_head_count,
             exec->indexed_rope_bf16_head_size,
             exec->indexed_rope_bf16_rotary_dimension,
             exec->indexed_rope_bf16_table_position_count,
             &tensor_element_count, &table_element_count, &work_item_count) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         same_context(exec->owner, exec->stream->owner) &&
         same_context(exec->owner, exec->fill_buffer->owner) &&
         same_context(exec->owner, exec->indexed_rope_bf16_input->owner) &&
         same_context(exec->owner, exec->indexed_rope_bf16_cos->owner) &&
         same_context(exec->owner, exec->indexed_rope_bf16_sin->owner) &&
         same_context(exec->owner, exec->indexed_rope_bf16_positions->owner) &&
         exec->fill_buffer->device_data != nullptr &&
         exec->indexed_rope_bf16_input->device_data != nullptr &&
         exec->indexed_rope_bf16_cos->device_data != nullptr &&
         exec->indexed_rope_bf16_sin->device_data != nullptr &&
         exec->indexed_rope_bf16_positions->device_data != nullptr &&
         tensor_element_count <= exec->indexed_rope_bf16_input->byte_len /
                                     sizeof(__nv_bfloat16) &&
         tensor_element_count <= exec->fill_buffer->byte_len /
                                     sizeof(__nv_bfloat16) &&
         table_element_count <= exec->indexed_rope_bf16_cos->byte_len /
                                    sizeof(float) &&
         table_element_count <= exec->indexed_rope_bf16_sin->byte_len /
                                    sizeof(float) &&
         exec->indexed_rope_bf16_active_row_count <=
             exec->indexed_rope_bf16_positions->byte_len / sizeof(uint32_t) &&
         exec->stream->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->fill_buffer->active_uses.load(std::memory_order_acquire) == 1 &&
         exec->indexed_rope_bf16_input->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->indexed_rope_bf16_cos->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->indexed_rope_bf16_sin->active_uses.load(
             std::memory_order_acquire) == 1 &&
         exec->indexed_rope_bf16_positions->active_uses.load(
             std::memory_order_acquire) == 1;
}

bool ragged_paged_kv_cache_write_bf16_state_has_expected_leases(
    const RileyCudaRaggedPagedKvCacheWriteBf16State& state,
    bool held) noexcept {
  return state.key_source_lease_held == held &&
         state.value_source_lease_held == held &&
         state.value_pool_lease_held == held &&
         state.sequence_block_offsets_lease_held == held &&
         state.block_ids_lease_held == held &&
         state.valid_tokens_lease_held == held &&
         state.row_sequence_slots_lease_held == held &&
         state.row_positions_lease_held == held;
}

bool ragged_paged_kv_cache_write_bf16_state_geometry_matches(
    const RileyCudaRaggedPagedKvCacheWriteBf16State& left,
    const RileyCudaRaggedPagedKvCacheWriteBf16State& right) noexcept {
  return left.key_source == right.key_source &&
         left.value_source == right.value_source &&
         left.key_pool == right.key_pool && left.value_pool == right.value_pool &&
         left.sequence_block_offsets == right.sequence_block_offsets &&
         left.block_ids == right.block_ids && left.valid_tokens == right.valid_tokens &&
         left.row_sequence_slots == right.row_sequence_slots &&
         left.row_positions == right.row_positions &&
         left.sequence_count == right.sequence_count &&
         left.block_count == right.block_count &&
         left.active_row_count == right.active_row_count &&
         left.physical_block_count == right.physical_block_count &&
         left.key_value_head_count == right.key_value_head_count &&
         left.head_size == right.head_size &&
         left.query_head_count == right.query_head_count &&
         left.output_row_count == right.output_row_count &&
         left.attention_scale == right.attention_scale &&
         left.grouped_attention == right.grouped_attention;
}

bool ragged_paged_kv_cache_write_bf16_state_is_valid_common(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* fill_buffer,
    const RileyCudaRaggedPagedKvCacheWriteBf16State& state,
    bool leases_held) noexcept {
  uint64_t source_element_count = 0;
  uint64_t pool_element_count = 0;
  uint64_t output_element_count = 0;
  const bool is_attention = state.grouped_attention;
  const bool shape_is_valid =
      is_attention
          ? grouped_ragged_paged_attention_bf16_shape_is_valid(
                state.sequence_count, state.block_count,
                state.active_row_count, state.physical_block_count,
                state.query_head_count, state.key_value_head_count,
                state.output_row_count, state.attention_scale,
                &source_element_count, &pool_element_count,
                &output_element_count)
          : ragged_paged_kv_cache_write_bf16_shape_is_valid(
                state.sequence_count, state.block_count,
                state.active_row_count, state.physical_block_count,
                state.key_value_head_count, state.head_size,
                &source_element_count, &pool_element_count);
  if (owner == nullptr || stream == nullptr || fill_buffer == nullptr ||
      state.key_pool != fill_buffer ||
      (!is_attention &&
       (state.query_head_count != 0 || state.output_row_count != 0 ||
        state.attention_scale != 0.0F)) ||
      (is_attention && state.head_size != kGraphRaggedAttentionHeadSize) ||
      !shape_is_valid ||
      !ragged_paged_kv_cache_write_bf16_state_has_expected_leases(
          state, leases_held)) {
    return false;
  }
  RileyCudaDeviceBuffer* const buffers[] = {
      state.key_source,           state.value_source,
      state.key_pool,             state.value_pool,
      state.sequence_block_offsets, state.block_ids,
      state.valid_tokens,         state.row_sequence_slots,
      state.row_positions,
  };
  constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  if (!same_context(owner, stream->owner)) {
    return false;
  }
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index] == nullptr || buffers[index]->owner == nullptr ||
        buffers[index]->device_data == nullptr ||
        !same_context(owner, buffers[index]->owner) ||
        buffers[index]->active_uses.load(std::memory_order_acquire) != 1) {
      return false;
    }
    for (size_t other = index + 1; other < kBufferCount; ++other) {
      if (buffers[index] == buffers[other]) {
        return false;
      }
    }
  }
  if (stream->active_uses.load(std::memory_order_acquire) != 1 ||
      source_element_count >
          state.key_source->byte_len / sizeof(__nv_bfloat16) ||
      (is_attention
           ? (pool_element_count >
                  state.value_source->byte_len / sizeof(__nv_bfloat16) ||
              output_element_count >
                  state.key_pool->byte_len / sizeof(__nv_bfloat16))
           : (source_element_count >
                  state.value_source->byte_len / sizeof(__nv_bfloat16) ||
              pool_element_count >
                  state.key_pool->byte_len / sizeof(__nv_bfloat16))) ||
      pool_element_count >
          state.value_pool->byte_len / sizeof(__nv_bfloat16) ||
      state.sequence_count + 1 >
          state.sequence_block_offsets->byte_len / sizeof(uint32_t) ||
      state.block_count > state.block_ids->byte_len / sizeof(uint32_t) ||
      state.block_count > state.valid_tokens->byte_len / sizeof(uint16_t) ||
      state.active_row_count >
          state.row_sequence_slots->byte_len / sizeof(uint32_t) ||
      state.active_row_count > state.row_positions->byte_len / sizeof(uint32_t)) {
    return false;
  }
  return true;
}

bool is_ragged_paged_fixed_address_operation(
    RileyCudaGraphCaptureOperation operation) noexcept {
  return operation ==
             RileyCudaGraphCaptureOperation::kRaggedPagedKvCacheWriteBf16 ||
         operation == RileyCudaGraphCaptureOperation::
                          kGroupedRaggedPagedAttentionBf16;
}

bool ragged_paged_state_operation_matches(
    RileyCudaGraphCaptureOperation operation,
    const RileyCudaRaggedPagedKvCacheWriteBf16State& state) noexcept {
  return (operation ==
              RileyCudaGraphCaptureOperation::kRaggedPagedKvCacheWriteBf16 &&
          !state.grouped_attention) ||
         (operation == RileyCudaGraphCaptureOperation::
                           kGroupedRaggedPagedAttentionBf16 &&
          state.grouped_attention);
}

bool ragged_paged_kv_cache_write_bf16_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         ragged_paged_state_operation_matches(
             capture->operation, capture->ragged_paged_kv_write_bf16) &&
         capture->fill_lease_held && capture->fill_element_count == 0 &&
         capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 && !capture->h2d_source_lease_held &&
         capture->silu_input == nullptr && capture->silu_element_count == 0 &&
         capture->silu_enqueue_count == 0 && !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         indexed_rope_bf16_capture_fields_are_clear(capture) &&
         ragged_paged_kv_cache_write_bf16_state_is_valid_common(
             capture->owner, capture->stream, capture->fill_buffer,
             capture->ragged_paged_kv_write_bf16, true);
}

bool ragged_paged_kv_cache_write_bf16_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr &&
         ragged_paged_state_operation_matches(
             graph->operation, graph->ragged_paged_kv_write_bf16) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         indexed_rope_bf16_graph_fields_are_clear(graph) &&
         ragged_paged_kv_cache_write_bf16_state_is_valid_common(
             graph->owner, graph->stream, graph->fill_buffer,
             graph->ragged_paged_kv_write_bf16, true);
}

// The preallocated graph wrapper exists while the capture owns every lease.
// It mirrors the immutable pointers and geometry but deliberately carries no
// extra lease bits until successful capture end transfers ownership.
bool ragged_paged_kv_cache_write_bf16_prepared_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr &&
         ragged_paged_state_operation_matches(
             graph->operation, graph->ragged_paged_kv_write_bf16) &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         indexed_rope_bf16_graph_fields_are_clear(graph) &&
         ragged_paged_kv_cache_write_bf16_state_is_valid_common(
             graph->owner, graph->stream, graph->fill_buffer,
             graph->ragged_paged_kv_write_bf16, false);
}

bool ragged_paged_kv_cache_write_bf16_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr &&
         ragged_paged_state_operation_matches(
             exec->operation, exec->ragged_paged_kv_write_bf16) &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         indexed_rope_bf16_exec_fields_are_clear(exec) &&
         ragged_paged_kv_cache_write_bf16_state_is_valid_common(
             exec->owner, exec->stream, exec->fill_buffer,
             exec->ragged_paged_kv_write_bf16, true);
}

bool canonical_gemm_bf16_state_matches(
    const RileyCudaCanonicalGemmBf16GraphState& left,
    const RileyCudaCanonicalGemmBf16GraphState& right) noexcept {
  return left.plan == right.plan && left.input == right.input &&
         left.weight == right.weight && left.workspace == right.workspace &&
         left.input_byte_len == right.input_byte_len &&
         left.weight_byte_len == right.weight_byte_len &&
         left.output_byte_len == right.output_byte_len &&
         left.workspace_byte_len == right.workspace_byte_len;
}

bool canonical_gemm_bf16_capture_state_is_valid(
    const RileyCudaGraphCapture* capture) noexcept {
  return capture != nullptr &&
         is_canonical_gemm_bf16_operation(capture->operation) &&
         capture->fill_lease_held && capture->fill_element_count == 0 &&
         capture->fill_enqueue_count == 0 &&
         capture->h2d_source == nullptr && capture->h2d_byte_len == 0 &&
         capture->h2d_enqueue_count == 0 && !capture->h2d_source_lease_held &&
         capture->silu_input == nullptr && capture->silu_element_count == 0 &&
         capture->silu_enqueue_count == 0 && !capture->silu_input_lease_held &&
         capture->gated_multiply_activated_gate == nullptr &&
         capture->gated_multiply_up == nullptr &&
         capture->gated_multiply_element_count == 0 &&
         capture->gated_multiply_enqueue_count == 0 &&
         !capture->gated_multiply_activated_gate_lease_held &&
         !capture->gated_multiply_up_lease_held &&
         residual_add_capture_fields_are_clear(capture) &&
         canonical_rms_norm_capture_fields_are_clear(capture) &&
         bf16_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_capture_fields_are_clear(capture) &&
         bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) &&
         indexed_rope_bf16_capture_fields_are_clear(capture) &&
         ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(capture) &&
         riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
             capture->owner, capture->stream, capture->fill_buffer,
             capture->canonical_gemm_bf16, true);
}

bool canonical_gemm_bf16_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && is_canonical_gemm_bf16_operation(graph->operation) &&
         graph->canonical_gemm_bf16.enqueue_count == 1 &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         indexed_rope_bf16_graph_fields_are_clear(graph) &&
         ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(graph) &&
         riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
             graph->owner, graph->stream, graph->fill_buffer,
             graph->canonical_gemm_bf16, true);
}

// The preallocated graph wrapper mirrors the immutable GEMM plan and buffer
// identity, while capture remains the sole owner of all permanent leases.
bool canonical_gemm_bf16_prepared_graph_state_is_valid(
    const RileyCudaGraph* graph) noexcept {
  return graph != nullptr && is_canonical_gemm_bf16_operation(graph->operation) &&
         graph->canonical_gemm_bf16.enqueue_count == 0 &&
         graph->h2d_source == nullptr && graph->h2d_byte_len == 0 &&
         graph->silu_input == nullptr && graph->silu_element_count == 0 &&
         graph->gated_multiply_activated_gate == nullptr &&
         graph->gated_multiply_up == nullptr &&
         graph->gated_multiply_element_count == 0 &&
         residual_add_graph_fields_are_clear(graph) &&
         canonical_rms_norm_graph_fields_are_clear(graph) &&
         bf16_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_graph_fields_are_clear(graph) &&
         bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph) &&
         indexed_rope_bf16_graph_fields_are_clear(graph) &&
         ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(graph) &&
         riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
             graph->owner, graph->stream, graph->fill_buffer,
             graph->canonical_gemm_bf16, false);
}

bool canonical_gemm_bf16_exec_state_is_valid(
    const RileyCudaGraphExec* exec) noexcept {
  return exec != nullptr && is_canonical_gemm_bf16_operation(exec->operation) &&
         exec->canonical_gemm_bf16.enqueue_count == 1 &&
         exec->h2d_source == nullptr && exec->h2d_byte_len == 0 &&
         !exec->h2d_input_staged && exec->silu_input == nullptr &&
         exec->silu_element_count == 0 &&
         exec->gated_multiply_activated_gate == nullptr &&
         exec->gated_multiply_up == nullptr &&
         exec->gated_multiply_element_count == 0 &&
         residual_add_exec_fields_are_clear(exec) &&
         canonical_rms_norm_exec_fields_are_clear(exec) &&
         bf16_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_exec_fields_are_clear(exec) &&
         bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) &&
         indexed_rope_bf16_exec_fields_are_clear(exec) &&
         ragged_paged_kv_cache_write_bf16_exec_fields_are_clear(exec) &&
         riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
             exec->owner, exec->stream, exec->fill_buffer,
             exec->canonical_gemm_bf16, true);
}

bool graph_error_is_compatible(const RileyCudaGraphErrorInfo* error) noexcept {
  return error == nullptr || error->struct_size >= sizeof(*error);
}

bool graph_error_reserved_is_zero(
    const RileyCudaGraphErrorInfo* error) noexcept {
  if (error == nullptr) {
    return true;
  }
  if (error->reserved0 != 0) {
    return false;
  }
  for (size_t index = 0; index < 3; ++index) {
    if (error->reserved[index] != 0) {
      return false;
    }
  }
  return true;
}

void clear_graph_error(RileyCudaGraphErrorInfo* error,
                       RileyCudaGraphStage stage) noexcept {
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  const uint32_t struct_size = error->struct_size;
  std::memset(error, 0, sizeof(*error));
  error->struct_size = struct_size;
  error->graph_stage = stage;
}

void record_graph_outcome(RileyCudaGraphErrorInfo* error,
                          RileyCudaGraphStage stage, uint64_t capture_id,
                          uint64_t exec_id, bool submission_started,
                          bool completion_known,
                          bool resource_release_known,
                          bool poisoned) noexcept {
  clear_graph_error(error, stage);
  if (error == nullptr || error->struct_size < sizeof(*error)) {
    return;
  }
  error->capture_id = capture_id;
  error->exec_id = exec_id;
  error->submission_started = submission_started ? 1 : 0;
  error->completion_known = completion_known ? 1 : 0;
  error->resource_release_known = resource_release_known ? 1 : 0;
  error->poisoned = poisoned ? 1 : 0;
}

void record_capture_outcome(RileyCudaGraphErrorInfo* error,
                            RileyCudaGraphStage stage, uint64_t capture_id,
                            bool resource_release_known,
                            bool poisoned) noexcept {
  record_graph_outcome(error, stage, capture_id, 0, false, false,
                       resource_release_known, poisoned);
}

// This wrapper is released only after every native side effect is known. Keep
// the thread-local gate published until the child and stream leases have both
// released; a failed release leaves the owner published and fail-closed.
bool release_capture_owner(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->owner == nullptr ||
      capture->stream == nullptr || capture->capture_domain == nullptr ||
      capture->prepared_graph != nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kNone ||
      capture->fill_buffer != nullptr || capture->fill_element_count != 0 ||
      capture->fill_enqueue_count != 0 || capture->fill_lease_held ||
      capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
      capture->h2d_enqueue_count != 0 || capture->h2d_source_lease_held ||
      capture->silu_input != nullptr || capture->silu_element_count != 0 ||
      capture->silu_enqueue_count != 0 || capture->silu_input_lease_held ||
      capture->gated_multiply_activated_gate != nullptr ||
      capture->gated_multiply_up != nullptr ||
      capture->gated_multiply_element_count != 0 ||
      capture->gated_multiply_enqueue_count != 0 ||
      capture->gated_multiply_activated_gate_lease_held ||
      capture->gated_multiply_up_lease_held ||
      !residual_add_capture_fields_are_clear(capture) ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      !indexed_rope_bf16_capture_fields_are_clear(capture) ||
      !canonical_gemm_bf16_capture_fields_are_clear(capture) ||
      !ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(capture) ||
      capture->unreleased_graph != nullptr ||
      capture->deferred_close_head != nullptr ||
      capture->deferred_close_tail != nullptr) {
    return false;
  }
  RileyCudaContext* const owner = capture->owner;
  RileyCudaStream* const stream = capture->stream;
  if (!release_child(owner)) {
    return false;
  }
  if (!release_exclusive_use(stream->active_uses)) {
    return false;
  }
  if (!release_capture_domain_capture(capture->capture_domain)) {
    return false;
  }
  if (!clear_thread_graph_capture_owner(capture)) {
    return false;
  }
  capture->~RileyCudaGraphCapture();
  std::free(capture);
  return true;
}

// C05-5 reserves one existing device buffer before capture begins. It is not
// an asynchronous operation token: the exact address stays leased at one
// through captured-graph and graph-exec ownership, then returns to zero only
// after a known abort or graph close. Keep this tiny helper separate from the
// generic capture-owner release so C05-4 owners retain their original layout.
bool release_capture_fill_lease(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr) {
    return false;
  }
  if (capture->gated_multiply_activated_gate != nullptr ||
      capture->gated_multiply_up != nullptr ||
      capture->gated_multiply_element_count != 0 ||
      capture->gated_multiply_enqueue_count != 0 ||
      capture->gated_multiply_activated_gate_lease_held ||
      capture->gated_multiply_up_lease_held ||
      !residual_add_capture_fields_are_clear(capture) ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      !indexed_rope_bf16_capture_fields_are_clear(capture) ||
      !canonical_gemm_bf16_capture_fields_are_clear(capture) ||
      !ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(capture)) {
    return false;
  }
  if (!capture->fill_lease_held) {
    return capture->fill_buffer == nullptr &&
           capture->operation != RileyCudaGraphCaptureOperation::kFillF32;
  }
  if (capture->fill_buffer == nullptr ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    capture->operation = RileyCudaGraphCaptureOperation::kNone;
  }
  return true;
}

// The H2D source is a captured raw host pointer. It must stay leased alongside
// the destination device allocation for the entire capture/graph/exec
// lifetime; otherwise a normal pinned-buffer close could create a graph UAF.
bool release_capture_h2d_leases(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kH2D ||
      capture->fill_buffer == nullptr || capture->h2d_source == nullptr ||
      !capture->fill_lease_held || !capture->h2d_source_lease_held ||
      capture->h2d_byte_len == 0 ||
      capture->gated_multiply_activated_gate != nullptr ||
      capture->gated_multiply_up != nullptr ||
      capture->gated_multiply_element_count != 0 ||
      capture->gated_multiply_enqueue_count != 0 ||
      capture->gated_multiply_activated_gate_lease_held ||
      capture->gated_multiply_up_lease_held ||
      !residual_add_capture_fields_are_clear(capture) ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->h2d_source->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  // Both counters are verified before either deterministic 1->0 transition.
  if (!release_exclusive_use(capture->h2d_source->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->h2d_source = nullptr;
  capture->h2d_byte_len = 0;
  capture->h2d_enqueue_count = 0;
  capture->h2d_source_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-8 retains two distinct BF16 device allocations. Both are graph-visible
// raw addresses, so validate every immutable field and both 1->0 transitions
// before releasing either lease. A malformed raw ABI owner remains fail-closed.
bool release_capture_silu_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
      capture->fill_buffer == nullptr || capture->silu_input == nullptr ||
      capture->fill_buffer == capture->silu_input ||
      !capture->fill_lease_held || !capture->silu_input_lease_held ||
      capture->silu_element_count == 0 ||
      capture->gated_multiply_activated_gate != nullptr ||
      capture->gated_multiply_up != nullptr ||
      capture->gated_multiply_element_count != 0 ||
      capture->gated_multiply_enqueue_count != 0 ||
      capture->gated_multiply_activated_gate_lease_held ||
      capture->gated_multiply_up_lease_held ||
      !residual_add_capture_fields_are_clear(capture) ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->silu_input->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  if (!release_exclusive_use(capture->silu_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->silu_input = nullptr;
  capture->silu_element_count = 0;
  capture->silu_enqueue_count = 0;
  capture->silu_input_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-10 retains three distinct BF16 device allocations. Every raw address is
// graph-visible, so validate all immutable fields and every 1->0 transition
// before releasing any lease. A malformed raw ABI owner remains fail-closed.
bool release_capture_gated_multiply_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation !=
          RileyCudaGraphCaptureOperation::kGatedMultiplyBf16 ||
      capture->fill_buffer == nullptr ||
      capture->gated_multiply_activated_gate == nullptr ||
      capture->gated_multiply_up == nullptr ||
      capture->fill_buffer == capture->gated_multiply_activated_gate ||
      capture->fill_buffer == capture->gated_multiply_up ||
      capture->gated_multiply_activated_gate == capture->gated_multiply_up ||
      !capture->fill_lease_held ||
      !capture->gated_multiply_activated_gate_lease_held ||
      !capture->gated_multiply_up_lease_held ||
      capture->gated_multiply_element_count == 0 ||
      capture->fill_element_count != 0 || capture->fill_enqueue_count != 0 ||
      capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
      capture->h2d_enqueue_count != 0 || capture->h2d_source_lease_held ||
      capture->silu_input != nullptr || capture->silu_element_count != 0 ||
      capture->silu_enqueue_count != 0 || capture->silu_input_lease_held ||
      !residual_add_capture_fields_are_clear(capture) ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->gated_multiply_activated_gate->active_uses.load(
          std::memory_order_acquire) != 1 ||
      capture->gated_multiply_up->active_uses.load(std::memory_order_acquire) !=
          1) {
    return false;
  }
  if (!release_exclusive_use(capture->gated_multiply_up->active_uses) ||
      !release_exclusive_use(
          capture->gated_multiply_activated_gate->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->gated_multiply_activated_gate = nullptr;
  capture->gated_multiply_up = nullptr;
  capture->gated_multiply_element_count = 0;
  capture->gated_multiply_enqueue_count = 0;
  capture->gated_multiply_activated_gate_lease_held = false;
  capture->gated_multiply_up_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-11 retains three distinct BF16 device allocations. Every raw address is
// graph-visible, so validate all immutable fields and every 1->0 transition
// before releasing any lease. A malformed raw ABI owner remains fail-closed.
bool release_capture_residual_add_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr ||
      capture->operation != RileyCudaGraphCaptureOperation::kResidualAddBf16 ||
      capture->fill_buffer == nullptr || capture->residual_add_left == nullptr ||
      capture->residual_add_right == nullptr ||
      capture->fill_buffer == capture->residual_add_left ||
      capture->fill_buffer == capture->residual_add_right ||
      capture->residual_add_left == capture->residual_add_right ||
      !capture->fill_lease_held || !capture->residual_add_left_lease_held ||
      !capture->residual_add_right_lease_held ||
      capture->residual_add_element_count == 0 ||
      capture->fill_element_count != 0 || capture->fill_enqueue_count != 0 ||
      capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
      capture->h2d_enqueue_count != 0 || capture->h2d_source_lease_held ||
      capture->silu_input != nullptr || capture->silu_element_count != 0 ||
      capture->silu_enqueue_count != 0 || capture->silu_input_lease_held ||
      capture->gated_multiply_activated_gate != nullptr ||
      capture->gated_multiply_up != nullptr ||
      capture->gated_multiply_element_count != 0 ||
      capture->gated_multiply_enqueue_count != 0 ||
      capture->gated_multiply_activated_gate_lease_held ||
      capture->gated_multiply_up_lease_held ||
      !canonical_rms_norm_capture_fields_are_clear(capture) ||
      !bf16_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
      !bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->residual_add_left->active_uses.load(std::memory_order_acquire) !=
          1 ||
      capture->residual_add_right->active_uses.load(std::memory_order_acquire) !=
          1) {
    return false;
  }
  if (!release_exclusive_use(capture->residual_add_right->active_uses) ||
      !release_exclusive_use(capture->residual_add_left->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->residual_add_left = nullptr;
  capture->residual_add_right = nullptr;
  capture->residual_add_element_count = 0;
  capture->residual_add_enqueue_count = 0;
  capture->residual_add_left_lease_held = false;
  capture->residual_add_right_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-12 retains the canonical RMSNorm input, learned weight, and output for
// the full capture lifecycle. Validate the whole immutable state before any
// 1->0 transition so malformed raw ABI owners remain fail-closed.
bool release_capture_canonical_rms_norm_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!canonical_rms_norm_capture_state_is_valid(capture) ||
      capture->canonical_rms_norm_enqueue_count > 1) {
    return false;
  }
  if (!release_exclusive_use(
          capture->canonical_rms_norm_weight->active_uses) ||
      !release_exclusive_use(
          capture->canonical_rms_norm_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->canonical_rms_norm_input = nullptr;
  capture->canonical_rms_norm_weight = nullptr;
  capture->canonical_rms_norm_row_count = 0;
  capture->canonical_rms_norm_hidden_size = 0;
  capture->canonical_rms_norm_epsilon = 0.0F;
  capture->canonical_rms_norm_enqueue_count = 0;
  capture->canonical_rms_norm_input_lease_held = false;
  capture->canonical_rms_norm_weight_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-13 retains logits and U32 argmax results for the complete capture
// lifecycle. Validate first so every raw-address lease has a known 1->0
// transition and malformed ABI owners remain fail-closed.
bool release_capture_bf16_argmax_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!bf16_argmax_capture_state_is_valid(capture) ||
      capture->bf16_argmax_enqueue_count > 1) {
    return false;
  }
  if (!release_exclusive_use(capture->bf16_argmax_logits->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->bf16_argmax_logits = nullptr;
  capture->bf16_argmax_row_count = 0;
  capture->bf16_argmax_vocabulary_size = 0;
  capture->bf16_argmax_enqueue_count = 0;
  capture->bf16_argmax_logits_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-14 retains three distinct fixed-address allocations for the complete
// capture lifecycle. Validate every field before the first 1->0 transition:
// raw device indices may be malformed, but the leases themselves must never
// be ambiguously released.
bool release_capture_bf16_row_gather_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!bf16_row_gather_capture_state_is_valid(capture) ||
      capture->bf16_row_gather_enqueue_count > 1) {
    return false;
  }
  if (!release_exclusive_use(capture->bf16_row_gather_indices->active_uses) ||
      !release_exclusive_use(capture->bf16_row_gather_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->bf16_row_gather_input = nullptr;
  capture->bf16_row_gather_indices = nullptr;
  capture->bf16_row_gather_input_row_count = 0;
  capture->bf16_row_gather_output_row_count = 0;
  capture->bf16_row_gather_column_count = 0;
  capture->bf16_row_gather_enqueue_count = 0;
  capture->bf16_row_gather_input_lease_held = false;
  capture->bf16_row_gather_indices_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-15 retains input, U32 row indices, gathered BF16 logits, and U32
// argmax results. The terminal enqueue marker is intentionally releasable by
// abort: it preserves every lease until CUDA capture termination is known,
// but cannot become a retryable capture state.
bool release_capture_bf16_row_gather_argmax_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!bf16_row_gather_argmax_capture_state_is_valid(capture) ||
      (capture->bf16_row_gather_argmax_enqueue_count != 0 &&
       capture->bf16_row_gather_argmax_enqueue_count != 1 &&
       capture->bf16_row_gather_argmax_enqueue_count !=
           kBf16RowGatherArgmaxEnqueueTerminal)) {
    return false;
  }
  if (!release_exclusive_use(
          capture->bf16_row_gather_argmax_gathered_logits->active_uses) ||
      !release_exclusive_use(
          capture->bf16_row_gather_argmax_indices->active_uses) ||
      !release_exclusive_use(
          capture->bf16_row_gather_argmax_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->bf16_row_gather_argmax_input = nullptr;
  capture->bf16_row_gather_argmax_indices = nullptr;
  capture->bf16_row_gather_argmax_gathered_logits = nullptr;
  capture->bf16_row_gather_argmax_input_row_count = 0;
  capture->bf16_row_gather_argmax_output_row_count = 0;
  capture->bf16_row_gather_argmax_vocabulary_size = 0;
  capture->bf16_row_gather_argmax_enqueue_count = 0;
  capture->bf16_row_gather_argmax_input_lease_held = false;
  capture->bf16_row_gather_argmax_indices_lease_held = false;
  capture->bf16_row_gather_argmax_gathered_logits_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-16/C05-20 share this five-address ownership ledger, but operation tags
// and state predicates keep their recorded node chains disjoint. A terminal
// marker is abort-releasable only: a partially recorded node prefix keeps
// every raw address leased until CUDA capture termination is known.
bool release_capture_bf16_row_gather_argmax_d2h_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!bf16_d2h_capture_state_is_valid(capture) ||
      (capture->bf16_row_gather_argmax_d2h_enqueue_count != 0 &&
       capture->bf16_row_gather_argmax_d2h_enqueue_count != 1 &&
       capture->bf16_row_gather_argmax_d2h_enqueue_count !=
           (capture->operation ==
                    RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H
                ? kBf16EmbeddingStatusD2HEnqueueTerminal
                : kBf16RowGatherArgmaxD2HEnqueueTerminal))) {
    return false;
  }
  if (!release_exclusive_use(
          capture->bf16_row_gather_argmax_d2h_pinned_results->active_uses) ||
      !release_exclusive_use(
          capture->bf16_row_gather_argmax_d2h_gathered_logits->active_uses) ||
      !release_exclusive_use(
          capture->bf16_row_gather_argmax_d2h_indices->active_uses) ||
      !release_exclusive_use(
          capture->bf16_row_gather_argmax_d2h_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_input = nullptr;
  capture->bf16_row_gather_argmax_d2h_indices = nullptr;
  capture->bf16_row_gather_argmax_d2h_gathered_logits = nullptr;
  capture->bf16_row_gather_argmax_d2h_pinned_results = nullptr;
  capture->bf16_row_gather_argmax_d2h_input_row_count = 0;
  capture->bf16_row_gather_argmax_d2h_output_row_count = 0;
  capture->bf16_row_gather_argmax_d2h_vocabulary_size = 0;
  capture->bf16_row_gather_argmax_d2h_result_byte_len = 0;
  capture->bf16_row_gather_argmax_d2h_enqueue_count = 0;
  capture->bf16_row_gather_argmax_d2h_input_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_indices_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-17 keeps five raw device addresses fixed through capture, graph, and
// exec ownership. The terminal enqueue marker is abort-releasable only: after
// a failed node launch we cannot distinguish an unrecorded graph from a graph
// containing an accepted prefix without the one CUDA abort transition.
bool release_capture_indexed_rope_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!indexed_rope_bf16_capture_state_is_valid(capture) ||
      (capture->indexed_rope_bf16_enqueue_count != 0 &&
       capture->indexed_rope_bf16_enqueue_count != 1 &&
       capture->indexed_rope_bf16_enqueue_count !=
           kIndexedRopeBf16EnqueueTerminal)) {
    return false;
  }
  if (!release_exclusive_use(
          capture->indexed_rope_bf16_positions->active_uses) ||
      !release_exclusive_use(capture->indexed_rope_bf16_sin->active_uses) ||
      !release_exclusive_use(capture->indexed_rope_bf16_cos->active_uses) ||
      !release_exclusive_use(capture->indexed_rope_bf16_input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->indexed_rope_bf16_input = nullptr;
  capture->indexed_rope_bf16_cos = nullptr;
  capture->indexed_rope_bf16_sin = nullptr;
  capture->indexed_rope_bf16_positions = nullptr;
  capture->indexed_rope_bf16_active_row_count = 0;
  capture->indexed_rope_bf16_head_count = 0;
  capture->indexed_rope_bf16_head_size = 0;
  capture->indexed_rope_bf16_rotary_dimension = 0;
  capture->indexed_rope_bf16_table_position_count = 0;
  capture->indexed_rope_bf16_enqueue_count = 0;
  capture->indexed_rope_bf16_input_lease_held = false;
  capture->indexed_rope_bf16_cos_lease_held = false;
  capture->indexed_rope_bf16_sin_lease_held = false;
  capture->indexed_rope_bf16_positions_lease_held = false;
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-18 permanently leases its two sources, two pools, and five raw metadata
// allocations from begin until capture abort or graph/exec close. key_pool is
// deliberately the legacy fill_buffer lease; do not add a second key-pool
// counter to the operation state.
bool release_capture_ragged_paged_kv_cache_write_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!ragged_paged_kv_cache_write_bf16_capture_state_is_valid(capture)) {
    return false;
  }
  const uint32_t enqueue_count =
      capture->ragged_paged_kv_write_bf16.enqueue_count;
  if (enqueue_count != 0 && enqueue_count != 1 &&
      enqueue_count != kRaggedPagedKvCacheWriteBf16EnqueueTerminal) {
    return false;
  }
  RileyCudaRaggedPagedKvCacheWriteBf16State& state =
      capture->ragged_paged_kv_write_bf16;
  if (!release_exclusive_use(state.row_positions->active_uses) ||
      !release_exclusive_use(state.row_sequence_slots->active_uses) ||
      !release_exclusive_use(state.valid_tokens->active_uses) ||
      !release_exclusive_use(state.block_ids->active_uses) ||
      !release_exclusive_use(state.sequence_block_offsets->active_uses) ||
      !release_exclusive_use(state.value_pool->active_uses) ||
      !release_exclusive_use(state.value_source->active_uses) ||
      !release_exclusive_use(state.key_source->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  state = RileyCudaRaggedPagedKvCacheWriteBf16State{};
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

// C05-21 leases an opaque, already-prepared cuBLASLt GEMM plan plus three
// non-output allocations. The legacy fill-buffer slot is the fixed output;
// it retains the fourth device allocation without duplicating its counter in
// the operation state. A failed cublasLt submission is abort-releasable only.
bool release_capture_canonical_gemm_bf16_leases(
    RileyCudaGraphCapture* capture) noexcept {
  if (!canonical_gemm_bf16_capture_state_is_valid(capture)) {
    return false;
  }
  RileyCudaCanonicalGemmBf16GraphState& state =
      capture->canonical_gemm_bf16;
  if (state.enqueue_count != 0 && state.enqueue_count != 1 &&
      state.enqueue_count != kCanonicalGemmBf16EnqueueTerminal) {
    return false;
  }
  if (!release_exclusive_use(state.workspace->active_uses) ||
      !release_exclusive_use(state.weight->active_uses) ||
      !release_exclusive_use(state.input->active_uses) ||
      !release_exclusive_use(capture->fill_buffer->active_uses) ||
      !riley_cuda_internal::release_canonical_gemm_bf16_graph_plan_lease(
          state.plan)) {
    return false;
  }
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  state = RileyCudaCanonicalGemmBf16GraphState{};
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  return true;
}

bool destroy_prepared_graph_storage(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->prepared_graph == nullptr) {
    return capture != nullptr;
  }
  RileyCudaGraph* const graph = capture->prepared_graph;
  if (graph->owner != capture->owner || graph->stream != capture->stream ||
      graph->graph != nullptr || graph->owns_capture_leases) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kResidualAddBf16 &&
      (!residual_add_capture_fields_are_clear(capture) ||
       !residual_add_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation !=
          RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16 &&
      (!canonical_rms_norm_capture_fields_are_clear(capture) ||
       !canonical_rms_norm_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kBf16Argmax &&
      (!bf16_argmax_capture_fields_are_clear(capture) ||
       !bf16_argmax_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kBf16RowGather &&
      (!bf16_row_gather_capture_fields_are_clear(capture) ||
       !bf16_row_gather_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation !=
          RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax &&
      (!bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
       !bf16_row_gather_argmax_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_bf16_d2h_operation(capture->operation) &&
      (!bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
       !bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kIndexedRopeBf16 &&
      (!indexed_rope_bf16_capture_fields_are_clear(capture) ||
       !indexed_rope_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_canonical_gemm_bf16_operation(capture->operation) &&
      (!canonical_gemm_bf16_capture_fields_are_clear(capture) ||
       !canonical_gemm_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_ragged_paged_fixed_address_operation(capture->operation) &&
      (!ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(capture) ||
       !ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kH2D) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kH2D ||
        graph->h2d_source != capture->h2d_source ||
        graph->h2d_byte_len != capture->h2d_byte_len ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kSiluBf16) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != capture->silu_input ||
        graph->silu_element_count != capture->silu_element_count ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kGatedMultiplyBf16) {
    if (graph->operation !=
            RileyCudaGraphCaptureOperation::kGatedMultiplyBf16 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        graph->gated_multiply_activated_gate !=
            capture->gated_multiply_activated_gate ||
        graph->gated_multiply_up != capture->gated_multiply_up ||
        graph->gated_multiply_element_count !=
            capture->gated_multiply_element_count) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kResidualAddBf16) {
    if (graph->operation != RileyCudaGraphCaptureOperation::kResidualAddBf16 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0 ||
        graph->residual_add_left != capture->residual_add_left ||
        graph->residual_add_right != capture->residual_add_right ||
        graph->residual_add_element_count !=
            capture->residual_add_element_count) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16) {
    if (!canonical_rms_norm_capture_state_is_valid(capture) ||
        !canonical_rms_norm_graph_state_is_valid(graph) ||
        graph->canonical_rms_norm_input !=
            capture->canonical_rms_norm_input ||
        graph->canonical_rms_norm_weight !=
            capture->canonical_rms_norm_weight ||
        graph->canonical_rms_norm_row_count !=
            capture->canonical_rms_norm_row_count ||
        graph->canonical_rms_norm_hidden_size !=
            capture->canonical_rms_norm_hidden_size ||
        graph->canonical_rms_norm_epsilon !=
            capture->canonical_rms_norm_epsilon) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kBf16Argmax) {
    if (!bf16_argmax_capture_state_is_valid(capture) ||
        !bf16_argmax_graph_state_is_valid(graph) ||
        graph->bf16_argmax_logits != capture->bf16_argmax_logits ||
        graph->bf16_argmax_row_count != capture->bf16_argmax_row_count ||
        graph->bf16_argmax_vocabulary_size !=
            capture->bf16_argmax_vocabulary_size) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGather) {
    if (!bf16_row_gather_capture_state_is_valid(capture) ||
        !bf16_row_gather_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_input != capture->bf16_row_gather_input ||
        graph->bf16_row_gather_indices != capture->bf16_row_gather_indices ||
        graph->bf16_row_gather_input_row_count !=
            capture->bf16_row_gather_input_row_count ||
        graph->bf16_row_gather_output_row_count !=
            capture->bf16_row_gather_output_row_count ||
        graph->bf16_row_gather_column_count !=
            capture->bf16_row_gather_column_count) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax) {
    if (!bf16_row_gather_argmax_capture_state_is_valid(capture) ||
        !bf16_row_gather_argmax_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_argmax_input !=
            capture->bf16_row_gather_argmax_input ||
        graph->bf16_row_gather_argmax_indices !=
            capture->bf16_row_gather_argmax_indices ||
        graph->bf16_row_gather_argmax_gathered_logits !=
            capture->bf16_row_gather_argmax_gathered_logits ||
        graph->bf16_row_gather_argmax_input_row_count !=
            capture->bf16_row_gather_argmax_input_row_count ||
        graph->bf16_row_gather_argmax_output_row_count !=
            capture->bf16_row_gather_argmax_output_row_count ||
        graph->bf16_row_gather_argmax_vocabulary_size !=
            capture->bf16_row_gather_argmax_vocabulary_size) {
      return false;
    }
  } else if (is_bf16_d2h_operation(capture->operation)) {
    if (!bf16_d2h_capture_state_is_valid(capture) ||
        !bf16_d2h_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_argmax_d2h_input !=
            capture->bf16_row_gather_argmax_d2h_input ||
        graph->bf16_row_gather_argmax_d2h_indices !=
            capture->bf16_row_gather_argmax_d2h_indices ||
        graph->bf16_row_gather_argmax_d2h_gathered_logits !=
            capture->bf16_row_gather_argmax_d2h_gathered_logits ||
        graph->bf16_row_gather_argmax_d2h_pinned_results !=
            capture->bf16_row_gather_argmax_d2h_pinned_results ||
        graph->bf16_row_gather_argmax_d2h_input_row_count !=
            capture->bf16_row_gather_argmax_d2h_input_row_count ||
        graph->bf16_row_gather_argmax_d2h_output_row_count !=
            capture->bf16_row_gather_argmax_d2h_output_row_count ||
        graph->bf16_row_gather_argmax_d2h_vocabulary_size !=
            capture->bf16_row_gather_argmax_d2h_vocabulary_size ||
        graph->bf16_row_gather_argmax_d2h_result_byte_len !=
            capture->bf16_row_gather_argmax_d2h_result_byte_len) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kIndexedRopeBf16) {
    if (!indexed_rope_bf16_capture_state_is_valid(capture) ||
        !indexed_rope_bf16_graph_state_is_valid(graph) ||
        graph->indexed_rope_bf16_input != capture->indexed_rope_bf16_input ||
        graph->indexed_rope_bf16_cos != capture->indexed_rope_bf16_cos ||
        graph->indexed_rope_bf16_sin != capture->indexed_rope_bf16_sin ||
        graph->indexed_rope_bf16_positions !=
            capture->indexed_rope_bf16_positions ||
        graph->indexed_rope_bf16_active_row_count !=
            capture->indexed_rope_bf16_active_row_count ||
        graph->indexed_rope_bf16_head_count !=
            capture->indexed_rope_bf16_head_count ||
        graph->indexed_rope_bf16_head_size !=
            capture->indexed_rope_bf16_head_size ||
        graph->indexed_rope_bf16_rotary_dimension !=
            capture->indexed_rope_bf16_rotary_dimension ||
        graph->indexed_rope_bf16_table_position_count !=
            capture->indexed_rope_bf16_table_position_count) {
      return false;
    }
  } else if (is_canonical_gemm_bf16_operation(capture->operation)) {
    if (!canonical_gemm_bf16_capture_state_is_valid(capture) ||
        !canonical_gemm_bf16_prepared_graph_state_is_valid(graph) ||
        !canonical_gemm_bf16_state_matches(capture->canonical_gemm_bf16,
                                            graph->canonical_gemm_bf16)) {
      return false;
    }
  } else if (is_ragged_paged_fixed_address_operation(capture->operation)) {
    if (!ragged_paged_kv_cache_write_bf16_capture_state_is_valid(capture) ||
        !ragged_paged_kv_cache_write_bf16_prepared_graph_state_is_valid(graph) ||
        !ragged_paged_kv_cache_write_bf16_state_geometry_matches(
            capture->ragged_paged_kv_write_bf16,
            graph->ragged_paged_kv_write_bf16)) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kNone) {
    // C05-5's historical cleanup releases the fixed-buffer lease before it
    // frees this preallocated graph wrapper. That order is valid only for a
    // fill graph, which has no pinned source pointer to preserve.
    if (graph->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0) {
      return false;
    }
  } else {
    return false;
  }
  graph->~RileyCudaGraph();
  std::free(graph);
  capture->prepared_graph = nullptr;
  return true;
}

// Move only the capture-domain/TLS ownership off the capture wrapper. The
// context-child, stream, and buffer leases deliberately remain acquired and
// become the returned graph's permanent resource guard.
bool transfer_capture_owner_to_graph(RileyCudaGraphCapture* capture) noexcept {
  if (capture == nullptr || capture->owner == nullptr ||
      capture->stream == nullptr || capture->capture_domain == nullptr ||
      capture->prepared_graph == nullptr || capture->fill_buffer == nullptr ||
      !capture->fill_lease_held || capture->deferred_close_head != nullptr ||
      capture->deferred_close_tail != nullptr || capture->unreleased_graph != nullptr) {
    return false;
  }
  RileyCudaGraph* const graph = capture->prepared_graph;
  if (graph->owner != capture->owner || graph->stream != capture->stream ||
      graph->fill_buffer != capture->fill_buffer ||
      graph->operation != capture->operation || graph->graph == nullptr ||
      graph->owns_capture_leases ||
      capture->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      capture->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kResidualAddBf16 &&
      (!residual_add_capture_fields_are_clear(capture) ||
       !residual_add_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation !=
          RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16 &&
      (!canonical_rms_norm_capture_fields_are_clear(capture) ||
       !canonical_rms_norm_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kBf16Argmax &&
      (!bf16_argmax_capture_fields_are_clear(capture) ||
       !bf16_argmax_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kBf16RowGather &&
      (!bf16_row_gather_capture_fields_are_clear(capture) ||
       !bf16_row_gather_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation !=
          RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax &&
      (!bf16_row_gather_argmax_capture_fields_are_clear(capture) ||
       !bf16_row_gather_argmax_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_bf16_d2h_operation(capture->operation) &&
      (!bf16_row_gather_argmax_d2h_capture_fields_are_clear(capture) ||
       !bf16_row_gather_argmax_d2h_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation != RileyCudaGraphCaptureOperation::kIndexedRopeBf16 &&
      (!indexed_rope_bf16_capture_fields_are_clear(capture) ||
       !indexed_rope_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_canonical_gemm_bf16_operation(capture->operation) &&
      (!canonical_gemm_bf16_capture_fields_are_clear(capture) ||
       !canonical_gemm_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (!is_ragged_paged_fixed_address_operation(capture->operation) &&
      (!ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(capture) ||
       !ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(graph))) {
    return false;
  }
  if (capture->operation == RileyCudaGraphCaptureOperation::kFillF32) {
    if (graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held || graph->silu_input != nullptr ||
        graph->silu_element_count != 0 || capture->silu_input != nullptr ||
        capture->silu_element_count != 0 || capture->silu_input_lease_held ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0 ||
        capture->gated_multiply_activated_gate != nullptr ||
        capture->gated_multiply_up != nullptr ||
        capture->gated_multiply_element_count != 0 ||
        capture->gated_multiply_enqueue_count != 0 ||
        capture->gated_multiply_activated_gate_lease_held ||
        capture->gated_multiply_up_lease_held) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kH2D) {
    if (capture->h2d_source == nullptr || !capture->h2d_source_lease_held ||
        capture->h2d_byte_len == 0 ||
        graph->h2d_source != capture->h2d_source ||
        graph->h2d_byte_len != capture->h2d_byte_len ||
        capture->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        graph->silu_input != nullptr || graph->silu_element_count != 0 ||
        capture->silu_input != nullptr || capture->silu_element_count != 0 ||
        capture->silu_input_lease_held ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0 ||
        capture->gated_multiply_activated_gate != nullptr ||
        capture->gated_multiply_up != nullptr ||
        capture->gated_multiply_element_count != 0 ||
        capture->gated_multiply_enqueue_count != 0 ||
        capture->gated_multiply_activated_gate_lease_held ||
        capture->gated_multiply_up_lease_held) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kSiluBf16) {
    if (capture->silu_input == nullptr ||
        capture->silu_input == capture->fill_buffer ||
        !capture->silu_input_lease_held ||
        capture->silu_element_count == 0 ||
        graph->silu_input != capture->silu_input ||
        graph->silu_element_count != capture->silu_element_count ||
        capture->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0 ||
        capture->gated_multiply_activated_gate != nullptr ||
        capture->gated_multiply_up != nullptr ||
        capture->gated_multiply_element_count != 0 ||
        capture->gated_multiply_enqueue_count != 0 ||
        capture->gated_multiply_activated_gate_lease_held ||
        capture->gated_multiply_up_lease_held) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kGatedMultiplyBf16) {
    if (capture->gated_multiply_activated_gate == nullptr ||
        capture->gated_multiply_up == nullptr ||
        capture->gated_multiply_activated_gate == capture->gated_multiply_up ||
        capture->gated_multiply_activated_gate == capture->fill_buffer ||
        capture->gated_multiply_up == capture->fill_buffer ||
        !capture->gated_multiply_activated_gate_lease_held ||
        !capture->gated_multiply_up_lease_held ||
        capture->gated_multiply_element_count == 0 ||
        graph->gated_multiply_activated_gate !=
            capture->gated_multiply_activated_gate ||
        graph->gated_multiply_up != capture->gated_multiply_up ||
        graph->gated_multiply_element_count !=
            capture->gated_multiply_element_count ||
        capture->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        capture->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held || graph->silu_input != nullptr ||
        graph->silu_element_count != 0 || capture->silu_input != nullptr ||
        capture->silu_element_count != 0 || capture->silu_input_lease_held) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kResidualAddBf16) {
    if (capture->residual_add_left == nullptr ||
        capture->residual_add_right == nullptr ||
        capture->residual_add_left == capture->residual_add_right ||
        capture->residual_add_left == capture->fill_buffer ||
        capture->residual_add_right == capture->fill_buffer ||
        !capture->residual_add_left_lease_held ||
        !capture->residual_add_right_lease_held ||
        capture->residual_add_element_count == 0 ||
        graph->residual_add_left != capture->residual_add_left ||
        graph->residual_add_right != capture->residual_add_right ||
        graph->residual_add_element_count != capture->residual_add_element_count ||
        capture->residual_add_left->active_uses.load(
            std::memory_order_acquire) != 1 ||
        capture->residual_add_right->active_uses.load(
            std::memory_order_acquire) != 1 ||
        graph->h2d_source != nullptr || graph->h2d_byte_len != 0 ||
        capture->h2d_source != nullptr || capture->h2d_byte_len != 0 ||
        capture->h2d_source_lease_held || graph->silu_input != nullptr ||
        graph->silu_element_count != 0 || capture->silu_input != nullptr ||
        capture->silu_element_count != 0 || capture->silu_input_lease_held ||
        graph->gated_multiply_activated_gate != nullptr ||
        graph->gated_multiply_up != nullptr ||
        graph->gated_multiply_element_count != 0 ||
        capture->gated_multiply_activated_gate != nullptr ||
        capture->gated_multiply_up != nullptr ||
        capture->gated_multiply_element_count != 0 ||
        capture->gated_multiply_enqueue_count != 0 ||
        capture->gated_multiply_activated_gate_lease_held ||
        capture->gated_multiply_up_lease_held) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16) {
    if (!canonical_rms_norm_capture_state_is_valid(capture) ||
        !canonical_rms_norm_graph_state_is_valid(graph) ||
        graph->canonical_rms_norm_input !=
            capture->canonical_rms_norm_input ||
        graph->canonical_rms_norm_weight !=
            capture->canonical_rms_norm_weight ||
        graph->canonical_rms_norm_row_count !=
            capture->canonical_rms_norm_row_count ||
        graph->canonical_rms_norm_hidden_size !=
            capture->canonical_rms_norm_hidden_size ||
        graph->canonical_rms_norm_epsilon !=
            capture->canonical_rms_norm_epsilon) {
      return false;
    }
  } else if (capture->operation == RileyCudaGraphCaptureOperation::kBf16Argmax) {
    if (!bf16_argmax_capture_state_is_valid(capture) ||
        !bf16_argmax_graph_state_is_valid(graph) ||
        graph->bf16_argmax_logits != capture->bf16_argmax_logits ||
        graph->bf16_argmax_row_count != capture->bf16_argmax_row_count ||
        graph->bf16_argmax_vocabulary_size !=
            capture->bf16_argmax_vocabulary_size) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGather) {
    if (!bf16_row_gather_capture_state_is_valid(capture) ||
        !bf16_row_gather_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_input != capture->bf16_row_gather_input ||
        graph->bf16_row_gather_indices != capture->bf16_row_gather_indices ||
        graph->bf16_row_gather_input_row_count !=
            capture->bf16_row_gather_input_row_count ||
        graph->bf16_row_gather_output_row_count !=
            capture->bf16_row_gather_output_row_count ||
        graph->bf16_row_gather_column_count !=
            capture->bf16_row_gather_column_count) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax) {
    if (capture->bf16_row_gather_argmax_enqueue_count != 1 ||
        !bf16_row_gather_argmax_capture_state_is_valid(capture) ||
        !bf16_row_gather_argmax_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_argmax_input !=
            capture->bf16_row_gather_argmax_input ||
        graph->bf16_row_gather_argmax_indices !=
            capture->bf16_row_gather_argmax_indices ||
        graph->bf16_row_gather_argmax_gathered_logits !=
            capture->bf16_row_gather_argmax_gathered_logits ||
        graph->bf16_row_gather_argmax_input_row_count !=
            capture->bf16_row_gather_argmax_input_row_count ||
        graph->bf16_row_gather_argmax_output_row_count !=
            capture->bf16_row_gather_argmax_output_row_count ||
        graph->bf16_row_gather_argmax_vocabulary_size !=
            capture->bf16_row_gather_argmax_vocabulary_size) {
      return false;
    }
  } else if (is_bf16_d2h_operation(capture->operation)) {
    if (capture->bf16_row_gather_argmax_d2h_enqueue_count != 1 ||
        !bf16_d2h_capture_state_is_valid(capture) ||
        !bf16_d2h_graph_state_is_valid(graph) ||
        graph->bf16_row_gather_argmax_d2h_input !=
            capture->bf16_row_gather_argmax_d2h_input ||
        graph->bf16_row_gather_argmax_d2h_indices !=
            capture->bf16_row_gather_argmax_d2h_indices ||
        graph->bf16_row_gather_argmax_d2h_gathered_logits !=
            capture->bf16_row_gather_argmax_d2h_gathered_logits ||
        graph->bf16_row_gather_argmax_d2h_pinned_results !=
            capture->bf16_row_gather_argmax_d2h_pinned_results ||
        graph->bf16_row_gather_argmax_d2h_input_row_count !=
            capture->bf16_row_gather_argmax_d2h_input_row_count ||
        graph->bf16_row_gather_argmax_d2h_output_row_count !=
            capture->bf16_row_gather_argmax_d2h_output_row_count ||
        graph->bf16_row_gather_argmax_d2h_vocabulary_size !=
            capture->bf16_row_gather_argmax_d2h_vocabulary_size ||
        graph->bf16_row_gather_argmax_d2h_result_byte_len !=
            capture->bf16_row_gather_argmax_d2h_result_byte_len) {
      return false;
    }
  } else if (capture->operation ==
             RileyCudaGraphCaptureOperation::kIndexedRopeBf16) {
    if (capture->indexed_rope_bf16_enqueue_count != 1 ||
        !indexed_rope_bf16_capture_state_is_valid(capture) ||
        !indexed_rope_bf16_graph_state_is_valid(graph) ||
        graph->indexed_rope_bf16_input != capture->indexed_rope_bf16_input ||
        graph->indexed_rope_bf16_cos != capture->indexed_rope_bf16_cos ||
        graph->indexed_rope_bf16_sin != capture->indexed_rope_bf16_sin ||
        graph->indexed_rope_bf16_positions !=
            capture->indexed_rope_bf16_positions ||
        graph->indexed_rope_bf16_active_row_count !=
            capture->indexed_rope_bf16_active_row_count ||
        graph->indexed_rope_bf16_head_count !=
            capture->indexed_rope_bf16_head_count ||
        graph->indexed_rope_bf16_head_size !=
            capture->indexed_rope_bf16_head_size ||
        graph->indexed_rope_bf16_rotary_dimension !=
            capture->indexed_rope_bf16_rotary_dimension ||
        graph->indexed_rope_bf16_table_position_count !=
            capture->indexed_rope_bf16_table_position_count) {
      return false;
    }
  } else if (is_canonical_gemm_bf16_operation(capture->operation)) {
    if (capture->canonical_gemm_bf16.enqueue_count != 1 ||
        !canonical_gemm_bf16_capture_state_is_valid(capture) ||
        !canonical_gemm_bf16_prepared_graph_state_is_valid(graph) ||
        !canonical_gemm_bf16_state_matches(capture->canonical_gemm_bf16,
                                            graph->canonical_gemm_bf16)) {
      return false;
    }
  } else if (is_ragged_paged_fixed_address_operation(capture->operation)) {
    if (capture->ragged_paged_kv_write_bf16.enqueue_count != 1 ||
        !ragged_paged_kv_cache_write_bf16_capture_state_is_valid(capture) ||
        !ragged_paged_kv_cache_write_bf16_prepared_graph_state_is_valid(graph) ||
        !ragged_paged_kv_cache_write_bf16_state_geometry_matches(
            capture->ragged_paged_kv_write_bf16,
            graph->ragged_paged_kv_write_bf16)) {
      return false;
    }
  } else {
    return false;
  }
  if (!release_capture_domain_capture(capture->capture_domain) ||
      !clear_thread_graph_capture_owner(capture)) {
    return false;
  }
  graph->owns_capture_leases = true;
  if (is_canonical_gemm_bf16_operation(capture->operation)) {
    graph->canonical_gemm_bf16 = capture->canonical_gemm_bf16;
    capture->canonical_gemm_bf16 = RileyCudaCanonicalGemmBf16GraphState{};
  }
  if (is_ragged_paged_fixed_address_operation(capture->operation)) {
    graph->ragged_paged_kv_write_bf16 =
        capture->ragged_paged_kv_write_bf16;
    capture->ragged_paged_kv_write_bf16 =
        RileyCudaRaggedPagedKvCacheWriteBf16State{};
  }
  capture->prepared_graph = nullptr;
  capture->fill_buffer = nullptr;
  capture->fill_element_count = 0;
  capture->fill_enqueue_count = 0;
  capture->fill_lease_held = false;
  capture->h2d_source = nullptr;
  capture->h2d_byte_len = 0;
  capture->h2d_enqueue_count = 0;
  capture->h2d_source_lease_held = false;
  capture->silu_input = nullptr;
  capture->silu_element_count = 0;
  capture->silu_enqueue_count = 0;
  capture->silu_input_lease_held = false;
  capture->gated_multiply_activated_gate = nullptr;
  capture->gated_multiply_up = nullptr;
  capture->gated_multiply_element_count = 0;
  capture->gated_multiply_enqueue_count = 0;
  capture->gated_multiply_activated_gate_lease_held = false;
  capture->gated_multiply_up_lease_held = false;
  capture->residual_add_left = nullptr;
  capture->residual_add_right = nullptr;
  capture->residual_add_element_count = 0;
  capture->residual_add_enqueue_count = 0;
  capture->residual_add_left_lease_held = false;
  capture->residual_add_right_lease_held = false;
  capture->canonical_rms_norm_input = nullptr;
  capture->canonical_rms_norm_weight = nullptr;
  capture->canonical_rms_norm_row_count = 0;
  capture->canonical_rms_norm_hidden_size = 0;
  capture->canonical_rms_norm_epsilon = 0.0F;
  capture->canonical_rms_norm_enqueue_count = 0;
  capture->canonical_rms_norm_input_lease_held = false;
  capture->canonical_rms_norm_weight_lease_held = false;
  capture->bf16_argmax_logits = nullptr;
  capture->bf16_argmax_row_count = 0;
  capture->bf16_argmax_vocabulary_size = 0;
  capture->bf16_argmax_enqueue_count = 0;
  capture->bf16_argmax_logits_lease_held = false;
  capture->bf16_row_gather_input = nullptr;
  capture->bf16_row_gather_indices = nullptr;
  capture->bf16_row_gather_input_row_count = 0;
  capture->bf16_row_gather_output_row_count = 0;
  capture->bf16_row_gather_column_count = 0;
  capture->bf16_row_gather_enqueue_count = 0;
  capture->bf16_row_gather_input_lease_held = false;
  capture->bf16_row_gather_indices_lease_held = false;
  capture->bf16_row_gather_argmax_input = nullptr;
  capture->bf16_row_gather_argmax_indices = nullptr;
  capture->bf16_row_gather_argmax_gathered_logits = nullptr;
  capture->bf16_row_gather_argmax_input_row_count = 0;
  capture->bf16_row_gather_argmax_output_row_count = 0;
  capture->bf16_row_gather_argmax_vocabulary_size = 0;
  capture->bf16_row_gather_argmax_enqueue_count = 0;
  capture->bf16_row_gather_argmax_input_lease_held = false;
  capture->bf16_row_gather_argmax_indices_lease_held = false;
  capture->bf16_row_gather_argmax_gathered_logits_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_input = nullptr;
  capture->bf16_row_gather_argmax_d2h_indices = nullptr;
  capture->bf16_row_gather_argmax_d2h_gathered_logits = nullptr;
  capture->bf16_row_gather_argmax_d2h_pinned_results = nullptr;
  capture->bf16_row_gather_argmax_d2h_input_row_count = 0;
  capture->bf16_row_gather_argmax_d2h_output_row_count = 0;
  capture->bf16_row_gather_argmax_d2h_vocabulary_size = 0;
  capture->bf16_row_gather_argmax_d2h_result_byte_len = 0;
  capture->bf16_row_gather_argmax_d2h_enqueue_count = 0;
  capture->bf16_row_gather_argmax_d2h_input_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_indices_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held = false;
  capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held = false;
  capture->indexed_rope_bf16_input = nullptr;
  capture->indexed_rope_bf16_cos = nullptr;
  capture->indexed_rope_bf16_sin = nullptr;
  capture->indexed_rope_bf16_positions = nullptr;
  capture->indexed_rope_bf16_active_row_count = 0;
  capture->indexed_rope_bf16_head_count = 0;
  capture->indexed_rope_bf16_head_size = 0;
  capture->indexed_rope_bf16_rotary_dimension = 0;
  capture->indexed_rope_bf16_table_position_count = 0;
  capture->indexed_rope_bf16_enqueue_count = 0;
  capture->indexed_rope_bf16_input_lease_held = false;
  capture->indexed_rope_bf16_cos_lease_held = false;
  capture->indexed_rope_bf16_sin_lease_held = false;
  capture->indexed_rope_bf16_positions_lease_held = false;
  capture->canonical_gemm_bf16 = RileyCudaCanonicalGemmBf16GraphState{};
  capture->ragged_paged_kv_write_bf16 =
      RileyCudaRaggedPagedKvCacheWriteBf16State{};
  capture->operation = RileyCudaGraphCaptureOperation::kNone;
  capture->~RileyCudaGraphCapture();
  std::free(capture);
  return true;
}

bool release_graph_leases(RileyCudaContext* owner, RileyCudaStream* stream,
                          RileyCudaDeviceBuffer* buffer) noexcept {
  if (owner == nullptr || stream == nullptr || buffer == nullptr ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, buffer->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  // Validate every counter first; each release is then a deterministic 1->0
  // transition. Any impossible underflow is retained fail-closed by callers.
  return release_exclusive_use(buffer->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

// H2D graph ownership adds a pinned source lease to C05-5's existing stream
// and device-buffer set. Verify every resource before releasing any of them;
// an impossible raw-ABI corruption stays fail-closed rather than exposing a
// captured pointer whose lifetime is no longer known.
bool release_graph_h2d_leases(RileyCudaContext* owner, RileyCudaStream* stream,
                              RileyCudaDeviceBuffer* destination,
                              RileyCudaPinnedHostBuffer* source) noexcept {
  if (owner == nullptr || stream == nullptr || destination == nullptr ||
      source == nullptr || !same_context(owner, stream->owner) ||
      !same_context(owner, destination->owner) ||
      !same_context(owner, source->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      destination->active_uses.load(std::memory_order_acquire) != 1 ||
      source->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(source->active_uses) &&
         release_exclusive_use(destination->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_silu_bf16_leases(RileyCudaContext* owner,
                                    RileyCudaStream* stream,
                                    RileyCudaDeviceBuffer* input,
                                    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      output == nullptr || input == output || !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) || !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_gated_multiply_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* activated_gate, RileyCudaDeviceBuffer* up,
    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || activated_gate == nullptr ||
      up == nullptr || output == nullptr || activated_gate == up ||
      activated_gate == output || up == output ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, activated_gate->owner) ||
      !same_context(owner, up->owner) || !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      activated_gate->active_uses.load(std::memory_order_acquire) != 1 ||
      up->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(up->active_uses) &&
         release_exclusive_use(activated_gate->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_residual_add_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* left, RileyCudaDeviceBuffer* right,
    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || left == nullptr ||
      right == nullptr || output == nullptr || left == right ||
      left == output || right == output || !same_context(owner, stream->owner) ||
      !same_context(owner, left->owner) || !same_context(owner, right->owner) ||
      !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      left->active_uses.load(std::memory_order_acquire) != 1 ||
      right->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(right->active_uses) &&
         release_exclusive_use(left->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_canonical_rms_norm_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* weight,
    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      weight == nullptr || output == nullptr || input == weight ||
      input == output || weight == output ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) || !same_context(owner, weight->owner) ||
      !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      weight->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(weight->active_uses) &&
         release_exclusive_use(input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_bf16_argmax_leases(RileyCudaContext* owner,
                                      RileyCudaStream* stream,
                                      RileyCudaDeviceBuffer* logits,
                                      RileyCudaDeviceBuffer* results) noexcept {
  if (owner == nullptr || stream == nullptr || logits == nullptr ||
      results == nullptr || logits == results ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, logits->owner) || !same_context(owner, results->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      logits->active_uses.load(std::memory_order_acquire) != 1 ||
      results->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(logits->active_uses) &&
         release_exclusive_use(results->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_bf16_row_gather_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      row_indices == nullptr || output == nullptr || input == row_indices ||
      input == output || row_indices == output ||
      !same_context(owner, stream->owner) || !same_context(owner, input->owner) ||
      !same_context(owner, row_indices->owner) ||
      !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      row_indices->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(row_indices->active_uses) &&
         release_exclusive_use(input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_bf16_row_gather_argmax_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits,
    RileyCudaDeviceBuffer* results) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      row_indices == nullptr || gathered_logits == nullptr || results == nullptr ||
      input == row_indices || input == gathered_logits || input == results ||
      row_indices == gathered_logits || row_indices == results ||
      gathered_logits == results || !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) ||
      !same_context(owner, row_indices->owner) ||
      !same_context(owner, gathered_logits->owner) ||
      !same_context(owner, results->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      row_indices->active_uses.load(std::memory_order_acquire) != 1 ||
      gathered_logits->active_uses.load(std::memory_order_acquire) != 1 ||
      results->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(gathered_logits->active_uses) &&
         release_exclusive_use(row_indices->active_uses) &&
         release_exclusive_use(input->active_uses) &&
         release_exclusive_use(results->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_bf16_row_gather_argmax_d2h_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits, RileyCudaDeviceBuffer* results,
    RileyCudaPinnedHostBuffer* pinned_results) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      row_indices == nullptr || gathered_logits == nullptr || results == nullptr ||
      pinned_results == nullptr || input == row_indices ||
      input == gathered_logits || input == results ||
      row_indices == gathered_logits || row_indices == results ||
      gathered_logits == results || !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) ||
      !same_context(owner, row_indices->owner) ||
      !same_context(owner, gathered_logits->owner) ||
      !same_context(owner, results->owner) ||
      !same_context(owner, pinned_results->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      row_indices->active_uses.load(std::memory_order_acquire) != 1 ||
      gathered_logits->active_uses.load(std::memory_order_acquire) != 1 ||
      results->active_uses.load(std::memory_order_acquire) != 1 ||
      pinned_results->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(pinned_results->active_uses) &&
         release_exclusive_use(gathered_logits->active_uses) &&
         release_exclusive_use(row_indices->active_uses) &&
         release_exclusive_use(input->active_uses) &&
         release_exclusive_use(results->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_indexed_rope_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* cos,
    RileyCudaDeviceBuffer* sin, RileyCudaDeviceBuffer* positions,
    RileyCudaDeviceBuffer* output) noexcept {
  if (owner == nullptr || stream == nullptr || input == nullptr ||
      cos == nullptr || sin == nullptr || positions == nullptr ||
      output == nullptr || input == cos || input == sin ||
      input == positions || input == output || cos == sin ||
      cos == positions || cos == output || sin == positions ||
      sin == output || positions == output ||
      !same_context(owner, stream->owner) ||
      !same_context(owner, input->owner) || !same_context(owner, cos->owner) ||
      !same_context(owner, sin->owner) ||
      !same_context(owner, positions->owner) ||
      !same_context(owner, output->owner) ||
      stream->active_uses.load(std::memory_order_acquire) != 1 ||
      input->active_uses.load(std::memory_order_acquire) != 1 ||
      cos->active_uses.load(std::memory_order_acquire) != 1 ||
      sin->active_uses.load(std::memory_order_acquire) != 1 ||
      positions->active_uses.load(std::memory_order_acquire) != 1 ||
      output->active_uses.load(std::memory_order_acquire) != 1) {
    return false;
  }
  return release_exclusive_use(positions->active_uses) &&
         release_exclusive_use(sin->active_uses) &&
         release_exclusive_use(cos->active_uses) &&
         release_exclusive_use(input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_ragged_paged_kv_cache_write_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* key_pool,
    const RileyCudaRaggedPagedKvCacheWriteBf16State& state) noexcept {
  if (!ragged_paged_kv_cache_write_bf16_state_is_valid_common(
          owner, stream, key_pool, state, true)) {
    return false;
  }
  return release_exclusive_use(state.row_positions->active_uses) &&
         release_exclusive_use(state.row_sequence_slots->active_uses) &&
         release_exclusive_use(state.valid_tokens->active_uses) &&
         release_exclusive_use(state.block_ids->active_uses) &&
         release_exclusive_use(state.sequence_block_offsets->active_uses) &&
         release_exclusive_use(state.value_pool->active_uses) &&
         release_exclusive_use(state.value_source->active_uses) &&
         release_exclusive_use(state.key_source->active_uses) &&
         release_exclusive_use(key_pool->active_uses) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

bool release_graph_canonical_gemm_bf16_leases(
    RileyCudaContext* owner, RileyCudaStream* stream,
    RileyCudaDeviceBuffer* output,
    const RileyCudaCanonicalGemmBf16GraphState& state) noexcept {
  if (state.enqueue_count != 1 ||
      !riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
          owner, stream, output, state, true)) {
    return false;
  }
  return release_exclusive_use(state.workspace->active_uses) &&
         release_exclusive_use(state.weight->active_uses) &&
         release_exclusive_use(state.input->active_uses) &&
         release_exclusive_use(output->active_uses) &&
         riley_cuda_internal::release_canonical_gemm_bf16_graph_plan_lease(
             state.plan) &&
         release_exclusive_use(stream->active_uses) && release_child(owner);
}

// cudaStreamBeginCapture is documented to surface a prior asynchronous CUDA
// error. If it does, use a direct capture-state observation before deciding
// whether the returned owner must be retained for recovery. An observation
// failure itself is ambiguous and therefore treated as an active capture.
bool capture_may_be_active_after_failed_begin(RileyCudaStream* stream) noexcept {
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  const cudaError_t result = cudaStreamIsCapturing(stream->stream, &state);
  return result != cudaSuccess || state != cudaStreamCaptureStatusNone;
}

bool capture_end_is_known(RileyCudaStream* stream) noexcept {
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  return cudaStreamIsCapturing(stream->stream, &state) == cudaSuccess &&
         state == cudaStreamCaptureStatusNone;
}

// Riley cannot safely adopt an already-active capture that was initiated by a
// foreign CUDA caller. Observe the exact stream while its owning context is
// current before creating/publishing a Riley capture owner. An observation
// error is deliberately a denial rather than a guess: begin may otherwise
// return a local abort owner for a foreign graph and destroy foreign work.
RileyCudaStatus require_stream_capture_idle(RileyCudaStream* stream,
                                            RileyCudaErrorInfo* error,
                                            const char* operation) noexcept {
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "stream or its owner is null while observing capture state");
  }
  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        operation);
  cudaStreamCaptureStatus state = cudaStreamCaptureStatusActive;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaStreamIsCapturing(stream->stream, &state), error,
                           RILEY_CUDA_ERROR_STAGE_PREPARE, operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS &&
        state != cudaStreamCaptureStatusNone) {
      status = validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
          "stream already has an active or invalidated foreign CUDA capture");
    }
  }
  return scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                     operation);
}

// C05-5's fixed-buffer entry point deliberately has its own admission path.
// Its capture wrapper and future graph wrapper are both allocated before
// cudaStreamBeginCapture, and both exact resource leases are established
// before any CUDA entry. This keeps capture enqueue allocation-free and means
// the graph can later retain the same device address safely.
RileyCudaStatus capture_begin_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* fill_buffer,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || fill_buffer == nullptr || stream->owner == nullptr ||
      fill_buffer->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "stream, fill buffer, or their owner is null");
  }
  if (!same_context(stream->owner, fill_buffer->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "capture stream and fill buffer belong to different context owners");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > fill_buffer->byte_len / sizeof(float)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill element count exceeds the preallocated buffer");
  }
  if (fill_buffer->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill buffer has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "a stream command batch blocks fixed-fill graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginFillOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(fill_buffer->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "fixed f32 fill buffer has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool buffer_released =
        release_exclusive_use(fill_buffer->active_uses);
    if (!buffer_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a rejected fixed-fill buffer lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginFillOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginFillOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginFillOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginFillOperation,
                     "host allocation failed for fixed-fill capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kFillF32;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(fill_buffer->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginFillOperation,
                     "host allocation failed for captured graph owner");
  }
  capture->prepared_graph =
      new (graph_storage) RileyCudaGraph(stream->owner, stream, fill_buffer,
                                         capture_id,
                                         RileyCudaGraphCaptureOperation::kFillF32);
  capture->fill_buffer = fill_buffer;
  capture->fill_element_count = element_count;
  capture->fill_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool buffer_released = release_capture_fill_lease(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !buffer_released || !child_released ||
        !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a blocked fixed-fill capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool buffer_released = release_capture_fill_lease(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !buffer_released ||
        !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginFillOperation,
                            "failed to release a rejected fixed-fill capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginFillOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginFillOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginFillOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginFillOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);

  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool buffer_released = release_capture_fill_lease(capture);
  const bool capture_released =
      graph_released && buffer_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginFillOperation,
                          "failed to release an unstarted fixed-fill capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-7 is deliberately a sibling admission path rather than an extension of
// the fixed-fill geometry. It captures exactly one whole-allocation H2D node
// and acquires all three permanent resource leases before cudaStreamBeginCapture
// can make their raw pointers observable to CUDA.
RileyCudaStatus capture_begin_h2d_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* destination,
    RileyCudaPinnedHostBuffer* source, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || destination == nullptr || source == nullptr ||
      stream->owner == nullptr || destination->owner == nullptr ||
      source->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "stream, H2D source, destination, or their owner is null");
  }
  if (!same_context(stream->owner, destination->owner) ||
      !same_context(stream->owner, source->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "capture stream, H2D source, and destination must share one context owner");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "only thread-local capture mode is admitted");
  }
  if (source->byte_len == 0 || source->byte_len != destination->byte_len) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D requires equal nonzero whole source and destination slabs");
  }
  if (source->host_data == nullptr || destination->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D source or destination has no live allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "a stream command batch blocks fixed-address graph H2D capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginH2DOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(source->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D source has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(destination->active_uses)) {
    const bool source_released = release_exclusive_use(source->active_uses);
    if (!source_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a rejected graph H2D source lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "graph H2D destination has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool destination_released =
        release_exclusive_use(destination->active_uses);
    const bool source_released = release_exclusive_use(source->active_uses);
    if (!destination_released || !source_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release rejected graph H2D resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginH2DOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginH2DOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginH2DOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginH2DOperation,
                     "host allocation failed for graph H2D capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kH2D;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(destination->active_uses);
    (void)release_exclusive_use(source->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginH2DOperation,
                     "host allocation failed for captured graph H2D owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, destination, capture_id,
      RileyCudaGraphCaptureOperation::kH2D, source, source->byte_len);
  capture->fill_buffer = destination;
  capture->fill_lease_held = true;
  capture->h2d_source = source;
  capture->h2d_byte_len = source->byte_len;
  capture->h2d_source_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_h2d_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a blocked graph H2D capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_h2d_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginH2DOperation,
                            "failed to release a rejected graph H2D capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginH2DOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginH2DOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginH2DOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginH2DOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_h2d_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginH2DOperation,
                          "failed to release an unstarted graph H2D capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-8 is intentionally a separate sibling admission path. The two device
// pointers and fixed BF16 geometry are prepared before capture begins, and both
// device leases remain held through graph/exec close. It does not generalize
// the eager subspan/aliasing SiLU ABI.
RileyCudaStatus capture_begin_silu_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* output, uint64_t element_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || output == nullptr ||
      stream->owner == nullptr || input->owner == nullptr ||
      output->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "stream, BF16 SiLU input, output, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "capture stream, BF16 SiLU input, and output must share one context owner");
  }
  if (input == output) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU requires distinct input and output allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "fixed BF16 SiLU element count exceeds an input or output allocation");
  }
  if (input->device_data == nullptr || output->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU input or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "a stream command batch blocks fixed-address BF16 SiLU graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginSiluBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!input_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginSiluBf16Operation,
                            "failed to release a rejected graph BF16 SiLU input lease");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "graph BF16 SiLU output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!output_released || !input_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginSiluBf16Operation,
                            "failed to release rejected graph BF16 SiLU resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginSiluBf16Operation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginSiluBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginSiluBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginSiluBf16Operation,
                     "host allocation failed for graph BF16 SiLU capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kSiluBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginSiluBf16Operation,
                     "host allocation failed for captured graph BF16 SiLU owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kSiluBf16, nullptr, 0, input,
      element_count);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->silu_input = input;
  capture->silu_element_count = element_count;
  capture->silu_input_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_silu_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginSiluBf16Operation,
          "failed to release a blocked graph BF16 SiLU capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_silu_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginSiluBf16Operation,
          "failed to release a rejected graph BF16 SiLU capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginSiluBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginSiluBf16Operation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginSiluBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginSiluBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_silu_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginSiluBf16Operation,
                          "failed to release an unstarted graph BF16 SiLU capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-10 is a sibling of C05-8, not a fused graph. The three device pointers
// and fixed BF16 geometry are established before capture begins and remain
// leased through graph/exec close. It does not generalize eager spans,
// aliasing, fresh inputs, or SiLU-plus-multiply fusion.
RileyCudaStatus capture_begin_gated_multiply_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* activated_gate,
    RileyCudaDeviceBuffer* up, RileyCudaDeviceBuffer* output,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginGatedMultiplyBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || activated_gate == nullptr || up == nullptr ||
      output == nullptr || stream->owner == nullptr ||
      activated_gate->owner == nullptr || up->owner == nullptr ||
      output->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "stream, BF16 activated gate, up, output, or their owner is null");
  }
  if (!same_context(stream->owner, activated_gate->owner) ||
      !same_context(stream->owner, up->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "capture stream and BF16 gated-multiply allocations must share one context owner");
  }
  if (activated_gate == up || activated_gate == output || up == output) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "graph BF16 gated multiply requires three distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginGatedMultiplyBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > activated_gate->byte_len / sizeof(__nv_bfloat16) ||
      element_count > up->byte_len / sizeof(__nv_bfloat16) ||
      element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "fixed BF16 gated-multiply element count exceeds an input or output allocation");
  }
  if (activated_gate->device_data == nullptr || up->device_data == nullptr ||
      output->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "graph BF16 gated-multiply input or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "a stream command batch blocks fixed-address BF16 gated-multiply graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginGatedMultiplyBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(activated_gate->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "graph BF16 activated-gate input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(up->active_uses)) {
    const bool activated_gate_released =
        release_exclusive_use(activated_gate->active_uses);
    if (!activated_gate_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
          "failed to release a rejected graph BF16 activated-gate lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "graph BF16 up input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool up_released = release_exclusive_use(up->active_uses);
    const bool activated_gate_released =
        release_exclusive_use(activated_gate->active_uses);
    if (!up_released || !activated_gate_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
          "failed to release rejected graph BF16 gated-multiply input leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "graph BF16 gated-multiply output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool up_released = release_exclusive_use(up->active_uses);
    const bool activated_gate_released =
        release_exclusive_use(activated_gate->active_uses);
    if (!output_released || !up_released || !activated_gate_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
          "failed to release rejected graph BF16 gated-multiply resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(up->active_uses);
    (void)release_exclusive_use(activated_gate->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginGatedMultiplyBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(up->active_uses);
    (void)release_exclusive_use(activated_gate->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginGatedMultiplyBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(up->active_uses);
    (void)release_exclusive_use(activated_gate->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginGatedMultiplyBf16Operation,
        "host allocation failed for graph BF16 gated-multiply capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(up->active_uses);
    (void)release_exclusive_use(activated_gate->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginGatedMultiplyBf16Operation,
        "host allocation failed for captured graph BF16 gated-multiply owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kGatedMultiplyBf16, nullptr, 0, nullptr,
      0, activated_gate, up, element_count);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->gated_multiply_activated_gate = activated_gate;
  capture->gated_multiply_up = up;
  capture->gated_multiply_element_count = element_count;
  capture->gated_multiply_activated_gate_lease_held = true;
  capture->gated_multiply_up_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_gated_multiply_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
          "failed to release a blocked graph BF16 gated-multiply capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_gated_multiply_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
          "failed to release a rejected graph BF16 gated-multiply capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginGatedMultiplyBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginGatedMultiplyBf16Operation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginGatedMultiplyBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginGatedMultiplyBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_gated_multiply_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginGatedMultiplyBf16Operation,
        "failed to release an unstarted graph BF16 gated-multiply capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-11 is a sibling of C05-10, not a fused residual-plus-normalization
// graph. The two fixed BF16 inputs and fixed BF16 output remain independently
// leased for the entire capture/graph/exec lifecycle. It deliberately admits
// neither aliases, offsets, fresh replay inputs, nor a fused RMSNorm.
RileyCudaStatus capture_begin_residual_add_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* left,
    RileyCudaDeviceBuffer* right, RileyCudaDeviceBuffer* output,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginResidualAddBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || left == nullptr || right == nullptr ||
      output == nullptr || stream->owner == nullptr || left->owner == nullptr ||
      right->owner == nullptr || output->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "stream, BF16 residual left, right, output, or their owner is null");
  }
  if (!same_context(stream->owner, left->owner) ||
      !same_context(stream->owner, right->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "capture stream and BF16 residual-add allocations must share one context owner");
  }
  if (left == right || left == output || right == output) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "graph BF16 residual add requires three distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginResidualAddBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  if (element_count == 0 ||
      element_count > left->byte_len / sizeof(__nv_bfloat16) ||
      element_count > right->byte_len / sizeof(__nv_bfloat16) ||
      element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "fixed BF16 residual-add element count exceeds an input or output allocation");
  }
  if (left->device_data == nullptr || right->device_data == nullptr ||
      output->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "graph BF16 residual-add input or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "a stream command batch blocks fixed-address BF16 residual-add graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginResidualAddBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(left->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "graph BF16 residual-left input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(right->active_uses)) {
    const bool left_released = release_exclusive_use(left->active_uses);
    if (!left_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
          "failed to release a rejected graph BF16 residual-left lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "graph BF16 residual-right input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool right_released = release_exclusive_use(right->active_uses);
    const bool left_released = release_exclusive_use(left->active_uses);
    if (!right_released || !left_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
          "failed to release rejected graph BF16 residual-add input leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "graph BF16 residual-add output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool right_released = release_exclusive_use(right->active_uses);
    const bool left_released = release_exclusive_use(left->active_uses);
    if (!output_released || !right_released || !left_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
          "failed to release rejected graph BF16 residual-add resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(right->active_uses);
    (void)release_exclusive_use(left->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginResidualAddBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(right->active_uses);
    (void)release_exclusive_use(left->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginResidualAddBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(right->active_uses);
    (void)release_exclusive_use(left->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginResidualAddBf16Operation,
        "host allocation failed for graph BF16 residual-add capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kResidualAddBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(right->active_uses);
    (void)release_exclusive_use(left->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginResidualAddBf16Operation,
        "host allocation failed for captured graph BF16 residual-add owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kResidualAddBf16, nullptr, 0, nullptr,
      0, nullptr, nullptr, 0, left, right, element_count);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->residual_add_left = left;
  capture->residual_add_right = right;
  capture->residual_add_element_count = element_count;
  capture->residual_add_left_lease_held = true;
  capture->residual_add_right_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_residual_add_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
          "failed to release a blocked graph BF16 residual-add capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_residual_add_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
          "failed to release a rejected graph BF16 residual-add capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginResidualAddBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginResidualAddBf16Operation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginResidualAddBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginResidualAddBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_residual_add_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginResidualAddBf16Operation,
        "failed to release an unstarted graph BF16 residual-add capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-12 deliberately captures only the generic eager BF16 RMSNorm contract.
// Its three fixed allocations and exact reduction geometry stay leased for the
// complete capture/graph/exec lifetime; profile-specific and fused variants
// use different arithmetic contracts and therefore do not enter this path.
RileyCudaStatus capture_begin_canonical_rms_norm_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* weight, RileyCudaDeviceBuffer* output,
    uint64_t row_count, uint64_t hidden_size, float epsilon,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalRmsNormBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || weight == nullptr ||
      output == nullptr || stream->owner == nullptr || input->owner == nullptr ||
      weight->owner == nullptr || output->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "stream, canonical RMSNorm input, weight, output, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, weight->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "capture stream and canonical RMSNorm allocations must share one context owner");
  }
  if (input == weight || input == output || weight == output) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "graph canonical RMSNorm requires three distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalRmsNormBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t element_count = 0;
  if (!canonical_rms_norm_element_count(row_count, hidden_size, &element_count) ||
      element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      hidden_size > weight->byte_len / sizeof(__nv_bfloat16) ||
      element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "canonical RMSNorm rows, hidden size, or allocation capacity is invalid");
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalRmsNormBf16Operation,
                            "canonical RMSNorm epsilon must be finite and positive");
  }
  if (input->device_data == nullptr || weight->device_data == nullptr ||
      output->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "canonical RMSNorm input, weight, or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "a stream command batch blocks fixed-address canonical RMSNorm graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginCanonicalRmsNormBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "canonical RMSNorm input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(weight->active_uses)) {
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
          "failed to release a rejected canonical RMSNorm input lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "canonical RMSNorm weight has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool weight_released = release_exclusive_use(weight->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!weight_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
          "failed to release rejected canonical RMSNorm input leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "canonical RMSNorm output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool weight_released = release_exclusive_use(weight->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!output_released || !weight_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
          "failed to release rejected canonical RMSNorm resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(weight->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginCanonicalRmsNormBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(weight->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginCanonicalRmsNormBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(weight->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginCanonicalRmsNormBf16Operation,
        "host allocation failed for canonical RMSNorm graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(weight->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginCanonicalRmsNormBf16Operation,
        "host allocation failed for captured canonical RMSNorm graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16, nullptr, 0,
      nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, input, weight,
      row_count, hidden_size, epsilon);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->canonical_rms_norm_input = input;
  capture->canonical_rms_norm_weight = weight;
  capture->canonical_rms_norm_row_count = row_count;
  capture->canonical_rms_norm_hidden_size = hidden_size;
  capture->canonical_rms_norm_epsilon = epsilon;
  capture->canonical_rms_norm_input_lease_held = true;
  capture->canonical_rms_norm_weight_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_canonical_rms_norm_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
          "failed to release a blocked canonical RMSNorm graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_canonical_rms_norm_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
          "failed to release a rejected canonical RMSNorm graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalRmsNormBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginCanonicalRmsNormBf16Operation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginCanonicalRmsNormBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginCanonicalRmsNormBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_canonical_rms_norm_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalRmsNormBf16Operation,
        "failed to release an unstarted canonical RMSNorm graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-13 captures one exact eager-equivalent deterministic BF16 argmax. It
// deliberately owns only fixed logits/result addresses and has no C07 row
// gather, host result handling, sampling, or executor connection.
RileyCudaStatus capture_begin_bf16_argmax_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* logits,
    RileyCudaDeviceBuffer* results, uint64_t row_count,
    uint64_t vocabulary_size, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16ArgmaxOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || logits == nullptr || results == nullptr ||
      stream->owner == nullptr || logits->owner == nullptr ||
      results->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "stream, BF16 logits, U32 results, or their owner is null");
  }
  if (!same_context(stream->owner, logits->owner) ||
      !same_context(stream->owner, results->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "capture stream, logits, and results must share one context owner");
  }
  if (logits == results) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "graph deterministic BF16 argmax requires distinct logits and result allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16ArgmaxOperation,
                            "only thread-local capture mode is admitted");
  }
  if (row_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16ArgmaxOperation,
                            "graph deterministic BF16 argmax requires nonzero row_count");
  }
  if (vocabulary_size == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16ArgmaxOperation,
                            "vocabulary_size must be greater than zero");
  }
  if (vocabulary_size > UINT32_MAX) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16ArgmaxOperation,
                            "vocabulary_size exceeds the U32 token-id contract");
  }
  uint64_t logit_element_count = 0;
  if (!bf16_argmax_shape_is_valid(row_count, vocabulary_size,
                                  &logit_element_count) ||
      logit_element_count > logits->byte_len / sizeof(__nv_bfloat16) ||
      row_count > results->byte_len / sizeof(RileyCudaBf16ArgmaxResult)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "deterministic BF16 argmax shape overflows or exceeds logits/results capacity");
  }
  if (logits->device_data == nullptr || results->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "graph deterministic BF16 argmax logits or results has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "a stream command batch blocks fixed-address deterministic BF16 argmax graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginBf16ArgmaxOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(logits->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "graph deterministic BF16 argmax logits has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(results->active_uses)) {
    const bool logits_released = release_exclusive_use(logits->active_uses);
    if (!logits_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16ArgmaxOperation,
          "failed to release a rejected graph deterministic BF16 argmax logits lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "graph deterministic BF16 argmax results has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool results_released = release_exclusive_use(results->active_uses);
    const bool logits_released = release_exclusive_use(logits->active_uses);
    if (!results_released || !logits_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16ArgmaxOperation,
          "failed to release rejected graph deterministic BF16 argmax resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(logits->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16ArgmaxOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(logits->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16ArgmaxOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(logits->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16ArgmaxOperation,
        "host allocation failed for deterministic BF16 argmax graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kBf16Argmax;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(logits->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16ArgmaxOperation,
        "host allocation failed for captured deterministic BF16 argmax graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, results, capture_id,
      RileyCudaGraphCaptureOperation::kBf16Argmax, nullptr, 0, nullptr, 0,
      nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, 0,
      0.0F, logits, row_count, vocabulary_size);
  capture->fill_buffer = results;
  capture->fill_lease_held = true;
  capture->bf16_argmax_logits = logits;
  capture->bf16_argmax_row_count = row_count;
  capture->bf16_argmax_vocabulary_size = vocabulary_size;
  capture->bf16_argmax_logits_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_bf16_argmax_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16ArgmaxOperation,
          "failed to release a blocked deterministic BF16 argmax graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_bf16_argmax_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16ArgmaxOperation,
          "failed to release a rejected deterministic BF16 argmax graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16ArgmaxOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginBf16ArgmaxOperation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginBf16ArgmaxOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginBf16ArgmaxOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_bf16_argmax_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16ArgmaxOperation,
        "failed to release an unstarted deterministic BF16 argmax graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-14 is deliberately a narrow fixed-address capture: BF16 input,
// contiguous U32 row indices, and BF16 output are leased before CUDA capture
// begins and remain immutable through graph/exec close. The native raw ABI
// does not retain a host index mirror; invalid device indices are handled by
// the captured kernel exactly as eager row_gather does.
RileyCudaStatus capture_begin_bf16_row_gather_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices, RileyCudaDeviceBuffer* output,
    uint64_t input_row_count, uint64_t output_row_count,
    uint64_t column_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherOperation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || row_indices == nullptr ||
      output == nullptr || stream->owner == nullptr || input->owner == nullptr ||
      row_indices->owner == nullptr || output->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "stream, BF16 input, U32 row indices, BF16 output, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, row_indices->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "capture stream, BF16 input, U32 row indices, and BF16 output must share one context owner");
  }
  if (input == row_indices || input == output || row_indices == output) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "graph BF16 row gather requires distinct input, row-index, and output allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherOperation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t input_element_count = 0;
  uint64_t output_element_count = 0;
  if (!bf16_row_gather_shape_is_valid(
          input_row_count, output_row_count, column_count,
          &input_element_count, &output_element_count) ||
      input_element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      output_row_count > row_indices->byte_len / sizeof(uint32_t) ||
      output_element_count > output->byte_len / sizeof(__nv_bfloat16)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "BF16 row-gather geometry is zero, overflows, or exceeds input/index/output capacity");
  }
  if (input->device_data == nullptr || row_indices->device_data == nullptr ||
      output->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "graph BF16 row-gather input, indices, or output has no live device allocation");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "a stream command batch blocks fixed-address BF16 row-gather graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginBf16RowGatherOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "graph BF16 row-gather input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(row_indices->active_uses)) {
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
          "failed to release a rejected graph BF16 row-gather input lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "graph BF16 row-gather indices has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(output->active_uses)) {
    const bool indices_released =
        release_exclusive_use(row_indices->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!indices_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
          "failed to release rejected graph BF16 row-gather input/index leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "graph BF16 row-gather output has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool output_released = release_exclusive_use(output->active_uses);
    const bool indices_released =
        release_exclusive_use(row_indices->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!output_released || !indices_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
          "failed to release rejected graph BF16 row-gather resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16RowGatherOperation,
        "host allocation failed for BF16 row-gather graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kBf16RowGather;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(output->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16RowGatherOperation,
        "host allocation failed for captured BF16 row-gather graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kBf16RowGather, nullptr, 0, nullptr, 0,
      nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, 0,
      0.0F, nullptr, 0, 0, input, row_indices, input_row_count,
      output_row_count, column_count);
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->bf16_row_gather_input = input;
  capture->bf16_row_gather_indices = row_indices;
  capture->bf16_row_gather_input_row_count = input_row_count;
  capture->bf16_row_gather_output_row_count = output_row_count;
  capture->bf16_row_gather_column_count = column_count;
  capture->bf16_row_gather_input_lease_held = true;
  capture->bf16_row_gather_indices_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_bf16_row_gather_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
          "failed to release a blocked BF16 row-gather graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_bf16_row_gather_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
          "failed to release a rejected BF16 row-gather graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginBf16RowGatherOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginBf16RowGatherOperation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginBf16RowGatherOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginBf16RowGatherOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_bf16_row_gather_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginBf16RowGatherOperation,
        "failed to release an unstarted BF16 row-gather graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-15 captures exactly a fixed-address row-gather node followed by the
// deterministic C05-13 BF16 argmax node. The gathered logits slab is an
// explicit fourth device allocation rather than an implicit temporary, so the
// dependency and every raw address remain stable through graph replay.
RileyCudaStatus capture_begin_bf16_row_gather_argmax_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits,
    RileyCudaDeviceBuffer* results, uint64_t input_row_count,
    uint64_t output_row_count, uint64_t vocabulary_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherArgmaxOperation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || row_indices == nullptr ||
      gathered_logits == nullptr || results == nullptr ||
      stream->owner == nullptr || input->owner == nullptr ||
      row_indices->owner == nullptr || gathered_logits->owner == nullptr ||
      results->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "stream, input, row indices, gathered logits, results, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, row_indices->owner) ||
      !same_context(stream->owner, gathered_logits->owner) ||
      !same_context(stream->owner, results->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "capture stream, input, row indices, gathered logits, and results must share one context owner");
  }
  if (input == row_indices || input == gathered_logits || input == results ||
      row_indices == gathered_logits || row_indices == results ||
      gathered_logits == results) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax requires four distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherArgmaxOperation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  if (!bf16_row_gather_argmax_shape_is_valid(
          input_row_count, output_row_count, vocabulary_size,
          &input_element_count, &gathered_element_count) ||
      input_element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      output_row_count > row_indices->byte_len / sizeof(uint32_t) ||
      gathered_element_count >
          gathered_logits->byte_len / sizeof(__nv_bfloat16) ||
      output_row_count >
          results->byte_len / sizeof(RileyCudaBf16ArgmaxResult)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "BF16 row-gather argmax geometry is zero, overflows, or exceeds fixed allocation capacity");
  }
  if (input->device_data == nullptr || row_indices->device_data == nullptr ||
      gathered_logits->device_data == nullptr || results->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax allocation has no live device storage");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "a stream command batch blocks fixed-address BF16 row-gather argmax graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginBf16RowGatherArgmaxOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax input has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(row_indices->active_uses)) {
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release a rejected BF16 row-gather argmax input lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax indices has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(gathered_logits->active_uses)) {
    const bool indices_released =
        release_exclusive_use(row_indices->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!indices_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release rejected BF16 row-gather argmax input/index leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax gathered logits has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(results->active_uses)) {
    const bool gathered_released =
        release_exclusive_use(gathered_logits->active_uses);
    const bool indices_released =
        release_exclusive_use(row_indices->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!gathered_released || !indices_released || !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release rejected BF16 row-gather argmax resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "graph BF16 row-gather argmax results has an active asynchronous use");
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool results_released = release_exclusive_use(results->active_uses);
    const bool gathered_released =
        release_exclusive_use(gathered_logits->active_uses);
    const bool indices_released =
        release_exclusive_use(row_indices->active_uses);
    const bool input_released = release_exclusive_use(input->active_uses);
    if (!results_released || !gathered_released || !indices_released ||
        !input_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release rejected BF16 row-gather argmax resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(gathered_logits->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherArgmaxOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(gathered_logits->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherArgmaxOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(gathered_logits->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16RowGatherArgmaxOperation,
        "host allocation failed for BF16 row-gather argmax graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    (void)release_exclusive_use(results->active_uses);
    (void)release_exclusive_use(gathered_logits->active_uses);
    (void)release_exclusive_use(row_indices->active_uses);
    (void)release_exclusive_use(input->active_uses);
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginBf16RowGatherArgmaxOperation,
        "host allocation failed for captured BF16 row-gather argmax graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, results, capture_id,
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax, nullptr, 0,
      nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0.0F, nullptr, 0, 0, nullptr, nullptr, 0, 0, 0,
      input, row_indices, gathered_logits, input_row_count, output_row_count,
      vocabulary_size);
  capture->fill_buffer = results;
  capture->fill_lease_held = true;
  capture->bf16_row_gather_argmax_input = input;
  capture->bf16_row_gather_argmax_indices = row_indices;
  capture->bf16_row_gather_argmax_gathered_logits = gathered_logits;
  capture->bf16_row_gather_argmax_input_row_count = input_row_count;
  capture->bf16_row_gather_argmax_output_row_count = output_row_count;
  capture->bf16_row_gather_argmax_vocabulary_size = vocabulary_size;
  capture->bf16_row_gather_argmax_input_lease_held = true;
  capture->bf16_row_gather_argmax_indices_lease_held = true;
  capture->bf16_row_gather_argmax_gathered_logits_lease_held = true;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release a blocked BF16 row-gather argmax graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxOperation,
          "failed to release a rejected BF16 row-gather argmax graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kBeginBf16RowGatherArgmaxOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginBf16RowGatherArgmaxOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginBf16RowGatherArgmaxOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_bf16_row_gather_argmax_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kBeginBf16RowGatherArgmaxOperation,
        "failed to release an unstarted BF16 row-gather argmax graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-16 retains the C05-15 dependency chain and one exact pinned result
// destination. The ownership setup intentionally finishes before
// cudaStreamBeginCapture so every captured raw address is already leased.
RileyCudaStatus capture_begin_bf16_row_gather_argmax_d2h_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits, RileyCudaDeviceBuffer* results,
    RileyCudaPinnedHostBuffer* pinned_results, uint64_t input_row_count,
    uint64_t output_row_count, uint64_t vocabulary_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || row_indices == nullptr ||
      gathered_logits == nullptr || results == nullptr ||
      pinned_results == nullptr || stream->owner == nullptr ||
      input->owner == nullptr || row_indices->owner == nullptr ||
      gathered_logits->owner == nullptr || results->owner == nullptr ||
      pinned_results->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "stream, four device allocations, pinned results, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, row_indices->owner) ||
      !same_context(stream->owner, gathered_logits->owner) ||
      !same_context(stream->owner, results->owner) ||
      !same_context(stream->owner, pinned_results->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "capture stream, device allocations, and pinned results must share one context owner");
  }
  if (input == row_indices || input == gathered_logits || input == results ||
      row_indices == gathered_logits || row_indices == results ||
      gathered_logits == results) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H requires four distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t input_element_count = 0;
  uint64_t gathered_element_count = 0;
  uint64_t result_byte_len = 0;
  if (!bf16_row_gather_argmax_d2h_shape_is_valid(
          input_row_count, output_row_count, vocabulary_size,
          &input_element_count, &gathered_element_count, &result_byte_len) ||
      input_element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      output_row_count > row_indices->byte_len / sizeof(uint32_t) ||
      gathered_element_count >
          gathered_logits->byte_len / sizeof(__nv_bfloat16) ||
      result_byte_len > results->byte_len ||
      pinned_results->byte_len != result_byte_len) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "BF16 row-gather argmax D2H geometry, capacity, or exact pinned result length is invalid");
  }
  if (input->device_data == nullptr || row_indices->device_data == nullptr ||
      gathered_logits->device_data == nullptr || results->device_data == nullptr ||
      pinned_results->host_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H allocation has no live storage");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "a stream command batch blocks fixed-address BF16 row-gather argmax D2H graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginBf16RowGatherArgmaxD2HOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  bool input_lease_held = false;
  bool indices_lease_held = false;
  bool gathered_lease_held = false;
  bool results_lease_held = false;
  bool pinned_results_lease_held = false;
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    if (pinned_results_lease_held) {
      released = release_exclusive_use(pinned_results->active_uses) && released;
      pinned_results_lease_held = false;
    }
    if (results_lease_held) {
      released = release_exclusive_use(results->active_uses) && released;
      results_lease_held = false;
    }
    if (gathered_lease_held) {
      released = release_exclusive_use(gathered_logits->active_uses) && released;
      gathered_lease_held = false;
    }
    if (indices_lease_held) {
      released = release_exclusive_use(row_indices->active_uses) && released;
      indices_lease_held = false;
    }
    if (input_lease_held) {
      released = release_exclusive_use(input->active_uses) && released;
      input_lease_held = false;
    }
    return released;
  };
  if (!try_acquire_exclusive_use(input->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H input has an active asynchronous use");
  }
  input_lease_held = true;
  if (!try_acquire_exclusive_use(row_indices->active_uses)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release a rejected BF16 row-gather argmax D2H input lease");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H indices has an active asynchronous use");
  }
  indices_lease_held = true;
  if (!try_acquire_exclusive_use(gathered_logits->active_uses)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release rejected BF16 row-gather argmax D2H input/index leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H gathered logits has an active asynchronous use");
  }
  gathered_lease_held = true;
  if (!try_acquire_exclusive_use(results->active_uses)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release rejected BF16 row-gather argmax D2H device leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H results has an active asynchronous use");
  }
  results_lease_held = true;
  if (!try_acquire_exclusive_use(pinned_results->active_uses)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release rejected BF16 row-gather argmax D2H device-result leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "graph BF16 row-gather argmax D2H pinned results has an active copy token");
  }
  pinned_results_lease_held = true;
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release rejected BF16 row-gather argmax D2H resource leases");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherArgmaxD2HOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16RowGatherArgmaxD2HOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool released = release_initial_leases();
    if (!child_released || !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "failed to release graph capture allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginBf16RowGatherArgmaxD2HOperation,
                     "host allocation failed for BF16 row-gather argmax D2H graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation =
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool released = release_initial_leases();
    if (!child_released || !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16RowGatherArgmaxD2HOperation,
                            "failed to release captured graph allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginBf16RowGatherArgmaxD2HOperation,
                     "host allocation failed for captured BF16 row-gather argmax D2H graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, results, capture_id,
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H, nullptr, 0,
      nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0.0F, nullptr, 0, 0, nullptr, nullptr, 0, 0, 0,
      nullptr, nullptr, nullptr, 0, 0, 0, input, row_indices,
      gathered_logits, pinned_results, input_row_count, output_row_count,
      vocabulary_size, result_byte_len);
  capture->fill_buffer = results;
  capture->fill_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_input = input;
  capture->bf16_row_gather_argmax_d2h_indices = row_indices;
  capture->bf16_row_gather_argmax_d2h_gathered_logits = gathered_logits;
  capture->bf16_row_gather_argmax_d2h_pinned_results = pinned_results;
  capture->bf16_row_gather_argmax_d2h_input_row_count = input_row_count;
  capture->bf16_row_gather_argmax_d2h_output_row_count = output_row_count;
  capture->bf16_row_gather_argmax_d2h_vocabulary_size = vocabulary_size;
  capture->bf16_row_gather_argmax_d2h_result_byte_len = result_byte_len;
  capture->bf16_row_gather_argmax_d2h_input_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_indices_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held = true;
  // Ownership moves to the capture fields; rollback is now deliberately
  // operation-specific so the prepared graph can validate every raw address.
  input_lease_held = false;
  indices_lease_held = false;
  gathered_lease_held = false;
  results_lease_held = false;
  pinned_results_lease_held = false;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_d2h_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release a blocked BF16 row-gather argmax D2H graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_d2h_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16RowGatherArgmaxD2HOperation,
          "failed to release a rejected BF16 row-gather argmax D2H graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kBeginBf16RowGatherArgmaxD2HOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginBf16RowGatherArgmaxD2HOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginBf16RowGatherArgmaxD2HOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_bf16_row_gather_argmax_d2h_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kBeginBf16RowGatherArgmaxD2HOperation,
        "failed to release an unstarted BF16 row-gather argmax D2H graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-20 keeps eager embedding's validation/status boundary inside one fixed
// graph. The C05-16 ledger supplies the five permanent leases without
// reusing C05-16 arithmetic or result interpretation.
RileyCudaStatus capture_begin_bf16_embedding_status_d2h_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* table,
    RileyCudaDeviceBuffer* token_ids, RileyCudaDeviceBuffer* output,
    RileyCudaDeviceBuffer* device_error_scratch,
    RileyCudaPinnedHostBuffer* pinned_report, uint64_t token_count,
    uint64_t vocabulary_size, uint64_t hidden_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16EmbeddingStatusD2HOperation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || table == nullptr || token_ids == nullptr ||
      output == nullptr || device_error_scratch == nullptr ||
      pinned_report == nullptr || stream->owner == nullptr ||
      table->owner == nullptr || token_ids->owner == nullptr ||
      output->owner == nullptr || device_error_scratch->owner == nullptr ||
      pinned_report->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "stream, four embedding device allocations, pinned report, or their owner is null");
  }
  RileyCudaDeviceBuffer* const buffers[] = {
      table, token_ids, output, device_error_scratch,
  };
  constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (!same_context(stream->owner, buffers[index]->owner)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginBf16EmbeddingStatusD2HOperation,
          "capture stream and embedding device allocations must share one context owner");
    }
    for (size_t other = index + 1; other < kBufferCount; ++other) {
      if (buffers[index] == buffers[other]) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
            RILEY_CUDA_ERROR_STAGE_VALIDATION,
            kBeginBf16EmbeddingStatusD2HOperation,
            "graph BF16 embedding status D2H requires four distinct device allocations");
      }
    }
  }
  if (!same_context(stream->owner, pinned_report->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "capture stream and pinned embedding report must share one context owner");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16EmbeddingStatusD2HOperation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t table_element_count = 0;
  uint64_t output_element_count = 0;
  if (!bf16_embedding_status_d2h_shape_is_valid(
          vocabulary_size, token_count, hidden_size, &table_element_count,
          &output_element_count) ||
      table_element_count > table->byte_len / sizeof(__nv_bfloat16) ||
      token_count > token_ids->byte_len / sizeof(uint32_t) ||
      output_element_count > output->byte_len / sizeof(__nv_bfloat16) ||
      device_error_scratch->byte_len !=
          sizeof(RileyCudaEmbeddingErrorReport) ||
      reinterpret_cast<uintptr_t>(device_error_scratch->device_data) %
              alignof(RileyCudaEmbeddingErrorReport) !=
          0 ||
      pinned_report->byte_len != sizeof(RileyCudaEmbeddingErrorReport)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "BF16 embedding status D2H geometry, exact report capacity, or alignment is invalid");
  }
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index]->device_data == nullptr) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginBf16EmbeddingStatusD2HOperation,
          "graph BF16 embedding status D2H allocation has no live device storage");
    }
  }
  if (pinned_report->host_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "graph BF16 embedding status D2H report has no live pinned host storage");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "a stream command batch blocks fixed-address BF16 embedding status D2H graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginBf16EmbeddingStatusD2HOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  bool resource_leases_held[kBufferCount] = {};
  bool pinned_lease_held = false;
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    if (pinned_lease_held) {
      released = release_exclusive_use(pinned_report->active_uses) && released;
      pinned_lease_held = false;
    }
    for (size_t index = kBufferCount; index != 0; --index) {
      const size_t resource_index = index - 1;
      if (resource_leases_held[resource_index]) {
        released = release_exclusive_use(
                       buffers[resource_index]->active_uses) &&
                   released;
        resource_leases_held[resource_index] = false;
      }
    }
    return released;
  };
  const auto reject_busy = [&](const char* message) noexcept {
    if (!release_initial_leases()) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16EmbeddingStatusD2HOperation,
          "failed to release rejected BF16 embedding status D2H graph resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginBf16EmbeddingStatusD2HOperation, message);
  };
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (!try_acquire_exclusive_use(buffers[index]->active_uses)) {
      return reject_busy(
          "a fixed BF16 embedding status D2H device allocation has an active asynchronous use");
    }
    resource_leases_held[index] = true;
  }
  if (!try_acquire_exclusive_use(pinned_report->active_uses)) {
    return reject_busy(
        "the fixed BF16 embedding status D2H pinned report has an active copy token");
  }
  pinned_lease_held = true;
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return reject_busy("stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16EmbeddingStatusD2HOperation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16EmbeddingStatusD2HOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginBf16EmbeddingStatusD2HOperation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginBf16EmbeddingStatusD2HOperation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16EmbeddingStatusD2HOperation,
          "failed to release graph capture allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginBf16EmbeddingStatusD2HOperation,
                     "host allocation failed for BF16 embedding status D2H graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16EmbeddingStatusD2HOperation,
          "failed to release captured graph allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginBf16EmbeddingStatusD2HOperation,
                     "host allocation failed for captured BF16 embedding status D2H graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H, nullptr, 0,
      nullptr, 0, nullptr, nullptr, 0, nullptr, nullptr, 0, nullptr,
      nullptr, 0, 0, 0.0F, nullptr, 0, 0, nullptr, nullptr, 0, 0, 0,
      nullptr, nullptr, nullptr, 0, 0, 0, table, token_ids,
      device_error_scratch, pinned_report, vocabulary_size, token_count,
      hidden_size, sizeof(RileyCudaEmbeddingErrorReport));
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_input = table;
  capture->bf16_row_gather_argmax_d2h_indices = token_ids;
  capture->bf16_row_gather_argmax_d2h_gathered_logits = device_error_scratch;
  capture->bf16_row_gather_argmax_d2h_pinned_results = pinned_report;
  capture->bf16_row_gather_argmax_d2h_input_row_count = vocabulary_size;
  capture->bf16_row_gather_argmax_d2h_output_row_count = token_count;
  capture->bf16_row_gather_argmax_d2h_vocabulary_size = hidden_size;
  capture->bf16_row_gather_argmax_d2h_result_byte_len =
      sizeof(RileyCudaEmbeddingErrorReport);
  capture->bf16_row_gather_argmax_d2h_input_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_indices_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_gathered_logits_lease_held = true;
  capture->bf16_row_gather_argmax_d2h_pinned_results_lease_held = true;
  for (size_t index = 0; index < kBufferCount; ++index) {
    resource_leases_held[index] = false;
  }
  pinned_lease_held = false;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_d2h_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16EmbeddingStatusD2HOperation,
          "failed to release a blocked BF16 embedding status D2H graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_bf16_row_gather_argmax_d2h_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginBf16EmbeddingStatusD2HOperation,
          "failed to release a rejected BF16 embedding status D2H graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginBf16EmbeddingStatusD2HOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kBeginBf16EmbeddingStatusD2HOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginBf16EmbeddingStatusD2HOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginBf16EmbeddingStatusD2HOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }
  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_bf16_row_gather_argmax_d2h_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kBeginBf16EmbeddingStatusD2HOperation,
        "failed to release an unstarted BF16 embedding status D2H graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-17 captures the eager indexed-RoPE BF16 arithmetic into a single fixed
// node. All capacity and host-mirror checks finish before any lease is held;
// only device pointers and immutable geometry are retained after capture
// starts, so this slice cannot become an implicit H2D positions update path.
RileyCudaStatus capture_begin_indexed_rope_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* cos, RileyCudaDeviceBuffer* sin,
    RileyCudaDeviceBuffer* positions, RileyCudaDeviceBuffer* output,
    const uint32_t* positions_mirror, uint64_t positions_mirror_len,
    uint64_t active_row_count, uint64_t head_count, uint64_t head_size,
    uint64_t rotary_dimension, uint64_t table_position_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginIndexedRopeBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || input == nullptr || cos == nullptr || sin == nullptr ||
      positions == nullptr || output == nullptr || stream->owner == nullptr ||
      input->owner == nullptr || cos->owner == nullptr || sin->owner == nullptr ||
      positions->owner == nullptr || output->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "stream, five indexed-RoPE device allocations, or their owner is null");
  }
  if (!same_context(stream->owner, input->owner) ||
      !same_context(stream->owner, cos->owner) ||
      !same_context(stream->owner, sin->owner) ||
      !same_context(stream->owner, positions->owner) ||
      !same_context(stream->owner, output->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "capture stream and indexed-RoPE allocations must share one context owner");
  }
  if (input == cos || input == sin || input == positions || input == output ||
      cos == sin || cos == positions || cos == output || sin == positions ||
      sin == output || positions == output) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "graph BF16 indexed-RoPE requires five distinct device allocations");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginIndexedRopeBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t tensor_element_count = 0;
  uint64_t table_element_count = 0;
  uint64_t work_item_count = 0;
  if (!indexed_rope_bf16_shape_is_valid(
          active_row_count, head_count, head_size, rotary_dimension,
          table_position_count, &tensor_element_count, &table_element_count,
          &work_item_count) ||
      tensor_element_count > input->byte_len / sizeof(__nv_bfloat16) ||
      tensor_element_count > output->byte_len / sizeof(__nv_bfloat16) ||
      table_element_count > cos->byte_len / sizeof(float) ||
      table_element_count > sin->byte_len / sizeof(float) ||
      active_row_count > positions->byte_len / sizeof(uint32_t)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "BF16 indexed-RoPE geometry is zero, invalid, overflows, or exceeds fixed capacity");
  }
  if (positions_mirror == nullptr || positions_mirror_len != active_row_count) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "indexed-RoPE positions mirror must be non-null and exactly active_row_count entries");
  }
  for (uint64_t row = 0; row < positions_mirror_len; ++row) {
    if (static_cast<uint64_t>(positions_mirror[row]) >= table_position_count) {
      return validation_error(
          error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
          "indexed-RoPE positions mirror contains a table-position out of range");
    }
  }
  if (input->device_data == nullptr || cos->device_data == nullptr ||
      sin->device_data == nullptr || positions->device_data == nullptr ||
      output->device_data == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "graph BF16 indexed-RoPE allocation has no live device storage");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "a stream command batch blocks fixed-address BF16 indexed-RoPE graph capture");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginIndexedRopeBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  bool input_lease_held = false;
  bool cos_lease_held = false;
  bool sin_lease_held = false;
  bool positions_lease_held = false;
  bool output_lease_held = false;
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    if (output_lease_held) {
      released = release_exclusive_use(output->active_uses) && released;
      output_lease_held = false;
    }
    if (positions_lease_held) {
      released = release_exclusive_use(positions->active_uses) && released;
      positions_lease_held = false;
    }
    if (sin_lease_held) {
      released = release_exclusive_use(sin->active_uses) && released;
      sin_lease_held = false;
    }
    if (cos_lease_held) {
      released = release_exclusive_use(cos->active_uses) && released;
      cos_lease_held = false;
    }
    if (input_lease_held) {
      released = release_exclusive_use(input->active_uses) && released;
      input_lease_held = false;
    }
    return released;
  };
  const auto reject_busy = [&](const char* resource) noexcept {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release rejected BF16 indexed-RoPE leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginIndexedRopeBf16Operation, resource);
  };
  if (!try_acquire_exclusive_use(input->active_uses)) {
    return reject_busy("graph BF16 indexed-RoPE input has an active asynchronous use");
  }
  input_lease_held = true;
  if (!try_acquire_exclusive_use(cos->active_uses)) {
    return reject_busy("graph BF16 indexed-RoPE cosine table has an active asynchronous use");
  }
  cos_lease_held = true;
  if (!try_acquire_exclusive_use(sin->active_uses)) {
    return reject_busy("graph BF16 indexed-RoPE sine table has an active asynchronous use");
  }
  sin_lease_held = true;
  if (!try_acquire_exclusive_use(positions->active_uses)) {
    return reject_busy("graph BF16 indexed-RoPE positions has an active asynchronous use");
  }
  positions_lease_held = true;
  if (!try_acquire_exclusive_use(output->active_uses)) {
    return reject_busy("graph BF16 indexed-RoPE output has an active asynchronous use");
  }
  output_lease_held = true;
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return reject_busy("stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginIndexedRopeBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    const bool released = release_initial_leases();
    if (!released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginIndexedRopeBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool released = release_initial_leases();
    if (!child_released || !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release graph capture allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginIndexedRopeBf16Operation,
                     "host allocation failed for BF16 indexed-RoPE graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool released = release_initial_leases();
    if (!child_released || !released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release captured graph allocation rollback leases");
    }
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE,
                     kBeginIndexedRopeBf16Operation,
                     "host allocation failed for captured BF16 indexed-RoPE graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kIndexedRopeBf16);
  RileyCudaGraph* const graph = capture->prepared_graph;
  graph->indexed_rope_bf16_input = input;
  graph->indexed_rope_bf16_cos = cos;
  graph->indexed_rope_bf16_sin = sin;
  graph->indexed_rope_bf16_positions = positions;
  graph->indexed_rope_bf16_active_row_count = active_row_count;
  graph->indexed_rope_bf16_head_count = head_count;
  graph->indexed_rope_bf16_head_size = head_size;
  graph->indexed_rope_bf16_rotary_dimension = rotary_dimension;
  graph->indexed_rope_bf16_table_position_count = table_position_count;
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->indexed_rope_bf16_input = input;
  capture->indexed_rope_bf16_cos = cos;
  capture->indexed_rope_bf16_sin = sin;
  capture->indexed_rope_bf16_positions = positions;
  capture->indexed_rope_bf16_active_row_count = active_row_count;
  capture->indexed_rope_bf16_head_count = head_count;
  capture->indexed_rope_bf16_head_size = head_size;
  capture->indexed_rope_bf16_rotary_dimension = rotary_dimension;
  capture->indexed_rope_bf16_table_position_count = table_position_count;
  capture->indexed_rope_bf16_input_lease_held = true;
  capture->indexed_rope_bf16_cos_lease_held = true;
  capture->indexed_rope_bf16_sin_lease_held = true;
  capture->indexed_rope_bf16_positions_lease_held = true;
  input_lease_held = false;
  cos_lease_held = false;
  sin_lease_held = false;
  positions_lease_held = false;
  output_lease_held = false;

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_indexed_rope_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release a blocked BF16 indexed-RoPE graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released = release_capture_indexed_rope_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginIndexedRopeBf16Operation,
                            "failed to release a rejected BF16 indexed-RoPE graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginIndexedRopeBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE, kBeginIndexedRopeBf16Operation,
      capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginIndexedRopeBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginIndexedRopeBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released = release_capture_indexed_rope_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                          kBeginIndexedRopeBf16Operation,
                          "failed to release an unstarted BF16 indexed-RoPE graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-18 captures exactly the existing eager ragged paged-KV write kernel.
// The safe Rust boundary uses a temporary PackedBatchHostV1 witness, but this
// native owner retains only fixed device allocations and immutable geometry;
// it never stages or retains host metadata.
RileyCudaStatus capture_begin_ragged_paged_kv_cache_write_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* key_source,
    RileyCudaDeviceBuffer* value_source, RileyCudaDeviceBuffer* key_pool,
    RileyCudaDeviceBuffer* value_pool,
    RileyCudaDeviceBuffer* sequence_block_offsets,
    RileyCudaDeviceBuffer* block_ids, RileyCudaDeviceBuffer* valid_tokens,
    RileyCudaDeviceBuffer* row_sequence_slots,
    RileyCudaDeviceBuffer* row_positions, uint64_t sequence_count,
    uint64_t block_count, uint64_t active_row_count,
    uint64_t physical_block_count, uint64_t key_value_head_count,
    uint64_t head_size, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "stream or its owner is null");
  }
  RileyCudaDeviceBuffer* const buffers[] = {
      key_source,           value_source,          key_pool,
      value_pool,           sequence_block_offsets, block_ids,
      valid_tokens,         row_sequence_slots,    row_positions,
  };
  constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index] == nullptr || buffers[index]->owner == nullptr) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "one of the nine fixed ragged paged-KV allocations or its owner is null");
    }
    if (!same_context(stream->owner, buffers[index]->owner)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "capture stream and ragged paged-KV allocations must share one context owner");
    }
    for (size_t other = index + 1; other < kBufferCount; ++other) {
      if (buffers[index] == buffers[other]) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
            RILEY_CUDA_ERROR_STAGE_VALIDATION,
            kBeginRaggedPagedKvCacheWriteBf16Operation,
            "graph BF16 ragged paged-KV write requires nine distinct device allocations");
      }
    }
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "only thread-local capture mode is admitted");
  }
  uint64_t source_element_count = 0;
  uint64_t pool_element_count = 0;
  if (!ragged_paged_kv_cache_write_bf16_shape_is_valid(
          sequence_count, block_count, active_row_count, physical_block_count,
          key_value_head_count, head_size, &source_element_count,
          &pool_element_count) ||
      source_element_count > key_source->byte_len / sizeof(__nv_bfloat16) ||
      source_element_count > value_source->byte_len / sizeof(__nv_bfloat16) ||
      pool_element_count > key_pool->byte_len / sizeof(__nv_bfloat16) ||
      pool_element_count > value_pool->byte_len / sizeof(__nv_bfloat16) ||
      sequence_count + 1 >
          sequence_block_offsets->byte_len / sizeof(uint32_t) ||
      block_count > block_ids->byte_len / sizeof(uint32_t) ||
      block_count > valid_tokens->byte_len / sizeof(uint16_t) ||
      active_row_count > row_sequence_slots->byte_len / sizeof(uint32_t) ||
      active_row_count > row_positions->byte_len / sizeof(uint32_t)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "ragged paged-KV geometry is zero, invalid, overflows, or exceeds fixed capacity");
  }
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index]->device_data == nullptr) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "a graph BF16 ragged paged-KV allocation has no live device storage");
    }
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "a stream command batch blocks fixed-address BF16 ragged paged-KV graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginRaggedPagedKvCacheWriteBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  bool resource_leases_held[kBufferCount] = {};
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    for (size_t index = kBufferCount; index != 0; --index) {
      const size_t resource_index = index - 1;
      if (resource_leases_held[resource_index]) {
        released = release_exclusive_use(
                       buffers[resource_index]->active_uses) &&
                   released;
        resource_leases_held[resource_index] = false;
      }
    }
    return released;
  };
  const auto reject_busy = [&](const char* message) noexcept {
    if (!release_initial_leases()) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "failed to release rejected BF16 ragged paged-KV graph resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            message);
  };
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (!try_acquire_exclusive_use(buffers[index]->active_uses)) {
      return reject_busy(
          "a fixed ragged paged-KV graph allocation has an active asynchronous use");
    }
    resource_leases_held[index] = true;
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return reject_busy("stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginRaggedPagedKvCacheWriteBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginRaggedPagedKvCacheWriteBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "failed to release graph capture allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "host allocation failed for BF16 ragged paged-KV graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation =
      RileyCudaGraphCaptureOperation::kRaggedPagedKvCacheWriteBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginRaggedPagedKvCacheWriteBf16Operation,
                            "failed to release captured graph allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "host allocation failed for captured BF16 ragged paged-KV graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, key_pool, capture_id,
      RileyCudaGraphCaptureOperation::kRaggedPagedKvCacheWriteBf16);
  RileyCudaGraph* const graph = capture->prepared_graph;
  RileyCudaRaggedPagedKvCacheWriteBf16State graph_state{};
  graph_state.key_source = key_source;
  graph_state.value_source = value_source;
  graph_state.key_pool = key_pool;
  graph_state.value_pool = value_pool;
  graph_state.sequence_block_offsets = sequence_block_offsets;
  graph_state.block_ids = block_ids;
  graph_state.valid_tokens = valid_tokens;
  graph_state.row_sequence_slots = row_sequence_slots;
  graph_state.row_positions = row_positions;
  graph_state.sequence_count = sequence_count;
  graph_state.block_count = block_count;
  graph_state.active_row_count = active_row_count;
  graph_state.physical_block_count = physical_block_count;
  graph_state.key_value_head_count = key_value_head_count;
  graph_state.head_size = head_size;
  graph->ragged_paged_kv_write_bf16 = graph_state;
  capture->fill_buffer = key_pool;
  capture->fill_lease_held = true;
  capture->ragged_paged_kv_write_bf16 = graph_state;
  capture->ragged_paged_kv_write_bf16.key_source_lease_held = true;
  capture->ragged_paged_kv_write_bf16.value_source_lease_held = true;
  capture->ragged_paged_kv_write_bf16.value_pool_lease_held = true;
  capture->ragged_paged_kv_write_bf16.sequence_block_offsets_lease_held = true;
  capture->ragged_paged_kv_write_bf16.block_ids_lease_held = true;
  capture->ragged_paged_kv_write_bf16.valid_tokens_lease_held = true;
  capture->ragged_paged_kv_write_bf16.row_sequence_slots_lease_held = true;
  capture->ragged_paged_kv_write_bf16.row_positions_lease_held = true;
  for (size_t index = 0; index < kBufferCount; ++index) {
    resource_leases_held[index] = false;
  }

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "failed to release a blocked BF16 ragged paged-KV graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginRaggedPagedKvCacheWriteBf16Operation,
          "failed to release a rejected BF16 ragged paged-KV graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kBeginRaggedPagedKvCacheWriteBf16Operation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(
          begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
          kBeginRaggedPagedKvCacheWriteBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginRaggedPagedKvCacheWriteBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kBeginRaggedPagedKvCacheWriteBf16Operation,
        "failed to release an unstarted BF16 ragged paged-KV graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-19 captures the eager grouped D64 paged-attention dispatch without
// retaining the Rust-side PackedBatch host witness. The immutable native
// ledger intentionally uses C05-18's nine-buffer ownership slots: query is
// key_source, K pool is value_source, output is the legacy key_pool/fill slot,
// and V pool remains value_pool. That keeps one permanent lease per fixed
// graph address through capture, graph, and exec ownership.
RileyCudaStatus capture_begin_grouped_ragged_paged_attention_bf16_impl(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* query,
    RileyCudaDeviceBuffer* key_pool, RileyCudaDeviceBuffer* value_pool,
    RileyCudaDeviceBuffer* output,
    RileyCudaDeviceBuffer* sequence_block_offsets,
    RileyCudaDeviceBuffer* block_ids, RileyCudaDeviceBuffer* valid_tokens,
    RileyCudaDeviceBuffer* row_sequence_slots,
    RileyCudaDeviceBuffer* row_positions, uint64_t sequence_count,
    uint64_t block_count, uint64_t active_row_count,
    uint64_t physical_block_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t output_row_count, float scale,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "stream or its owner is null");
  }
  RileyCudaDeviceBuffer* const buffers[] = {
      query, key_pool, value_pool, output, sequence_block_offsets, block_ids,
      valid_tokens, row_sequence_slots, row_positions,
  };
  constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index] == nullptr || buffers[index]->owner == nullptr) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "one of the nine fixed grouped ragged-attention allocations or its owner is null");
    }
    if (!same_context(stream->owner, buffers[index]->owner)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "capture stream and grouped ragged-attention allocations must share one context owner");
    }
    for (size_t other = index + 1; other < kBufferCount; ++other) {
      if (buffers[index] == buffers[other]) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
            RILEY_CUDA_ERROR_STAGE_VALIDATION,
            kBeginGroupedRaggedPagedAttentionBf16Operation,
            "graph grouped ragged attention requires nine distinct device allocations");
      }
    }
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "only thread-local capture mode is admitted");
  }
  uint64_t query_element_count = 0;
  uint64_t pool_element_count = 0;
  uint64_t output_element_count = 0;
  if (!grouped_ragged_paged_attention_bf16_shape_is_valid(
          sequence_count, block_count, active_row_count, physical_block_count,
          query_head_count, key_value_head_count, output_row_count, scale,
          &query_element_count, &pool_element_count, &output_element_count) ||
      query_element_count > query->byte_len / sizeof(__nv_bfloat16) ||
      pool_element_count > key_pool->byte_len / sizeof(__nv_bfloat16) ||
      pool_element_count > value_pool->byte_len / sizeof(__nv_bfloat16) ||
      output_element_count > output->byte_len / sizeof(__nv_bfloat16) ||
      sequence_count + 1 >
          sequence_block_offsets->byte_len / sizeof(uint32_t) ||
      block_count > block_ids->byte_len / sizeof(uint32_t) ||
      block_count > valid_tokens->byte_len / sizeof(uint16_t) ||
      active_row_count > row_sequence_slots->byte_len / sizeof(uint32_t) ||
      active_row_count > row_positions->byte_len / sizeof(uint32_t)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "grouped ragged-attention geometry is zero, invalid, overflows, or exceeds fixed capacity");
  }
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (buffers[index]->device_data == nullptr) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "a graph grouped ragged-attention allocation has no live device storage");
    }
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "a stream command batch blocks fixed-address grouped ragged-attention graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginGroupedRaggedPagedAttentionBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  bool resource_leases_held[kBufferCount] = {};
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    for (size_t index = kBufferCount; index != 0; --index) {
      const size_t resource_index = index - 1;
      if (resource_leases_held[resource_index]) {
        released = release_exclusive_use(
                       buffers[resource_index]->active_uses) &&
                   released;
        resource_leases_held[resource_index] = false;
      }
    }
    return released;
  };
  const auto reject_busy = [&](const char* message) noexcept {
    if (!release_initial_leases()) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "failed to release rejected grouped ragged-attention graph resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginGroupedRaggedPagedAttentionBf16Operation,
                            message);
  };
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (!try_acquire_exclusive_use(buffers[index]->active_uses)) {
      return reject_busy(
          "a fixed grouped ragged-attention graph allocation has an active asynchronous use");
    }
    resource_leases_held[index] = true;
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return reject_busy("stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginGroupedRaggedPagedAttentionBf16Operation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginGroupedRaggedPagedAttentionBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginGroupedRaggedPagedAttentionBf16Operation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginGroupedRaggedPagedAttentionBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "failed to release graph capture allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "host allocation failed for grouped ragged-attention graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation =
      RileyCudaGraphCaptureOperation::kGroupedRaggedPagedAttentionBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "failed to release captured graph allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "host allocation failed for captured grouped ragged-attention graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kGroupedRaggedPagedAttentionBf16);
  RileyCudaGraph* const graph = capture->prepared_graph;
  RileyCudaRaggedPagedKvCacheWriteBf16State graph_state{};
  graph_state.key_source = query;
  graph_state.value_source = key_pool;
  graph_state.key_pool = output;
  graph_state.value_pool = value_pool;
  graph_state.sequence_block_offsets = sequence_block_offsets;
  graph_state.block_ids = block_ids;
  graph_state.valid_tokens = valid_tokens;
  graph_state.row_sequence_slots = row_sequence_slots;
  graph_state.row_positions = row_positions;
  graph_state.sequence_count = sequence_count;
  graph_state.block_count = block_count;
  graph_state.active_row_count = active_row_count;
  graph_state.physical_block_count = physical_block_count;
  graph_state.key_value_head_count = key_value_head_count;
  graph_state.head_size = kGraphRaggedAttentionHeadSize;
  graph_state.query_head_count = query_head_count;
  graph_state.output_row_count = output_row_count;
  graph_state.attention_scale = scale;
  graph_state.grouped_attention = true;
  graph->ragged_paged_kv_write_bf16 = graph_state;
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->ragged_paged_kv_write_bf16 = graph_state;
  capture->ragged_paged_kv_write_bf16.key_source_lease_held = true;
  capture->ragged_paged_kv_write_bf16.value_source_lease_held = true;
  capture->ragged_paged_kv_write_bf16.value_pool_lease_held = true;
  capture->ragged_paged_kv_write_bf16.sequence_block_offsets_lease_held =
      true;
  capture->ragged_paged_kv_write_bf16.block_ids_lease_held = true;
  capture->ragged_paged_kv_write_bf16.valid_tokens_lease_held = true;
  capture->ragged_paged_kv_write_bf16.row_sequence_slots_lease_held = true;
  capture->ragged_paged_kv_write_bf16.row_positions_lease_held = true;
  for (size_t index = 0; index < kBufferCount; ++index) {
    resource_leases_held[index] = false;
  }

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "failed to release a blocked grouped ragged-attention graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginGroupedRaggedPagedAttentionBf16Operation,
          "failed to release a rejected grouped ragged-attention graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kBeginGroupedRaggedPagedAttentionBf16Operation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(
          begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
          kBeginGroupedRaggedPagedAttentionBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginGroupedRaggedPagedAttentionBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }
  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_ragged_paged_kv_cache_write_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kBeginGroupedRaggedPagedAttentionBf16Operation,
        "failed to release an unstarted grouped ragged-attention graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

// C05-21 records one real cuBLASLt matmul into a thread-local CUDA Graph.
// The opaque GEMM plan remains owned by its public plan handle; this graph
// holds only its exclusive-use lease alongside the four fixed allocations.
RileyCudaStatus capture_begin_canonical_gemm_bf16_impl(
    RileyCudaStream* stream, RileyCudaGemmPlan* plan,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* weight,
    RileyCudaDeviceBuffer* output, RileyCudaDeviceBuffer* workspace,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalGemmBf16Operation,
                            "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalGemmBf16Operation,
                            "stream or its owner is null");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalGemmBf16Operation,
                            "only thread-local capture mode is admitted");
  }

  RileyCudaCanonicalGemmBf16GraphState graph_state{};
  RileyCudaStatus status =
      riley_cuda_internal::preflight_canonical_gemm_bf16_graph_state(
          plan, stream, input, weight, output, workspace, &graph_state,
          error, kBeginCanonicalGemmBf16Operation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch() || command_batch_is_active(stream)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "a stream command batch blocks canonical cuBLASLt GEMM graph capture");
  }
  const RileyCudaStatus idle_status = require_stream_capture_idle(
      stream, error, kBeginCanonicalGemmBf16Operation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }

  RileyCudaDeviceBuffer* const buffers[] = {input, weight, output, workspace};
  constexpr size_t kBufferCount = sizeof(buffers) / sizeof(buffers[0]);
  bool plan_lease_held = false;
  bool resource_leases_held[kBufferCount] = {};
  bool stream_lease_held = false;
  const auto release_initial_leases = [&]() noexcept {
    bool released = true;
    if (stream_lease_held) {
      released = release_exclusive_use(stream->active_uses) && released;
      stream_lease_held = false;
    }
    for (size_t index = kBufferCount; index != 0; --index) {
      const size_t resource_index = index - 1;
      if (resource_leases_held[resource_index]) {
        released = release_exclusive_use(
                       buffers[resource_index]->active_uses) &&
                   released;
        resource_leases_held[resource_index] = false;
      }
    }
    if (plan_lease_held) {
      released =
                     riley_cuda_internal::release_canonical_gemm_bf16_graph_plan_lease(
                         plan) &&
                 released;
      plan_lease_held = false;
    }
    return released;
  };
  const auto reject_busy = [&](const char* message) noexcept {
    if (!release_initial_leases()) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginCanonicalGemmBf16Operation,
          "failed to release rejected canonical GEMM graph resource leases");
    }
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginCanonicalGemmBf16Operation, message);
  };

  status = riley_cuda_internal::acquire_canonical_gemm_bf16_graph_plan_lease(
      plan, error, kBeginCanonicalGemmBf16Operation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  plan_lease_held = true;
  for (size_t index = 0; index < kBufferCount; ++index) {
    if (!try_acquire_exclusive_use(buffers[index]->active_uses)) {
      return reject_busy(
          "a fixed canonical GEMM graph allocation has an active asynchronous use");
    }
    resource_leases_held[index] = true;
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return reject_busy("stream has an active asynchronous use or capture");
  }
  stream_lease_held = true;

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginCanonicalGemmBf16Operation,
                            "failed to release exhausted graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginCanonicalGemmBf16Operation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    if (!release_initial_leases()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginCanonicalGemmBf16Operation,
                            "failed to release rejected graph capture resource leases");
    }
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginCanonicalGemmBf16Operation,
                          "context child-resource counter overflow");
  }
  void* capture_storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (capture_storage == nullptr) {
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginCanonicalGemmBf16Operation,
                            "failed to release graph capture allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginCanonicalGemmBf16Operation,
        "host allocation failed for canonical GEMM graph capture owner");
  }
  auto* capture = new (capture_storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  capture->operation = RileyCudaGraphCaptureOperation::kCanonicalGemmBf16;
  void* graph_storage = std::calloc(1, sizeof(RileyCudaGraph));
  if (graph_storage == nullptr) {
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    const bool child_released = release_child(stream->owner);
    const bool leases_released = release_initial_leases();
    if (!child_released || !leases_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginCanonicalGemmBf16Operation,
                            "failed to release captured graph allocation rollback leases");
    }
    return set_error(
        error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
        RILEY_CUDA_ERROR_DOMAIN_INTERNAL, RILEY_CUDA_ERROR_STAGE_CREATE,
        kBeginCanonicalGemmBf16Operation,
        "host allocation failed for captured canonical GEMM graph owner");
  }
  capture->prepared_graph = new (graph_storage) RileyCudaGraph(
      stream->owner, stream, output, capture_id,
      RileyCudaGraphCaptureOperation::kCanonicalGemmBf16);
  capture->prepared_graph->canonical_gemm_bf16 = graph_state;
  capture->fill_buffer = output;
  capture->fill_lease_held = true;
  capture->canonical_gemm_bf16 = graph_state;
  capture->canonical_gemm_bf16.plan_lease_held = true;
  capture->canonical_gemm_bf16.input_lease_held = true;
  capture->canonical_gemm_bf16.weight_lease_held = true;
  capture->canonical_gemm_bf16.workspace_lease_held = true;
  plan_lease_held = false;
  for (size_t index = 0; index < kBufferCount; ++index) {
    resource_leases_held[index] = false;
  }

  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_canonical_gemm_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!graph_released || !leases_released || !child_released ||
        !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginCanonicalGemmBf16Operation,
          "failed to release a blocked canonical GEMM graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool graph_released = destroy_prepared_graph_storage(capture);
    const bool leases_released =
        release_capture_canonical_gemm_bf16_leases(capture);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_initial_leases();
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !graph_released || !leases_released ||
        !child_released || !stream_released) {
      return internal_error(
          error, RILEY_CUDA_ERROR_STAGE_CLOSE,
          kBeginCanonicalGemmBf16Operation,
          "failed to release a rejected canonical GEMM graph capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginCanonicalGemmBf16Operation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginCanonicalGemmBf16Operation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result = cudaStreamBeginCapture(
        stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginCanonicalGemmBf16Operation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginCanonicalGemmBf16Operation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (capture_may_be_active) {
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  const bool graph_released = destroy_prepared_graph_storage(capture);
  const bool leases_released =
      release_capture_canonical_gemm_bf16_leases(capture);
  const bool capture_released =
      graph_released && leases_released && release_capture_owner(capture);
  if (!capture_released) {
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginCanonicalGemmBf16Operation,
        "failed to release an unstarted canonical GEMM graph capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_graph_capture_query_capability(
    RileyCudaGraphCaptureOperationKind operation,
    RileyCudaGraphCaptureCapability* out_capability,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  // This query is deliberately outside every context/stream/capture path. It
  // is evidence for a named C05 vertical slice, not a runtime probe or an
  // admission shortcut for a larger graph.
  clear_error(error);
  if (out_capability != nullptr) {
    *out_capability = RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN;
  }
  if (out_capability == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kQueryCaptureCapabilityOperation,
                            "out_capability is null");
  }
  switch (operation) {
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_FILL_F32:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_H2D:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_SILU_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GATED_MULTIPLY_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RESIDUAL_ADD_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_RMS_NORM_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ARGMAX:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER_ARGMAX:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_ROW_GATHER_ARGMAX_D2H:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_BF16_EMBEDDING_STATUS_D2H:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_INDEXED_ROPE_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_RAGGED_PAGED_KV_CACHE_WRITE_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_GROUPED_RAGGED_PAGED_ATTENTION_BF16:
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_CANONICAL_GEMM_BF16:
      *out_capability = RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED;
      break;
    case RILEY_CUDA_GRAPH_CAPTURE_OPERATION_KIND_UNKNOWN:
    default:
      // A newer caller can ask this older archive about a newer operation
      // value. Return the closed unknown value so it cannot accidentally
      // inherit capture admission from a sibling operation.
      break;
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin(
    RileyCudaStream* stream, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_capture != nullptr) {
    *out_capture = nullptr;
  }
  if (out_capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation, "out_capture is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN);
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation, "stream or its owner is null");
  }
  if (mode != RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "only thread-local capture mode is admitted");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "a prior CUDA context-stack restoration failed");
  }
  if (thread_has_active_graph_capture()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }
  if (thread_has_active_command_batch()) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread has an active CUDA stream command batch");
  }
  if (command_batch_is_active(stream)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "stream already has an active command batch");
  }
  const RileyCudaStatus idle_status =
      require_stream_capture_idle(stream, error, kBeginOperation);
  if (idle_status != RILEY_CUDA_STATUS_SUCCESS) {
    return idle_status;
  }
  if (!try_acquire_exclusive_use(stream->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kBeginOperation,
                            "stream has an active asynchronous use or capture");
  }

  const uint64_t capture_id = next_graph_capture_id();
  if (capture_id == 0) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginOperation,
                          "CUDA Graph capture ID space is exhausted");
  }
  if (!retain_child(stream->owner)) {
    (void)release_exclusive_use(stream->active_uses);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kBeginOperation,
                          "context child-resource counter overflow");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaGraphCapture));
  if (storage == nullptr) {
    (void)release_child(stream->owner);
    (void)release_exclusive_use(stream->active_uses);
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kBeginOperation,
                     "host allocation failed");
  }
  auto* capture = new (storage) RileyCudaGraphCapture{
      stream->owner, stream, stream->owner->capture_domain,
      native_thread_token(), capture_id};
  if (!try_begin_capture_domain(capture->capture_domain)) {
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginOperation,
                            "failed to release a blocked capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "the CUDA primary context has a pending copy, fill, or broad control operation");
  }
  if (!try_publish_thread_graph_capture(capture)) {
    const bool domain_released =
        release_capture_domain_capture(capture->capture_domain);
    const bool child_released = release_child(stream->owner);
    const bool stream_released = release_exclusive_use(stream->active_uses);
    capture->~RileyCudaGraphCapture();
    std::free(capture);
    if (!domain_released || !child_released || !stream_released) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kBeginOperation,
                            "failed to release a rejected capture owner");
    }
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kBeginOperation,
        "this host thread already owns a thread-local CUDA Graph capture");
  }

  CurrentContext scope(stream->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                                        kBeginOperation, capture);
  bool capture_may_be_active = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const cudaError_t begin_result =
        cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_result == cudaSuccess) {
      capture->capture_started = true;
      capture_may_be_active = true;
    } else {
      status = runtime_error(begin_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                             kBeginOperation);
      capture_may_be_active = capture_may_be_active_after_failed_begin(stream);
      capture->capture_started = capture_may_be_active;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kBeginOperation);
  const bool restoration_known =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);

  if (capture_may_be_active) {
    // A non-success status can still accompany an entered capture if CUDA
    // surfaced a deferred asynchronous error. Returning the owner permits the
    // safe Rust boundary to run the same one-shot abort/recovery path before it
    // reports that error. A restoration failure marks that owner poisoned but
    // is likewise retained rather than silently abandoning active capture.
    *out_capture = capture;
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, capture_id,
                           false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                      !restoration_known);
    return status;
  }

  if (!release_capture_owner(capture)) {
    // The capture never began, but an ownership-counter corruption must still
    // strand the stream lease rather than permit unsound reuse.
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE, kBeginOperation,
                          "failed to release an unstarted capture owner");
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN, 0, true,
                         !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_abort(
    RileyCudaGraphCapture** capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kAbortOperation, "capture pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kAbortOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  if (*capture == nullptr) {
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, 0, true,
                           false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  RileyCudaGraphCapture* const owner = *capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture owner has a null context or stream");
  }
  if (owner->owner_thread != native_thread_token()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kAbortOperation,
                            "thread-local capture must end on its begin thread");
  }
  if (!thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kAbortOperation,
        "the supplied capture owner is not active on this host thread");
  }
  if (!owner->capture_started) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture owner was not marked active");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kAbortOperation,
                          "capture stream lease was corrupted");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kAbortOperation, owner);
  bool end_attempted = false;
  bool termination_known = false;
  bool graph_release_known = false;
  cudaGraph_t returned_graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    end_attempted = true;
    const cudaError_t end_result =
        cudaStreamEndCapture(owner->stream->stream, &returned_graph);
    if (end_result == cudaSuccess) {
      termination_known = true;
    } else {
      status = runtime_error(end_result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                             kAbortOperation);
      termination_known = capture_end_is_known(owner->stream);
    }
    if (termination_known) {
      graph_release_known = true;
      if (returned_graph != nullptr) {
        const cudaError_t destroy_result = cudaGraphDestroy(returned_graph);
        if (destroy_result != cudaSuccess) {
          // cudaGraphDestroy may report a deferred error after consuming the
          // resource. Preserve the opaque graph only in the intentionally
          // leaked owner and never issue a second destroy attempt.
          owner->unreleased_graph = returned_graph;
          graph_release_known = false;
          if (status == RILEY_CUDA_STATUS_SUCCESS) {
            status = runtime_error(destroy_result, error,
                                   RILEY_CUDA_ERROR_STAGE_CLOSE,
                                   kAbortOperation);
          }
        }
      }
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kAbortOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);

  if (!end_attempted) {
    // No CUDA end attempt occurred, so a raw caller may retry after correcting
    // the precondition. The safe Rust wrapper deliberately abandons this
    // handle instead of retrying from Drop.
    record_capture_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                           false, !restoration_known);
    return status;
  }

  // End capture is a one-shot CUDA lifecycle transition. Consume the caller's
  // raw handle before reporting any result, even when capture termination or
  // graph destruction cannot be proven. The retained owner/lease below is an
  // intentional fail-closed leak, never a retryable dangling pointer.
  *capture = nullptr;

  // The ThreadLocal gate remains published after cudaStreamEndCapture,
  // cudaGraphDestroy, and capture-context restoration are all known. Drain
  // only in that state, before releasing the graph child, stream lease, domain
  // admission, or TLS owner. Each callback receives `owner` as the exact
  // CurrentContext bypass and can close a resource belonging to another Riley
  // context. A failed drain intentionally strands its remaining FIFO plus all
  // capture leases; reissuing CUDA closes after an ambiguous error is unsafe.
  bool released = false;
  if (termination_known && graph_release_known && restoration_known) {
    // CUDA has physically left capture and the transient graph is gone, but
    // the exact TLS owner intentionally remains published until deferred safe
    // resource cleanup succeeds. This narrowly permits a childless foreign
    // context lease release through the matching capture-domain control gate.
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS) {
      const bool is_h2d =
          owner->operation == RileyCudaGraphCaptureOperation::kH2D;
      const bool is_silu_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
      const bool is_gated_multiply_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
      const bool is_residual_add_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kResidualAddBf16;
      const bool is_canonical_rms_norm_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
      const bool is_bf16_argmax =
          owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
      const bool is_bf16_row_gather =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kBf16RowGather;
      const bool is_bf16_row_gather_argmax =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
      const bool is_bf16_row_gather_argmax_d2h =
          is_bf16_d2h_operation(owner->operation);
      const bool is_indexed_rope_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
      const bool is_canonical_gemm_bf16 =
          is_canonical_gemm_bf16_operation(owner->operation);
      const bool is_ragged_paged_kv_write_bf16 =
          is_ragged_paged_fixed_address_operation(owner->operation);
      const bool is_fill_or_generic =
          owner->operation == RileyCudaGraphCaptureOperation::kFillF32 ||
          owner->operation == RileyCudaGraphCaptureOperation::kNone;
      const bool release_graph_first =
          is_h2d || is_silu_bf16 || is_gated_multiply_bf16 ||
          is_residual_add_bf16 || is_canonical_rms_norm_bf16 ||
          is_bf16_argmax || is_bf16_row_gather ||
          is_bf16_row_gather_argmax || is_bf16_row_gather_argmax_d2h ||
          is_indexed_rope_bf16 || is_canonical_gemm_bf16 ||
          is_ragged_paged_kv_write_bf16;
      const bool prepared_graph_released =
          release_graph_first ? destroy_prepared_graph_storage(owner) : true;
      const bool operation_released =
          prepared_graph_released &&
          (is_h2d ? release_capture_h2d_leases(owner)
                  : is_silu_bf16 ? release_capture_silu_bf16_leases(owner)
                                 : is_gated_multiply_bf16
                                       ? release_capture_gated_multiply_bf16_leases(
                                             owner)
                                 : is_residual_add_bf16
                                       ? release_capture_residual_add_bf16_leases(
                                             owner)
                                 : is_canonical_rms_norm_bf16
                                       ? release_capture_canonical_rms_norm_bf16_leases(
                                             owner)
                                 : is_bf16_argmax
                                       ? release_capture_bf16_argmax_leases(owner)
                                 : is_bf16_row_gather
                                       ? release_capture_bf16_row_gather_leases(
                                             owner)
                                 : is_bf16_row_gather_argmax
                                       ? release_capture_bf16_row_gather_argmax_leases(
                                             owner)
                                 : is_bf16_row_gather_argmax_d2h
                                       ? release_capture_bf16_row_gather_argmax_d2h_leases(
                                             owner)
                                 : is_indexed_rope_bf16
                                       ? release_capture_indexed_rope_bf16_leases(
                                             owner)
                                 : is_canonical_gemm_bf16
                                       ? release_capture_canonical_gemm_bf16_leases(
                                             owner)
                                 : is_ragged_paged_kv_write_bf16
                                       ? release_capture_ragged_paged_kv_cache_write_bf16_leases(
                                             owner)
                                 : is_fill_or_generic
                                       ? release_capture_fill_lease(owner)
                                       : false);
      const bool cleanup_graph_released =
          release_graph_first
              ? prepared_graph_released
              : operation_released && destroy_prepared_graph_storage(owner);
      released = cleanup_graph_released && operation_released &&
                 release_capture_owner(owner);
      if (!released && status == RILEY_CUDA_STATUS_SUCCESS) {
        status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                kAbortOperation,
                                "failed to release recovered capture owner");
      }
    } else if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status;
      if (error != nullptr && error->struct_size >= sizeof(*error)) {
        *error = deferred_close_error;
      }
    }
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT, capture_id,
                         released, !released);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_fill_f32(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* buffer,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_impl(stream, buffer, element_count, mode, out_capture,
                            out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_h2d(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* destination,
    RileyCudaPinnedHostBuffer* source, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_h2d_impl(stream, destination, source, mode, out_capture,
                                out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_silu_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* output, uint64_t element_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_silu_bf16_impl(stream, input, output, element_count,
                                      mode, out_capture, out_graph_error,
                                      error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_gated_multiply_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* activated_gate,
    RileyCudaDeviceBuffer* up, RileyCudaDeviceBuffer* output,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_gated_multiply_bf16_impl(
      stream, activated_gate, up, output, element_count, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_residual_add_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* left,
    RileyCudaDeviceBuffer* right, RileyCudaDeviceBuffer* output,
    uint64_t element_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_residual_add_bf16_impl(
      stream, left, right, output, element_count, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_canonical_rms_norm_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* weight, RileyCudaDeviceBuffer* output,
    uint64_t row_count, uint64_t hidden_size, float epsilon,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_canonical_rms_norm_bf16_impl(
      stream, input, weight, output, row_count, hidden_size, epsilon, mode,
      out_capture, out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_bf16_argmax(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* logits,
    RileyCudaDeviceBuffer* results, uint64_t row_count,
    uint64_t vocabulary_size, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_bf16_argmax_impl(
      stream, logits, results, row_count, vocabulary_size, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_bf16_row_gather(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices, RileyCudaDeviceBuffer* output,
    uint64_t input_row_count, uint64_t output_row_count,
    uint64_t column_count, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_bf16_row_gather_impl(
      stream, input, row_indices, output, input_row_count, output_row_count,
      column_count, mode, out_capture, out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_bf16_row_gather_argmax(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits,
    RileyCudaDeviceBuffer* results, uint64_t input_row_count,
    uint64_t output_row_count, uint64_t vocabulary_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_bf16_row_gather_argmax_impl(
      stream, input, row_indices, gathered_logits, results, input_row_count,
      output_row_count, vocabulary_size, mode, out_capture, out_graph_error,
      error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_bf16_row_gather_argmax_d2h(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* row_indices,
    RileyCudaDeviceBuffer* gathered_logits, RileyCudaDeviceBuffer* results,
    RileyCudaPinnedHostBuffer* pinned_results, uint64_t input_row_count,
    uint64_t output_row_count, uint64_t vocabulary_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_bf16_row_gather_argmax_d2h_impl(
      stream, input, row_indices, gathered_logits, results, pinned_results,
      input_row_count, output_row_count, vocabulary_size, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_bf16_embedding_status_d2h(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* table,
    RileyCudaDeviceBuffer* token_ids, RileyCudaDeviceBuffer* output,
    RileyCudaDeviceBuffer* device_error_scratch,
    RileyCudaPinnedHostBuffer* pinned_report, uint64_t token_count,
    uint64_t vocabulary_size, uint64_t hidden_size,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_bf16_embedding_status_d2h_impl(
      stream, table, token_ids, output, device_error_scratch, pinned_report,
      token_count, vocabulary_size, hidden_size, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_begin_indexed_rope_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* input,
    RileyCudaDeviceBuffer* cos, RileyCudaDeviceBuffer* sin,
    RileyCudaDeviceBuffer* positions, RileyCudaDeviceBuffer* output,
    const uint32_t* positions_mirror, uint64_t positions_mirror_len,
    uint64_t active_row_count, uint64_t head_count, uint64_t head_size,
    uint64_t rotary_dimension, uint64_t table_position_count,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_indexed_rope_bf16_impl(
      stream, input, cos, sin, positions, output, positions_mirror,
      positions_mirror_len, active_row_count, head_count, head_size,
      rotary_dimension, table_position_count, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_ragged_paged_kv_cache_write_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* key_source,
    RileyCudaDeviceBuffer* value_source, RileyCudaDeviceBuffer* key_pool,
    RileyCudaDeviceBuffer* value_pool,
    RileyCudaDeviceBuffer* sequence_block_offsets,
    RileyCudaDeviceBuffer* block_ids, RileyCudaDeviceBuffer* valid_tokens,
    RileyCudaDeviceBuffer* row_sequence_slots,
    RileyCudaDeviceBuffer* row_positions, uint64_t sequence_count,
    uint64_t block_count, uint64_t active_row_count,
    uint64_t physical_block_count, uint64_t key_value_head_count,
    uint64_t head_size, RileyCudaGraphCaptureMode mode,
    RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_ragged_paged_kv_cache_write_bf16_impl(
      stream, key_source, value_source, key_pool, value_pool,
      sequence_block_offsets, block_ids, valid_tokens, row_sequence_slots,
      row_positions, sequence_count, block_count, active_row_count,
      physical_block_count, key_value_head_count, head_size, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_grouped_ragged_paged_attention_bf16(
    RileyCudaStream* stream, RileyCudaDeviceBuffer* query,
    RileyCudaDeviceBuffer* key_pool, RileyCudaDeviceBuffer* value_pool,
    RileyCudaDeviceBuffer* output,
    RileyCudaDeviceBuffer* sequence_block_offsets,
    RileyCudaDeviceBuffer* block_ids, RileyCudaDeviceBuffer* valid_tokens,
    RileyCudaDeviceBuffer* row_sequence_slots,
    RileyCudaDeviceBuffer* row_positions, uint64_t sequence_count,
    uint64_t block_count, uint64_t active_row_count,
    uint64_t physical_block_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t output_row_count, float scale,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_grouped_ragged_paged_attention_bf16_impl(
      stream, query, key_pool, value_pool, output, sequence_block_offsets,
      block_ids, valid_tokens, row_sequence_slots, row_positions,
      sequence_count, block_count, active_row_count, physical_block_count,
      query_head_count, key_value_head_count, output_row_count, scale, mode,
      out_capture, out_graph_error, error);
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_begin_canonical_gemm_bf16(
    RileyCudaStream* stream, RileyCudaGemmPlan* plan,
    RileyCudaDeviceBuffer* input, RileyCudaDeviceBuffer* weight,
    RileyCudaDeviceBuffer* output, RileyCudaDeviceBuffer* workspace,
    RileyCudaGraphCaptureMode mode, RileyCudaGraphCapture** out_capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  return capture_begin_canonical_gemm_bf16_impl(
      stream, plan, input, weight, output, workspace, mode, out_capture,
      out_graph_error, error);
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_fill_f32(
    RileyCudaGraphCapture* capture, float value,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueFillOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation, "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kFillF32 ||
      !owner->fill_lease_held || owner->capture_terminated ||
      owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "capture owner is not a live fixed-fill capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueFillOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->fill_element_count == 0 ||
      owner->fill_element_count > owner->fill_buffer->byte_len / sizeof(float) ||
      owner->fill_buffer->device_data == nullptr ||
      owner->fill_enqueue_count == std::numeric_limits<uint32_t>::max()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed-fill capture owner has invalid immutable geometry");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed-fill capture resource lease is unavailable");
  }
  const uint64_t grid_x =
      ((owner->fill_element_count - 1) / kGraphFillThreads) + 1;
  if (grid_x > kMaximumGraphFillGridX) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueFillOperation,
                            "fixed f32 fill grid exceeds CUDA's x-dimension limit");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueFillOperation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_fill_f32<<<static_cast<unsigned int>(grid_x), kGraphFillThreads, 0,
                     owner->stream->stream>>>(
        static_cast<float*>(owner->fill_buffer->device_data),
        owner->fill_element_count, value);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueFillOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      ++owner->fill_enqueue_count;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueFillOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_h2d(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation, "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->h2d_source == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kH2D ||
      !owner->fill_lease_held || !owner->h2d_source_lease_held ||
      owner->capture_terminated || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "capture owner is not a live graph H2D capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueH2DOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->h2d_byte_len == 0 ||
      owner->h2d_byte_len != owner->fill_buffer->byte_len ||
      owner->h2d_byte_len != owner->h2d_source->byte_len ||
      owner->fill_buffer->device_data == nullptr ||
      owner->h2d_source->host_data == nullptr || owner->h2d_enqueue_count != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "graph H2D capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueH2DOperation,
                            "graph H2D capture resource lease is unavailable");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueH2DOperation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaMemcpyAsync(owner->fill_buffer->device_data, owner->h2d_source->host_data,
                        owner->h2d_byte_len, cudaMemcpyHostToDevice,
                        owner->stream->stream),
        error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueH2DOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->h2d_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueH2DOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_silu_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->silu_input == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kSiluBf16 ||
      !owner->fill_lease_held || !owner->silu_input_lease_held ||
      owner->capture_terminated || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "capture owner is not a live graph BF16 SiLU capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started || owner->silu_input == owner->fill_buffer ||
      owner->silu_element_count == 0 ||
      owner->silu_element_count >
          owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
      owner->silu_element_count >
          owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
      owner->silu_input->device_data == nullptr ||
      owner->fill_buffer->device_data == nullptr ||
      owner->silu_enqueue_count != 0 || owner->h2d_source != nullptr ||
      owner->h2d_byte_len != 0 || owner->h2d_source_lease_held ||
      owner->fill_element_count != 0 || owner->fill_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueSiluBf16Operation,
        "graph BF16 SiLU capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueSiluBf16Operation,
                            "graph BF16 SiLU capture resource lease is unavailable");
  }
  const uint64_t needed_blocks =
      ((owner->silu_element_count - 1) / kGraphSiluThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      needed_blocks < kMaximumGraphSiluBlocks ? needed_blocks
                                               : kMaximumGraphSiluBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kEnqueueSiluBf16Operation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_silu_bf16<<<grid_x, kGraphSiluThreads, 0, owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(owner->silu_input->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->silu_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueSiluBf16Operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->silu_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueSiluBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_gated_multiply_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueGatedMultiplyBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueGatedMultiplyBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->gated_multiply_activated_gate == nullptr ||
      owner->gated_multiply_up == nullptr ||
      owner->operation !=
          RileyCudaGraphCaptureOperation::kGatedMultiplyBf16 ||
      !owner->fill_lease_held ||
      !owner->gated_multiply_activated_gate_lease_held ||
      !owner->gated_multiply_up_lease_held || owner->capture_terminated ||
      owner->unreleased_graph != nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueGatedMultiplyBf16Operation,
        "capture owner is not a live graph BF16 gated-multiply capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueGatedMultiplyBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started ||
      owner->gated_multiply_activated_gate == owner->gated_multiply_up ||
      owner->gated_multiply_activated_gate == owner->fill_buffer ||
      owner->gated_multiply_up == owner->fill_buffer ||
      owner->gated_multiply_element_count == 0 ||
      owner->gated_multiply_element_count >
          owner->gated_multiply_activated_gate->byte_len /
              sizeof(__nv_bfloat16) ||
      owner->gated_multiply_element_count >
          owner->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
      owner->gated_multiply_element_count >
          owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
      owner->gated_multiply_activated_gate->device_data == nullptr ||
      owner->gated_multiply_up->device_data == nullptr ||
      owner->fill_buffer->device_data == nullptr ||
      owner->gated_multiply_enqueue_count != 0 ||
      owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
      owner->h2d_enqueue_count != 0 || owner->h2d_source_lease_held ||
      owner->silu_input != nullptr || owner->silu_element_count != 0 ||
      owner->silu_enqueue_count != 0 || owner->silu_input_lease_held ||
      owner->fill_element_count != 0 || owner->fill_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueGatedMultiplyBf16Operation,
        "graph BF16 gated-multiply capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->gated_multiply_activated_gate->active_uses.load(
          std::memory_order_acquire) != 1 ||
      owner->gated_multiply_up->active_uses.load(std::memory_order_acquire) !=
          1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueGatedMultiplyBf16Operation,
        "graph BF16 gated-multiply capture resource lease is unavailable");
  }
  const uint64_t needed_blocks =
      ((owner->gated_multiply_element_count - 1) / kGraphSiluThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      needed_blocks < kMaximumGraphSiluBlocks ? needed_blocks
                                               : kMaximumGraphSiluBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueGatedMultiplyBf16Operation,
      owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_gated_multiply_bf16<<<grid_x, kGraphSiluThreads, 0,
                                 owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->gated_multiply_activated_gate->device_data),
        static_cast<const __nv_bfloat16*>(owner->gated_multiply_up->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->gated_multiply_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueGatedMultiplyBf16Operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->gated_multiply_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueGatedMultiplyBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_residual_add_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueResidualAddBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueResidualAddBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      owner->residual_add_left == nullptr ||
      owner->residual_add_right == nullptr ||
      owner->operation != RileyCudaGraphCaptureOperation::kResidualAddBf16 ||
      !owner->fill_lease_held || !owner->residual_add_left_lease_held ||
      !owner->residual_add_right_lease_held || owner->capture_terminated ||
      owner->unreleased_graph != nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueResidualAddBf16Operation,
        "capture owner is not a live graph BF16 residual-add capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueResidualAddBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  if (!owner->capture_started ||
      owner->residual_add_left == owner->residual_add_right ||
      owner->residual_add_left == owner->fill_buffer ||
      owner->residual_add_right == owner->fill_buffer ||
      owner->residual_add_element_count == 0 ||
      owner->residual_add_element_count >
          owner->residual_add_left->byte_len / sizeof(__nv_bfloat16) ||
      owner->residual_add_element_count >
          owner->residual_add_right->byte_len / sizeof(__nv_bfloat16) ||
      owner->residual_add_element_count >
          owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
      owner->residual_add_left->device_data == nullptr ||
      owner->residual_add_right->device_data == nullptr ||
      owner->fill_buffer->device_data == nullptr ||
      owner->residual_add_enqueue_count != 0 ||
      owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
      owner->h2d_enqueue_count != 0 || owner->h2d_source_lease_held ||
      owner->silu_input != nullptr || owner->silu_element_count != 0 ||
      owner->silu_enqueue_count != 0 || owner->silu_input_lease_held ||
      owner->gated_multiply_activated_gate != nullptr ||
      owner->gated_multiply_up != nullptr ||
      owner->gated_multiply_element_count != 0 ||
      owner->gated_multiply_enqueue_count != 0 ||
      owner->gated_multiply_activated_gate_lease_held ||
      owner->gated_multiply_up_lease_held ||
      owner->fill_element_count != 0 || owner->fill_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueResidualAddBf16Operation,
        "graph BF16 residual-add capture has invalid immutable geometry or already enqueued its sole node");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->residual_add_left->active_uses.load(std::memory_order_acquire) !=
          1 ||
      owner->residual_add_right->active_uses.load(std::memory_order_acquire) !=
          1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueResidualAddBf16Operation,
        "graph BF16 residual-add capture resource lease is unavailable");
  }
  const uint64_t needed_blocks =
      ((owner->residual_add_element_count - 1) / kGraphSiluThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      needed_blocks < kMaximumGraphSiluBlocks ? needed_blocks
                                               : kMaximumGraphSiluBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueResidualAddBf16Operation,
      owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_residual_add_bf16<<<grid_x, kGraphSiluThreads, 0,
                              owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(owner->residual_add_left->device_data),
        static_cast<const __nv_bfloat16*>(
            owner->residual_add_right->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->residual_add_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueResidualAddBf16Operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->residual_add_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueResidualAddBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_canonical_rms_norm_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueCanonicalRmsNormBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueCanonicalRmsNormBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !canonical_rms_norm_capture_state_is_valid(owner) ||
      owner->canonical_rms_norm_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueCanonicalRmsNormBf16Operation,
        "capture owner is not a live unqueued canonical RMSNorm graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueCanonicalRmsNormBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  const uint32_t grid_x = static_cast<uint32_t>(
      owner->canonical_rms_norm_row_count < kMaximumGraphCanonicalRmsNormBlocks
          ? owner->canonical_rms_norm_row_count
          : kMaximumGraphCanonicalRmsNormBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueCanonicalRmsNormBf16Operation,
      owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_canonical_rms_norm_bf16<<<
        grid_x, kGraphCanonicalRmsNormThreads,
        kGraphCanonicalRmsNormThreads * sizeof(float), owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->canonical_rms_norm_input->device_data),
        static_cast<const __nv_bfloat16*>(
            owner->canonical_rms_norm_weight->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->canonical_rms_norm_row_count,
        owner->canonical_rms_norm_hidden_size,
        owner->canonical_rms_norm_epsilon);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueCanonicalRmsNormBf16Operation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->canonical_rms_norm_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueCanonicalRmsNormBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_bf16_argmax(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16ArgmaxOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueBf16ArgmaxOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !bf16_argmax_capture_state_is_valid(owner) ||
      owner->bf16_argmax_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16ArgmaxOperation,
        "capture owner is not a live unqueued deterministic BF16 argmax graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16ArgmaxOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  const uint32_t grid_x = static_cast<uint32_t>(
      owner->bf16_argmax_row_count < kMaximumGraphBf16ArgmaxBlocks
          ? owner->bf16_argmax_row_count
          : kMaximumGraphBf16ArgmaxBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueBf16ArgmaxOperation,
      owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_bf16_argmax_bf16<<<grid_x, kGraphBf16ArgmaxThreads, 0,
                              owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->bf16_argmax_logits->device_data),
        static_cast<RileyCudaBf16ArgmaxResult*>(
            owner->fill_buffer->device_data),
        owner->bf16_argmax_row_count, owner->bf16_argmax_vocabulary_size);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueBf16ArgmaxOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->bf16_argmax_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueBf16ArgmaxOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_bf16_row_gather(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16RowGatherOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueBf16RowGatherOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !bf16_row_gather_capture_state_is_valid(owner) ||
      owner->bf16_row_gather_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16RowGatherOperation,
        "capture owner is not a live unqueued BF16 row-gather graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueBf16RowGatherOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  // State validation above proves this multiplication cannot overflow and all
  // dimensions are nonzero. Enqueue remains allocation-free and does no
  // stream synchronization or graph-node mutation.
  const uint64_t output_element_count =
      owner->bf16_row_gather_output_row_count *
      owner->bf16_row_gather_column_count;
  const uint64_t requested_blocks =
      ((output_element_count - 1) / kGraphBf16RowGatherThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      requested_blocks < kMaximumGraphBf16RowGatherBlocks
          ? requested_blocks
          : kMaximumGraphBf16RowGatherBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueBf16RowGatherOperation,
      owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_bf16_row_gather_bf16<<<grid_x, kGraphBf16RowGatherThreads, 0,
                                  owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->bf16_row_gather_input->device_data),
        static_cast<const uint32_t*>(
            owner->bf16_row_gather_indices->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->bf16_row_gather_input_row_count,
        owner->bf16_row_gather_column_count, output_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueBf16RowGatherOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      owner->bf16_row_gather_enqueue_count = 1;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueBf16RowGatherOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_bf16_row_gather_argmax(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueBf16RowGatherArgmaxOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !bf16_row_gather_argmax_capture_state_is_valid(owner) ||
      owner->bf16_row_gather_argmax_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxOperation,
        "capture owner is not a live unqueued BF16 row-gather argmax graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  // State validation proves both products cannot overflow, every dimension is
  // nonzero, and vocabulary_size fits the argmax U32 token-id contract.
  const uint64_t gathered_element_count =
      owner->bf16_row_gather_argmax_output_row_count *
      owner->bf16_row_gather_argmax_vocabulary_size;
  const uint64_t requested_gather_blocks =
      ((gathered_element_count - 1) / kGraphBf16RowGatherThreads) + 1;
  const uint32_t gather_grid_x = static_cast<uint32_t>(
      requested_gather_blocks < kMaximumGraphBf16RowGatherBlocks
          ? requested_gather_blocks
          : kMaximumGraphBf16RowGatherBlocks);
  const uint32_t argmax_grid_x = static_cast<uint32_t>(
      owner->bf16_row_gather_argmax_output_row_count <
              kMaximumGraphBf16ArgmaxBlocks
          ? owner->bf16_row_gather_argmax_output_row_count
          : kMaximumGraphBf16ArgmaxBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueBf16RowGatherArgmaxOperation, owner);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    graph_bf16_row_gather_bf16<<<gather_grid_x, kGraphBf16RowGatherThreads,
                                  0, owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->bf16_row_gather_argmax_input->device_data),
        static_cast<const uint32_t*>(
            owner->bf16_row_gather_argmax_indices->device_data),
        static_cast<__nv_bfloat16*>(
            owner->bf16_row_gather_argmax_gathered_logits->device_data),
        owner->bf16_row_gather_argmax_input_row_count,
        owner->bf16_row_gather_argmax_vocabulary_size,
        gathered_element_count);
    status = runtime_error(cudaGetLastError(), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueBf16RowGatherArgmaxOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_bf16_argmax_bf16<<<argmax_grid_x, kGraphBf16ArgmaxThreads, 0,
                                owner->stream->stream>>>(
          static_cast<const __nv_bfloat16*>(
              owner->bf16_row_gather_argmax_gathered_logits->device_data),
          static_cast<RileyCudaBf16ArgmaxResult*>(
              owner->fill_buffer->device_data),
          owner->bf16_row_gather_argmax_output_row_count,
          owner->bf16_row_gather_argmax_vocabulary_size);
      status = runtime_error(cudaGetLastError(), error,
                             RILEY_CUDA_ERROR_STAGE_LAUNCH,
                             kEnqueueBf16RowGatherArgmaxOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      // The graph is complete only after both dependent nodes were accepted.
      owner->bf16_row_gather_argmax_enqueue_count = 1;
    } else {
      // A recorded prefix plus a failed later launch can leave CUDA capture
      // invalidated. Keep every lease and reject enqueue/end/instantiate until
      // the one abort transition has observed CUDA's terminal capture state.
      owner->bf16_row_gather_argmax_enqueue_count =
          kBf16RowGatherArgmaxEnqueueTerminal;
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueBf16RowGatherArgmaxOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_bf16_row_gather_argmax_d2h(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxD2HOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueBf16RowGatherArgmaxD2HOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !bf16_row_gather_argmax_d2h_capture_state_is_valid(owner) ||
      owner->bf16_row_gather_argmax_d2h_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxD2HOperation,
        "capture owner is not a live unqueued BF16 row-gather argmax D2H graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16RowGatherArgmaxD2HOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  const uint64_t gathered_element_count =
      owner->bf16_row_gather_argmax_d2h_output_row_count *
      owner->bf16_row_gather_argmax_d2h_vocabulary_size;
  const uint64_t requested_gather_blocks =
      ((gathered_element_count - 1) / kGraphBf16RowGatherThreads) + 1;
  const uint32_t gather_grid_x = static_cast<uint32_t>(
      requested_gather_blocks < kMaximumGraphBf16RowGatherBlocks
          ? requested_gather_blocks
          : kMaximumGraphBf16RowGatherBlocks);
  const uint32_t argmax_grid_x = static_cast<uint32_t>(
      owner->bf16_row_gather_argmax_d2h_output_row_count <
              kMaximumGraphBf16ArgmaxBlocks
          ? owner->bf16_row_gather_argmax_d2h_output_row_count
          : kMaximumGraphBf16ArgmaxBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueBf16RowGatherArgmaxD2HOperation, owner);
  bool node_submission_started = false;
  bool all_nodes_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    graph_bf16_row_gather_bf16<<<gather_grid_x, kGraphBf16RowGatherThreads,
                                  0, owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->bf16_row_gather_argmax_d2h_input->device_data),
        static_cast<const uint32_t*>(
            owner->bf16_row_gather_argmax_d2h_indices->device_data),
        static_cast<__nv_bfloat16*>(
            owner->bf16_row_gather_argmax_d2h_gathered_logits->device_data),
        owner->bf16_row_gather_argmax_d2h_input_row_count,
        owner->bf16_row_gather_argmax_d2h_vocabulary_size,
        gathered_element_count);
    status = runtime_error(cudaGetLastError(), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueBf16RowGatherArgmaxD2HOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_bf16_argmax_bf16<<<argmax_grid_x, kGraphBf16ArgmaxThreads, 0,
                                owner->stream->stream>>>(
          static_cast<const __nv_bfloat16*>(
              owner->bf16_row_gather_argmax_d2h_gathered_logits->device_data),
          static_cast<RileyCudaBf16ArgmaxResult*>(
              owner->fill_buffer->device_data),
          owner->bf16_row_gather_argmax_d2h_output_row_count,
          owner->bf16_row_gather_argmax_d2h_vocabulary_size);
      status = runtime_error(cudaGetLastError(), error,
                             RILEY_CUDA_ERROR_STAGE_LAUNCH,
                             kEnqueueBf16RowGatherArgmaxD2HOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(
          cudaMemcpyAsync(
              owner->bf16_row_gather_argmax_d2h_pinned_results->host_data,
              owner->fill_buffer->device_data,
              owner->bf16_row_gather_argmax_d2h_result_byte_len,
              cudaMemcpyDeviceToHost, owner->stream->stream),
          error, RILEY_CUDA_ERROR_STAGE_COPY,
          kEnqueueBf16RowGatherArgmaxD2HOperation);
    }
    all_nodes_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueBf16RowGatherArgmaxD2HOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!all_nodes_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    // Any recorded prefix, including a successful triple followed by failed
    // context restoration, is abort-only. End/instantiate must not observe it.
    owner->bf16_row_gather_argmax_d2h_enqueue_count =
        kBf16RowGatherArgmaxD2HEnqueueTerminal;
  } else if (all_nodes_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->bf16_row_gather_argmax_d2h_enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_bf16_embedding_status_d2h(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16EmbeddingStatusD2HOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueBf16EmbeddingStatusD2HOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->operation !=
          RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H ||
      owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !bf16_embedding_status_d2h_capture_state_is_valid(owner) ||
      !bf16_embedding_status_d2h_graph_state_is_valid(owner->prepared_graph) ||
      owner->bf16_row_gather_argmax_d2h_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16EmbeddingStatusD2HOperation,
        "capture owner is not a live unqueued BF16 embedding status D2H graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueBf16EmbeddingStatusD2HOperation,
        "thread-local capture must enqueue on its begin thread");
  }
  const uint64_t token_count =
      owner->bf16_row_gather_argmax_d2h_output_row_count;
  const uint64_t vocabulary_size =
      owner->bf16_row_gather_argmax_d2h_input_row_count;
  const uint64_t hidden_size =
      owner->bf16_row_gather_argmax_d2h_vocabulary_size;
  const uint64_t output_element_count = token_count * hidden_size;
  const uint64_t requested_token_blocks =
      ((token_count - 1) / kGraphBf16EmbeddingThreads) + 1;
  const uint64_t requested_output_blocks =
      ((output_element_count - 1) / kGraphBf16EmbeddingThreads) + 1;
  const uint32_t token_grid_x = static_cast<uint32_t>(
      requested_token_blocks < kMaximumGraphBf16EmbeddingBlocks
          ? requested_token_blocks
          : kMaximumGraphBf16EmbeddingBlocks);
  const uint32_t output_grid_x = static_cast<uint32_t>(
      requested_output_blocks < kMaximumGraphBf16EmbeddingBlocks
          ? requested_output_blocks
          : kMaximumGraphBf16EmbeddingBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueBf16EmbeddingStatusD2HOperation, owner);
  bool node_submission_started = false;
  bool all_nodes_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    auto* const report = static_cast<RileyCudaEmbeddingErrorReport*>(
        owner->bf16_row_gather_argmax_d2h_gathered_logits->device_data);
    const auto* const ids = static_cast<const uint32_t*>(
        owner->bf16_row_gather_argmax_d2h_indices->device_data);
    graph_reset_embedding_error<<<1, 1, 0, owner->stream->stream>>>(report);
    status = runtime_error(cudaGetLastError(), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueBf16EmbeddingStatusD2HOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_validate_embedding_tokens<<<token_grid_x, kGraphBf16EmbeddingThreads,
                                        0, owner->stream->stream>>>(
          ids, token_count, vocabulary_size, report);
      status = runtime_error(cudaGetLastError(), error,
                             RILEY_CUDA_ERROR_STAGE_LAUNCH,
                             kEnqueueBf16EmbeddingStatusD2HOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_bf16_embedding<<<output_grid_x, kGraphBf16EmbeddingThreads, 0,
                             owner->stream->stream>>>(
          static_cast<const __nv_bfloat16*>(
              owner->bf16_row_gather_argmax_d2h_input->device_data),
          ids, static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
          report, hidden_size, output_element_count);
      status = runtime_error(cudaGetLastError(), error,
                             RILEY_CUDA_ERROR_STAGE_LAUNCH,
                             kEnqueueBf16EmbeddingStatusD2HOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_finalize_embedding_error<<<1, 1, 0, owner->stream->stream>>>(
          ids, token_count, report);
      status = runtime_error(cudaGetLastError(), error,
                             RILEY_CUDA_ERROR_STAGE_LAUNCH,
                             kEnqueueBf16EmbeddingStatusD2HOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = runtime_error(
          cudaMemcpyAsync(
              owner->bf16_row_gather_argmax_d2h_pinned_results->host_data,
              report, sizeof(RileyCudaEmbeddingErrorReport),
              cudaMemcpyDeviceToHost, owner->stream->stream),
          error, RILEY_CUDA_ERROR_STAGE_COPY,
          kEnqueueBf16EmbeddingStatusD2HOperation);
    }
    all_nodes_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueBf16EmbeddingStatusD2HOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!all_nodes_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    owner->bf16_row_gather_argmax_d2h_enqueue_count =
        kBf16EmbeddingStatusD2HEnqueueTerminal;
  } else if (all_nodes_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->bf16_row_gather_argmax_d2h_enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_enqueue_indexed_rope_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueIndexedRopeBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueIndexedRopeBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !indexed_rope_bf16_capture_state_is_valid(owner) ||
      owner->indexed_rope_bf16_enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueIndexedRopeBf16Operation,
        "capture owner is not a live unqueued BF16 indexed-RoPE graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueIndexedRopeBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  uint64_t tensor_element_count = 0;
  uint64_t table_element_count = 0;
  uint64_t work_item_count = 0;
  if (!indexed_rope_bf16_shape_is_valid(
          owner->indexed_rope_bf16_active_row_count,
          owner->indexed_rope_bf16_head_count,
          owner->indexed_rope_bf16_head_size,
          owner->indexed_rope_bf16_rotary_dimension,
          owner->indexed_rope_bf16_table_position_count,
          &tensor_element_count, &table_element_count, &work_item_count)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEnqueueIndexedRopeBf16Operation,
        "BF16 indexed-RoPE capture owner has invalid immutable geometry");
  }
  // These values are independently checked by the state validator and are
  // intentionally recomputed here to reject a corrupted immutable shape
  // before deriving launch geometry.
  (void)tensor_element_count;
  (void)table_element_count;
  const uint64_t requested_blocks =
      ((work_item_count - 1) / kGraphBf16RowGatherThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      requested_blocks < kMaximumGraphBf16RowGatherBlocks
          ? requested_blocks
          : kMaximumGraphBf16RowGatherBlocks);

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kEnqueueIndexedRopeBf16Operation,
      owner);
  bool node_submission_started = false;
  bool node_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    graph_indexed_rope_bf16<<<grid_x, kGraphBf16RowGatherThreads, 0,
                              owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(
            owner->indexed_rope_bf16_input->device_data),
        static_cast<const float*>(owner->indexed_rope_bf16_cos->device_data),
        static_cast<const float*>(owner->indexed_rope_bf16_sin->device_data),
        static_cast<const uint32_t*>(
            owner->indexed_rope_bf16_positions->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        owner->indexed_rope_bf16_table_position_count,
        owner->indexed_rope_bf16_head_count,
        owner->indexed_rope_bf16_head_size,
        owner->indexed_rope_bf16_rotary_dimension, work_item_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueIndexedRopeBf16Operation);
    node_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueIndexedRopeBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!node_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    owner->indexed_rope_bf16_enqueue_count =
        kIndexedRopeBf16EnqueueTerminal;
  } else if (node_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->indexed_rope_bf16_enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_ragged_paged_kv_cache_write_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueRaggedPagedKvCacheWriteBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueRaggedPagedKvCacheWriteBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !ragged_paged_kv_cache_write_bf16_capture_state_is_valid(owner) ||
      !ragged_paged_kv_cache_write_bf16_prepared_graph_state_is_valid(
          owner->prepared_graph) ||
      !ragged_paged_kv_cache_write_bf16_state_geometry_matches(
          owner->ragged_paged_kv_write_bf16,
          owner->prepared_graph->ragged_paged_kv_write_bf16) ||
      owner->ragged_paged_kv_write_bf16.enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueRaggedPagedKvCacheWriteBf16Operation,
        "capture owner is not a live unqueued BF16 ragged paged-KV graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueRaggedPagedKvCacheWriteBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  const RileyCudaRaggedPagedKvCacheWriteBf16State& state =
      owner->ragged_paged_kv_write_bf16;
  uint64_t source_element_count = 0;
  uint64_t pool_element_count = 0;
  if (!ragged_paged_kv_cache_write_bf16_shape_is_valid(
          state.sequence_count, state.block_count, state.active_row_count,
          state.physical_block_count, state.key_value_head_count,
          state.head_size, &source_element_count, &pool_element_count)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueRaggedPagedKvCacheWriteBf16Operation,
        "BF16 ragged paged-KV capture owner has invalid immutable geometry");
  }
  (void)pool_element_count;
  const uint64_t requested_blocks =
      ((source_element_count - 1) / kGraphBf16RowGatherThreads) + 1;
  const uint32_t grid_x = static_cast<uint32_t>(
      requested_blocks < kMaximumGraphBf16RowGatherBlocks
          ? requested_blocks
          : kMaximumGraphBf16RowGatherBlocks);
  const GraphRaggedPagedKvDeviceBatch batch{
      static_cast<const uint32_t*>(state.sequence_block_offsets->device_data),
      static_cast<const uint32_t*>(state.block_ids->device_data),
      static_cast<const uint16_t*>(state.valid_tokens->device_data),
      static_cast<const uint32_t*>(state.row_sequence_slots->device_data),
      static_cast<const uint32_t*>(state.row_positions->device_data),
      state.sequence_count,
      state.block_count,
      state.active_row_count,
      state.physical_block_count,
  };

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueRaggedPagedKvCacheWriteBf16Operation, owner);
  bool node_submission_started = false;
  bool node_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    graph_ragged_paged_kv_cache_write_bf16<<<
        grid_x, kGraphBf16RowGatherThreads, 0, owner->stream->stream>>>(
        static_cast<const __nv_bfloat16*>(state.key_source->device_data),
        static_cast<const __nv_bfloat16*>(state.value_source->device_data),
        static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data),
        static_cast<__nv_bfloat16*>(state.value_pool->device_data), batch,
        state.key_value_head_count, state.head_size, source_element_count);
    status = runtime_error(cudaGetLastError(), error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueRaggedPagedKvCacheWriteBf16Operation);
    node_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueRaggedPagedKvCacheWriteBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!node_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    owner->ragged_paged_kv_write_bf16.enqueue_count =
        kRaggedPagedKvCacheWriteBf16EnqueueTerminal;
  } else if (node_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->ragged_paged_kv_write_bf16.enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_canonical_gemm_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueCanonicalGemmBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kEnqueueCanonicalGemmBf16Operation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !canonical_gemm_bf16_capture_state_is_valid(owner) ||
      !canonical_gemm_bf16_prepared_graph_state_is_valid(
          owner->prepared_graph) ||
      !canonical_gemm_bf16_state_matches(owner->canonical_gemm_bf16,
                                          owner->prepared_graph
                                              ->canonical_gemm_bf16) ||
      owner->canonical_gemm_bf16.enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueCanonicalGemmBf16Operation,
        "capture owner is not a live unqueued canonical cuBLASLt GEMM graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueCanonicalGemmBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueCanonicalGemmBf16Operation, owner);
  bool node_submission_started = false;
  bool node_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    status = riley_cuda_internal::enqueue_canonical_gemm_bf16_graph_matmul(
        owner->owner, owner->stream, owner->fill_buffer,
        owner->canonical_gemm_bf16, error, kEnqueueCanonicalGemmBf16Operation);
    node_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueCanonicalGemmBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!node_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    owner->canonical_gemm_bf16.enqueue_count =
        kCanonicalGemmBf16EnqueueTerminal;
  } else if (node_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->canonical_gemm_bf16.enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_capture_enqueue_grouped_ragged_paged_attention_bf16(
    RileyCudaGraphCapture* capture,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueGroupedRaggedPagedAttentionBf16Operation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE);
  if (capture == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueGroupedRaggedPagedAttentionBf16Operation,
        "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = capture;
  const uint64_t capture_id = owner->capture_id;
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, false);
  if (owner->operation != RileyCudaGraphCaptureOperation::
                              kGroupedRaggedPagedAttentionBf16 ||
      owner->prepared_graph == nullptr || !owner->capture_started ||
      owner->capture_terminated || owner->unreleased_graph != nullptr ||
      !ragged_paged_kv_cache_write_bf16_capture_state_is_valid(owner) ||
      !ragged_paged_kv_cache_write_bf16_prepared_graph_state_is_valid(
          owner->prepared_graph) ||
      !ragged_paged_kv_cache_write_bf16_state_geometry_matches(
          owner->ragged_paged_kv_write_bf16,
          owner->prepared_graph->ragged_paged_kv_write_bf16) ||
      owner->ragged_paged_kv_write_bf16.enqueue_count != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueGroupedRaggedPagedAttentionBf16Operation,
        "capture owner is not a live unqueued grouped ragged-attention graph capture");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueGroupedRaggedPagedAttentionBf16Operation,
        "thread-local capture must enqueue on its begin thread");
  }
  const RileyCudaRaggedPagedKvCacheWriteBf16State& state =
      owner->ragged_paged_kv_write_bf16;
  uint64_t query_element_count = 0;
  uint64_t pool_element_count = 0;
  uint64_t output_element_count = 0;
  if (!state.grouped_attention ||
      !grouped_ragged_paged_attention_bf16_shape_is_valid(
          state.sequence_count, state.block_count, state.active_row_count,
          state.physical_block_count, state.query_head_count,
          state.key_value_head_count, state.output_row_count,
          state.attention_scale, &query_element_count, &pool_element_count,
          &output_element_count)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kEnqueueGroupedRaggedPagedAttentionBf16Operation,
        "grouped ragged-attention capture owner has invalid immutable geometry");
  }
  (void)query_element_count;
  (void)pool_element_count;
  (void)output_element_count;
  const GraphRaggedPagedKvDeviceBatch batch{
      static_cast<const uint32_t*>(state.sequence_block_offsets->device_data),
      static_cast<const uint32_t*>(state.block_ids->device_data),
      static_cast<const uint16_t*>(state.valid_tokens->device_data),
      static_cast<const uint32_t*>(state.row_sequence_slots->device_data),
      static_cast<const uint32_t*>(state.row_positions->device_data),
      state.sequence_count,
      state.block_count,
      state.active_row_count,
      state.physical_block_count,
  };

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(
      error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
      kEnqueueGroupedRaggedPagedAttentionBf16Operation, owner);
  bool node_submission_started = false;
  bool node_accepted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    node_submission_started = true;
    const uint64_t query_head_group_size =
        state.query_head_count / state.key_value_head_count;
    if (query_head_group_size > 1 &&
        query_head_group_size <= kGraphRaggedAttentionMaximumWarpsPerBlock) {
      const dim3 grid(static_cast<uint32_t>(state.output_row_count),
                      static_cast<uint32_t>(state.key_value_head_count), 1);
      graph_grouped_ragged_paged_attention_gqa_shared_kv_bf16<<<
          grid,
          static_cast<uint32_t>(query_head_group_size *
                                kGraphRaggedAttentionWarpSize),
          0, owner->stream->stream>>>(
          static_cast<const __nv_bfloat16*>(state.key_source->device_data),
          static_cast<const __nv_bfloat16*>(state.value_source->device_data),
          static_cast<const __nv_bfloat16*>(state.value_pool->device_data),
          static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data), batch,
          state.query_head_count, state.key_value_head_count,
          state.attention_scale);
    } else {
      const uint32_t query_head_warps = static_cast<uint32_t>(
          state.query_head_count < kGraphRaggedAttentionMaximumWarpsPerBlock
              ? state.query_head_count
              : kGraphRaggedAttentionMaximumWarpsPerBlock);
      const uint32_t query_head_tiles = static_cast<uint32_t>(
          (state.query_head_count + query_head_warps - 1) /
          query_head_warps);
      const dim3 grid(static_cast<uint32_t>(state.output_row_count),
                      query_head_tiles, 1);
      graph_grouped_ragged_paged_attention_bf16<<<
          grid, query_head_warps * kGraphRaggedAttentionWarpSize, 0,
          owner->stream->stream>>>(
          static_cast<const __nv_bfloat16*>(state.key_source->device_data),
          static_cast<const __nv_bfloat16*>(state.value_source->device_data),
          static_cast<const __nv_bfloat16*>(state.value_pool->device_data),
          static_cast<__nv_bfloat16*>(owner->fill_buffer->device_data), batch,
          state.query_head_count, state.key_value_head_count,
          state.attention_scale);
    }
    status = runtime_error(cudaGetLastError(), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH,
                           kEnqueueGroupedRaggedPagedAttentionBf16Operation);
    node_accepted = status == RILEY_CUDA_STATUS_SUCCESS;
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kEnqueueGroupedRaggedPagedAttentionBf16Operation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (node_submission_started &&
      (!node_accepted || status != RILEY_CUDA_STATUS_SUCCESS ||
       !restoration_known)) {
    owner->ragged_paged_kv_write_bf16.enqueue_count =
        kGroupedRaggedPagedAttentionBf16EnqueueTerminal;
  } else if (node_accepted && status == RILEY_CUDA_STATUS_SUCCESS &&
             restoration_known) {
    owner->ragged_paged_kv_write_bf16.enqueue_count = 1;
  }
  record_capture_outcome(out_graph_error,
                         RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE, capture_id,
                         false, status != RILEY_CUDA_STATUS_SUCCESS ||
                                    !restoration_known);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_capture_end(
    RileyCudaGraphCapture** capture, RileyCudaGraph** out_graph,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_graph != nullptr) {
    *out_graph = nullptr;
  }
  if (capture == nullptr || out_graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture pointer or out_graph is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END);
  if (*capture == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner is null");
  }
  RileyCudaGraphCapture* const owner = *capture;
  const uint64_t capture_id = owner->capture_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                       capture_id, 0, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->prepared_graph == nullptr || owner->fill_buffer == nullptr ||
      !owner->fill_lease_held || owner->unreleased_graph != nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner is not a live fixed-operation capture");
  }
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  const bool is_gated_multiply_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  const bool is_residual_add_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kResidualAddBf16;
  const bool is_canonical_rms_norm_bf16 =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  const bool is_bf16_argmax =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
  const bool is_bf16_row_gather =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16RowGather;
  const bool is_bf16_row_gather_argmax =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  const bool is_bf16_row_gather_argmax_d2h =
      is_bf16_d2h_operation(owner->operation);
  const bool is_indexed_rope_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  const bool is_canonical_gemm_bf16 =
      is_canonical_gemm_bf16_operation(owner->operation);
  const bool is_ragged_paged_kv_write_bf16 =
      is_ragged_paged_fixed_address_operation(owner->operation);
  if ((!is_fill && !is_h2d && !is_silu_bf16 && !is_gated_multiply_bf16 &&
       !is_residual_add_bf16 && !is_canonical_rms_norm_bf16 &&
       !is_bf16_argmax && !is_bf16_row_gather &&
       !is_bf16_row_gather_argmax && !is_bf16_row_gather_argmax_d2h &&
       !is_indexed_rope_bf16 && !is_canonical_gemm_bf16 &&
       !is_ragged_paged_kv_write_bf16) ||
      (!is_residual_add_bf16 && !residual_add_capture_fields_are_clear(owner)) ||
      (!is_canonical_rms_norm_bf16 &&
       !canonical_rms_norm_capture_fields_are_clear(owner)) ||
      (!is_bf16_argmax && !bf16_argmax_capture_fields_are_clear(owner)) ||
      (!is_bf16_row_gather &&
       !bf16_row_gather_capture_fields_are_clear(owner)) ||
      (!is_bf16_row_gather_argmax &&
       !bf16_row_gather_argmax_capture_fields_are_clear(owner)) ||
      (!is_bf16_row_gather_argmax_d2h &&
       !bf16_row_gather_argmax_d2h_capture_fields_are_clear(owner)) ||
      (!is_indexed_rope_bf16 &&
       !indexed_rope_bf16_capture_fields_are_clear(owner)) ||
      (!is_canonical_gemm_bf16 &&
       !canonical_gemm_bf16_capture_fields_are_clear(owner)) ||
      (!is_ragged_paged_kv_write_bf16 &&
       !ragged_paged_kv_cache_write_bf16_capture_fields_are_clear(owner)) ||
      (is_fill && (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
                   owner->h2d_source_lease_held || owner->silu_input != nullptr ||
                   owner->silu_element_count != 0 ||
                   owner->silu_enqueue_count != 0 ||
                   owner->silu_input_lease_held ||
                   owner->gated_multiply_activated_gate != nullptr ||
                   owner->gated_multiply_up != nullptr ||
                   owner->gated_multiply_element_count != 0 ||
                   owner->gated_multiply_enqueue_count != 0 ||
                   owner->gated_multiply_activated_gate_lease_held ||
                   owner->gated_multiply_up_lease_held)) ||
      (is_h2d && (owner->h2d_source == nullptr ||
                  !owner->h2d_source_lease_held || owner->h2d_byte_len == 0 ||
                  owner->h2d_source->host_data == nullptr ||
                  owner->h2d_source->byte_len != owner->h2d_byte_len ||
                  owner->fill_buffer->byte_len != owner->h2d_byte_len ||
                  owner->silu_input != nullptr || owner->silu_element_count != 0 ||
                  owner->silu_enqueue_count != 0 ||
                  owner->silu_input_lease_held ||
                  owner->gated_multiply_activated_gate != nullptr ||
                  owner->gated_multiply_up != nullptr ||
                  owner->gated_multiply_element_count != 0 ||
                  owner->gated_multiply_enqueue_count != 0 ||
                  owner->gated_multiply_activated_gate_lease_held ||
                  owner->gated_multiply_up_lease_held)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_enqueue_count != 0 || owner->h2d_source_lease_held ||
        owner->silu_input == nullptr ||
        owner->silu_input == owner->fill_buffer ||
        !owner->silu_input_lease_held || owner->silu_element_count == 0 ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->fill_element_count != 0 || owner->fill_enqueue_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0 ||
        owner->gated_multiply_enqueue_count != 0 ||
        owner->gated_multiply_activated_gate_lease_held ||
        owner->gated_multiply_up_lease_held)) ||
      (is_gated_multiply_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_enqueue_count != 0 || owner->h2d_source_lease_held ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->silu_enqueue_count != 0 || owner->silu_input_lease_held ||
        owner->fill_element_count != 0 || owner->fill_enqueue_count != 0 ||
        owner->gated_multiply_activated_gate == nullptr ||
        owner->gated_multiply_up == nullptr ||
        owner->gated_multiply_activated_gate == owner->gated_multiply_up ||
        owner->gated_multiply_activated_gate == owner->fill_buffer ||
        owner->gated_multiply_up == owner->fill_buffer ||
        !owner->gated_multiply_activated_gate_lease_held ||
        !owner->gated_multiply_up_lease_held ||
        owner->gated_multiply_element_count == 0 ||
        !same_context(owner->owner,
                      owner->gated_multiply_activated_gate->owner) ||
        !same_context(owner->owner, owner->gated_multiply_up->owner) ||
        owner->gated_multiply_activated_gate->device_data == nullptr ||
        owner->gated_multiply_up->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_activated_gate->byte_len /
                sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16))) ||
      (is_residual_add_bf16 &&
       !residual_add_capture_state_is_valid(owner)) ||
      (is_canonical_rms_norm_bf16 &&
       !canonical_rms_norm_capture_state_is_valid(owner)) ||
      (is_bf16_argmax && !bf16_argmax_capture_state_is_valid(owner)) ||
      (is_bf16_row_gather &&
       !bf16_row_gather_capture_state_is_valid(owner)) ||
      (is_bf16_row_gather_argmax &&
       !bf16_row_gather_argmax_capture_state_is_valid(owner)) ||
      (is_bf16_row_gather_argmax_d2h &&
       !bf16_d2h_capture_state_is_valid(owner)) ||
      (is_indexed_rope_bf16 &&
       !indexed_rope_bf16_capture_state_is_valid(owner)) ||
      (is_canonical_gemm_bf16 &&
       !canonical_gemm_bf16_capture_state_is_valid(owner)) ||
      (is_ragged_paged_kv_write_bf16 &&
       !ragged_paged_kv_cache_write_bf16_capture_state_is_valid(owner))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture owner has invalid fixed-operation geometry");
  }
  if (owner->owner_thread != native_thread_token() ||
      !thread_graph_capture_is_owner(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "thread-local capture must end on its begin thread");
  }
  if (!owner->capture_started || owner->capture_terminated ||
      (is_fill && owner->fill_enqueue_count == 0) ||
      (is_h2d && owner->h2d_enqueue_count != 1) ||
      (is_silu_bf16 && owner->silu_enqueue_count != 1) ||
      (is_gated_multiply_bf16 &&
       owner->gated_multiply_enqueue_count != 1) ||
      (is_residual_add_bf16 && owner->residual_add_enqueue_count != 1) ||
      (is_canonical_rms_norm_bf16 &&
       owner->canonical_rms_norm_enqueue_count != 1) ||
      (is_bf16_argmax && owner->bf16_argmax_enqueue_count != 1) ||
      (is_bf16_row_gather && owner->bf16_row_gather_enqueue_count != 1) ||
      (is_bf16_row_gather_argmax &&
       owner->bf16_row_gather_argmax_enqueue_count != 1) ||
      (is_bf16_row_gather_argmax_d2h &&
       owner->bf16_row_gather_argmax_d2h_enqueue_count != 1) ||
      (is_indexed_rope_bf16 &&
       owner->indexed_rope_bf16_enqueue_count != 1) ||
      (is_canonical_gemm_bf16 &&
       owner->canonical_gemm_bf16.enqueue_count != 1) ||
      (is_ragged_paged_kv_write_bf16 &&
       owner->ragged_paged_kv_write_bf16.enqueue_count != 1)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kEndOperation,
                            "capture end requires its admitted operation enqueue contract");
  }
  if (owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      (is_h2d && owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1) ||
      (is_silu_bf16 &&
       owner->silu_input->active_uses.load(std::memory_order_acquire) != 1) ||
      (is_gated_multiply_bf16 &&
       (owner->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_residual_add_bf16 &&
       (owner->residual_add_left->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->residual_add_right->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_canonical_rms_norm_bf16 &&
       (owner->canonical_rms_norm_input->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->canonical_rms_norm_weight->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_bf16_argmax &&
       owner->bf16_argmax_logits->active_uses.load(
           std::memory_order_acquire) != 1) ||
      (is_bf16_row_gather &&
       (owner->bf16_row_gather_input->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_indices->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_bf16_row_gather_argmax &&
       (owner->bf16_row_gather_argmax_input->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_argmax_indices->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_argmax_gathered_logits->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_bf16_row_gather_argmax_d2h &&
       (owner->bf16_row_gather_argmax_d2h_input->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_argmax_d2h_indices->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_argmax_d2h_gathered_logits->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->bf16_row_gather_argmax_d2h_pinned_results->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_indexed_rope_bf16 &&
       (owner->indexed_rope_bf16_input->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->indexed_rope_bf16_cos->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->indexed_rope_bf16_sin->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->indexed_rope_bf16_positions->active_uses.load(
            std::memory_order_acquire) != 1)) ||
      (is_canonical_gemm_bf16 &&
       !riley_cuda_internal::canonical_gemm_bf16_graph_state_is_valid(
           owner->owner, owner->stream, owner->fill_buffer,
           owner->canonical_gemm_bf16, true))) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kEndOperation,
                          "fixed graph capture resource lease was corrupted");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kEndOperation, owner);
  bool end_attempted = false;
  bool termination_known = false;
  bool graph_release_known = false;
  cudaGraph_t returned_graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    end_attempted = true;
    const cudaError_t end_result =
        cudaStreamEndCapture(owner->stream->stream, &returned_graph);
    if (end_result == cudaSuccess) {
      termination_known = true;
    } else {
      status = runtime_error(end_result, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                             kEndOperation);
      termination_known = capture_end_is_known(owner->stream);
    }
    if (termination_known) {
      graph_release_known = true;
      // Successful end transfers the graph below. An error outcome instead
      // discards any returned graph exactly once so the capture owner can
      // still recover its Rust deferred-close ledger when that destruction is
      // fully known.
      if (status != RILEY_CUDA_STATUS_SUCCESS && returned_graph != nullptr) {
        const cudaError_t destroy_result = cudaGraphDestroy(returned_graph);
        if (destroy_result != cudaSuccess) {
          owner->unreleased_graph = returned_graph;
          graph_release_known = false;
        }
      }
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kEndOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!end_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                         capture_id, 0, false, false, false,
                         !restoration_known);
    return status;
  }

  // cudaStreamEndCapture is one-shot. Consume the raw input after the CUDA
  // attempt and retain any uncertain owner/lease state rather than permitting
  // a second end or destroy attempt through a stale handle.
  *capture = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && termination_known &&
      graph_release_known && restoration_known && returned_graph != nullptr) {
    owner->prepared_graph->graph = returned_graph;
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    RileyCudaGraph* const graph = owner->prepared_graph;
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS &&
        transfer_capture_owner_to_graph(owner)) {
      *out_graph = graph;
      // owner has been freed. Do not dereference it beyond this point.
      record_graph_outcome(out_graph_error,
                           RILEY_CUDA_GRAPH_STAGE_CAPTURE_END, capture_id, 0,
                           false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    if (deferred_close_status != RILEY_CUDA_STATUS_SUCCESS &&
        error != nullptr && error->struct_size >= sizeof(*error)) {
      *error = deferred_close_error;
    }
    if (deferred_close_status != RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status;
    } else {
      status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                              kEndOperation,
                              "failed to transfer the recovered capture owner to graph ownership");
    }
  } else if (termination_known && graph_release_known && restoration_known) {
    // A deferred end error can still leave capture physically terminated. In
    // that case discard the graph and run the same one-shot recovery as abort;
    // this lets the safe wrapper finish its TLS deferred-context ledger only
    // when every capture-local close has a known result.
    if (returned_graph == nullptr && status == RILEY_CUDA_STATUS_SUCCESS) {
      status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                              kEndOperation,
                              "cudaStreamEndCapture succeeded without a graph handle");
    }
    owner->capture_terminated = true;
    RileyCudaErrorInfo deferred_close_error{};
    deferred_close_error.struct_size = sizeof(deferred_close_error);
    const RileyCudaStatus deferred_close_status =
        drain_capture_deferred_closes(owner, &deferred_close_error);
    bool released = false;
    if (deferred_close_status == RILEY_CUDA_STATUS_SUCCESS) {
      const bool is_h2d =
          owner->operation == RileyCudaGraphCaptureOperation::kH2D;
      const bool is_silu_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
      const bool is_gated_multiply_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
      const bool is_residual_add_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kResidualAddBf16;
      const bool is_canonical_rms_norm_bf16 =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
      const bool is_bf16_argmax =
          owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
      const bool is_bf16_row_gather =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kBf16RowGather;
      const bool is_bf16_row_gather_argmax =
          owner->operation ==
          RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
      const bool is_bf16_row_gather_argmax_d2h =
          is_bf16_d2h_operation(owner->operation);
      const bool is_indexed_rope_bf16 =
          owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
      const bool is_canonical_gemm_bf16 =
          is_canonical_gemm_bf16_operation(owner->operation);
      const bool is_ragged_paged_kv_write_bf16 =
          is_ragged_paged_fixed_address_operation(owner->operation);
      const bool is_fill =
          owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
      const bool release_graph_first =
          is_h2d || is_silu_bf16 || is_gated_multiply_bf16 ||
          is_residual_add_bf16 || is_canonical_rms_norm_bf16 ||
          is_bf16_argmax || is_bf16_row_gather ||
          is_bf16_row_gather_argmax || is_bf16_row_gather_argmax_d2h ||
          is_indexed_rope_bf16 || is_canonical_gemm_bf16 ||
          is_ragged_paged_kv_write_bf16;
      const bool prepared_graph_released =
          release_graph_first ? destroy_prepared_graph_storage(owner) : true;
      const bool operation_released =
          prepared_graph_released &&
          (is_h2d ? release_capture_h2d_leases(owner)
                  : is_silu_bf16 ? release_capture_silu_bf16_leases(owner)
                                 : is_gated_multiply_bf16
                                       ? release_capture_gated_multiply_bf16_leases(
                                             owner)
                                 : is_residual_add_bf16
                                       ? release_capture_residual_add_bf16_leases(
                                             owner)
                                 : is_canonical_rms_norm_bf16
                                       ? release_capture_canonical_rms_norm_bf16_leases(
                                             owner)
                                 : is_bf16_argmax
                                       ? release_capture_bf16_argmax_leases(owner)
                                 : is_bf16_row_gather
                                       ? release_capture_bf16_row_gather_leases(
                                             owner)
                                 : is_bf16_row_gather_argmax
                                       ? release_capture_bf16_row_gather_argmax_leases(
                                             owner)
                                 : is_bf16_row_gather_argmax_d2h
                                       ? release_capture_bf16_row_gather_argmax_d2h_leases(
                                             owner)
                                 : is_indexed_rope_bf16
                                       ? release_capture_indexed_rope_bf16_leases(
                                             owner)
                                 : is_canonical_gemm_bf16
                                       ? release_capture_canonical_gemm_bf16_leases(
                                             owner)
                                 : is_ragged_paged_kv_write_bf16
                                       ? release_capture_ragged_paged_kv_cache_write_bf16_leases(
                                             owner)
                                 : is_fill ? release_capture_fill_lease(owner)
                                           : false);
      const bool cleanup_graph_released =
          release_graph_first
              ? prepared_graph_released
              : operation_released && destroy_prepared_graph_storage(owner);
      released = cleanup_graph_released && operation_released &&
                 release_capture_owner(owner);
    } else if (error != nullptr && error->struct_size >= sizeof(*error)) {
      *error = deferred_close_error;
    }
    if (!released && status == RILEY_CUDA_STATUS_SUCCESS) {
      status = deferred_close_status == RILEY_CUDA_STATUS_SUCCESS
                   ? internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                    kEndOperation,
                                    "failed to release recovered capture after graph end")
                   : deferred_close_status;
    }
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                         capture_id, 0, false, false, released, !released);
    return status;
  }

  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CAPTURE_END,
                       capture_id, 0, false, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_instantiate(
    RileyCudaGraph** graph, RileyCudaGraphExec** out_exec,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_exec != nullptr) {
    *out_exec = nullptr;
  }
  if (graph == nullptr || out_exec == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "graph pointer or out_exec is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE);
  if (*graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation, "graph owner is null");
  }
  RileyCudaGraph* const owner = *graph;
  const uint64_t capture_id = owner->capture_id;
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  const bool is_gated_multiply_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  const bool is_residual_add_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kResidualAddBf16;
  const bool is_canonical_rms_norm_bf16 =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  const bool is_bf16_argmax =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
  const bool is_bf16_row_gather =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16RowGather;
  const bool is_bf16_row_gather_argmax =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  const bool is_bf16_row_gather_argmax_d2h =
      is_bf16_d2h_operation(owner->operation);
  const bool is_indexed_rope_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  const bool is_canonical_gemm_bf16 =
      is_canonical_gemm_bf16_operation(owner->operation);
  const bool is_ragged_paged_kv_write_bf16 =
      is_ragged_paged_fixed_address_operation(owner->operation);
  if (is_residual_add_bf16 && !residual_add_graph_state_is_valid(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured residual-add graph has invalid fixed resource state");
  }
  if (!is_residual_add_bf16 && !residual_add_graph_fields_are_clear(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph mixes residual-add state with another operation");
  }
  if (is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured canonical RMSNorm graph has invalid fixed resource state");
  }
  if (!is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes canonical RMSNorm state with another operation");
  }
  if (is_bf16_argmax && !bf16_argmax_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured deterministic BF16 argmax graph has invalid fixed resource state");
  }
  if (!is_bf16_argmax && !bf16_argmax_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes deterministic BF16 argmax state with another operation");
  }
  if (is_bf16_row_gather && !bf16_row_gather_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured BF16 row-gather graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather && !bf16_row_gather_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes BF16 row-gather state with another operation");
  }
  if (is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured BF16 row-gather argmax graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes BF16 row-gather argmax state with another operation");
  }
  if (is_bf16_row_gather_argmax_d2h &&
      !bf16_d2h_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured BF16 row-gather argmax D2H graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax_d2h &&
      !bf16_row_gather_argmax_d2h_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
                            "captured graph mixes BF16 row-gather argmax D2H state with another operation");
  }
  if (is_indexed_rope_bf16 && !indexed_rope_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured BF16 indexed-RoPE graph has invalid fixed resource state");
  }
  if (!is_indexed_rope_bf16 &&
      !indexed_rope_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
                            "captured graph mixes BF16 indexed-RoPE state with another operation");
  }
  if (is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured canonical cuBLASLt GEMM graph has invalid fixed resource state");
  }
  if (!is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes canonical cuBLASLt GEMM state with another operation");
  }
  if (is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured BF16 ragged paged-KV graph has invalid fixed resource state");
  }
  if (!is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kInstantiateOperation,
        "captured graph mixes BF16 ragged paged-KV state with another operation");
  }
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16 && !is_gated_multiply_bf16 &&
       !is_residual_add_bf16 && !is_canonical_rms_norm_bf16 &&
       !is_bf16_argmax && !is_bf16_row_gather &&
       !is_bf16_row_gather_argmax && !is_bf16_row_gather_argmax_d2h &&
       !is_indexed_rope_bf16 && !is_canonical_gemm_bf16 &&
       !is_ragged_paged_kv_write_bf16) ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph resource lease is invalid");
  }
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input == nullptr || owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_gated_multiply_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate == nullptr ||
        owner->gated_multiply_up == nullptr ||
        owner->gated_multiply_activated_gate == owner->gated_multiply_up ||
        owner->gated_multiply_activated_gate == owner->fill_buffer ||
        owner->gated_multiply_up == owner->fill_buffer ||
        owner->gated_multiply_element_count == 0 ||
        !same_context(owner->owner,
                      owner->gated_multiply_activated_gate->owner) ||
        !same_context(owner->owner, owner->gated_multiply_up->owner) ||
        owner->gated_multiply_activated_gate->device_data == nullptr ||
        owner->gated_multiply_up->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_activated_gate->byte_len /
                sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph has invalid fixed-operation resource state");
  }
  if (owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kInstantiateOperation,
                            "captured graph context is poisoned");
  }
  const uint64_t exec_id = next_graph_exec_id();
  if (exec_id == 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kInstantiateOperation,
                          "CUDA Graph exec ID space is exhausted");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                       capture_id, exec_id, false, false, false, false);
  void* storage = std::calloc(1, sizeof(RileyCudaGraphExec));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kInstantiateOperation,
                     "host allocation failed for CUDA Graph exec owner");
  }
  auto* exec = new (storage) RileyCudaGraphExec(
      owner->owner, owner->stream, owner->fill_buffer, capture_id, exec_id,
      owner->operation, owner->h2d_source, owner->h2d_byte_len,
      owner->silu_input, owner->silu_element_count,
      owner->gated_multiply_activated_gate, owner->gated_multiply_up,
      owner->gated_multiply_element_count, owner->residual_add_left,
      owner->residual_add_right, owner->residual_add_element_count,
      owner->canonical_rms_norm_input, owner->canonical_rms_norm_weight,
      owner->canonical_rms_norm_row_count,
      owner->canonical_rms_norm_hidden_size,
      owner->canonical_rms_norm_epsilon, owner->bf16_argmax_logits,
      owner->bf16_argmax_row_count, owner->bf16_argmax_vocabulary_size,
      owner->bf16_row_gather_input, owner->bf16_row_gather_indices,
      owner->bf16_row_gather_input_row_count,
      owner->bf16_row_gather_output_row_count,
      owner->bf16_row_gather_column_count,
      owner->bf16_row_gather_argmax_input,
      owner->bf16_row_gather_argmax_indices,
      owner->bf16_row_gather_argmax_gathered_logits,
      owner->bf16_row_gather_argmax_input_row_count,
      owner->bf16_row_gather_argmax_output_row_count,
      owner->bf16_row_gather_argmax_vocabulary_size,
      owner->bf16_row_gather_argmax_d2h_input,
      owner->bf16_row_gather_argmax_d2h_indices,
      owner->bf16_row_gather_argmax_d2h_gathered_logits,
      owner->bf16_row_gather_argmax_d2h_pinned_results,
      owner->bf16_row_gather_argmax_d2h_input_row_count,
      owner->bf16_row_gather_argmax_d2h_output_row_count,
      owner->bf16_row_gather_argmax_d2h_vocabulary_size,
      owner->bf16_row_gather_argmax_d2h_result_byte_len);
  exec->indexed_rope_bf16_input = owner->indexed_rope_bf16_input;
  exec->indexed_rope_bf16_cos = owner->indexed_rope_bf16_cos;
  exec->indexed_rope_bf16_sin = owner->indexed_rope_bf16_sin;
  exec->indexed_rope_bf16_positions = owner->indexed_rope_bf16_positions;
  exec->indexed_rope_bf16_active_row_count =
      owner->indexed_rope_bf16_active_row_count;
  exec->indexed_rope_bf16_head_count = owner->indexed_rope_bf16_head_count;
  exec->indexed_rope_bf16_head_size = owner->indexed_rope_bf16_head_size;
  exec->indexed_rope_bf16_rotary_dimension =
      owner->indexed_rope_bf16_rotary_dimension;
  exec->indexed_rope_bf16_table_position_count =
      owner->indexed_rope_bf16_table_position_count;

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                                        kInstantiateOperation);
  bool instantiate_attempted = false;
  cudaGraphExec_t native_exec = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    instantiate_attempted = true;
    status = runtime_error(
        cudaGraphInstantiate(&native_exec, owner->graph, nullptr, nullptr, 0),
        error, RILEY_CUDA_ERROR_STAGE_CREATE, kInstantiateOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CREATE,
                       kInstantiateOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!instantiate_attempted) {
    exec->~RileyCudaGraphExec();
    std::free(exec);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Instantiation is one-shot from the safe ownership perspective. The graph
  // pointer is consumed after any CUDA instantiate attempt; an uncertain
  // native exec/graph pair is retained intentionally rather than retried.
  *graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known &&
      native_exec != nullptr) {
    exec->graph = owner->graph;
    exec->exec = native_exec;
    exec->owns_capture_leases = true;
    if (is_canonical_gemm_bf16) {
      exec->canonical_gemm_bf16 = owner->canonical_gemm_bf16;
      owner->canonical_gemm_bf16 = RileyCudaCanonicalGemmBf16GraphState{};
    }
    if (is_ragged_paged_kv_write_bf16) {
      exec->ragged_paged_kv_write_bf16 =
          owner->ragged_paged_kv_write_bf16;
      owner->ragged_paged_kv_write_bf16 =
          RileyCudaRaggedPagedKvCacheWriteBf16State{};
    }
    owner->graph = nullptr;
    owner->owns_capture_leases = false;
    owner->~RileyCudaGraph();
    std::free(owner);
    *out_exec = exec;
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                         capture_id, exec_id, false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  // Preserve any opaque CUDA outputs in deliberately leaked host owners. No
  // close/retry is safe after a failed instantiate call because CUDA may have
  // consumed or partially initialized either native object before surfacing a
  // deferred error.
  exec->graph = owner->graph;
  exec->exec = native_exec;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INSTANTIATE,
                       capture_id, exec_id, false, false, false, true);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kInstantiateOperation,
                          "cudaGraphInstantiate succeeded without an exec handle");
  }
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_exec_stage_h2d_source(
    RileyCudaGraphExec* exec, RileyCudaPinnedHostBuffer* source,
    const uint8_t* bytes, uint64_t byte_len,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kStageH2DOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE);
  if (exec == nullptr || source == nullptr || bytes == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kStageH2DOperation,
                            "graph H2D exec, retained source, or payload is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE,
                       capture_id, exec_id, false, false, false, false);
  if (exec->owner == nullptr || exec->stream == nullptr ||
      exec->fill_buffer == nullptr || exec->h2d_source == nullptr ||
      exec->operation != RileyCudaGraphCaptureOperation::kH2D ||
      exec->h2d_source != source || exec->graph == nullptr ||
      exec->exec == nullptr || !exec->owns_capture_leases ||
      exec->h2d_byte_len == 0 || byte_len != exec->h2d_byte_len ||
      source->host_data == nullptr || source->byte_len != exec->h2d_byte_len ||
      exec->fill_buffer->byte_len != exec->h2d_byte_len ||
      !same_context(exec->owner, exec->stream->owner) ||
      !same_context(exec->owner, exec->fill_buffer->owner) ||
      !same_context(exec->owner, source->owner) ||
      exec->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      source->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->silu_input != nullptr || exec->silu_element_count != 0 ||
      exec->gated_multiply_activated_gate != nullptr ||
      exec->gated_multiply_up != nullptr ||
      exec->gated_multiply_element_count != 0 ||
      !residual_add_exec_fields_are_clear(exec) ||
      !canonical_rms_norm_exec_fields_are_clear(exec) ||
      !bf16_argmax_exec_fields_are_clear(exec) ||
      !bf16_row_gather_exec_fields_are_clear(exec) ||
      !bf16_row_gather_argmax_exec_fields_are_clear(exec) ||
      !bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec) ||
      !indexed_rope_bf16_exec_fields_are_clear(exec) ||
      !canonical_gemm_bf16_exec_fields_are_clear(exec) ||
      !ragged_paged_kv_cache_write_bf16_exec_fields_are_clear(exec) ||
      exec->launch_in_flight || exec->h2d_input_staged || exec->poisoned ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kStageH2DOperation,
                            "graph H2D exec is busy, poisoned, or lost its exact retained resource lease");
  }
  // The graph node retains this exact pinned allocation address. This private
  // stage is deliberately the sole mutable path while its normal active-use
  // guard remains held; no CUDA call, node update, or pointer mutation occurs.
  std::memmove(source->host_data, bytes, static_cast<size_t>(byte_len));
  exec->h2d_input_staged = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_INPUT_STAGE,
                       capture_id, exec_id, false, false, false, false);
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_graph_exec_launch(
    RileyCudaGraphExec* exec, RileyCudaStream* stream,
    RileyCudaGraphLaunch** out_launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (out_launch != nullptr) {
    *out_launch = nullptr;
  }
  if (out_launch == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation, "out_launch is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH);
  if (exec == nullptr || stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation, "graph exec or stream is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  const bool is_fill =
      exec->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = exec->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      exec->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  const bool is_gated_multiply_bf16 =
      exec->operation == RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  const bool is_residual_add_bf16 =
      exec->operation == RileyCudaGraphCaptureOperation::kResidualAddBf16;
  const bool is_canonical_rms_norm_bf16 =
      exec->operation ==
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  const bool is_bf16_argmax =
      exec->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
  const bool is_bf16_row_gather =
      exec->operation == RileyCudaGraphCaptureOperation::kBf16RowGather;
  const bool is_bf16_row_gather_argmax =
      exec->operation ==
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  const bool is_bf16_row_gather_argmax_d2h =
      is_bf16_d2h_operation(exec->operation);
  const bool is_indexed_rope_bf16 =
      exec->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  const bool is_canonical_gemm_bf16 =
      is_canonical_gemm_bf16_operation(exec->operation);
  const bool is_ragged_paged_kv_write_bf16 =
      is_ragged_paged_fixed_address_operation(exec->operation);
  if (is_residual_add_bf16 && !residual_add_exec_state_is_valid(exec)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "residual-add graph exec has invalid fixed resource state");
  }
  if (!is_residual_add_bf16 && !residual_add_exec_fields_are_clear(exec)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec mixes residual-add state with another operation");
  }
  if (is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "canonical RMSNorm graph exec has invalid fixed resource state");
  }
  if (!is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes canonical RMSNorm state with another operation");
  }
  if (is_bf16_argmax && !bf16_argmax_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "deterministic BF16 argmax graph exec has invalid fixed resource state");
  }
  if (!is_bf16_argmax && !bf16_argmax_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes deterministic BF16 argmax state with another operation");
  }
  if (is_bf16_row_gather && !bf16_row_gather_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "BF16 row-gather graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather && !bf16_row_gather_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes BF16 row-gather state with another operation");
  }
  if (is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "BF16 row-gather argmax graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes BF16 row-gather argmax state with another operation");
  }
  if (is_bf16_row_gather_argmax_d2h &&
      !bf16_d2h_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "BF16 row-gather argmax D2H graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax_d2h &&
      !bf16_row_gather_argmax_d2h_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes BF16 row-gather argmax D2H state with another operation");
  }
  if (is_indexed_rope_bf16 && !indexed_rope_bf16_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "BF16 indexed-RoPE graph exec has invalid fixed resource state");
  }
  if (!is_indexed_rope_bf16 &&
      !indexed_rope_bf16_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
                            "graph exec mixes BF16 indexed-RoPE state with another operation");
  }
  if (is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "canonical cuBLASLt GEMM graph exec has invalid fixed resource state");
  }
  if (!is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes canonical cuBLASLt GEMM state with another operation");
  }
  if (is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_exec_state_is_valid(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "BF16 ragged paged-KV graph exec has invalid fixed resource state");
  }
  if (!is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_exec_fields_are_clear(exec)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kLaunchOperation,
        "graph exec mixes BF16 ragged paged-KV state with another operation");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                       capture_id, exec_id, false, false, false, false);
  if (exec->owner == nullptr || exec->stream == nullptr ||
      exec->fill_buffer == nullptr || exec->graph == nullptr ||
      exec->exec == nullptr || !exec->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16 && !is_gated_multiply_bf16 &&
       !is_residual_add_bf16 && !is_canonical_rms_norm_bf16 &&
       !is_bf16_argmax && !is_bf16_row_gather &&
       !is_bf16_row_gather_argmax && !is_bf16_row_gather_argmax_d2h &&
       !is_indexed_rope_bf16 && !is_canonical_gemm_bf16 &&
       !is_ragged_paged_kv_write_bf16)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec has invalid retained capture resources");
  }
  if (stream != exec->stream) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec must launch on its exact captured stream");
  }
  if (!same_context(exec->owner, stream->owner) ||
      !same_context(exec->owner, exec->fill_buffer->owner) ||
      exec->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      exec->launch_in_flight || exec->poisoned ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec is busy, poisoned, or lost its retained resource lease");
  }
  if ((is_fill &&
       (exec->h2d_source != nullptr || exec->h2d_byte_len != 0 ||
        exec->h2d_input_staged || exec->silu_input != nullptr ||
        exec->silu_element_count != 0 ||
        exec->gated_multiply_activated_gate != nullptr ||
        exec->gated_multiply_up != nullptr ||
        exec->gated_multiply_element_count != 0)) ||
      (is_h2d &&
       (exec->h2d_byte_len == 0 ||
        exec->h2d_source == nullptr ||
        !same_context(exec->owner, exec->h2d_source->owner) ||
        exec->h2d_source->host_data == nullptr ||
        exec->h2d_source->byte_len != exec->h2d_byte_len ||
        exec->fill_buffer->byte_len != exec->h2d_byte_len ||
        exec->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        !exec->h2d_input_staged || exec->silu_input != nullptr ||
        exec->silu_element_count != 0 ||
        exec->gated_multiply_activated_gate != nullptr ||
        exec->gated_multiply_up != nullptr ||
        exec->gated_multiply_element_count != 0)) ||
      (is_silu_bf16 &&
       (exec->h2d_source != nullptr || exec->h2d_byte_len != 0 ||
        exec->h2d_input_staged || exec->silu_input == nullptr ||
        exec->silu_input == exec->fill_buffer ||
        exec->silu_element_count == 0 ||
        !same_context(exec->owner, exec->silu_input->owner) ||
        exec->silu_input->device_data == nullptr ||
        exec->fill_buffer->device_data == nullptr ||
        exec->silu_element_count >
            exec->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        exec->silu_element_count >
            exec->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        exec->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        exec->gated_multiply_activated_gate != nullptr ||
        exec->gated_multiply_up != nullptr ||
        exec->gated_multiply_element_count != 0)) ||
      (is_gated_multiply_bf16 &&
       (exec->h2d_source != nullptr || exec->h2d_byte_len != 0 ||
        exec->h2d_input_staged || exec->silu_input != nullptr ||
        exec->silu_element_count != 0 ||
        exec->gated_multiply_activated_gate == nullptr ||
        exec->gated_multiply_up == nullptr ||
        exec->gated_multiply_activated_gate == exec->gated_multiply_up ||
        exec->gated_multiply_activated_gate == exec->fill_buffer ||
        exec->gated_multiply_up == exec->fill_buffer ||
        exec->gated_multiply_element_count == 0 ||
        !same_context(exec->owner,
                      exec->gated_multiply_activated_gate->owner) ||
        !same_context(exec->owner, exec->gated_multiply_up->owner) ||
        exec->gated_multiply_activated_gate->device_data == nullptr ||
        exec->gated_multiply_up->device_data == nullptr ||
        exec->fill_buffer->device_data == nullptr ||
        exec->gated_multiply_element_count >
            exec->gated_multiply_activated_gate->byte_len /
                sizeof(__nv_bfloat16) ||
        exec->gated_multiply_element_count >
            exec->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
        exec->gated_multiply_element_count >
            exec->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        exec->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        exec->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kLaunchOperation,
                            "graph exec has invalid fixed-operation replay state");
  }
  void* storage = std::calloc(1, sizeof(RileyCudaGraphLaunch));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kLaunchOperation,
                     "host allocation failed for CUDA Graph launch owner");
  }
  auto* launch = new (storage) RileyCudaGraphLaunch(exec, stream);

  CurrentContext scope(exec->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                        kLaunchOperation);
  bool launch_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    if (is_h2d) {
      // A launch attempt consumes its stage even if CUDA subsequently reports
      // a deferred error. Completion never restores this bit: every replay
      // must explicitly stage a new exact payload.
      exec->h2d_input_staged = false;
    }
    if (is_bf16_row_gather_argmax_d2h) {
      // A new replay supersedes the previous result receipt before CUDA sees
      // its fixed D2H destination, including a later deferred launch error.
      exec->bf16_row_gather_argmax_d2h_completion_visible = false;
    }
    launch_attempted = true;
    status = runtime_error(cudaGraphLaunch(exec->exec, stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_LAUNCH, kLaunchOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kLaunchOperation);
  const bool restoration_known =
      !exec->owner->restoration_failed.load(std::memory_order_acquire);
  if (!launch_attempted) {
    launch->~RileyCudaGraphLaunch();
    std::free(launch);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Once cudaGraphLaunch has been attempted, close/relaunch must remain
  // blocked even if CUDA reports a deferred failure. Give raw callers the
  // one completion owner, but fail closed if they decline to settle it.
  exec->launch_in_flight = true;
  *out_launch = launch;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                         capture_id, exec_id, true, false, false, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  exec->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_LAUNCH,
                       capture_id, exec_id, true, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_launch_complete(
    RileyCudaGraphLaunch** launch,
    RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (launch == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCompleteOperation, "launch pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCompleteOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION);
  if (*launch == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         0, 0, false, true, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraphLaunch* const owner = *launch;
  RileyCudaGraphExec* const exec = owner->exec;
  if (exec == nullptr || owner->stream == nullptr || exec->owner == nullptr ||
      exec->stream != owner->stream || !exec->launch_in_flight ||
      exec->exec == nullptr || !exec->owns_capture_leases) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCompleteOperation,
                            "graph launch owner is not a live completion boundary");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, false, false, false);

  CurrentContext scope(exec->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                                        kCompleteOperation);
  bool completion_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    completion_attempted = true;
    status = runtime_error(cudaStreamSynchronize(exec->stream->stream), error,
                           RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                           kCompleteOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                       kCompleteOperation);
  const bool restoration_known =
      !exec->owner->restoration_failed.load(std::memory_order_acquire);
  if (!completion_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         capture_id, exec_id, true, false, false,
                         !restoration_known);
    return status;
  }

  // Completion is also one-shot. Consume the raw owner even when a CUDA sync
  // reports an error; native resources remain poisoned and retained on an
  // ambiguous completion rather than accepting a second synchronize attempt.
  *launch = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    // A failed cudaGraphLaunch can surface a deferred error after submitting
    // work. This single successful synchronization proves the only in-flight
    // boundary has settled, so the launch-specific poison is recoverable and
    // the safe FFI may return the original launch error without stranding the
    // graph exec's permanent stream/buffer leases.
    exec->launch_in_flight = false;
    exec->poisoned = false;
    if (is_bf16_d2h_operation(exec->operation)) {
      exec->bf16_row_gather_argmax_d2h_completion_visible = true;
    }
    owner->~RileyCudaGraphLaunch();
    std::free(owner);
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         capture_id, exec_id, true, true, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  exec->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, false, false, true);
  return status;
}

extern "C" RileyCudaStatus
riley_cuda_graph_exec_read_bf16_row_gather_argmax_d2h_results(
    RileyCudaGraphExec* exec, uint8_t* destination,
    uint64_t destination_len, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16RowGatherArgmaxD2HResultsOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION);
  if (exec == nullptr || destination == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16RowGatherArgmaxD2HResultsOperation,
        "graph exec or result destination is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, false, false, false, false);
  if (exec->operation !=
          RileyCudaGraphCaptureOperation::kBf16RowGatherArgmaxD2H ||
      !bf16_row_gather_argmax_d2h_exec_state_is_valid(exec) ||
      exec->graph == nullptr || exec->exec == nullptr ||
      !exec->owns_capture_leases || exec->launch_in_flight || exec->poisoned ||
      !exec->bf16_row_gather_argmax_d2h_completion_visible ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16RowGatherArgmaxD2HResultsOperation,
        "BF16 row-gather argmax D2H results are not visible at a known graph completion boundary");
  }
  const uint64_t result_byte_len =
      exec->bf16_row_gather_argmax_d2h_result_byte_len;
  if (destination_len != result_byte_len ||
      result_byte_len > static_cast<uint64_t>(SIZE_MAX)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16RowGatherArgmaxD2HResultsOperation,
        "completion result destination length does not exactly match fixed D2H records");
  }
  std::memcpy(destination,
              exec->bf16_row_gather_argmax_d2h_pinned_results->host_data,
              static_cast<size_t>(result_byte_len));
  // The permanent pinned lease deliberately remains held, so generic pinned
  // buffer CPU access and close cannot invalidate the replayable graph node.
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, true, true, false);
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus
riley_cuda_graph_exec_read_bf16_embedding_status_d2h_report(
    RileyCudaGraphExec* exec, uint8_t* destination,
    uint64_t destination_len, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16EmbeddingStatusD2HReportOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION);
  if (exec == nullptr || destination == nullptr) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16EmbeddingStatusD2HReportOperation,
        "graph exec or embedding report destination is null");
  }
  const uint64_t capture_id = exec->capture_id;
  const uint64_t exec_id = exec->exec_id;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, false, false, false, false);
  if (exec->operation !=
          RileyCudaGraphCaptureOperation::kBf16EmbeddingStatusD2H ||
      !bf16_embedding_status_d2h_exec_state_is_valid(exec) ||
      exec->graph == nullptr || exec->exec == nullptr ||
      !exec->owns_capture_leases || exec->launch_in_flight || exec->poisoned ||
      !exec->bf16_row_gather_argmax_d2h_completion_visible ||
      exec->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16EmbeddingStatusD2HReportOperation,
        "BF16 embedding status D2H report is not visible at a known graph completion boundary");
  }
  const uint64_t report_byte_len =
      exec->bf16_row_gather_argmax_d2h_result_byte_len;
  if (destination_len != report_byte_len ||
      report_byte_len != sizeof(RileyCudaEmbeddingErrorReport) ||
      report_byte_len > static_cast<uint64_t>(SIZE_MAX)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,
        kReadBf16EmbeddingStatusD2HReportOperation,
        "completion report destination length does not exactly match the fixed embedding report");
  }

  // Do not cast the pinned byte allocation to the record type: the public
  // pinned-buffer ABI does not promise host pointer alignment. Validate a
  // local copy before publishing it so malformed device status can never be
  // mistaken for a user-visible token-range result.
  RileyCudaEmbeddingErrorReport report{};
  std::memcpy(&report,
              exec->bf16_row_gather_argmax_d2h_pinned_results->host_data,
              sizeof(report));
  const uint64_t vocabulary_size =
      exec->bf16_row_gather_argmax_d2h_input_row_count;
  const uint64_t token_count =
      exec->bf16_row_gather_argmax_d2h_output_row_count;
  const bool no_error = report.code == RILEY_CUDA_EMBEDDING_ERROR_NONE &&
                        report.token_position == 0 && report.token_id == 0;
  const bool token_out_of_range =
      report.code == RILEY_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE &&
      report.token_position < token_count && report.token_id >= vocabulary_size &&
      report.token_id <= UINT32_MAX;
  if (report.struct_size != sizeof(report) || report.reserved != 0 ||
      (!no_error && !token_out_of_range)) {
    // Completion itself is known, but an impossible record makes the reusable
    // graph and its fixed raw addresses fail-closed just like an ambiguous
    // CUDA completion. Do not publish malformed status data.
    exec->poisoned = true;
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                         capture_id, exec_id, true, true, false, true);
    return internal_error(
        error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
        kReadBf16EmbeddingStatusD2HReportOperation,
        "completed BF16 embedding status D2H report is malformed");
  }

  std::memcpy(destination, &report, sizeof(report));
  // A canonical OOB report is expected status data. It intentionally does
  // not poison this replayable graph or turn the raw read into an out-of-range
  // lifecycle error; the safe FFI decodes `code` after this call succeeds.
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_COMPLETION,
                       capture_id, exec_id, true, true, true, false);
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_graph_close(
    RileyCudaGraph** graph, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (graph == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation, "graph pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE);
  if (*graph == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE, 0, 0,
                         false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraph* const owner = *graph;
  const uint64_t capture_id = owner->capture_id;
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  const bool is_gated_multiply_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  const bool is_residual_add_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kResidualAddBf16;
  const bool is_canonical_rms_norm_bf16 =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  const bool is_bf16_argmax =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
  const bool is_bf16_row_gather =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16RowGather;
  const bool is_bf16_row_gather_argmax =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  const bool is_bf16_row_gather_argmax_d2h =
      is_bf16_d2h_operation(owner->operation);
  const bool is_indexed_rope_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  const bool is_canonical_gemm_bf16 =
      is_canonical_gemm_bf16_operation(owner->operation);
  const bool is_ragged_paged_kv_write_bf16 =
      is_ragged_paged_fixed_address_operation(owner->operation);
  if (is_residual_add_bf16 && !residual_add_graph_state_is_valid(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "residual-add graph has invalid fixed resource state");
  }
  if (!is_residual_add_bf16 && !residual_add_graph_fields_are_clear(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "captured graph mixes residual-add state with another operation");
  }
  if (is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "canonical RMSNorm graph has invalid fixed resource state");
  }
  if (!is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes canonical RMSNorm state with another operation");
  }
  if (is_bf16_argmax && !bf16_argmax_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "deterministic BF16 argmax graph has invalid fixed resource state");
  }
  if (!is_bf16_argmax && !bf16_argmax_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes deterministic BF16 argmax state with another operation");
  }
  if (is_bf16_row_gather && !bf16_row_gather_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "BF16 row-gather graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather && !bf16_row_gather_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes BF16 row-gather state with another operation");
  }
  if (is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "BF16 row-gather argmax graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes BF16 row-gather argmax state with another operation");
  }
  if (is_bf16_row_gather_argmax_d2h &&
      !bf16_d2h_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "BF16 row-gather argmax D2H graph has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax_d2h &&
      !bf16_row_gather_argmax_d2h_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes BF16 row-gather argmax D2H state with another operation");
  }
  if (is_indexed_rope_bf16 && !indexed_rope_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "BF16 indexed-RoPE graph has invalid fixed resource state");
  }
  if (!is_indexed_rope_bf16 &&
      !indexed_rope_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
                            "captured graph mixes BF16 indexed-RoPE state with another operation");
  }
  if (is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "canonical cuBLASLt GEMM graph has invalid fixed resource state");
  }
  if (!is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes canonical cuBLASLt GEMM state with another operation");
  }
  if (is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_graph_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "BF16 ragged paged-KV graph has invalid fixed resource state");
  }
  if (!is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_graph_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseGraphOperation,
        "captured graph mixes BF16 ragged paged-KV state with another operation");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, 0, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      !owner->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16 && !is_gated_multiply_bf16 &&
       !is_residual_add_bf16 && !is_canonical_rms_norm_bf16 &&
       !is_bf16_argmax && !is_bf16_row_gather &&
       !is_bf16_row_gather_argmax && !is_bf16_row_gather_argmax_d2h &&
       !is_indexed_rope_bf16 && !is_canonical_gemm_bf16 &&
       !is_ragged_paged_kv_write_bf16) ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "captured graph has invalid retained resource leases");
  }
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input == nullptr || owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_gated_multiply_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate == nullptr ||
        owner->gated_multiply_up == nullptr ||
        owner->gated_multiply_activated_gate == owner->gated_multiply_up ||
        owner->gated_multiply_activated_gate == owner->fill_buffer ||
        owner->gated_multiply_up == owner->fill_buffer ||
        owner->gated_multiply_element_count == 0 ||
        !same_context(owner->owner,
                      owner->gated_multiply_activated_gate->owner) ||
        !same_context(owner->owner, owner->gated_multiply_up->owner) ||
        owner->gated_multiply_activated_gate->device_data == nullptr ||
        owner->gated_multiply_up->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_activated_gate->byte_len /
                sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseGraphOperation,
                            "captured graph has invalid fixed-operation resource state");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kCloseGraphOperation);
  bool destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    destroy_attempted = true;
    status = runtime_error(cudaGraphDestroy(owner->graph), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kCloseGraphOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kCloseGraphOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!destroy_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                         capture_id, 0, false, false, false,
                         !restoration_known);
    return status;
  }

  *graph = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && restoration_known) {
    owner->graph = nullptr;
    const bool released =
        is_h2d ? release_graph_h2d_leases(owner->owner, owner->stream,
                                           owner->fill_buffer,
                                           owner->h2d_source)
               : is_silu_bf16
                     ? release_graph_silu_bf16_leases(owner->owner,
                                                       owner->stream,
                                                       owner->silu_input,
                                                       owner->fill_buffer)
                     : is_gated_multiply_bf16
                           ? release_graph_gated_multiply_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->gated_multiply_activated_gate,
                                 owner->gated_multiply_up, owner->fill_buffer)
                     : is_residual_add_bf16
                           ? release_graph_residual_add_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->residual_add_left,
                                 owner->residual_add_right, owner->fill_buffer)
                     : is_canonical_rms_norm_bf16
                           ? release_graph_canonical_rms_norm_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->canonical_rms_norm_input,
                                 owner->canonical_rms_norm_weight,
                                 owner->fill_buffer)
                     : is_bf16_argmax
                           ? release_graph_bf16_argmax_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_argmax_logits,
                                 owner->fill_buffer)
                     : is_bf16_row_gather
                           ? release_graph_bf16_row_gather_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_input,
                                 owner->bf16_row_gather_indices,
                                 owner->fill_buffer)
                     : is_bf16_row_gather_argmax
                           ? release_graph_bf16_row_gather_argmax_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_argmax_input,
                                 owner->bf16_row_gather_argmax_indices,
                                 owner->bf16_row_gather_argmax_gathered_logits,
                                 owner->fill_buffer)
                     : is_bf16_row_gather_argmax_d2h
                           ? release_graph_bf16_row_gather_argmax_d2h_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_argmax_d2h_input,
                                 owner->bf16_row_gather_argmax_d2h_indices,
                                 owner->bf16_row_gather_argmax_d2h_gathered_logits,
                                 owner->fill_buffer,
                                 owner->bf16_row_gather_argmax_d2h_pinned_results)
                     : is_indexed_rope_bf16
                           ? release_graph_indexed_rope_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->indexed_rope_bf16_input,
                                 owner->indexed_rope_bf16_cos,
                                 owner->indexed_rope_bf16_sin,
                                 owner->indexed_rope_bf16_positions,
                                 owner->fill_buffer)
                     : is_canonical_gemm_bf16
                           ? release_graph_canonical_gemm_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->fill_buffer,
                                 owner->canonical_gemm_bf16)
                     : is_ragged_paged_kv_write_bf16
                           ? release_graph_ragged_paged_kv_cache_write_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->fill_buffer,
                                 owner->ragged_paged_kv_write_bf16)
                     : release_graph_leases(owner->owner, owner->stream,
                                            owner->fill_buffer);
    if (released) {
      owner->owns_capture_leases = false;
      owner->~RileyCudaGraph();
      std::free(owner);
      record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                           capture_id, 0, false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseGraphOperation,
                            "failed to release graph stream, retained buffers, or context lease");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, 0, false, false, false, true);
  return status;
}

extern "C" RileyCudaStatus riley_cuda_graph_exec_close(
    RileyCudaGraphExec** exec, RileyCudaGraphErrorInfo* out_graph_error,
    RileyCudaErrorInfo* error) noexcept {
  using riley_cuda_internal::clear_error;

  clear_error(error);
  if (exec == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation, "graph exec pointer is null");
  }
  if (!graph_error_is_compatible(out_graph_error) ||
      !graph_error_reserved_is_zero(out_graph_error)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "out_graph_error has an incompatible struct_size or nonzero reserved fields");
  }
  clear_graph_error(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE);
  if (*exec == nullptr) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE, 0, 0,
                         false, false, true, false);
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaGraphExec* const owner = *exec;
  const uint64_t capture_id = owner->capture_id;
  const uint64_t exec_id = owner->exec_id;
  const bool is_fill =
      owner->operation == RileyCudaGraphCaptureOperation::kFillF32;
  const bool is_h2d = owner->operation == RileyCudaGraphCaptureOperation::kH2D;
  const bool is_silu_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kSiluBf16;
  const bool is_gated_multiply_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kGatedMultiplyBf16;
  const bool is_residual_add_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kResidualAddBf16;
  const bool is_canonical_rms_norm_bf16 =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kCanonicalRmsNormBf16;
  const bool is_bf16_argmax =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16Argmax;
  const bool is_bf16_row_gather =
      owner->operation == RileyCudaGraphCaptureOperation::kBf16RowGather;
  const bool is_bf16_row_gather_argmax =
      owner->operation ==
      RileyCudaGraphCaptureOperation::kBf16RowGatherArgmax;
  const bool is_bf16_row_gather_argmax_d2h =
      is_bf16_d2h_operation(owner->operation);
  const bool is_indexed_rope_bf16 =
      owner->operation == RileyCudaGraphCaptureOperation::kIndexedRopeBf16;
  const bool is_canonical_gemm_bf16 =
      is_canonical_gemm_bf16_operation(owner->operation);
  const bool is_ragged_paged_kv_write_bf16 =
      is_ragged_paged_fixed_address_operation(owner->operation);
  if (is_residual_add_bf16 && !residual_add_exec_state_is_valid(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "residual-add graph exec has invalid fixed resource state");
  }
  if (!is_residual_add_bf16 && !residual_add_exec_fields_are_clear(owner)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "graph exec mixes residual-add state with another operation");
  }
  if (is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "canonical RMSNorm graph exec has invalid fixed resource state");
  }
  if (!is_canonical_rms_norm_bf16 &&
      !canonical_rms_norm_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes canonical RMSNorm state with another operation");
  }
  if (is_bf16_argmax && !bf16_argmax_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "deterministic BF16 argmax graph exec has invalid fixed resource state");
  }
  if (!is_bf16_argmax && !bf16_argmax_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes deterministic BF16 argmax state with another operation");
  }
  if (is_bf16_row_gather && !bf16_row_gather_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "BF16 row-gather graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather && !bf16_row_gather_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes BF16 row-gather state with another operation");
  }
  if (is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "BF16 row-gather argmax graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax &&
      !bf16_row_gather_argmax_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes BF16 row-gather argmax state with another operation");
  }
  if (is_bf16_row_gather_argmax_d2h &&
      !bf16_d2h_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "BF16 row-gather argmax D2H graph exec has invalid fixed resource state");
  }
  if (!is_bf16_row_gather_argmax_d2h &&
      !bf16_row_gather_argmax_d2h_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes BF16 row-gather argmax D2H state with another operation");
  }
  if (is_indexed_rope_bf16 && !indexed_rope_bf16_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "BF16 indexed-RoPE graph exec has invalid fixed resource state");
  }
  if (!is_indexed_rope_bf16 &&
      !indexed_rope_bf16_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
                            "graph exec mixes BF16 indexed-RoPE state with another operation");
  }
  if (is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "canonical cuBLASLt GEMM graph exec has invalid fixed resource state");
  }
  if (!is_canonical_gemm_bf16 &&
      !canonical_gemm_bf16_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes canonical cuBLASLt GEMM state with another operation");
  }
  if (is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_exec_state_is_valid(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "BF16 ragged paged-KV graph exec has invalid fixed resource state");
  }
  if (!is_ragged_paged_kv_write_bf16 &&
      !ragged_paged_kv_cache_write_bf16_exec_fields_are_clear(owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCloseExecOperation,
        "graph exec mixes BF16 ragged paged-KV state with another operation");
  }
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, false);
  if (owner->owner == nullptr || owner->stream == nullptr ||
      owner->fill_buffer == nullptr || owner->graph == nullptr ||
      owner->exec == nullptr || !owner->owns_capture_leases ||
      (!is_fill && !is_h2d && !is_silu_bf16 && !is_gated_multiply_bf16 &&
       !is_residual_add_bf16 && !is_canonical_rms_norm_bf16 &&
       !is_bf16_argmax && !is_bf16_row_gather &&
       !is_bf16_row_gather_argmax && !is_bf16_row_gather_argmax_d2h &&
       !is_indexed_rope_bf16 && !is_canonical_gemm_bf16 &&
       !is_ragged_paged_kv_write_bf16) ||
      !same_context(owner->owner, owner->stream->owner) ||
      !same_context(owner->owner, owner->fill_buffer->owner) ||
      owner->stream->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->fill_buffer->active_uses.load(std::memory_order_acquire) != 1 ||
      owner->launch_in_flight || owner->poisoned ||
      owner->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "graph exec is busy, poisoned, or lost its retained resource lease");
  }
  if ((is_fill &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_input_staged || owner->silu_input != nullptr ||
        owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_h2d &&
       (owner->h2d_byte_len == 0 ||
        owner->h2d_source == nullptr ||
        !same_context(owner->owner, owner->h2d_source->owner) ||
        owner->h2d_source->host_data == nullptr ||
        owner->h2d_source->byte_len != owner->h2d_byte_len ||
        owner->fill_buffer->byte_len != owner->h2d_byte_len ||
        owner->h2d_source->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->silu_input != nullptr || owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_silu_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_input_staged || owner->silu_input == nullptr ||
        owner->silu_input == owner->fill_buffer ||
        owner->silu_element_count == 0 ||
        !same_context(owner->owner, owner->silu_input->owner) ||
        owner->silu_input->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->silu_element_count >
            owner->silu_input->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->silu_input->active_uses.load(std::memory_order_acquire) != 1 ||
        owner->gated_multiply_activated_gate != nullptr ||
        owner->gated_multiply_up != nullptr ||
        owner->gated_multiply_element_count != 0)) ||
      (is_gated_multiply_bf16 &&
       (owner->h2d_source != nullptr || owner->h2d_byte_len != 0 ||
        owner->h2d_input_staged || owner->silu_input != nullptr ||
        owner->silu_element_count != 0 ||
        owner->gated_multiply_activated_gate == nullptr ||
        owner->gated_multiply_up == nullptr ||
        owner->gated_multiply_activated_gate == owner->gated_multiply_up ||
        owner->gated_multiply_activated_gate == owner->fill_buffer ||
        owner->gated_multiply_up == owner->fill_buffer ||
        owner->gated_multiply_element_count == 0 ||
        !same_context(owner->owner,
                      owner->gated_multiply_activated_gate->owner) ||
        !same_context(owner->owner, owner->gated_multiply_up->owner) ||
        owner->gated_multiply_activated_gate->device_data == nullptr ||
        owner->gated_multiply_up->device_data == nullptr ||
        owner->fill_buffer->device_data == nullptr ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_activated_gate->byte_len /
                sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->gated_multiply_up->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_element_count >
            owner->fill_buffer->byte_len / sizeof(__nv_bfloat16) ||
        owner->gated_multiply_activated_gate->active_uses.load(
            std::memory_order_acquire) != 1 ||
        owner->gated_multiply_up->active_uses.load(
            std::memory_order_acquire) != 1))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseExecOperation,
                            "graph exec has invalid fixed-operation resource state");
  }

  CurrentContext scope(owner->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kCloseExecOperation);
  bool exec_destroy_attempted = false;
  bool graph_destroy_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    exec_destroy_attempted = true;
    status = runtime_error(cudaGraphExecDestroy(owner->exec), error,
                           RILEY_CUDA_ERROR_STAGE_CLOSE, kCloseExecOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      graph_destroy_attempted = true;
      status = runtime_error(cudaGraphDestroy(owner->graph), error,
                             RILEY_CUDA_ERROR_STAGE_CLOSE,
                             kCloseExecOperation);
    }
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kCloseExecOperation);
  const bool restoration_known =
      !owner->owner->restoration_failed.load(std::memory_order_acquire);
  if (!exec_destroy_attempted) {
    record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                         capture_id, exec_id, false, false, false,
                         !restoration_known);
    return status;
  }

  // Both native destroy operations are one-shot. A failure after the first
  // call retains every graph lease permanently; retrying could double-destroy
  // either opaque CUDA object.
  *exec = nullptr;
  if (status == RILEY_CUDA_STATUS_SUCCESS && graph_destroy_attempted &&
      restoration_known) {
    owner->exec = nullptr;
    owner->graph = nullptr;
    const bool released =
        is_h2d ? release_graph_h2d_leases(owner->owner, owner->stream,
                                           owner->fill_buffer,
                                           owner->h2d_source)
               : is_silu_bf16
                     ? release_graph_silu_bf16_leases(owner->owner,
                                                       owner->stream,
                                                       owner->silu_input,
                                                       owner->fill_buffer)
                     : is_gated_multiply_bf16
                           ? release_graph_gated_multiply_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->gated_multiply_activated_gate,
                                 owner->gated_multiply_up, owner->fill_buffer)
                     : is_residual_add_bf16
                           ? release_graph_residual_add_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->residual_add_left,
                                 owner->residual_add_right, owner->fill_buffer)
                     : is_canonical_rms_norm_bf16
                           ? release_graph_canonical_rms_norm_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->canonical_rms_norm_input,
                                 owner->canonical_rms_norm_weight,
                                 owner->fill_buffer)
                     : is_bf16_argmax
                           ? release_graph_bf16_argmax_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_argmax_logits,
                                 owner->fill_buffer)
                     : is_bf16_row_gather
                           ? release_graph_bf16_row_gather_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_input,
                                 owner->bf16_row_gather_indices,
                                 owner->fill_buffer)
                     : is_bf16_row_gather_argmax
                           ? release_graph_bf16_row_gather_argmax_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_argmax_input,
                                 owner->bf16_row_gather_argmax_indices,
                                 owner->bf16_row_gather_argmax_gathered_logits,
                                 owner->fill_buffer)
                     : is_bf16_row_gather_argmax_d2h
                           ? release_graph_bf16_row_gather_argmax_d2h_leases(
                                 owner->owner, owner->stream,
                                 owner->bf16_row_gather_argmax_d2h_input,
                                 owner->bf16_row_gather_argmax_d2h_indices,
                                 owner->bf16_row_gather_argmax_d2h_gathered_logits,
                                 owner->fill_buffer,
                                 owner->bf16_row_gather_argmax_d2h_pinned_results)
                     : is_indexed_rope_bf16
                           ? release_graph_indexed_rope_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->indexed_rope_bf16_input,
                                 owner->indexed_rope_bf16_cos,
                                 owner->indexed_rope_bf16_sin,
                                 owner->indexed_rope_bf16_positions,
                                 owner->fill_buffer)
                     : is_canonical_gemm_bf16
                           ? release_graph_canonical_gemm_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->fill_buffer,
                                 owner->canonical_gemm_bf16)
                     : is_ragged_paged_kv_write_bf16
                           ? release_graph_ragged_paged_kv_cache_write_bf16_leases(
                                 owner->owner, owner->stream,
                                 owner->fill_buffer,
                                 owner->ragged_paged_kv_write_bf16)
                     : release_graph_leases(owner->owner, owner->stream,
                                            owner->fill_buffer);
    if (released) {
      owner->owns_capture_leases = false;
      owner->~RileyCudaGraphExec();
      std::free(owner);
      record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                           capture_id, exec_id, false, false, true, false);
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    status = internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseExecOperation,
                            "failed to release graph exec stream, retained buffers, or context lease");
  }
  owner->poisoned = true;
  record_graph_outcome(out_graph_error, RILEY_CUDA_GRAPH_STAGE_CLOSE,
                       capture_id, exec_id, false, false, false, true);
  return status;
}
