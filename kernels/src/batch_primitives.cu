#include "ffi_internal.hpp"
#include "fixed37_reduction.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::command_batch_is_owned_by_current_thread;
using riley_cuda_internal::command_batch_register_use;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::same_context;
using riley_cuda_internal::try_acquire_exclusive_use;
using riley_cuda_internal::validation_error;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaximumBlocks = 65535;
// D128 two-stage ragged decode owns seven tensor spans plus the five packed
// metadata spans. Keep room for the additive ABI without constraining legacy
// D64 paths, which still use at most nine distinct buffers.
constexpr size_t kMaximumBatchBuffers = 16;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
// Grouped GQA execution keeps the query-head warp's reduction and online
// token order intact, while allowing the query heads that share one KV head to
// reuse the same BF16 K/V tile from shared memory. This cap also preserves a
// bounded generic fallback for shapes that cannot use the shared-KV launch.
constexpr uint32_t kRaggedAttentionMaximumWarpsPerBlock = 16;
constexpr uint32_t kRaggedAttentionThreads =
    kWarpSize * kRaggedAttentionMaximumWarpsPerBlock;
constexpr uint64_t kAttentionHeadSize = 64;
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount = 16;
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount = 2;
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128HeadSize = 128;
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128StateStride =
    kNativeBf16RaggedPagedSplitGqaD128HeadSize + 2;
constexpr uint64_t kNativeBf16RaggedPagedSplitGqaD128TransitionWidth = 2;
constexpr uint32_t kNativeBf16RaggedPagedSplitGqaD128ReductionThreads = 128;
constexpr uint32_t kNativeBf16RaggedD128ReplayLookahead = 4;
constexpr uint32_t kRaggedAttentionSharedTileTokens =
    RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
constexpr uint64_t kFixed37RaggedMaximumTokenCount = 8192;
constexpr uint64_t kFixed37RaggedDepthPartialCount =
    riley_cuda_fixed37::chunk_count(kAttentionHeadSize);
constexpr uint64_t kFixed37RaggedMaximumTokenPartialCount =
    riley_cuda_fixed37::chunk_count(
        kFixed37RaggedMaximumTokenCount);
constexpr uint64_t kFixed37RaggedMaximumSharedBytes =
    (kFixed37RaggedMaximumTokenCount +
     2 * kFixed37RaggedMaximumTokenPartialCount) *
    sizeof(float);
constexpr uint64_t kMaximumGridX = 2147483647;
constexpr uint64_t kMaximumGridYOrZ = 65535;

static_assert(sizeof(RileyCudaIndexedRopeParams) == 320,
              "indexed RoPE ABI size changed");
static_assert(offsetof(RileyCudaIndexedRopeParams, input) == 8,
              "indexed RoPE input offset changed");
static_assert(offsetof(RileyCudaIndexedRopeParams, active_row_count) == 248,
              "indexed RoPE dimension offset changed");
static_assert(offsetof(RileyCudaIndexedRopeParams, reserved) == 288,
              "indexed RoPE reserved tail changed");
static_assert(sizeof(RileyCudaRowGatherParams) == 208,
              "row gather ABI size changed");
static_assert(offsetof(RileyCudaRowGatherParams, input_row_count) == 152,
              "row gather dimension offset changed");
static_assert(offsetof(RileyCudaRowGatherParams, reserved) == 176,
              "row gather reserved tail changed");
static_assert(sizeof(RileyCudaBf16ArgmaxResult) == 8,
              "BF16 argmax result ABI size changed");
static_assert(offsetof(RileyCudaBf16ArgmaxResult, status) == 4,
              "BF16 argmax result status offset changed");
static_assert(sizeof(RileyCudaBf16ArgmaxParams) == 152,
              "BF16 argmax params ABI size changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, logits) == 8,
              "BF16 argmax logits offset changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, results) == 56,
              "BF16 argmax results offset changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, row_count) == 104,
              "BF16 argmax dimension offset changed");
static_assert(offsetof(RileyCudaBf16ArgmaxParams, reserved) == 120,
              "BF16 argmax reserved tail changed");
static_assert(sizeof(RileyCudaPackedBatchV1) == 320,
              "packed batch ABI size changed");
static_assert(offsetof(RileyCudaPackedBatchV1,
                       sequence_block_offsets) == 8,
              "packed batch offsets span changed");
static_assert(offsetof(RileyCudaPackedBatchV1, sequence_count) == 248,
              "packed batch dimension offset changed");
static_assert(offsetof(RileyCudaPackedBatchV1, reserved) == 288,
              "packed batch reserved tail changed");
static_assert(sizeof(RileyCudaRaggedPagedKvCacheWriteParams) == 568,
              "ragged paged KV write ABI size changed");
static_assert(offsetof(RileyCudaRaggedPagedKvCacheWriteParams, batch) ==
                  200,
              "ragged paged KV write batch offset changed");
static_assert(offsetof(RileyCudaRaggedPagedKvCacheWriteParams,
                       key_value_head_count) == 520,
              "ragged paged KV write dimension offset changed");
static_assert(sizeof(RileyCudaRaggedPagedAttentionParams) == 592,
              "ragged paged attention ABI size changed");
static_assert(offsetof(RileyCudaRaggedPagedAttentionParams, batch) == 200,
              "ragged paged attention batch offset changed");
static_assert(offsetof(RileyCudaRaggedPagedAttentionParams,
                       output_row_count) == 544,
              "ragged paged attention output-row offset changed");
static_assert(offsetof(RileyCudaRaggedPagedAttentionParams, scale) == 552,
              "ragged paged attention scale offset changed");
static_assert(RILEY_CUDA_NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_V2_VERSION == 2,
              "native BF16 ragged split-GQA V2 ABI version changed");
static_assert(sizeof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2) == 744,
              "native BF16 ragged split-GQA V2 ABI size changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             format_version) == 4,
    "native BF16 ragged split-GQA V2 format-version offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2, partial_states) ==
        152,
    "native BF16 ragged split-GQA V2 partial-state offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             reduction_steps) == 200,
    "native BF16 ragged split-GQA V2 reduction-step offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             reduction_normalizers) == 248,
    "native BF16 ragged split-GQA V2 reduction-normalizer offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2, output) == 296,
    "native BF16 ragged split-GQA V2 output offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2, batch) == 344,
    "native BF16 ragged split-GQA V2 packed-batch offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             query_head_count) == 664,
    "native BF16 ragged split-GQA V2 dimensions offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             partial_state_capacity) == 696,
    "native BF16 ragged split-GQA V2 capacity offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
             output_row_count) == 688,
    "native BF16 ragged split-GQA V2 output-row offset changed");
static_assert(offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2,
                       scale) == 704,
              "native BF16 ragged split-GQA V2 scale offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2, reserved) == 712,
    "native BF16 ragged split-GQA V2 reserved tail changed");
static_assert(RILEY_CUDA_NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_V3_VERSION == 3,
              "native BF16 ragged split-GQA V3 ABI version changed");
static_assert(sizeof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3) == 752,
              "native BF16 ragged split-GQA V3 ABI size changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             format_version) == 4,
    "native BF16 ragged split-GQA V3 format-version offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3, partial_states) ==
        152,
    "native BF16 ragged split-GQA V3 partial-state offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             reduction_steps) == 200,
    "native BF16 ragged split-GQA V3 reduction-step offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             reduction_normalizers) == 248,
    "native BF16 ragged split-GQA V3 reduction-normalizer offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3, output) == 296,
    "native BF16 ragged split-GQA V3 output offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3, batch) == 344,
    "native BF16 ragged split-GQA V3 packed-batch offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             query_head_count) == 664,
    "native BF16 ragged split-GQA V3 dimensions offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             partial_state_capacity) == 696,
    "native BF16 ragged split-GQA V3 capacity offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             output_row_count) == 688,
    "native BF16 ragged split-GQA V3 output-row offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
             launch_partial_state_count) == 704,
    "native BF16 ragged split-GQA V3 launch-count offset changed");
static_assert(offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3,
                       scale) == 712,
              "native BF16 ragged split-GQA V3 scale offset changed");
static_assert(
    offsetof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3, reserved) == 720,
    "native BF16 ragged split-GQA V3 reserved tail changed");
static_assert(sizeof(RileyCudaFixed37RaggedPagedAttentionParams) == 600,
              "fixed37 ragged paged attention ABI size changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, struct_size) == 0,
    "fixed37 ragged paged attention struct-size offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved0) == 4,
    "fixed37 ragged paged attention reserved0 offset changed");
static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams, query) ==
                  8,
              "fixed37 ragged paged attention query offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, key_pool) == 56,
    "fixed37 ragged paged attention key-pool offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, value_pool) == 104,
    "fixed37 ragged paged attention value-pool offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, output) == 152,
    "fixed37 ragged paged attention output offset changed");
static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams, batch) ==
                  200,
              "fixed37 ragged paged attention batch offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
             query_head_count) == 520,
    "fixed37 ragged paged attention QH offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
             key_value_head_count) == 528,
    "fixed37 ragged paged attention KVH offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, head_size) == 536,
    "fixed37 ragged paged attention head-size offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
             output_row_count) == 544,
    "fixed37 ragged paged attention output-row offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
             maximum_logical_token_count) == 552,
    "fixed37 ragged paged attention maximum-T offset changed");
static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams, scale) ==
                  560,
              "fixed37 ragged paged attention scale offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved1) == 564,
    "fixed37 ragged paged attention reserved1 offset changed");
static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved) == 568,
    "fixed37 ragged paged attention reserved tail changed");
static_assert(kAttentionHeadSize == 2 * kWarpSize,
              "ragged attention lane ownership changed");
static_assert(kThreads % kWarpSize == 0,
              "BF16 argmax requires a whole number of warps");
static_assert(kFixed37RaggedDepthPartialCount == 2,
              "fixed37 ragged D64 partial shape changed");
static_assert(kFixed37RaggedMaximumTokenPartialCount == 222,
              "fixed37 ragged T8192 partial shape changed");
static_assert(kFixed37RaggedMaximumSharedBytes == 34544,
              "fixed37 ragged maximum shared-memory shape changed");
static_assert(kFixed37RaggedMaximumSharedBytes <= 48 * 1024,
              "fixed37 ragged shared memory exceeds the base CUDA limit");

struct ResolvedSpan {
  RileyCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
  RileyCudaDType dtype;
};

struct ResolvedBatch {
  ResolvedSpan sequence_block_offsets;
  ResolvedSpan block_ids;
  ResolvedSpan valid_tokens;
  ResolvedSpan row_sequence_slots;
  ResolvedSpan row_positions;
};

struct DeviceBatch {
  const uint32_t* sequence_block_offsets;
  const uint32_t* block_ids;
  const uint16_t* valid_tokens;
  const uint32_t* row_sequence_slots;
  const uint32_t* row_positions;
  uint64_t sequence_count;
  uint64_t block_count;
  uint64_t active_row_count;
  uint64_t physical_block_count;
  // Non-null in a dynamic step capture: `[0]` is the step's active row count.
  const uint32_t* dynamic_counts;
};

__device__ __forceinline__ uint64_t active_rows(const DeviceBatch& batch) {
  return batch.dynamic_counts != nullptr
             ? static_cast<uint64_t>(batch.dynamic_counts[0])
             : batch.active_row_count;
}

bool checked_add(uint64_t left, uint64_t right, uint64_t* output) noexcept {
  if (output == nullptr ||
      right > std::numeric_limits<uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool checked_multiply(uint64_t left, uint64_t right,
                      uint64_t* output) noexcept {
  if (output == nullptr ||
      (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *output = left * right;
  return true;
}

bool checked_product3(uint64_t first, uint64_t second, uint64_t third,
                      uint64_t* output) noexcept {
  uint64_t partial = 0;
  return checked_multiply(first, second, &partial) &&
         checked_multiply(partial, third, output);
}

bool checked_product4(uint64_t first, uint64_t second, uint64_t third,
                      uint64_t fourth, uint64_t* output) noexcept {
  uint64_t partial = 0;
  return checked_product3(first, second, third, &partial) &&
         checked_multiply(partial, fourth, output);
}

RileyCudaStatus fixed37_ragged_shared_bytes(
    uint64_t maximum_logical_token_count, uint64_t* token_partial_count,
    uint64_t* shared_bytes, RileyCudaErrorInfo* error,
    const char* operation) noexcept {
  if (token_partial_count == nullptr || shared_bytes == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "internal fixed37 ragged shared-memory output is null");
  }
  const uint64_t chunks =
      riley_cuda_fixed37::chunk_count(maximum_logical_token_count);
  if (chunks == 0 ||
      chunks > riley_cuda_fixed37::kMaximumChunkCount) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "fixed37 ragged maximum T exceeds the chunk-partial capacity");
  }
  const uint64_t partial_capacity =
      chunks < kFixed37RaggedDepthPartialCount
          ? kFixed37RaggedDepthPartialCount
          : chunks;
  uint64_t value_bytes = 0;
  uint64_t reduction_bytes = 0;
  if (!checked_multiply(maximum_logical_token_count, sizeof(float),
                        &value_bytes) ||
      !checked_multiply(partial_capacity, 2 * sizeof(float),
                        &reduction_bytes) ||
      !checked_add(value_bytes, reduction_bytes, shared_bytes)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "fixed37 ragged shared-memory size overflows uint64_t");
  }
  *token_partial_count = chunks;
  return RILEY_CUDA_STATUS_SUCCESS;
}

bool reserved_is_zero(const uint64_t* reserved, size_t count) noexcept {
  if (reserved == nullptr) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    if (reserved[index] != 0) {
      return false;
    }
  }
  return true;
}

uint64_t dtype_size(RileyCudaDType dtype) noexcept {
  switch (dtype) {
    case RILEY_CUDA_DTYPE_F32:
    case RILEY_CUDA_DTYPE_U32:
      return 4;
    case RILEY_CUDA_DTYPE_BF16:
    case RILEY_CUDA_DTYPE_U16:
      return 2;
    case RILEY_CUDA_DTYPE_U8:
      return 1;
    default:
      return 0;
  }
}

bool arithmetic_dtype(RileyCudaDType dtype) noexcept {
  return dtype == RILEY_CUDA_DTYPE_F32 ||
         dtype == RILEY_CUDA_DTYPE_BF16;
}

RileyCudaStatus typed_bytes(uint64_t element_count,
                                RileyCudaDType dtype, uint64_t* output,
                                RileyCudaErrorInfo* error,
                                const char* operation) noexcept {
  const uint64_t width = dtype_size(dtype);
  if (width == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "batch span has an unsupported dtype");
  }
  if (!checked_multiply(element_count, width, output)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "batch byte length overflows uint64_t");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus resolve_span(const RileyCudaBufferSpan& span,
                                 RileyCudaDType expected_dtype,
                                 uint64_t required_bytes,
                                 ResolvedSpan* output,
                                 RileyCudaErrorInfo* error,
                                 const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal resolved span is null");
  }
  if (span.struct_size < sizeof(span)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span has an incompatible struct_size");
  }
  if (!reserved_is_zero(span.reserved, 2)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span reserved fields must be zero");
  }
  if (span.dtype != expected_dtype) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span dtype does not match the batch contract");
  }
  const uint64_t alignment = dtype_size(span.dtype);
  if (alignment == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span dtype is invalid");
  }
  if (span.buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span handle is null");
  }
  if (span.byte_offset % alignment != 0 || span.byte_len % alignment != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span offset or length is not dtype-aligned");
  }
  if (span.byte_offset > span.buffer->byte_len ||
      span.byte_len > span.buffer->byte_len - span.byte_offset) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "declared span exceeds the opaque allocation");
  }
  if (required_bytes > span.byte_len) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "required bytes exceed the declared span capacity");
  }
  if (required_bytes != 0 && span.buffer->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "non-empty span refers to a zero-byte allocation");
  }

  uint8_t* data = nullptr;
  if (span.buffer->device_data != nullptr) {
    data = static_cast<uint8_t*>(span.buffer->device_data) +
           static_cast<size_t>(span.byte_offset);
  }
  *output = ResolvedSpan{span.buffer, data, span.byte_offset, required_bytes,
                         span.dtype};
  return RILEY_CUDA_STATUS_SUCCESS;
}

bool overlaps(const ResolvedSpan& left, const ResolvedSpan& right) noexcept {
  if (left.buffer != right.buffer || left.used_bytes == 0 ||
      right.used_bytes == 0) {
    return false;
  }
  const uint64_t left_end = left.byte_offset + left.used_bytes;
  const uint64_t right_end = right.byte_offset + right.used_bytes;
  return left.byte_offset < right_end && right.byte_offset < left_end;
}

bool exact_alias(const ResolvedSpan& left, const ResolvedSpan& right) noexcept {
  return left.buffer == right.buffer &&
         left.byte_offset == right.byte_offset &&
         left.used_bytes == right.used_bytes;
}

RileyCudaStatus reject_overlap(const ResolvedSpan& writable,
                                   const ResolvedSpan& other,
                                   bool exact_alias_allowed,
                                   RileyCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(writable, other) &&
      !(exact_alias_allowed && exact_alias(writable, other))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "unsupported writable/touched span overlap");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus validate_contexts(RileyCudaStream* stream,
                                      const ResolvedSpan* spans, size_t count,
                                      RileyCudaErrorInfo* error,
                                      const char* operation) noexcept {
  if (stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "stream is null");
  }
  if (stream->owner == nullptr ||
      stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "CUDA context owner is missing or poisoned by a prior restoration failure");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!same_context(stream->owner, spans[index].buffer->owner)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
          "stream and batch spans belong to different context owners");
    }
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

class ExclusiveUses final {
 public:
  explicit ExclusiveUses(RileyCudaStream* stream) noexcept
      : stream_(stream),
        buffers_{},
        buffer_count_(0),
        acquired_count_(0),
        stream_acquired_(false),
        command_batch_(false) {}

  ExclusiveUses(const ExclusiveUses&) = delete;
  ExclusiveUses& operator=(const ExclusiveUses&) = delete;

  bool add(RileyCudaDeviceBuffer* buffer) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (buffers_[index] == buffer) {
        return true;
      }
    }
    if (buffer == nullptr || buffer_count_ == kMaximumBatchBuffers) {
      return false;
    }
    buffers_[buffer_count_++] = buffer;
    return true;
  }

  RileyCudaStatus acquire(RileyCudaErrorInfo* error,
                              const char* operation) noexcept {
    if (command_batch_is_active(stream_)) {
      if (!command_batch_is_owned_by_current_thread(stream_)) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_STATE,
            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
            "an active stream command batch is owned by another thread");
      }
      command_batch_ = true;
      for (size_t index = 0; index < buffer_count_; ++index) {
        const RileyCudaStatus status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error, operation,
            "a batch buffer already has an active asynchronous use");
        if (status != RILEY_CUDA_STATUS_SUCCESS) {
          return status;
        }
      }
      return RILEY_CUDA_STATUS_SUCCESS;
    }
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        release_acquired();
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_STATE,
            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
            "a batch buffer already has an active asynchronous use");
      }
      ++acquired_count_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      release_acquired();
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
          "the stream already has an active asynchronous use");
    }
    stream_acquired_ = true;
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  bool release_completed() noexcept {
    if (command_batch_) {
      return true;
    }
    bool valid = true;
    if (stream_acquired_) {
      valid = release_exclusive_use(stream_->active_uses) && valid;
      stream_acquired_ = false;
    }
    while (acquired_count_ != 0) {
      --acquired_count_;
      valid = release_exclusive_use(buffers_[acquired_count_]->active_uses) &&
              valid;
    }
    return valid;
  }

  bool command_batch() const noexcept { return command_batch_; }

 private:
  void release_acquired() noexcept {
    while (acquired_count_ != 0) {
      --acquired_count_;
      (void)release_exclusive_use(buffers_[acquired_count_]->active_uses);
    }
  }

  RileyCudaStream* stream_;
  RileyCudaDeviceBuffer* buffers_[kMaximumBatchBuffers];
  size_t buffer_count_;
  size_t acquired_count_;
  bool stream_acquired_;
  bool command_batch_;
};

uint32_t block_count(uint64_t work_items) noexcept {
  if (work_items == 0) {
    return 0;
  }
  const uint64_t needed = ((work_items - 1) / kThreads) + 1;
  return static_cast<uint32_t>(
      needed < kMaximumBlocks ? needed : kMaximumBlocks);
}

RileyCudaStatus launch_status(RileyCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RILEY_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RileyCudaStatus prior_launch_status(RileyCudaErrorInfo* error,
                                        const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RILEY_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RileyCudaStatus complete_execution(ExclusiveUses* uses,
                                       CurrentContext* scope,
                                       RileyCudaStream* stream,
                                       RileyCudaStatus operation_status,
                                       bool launch_attempted,
                                       RileyCudaErrorInfo* error,
                                       const char* operation) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  }
  bool completion_confirmed = !launch_attempted;
  RileyCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t synchronize_result =
        cudaStreamSynchronize(stream->stream);
    completion_confirmed = synchronize_result == cudaSuccess;
    if (!completion_confirmed) {
      status = runtime_error(synchronize_result, error,
                             RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                             operation);
    }
  }
  status = scope->leave(status, error,
                        RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (completion_confirmed && restoration_confirmed) {
    if (!uses->release_completed()) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            operation,
                            "exclusive-use accounting was corrupted");
    }
  }
  return status;
}

RileyCudaStatus validate_packed_batch(
    const RileyCudaPackedBatchV1& batch, RileyCudaErrorInfo* error,
    const char* operation) noexcept {
  if (batch.struct_size < sizeof(batch)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch has an incompatible struct_size");
  }
  if (batch.format_version != RILEY_CUDA_PACKED_BATCH_VERSION ||
      batch.block_size != RILEY_CUDA_PAGED_KV_BLOCK_SIZE) {
    return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch version or block size is unsupported");
  }
  if (batch.reserved0 != 0 || !reserved_is_zero(batch.reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch reserved fields must be zero");
  }
  if (batch.sequence_count == 0 || batch.block_count == 0 ||
      batch.active_row_count == 0 || batch.physical_block_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch dimensions must be greater than zero");
  }
  constexpr uint64_t kMaximumU32 =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  if (batch.sequence_count > kMaximumU32 || batch.block_count > kMaximumU32 ||
      batch.active_row_count > kMaximumU32 ||
      batch.physical_block_count > kMaximumU32) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch dimensions exceed U32 metadata range");
  }
  if (batch.block_count > batch.physical_block_count) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch uses more logical blocks than the physical pool");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus resolve_packed_batch(
    const RileyCudaPackedBatchV1& batch, ResolvedBatch* output,
    RileyCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal resolved batch is null");
  }
  uint64_t offset_count = 0;
  if (!checked_add(batch.sequence_count, 1, &offset_count)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed CSR offset count overflows uint64_t");
  }
  uint64_t offset_bytes = 0;
  uint64_t block_id_bytes = 0;
  uint64_t valid_token_bytes = 0;
  uint64_t row_bytes = 0;
  RileyCudaStatus status = typed_bytes(
      offset_count, RILEY_CUDA_DTYPE_U32, &offset_bytes, error, operation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.block_count, RILEY_CUDA_DTYPE_U32,
                         &block_id_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.block_count, RILEY_CUDA_DTYPE_U16,
                         &valid_token_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.active_row_count, RILEY_CUDA_DTYPE_U32,
                         &row_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.sequence_block_offsets,
                          RILEY_CUDA_DTYPE_U32, offset_bytes,
                          &output->sequence_block_offsets, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.block_ids, RILEY_CUDA_DTYPE_U32,
                          block_id_bytes, &output->block_ids, error,
                          operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.valid_tokens, RILEY_CUDA_DTYPE_U16,
                          valid_token_bytes, &output->valid_tokens, error,
                          operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.row_sequence_slots,
                          RILEY_CUDA_DTYPE_U32, row_bytes,
                          &output->row_sequence_slots, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.row_positions, RILEY_CUDA_DTYPE_U32,
                          row_bytes, &output->row_positions, error,
                          operation);
  }
  return status;
}

DeviceBatch device_batch(const RileyCudaPackedBatchV1& batch,
                         const ResolvedBatch& resolved) noexcept {
  return DeviceBatch{
      reinterpret_cast<const uint32_t*>(
          resolved.sequence_block_offsets.data),
      reinterpret_cast<const uint32_t*>(resolved.block_ids.data),
      reinterpret_cast<const uint16_t*>(resolved.valid_tokens.data),
      reinterpret_cast<const uint32_t*>(resolved.row_sequence_slots.data),
      reinterpret_cast<const uint32_t*>(resolved.row_positions.data),
      // Like the block count below, the sequence count only bounds indices the
      // host validates every step; a recorded step must not freeze it.
      riley_cuda_internal::thread_has_active_step_capture()
          ? static_cast<uint64_t>(UINT32_MAX)
          : batch.sequence_count,
      // The block count is only an upper bound for the CSR offsets read on the
      // device. A recorded step must not freeze this step's value, so it uses
      // the widest bound; the host validates the offsets of every step.
      riley_cuda_internal::thread_has_active_step_capture()
          ? static_cast<uint64_t>(UINT32_MAX)
          : batch.block_count,
      batch.active_row_count,
      batch.physical_block_count,
      riley_cuda_internal::thread_step_capture_dynamic().counts,
  };
}

template <typename T>
__device__ float load_f32(const T* values, uint64_t index);

template <>
__device__ float load_f32<float>(const float* values, uint64_t index) {
  return values[index];
}

template <>
__device__ float load_f32<__nv_bfloat16>(const __nv_bfloat16* values,
                                         uint64_t index) {
  return __bfloat162float(values[index]);
}

template <typename T>
__device__ void store_f32(T* values, uint64_t index, float value);

template <>
__device__ void store_f32<float>(float* values, uint64_t index, float value) {
  values[index] = value;
}

template <>
__device__ void store_f32<__nv_bfloat16>(__nv_bfloat16* values,
                                         uint64_t index, float value) {
  values[index] = __float2bfloat16_rn(value);
}

template <typename T>
__device__ float round_to_storage(float value) {
  return value;
}

template <>
__device__ float round_to_storage<__nv_bfloat16>(float value) {
  return __bfloat162float(__float2bfloat16_rn(value));
}

template <typename T>
__device__ float multiply_in_storage(float left, float right) {
  return left * right;
}

template <>
__device__ float multiply_in_storage<__nv_bfloat16>(float left,
                                                     float right) {
  return __bfloat162float(__float2bfloat16_rn(left * right));
}

template <typename T>
__global__ void indexed_rope_kernel(
    const T* input, const float* cos, const float* sin,
    const uint32_t* positions, T* output, uint64_t table_position_count,
    uint64_t head_count, uint64_t head_size, uint64_t rotary_dimension,
    uint64_t work_item_count) {
  const uint64_t half = rotary_dimension / 2;
  const uint64_t tail = head_size - rotary_dimension;
  const uint64_t units_per_head = half + tail;
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
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
        store_f32(output, first_index, CUDART_NAN_F);
        store_f32(output, second_index, CUDART_NAN_F);
        continue;
      }
      const uint64_t table_index = position * half + unit;
      const float first_value = load_f32(input, first_index);
      const float second_value = load_f32(input, second_index);
      const float cosine = round_to_storage<T>(cos[table_index]);
      const float sine = round_to_storage<T>(sin[table_index]);
      const float first_cosine = multiply_in_storage<T>(first_value, cosine);
      const float second_sine = multiply_in_storage<T>(second_value, sine);
      const float second_cosine = multiply_in_storage<T>(second_value, cosine);
      const float first_sine = multiply_in_storage<T>(first_value, sine);
      store_f32(output, first_index, first_cosine - second_sine);
      store_f32(output, second_index, second_cosine + first_sine);
    } else {
      const uint64_t dimension = rotary_dimension + (unit - half);
      output[base + dimension] = input[base + dimension];
    }
  }
}

template <typename T>
__global__ void row_gather_kernel(const T* input, const uint32_t* row_indices,
                                  T* output, uint64_t input_row_count,
                                  uint64_t column_count,
                                  uint64_t output_element_count,
                                  const uint32_t* dynamic_counts) {
  // In a dynamic step capture the output row count is read on the device.
  if (dynamic_counts != nullptr) {
    output_element_count =
        static_cast<uint64_t>(dynamic_counts[1]) * column_count;
  }
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < output_element_count; index += stride) {
    const uint64_t output_row = index / column_count;
    const uint64_t column = index - output_row * column_count;
    const uint64_t input_row = row_indices[output_row];
    if (input_row >= input_row_count) {
      store_f32(output, index, CUDART_NAN_F);
    } else {
      output[index] = input[input_row * column_count + column];
    }
  }
}

__device__ __forceinline__ void select_argmax_candidate(
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

__device__ __forceinline__ void reduce_argmax_warp(
    float* selected_value, uint32_t* selected_token,
    uint32_t* non_finite) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    const float candidate_value =
        __shfl_down_sync(kFullWarpMask, *selected_value, offset);
    const uint32_t candidate_token =
        __shfl_down_sync(kFullWarpMask, *selected_token, offset);
    const uint32_t candidate_non_finite =
        __shfl_down_sync(kFullWarpMask, *non_finite, offset);
    if (lane + offset < kWarpSize) {
      *non_finite |= candidate_non_finite;
      select_argmax_candidate(candidate_value, candidate_token,
                              selected_value, selected_token);
    }
  }
}

// One 1024-thread block per row: each thread scans a quarter of the columns a
// 256-thread block would, eight logits per load, and the selection (largest
// value, lowest token on ties) does not depend on how the columns are
// partitioned.
constexpr uint32_t kArgmaxThreads = 1024;

__global__ __launch_bounds__(kArgmaxThreads) void bf16_argmax_kernel(
    const __nv_bfloat16* logits, RileyCudaBf16ArgmaxResult* results,
    uint64_t row_count, uint64_t vocabulary_size, uint64_t candidate_count,
    const uint32_t* dynamic_counts) {
  constexpr uint32_t kWarpCount = kArgmaxThreads / kWarpSize;
  // In a dynamic step capture the output row count is read on the device.
  if (dynamic_counts != nullptr) {
    row_count = dynamic_counts[1];
  }
  __shared__ float warp_values[kWarpCount];
  __shared__ uint32_t warp_tokens[kWarpCount];
  __shared__ uint32_t warp_non_finite[kWarpCount];

  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    float selected_value = -CUDART_INF_F;
    uint32_t selected_token = RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID;
    uint32_t non_finite = 0;
    const uint64_t row_base = row * vocabulary_size;
    const auto consider = [&](uint64_t column, __nv_bfloat16 logit) {
      const float value = __bfloat162float(logit);
      if (!isfinite(value)) {
        non_finite = 1;
        return;
      }
      // Columns past the candidate bound are still checked for finiteness
      // but can never be selected.
      if (column >= candidate_count) {
        return;
      }
      select_argmax_candidate(value, static_cast<uint32_t>(column),
                              &selected_value, &selected_token);
    };
    // The selection does not depend on the order columns are visited in, so
    // the 16-byte-aligned middle of the row is read eight logits per load;
    // the unaligned head and the tail are read one logit at a time.
    const __nv_bfloat16* row_logits = logits + row_base;
    const uint64_t misaligned =
        ((16 - (reinterpret_cast<uintptr_t>(row_logits) & 15)) & 15) / 2;
    const uint64_t head =
        misaligned < vocabulary_size ? misaligned : vocabulary_size;
    const uint64_t group_count = (vocabulary_size - head) / 8;
    const uint64_t tail_begin = head + group_count * 8;
    const uint64_t single_count = head + (vocabulary_size - tail_begin);
    for (uint64_t single = threadIdx.x; single < single_count;
         single += blockDim.x) {
      const uint64_t column =
          single < head ? single : tail_begin + (single - head);
      consider(column, row_logits[column]);
    }
    for (uint64_t group = threadIdx.x; group < group_count;
         group += blockDim.x) {
      const uint64_t first_column = head + group * 8;
      const uint4 packed =
          *reinterpret_cast<const uint4*>(row_logits + first_column);
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (uint32_t element = 0; element < 8; ++element) {
        const uint16_t bits = static_cast<uint16_t>(
            words[element / 2] >> (16 * (element % 2)));
        consider(first_column + element, __ushort_as_bfloat16(bits));
      }
    }

    reduce_argmax_warp(&selected_value, &selected_token, &non_finite);
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
      reduce_argmax_warp(&selected_value, &selected_token, &non_finite);
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

__device__ __forceinline__ bool resolve_row_cache_base(
    const DeviceBatch& batch, uint64_t row, uint64_t logical_position,
    uint64_t key_value_head, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t* output) {
  if (output == nullptr || row >= active_rows(batch)) {
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

__global__ void ragged_paged_kv_cache_write_kernel(
    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool, DeviceBatch batch,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t packed_row = index / head_size;
    const uint64_t key_value_head = packed_row % key_value_head_count;
    const uint64_t row = packed_row / key_value_head_count;
    const uint64_t logical_position = batch.row_positions[row];
    uint64_t cache_base = 0;
    if (resolve_row_cache_base(batch, row, logical_position, key_value_head,
                               key_value_head_count, head_size, &cache_base)) {
      key_pool[cache_base + depth] = key_source[index];
      value_pool[cache_base + depth] = value_source[index];
    }
  }
}

// qkv_split_bias_rope_kernel and ragged_paged_kv_cache_write_kernel in one
// launch. Every thread forms its query, key or value elements exactly as the
// split kernel does and stores them to the same workspace outputs; a key or
// value element of a row that resolves to a cache slot is also stored there,
// which is the copy the separate write kernel would make from the workspace.
__global__ void qkv_split_bias_rope_kv_write_kernel(
    const __nv_bfloat16* merged, const __nv_bfloat16* query_bias,
    const __nv_bfloat16* key_bias, const __nv_bfloat16* value_bias,
    const float* cos, const float* sin, const uint32_t* positions,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool, DeviceBatch batch,
    uint64_t key_value_head_count, uint64_t query_width,
    uint64_t key_value_width, uint64_t head_size,
    uint64_t table_position_count, uint64_t rotated_row_count,
    uint64_t work_item_count, const uint32_t* dynamic_counts) {
  if (dynamic_counts != nullptr) {
    rotated_row_count = dynamic_counts[0];
  }
  const uint64_t half = head_size / 2;
  const uint64_t merged_width = query_width + 2 * key_value_width;
  const uint64_t query_pairs = query_width / 2;
  const uint64_t key_pairs = key_value_width / 2;
  const uint64_t work_width = query_pairs + key_pairs + key_value_width;
  const uint64_t first =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t work = first; work < work_item_count; work += stride) {
    const uint64_t row = work / work_width;
    const uint64_t column = work - row * work_width;
    const uint64_t merged_row = row * merged_width;
    if (column >= query_pairs + key_pairs) {
      const uint64_t local = column - query_pairs - key_pairs;
      const float projected = __bfloat162float(
          merged[merged_row + query_width + key_value_width + local]);
      const __nv_bfloat16 stored = __float2bfloat16_rn(
          __fadd_rn(projected, __bfloat162float(value_bias[local])));
      value[row * key_value_width + local] = stored;
      uint64_t cache_base = 0;
      if (resolve_row_cache_base(batch, row, batch.row_positions[row],
                                 local / head_size, key_value_head_count,
                                 head_size, &cache_base)) {
        value_pool[cache_base + local % head_size] = stored;
      }
      continue;
    }
    if (row >= rotated_row_count) {
      continue;
    }
    const bool is_query = column < query_pairs;
    const uint64_t pair = is_query ? column : column - query_pairs;
    const uint64_t unit = pair % half;
    const uint64_t first_local = (pair / half) * head_size + unit;
    const uint64_t second_local = first_local + half;
    const uint64_t section = is_query ? 0 : query_width;
    const __nv_bfloat16* bias = is_query ? query_bias : key_bias;
    __nv_bfloat16* output = is_query ? query : key;
    const uint64_t output_row = row * (is_query ? query_width : key_value_width);
    const uint64_t position = positions[row];
    __nv_bfloat16 first_output = __float2bfloat16_rn(CUDART_NAN_F);
    __nv_bfloat16 second_output = __float2bfloat16_rn(CUDART_NAN_F);
    if (position < table_position_count) {
      const float first_value = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
          __bfloat162float(merged[merged_row + section + first_local]),
          __bfloat162float(bias[first_local]))));
      const float second_value = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
          __bfloat162float(merged[merged_row + section + second_local]),
          __bfloat162float(bias[second_local]))));
      const uint64_t table_index = position * half + unit;
      const float cosine = round_to_storage<__nv_bfloat16>(cos[table_index]);
      const float sine = round_to_storage<__nv_bfloat16>(sin[table_index]);
      const float first_cosine =
          multiply_in_storage<__nv_bfloat16>(first_value, cosine);
      const float second_sine =
          multiply_in_storage<__nv_bfloat16>(second_value, sine);
      const float second_cosine =
          multiply_in_storage<__nv_bfloat16>(second_value, cosine);
      const float first_sine =
          multiply_in_storage<__nv_bfloat16>(first_value, sine);
      first_output = __float2bfloat16_rn(first_cosine - second_sine);
      second_output = __float2bfloat16_rn(second_cosine + first_sine);
    }
    output[output_row + first_local] = first_output;
    output[output_row + second_local] = second_output;
    if (!is_query) {
      uint64_t cache_base = 0;
      if (resolve_row_cache_base(batch, row, batch.row_positions[row],
                                 pair / half, key_value_head_count, head_size,
                                 &cache_base)) {
        key_pool[cache_base + unit] = first_output;
        key_pool[cache_base + unit + half] = second_output;
      }
    }
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

// Same pairwise tree as warp_sum on lane 0, with the result on every lane.
__device__ __forceinline__ float warp_sum_all_lanes(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    value += __shfl_xor_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float staged_attention_score(float dot_product,
                                                         float scale) {
  // Preserve the established BF16 attention-score contract: the reference,
  // online-prefill, and optimized-decode paths round once after QK and once
  // after scaling before softmax state is updated.
  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(staged_dot) * scale));
}

__device__ __forceinline__ void update_online_state(
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

__device__ __forceinline__ float update_numerator(float numerator,
                                                   float value, float alpha,
                                                   float beta) {
  if (beta == 0.0F) {
    return alpha * numerator;
  }
  if (alpha == 0.0F) {
    return beta * value;
  }
  return fmaf(beta, value, alpha * numerator);
}

// Keep the original one-warp launch and operation ordering available as the
// compatibility path.  The grouped-head kernel below is selected only through
// the additive ABI entry point so that a single binary can benchmark and roll
// back the two launch geometries without changing the numerical contract.
__global__ __launch_bounds__(kWarpSize) void ragged_paged_attention_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    DeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x;
  const uint64_t row = blockIdx.x;
  const uint64_t query_head = blockIdx.y;
  const uint64_t output_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  if (row >= active_rows(batch)) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    output[output_base + lane] = zero;
    output[output_base + lane + kWarpSize] = zero;
    return;
  }
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high =
      __bfloat162float(query[query_base + lane + kWarpSize]);

  const uint64_t logical_position = batch.row_positions[row];
  const uint64_t logical_token_count = logical_position + 1;
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  bool valid = true;
  for (uint64_t logical_token = 0; logical_token < logical_token_count;
       ++logical_token) {
    uint64_t cache_base = 0;
    if (!resolve_row_cache_base(batch, row, logical_token, key_value_head,
                                key_value_head_count, kAttentionHeadSize,
                                &cache_base)) {
      valid = false;
      break;
    }
    const float key_low = __bfloat162float(key_pool[cache_base + lane]);
    const float key_high =
        __bfloat162float(key_pool[cache_base + lane + kWarpSize]);
    float score = fmaf(query_low, key_low, query_high * key_high);
    score = warp_sum(score);
    score = staged_attention_score(
        __shfl_sync(kFullWarpMask, score, 0), scale);

    float alpha = 0.0F;
    float beta = 0.0F;
    if (lane == 0) {
      update_online_state(score, &maximum, &denominator, &alpha, &beta);
    }
    alpha = __shfl_sync(kFullWarpMask, alpha, 0);
    beta = __shfl_sync(kFullWarpMask, beta, 0);
    numerator_low = update_numerator(
        numerator_low, __bfloat162float(value_pool[cache_base + lane]), alpha,
        beta);
    numerator_high = update_numerator(
        numerator_high,
        __bfloat162float(value_pool[cache_base + lane + kWarpSize]), alpha,
        beta);
  }

  // Only lane zero owns the scalar online state. Broadcast its final
  // denominator before every lane normalizes its two value dimensions.
  denominator = __shfl_sync(kFullWarpMask, denominator, 0);
  const float inverse_denominator =
      !valid ? CUDART_NAN_F
             : (isnan(denominator)
                    ? CUDART_NAN_F
                    : (denominator > 0.0F ? 1.0F / denominator : 0.0F));
  const float output_low = numerator_low * inverse_denominator;
  const float output_high = numerator_high * inverse_denominator;
  output[output_base + lane] = __float2bfloat16_rn(output_low);
  output[output_base + lane + kWarpSize] =
      __float2bfloat16_rn(output_high);
}

__global__ __launch_bounds__(kRaggedAttentionThreads)
void ragged_paged_attention_grouped_heads_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    DeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint64_t row = blockIdx.x;
  const uint64_t warps_per_block = blockDim.x / kWarpSize;
  const uint64_t query_head =
      static_cast<uint64_t>(blockIdx.y) * warps_per_block + warp;
  if (query_head >= query_head_count) {
    return;
  }
  const uint64_t output_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  if (row >= active_rows(batch)) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    output[output_base + lane] = zero;
    output[output_base + lane + kWarpSize] = zero;
    return;
  }
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high =
      __bfloat162float(query[query_base + lane + kWarpSize]);

  const uint64_t logical_position = batch.row_positions[row];
  const uint64_t logical_token_count = logical_position + 1;
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  bool valid = true;
  for (uint64_t logical_token = 0; logical_token < logical_token_count;
       ++logical_token) {
    uint64_t cache_base = 0;
    if (!resolve_row_cache_base(batch, row, logical_token, key_value_head,
                                key_value_head_count, kAttentionHeadSize,
                                &cache_base)) {
      valid = false;
      break;
    }
    const float key_low = __bfloat162float(key_pool[cache_base + lane]);
    const float key_high =
        __bfloat162float(key_pool[cache_base + lane + kWarpSize]);
    float score = fmaf(query_low, key_low, query_high * key_high);
    score = warp_sum(score);
    score = staged_attention_score(
        __shfl_sync(kFullWarpMask, score, 0), scale);

    float alpha = 0.0F;
    float beta = 0.0F;
    if (lane == 0) {
      update_online_state(score, &maximum, &denominator, &alpha, &beta);
    }
    alpha = __shfl_sync(kFullWarpMask, alpha, 0);
    beta = __shfl_sync(kFullWarpMask, beta, 0);
    numerator_low = update_numerator(
        numerator_low, __bfloat162float(value_pool[cache_base + lane]), alpha,
        beta);
    numerator_high = update_numerator(
        numerator_high,
        __bfloat162float(value_pool[cache_base + lane + kWarpSize]), alpha,
        beta);
  }

  // Only lane zero owns the scalar online state. Broadcast its final
  // denominator before every lane normalizes its two value dimensions.
  denominator = __shfl_sync(kFullWarpMask, denominator, 0);
  const float inverse_denominator =
      !valid ? CUDART_NAN_F
             : (isnan(denominator)
                    ? CUDART_NAN_F
                    : (denominator > 0.0F ? 1.0F / denominator : 0.0F));
  const float output_low = numerator_low * inverse_denominator;
  const float output_high = numerator_high * inverse_denominator;
  output[output_base + lane] = __float2bfloat16_rn(output_low);
  output[output_base + lane + kWarpSize] =
      __float2bfloat16_rn(output_high);
}

// GQA decode specialization for group sizes in [2, 16]. One CTA owns a
// `(row, key_value_head)` pair and has one warp per associated query head.
// It deliberately preserves the existing per-warp D64 reduction, staged
// score rounding, online-softmax token order, and numerator update order.
// Only immutable BF16 K/V loads and row-to-page address resolution are shared
// across the independent query heads.
__global__ __launch_bounds__(kRaggedAttentionThreads)
void ragged_paged_attention_gqa_shared_kv_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    DeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint64_t row = blockIdx.x;
  const uint64_t warps_per_block = blockDim.x / kWarpSize;
  const uint64_t key_value_head = blockIdx.y;
  const uint64_t query_head = key_value_head * warps_per_block + warp;
  const uint64_t output_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  if (row >= active_rows(batch)) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    output[output_base + lane] = zero;
    output[output_base + lane + kWarpSize] = zero;
    return;
  }

  // The host dispatch below launches exactly QH/KVH warps for this KV head.
  // Do not add a per-warp early return here: the shared-K/V staging path uses
  // CTA barriers, so every launched warp must participate in them.
  const uint64_t query_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high =
      __bfloat162float(query[query_base + lane + kWarpSize]);
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;

  // BF16 shared storage preserves the exact global-storage representation. A
  // tile has the same width as a paged KV block, so no preload crosses a
  // physical-page boundary for the normal aligned tiles.
  __shared__ __nv_bfloat16 shared_keys[kRaggedAttentionSharedTileTokens]
                                      [kAttentionHeadSize];
  __shared__ __nv_bfloat16 shared_values[kRaggedAttentionSharedTileTokens]
                                        [kAttentionHeadSize];

  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  bool valid = true;

  for (uint64_t tile_start = 0; tile_start < logical_token_count;
       tile_start += kRaggedAttentionSharedTileTokens) {
    // Every consumer has completed reads from the prior tile before the CTA
    // overwrites the shared K/V staging area. The first tile has nothing to
    // protect.
    if (tile_start != 0) {
      __syncthreads();
    }
    const uint64_t remaining = logical_token_count - tile_start;
    const uint32_t tile_count = static_cast<uint32_t>(
        remaining < kRaggedAttentionSharedTileTokens
            ? remaining
            : kRaggedAttentionSharedTileTokens);

    // Each query-head warp preloads disjoint token offsets. K/V values are
    // still fetched exactly once per KV head, but the otherwise idle consumer
    // warps now overlap the immutable page-address and global-memory work.
    // The broadcasts remain warp-local, so no value crosses KV-head groups.
    bool tile_valid = true;
    for (uint32_t tile_offset = warp; tile_offset < tile_count;
         tile_offset += static_cast<uint32_t>(warps_per_block)) {
      uint32_t cache_base_low = 0;
      uint32_t cache_base_high = 0;
      int token_valid = 0;
      if (lane == 0) {
        uint64_t cache_base = 0;
        token_valid = resolve_row_cache_base(
                          batch, row, tile_start + tile_offset,
                          key_value_head, key_value_head_count,
                          kAttentionHeadSize, &cache_base)
                          ? 1
                          : 0;
        cache_base_low = static_cast<uint32_t>(cache_base);
        cache_base_high = static_cast<uint32_t>(cache_base >> 32);
      }
      token_valid = __shfl_sync(kFullWarpMask, token_valid, 0);
      cache_base_low = __shfl_sync(kFullWarpMask, cache_base_low, 0);
      cache_base_high = __shfl_sync(kFullWarpMask, cache_base_high, 0);
      if (token_valid == 0) {
        tile_valid = false;
        continue;
      }
      const uint64_t cache_base =
          (static_cast<uint64_t>(cache_base_high) << 32) | cache_base_low;
      shared_keys[tile_offset][lane] = key_pool[cache_base + lane];
      shared_keys[tile_offset][lane + kWarpSize] =
          key_pool[cache_base + lane + kWarpSize];
      shared_values[tile_offset][lane] = value_pool[cache_base + lane];
      shared_values[tile_offset][lane + kWarpSize] =
          value_pool[cache_base + lane + kWarpSize];
    }

    // This barrier publishes all K/V loads and reduces malformed metadata to
    // one CTA-uniform decision. On an invalid logical token the legacy kernel
    // also produces NaN, so no partially loaded tile is consumed here.
    if (__syncthreads_or(tile_valid ? 0 : 1) != 0) {
      valid = false;
      break;
    }
    for (uint32_t tile_offset = 0; tile_offset < tile_count; ++tile_offset) {
      const float key_low =
          __bfloat162float(shared_keys[tile_offset][lane]);
      const float key_high =
          __bfloat162float(shared_keys[tile_offset][lane + kWarpSize]);
      float score = fmaf(query_low, key_low, query_high * key_high);
      score = warp_sum(score);
      score = staged_attention_score(
          __shfl_sync(kFullWarpMask, score, 0), scale);

      float alpha = 0.0F;
      float beta = 0.0F;
      if (lane == 0) {
        update_online_state(score, &maximum, &denominator, &alpha, &beta);
      }
      alpha = __shfl_sync(kFullWarpMask, alpha, 0);
      beta = __shfl_sync(kFullWarpMask, beta, 0);
      numerator_low = update_numerator(
          numerator_low,
          __bfloat162float(shared_values[tile_offset][lane]), alpha, beta);
      numerator_high = update_numerator(
          numerator_high,
          __bfloat162float(shared_values[tile_offset][lane + kWarpSize]),
          alpha, beta);
    }
  }

  denominator = __shfl_sync(kFullWarpMask, denominator, 0);
  const float inverse_denominator =
      !valid ? CUDART_NAN_F
             : (isnan(denominator)
                    ? CUDART_NAN_F
                    : (denominator > 0.0F ? 1.0F / denominator : 0.0F));
  output[output_base + lane] =
      __float2bfloat16_rn(numerator_low * inverse_denominator);
  output[output_base + lane + kWarpSize] =
      __float2bfloat16_rn(numerator_high * inverse_denominator);
}

__global__ __launch_bounds__(riley_cuda_fixed37::kThreadsPerBlock)
void fixed37_ragged_paged_attention_two_pass_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, __nv_bfloat16* output,
    DeviceBatch batch, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t maximum_logical_token_count,
    float scale, uint64_t maximum_token_partial_count) {
  extern __shared__ float shared_values[];
  __shared__ uint32_t has_nan;
  __shared__ uint64_t logical_token_count_shared;
  float* values = shared_values;
  float* first = values + maximum_logical_token_count;
  const uint64_t partial_capacity =
      maximum_token_partial_count < kFixed37RaggedDepthPartialCount
          ? kFixed37RaggedDepthPartialCount
          : maximum_token_partial_count;
  float* second = first + partial_capacity;

  const uint64_t row = blockIdx.x;
  const uint64_t query_head = blockIdx.y;
  const uint64_t output_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;
  if (row >= active_rows(batch)) {
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
    for (uint64_t depth = threadIdx.x; depth < kAttentionHeadSize;
         depth += blockDim.x) {
      output[output_base + depth] = zero;
    }
    return;
  }

  if (threadIdx.x == 0) {
    const uint64_t logical_position = batch.row_positions[row];
    logical_token_count_shared =
        logical_position < maximum_logical_token_count
            ? logical_position + 1
            : 0;
    has_nan = 0;
  }
  __syncthreads();
  const uint64_t logical_token_count = logical_token_count_shared;
  if (logical_token_count == 0) {
    const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
    for (uint64_t depth = threadIdx.x; depth < kAttentionHeadSize;
         depth += blockDim.x) {
      output[output_base + depth] = nan;
    }
    return;
  }

  const uint64_t token_partial_count =
      ((logical_token_count - 1) /
       riley_cuda_fixed37::kChunkElements) +
      1;
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base =
      (row * query_head_count + query_head) * kAttentionHeadSize;

  // Pass one recomputes D64 QK in two logical-depth chunks. Every chunk is an
  // ascending fmaf left fold, followed by the fixed adjacent balanced merge;
  // the score is rounded raw-BF16 then scaled-BF16 before token reduction.
  float chunk_maximum = -CUDART_INF_F;
  for (uint64_t token = 0; token < logical_token_count; ++token) {
    uint64_t key_base = 0;
    const bool valid = resolve_row_cache_base(
        batch, row, token, key_value_head, key_value_head_count,
        kAttentionHeadSize, &key_base);
    for (uint64_t chunk = threadIdx.x;
         chunk < kFixed37RaggedDepthPartialCount; chunk += blockDim.x) {
      const uint64_t begin =
          chunk * riley_cuda_fixed37::kChunkElements;
      uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
      if (end > kAttentionHeadSize) {
        end = kAttentionHeadSize;
      }
      float accumulator = valid ? 0.0F : CUDART_NAN_F;
      if (valid) {
        for (uint64_t depth = begin; depth < end; ++depth) {
          accumulator = fmaf(
              __bfloat162float(query[query_base + depth]),
              __bfloat162float(key_pool[key_base + depth]), accumulator);
        }
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float dot = riley_cuda_fixed37::balanced_sum(
        first, second, kFixed37RaggedDepthPartialCount);
    if (threadIdx.x == 0) {
      const float score = staged_attention_score(dot, scale);
      if (isnan(score)) {
        has_nan = 1;
      }
      if (token % riley_cuda_fixed37::kChunkElements == 0) {
        chunk_maximum = -CUDART_INF_F;
      }
      chunk_maximum = fmaxf(chunk_maximum, score);
      if (token % riley_cuda_fixed37::kChunkElements ==
              riley_cuda_fixed37::kChunkElements - 1 ||
          token + 1 == logical_token_count) {
        values[token / riley_cuda_fixed37::kChunkElements] =
            chunk_maximum;
      }
    }
    __syncthreads();
  }

  for (uint64_t chunk = threadIdx.x; chunk < token_partial_count;
       chunk += blockDim.x) {
    first[chunk] = values[chunk];
  }
  __syncthreads();
  const float maximum = riley_cuda_fixed37::balanced_max(
      first, second, token_partial_count);
  if (has_nan != 0 || !isfinite(maximum)) {
    const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
    for (uint64_t depth = threadIdx.x; depth < kAttentionHeadSize;
         depth += blockDim.x) {
      output[output_base + depth] = nan;
    }
    return;
  }
  __syncthreads();

  // Pass two recomputes QK, materializes only exp(T) in shared memory, and
  // narrows each probability to BF16 before logical-token-zero-anchored AV.
  for (uint64_t token = 0; token < logical_token_count; ++token) {
    uint64_t key_base = 0;
    const bool valid = resolve_row_cache_base(
        batch, row, token, key_value_head, key_value_head_count,
        kAttentionHeadSize, &key_base);
    for (uint64_t chunk = threadIdx.x;
         chunk < kFixed37RaggedDepthPartialCount; chunk += blockDim.x) {
      const uint64_t begin =
          chunk * riley_cuda_fixed37::kChunkElements;
      uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
      if (end > kAttentionHeadSize) {
        end = kAttentionHeadSize;
      }
      float accumulator = valid ? 0.0F : CUDART_NAN_F;
      if (valid) {
        for (uint64_t depth = begin; depth < end; ++depth) {
          accumulator = fmaf(
              __bfloat162float(query[query_base + depth]),
              __bfloat162float(key_pool[key_base + depth]), accumulator);
        }
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float dot = riley_cuda_fixed37::balanced_sum(
        first, second, kFixed37RaggedDepthPartialCount);
    if (threadIdx.x == 0) {
      values[token] = expf(__fsub_rn(staged_attention_score(dot, scale),
                                    maximum));
    }
    __syncthreads();
  }

  for (uint64_t chunk = threadIdx.x; chunk < token_partial_count;
       chunk += blockDim.x) {
    const uint64_t begin =
        chunk * riley_cuda_fixed37::kChunkElements;
    uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
    if (end > logical_token_count) {
      end = logical_token_count;
    }
    float sum = 0.0F;
    for (uint64_t token = begin; token < end; ++token) {
      sum = __fadd_rn(sum, values[token]);
    }
    first[chunk] = sum;
  }
  __syncthreads();
  const float denominator = riley_cuda_fixed37::balanced_sum(
      first, second, token_partial_count);
  for (uint64_t token = threadIdx.x; token < logical_token_count;
       token += blockDim.x) {
    const __nv_bfloat16 probability =
        __float2bfloat16_rn(values[token] / denominator);
    values[token] = __bfloat162float(probability);
  }
  __syncthreads();

  for (uint64_t depth = 0; depth < kAttentionHeadSize; ++depth) {
    for (uint64_t chunk = threadIdx.x; chunk < token_partial_count;
         chunk += blockDim.x) {
      const uint64_t begin =
          chunk * riley_cuda_fixed37::kChunkElements;
      uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
      if (end > logical_token_count) {
        end = logical_token_count;
      }
      float accumulator = 0.0F;
      for (uint64_t token = begin; token < end; ++token) {
        uint64_t value_base = 0;
        if (!resolve_row_cache_base(
                batch, row, token, key_value_head, key_value_head_count,
                kAttentionHeadSize, &value_base)) {
          accumulator = CUDART_NAN_F;
          break;
        }
        // Deliberately unconditional: the fixed37 contract preserves IEEE
        // 0*Inf -> qNaN instead of short-circuiting zero probabilities.
        accumulator = fmaf(
            values[token],
            __bfloat162float(value_pool[value_base + depth]), accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float result = riley_cuda_fixed37::balanced_sum(
        first, second, token_partial_count);
    if (threadIdx.x == 0) {
      output[output_base + depth] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_indexed_rope_execute(
    const RileyCudaIndexedRopeParams* params, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute row-indexed non-interleaved Llama RoPE";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaIndexedRopeParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype ||
      params->cos.dtype != RILEY_CUDA_DTYPE_F32 ||
      params->sin.dtype != RILEY_CUDA_DTYPE_F32 ||
      params->positions.dtype != RILEY_CUDA_DTYPE_U32) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "indexed RoPE requires matching F32/BF16 tensors, F32 tables, and U32 positions");
  }
  if (params->head_count == 0 || params->head_size == 0 ||
      params->rotary_dimension == 0 ||
      params->rotary_dimension > params->head_size ||
      params->rotary_dimension % 2 != 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "head dimensions must be non-zero and rotary_dimension even and <= head_size");
  }
  if (params->active_row_count != 0 &&
      params->table_position_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "a non-empty indexed RoPE batch requires a position table");
  }

  const uint64_t half = params->rotary_dimension / 2;
  const uint64_t tail = params->head_size - params->rotary_dimension;
  uint64_t head_rows = 0;
  uint64_t tensor_elements = 0;
  uint64_t table_elements = 0;
  uint64_t work_units_per_head = 0;
  uint64_t work_items = 0;
  if (!checked_multiply(params->active_row_count, params->head_count,
                        &head_rows) ||
      !checked_multiply(head_rows, params->head_size, &tensor_elements) ||
      !checked_multiply(params->table_position_count, half,
                        &table_elements) ||
      !checked_add(half, tail, &work_units_per_head) ||
      !checked_multiply(head_rows, work_units_per_head, &work_items)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "indexed RoPE shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t table_bytes = 0;
  uint64_t position_bytes = 0;
  RileyCudaStatus status =
      typed_bytes(tensor_elements, params->input.dtype, &tensor_bytes, error,
                  kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table_elements, RILEY_CUDA_DTYPE_F32,
                         &table_bytes, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(params->active_row_count,
                         RILEY_CUDA_DTYPE_U32, &position_bytes, error,
                         kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan input{};
  ResolvedSpan cos{};
  ResolvedSpan sin{};
  ResolvedSpan positions{};
  ResolvedSpan output{};
  status = resolve_span(params->input, params->input.dtype, tensor_bytes,
                        &input, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->cos, RILEY_CUDA_DTYPE_F32, table_bytes,
                          &cos, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->sin, RILEY_CUDA_DTYPE_F32, table_bytes,
                          &sin, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->positions, RILEY_CUDA_DTYPE_U32,
                          position_bytes, &positions, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, params->input.dtype, tensor_bytes,
                          &output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, true, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, cos, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, sin, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, positions, false, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, cos, sin, positions, output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(cos.buffer) ||
      !uses.add(sin.buffer) || !uses.add(positions.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "batch buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->active_row_count == 0) {
    return uses.release_completed()
               ? RILEY_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (params->input.dtype == RILEY_CUDA_DTYPE_F32) {
      indexed_rope_kernel<float>
          <<<block_count(work_items), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const float*>(input.data),
              reinterpret_cast<const float*>(cos.data),
              reinterpret_cast<const float*>(sin.data),
              reinterpret_cast<const uint32_t*>(positions.data),
              reinterpret_cast<float*>(output.data),
              params->table_position_count, params->head_count,
              params->head_size, params->rotary_dimension, work_items);
    } else {
      indexed_rope_kernel<__nv_bfloat16>
          <<<block_count(work_items), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data),
              reinterpret_cast<const float*>(cos.data),
              reinterpret_cast<const float*>(sin.data),
              reinterpret_cast<const uint32_t*>(positions.data),
              reinterpret_cast<__nv_bfloat16*>(output.data),
              params->table_position_count, params->head_count,
              params->head_size, params->rotary_dimension, work_items);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RileyCudaStatus riley_cuda_row_gather_execute(
    const RileyCudaRowGatherParams* params, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute row gather";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaRowGatherParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype ||
      params->row_indices.dtype != RILEY_CUDA_DTYPE_U32) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "row gather requires matching F32/BF16 tensors and U32 indices");
  }
  if (params->input_row_count == 0 || params->column_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "input_row_count and column_count must be greater than zero");
  }
  uint64_t input_elements = 0;
  uint64_t output_elements = 0;
  if (!checked_multiply(params->input_row_count, params->column_count,
                        &input_elements) ||
      !checked_multiply(params->output_row_count, params->column_count,
                        &output_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "row gather matrix shape overflows uint64_t");
  }
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t index_bytes = 0;
  RileyCudaStatus status =
      typed_bytes(input_elements, params->input.dtype, &input_bytes, error,
                  kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, params->input.dtype, &output_bytes,
                         error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(params->output_row_count,
                         RILEY_CUDA_DTYPE_U32, &index_bytes, error,
                         kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan input{};
  ResolvedSpan row_indices{};
  ResolvedSpan output{};
  status = resolve_span(params->input, params->input.dtype, input_bytes,
                        &input, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->row_indices, RILEY_CUDA_DTYPE_U32,
                          index_bytes, &row_indices, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, params->input.dtype, output_bytes,
                          &output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, row_indices, false, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, row_indices, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(row_indices.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "batch buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->output_row_count == 0) {
    return uses.release_completed()
               ? RILEY_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    // A dynamic step capture launches over the output row extent; the kernel
    // reads the step's output row count on the device.
    const auto& dynamic = riley_cuda_internal::thread_step_capture_dynamic();
    const uint64_t launch_elements =
        dynamic.counts != nullptr ? dynamic.output_rows * params->column_count
                                  : output_elements;
    if (params->input.dtype == RILEY_CUDA_DTYPE_F32) {
      row_gather_kernel<float>
          <<<block_count(launch_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const float*>(input.data),
              reinterpret_cast<const uint32_t*>(row_indices.data),
              reinterpret_cast<float*>(output.data), params->input_row_count,
              params->column_count, output_elements, dynamic.counts);
    } else {
      row_gather_kernel<__nv_bfloat16>
          <<<block_count(launch_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data),
              reinterpret_cast<const uint32_t*>(row_indices.data),
              reinterpret_cast<__nv_bfloat16*>(output.data),
              params->input_row_count, params->column_count, output_elements,
              dynamic.counts);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RileyCudaStatus riley_cuda_bf16_argmax_execute(
    const RileyCudaBf16ArgmaxParams* params, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute deterministic BF16 argmax";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaBf16ArgmaxParams stable_params = *params;
  params = &stable_params;
  // reserved[0] is the optional candidate bound: zero selects among every
  // column, otherwise only token IDs below it can win.
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved + 1, 3) ||
      params->reserved[0] > params->vocabulary_size) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero and the candidate bound within the vocabulary");
  }
  if (params->logits.dtype != RILEY_CUDA_DTYPE_BF16 ||
      params->results.dtype != RILEY_CUDA_DTYPE_U32) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "argmax requires BF16 logits and U32 result records");
  }
  if (params->vocabulary_size == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "vocabulary_size must be greater than zero");
  }
  if (params->vocabulary_size > UINT32_MAX) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "vocabulary_size exceeds the U32 token-id contract");
  }

  uint64_t logit_elements = 0;
  uint64_t result_elements = 0;
  if (!checked_multiply(params->row_count, params->vocabulary_size,
                        &logit_elements) ||
      !checked_multiply(params->row_count, uint64_t{2}, &result_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "argmax shape overflows uint64_t");
  }
  uint64_t logit_bytes = 0;
  uint64_t result_bytes = 0;
  RileyCudaStatus status =
      typed_bytes(logit_elements, RILEY_CUDA_DTYPE_BF16, &logit_bytes,
                  error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(result_elements, RILEY_CUDA_DTYPE_U32,
                         &result_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan logits{};
  ResolvedSpan results{};
  status = resolve_span(params->logits, RILEY_CUDA_DTYPE_BF16,
                        logit_bytes, &logits, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->results, RILEY_CUDA_DTYPE_U32,
                          result_bytes, &results, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(results, logits, false, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {logits, results};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(logits.buffer) || !uses.add(results.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "batch buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->row_count == 0) {
    return uses.release_completed()
               ? RILEY_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const auto& dynamic = riley_cuda_internal::thread_step_capture_dynamic();
    const uint64_t launch_rows =
        dynamic.counts != nullptr ? dynamic.output_rows : params->row_count;
    const uint32_t blocks = static_cast<uint32_t>(
        launch_rows < kMaximumBlocks ? launch_rows : kMaximumBlocks);
    bf16_argmax_kernel<<<blocks, kArgmaxThreads, 0, stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data),
        reinterpret_cast<RileyCudaBf16ArgmaxResult*>(results.data),
        params->row_count, params->vocabulary_size,
        params->reserved[0] != 0 ? params->reserved[0]
                                 : params->vocabulary_size,
        dynamic.counts);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RileyCudaStatus
riley_cuda_ragged_paged_kv_cache_write_execute(
    const RileyCudaRaggedPagedKvCacheWriteParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "write ragged paged KV cache";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaRaggedPagedKvCacheWriteParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RileyCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->key_value_head_count == 0 || params->head_size == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "KV head dimensions must be greater than zero");
  }

  uint64_t source_elements = 0;
  uint64_t pool_elements = 0;
  if (!checked_product3(params->batch.active_row_count,
                        params->key_value_head_count, params->head_size,
                        &source_elements) ||
      !checked_product4(params->batch.physical_block_count,
                        params->key_value_head_count,
                        RILEY_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged paged KV tensor shape overflows uint64_t");
  }
  uint64_t source_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(source_elements, RILEY_CUDA_DTYPE_BF16,
                       &source_bytes, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RILEY_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan key_source{};
  ResolvedSpan value_source{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedBatch batch{};
  status = resolve_span(params->key_source, RILEY_CUDA_DTYPE_BF16,
                        source_bytes, &key_source, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_source, RILEY_CUDA_DTYPE_BF16,
                          source_bytes, &value_source, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  status = reject_overlap(key_pool, key_source, false, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_pool, value_source, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_pool, value_pool, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_pool, key_source, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_pool, value_source, false, error,
                            kOperation);
  }
  for (size_t index = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(key_pool, metadata[index], false, error,
                            kOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = reject_overlap(value_pool, metadata[index], false, error,
                              kOperation);
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      key_source, value_source, key_pool, value_pool,
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    // A dynamic step capture covers every dense row; rows at or beyond the
    // device-side active count resolve to no cache slot and are skipped.
    const auto& dynamic = riley_cuda_internal::thread_step_capture_dynamic();
    const uint64_t launch_elements =
        dynamic.counts != nullptr
            ? dynamic.dense_rows * params->key_value_head_count *
                  params->head_size
            : source_elements;
    auto& deferred = riley_cuda_internal::thread_deferred_qkv_split();
    if (deferred.armed) {
      // The split that produced these sources deferred its launch: run the
      // split and this write in one kernel.
      const riley_cuda_internal::DeferredQkvSplit split = deferred;
      deferred = {};
      if (split.stream != stream->stream || split.key != key_source.data ||
          split.value != value_source.data ||
          split.head_size != params->head_size ||
          split.key_value_width !=
              params->key_value_head_count * params->head_size) {
        status = validation_error(
            error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
            "KV cache write does not match the deferred merged QKV split");
      } else {
        const uint64_t work_items =
            split.row_count * (split.query_width / 2 +
                               split.key_value_width / 2 +
                               split.key_value_width);
        qkv_split_bias_rope_kv_write_kernel
            <<<block_count(work_items), kThreads, 0, stream->stream>>>(
                static_cast<const __nv_bfloat16*>(split.merged),
                static_cast<const __nv_bfloat16*>(split.query_bias),
                static_cast<const __nv_bfloat16*>(split.key_bias),
                static_cast<const __nv_bfloat16*>(split.value_bias),
                static_cast<const float*>(split.cos),
                static_cast<const float*>(split.sin),
                static_cast<const uint32_t*>(split.positions),
                static_cast<__nv_bfloat16*>(split.query),
                static_cast<__nv_bfloat16*>(split.key),
                static_cast<__nv_bfloat16*>(split.value),
                reinterpret_cast<__nv_bfloat16*>(key_pool.data),
                reinterpret_cast<__nv_bfloat16*>(value_pool.data),
                device_batch(params->batch, batch),
                params->key_value_head_count, split.query_width,
                split.key_value_width, split.head_size,
                split.table_position_count, split.rotated_row_count,
                work_items, dynamic.counts);
        status = launch_status(error, kOperation);
      }
    } else {
    ragged_paged_kv_cache_write_kernel
        <<<block_count(launch_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key_source.data),
            reinterpret_cast<const __nv_bfloat16*>(value_source.data),
            reinterpret_cast<__nv_bfloat16*>(key_pool.data),
            reinterpret_cast<__nv_bfloat16*>(value_pool.data),
            device_batch(params->batch, batch), params->key_value_head_count,
            params->head_size, launch_elements);
    status = launch_status(error, kOperation);
    }
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

namespace {

RileyCudaStatus execute_ragged_paged_attention(
    const RileyCudaRaggedPagedAttentionParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error,
    bool grouped_heads) noexcept {
  constexpr const char* kOperation = "execute ragged causal paged attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaRaggedPagedAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RileyCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->query_head_count == 0 ||
      params->key_value_head_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention head counts must be greater than zero");
  }
  if (params->head_size != kAttentionHeadSize) {
    return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged paged attention supports head_size=64 only");
  }
  if (params->query_head_count % params->key_value_head_count != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "key_value_head_count must divide query_head_count");
  }
  if (!std::isfinite(params->scale) || params->scale <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention scale must be finite and greater than zero");
  }
  if (params->output_row_count < params->batch.active_row_count) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "output_row_count is smaller than active_row_count");
  }
  if (params->output_row_count > kMaximumGridX ||
      params->query_head_count > kMaximumGridYOrZ) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged attention launch dimensions exceed the CUDA grid contract");
  }

  uint64_t query_elements = 0;
  uint64_t output_elements = 0;
  uint64_t pool_elements = 0;
  if (!checked_product3(params->batch.active_row_count,
                        params->query_head_count, params->head_size,
                        &query_elements) ||
      !checked_product3(params->output_row_count, params->query_head_count,
                        params->head_size, &output_elements) ||
      !checked_product4(params->batch.physical_block_count,
                        params->key_value_head_count,
                        RILEY_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged attention tensor shape overflows uint64_t");
  }
  uint64_t query_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(query_elements, RILEY_CUDA_DTYPE_BF16,
                       &query_bytes, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, RILEY_CUDA_DTYPE_BF16,
                         &output_bytes, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RILEY_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan output{};
  ResolvedBatch batch{};
  status = resolve_span(params->query, RILEY_CUDA_DTYPE_BF16,
                        query_bytes, &query, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RILEY_CUDA_DTYPE_BF16,
                          output_bytes, &output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = reject_overlap(output, query, false, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key_pool, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value_pool, false, error, kOperation);
  }
  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  for (size_t index = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, metadata[index], false, error,
                            kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      query, key_pool, value_pool, output, batch.sequence_block_offsets,
      batch.block_ids, batch.valid_tokens, batch.row_sequence_slots,
      batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const uint64_t query_head_group_size =
        params->query_head_count / params->key_value_head_count;
    if (grouped_heads && query_head_group_size > 1 &&
        query_head_group_size <= kRaggedAttentionMaximumWarpsPerBlock) {
      // One CTA per KV head lets its GQA query-head warps reuse the same
      // immutable paged K/V tile. Shapes outside the bounded GQA group fall
      // through to the existing generic grouped-head path below.
      const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                      static_cast<uint32_t>(params->key_value_head_count),
                      1);
      ragged_paged_attention_gqa_shared_kv_kernel<<<
          grid, static_cast<uint32_t>(query_head_group_size * kWarpSize), 0,
          stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<__nv_bfloat16*>(output.data),
          device_batch(params->batch, batch), params->query_head_count,
          params->key_value_head_count, params->scale);
    } else if (grouped_heads) {
      const uint32_t query_head_warps = static_cast<uint32_t>(
          params->query_head_count < kRaggedAttentionMaximumWarpsPerBlock
              ? params->query_head_count
              : kRaggedAttentionMaximumWarpsPerBlock);
      const uint32_t query_head_tiles = static_cast<uint32_t>(
          (params->query_head_count + query_head_warps - 1) /
          query_head_warps);
      const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                      query_head_tiles, 1);
      ragged_paged_attention_grouped_heads_kernel<<<
          grid, query_head_warps * kWarpSize, 0, stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<__nv_bfloat16*>(output.data),
          device_batch(params->batch, batch), params->query_head_count,
          params->key_value_head_count, params->scale);
    } else {
      const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                      static_cast<uint32_t>(params->query_head_count), 1);
      ragged_paged_attention_kernel<<<grid, kWarpSize, 0, stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<__nv_bfloat16*>(output.data),
          device_batch(params->batch, batch), params->query_head_count,
          params->key_value_head_count, params->scale);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

}  // namespace

extern "C" RileyCudaStatus
riley_cuda_ragged_paged_attention_execute(
    const RileyCudaRaggedPagedAttentionParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  return execute_ragged_paged_attention(params, stream, error, false);
}

extern "C" RileyCudaStatus
riley_cuda_ragged_paged_attention_grouped_heads_execute(
    const RileyCudaRaggedPagedAttentionParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  return execute_ragged_paged_attention(params, stream, error, true);
}

extern "C" RileyCudaStatus
riley_cuda_fixed37_ragged_paged_attention_two_pass_execute(
    const RileyCudaFixed37RaggedPagedAttentionParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 two-pass ragged causal paged attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaFixed37RaggedPagedAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RileyCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->query_head_count == 0 ||
      params->key_value_head_count == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention head counts must be greater than zero");
  }
  if (params->head_size != kAttentionHeadSize) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged paged attention supports head_size=64 only");
  }
  if (params->query_head_count % params->key_value_head_count != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "key_value_head_count must divide query_head_count");
  }
  if (params->maximum_logical_token_count == 0) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "maximum_logical_token_count must be greater than zero");
  }
  if (params->maximum_logical_token_count >
      kFixed37RaggedMaximumTokenCount) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged paged attention supports maximum logical T<=8192 only");
  }
  if (!std::isfinite(params->scale) || params->scale <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention scale must be finite and greater than zero");
  }
  if (params->output_row_count < params->batch.active_row_count) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "output_row_count is smaller than active_row_count");
  }
  if (params->output_row_count > kMaximumGridX ||
      params->query_head_count > kMaximumGridYOrZ) {
    return validation_error(
        error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged attention launch dimensions exceed the CUDA grid contract");
  }

  uint64_t maximum_token_partial_count = 0;
  uint64_t shared_bytes = 0;
  status = fixed37_ragged_shared_bytes(
      params->maximum_logical_token_count, &maximum_token_partial_count,
      &shared_bytes, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  uint64_t query_elements = 0;
  uint64_t output_elements = 0;
  uint64_t pool_elements = 0;
  if (!checked_product3(params->batch.active_row_count,
                        params->query_head_count, params->head_size,
                        &query_elements) ||
      !checked_product3(params->output_row_count, params->query_head_count,
                        params->head_size, &output_elements) ||
      !checked_product4(params->batch.physical_block_count,
                        params->key_value_head_count,
                        RILEY_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "fixed37 ragged attention tensor shape overflows uint64_t");
  }
  uint64_t query_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(query_elements, RILEY_CUDA_DTYPE_BF16,
                       &query_bytes, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, RILEY_CUDA_DTYPE_BF16,
                         &output_bytes, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RILEY_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan output{};
  ResolvedBatch batch{};
  status = resolve_span(params->query, RILEY_CUDA_DTYPE_BF16,
                        query_bytes, &query, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RILEY_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RILEY_CUDA_DTYPE_BF16,
                          output_bytes, &output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = reject_overlap(output, query, false, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key_pool, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value_pool, false, error, kOperation);
  }
  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  for (size_t index = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, metadata[index], false, error,
                            kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      query, key_pool, value_pool, output, batch.sequence_block_offsets,
      batch.block_ids, batch.valid_tokens, batch.row_sequence_slots,
      batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                    static_cast<uint32_t>(params->query_head_count), 1);
    fixed37_ragged_paged_attention_two_pass_kernel<<<
        grid, riley_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
        reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        device_batch(params->batch, batch), params->query_head_count,
        params->key_value_head_count,
        params->maximum_logical_token_count, params->scale,
        maximum_token_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

namespace {

struct NativeBf16RaggedPartialTransitionV2 {
  float left_scale;
  float right_scale;
};

__device__ __forceinline__ bool resolve_ragged_partial_block(
    const DeviceBatch& batch, uint64_t row, uint64_t logical_block,
    uint64_t key_value_head, uint64_t* block_head_base,
    uint64_t* token_count) {
  if (block_head_base == nullptr || token_count == nullptr ||
      row >= active_rows(batch)) {
    return false;
  }
  const uint64_t sequence = batch.row_sequence_slots[row];
  if (sequence >= batch.sequence_count) {
    return false;
  }
  const uint64_t block_begin = batch.sequence_block_offsets[sequence];
  const uint64_t block_end = batch.sequence_block_offsets[sequence + 1];
  if (block_begin > block_end || block_end > batch.block_count ||
      logical_block >= block_end - block_begin) {
    return false;
  }
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;
  const uint64_t block_token_begin =
      logical_block * RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  if (block_token_begin >= logical_token_count) {
    return false;
  }
  const uint64_t block_index = block_begin + logical_block;
  const uint64_t physical_block = batch.block_ids[block_index];
  const uint64_t valid = batch.valid_tokens[block_index];
  uint64_t required = logical_token_count - block_token_begin;
  if (required > RILEY_CUDA_PAGED_KV_BLOCK_SIZE) {
    required = RILEY_CUDA_PAGED_KV_BLOCK_SIZE;
  }
  if (physical_block >= batch.physical_block_count || valid < required ||
      valid == 0 || valid > RILEY_CUDA_PAGED_KV_BLOCK_SIZE) {
    return false;
  }
  *block_head_base =
      (physical_block *
           kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount +
       key_value_head) *
      RILEY_CUDA_PAGED_KV_BLOCK_SIZE *
      kNativeBf16RaggedPagedSplitGqaD128HeadSize;
  *token_count = required;
  return true;
}

__global__ __launch_bounds__(kWarpSize) void
native_bf16_ragged_paged_split_gqa_d128_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, float* partial_states,
    DeviceBatch batch, uint64_t partial_state_capacity, float scale) {
  const uint32_t lane = threadIdx.x;
  const uint64_t logical_block = blockIdx.x;
  const uint64_t query_head = blockIdx.y;
  const uint64_t row = blockIdx.z;
  if (row >= active_rows(batch) ||
      query_head >= kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount) {
    return;
  }
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;
  const uint64_t partial_state_count =
      ((logical_token_count - 1) / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) + 1;
  // Capacity tails remain untouched. This is also what makes each row's
  // state prefix independently reducible by the established V1 generic path.
  if (logical_block >= partial_state_count ||
      logical_block >= partial_state_capacity) {
    return;
  }

  const uint64_t group_size =
      kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount /
      kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base =
      (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head) *
      kNativeBf16RaggedPagedSplitGqaD128HeadSize;
  uint64_t block_head_base = 0;
  uint64_t token_count = 0;
  const bool valid_block = resolve_ragged_partial_block(
      batch, row, logical_block, key_value_head, &block_head_base,
      &token_count);

  const float query0 = __bfloat162float(query[query_base + lane]);
  const float query1 =
      __bfloat162float(query[query_base + lane + kWarpSize]);
  const float query2 =
      __bfloat162float(query[query_base + lane + 2 * kWarpSize]);
  const float query3 =
      __bfloat162float(query[query_base + lane + 3 * kWarpSize]);
  float maximum = valid_block ? -CUDART_INF_F : CUDART_NAN_F;
  float denominator = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator0 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator1 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator2 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator3 = valid_block ? 0.0F : CUDART_NAN_F;
  if (valid_block) {
    for (uint64_t token_in_block = 0; token_in_block < token_count;
         ++token_in_block) {
      const uint64_t cache_base =
          block_head_base + token_in_block *
                                kNativeBf16RaggedPagedSplitGqaD128HeadSize;
      const float key0 = __bfloat162float(key_pool[cache_base + lane]);
      const float key1 =
          __bfloat162float(key_pool[cache_base + lane + kWarpSize]);
      const float key2 =
          __bfloat162float(key_pool[cache_base + lane + 2 * kWarpSize]);
      const float key3 =
          __bfloat162float(key_pool[cache_base + lane + 3 * kWarpSize]);
      float score = fmaf(query0, key0, query1 * key1);
      score = fmaf(query2, key2, score);
      score = fmaf(query3, key3, score);
      score = warp_sum(score);
      score = staged_attention_score(
          __shfl_sync(kFullWarpMask, score, 0), scale);

      float alpha = 0.0F;
      float beta = 0.0F;
      if (lane == 0) {
        update_online_state(score, &maximum, &denominator, &alpha, &beta);
      }
      alpha = __shfl_sync(kFullWarpMask, alpha, 0);
      beta = __shfl_sync(kFullWarpMask, beta, 0);
      numerator0 = update_numerator(
          numerator0, __bfloat162float(value_pool[cache_base + lane]), alpha,
          beta);
      numerator1 = update_numerator(
          numerator1,
          __bfloat162float(value_pool[cache_base + lane + kWarpSize]), alpha,
          beta);
      numerator2 = update_numerator(
          numerator2,
          __bfloat162float(value_pool[cache_base + lane + 2 * kWarpSize]),
          alpha, beta);
      numerator3 = update_numerator(
          numerator3,
          __bfloat162float(value_pool[cache_base + lane + 3 * kWarpSize]),
          alpha, beta);
    }
  }
  const uint64_t state_base =
      (((row * partial_state_capacity + logical_block) *
            kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
        query_head) *
       kNativeBf16RaggedPagedSplitGqaD128StateStride);
  if (lane == 0) {
    partial_states[state_base] = maximum;
    partial_states[state_base + 1] = denominator;
  }
  partial_states[state_base + 2 + lane] = numerator0;
  partial_states[state_base + 2 + lane + kWarpSize] = numerator1;
  partial_states[state_base + 2 + lane + 2 * kWarpSize] = numerator2;
  partial_states[state_base + 2 + lane + 3 * kWarpSize] = numerator3;
}

// Row kinds for the D128 two-stage attention when the prompt rows take the
// tensor-core prefill producer: a prompt row is a token of a request's prompt,
// whatever chunk it arrives in; every other row keeps the shared-KV producer.
constexpr uint32_t kNativeBf16RaggedD128PromptRowKind = 1;
// The prefill producer writes one partial state per four logical pages.
constexpr uint64_t kNativeBf16PrefillD128MmaSegmentPages = 4;
constexpr uint64_t kNativeBf16PrefillD128MmaTileRows = 16;

__device__ __forceinline__ uint32_t pack_bf16_pair(float first, float second) {
  return static_cast<uint32_t>(
             __bfloat16_as_ushort(__float2bfloat16_rn(first))) |
         (static_cast<uint32_t>(
              __bfloat16_as_ushort(__float2bfloat16_rn(second)))
          << 16);
}

__device__ __forceinline__ void mma_m16n8k16_bf16(float* accumulator,
                                                  uint32_t a0, uint32_t a1,
                                                  uint32_t a2, uint32_t a3,
                                                  uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]),
        "+f"(accumulator[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Tensor-core partial-state producer for prompt rows. One CTA owns a
// `(16-row tile, key_value_head, 4-page segment)`; its eight warps are the
// eight query heads of that KV head and share one staged K tile and one
// transposed V tile per page. Within the tile, each run of consecutive prompt
// rows of one sequence is processed in turn with the other rows masked.
//
// For a row, a query head and a segment the state `[m, l, n[128]]` is the
// online-softmax fold of the segment's pages in ascending order: scores are
// the FP32 m16n8k16 accumulation over depth times `scale`, keys after the
// row's own position score -Inf, probabilities are rounded to BF16 before
// they weight the denominator and the value MMA. An MMA output element
// depends only on its own row and column, so a row's state does not depend
// on which rows share its tile, on the tile's alignment or on the chunk the
// row arrived in. This is a different rounding from the shared-KV producer
// and therefore a separately selected numerical mode for prompt rows.
__global__ __launch_bounds__(256)
void native_bf16_prefill_d128_mma_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, float* partial_states,
    DeviceBatch batch, uint64_t partial_state_capacity, float scale,
    const uint32_t* row_kinds) {
  constexpr uint32_t kHead = 128;
  constexpr uint32_t kPage = 16;
  __shared__ __align__(16) __nv_bfloat16 key_tiles[2][kPage][kHead];
  __shared__ __align__(16) __nv_bfloat16 value_tiles[2][kHead][kPage];
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint32_t group = lane / 4;
  const uint32_t pair = lane % 4;
  const uint64_t key_value_head = blockIdx.y;
  const uint64_t query_head = key_value_head * 8 + warp;
  const uint64_t segment = blockIdx.z;
  const uint64_t first_page = segment * kNativeBf16PrefillD128MmaSegmentPages;
  const uint64_t tile_begin =
      static_cast<uint64_t>(blockIdx.x) * kNativeBf16PrefillD128MmaTileRows;
  const uint64_t row_count = active_rows(batch);
  const uint64_t tile_end =
      row_count < tile_begin + kNativeBf16PrefillD128MmaTileRows
          ? row_count
          : tile_begin + kNativeBf16PrefillD128MmaTileRows;
  // Everything that steers the loops and barriers below is CTA-uniform.
  if (row_kinds == nullptr || segment >= partial_state_capacity ||
      key_value_head >= kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount) {
    return;
  }
  const uint64_t lane_row0 = tile_begin + group;
  const uint64_t lane_row1 = lane_row0 + 8;
  uint64_t run_begin = tile_begin;
  while (run_begin < tile_end) {
    if (row_kinds[run_begin] != kNativeBf16RaggedD128PromptRowKind) {
      ++run_begin;
      continue;
    }
    const uint32_t sequence = batch.row_sequence_slots[run_begin];
    uint64_t run_end = run_begin + 1;
    while (run_end < tile_end &&
           row_kinds[run_end] == kNativeBf16RaggedD128PromptRowKind &&
           batch.row_sequence_slots[run_end] == sequence &&
           batch.row_positions[run_end] ==
               batch.row_positions[run_end - 1] + 1) {
      ++run_end;
    }
    const uint64_t run_last_row = run_end - 1;
    const uint64_t run_pages =
        static_cast<uint64_t>(batch.row_positions[run_last_row]) / kPage + 1;
    const uint64_t this_run_begin = run_begin;
    run_begin = run_end;
    if (first_page >= run_pages) {
      continue;
    }
    const uint64_t end_page =
        run_pages < first_page + kNativeBf16PrefillD128MmaSegmentPages
            ? run_pages
            : first_page + kNativeBf16PrefillD128MmaSegmentPages;
    const bool live0 = lane_row0 >= this_run_begin && lane_row0 < run_end;
    const bool live1 = lane_row1 >= this_run_begin && lane_row1 < run_end;
    const uint64_t position0 = live0 ? batch.row_positions[lane_row0] : 0;
    const uint64_t position1 = live1 ? batch.row_positions[lane_row1] : 0;

    uint32_t query_operand[8][4];
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) {
      const __nv_bfloat16* q0 =
          query +
          ((live0 ? lane_row0 : this_run_begin) *
               kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
           query_head) *
              kHead +
          step * 16 + 2 * pair;
      const __nv_bfloat16* q1 =
          query +
          ((live1 ? lane_row1 : this_run_begin) *
               kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
           query_head) *
              kHead +
          step * 16 + 2 * pair;
      query_operand[step][0] =
          live0 ? *reinterpret_cast<const uint32_t*>(q0) : 0;
      query_operand[step][1] =
          live1 ? *reinterpret_cast<const uint32_t*>(q1) : 0;
      query_operand[step][2] =
          live0 ? *reinterpret_cast<const uint32_t*>(q0 + 8) : 0;
      query_operand[step][3] =
          live1 ? *reinterpret_cast<const uint32_t*>(q1 + 8) : 0;
    }

    // Tokens at or past the page's valid count are staged as zero, so a
    // masked key multiplies a zero probability with a finite value.
    const auto load_page = [&](uint64_t page, uint32_t* key_words,
                               uint32_t* value_words) {
      uint64_t block_head_base = 0;
      uint64_t token_count = 0;
      const bool valid = resolve_ragged_partial_block(
          batch, run_last_row, page, key_value_head, &block_head_base,
          &token_count);
#pragma unroll
      for (uint32_t part = 0; part < 4; ++part) {
        const uint32_t word = threadIdx.x + part * blockDim.x;
        const uint32_t token = word / (kHead / 2);
        const uint32_t column = 2 * (word % (kHead / 2));
        const bool present = valid && token < token_count;
        const uint64_t element = block_head_base + token * kHead + column;
        key_words[part] =
            present ? *reinterpret_cast<const uint32_t*>(key_pool + element)
                    : 0;
        value_words[part] =
            present ? *reinterpret_cast<const uint32_t*>(value_pool + element)
                    : 0;
      }
    };
    const auto store_page = [&](uint32_t buffer, const uint32_t* key_words,
                                const uint32_t* value_words) {
#pragma unroll
      for (uint32_t part = 0; part < 4; ++part) {
        const uint32_t word = threadIdx.x + part * blockDim.x;
        const uint32_t token = word / (kHead / 2);
        const uint32_t column = 2 * (word % (kHead / 2));
        *reinterpret_cast<uint32_t*>(&key_tiles[buffer][token][column]) =
            key_words[part];
        value_tiles[buffer][column][token] = __ushort_as_bfloat16(
            static_cast<uint16_t>(value_words[part]));
        value_tiles[buffer][column + 1][token] = __ushort_as_bfloat16(
            static_cast<uint16_t>(value_words[part] >> 16));
      }
    };

    float numerator[16][4];
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
      numerator[dim][0] = 0.0F;
      numerator[dim][1] = 0.0F;
      numerator[dim][2] = 0.0F;
      numerator[dim][3] = 0.0F;
    }
    float maximum0 = -CUDART_INF_F;
    float maximum1 = -CUDART_INF_F;
    float denominator0 = 0.0F;
    float denominator1 = 0.0F;
    {
      uint32_t key_words[4];
      uint32_t value_words[4];
      load_page(first_page, key_words, value_words);
      store_page(static_cast<uint32_t>(first_page & 1), key_words,
                 value_words);
    }
    __syncthreads();
    for (uint64_t page = first_page; page < end_page; ++page) {
      const uint32_t buffer = static_cast<uint32_t>(page & 1);
      // The next page is loaded before this page's arithmetic and stored
      // after it, so its memory latency overlaps the MMAs.
      uint32_t next_key_words[4];
      uint32_t next_value_words[4];
      const bool more = page + 1 < end_page;
      if (more) {
        load_page(page + 1, next_key_words, next_value_words);
      }
      float score[2][4] = {{0.0F, 0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F, 0.0F}};
#pragma unroll
      for (uint32_t step = 0; step < 8; ++step) {
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          const __nv_bfloat16* key_operand =
              &key_tiles[buffer][half * 8 + group][step * 16 + 2 * pair];
          mma_m16n8k16_bf16(
              score[half], query_operand[step][0], query_operand[step][1],
              query_operand[step][2], query_operand[step][3],
              *reinterpret_cast<const uint32_t*>(key_operand),
              *reinterpret_cast<const uint32_t*>(key_operand + 8));
        }
      }
      float page_maximum0 = -CUDART_INF_F;
      float page_maximum1 = -CUDART_INF_F;
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
#pragma unroll
        for (uint32_t element = 0; element < 2; ++element) {
          const uint64_t key_position =
              page * kPage + half * 8 + 2 * pair + element;
          score[half][element] = (live0 && key_position <= position0)
                                     ? score[half][element] * scale
                                     : -CUDART_INF_F;
          score[half][2 + element] = (live1 && key_position <= position1)
                                         ? score[half][2 + element] * scale
                                         : -CUDART_INF_F;
          page_maximum0 = fmaxf(page_maximum0, score[half][element]);
          page_maximum1 = fmaxf(page_maximum1, score[half][2 + element]);
        }
      }
      // A row's sixteen scores of this page sit in the four lanes of its group.
      page_maximum0 = fmaxf(page_maximum0,
                            __shfl_xor_sync(kFullWarpMask, page_maximum0, 1));
      page_maximum0 = fmaxf(page_maximum0,
                            __shfl_xor_sync(kFullWarpMask, page_maximum0, 2));
      page_maximum1 = fmaxf(page_maximum1,
                            __shfl_xor_sync(kFullWarpMask, page_maximum1, 1));
      page_maximum1 = fmaxf(page_maximum1,
                            __shfl_xor_sync(kFullWarpMask, page_maximum1, 2));
      const float next_maximum0 = fmaxf(maximum0, page_maximum0);
      const float next_maximum1 = fmaxf(maximum1, page_maximum1);
      const bool empty0 = next_maximum0 == -CUDART_INF_F;
      const bool empty1 = next_maximum1 == -CUDART_INF_F;
      const float alpha0 = empty0 ? 1.0F : expf(maximum0 - next_maximum0);
      const float alpha1 = empty1 ? 1.0F : expf(maximum1 - next_maximum1);
      maximum0 = next_maximum0;
      maximum1 = next_maximum1;
      float page_sum0 = 0.0F;
      float page_sum1 = 0.0F;
      uint32_t probability_operand[4];
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        float probability[4];
#pragma unroll
        for (uint32_t element = 0; element < 2; ++element) {
          probability[element] = __bfloat162float(__float2bfloat16_rn(
              empty0 ? 0.0F : expf(score[half][element] - next_maximum0)));
          probability[2 + element] = __bfloat162float(__float2bfloat16_rn(
              empty1 ? 0.0F
                     : expf(score[half][2 + element] - next_maximum1)));
          page_sum0 += probability[element];
          page_sum1 += probability[2 + element];
        }
        probability_operand[half * 2] =
            pack_bf16_pair(probability[0], probability[1]);
        probability_operand[half * 2 + 1] =
            pack_bf16_pair(probability[2], probability[3]);
      }
      page_sum0 += __shfl_xor_sync(kFullWarpMask, page_sum0, 1);
      page_sum0 += __shfl_xor_sync(kFullWarpMask, page_sum0, 2);
      page_sum1 += __shfl_xor_sync(kFullWarpMask, page_sum1, 1);
      page_sum1 += __shfl_xor_sync(kFullWarpMask, page_sum1, 2);
      denominator0 = fmaf(denominator0, alpha0, page_sum0);
      denominator1 = fmaf(denominator1, alpha1, page_sum1);
#pragma unroll
      for (uint32_t dim = 0; dim < 16; ++dim) {
        numerator[dim][0] *= alpha0;
        numerator[dim][1] *= alpha0;
        numerator[dim][2] *= alpha1;
        numerator[dim][3] *= alpha1;
        const __nv_bfloat16* value_operand =
            &value_tiles[buffer][dim * 8 + group][2 * pair];
        mma_m16n8k16_bf16(
            numerator[dim], probability_operand[0], probability_operand[1],
            probability_operand[2], probability_operand[3],
            *reinterpret_cast<const uint32_t*>(value_operand),
            *reinterpret_cast<const uint32_t*>(value_operand + 8));
      }
      if (more) {
        store_page(static_cast<uint32_t>((page + 1) & 1), next_key_words,
                   next_value_words);
      }
      __syncthreads();
    }

    // A row writes this segment only if the segment starts inside its context.
    const bool write0 = live0 && first_page < position0 / kPage + 1;
    const bool write1 = live1 && first_page < position1 / kPage + 1;
    float* const state0 =
        partial_states +
        ((lane_row0 * partial_state_capacity + segment) *
             kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
         query_head) *
            kNativeBf16RaggedPagedSplitGqaD128StateStride;
    float* const state1 =
        partial_states +
        ((lane_row1 * partial_state_capacity + segment) *
             kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
         query_head) *
            kNativeBf16RaggedPagedSplitGqaD128StateStride;
    if (write0 && pair == 0) {
      state0[0] = maximum0;
      state0[1] = denominator0;
    }
    if (write1 && pair == 0) {
      state1[0] = maximum1;
      state1[1] = denominator1;
    }
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
      if (write0) {
        state0[2 + dim * 8 + 2 * pair] = numerator[dim][0];
        state0[2 + dim * 8 + 2 * pair + 1] = numerator[dim][1];
      }
      if (write1) {
        state1[2 + dim * 8 + 2 * pair] = numerator[dim][2];
        state1[2 + dim * 8 + 2 * pair + 1] = numerator[dim][3];
      }
    }
  }
}

// Tensor-core partial-state producer for the D128 two-stage attention. One
// single-warp CTA owns a `(logical_block, key_value_head, row)` page, the
// launch geometry of the shared-KV producer, and writes the page's state
// `[m, l, n[128]]` for all eight query heads of the KV head:
//   1. score = BF16(BF16(q . k) * scale): eight m16n8k16 MMAs (tokens x
//      heads) accumulate the depth steps in ascending order into one FP32
//      accumulator, with the paged producers' two BF16 roundings. Tokens
//      past the page's valid count score -Inf.
//   2. m = max of the page's scores; e = exp(score - m); l = sum of e (the
//      two tokens of a lane, then a fixed lane tree).
//   3. n = sum of e * v with the full FP32 weight: e is split exactly into
//      three BF16 pieces (e = high + middle + low) and each piece is one MMA
//      operand (depth x heads over tokens), so no weight bit is dropped.
// A state depends only on its own page and row, so it does not depend on the
// other rows of the batch or on the chunk a prompt token arrives in. It is
// the same page-local softmax state as the shared-KV producer's, computed in
// a different order, so it differs from it in rounding only; it is a
// separately selected numerical mode. An unresolvable page writes a NaN
// state; a non-finite score yields a NaN state through its weights.
__global__ __launch_bounds__(kWarpSize)
void native_bf16_ragged_paged_split_gqa_d128_tensor_core_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, float* partial_states,
    DeviceBatch batch, uint64_t partial_state_capacity, float scale) {
  constexpr uint32_t kHead = 128;
  constexpr uint32_t kGroupHeads = 8;
  const uint32_t lane = threadIdx.x;
  const uint32_t group = lane / 4;
  const uint32_t pair = lane % 4;
  const uint64_t logical_block = blockIdx.x;
  const uint64_t key_value_head = blockIdx.y;
  const uint64_t row = blockIdx.z;
  // Every exit before the shuffles below is uniform across the warp.
  if (row >= active_rows(batch) ||
      key_value_head >= kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount) {
    return;
  }
  const uint64_t partial_state_count =
      static_cast<uint64_t>(batch.row_positions[row]) /
          RILEY_CUDA_PAGED_KV_BLOCK_SIZE + 1;
  if (logical_block >= partial_state_count ||
      logical_block >= partial_state_capacity) {
    return;
  }
  // This lane holds query heads `2 * pair` and `2 * pair + 1` of the group.
  float* const state0 =
      partial_states +
      ((row * partial_state_capacity + logical_block) *
           kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
       key_value_head * kGroupHeads + 2 * pair) *
          kNativeBf16RaggedPagedSplitGqaD128StateStride;
  float* const state1 = state0 + kNativeBf16RaggedPagedSplitGqaD128StateStride;
  uint64_t block_head_base = 0;
  uint64_t token_count = 0;
  const bool valid_block = resolve_ragged_partial_block(
      batch, row, logical_block, key_value_head, &block_head_base,
      &token_count);
  float maximum0 = CUDART_NAN_F;
  float maximum1 = CUDART_NAN_F;
  float denominator0 = CUDART_NAN_F;
  float denominator1 = CUDART_NAN_F;
  float numerator[8][4];
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) {
    numerator[step][0] = valid_block ? 0.0F : CUDART_NAN_F;
    numerator[step][1] = numerator[step][0];
    numerator[step][2] = numerator[step][0];
    numerator[step][3] = numerator[step][0];
  }
  if (valid_block) {
    // Query operand B[k = depth][n = head]: head `group` of this KV head.
    const __nv_bfloat16* const query_head =
        query +
        (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
         key_value_head * kGroupHeads + group) *
            kHead +
        2 * pair;
    // Key operand A[m = token][k = depth]: tokens `group` and `group + 8`.
    const __nv_bfloat16* const key_low =
        key_pool + block_head_base + group * kHead + 2 * pair;
    const __nv_bfloat16* const key_high = key_low + 8 * kHead;
    // The eight depth steps accumulate into one FP32 accumulator in
    // ascending order, the way a tensor-core GEMM walks its inner dimension.
    float dot[4] = {0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) {
      mma_m16n8k16_bf16(
          dot, *reinterpret_cast<const uint32_t*>(key_low + 16 * step),
          *reinterpret_cast<const uint32_t*>(key_high + 16 * step),
          *reinterpret_cast<const uint32_t*>(key_low + 16 * step + 8),
          *reinterpret_cast<const uint32_t*>(key_high + 16 * step + 8),
          *reinterpret_cast<const uint32_t*>(query_head + 16 * step),
          *reinterpret_cast<const uint32_t*>(query_head + 16 * step + 8));
    }
    // [token group, head 2 pair], [token group, head 2 pair + 1],
    // [token group + 8, head 2 pair], [token group + 8, head 2 pair + 1].
    float score[4];
#pragma unroll
    for (uint32_t element = 0; element < 4; ++element) {
      const bool present = (element < 2 ? group : group + 8) < token_count;
      score[element] = present ? staged_attention_score(dot[element], scale)
                               : -CUDART_INF_F;
    }
    maximum0 = fmaxf(score[0], score[2]);
    maximum1 = fmaxf(score[1], score[3]);
#pragma unroll
    for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) {
      maximum0 =
          fmaxf(maximum0, __shfl_xor_sync(kFullWarpMask, maximum0, offset));
      maximum1 =
          fmaxf(maximum1, __shfl_xor_sync(kFullWarpMask, maximum1, offset));
    }
    float weight[4];
    weight[0] = expf(score[0] - maximum0);
    weight[1] = expf(score[1] - maximum1);
    weight[2] = expf(score[2] - maximum0);
    weight[3] = expf(score[3] - maximum1);
    denominator0 = weight[0] + weight[2];
    denominator1 = weight[1] + weight[3];
#pragma unroll
    for (uint32_t offset = 4; offset < kWarpSize; offset *= 2) {
      denominator0 += __shfl_xor_sync(kFullWarpMask, denominator0, offset);
      denominator1 += __shfl_xor_sync(kFullWarpMask, denominator1, offset);
    }
    // Weight operands B[k = token][n = head group], one per BF16 piece: the
    // operand of lane `(group, pair)` is head `group` at tokens
    // `2 * pair (+1, +8, +9)`, whose weights sit in lanes
    // `(2 * pair, group / 2)` and `(2 * pair + 1, group / 2)`. Each
    // subtraction is exact, and the third piece holds what is left.
    const uint32_t source_even = 8 * pair + group / 2;
    const uint32_t source_odd = source_even + 4;
    const uint32_t source_shift = 16 * (group % 2);
    uint32_t weight_operand[3][2];
#pragma unroll
    for (uint32_t piece = 0; piece < 3; ++piece) {
      float rounded[4];
#pragma unroll
      for (uint32_t element = 0; element < 4; ++element) {
        rounded[element] =
            __bfloat162float(__float2bfloat16_rn(weight[element]));
        weight[element] -= rounded[element];
      }
      const uint32_t low_tokens = pack_bf16_pair(rounded[0], rounded[1]);
      const uint32_t high_tokens = pack_bf16_pair(rounded[2], rounded[3]);
      const uint32_t even_low =
          __shfl_sync(kFullWarpMask, low_tokens, source_even);
      const uint32_t even_high =
          __shfl_sync(kFullWarpMask, high_tokens, source_even);
      const uint32_t odd_low =
          __shfl_sync(kFullWarpMask, low_tokens, source_odd);
      const uint32_t odd_high =
          __shfl_sync(kFullWarpMask, high_tokens, source_odd);
      weight_operand[piece][0] =
          ((even_low >> source_shift) & 0xffffU) |
          (((odd_low >> source_shift) & 0xffffU) << 16);
      weight_operand[piece][1] =
          ((even_high >> source_shift) & 0xffffU) |
          (((odd_high >> source_shift) & 0xffffU) << 16);
    }
    // Value operand A[m = depth][k = token]: depths `16 step + group (+8)`;
    // tokens past the page's valid count are staged as zero.
    const uint16_t* const value_token =
        reinterpret_cast<const uint16_t*>(value_pool + block_head_base +
                                          2 * pair * kHead + group);
    const bool present0 = 2 * pair < token_count;
    const bool present1 = 2 * pair + 1 < token_count;
    const bool present2 = 2 * pair + 8 < token_count;
    const bool present3 = 2 * pair + 9 < token_count;
#pragma unroll
    for (uint32_t step = 0; step < 8; ++step) {
      const uint16_t* const token0 = value_token + 16 * step;
      const uint16_t* const token1 = token0 + kHead;
      const uint16_t* const token2 = token0 + 8 * kHead;
      const uint16_t* const token3 = token2 + kHead;
      const uint32_t value0 =
          (present0 ? static_cast<uint32_t>(token0[0]) : 0U) |
          ((present1 ? static_cast<uint32_t>(token1[0]) : 0U) << 16);
      const uint32_t value1 =
          (present0 ? static_cast<uint32_t>(token0[8]) : 0U) |
          ((present1 ? static_cast<uint32_t>(token1[8]) : 0U) << 16);
      const uint32_t value2 =
          (present2 ? static_cast<uint32_t>(token2[0]) : 0U) |
          ((present3 ? static_cast<uint32_t>(token3[0]) : 0U) << 16);
      const uint32_t value3 =
          (present2 ? static_cast<uint32_t>(token2[8]) : 0U) |
          ((present3 ? static_cast<uint32_t>(token3[8]) : 0U) << 16);
#pragma unroll
      for (uint32_t piece = 0; piece < 3; ++piece) {
        mma_m16n8k16_bf16(numerator[step], value0, value1, value2, value3,
                          weight_operand[piece][0], weight_operand[piece][1]);
      }
    }
  }
  if (group == 0) {
    state0[0] = maximum0;
    state0[1] = denominator0;
    state1[0] = maximum1;
    state1[1] = denominator1;
  }
#pragma unroll
  for (uint32_t step = 0; step < 8; ++step) {
    state0[2 + 16 * step + group] = numerator[step][0];
    state1[2 + 16 * step + group] = numerator[step][1];
    state0[2 + 16 * step + group + 8] = numerator[step][2];
    state1[2 + 16 * step + group + 8] = numerator[step][3];
  }
}

// Shared-KV launch geometry for the D128 partial-state producer. One CTA owns
// a `(logical_block, key_value_head, row)` page and has one warp per query
// head of that KV head. The page's immutable BF16 K/V tile is read from global
// memory once and staged in shared memory with its exact storage bits; every
// warp then executes the same per-head score reduction, staged rounding,
// online token order, and numerator updates as the one-warp producer above,
// and writes the identical `[m, l, n[128]]` state.
constexpr uint32_t kNativeBf16RaggedPagedSplitGqaD128SharedKvWarps =
    static_cast<uint32_t>(kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount /
                          kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount);
constexpr uint32_t kNativeBf16RaggedPagedSplitGqaD128SharedKvThreads =
    kNativeBf16RaggedPagedSplitGqaD128SharedKvWarps * kWarpSize;
constexpr uint32_t kNativeBf16RaggedPagedSplitGqaD128TileElements =
    static_cast<uint32_t>(RILEY_CUDA_PAGED_KV_BLOCK_SIZE *
                          kNativeBf16RaggedPagedSplitGqaD128HeadSize);

__global__ __launch_bounds__(kNativeBf16RaggedPagedSplitGqaD128SharedKvThreads)
void native_bf16_ragged_paged_split_gqa_d128_shared_kv_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, float* partial_states,
    DeviceBatch batch, uint64_t partial_state_capacity, float scale,
    const uint32_t* row_kinds) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;
  const uint64_t logical_block = blockIdx.x;
  const uint64_t key_value_head = blockIdx.y;
  const uint64_t row = blockIdx.z;
  // Every exit before the barrier below depends only on CTA-uniform values.
  if (row >= active_rows(batch) ||
      key_value_head >= kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount) {
    return;
  }
  // Prompt rows are produced by the tensor-core prefill kernel instead.
  if (row_kinds != nullptr &&
      row_kinds[row] == kNativeBf16RaggedD128PromptRowKind) {
    return;
  }
  const uint64_t logical_token_count =
      static_cast<uint64_t>(batch.row_positions[row]) + 1;
  const uint64_t partial_state_count =
      ((logical_token_count - 1) / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) + 1;
  if (logical_block >= partial_state_count ||
      logical_block >= partial_state_capacity) {
    return;
  }

  const uint64_t query_head =
      key_value_head * kNativeBf16RaggedPagedSplitGqaD128SharedKvWarps + warp;
  const uint64_t query_base =
      (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head) *
      kNativeBf16RaggedPagedSplitGqaD128HeadSize;
  uint64_t block_head_base = 0;
  uint64_t token_count = 0;
  const bool valid_block = resolve_ragged_partial_block(
      batch, row, logical_block, key_value_head, &block_head_base,
      &token_count);

  // The tiles are staged already widened to FP32 (an exact conversion), so
  // the per-token loop reads them without converting again in every warp.
  __shared__ float shared_keys[kNativeBf16RaggedPagedSplitGqaD128TileElements];
  __shared__ float
      shared_values[kNativeBf16RaggedPagedSplitGqaD128TileElements];
  if (valid_block) {
    const uint32_t tile_elements = static_cast<uint32_t>(
        token_count * kNativeBf16RaggedPagedSplitGqaD128HeadSize);
    for (uint32_t element = threadIdx.x; element < tile_elements;
         element += kNativeBf16RaggedPagedSplitGqaD128SharedKvThreads) {
      shared_keys[element] =
          __bfloat162float(key_pool[block_head_base + element]);
      shared_values[element] =
          __bfloat162float(value_pool[block_head_base + element]);
    }
  }
  __syncthreads();

  const float query0 = __bfloat162float(query[query_base + lane]);
  const float query1 =
      __bfloat162float(query[query_base + lane + kWarpSize]);
  const float query2 =
      __bfloat162float(query[query_base + lane + 2 * kWarpSize]);
  const float query3 =
      __bfloat162float(query[query_base + lane + 3 * kWarpSize]);
  float maximum = valid_block ? -CUDART_INF_F : CUDART_NAN_F;
  float denominator = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator0 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator1 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator2 = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator3 = valid_block ? 0.0F : CUDART_NAN_F;
  if (valid_block) {
    for (uint64_t token_in_block = 0; token_in_block < token_count;
         ++token_in_block) {
      const uint32_t tile_base = static_cast<uint32_t>(
          token_in_block * kNativeBf16RaggedPagedSplitGqaD128HeadSize);
      const float key0 = shared_keys[tile_base + lane];
      const float key1 = shared_keys[tile_base + lane + kWarpSize];
      const float key2 = shared_keys[tile_base + lane + 2 * kWarpSize];
      const float key3 = shared_keys[tile_base + lane + 3 * kWarpSize];
      float score = fmaf(query0, key0, query1 * key1);
      score = fmaf(query2, key2, score);
      score = fmaf(query3, key3, score);
      // An xor butterfly adds, at every level, the same two subtree sums the
      // shfl_down tree adds on lane 0, so every lane ends with lane 0's bits.
      // Each lane then runs the scalar online update itself and no score,
      // alpha, or beta broadcast is needed.
      score = staged_attention_score(warp_sum_all_lanes(score), scale);

      float alpha = 0.0F;
      float beta = 0.0F;
      update_online_state(score, &maximum, &denominator, &alpha, &beta);
      const float value0 = shared_values[tile_base + lane];
      const float value1 = shared_values[tile_base + lane + kWarpSize];
      const float value2 = shared_values[tile_base + lane + 2 * kWarpSize];
      const float value3 = shared_values[tile_base + lane + 3 * kWarpSize];
      // update_numerator's cases, selected once per token: alpha and beta are
      // uniform across the warp.
      if (beta == 0.0F) {
        numerator0 = alpha * numerator0;
        numerator1 = alpha * numerator1;
        numerator2 = alpha * numerator2;
        numerator3 = alpha * numerator3;
      } else if (alpha == 0.0F) {
        numerator0 = beta * value0;
        numerator1 = beta * value1;
        numerator2 = beta * value2;
        numerator3 = beta * value3;
      } else {
        numerator0 = fmaf(beta, value0, alpha * numerator0);
        numerator1 = fmaf(beta, value1, alpha * numerator1);
        numerator2 = fmaf(beta, value2, alpha * numerator2);
        numerator3 = fmaf(beta, value3, alpha * numerator3);
      }
    }
  }
  const uint64_t state_base =
      (((row * partial_state_capacity + logical_block) *
            kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
        query_head) *
       kNativeBf16RaggedPagedSplitGqaD128StateStride);
  if (lane == 0) {
    partial_states[state_base] = maximum;
    partial_states[state_base + 1] = denominator;
  }
  partial_states[state_base + 2 + lane] = numerator0;
  partial_states[state_base + 2 + lane + kWarpSize] = numerator1;
  partial_states[state_base + 2 + lane + 2 * kWarpSize] = numerator2;
  partial_states[state_base + 2 + lane + 3 * kWarpSize] = numerator3;
}

__device__ __forceinline__ NativeBf16RaggedPartialTransitionV2
merge_ragged_partial_metadata_v2(float other_maximum, float other_denominator,
                                 float* maximum, float* denominator) {
  if (other_denominator == 0.0F) {
    return NativeBf16RaggedPartialTransitionV2{1.0F, 0.0F};
  }
  if (*denominator == 0.0F) {
    *maximum = other_maximum;
    *denominator = other_denominator;
    return NativeBf16RaggedPartialTransitionV2{0.0F, 1.0F};
  }
  if (isnan(*maximum) || isnan(other_maximum) || isnan(*denominator) ||
      isnan(other_denominator)) {
    *maximum = CUDART_NAN_F;
    *denominator = CUDART_NAN_F;
    return NativeBf16RaggedPartialTransitionV2{CUDART_NAN_F, CUDART_NAN_F};
  }
  const bool left_positive_infinity = isinf(*maximum) && *maximum > 0.0F;
  const bool right_positive_infinity =
      isinf(other_maximum) && other_maximum > 0.0F;
  if (left_positive_infinity || right_positive_infinity) {
    if (left_positive_infinity && right_positive_infinity) {
      *denominator += other_denominator;
      return NativeBf16RaggedPartialTransitionV2{1.0F, 1.0F};
    }
    if (right_positive_infinity) {
      *maximum = other_maximum;
      *denominator = other_denominator;
      return NativeBf16RaggedPartialTransitionV2{0.0F, 1.0F};
    }
    return NativeBf16RaggedPartialTransitionV2{1.0F, 0.0F};
  }
  const float next_maximum = fmaxf(*maximum, other_maximum);
  const float left_scale = expf(*maximum - next_maximum);
  const float right_scale = expf(other_maximum - next_maximum);
  *denominator = fmaf(left_scale, *denominator,
                      right_scale * other_denominator);
  *maximum = next_maximum;
  return NativeBf16RaggedPartialTransitionV2{left_scale, right_scale};
}

__device__ __forceinline__ float apply_ragged_partial_transition_v2(
    float numerator, float other_numerator,
    NativeBf16RaggedPartialTransitionV2 transition,
    bool both_positive_infinity) {
  if (transition.right_scale == 0.0F) {
    return transition.left_scale == 1.0F
               ? numerator
               : transition.left_scale * numerator;
  }
  if (transition.left_scale == 0.0F) {
    return transition.right_scale == 1.0F
               ? other_numerator
               : transition.right_scale * other_numerator;
  }
  if (both_positive_infinity) {
    return numerator + other_numerator;
  }
  return fmaf(transition.right_scale, other_numerator,
              transition.left_scale * numerator);
}

__global__ __launch_bounds__(kWarpSize) void
native_bf16_ragged_paged_split_gqa_d128_transition_kernel(
    const float* partial_states, float* reduction_steps,
    float* reduction_normalizers, const uint32_t* row_positions,
    uint64_t partial_state_capacity, uint64_t launch_partial_state_count,
    uint32_t reduction_order) {
  if (threadIdx.x != 0) {
    return;
  }
  const uint64_t query_head = blockIdx.x;
  const uint64_t row = blockIdx.y;
  const uint64_t logical_token_count =
      static_cast<uint64_t>(row_positions[row]) + 1;
  const uint64_t partial_state_count =
      ((logical_token_count - 1) / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) + 1;
  const uint64_t normalizer_index =
      row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head;
  if (partial_state_count > partial_state_capacity ||
      partial_state_count > launch_partial_state_count) {
    // Raw C callers cannot make the reducer read beyond its declared
    // per-row workspace or stale producer tails. The output stage recognizes
    // this normalizer as an invalid row and writes qNaN without consuming
    // transition tails.
    reduction_normalizers[normalizer_index] = CUDART_NAN_F;
    return;
  }
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  const uint64_t pair_base =
      row * partial_state_capacity *
      kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head;
  uint64_t ordinal = 0;
  // The scalar merge chain is strictly ordered, but its inputs are not. Issue
  // the next four `(m, l)` loads together so their memory latency overlaps;
  // the merges below still run one partition at a time in the same order.
  for (; ordinal + kNativeBf16RaggedD128ReplayLookahead <= partial_state_count;
       ordinal += kNativeBf16RaggedD128ReplayLookahead) {
    float other_maximum[kNativeBf16RaggedD128ReplayLookahead];
    float other_denominator[kNativeBf16RaggedD128ReplayLookahead];
    uint64_t pair_index[kNativeBf16RaggedD128ReplayLookahead];
#pragma unroll
    for (uint32_t ahead = 0; ahead < kNativeBf16RaggedD128ReplayLookahead;
         ++ahead) {
      const uint64_t partition =
          reduction_order == RILEY_CUDA_DECODE_REDUCTION_ASCENDING
              ? ordinal + ahead
              : partial_state_count - 1 - (ordinal + ahead);
      pair_index[ahead] =
          pair_base +
          partition * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount;
      const uint64_t state_base =
          pair_index[ahead] * kNativeBf16RaggedPagedSplitGqaD128StateStride;
      other_maximum[ahead] = partial_states[state_base];
      other_denominator[ahead] = partial_states[state_base + 1];
    }
#pragma unroll
    for (uint32_t ahead = 0; ahead < kNativeBf16RaggedD128ReplayLookahead;
         ++ahead) {
      const NativeBf16RaggedPartialTransitionV2 transition =
          merge_ragged_partial_metadata_v2(
              other_maximum[ahead], other_denominator[ahead], &maximum,
              &denominator);
      const uint64_t transition_base =
          pair_index[ahead] *
          kNativeBf16RaggedPagedSplitGqaD128TransitionWidth;
      reduction_steps[transition_base] = transition.left_scale;
      reduction_steps[transition_base + 1] = transition.right_scale;
    }
  }
  for (; ordinal < partial_state_count; ++ordinal) {
    const uint64_t partition =
        reduction_order == RILEY_CUDA_DECODE_REDUCTION_ASCENDING
            ? ordinal
            : partial_state_count - 1 - ordinal;
    const uint64_t state_base =
        (((row * partial_state_capacity + partition) *
              kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
          query_head) *
         kNativeBf16RaggedPagedSplitGqaD128StateStride);
    const NativeBf16RaggedPartialTransitionV2 transition =
        merge_ragged_partial_metadata_v2(
            partial_states[state_base], partial_states[state_base + 1],
            &maximum, &denominator);
    const uint64_t transition_base =
        (((row * partial_state_capacity + partition) *
              kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
          query_head) *
         kNativeBf16RaggedPagedSplitGqaD128TransitionWidth);
    reduction_steps[transition_base] = transition.left_scale;
    reduction_steps[transition_base + 1] = transition.right_scale;
  }
  reduction_normalizers[normalizer_index] = denominator;
}

__global__ __launch_bounds__(
    kNativeBf16RaggedPagedSplitGqaD128ReductionThreads) void
native_bf16_ragged_paged_split_gqa_d128_transition_reduce_kernel(
    const float* partial_states, const float* reduction_steps,
    const float* reduction_normalizers, const uint32_t* row_positions,
    __nv_bfloat16* output, uint64_t active_row_count,
    uint64_t partial_state_capacity, uint64_t launch_partial_state_count,
    uint32_t reduction_order) {
  const uint64_t query_head = blockIdx.x;
  const uint64_t row = blockIdx.y;
  const uint64_t depth = threadIdx.x;
  if (depth >= kNativeBf16RaggedPagedSplitGqaD128HeadSize) {
    return;
  }
  const uint64_t output_base =
      (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head) *
      kNativeBf16RaggedPagedSplitGqaD128HeadSize;
  if (row >= active_row_count) {
    output[output_base + depth] = __float2bfloat16_rn(0.0F);
    return;
  }
  const uint64_t logical_token_count =
      static_cast<uint64_t>(row_positions[row]) + 1;
  const uint64_t partial_state_count =
      ((logical_token_count - 1) / RILEY_CUDA_PAGED_KV_BLOCK_SIZE) + 1;
  if (partial_state_count > partial_state_capacity ||
      partial_state_count > launch_partial_state_count) {
    output[output_base + depth] = __float2bfloat16_rn(CUDART_NAN_F);
    return;
  }
  float numerator = 0.0F;
  const uint64_t pair_base =
      row * partial_state_capacity *
      kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head;
  uint64_t ordinal = 0;
  bool poisoned = false;
  // Same replay order as the scalar loop below; only the loads of the next
  // four partitions are issued together. The `+Inf` probe reads the state
  // maximum unconditionally (always in bounds) instead of on demand.
  for (; !poisoned &&
         ordinal + kNativeBf16RaggedD128ReplayLookahead <= partial_state_count;
       ordinal += kNativeBf16RaggedD128ReplayLookahead) {
    float left_scale[kNativeBf16RaggedD128ReplayLookahead];
    float right_scale[kNativeBf16RaggedD128ReplayLookahead];
    float other_numerator[kNativeBf16RaggedD128ReplayLookahead];
    float other_maximum[kNativeBf16RaggedD128ReplayLookahead];
#pragma unroll
    for (uint32_t ahead = 0; ahead < kNativeBf16RaggedD128ReplayLookahead;
         ++ahead) {
      const uint64_t partition =
          reduction_order == RILEY_CUDA_DECODE_REDUCTION_ASCENDING
              ? ordinal + ahead
              : partial_state_count - 1 - (ordinal + ahead);
      const uint64_t pair_index =
          pair_base +
          partition * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount;
      const uint64_t state_base =
          pair_index * kNativeBf16RaggedPagedSplitGqaD128StateStride;
      const uint64_t transition_base =
          pair_index * kNativeBf16RaggedPagedSplitGqaD128TransitionWidth;
      left_scale[ahead] = reduction_steps[transition_base];
      right_scale[ahead] = reduction_steps[transition_base + 1];
      other_numerator[ahead] = partial_states[state_base + 2 + depth];
      other_maximum[ahead] = partial_states[state_base];
    }
#pragma unroll
    for (uint32_t ahead = 0; ahead < kNativeBf16RaggedD128ReplayLookahead;
         ++ahead) {
      if (poisoned) {
        continue;
      }
      const NativeBf16RaggedPartialTransitionV2 transition{
          left_scale[ahead], right_scale[ahead]};
      if (isnan(transition.left_scale) || isnan(transition.right_scale)) {
        numerator = CUDART_NAN_F;
        poisoned = true;
        continue;
      }
      const bool both_positive_infinity =
          transition.left_scale == 1.0F && transition.right_scale == 1.0F &&
          isinf(other_maximum[ahead]) && other_maximum[ahead] > 0.0F;
      numerator = apply_ragged_partial_transition_v2(
          numerator, other_numerator[ahead], transition,
          both_positive_infinity);
    }
  }
  for (; !poisoned && ordinal < partial_state_count; ++ordinal) {
    const uint64_t partition =
        reduction_order == RILEY_CUDA_DECODE_REDUCTION_ASCENDING
            ? ordinal
            : partial_state_count - 1 - ordinal;
    const uint64_t state_base =
        (((row * partial_state_capacity + partition) *
              kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
          query_head) *
         kNativeBf16RaggedPagedSplitGqaD128StateStride);
    const uint64_t transition_base =
        (((row * partial_state_capacity + partition) *
              kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
          query_head) *
         kNativeBf16RaggedPagedSplitGqaD128TransitionWidth);
    const NativeBf16RaggedPartialTransitionV2 transition{
        reduction_steps[transition_base], reduction_steps[transition_base + 1]};
    if (isnan(transition.left_scale) || isnan(transition.right_scale)) {
      numerator = CUDART_NAN_F;
      break;
    }
    const bool both_positive_infinity =
        transition.left_scale == 1.0F && transition.right_scale == 1.0F &&
        isinf(partial_states[state_base]) && partial_states[state_base] > 0.0F;
    numerator = apply_ragged_partial_transition_v2(
        numerator, partial_states[state_base + 2 + depth], transition,
        both_positive_infinity);
  }
  const float denominator = reduction_normalizers[
      row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head];
  const float inverse_denominator =
      isnan(denominator)
          ? CUDART_NAN_F
          : (denominator > 0.0F ? 1.0F / denominator : 0.0F);
  output[output_base + depth] = __float2bfloat16_rn(numerator * inverse_denominator);
}

// Global-max merge of the per-page partial states: one kernel in place of the
// ordered transition + replay pair. For each `(row, query_head)` it takes the
// maximum page maximum `M`, weights every page by `exp(m_p - M)`, and divides
// the weighted numerator sum by the weighted denominator sum. The denominator
// is reduced with a fixed stride-128 fold and adjacent tree; each output depth
// folds pages in ascending order. The result depends only on the row's own pages,
// so it is deterministic and independent of the batch. It is mathematically
// the same softmax merge as the ordered pair but rounds differently, so it is
// a separately selected numerical mode. Non-finite page maxima (other than
// the `-Inf` of an empty page) yield NaN outputs.
constexpr uint32_t kNativeBf16RaggedD128GlobalMergeThreads = 128;
constexpr uint64_t kNativeBf16RaggedD128GlobalMergeMaximumPages = 128;

__device__ __forceinline__ float native_d128_nan_propagating_max(float left,
                                                                 float right) {
  return (isnan(left) || isnan(right)) ? CUDART_NAN_F : fmaxf(left, right);
}

__global__ __launch_bounds__(kNativeBf16RaggedD128GlobalMergeThreads) void
native_bf16_ragged_paged_split_gqa_d128_global_merge_kernel(
    const float* partial_states, const uint32_t* row_positions,
    __nv_bfloat16* output, uint64_t active_row_count,
    uint64_t partial_state_capacity, uint64_t launch_partial_state_count,
    const uint32_t* dynamic_counts, const uint32_t* row_kinds) {
  __shared__ float page_weights[kNativeBf16RaggedD128GlobalMergeMaximumPages];
  if (dynamic_counts != nullptr) {
    active_row_count = dynamic_counts[0];
  }
  __shared__ float reduction[kNativeBf16RaggedD128GlobalMergeThreads];
  const uint64_t query_head = blockIdx.x;
  const uint64_t row = blockIdx.y;
  const uint32_t depth = threadIdx.x;
  const uint64_t output_base =
      (row * kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount + query_head) *
      kNativeBf16RaggedPagedSplitGqaD128HeadSize;
  // Every early exit below depends only on CTA-uniform values.
  if (row >= active_row_count) {
    output[output_base + depth] = __float2bfloat16_rn(0.0F);
    return;
  }
  const uint64_t logical_page_count =
      static_cast<uint64_t>(row_positions[row]) /
          RILEY_CUDA_PAGED_KV_BLOCK_SIZE + 1;
  // A prompt row holds one state per four-page segment.
  const uint64_t page_count =
      (row_kinds != nullptr &&
       row_kinds[row] == kNativeBf16RaggedD128PromptRowKind)
          ? (logical_page_count + kNativeBf16PrefillD128MmaSegmentPages - 1) /
                kNativeBf16PrefillD128MmaSegmentPages
          : logical_page_count;
  if (logical_page_count > partial_state_capacity ||
      logical_page_count > launch_partial_state_count) {
    output[output_base + depth] = __float2bfloat16_rn(CUDART_NAN_F);
    return;
  }
  const uint64_t page_stride =
      kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount *
      kNativeBf16RaggedPagedSplitGqaD128StateStride;
  const float* const states =
      partial_states +
      (row * partial_state_capacity *
           kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount +
       query_head) *
          kNativeBf16RaggedPagedSplitGqaD128StateStride;

  // Thread `depth` folds pages `depth, depth + 128, ...` in ascending order;
  // with at most 128 pages each thread owns at most one page.
  float local_maximum = -CUDART_INF_F;
  for (uint64_t page = depth; page < page_count;
       page += kNativeBf16RaggedD128GlobalMergeThreads) {
    local_maximum = native_d128_nan_propagating_max(
        local_maximum, states[page * page_stride]);
  }
  reduction[depth] = local_maximum;
  __syncthreads();
  for (uint32_t offset = kNativeBf16RaggedD128GlobalMergeThreads / 2;
       offset != 0; offset /= 2) {
    if (depth < offset) {
      reduction[depth] = native_d128_nan_propagating_max(
          reduction[depth], reduction[depth + offset]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  __syncthreads();
  float local_denominator = 0.0F;
  for (uint64_t page = depth; page < page_count;
       page += kNativeBf16RaggedD128GlobalMergeThreads) {
    const float weight = expf(states[page * page_stride] - maximum);
    if (page < kNativeBf16RaggedD128GlobalMergeMaximumPages) {
      page_weights[page] = weight;
    }
    local_denominator =
        fmaf(weight, states[page * page_stride + 1], local_denominator);
  }
  reduction[depth] = local_denominator;
  __syncthreads();
  for (uint32_t offset = kNativeBf16RaggedD128GlobalMergeThreads / 2;
       offset != 0; offset /= 2) {
    if (depth < offset) {
      reduction[depth] += reduction[depth + offset];
    }
    __syncthreads();
  }
  const float denominator = reduction[0];
  const float inverse_denominator =
      isnan(denominator)
          ? CUDART_NAN_F
          : (denominator > 0.0F ? 1.0F / denominator : 0.0F);

  const float* const numerators = states + 2 + depth;
  float numerator = 0.0F;
  uint64_t page = 0;
  const uint64_t cached_pages =
      page_count < kNativeBf16RaggedD128GlobalMergeMaximumPages
          ? page_count
          : kNativeBf16RaggedD128GlobalMergeMaximumPages;
  // Ascending page order; the next four inputs are loaded together so their
  // memory latency overlaps.
  for (; page + 4 <= cached_pages; page += 4) {
    const float value0 = numerators[page * page_stride];
    const float value1 = numerators[(page + 1) * page_stride];
    const float value2 = numerators[(page + 2) * page_stride];
    const float value3 = numerators[(page + 3) * page_stride];
    numerator = fmaf(page_weights[page], value0, numerator);
    numerator = fmaf(page_weights[page + 1], value1, numerator);
    numerator = fmaf(page_weights[page + 2], value2, numerator);
    numerator = fmaf(page_weights[page + 3], value3, numerator);
  }
  for (; page < cached_pages; ++page) {
    numerator =
        fmaf(page_weights[page], numerators[page * page_stride], numerator);
  }
  // Pages beyond the cached weights recompute the identical weight value.
  for (; page < page_count; ++page) {
    numerator = fmaf(expf(states[page * page_stride] - maximum),
                     numerators[page * page_stride], numerator);
  }
  output[output_base + depth] =
      __float2bfloat16_rn(numerator * inverse_denominator);
}

struct NativeBf16RaggedPagedSplitGqaD128ByteCounts {
  uint64_t query_bytes;
  uint64_t pool_bytes;
  uint64_t partial_state_bytes;
  uint64_t reduction_step_bytes;
  uint64_t reduction_normalizer_bytes;
  uint64_t output_bytes;
};

RileyCudaStatus native_bf16_ragged_paged_split_gqa_d128_byte_counts(
    const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3& params,
    NativeBf16RaggedPagedSplitGqaD128ByteCounts* output,
    RileyCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                          "native ragged D128 byte-count output is null");
  }
  uint64_t query_elements = 0;
  uint64_t pool_elements = 0;
  uint64_t partial_state_elements = 0;
  uint64_t reduction_step_elements = 0;
  uint64_t reduction_normalizer_elements = 0;
  uint64_t output_elements = 0;
  if (!checked_product3(params.batch.active_row_count,
                        params.query_head_count, params.head_size,
                        &query_elements) ||
      !checked_product4(params.batch.physical_block_count,
                        params.key_value_head_count,
                        RILEY_CUDA_PAGED_KV_BLOCK_SIZE, params.head_size,
                        &pool_elements) ||
      !checked_product4(params.output_row_count, params.partial_state_capacity,
                        params.query_head_count,
                        kNativeBf16RaggedPagedSplitGqaD128StateStride,
                        &partial_state_elements) ||
      !checked_product4(params.output_row_count, params.partial_state_capacity,
                        params.query_head_count,
                        kNativeBf16RaggedPagedSplitGqaD128TransitionWidth,
                        &reduction_step_elements) ||
      !checked_multiply(params.output_row_count, params.query_head_count,
                        &reduction_normalizer_elements) ||
      !checked_product3(params.output_row_count, params.query_head_count,
                        params.head_size, &output_elements)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "native ragged D128 tensor shape overflows uint64_t");
  }
  RileyCudaStatus status = typed_bytes(query_elements, RILEY_CUDA_DTYPE_BF16,
                                       &output->query_bytes, error, operation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RILEY_CUDA_DTYPE_BF16,
                         &output->pool_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(partial_state_elements, RILEY_CUDA_DTYPE_F32,
                         &output->partial_state_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(reduction_step_elements, RILEY_CUDA_DTYPE_F32,
                         &output->reduction_step_bytes, error, operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(reduction_normalizer_elements, RILEY_CUDA_DTYPE_F32,
                         &output->reduction_normalizer_bytes, error,
                         operation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, RILEY_CUDA_DTYPE_BF16,
                         &output->output_bytes, error, operation);
  }
  return status;
}

}  // namespace

extern "C" RileyCudaStatus
riley_cuda_native_bf16_ragged_paged_split_gqa_d128_two_stage_execute(
    const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute native BF16 ragged paged split-GQA D128 two-stage V2";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV2 stable_params = *params;
  if (stable_params.format_version !=
          RILEY_CUDA_NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_V2_VERSION ||
      !reserved_is_zero(stable_params.reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 V2 version or reserved fields are invalid");
  }

  // Preserve the published V2 static-P producer-grid contract by adapting it
  // to the additive V3 descriptor with L=P. V3 performs the shared geometry,
  // span, ownership, and launch validation without ever reading beyond V2.
  const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3 v3_params{
      static_cast<uint32_t>(sizeof(RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3)),
      RILEY_CUDA_NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_V3_VERSION,
      stable_params.query,
      stable_params.key_pool,
      stable_params.value_pool,
      stable_params.partial_states,
      stable_params.reduction_steps,
      stable_params.reduction_normalizers,
      stable_params.output,
      stable_params.batch,
      stable_params.query_head_count,
      stable_params.key_value_head_count,
      stable_params.head_size,
      stable_params.output_row_count,
      stable_params.partial_state_capacity,
      stable_params.partial_state_capacity,
      stable_params.scale,
      stable_params.reduction_order,
      {0, 0, 0, 0}};
  return riley_cuda_native_bf16_ragged_paged_split_gqa_d128_two_stage_v3_execute(
      &v3_params, stream, error);
}

extern "C" RileyCudaStatus
riley_cuda_native_bf16_ragged_paged_split_gqa_d128_two_stage_v3_execute(
    const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute native BF16 ragged paged split-GQA D128 two-stage";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaNativeBf16RaggedPagedSplitGqaParamsV3 stable_params = *params;
  params = &stable_params;
  if (params->format_version !=
          RILEY_CUDA_NATIVE_BF16_RAGGED_PAGED_SPLIT_GQA_V3_VERSION ||
      params->reserved[0] > 2 || params->reserved[1] > 1 ||
      (params->reserved[0] == 2 && params->reserved[1] != 1) ||
      (params->reserved[2] != 0 &&
       (params->reserved[0] != 1 || params->reserved[1] != 1)) ||
      params->reserved[3] != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 V3 version or reserved fields are invalid");
  }
  RileyCudaStatus status = validate_packed_batch(params->batch, error,
                                                  kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->query_head_count !=
          kNativeBf16RaggedPagedSplitGqaD128QueryHeadCount ||
      params->key_value_head_count !=
          kNativeBf16RaggedPagedSplitGqaD128KeyValueHeadCount ||
      params->head_size != kNativeBf16RaggedPagedSplitGqaD128HeadSize) {
    return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 V3 requires QH=16, KVH=2, and D=128");
  }
  if (params->partial_state_capacity == 0 ||
      params->launch_partial_state_count == 0 ||
      params->launch_partial_state_count > params->partial_state_capacity ||
      params->output_row_count < params->batch.active_row_count) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 workspace capacity, launch extent, or output rows are invalid");
  }
  if (params->launch_partial_state_count > kMaximumGridX ||
      params->partial_state_capacity > kMaximumGridX ||
      params->output_row_count > kMaximumGridYOrZ ||
      params->batch.active_row_count > kMaximumGridYOrZ) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 launch dimensions exceed CUDA grid limits");
  }
  if (!std::isfinite(params->scale) || params->scale <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 scale must be finite and positive");
  }
  if (params->reduction_order != RILEY_CUDA_DECODE_REDUCTION_ASCENDING &&
      params->reduction_order != RILEY_CUDA_DECODE_REDUCTION_DESCENDING) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 reduction order is unsupported");
  }
  if (params->reserved[1] == 1 &&
      params->reduction_order != RILEY_CUDA_DECODE_REDUCTION_ASCENDING) {
    return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 global merge requires ascending order");
  }

  NativeBf16RaggedPagedSplitGqaD128ByteCounts bytes{};
  status = native_bf16_ragged_paged_split_gqa_d128_byte_counts(
      *params, &bytes, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan partial_states{};
  ResolvedSpan reduction_steps{};
  ResolvedSpan reduction_normalizers{};
  ResolvedSpan output{};
  ResolvedBatch batch{};
  status = resolve_span(params->query, RILEY_CUDA_DTYPE_BF16,
                        bytes.query_bytes, &query, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RILEY_CUDA_DTYPE_BF16,
                          bytes.pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RILEY_CUDA_DTYPE_BF16,
                          bytes.pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->partial_states, RILEY_CUDA_DTYPE_F32,
                          bytes.partial_state_bytes, &partial_states, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->reduction_steps, RILEY_CUDA_DTYPE_F32,
                          bytes.reduction_step_bytes, &reduction_steps, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->reduction_normalizers, RILEY_CUDA_DTYPE_F32,
                          bytes.reduction_normalizer_bytes,
                          &reduction_normalizers, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RILEY_CUDA_DTYPE_BF16,
                          bytes.output_bytes, &output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan inputs[] = {
      query, key_pool, value_pool, batch.sequence_block_offsets,
      batch.block_ids, batch.valid_tokens, batch.row_sequence_slots,
      batch.row_positions};
  const ResolvedSpan writable[] = {
      partial_states, reduction_steps, reduction_normalizers, output};
  for (size_t writable_index = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && writable_index < 4;
       ++writable_index) {
    for (size_t input_index = 0; input_index < 8; ++input_index) {
      status = reject_overlap(writable[writable_index], inputs[input_index],
                              false, error, kOperation);
      if (status != RILEY_CUDA_STATUS_SUCCESS) {
        break;
      }
    }
    for (size_t prior_index = 0;
         status == RILEY_CUDA_STATUS_SUCCESS && prior_index < writable_index;
         ++prior_index) {
      status = reject_overlap(writable[writable_index],
                              writable[prior_index], false, error,
                              kOperation);
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {
      query,
      key_pool,
      value_pool,
      partial_states,
      reduction_steps,
      reduction_normalizers,
      output,
      batch.sequence_block_offsets,
      batch.block_ids,
      batch.valid_tokens,
      batch.row_sequence_slots,
      batch.row_positions};
  status = validate_contexts(stream, spans, 12, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (riley_cuda_internal::thread_has_active_graph_capture()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "native ragged D128 V3 is eager-only and cannot run during graph capture");
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 12; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "native ragged D128 buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  // A recorded step launches the producer over the full per-row capacity:
  // pages beyond a row's own count exit immediately, and the extent then does
  // not depend on this step's longest row.
  const uint64_t launch_extent =
      riley_cuda_internal::thread_has_active_step_capture()
          ? params->partial_state_capacity
          : params->launch_partial_state_count;
  // A dynamic step capture launches the producer over every dense row; rows
  // at or beyond the device-side active count exit immediately. Only the
  // shared-KV and tensor-core producers with the global-max merge support it.
  const auto& dynamic = riley_cuda_internal::thread_step_capture_dynamic();
  if (dynamic.counts != nullptr &&
      (params->reserved[0] == 0 || params->reserved[1] != 1 ||
       dynamic.dense_rows > params->output_row_count ||
       dynamic.dense_rows > kMaximumGridYOrZ)) {
    return validation_error(error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "a dynamic step capture requires the shared-KV or tensor-core producer, the global-max merge, and dense rows within the output rows");
  }
  const uint64_t launch_rows = dynamic.counts != nullptr
                                   ? dynamic.dense_rows
                                   : params->batch.active_row_count;
  // reserved[2] != 0 selects the tensor-core producer for prompt rows: the
  // row kinds are U32 words in the row-position buffer starting at word
  // `reserved[2] - 1`, one per launched row.
  const uint32_t* row_kinds = nullptr;
  if (params->reserved[2] != 0) {
    const uint64_t first_word = params->reserved[2] - 1;
    const RileyCudaDeviceBuffer* positions = batch.row_positions.buffer;
    if (positions == nullptr || batch.row_positions.byte_offset != 0 ||
        first_word > positions->byte_len / sizeof(uint32_t) ||
        launch_rows > positions->byte_len / sizeof(uint32_t) - first_word) {
      return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                              "row kinds lie outside the row-position buffer");
    }
    row_kinds =
        reinterpret_cast<const uint32_t*>(batch.row_positions.data) + first_word;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const dim3 producer_grid(
        static_cast<uint32_t>(launch_extent),
        static_cast<uint32_t>(params->query_head_count),
        static_cast<uint32_t>(launch_rows));
    // reserved[0] selects the producer: 2 is the tensor-core producer (its
    // own rounding); 1 and 0 are the two launch geometries of the per-token
    // producer, which write bit-identical partial states.
    if (params->reserved[0] == 2) {
      const dim3 tensor_core_grid(
          static_cast<uint32_t>(launch_extent),
          static_cast<uint32_t>(params->key_value_head_count),
          static_cast<uint32_t>(launch_rows));
      native_bf16_ragged_paged_split_gqa_d128_tensor_core_partial_state_kernel<<<
          tensor_core_grid, kWarpSize, 0, stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<float*>(partial_states.data),
          device_batch(params->batch, batch), params->partial_state_capacity,
          params->scale);
    } else if (params->reserved[0] == 1) {
      const dim3 shared_kv_grid(
          static_cast<uint32_t>(launch_extent),
          static_cast<uint32_t>(params->key_value_head_count),
          static_cast<uint32_t>(launch_rows));
      native_bf16_ragged_paged_split_gqa_d128_shared_kv_partial_state_kernel<<<
          shared_kv_grid, kNativeBf16RaggedPagedSplitGqaD128SharedKvThreads, 0,
          stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<float*>(partial_states.data),
          device_batch(params->batch, batch), params->partial_state_capacity,
          params->scale, row_kinds);
      if (row_kinds != nullptr) {
        status = launch_status(error, kOperation);
      }
      if (row_kinds != nullptr && status == RILEY_CUDA_STATUS_SUCCESS) {
        const dim3 prefill_grid(
            static_cast<uint32_t>(
                (launch_rows + kNativeBf16PrefillD128MmaTileRows - 1) /
                kNativeBf16PrefillD128MmaTileRows),
            static_cast<uint32_t>(params->key_value_head_count),
            static_cast<uint32_t>(
                (launch_extent + kNativeBf16PrefillD128MmaSegmentPages - 1) /
                kNativeBf16PrefillD128MmaSegmentPages));
        native_bf16_prefill_d128_mma_partial_state_kernel<<<
            prefill_grid, kNativeBf16RaggedPagedSplitGqaD128SharedKvThreads,
            0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data),
            reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
            reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
            reinterpret_cast<float*>(partial_states.data),
            device_batch(params->batch, batch),
            params->partial_state_capacity, params->scale, row_kinds);
      }
    } else {
      native_bf16_ragged_paged_split_gqa_d128_partial_state_kernel<<<
          producer_grid, kWarpSize, 0, stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<float*>(partial_states.data),
          device_batch(params->batch, batch), params->partial_state_capacity,
          params->scale);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = launch_status(error, kOperation);
    }
  }
  // reserved[1] selects the page-merge stage: 0 keeps the ordered transition +
  // replay pair, 1 runs the single global-max merge kernel.
  if (status == RILEY_CUDA_STATUS_SUCCESS && params->reserved[1] == 1) {
    const dim3 merge_grid(static_cast<uint32_t>(params->query_head_count),
                          static_cast<uint32_t>(params->output_row_count), 1);
    native_bf16_ragged_paged_split_gqa_d128_global_merge_kernel<<<
        merge_grid, kNativeBf16RaggedD128GlobalMergeThreads, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<const uint32_t*>(batch.row_positions.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->batch.active_row_count, params->partial_state_capacity,
        launch_extent, dynamic.counts, row_kinds);
    status = launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS && params->reserved[1] == 0) {
    const dim3 transition_grid(
        static_cast<uint32_t>(params->query_head_count),
        static_cast<uint32_t>(params->batch.active_row_count), 1);
    native_bf16_ragged_paged_split_gqa_d128_transition_kernel<<<
        transition_grid, kWarpSize, 0, stream->stream>>>(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<float*>(reduction_steps.data),
        reinterpret_cast<float*>(reduction_normalizers.data),
        reinterpret_cast<const uint32_t*>(batch.row_positions.data),
        params->partial_state_capacity, launch_extent,
        params->reduction_order);
    status = launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS && params->reserved[1] == 0) {
    const dim3 reduce_grid(static_cast<uint32_t>(params->query_head_count),
                           static_cast<uint32_t>(params->output_row_count),
                           1);
    native_bf16_ragged_paged_split_gqa_d128_transition_reduce_kernel<<<
        reduce_grid, kNativeBf16RaggedPagedSplitGqaD128ReductionThreads, 0,
        stream->stream>>>(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<const float*>(reduction_steps.data),
        reinterpret_cast<const float*>(reduction_normalizers.data),
        reinterpret_cast<const uint32_t*>(batch.row_positions.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->batch.active_row_count, params->partial_state_capacity,
        launch_extent, params->reduction_order);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

namespace {

__global__ void qkv_split_bias_kernel(
    const __nv_bfloat16* merged, const __nv_bfloat16* query_bias,
    const __nv_bfloat16* key_bias, const __nv_bfloat16* value_bias,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    uint64_t query_width, uint64_t key_value_width, uint64_t element_count) {
  const uint64_t merged_width = query_width + 2 * key_value_width;
  const uint64_t first =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t row = index / merged_width;
    const uint64_t column = index % merged_width;
    const float projected = __bfloat162float(merged[index]);
    if (column < query_width) {
      query[row * query_width + column] = __float2bfloat16_rn(
          __fadd_rn(projected, __bfloat162float(query_bias[column])));
    } else if (column < query_width + key_value_width) {
      const uint64_t local = column - query_width;
      key[row * key_value_width + local] = __float2bfloat16_rn(
          __fadd_rn(projected, __bfloat162float(key_bias[local])));
    } else {
      const uint64_t local = column - query_width - key_value_width;
      value[row * key_value_width + local] = __float2bfloat16_rn(
          __fadd_rn(projected, __bfloat162float(value_bias[local])));
    }
  }
}

// Split + bias + indexed RoPE in one kernel. Each thread owns one rotary pair
// of a query or key head (or one value element): it forms the two biased BF16
// elements exactly as qkv_split_bias_kernel stores them and rotates them with
// the arithmetic of indexed_rope_kernel<__nv_bfloat16>.
__global__ void qkv_split_bias_rope_kernel(
    const __nv_bfloat16* merged, const __nv_bfloat16* query_bias,
    const __nv_bfloat16* key_bias, const __nv_bfloat16* value_bias,
    const float* cos, const float* sin, const uint32_t* positions,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    uint64_t query_width, uint64_t key_value_width, uint64_t head_size,
    uint64_t table_position_count, uint64_t rotated_row_count,
    uint64_t work_item_count, const uint32_t* dynamic_counts) {
  if (dynamic_counts != nullptr) {
    rotated_row_count = dynamic_counts[0];
  }
  const uint64_t half = head_size / 2;
  const uint64_t merged_width = query_width + 2 * key_value_width;
  // Work columns per row: one per rotary pair of Q and K, one per V element.
  const uint64_t query_pairs = query_width / 2;
  const uint64_t key_pairs = key_value_width / 2;
  const uint64_t work_width = query_pairs + key_pairs + key_value_width;
  const uint64_t first =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t work = first; work < work_item_count; work += stride) {
    const uint64_t row = work / work_width;
    const uint64_t column = work - row * work_width;
    const uint64_t merged_row = row * merged_width;
    if (column >= query_pairs + key_pairs) {
      const uint64_t local = column - query_pairs - key_pairs;
      const float projected = __bfloat162float(
          merged[merged_row + query_width + key_value_width + local]);
      value[row * key_value_width + local] = __float2bfloat16_rn(
          __fadd_rn(projected, __bfloat162float(value_bias[local])));
      continue;
    }
    if (row >= rotated_row_count) {
      continue;
    }
    const bool is_query = column < query_pairs;
    const uint64_t pair = is_query ? column : column - query_pairs;
    const uint64_t unit = pair % half;
    const uint64_t first_local = (pair / half) * head_size + unit;
    const uint64_t second_local = first_local + half;
    const uint64_t section = is_query ? 0 : query_width;
    const __nv_bfloat16* bias = is_query ? query_bias : key_bias;
    __nv_bfloat16* output = is_query ? query : key;
    const uint64_t output_row = row * (is_query ? query_width : key_value_width);
    const uint64_t position = positions[row];
    if (position >= table_position_count) {
      output[output_row + first_local] = __float2bfloat16_rn(CUDART_NAN_F);
      output[output_row + second_local] = __float2bfloat16_rn(CUDART_NAN_F);
      continue;
    }
    const float first_value = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
        __bfloat162float(merged[merged_row + section + first_local]),
        __bfloat162float(bias[first_local]))));
    const float second_value = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
        __bfloat162float(merged[merged_row + section + second_local]),
        __bfloat162float(bias[second_local]))));
    const uint64_t table_index = position * half + unit;
    const float cosine = round_to_storage<__nv_bfloat16>(cos[table_index]);
    const float sine = round_to_storage<__nv_bfloat16>(sin[table_index]);
    const float first_cosine =
        multiply_in_storage<__nv_bfloat16>(first_value, cosine);
    const float second_sine =
        multiply_in_storage<__nv_bfloat16>(second_value, sine);
    const float second_cosine =
        multiply_in_storage<__nv_bfloat16>(second_value, cosine);
    const float first_sine =
        multiply_in_storage<__nv_bfloat16>(first_value, sine);
    output[output_row + first_local] =
        __float2bfloat16_rn(first_cosine - second_sine);
    output[output_row + second_local] =
        __float2bfloat16_rn(second_cosine + first_sine);
  }
}

}  // namespace

extern "C" RileyCudaStatus
riley_cuda_stream_command_batch_step_capture_dynamic_counts(
    RileyCudaStream* stream, const RileyCudaBufferSpan* counts,
    uint64_t dense_rows, uint64_t output_rows,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "set step capture dynamic counts";
  clear_error(error);
  if (stream == nullptr || counts == nullptr || dense_rows == 0 ||
      output_rows == 0 || dense_rows > UINT32_MAX ||
      output_rows > UINT32_MAX) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "stream or counts is null, or a row extent is zero or too large");
  }
  if (!riley_cuda_internal::thread_has_active_step_capture()) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "this thread has no active step capture");
  }
  ResolvedSpan resolved{};
  const RileyCudaStatus status =
      resolve_span(*counts, RILEY_CUDA_DTYPE_U32, 2 * sizeof(uint32_t),
                   &resolved, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {resolved};
  const RileyCudaStatus context_status =
      validate_contexts(stream, spans, 1, error, kOperation);
  if (context_status != RILEY_CUDA_STATUS_SUCCESS) {
    return context_status;
  }
  auto& dynamic = riley_cuda_internal::thread_step_capture_dynamic();
  dynamic.counts = reinterpret_cast<const uint32_t*>(resolved.data);
  dynamic.dense_rows = dense_rows;
  dynamic.output_rows = output_rows;
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_qkv_split_bias_execute(
    const RileyCudaQkvSplitBiasParams* params, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute merged QKV split with bias";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaQkvSplitBiasParams stable_params = *params;
  params = &stable_params;
  const bool with_rope = params->reserved[0] == 1;
  // reserved[2] == 1 (RoPE mode, inside a command batch) defers the launch
  // to the KV cache write that follows; see DeferredQkvSplit.
  const bool deferred_launch = params->reserved[2] == 1;
  if (params->reserved0 != 0 || params->reserved[0] > 1 ||
      (!with_rope && params->reserved[1] != 0) ||
      (with_rope && params->reserved[1] > params->row_count) ||
      params->reserved[2] > 1 || (deferred_launch && !with_rope) ||
      params->reserved[3] != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (params->query_width == 0 || params->key_value_width == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "query_width and key_value_width must be non-zero");
  }
  if (with_rope &&
      (params->rope_head_size == 0 || params->rope_head_size % 2 != 0 ||
       params->query_width % params->rope_head_size != 0 ||
       params->key_value_width % params->rope_head_size != 0 ||
       params->rope_table_position_count == 0)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "rope_head_size must be even and divide both widths, with a non-empty table");
  }
  uint64_t two_key_value = 0;
  uint64_t merged_width = 0;
  uint64_t merged_elements = 0;
  uint64_t query_elements = 0;
  uint64_t key_value_elements = 0;
  if (!checked_multiply(params->key_value_width, 2, &two_key_value) ||
      params->query_width > UINT64_MAX - two_key_value) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "merged QKV width overflows uint64_t");
  }
  merged_width = params->query_width + two_key_value;
  uint64_t merged_bytes = 0;
  uint64_t query_bytes = 0;
  uint64_t key_value_bytes = 0;
  uint64_t query_bias_bytes = 0;
  uint64_t key_value_bias_bytes = 0;
  if (!checked_multiply(params->row_count, merged_width, &merged_elements) ||
      !checked_multiply(params->row_count, params->query_width,
                        &query_elements) ||
      !checked_multiply(params->row_count, params->key_value_width,
                        &key_value_elements) ||
      !checked_multiply(merged_elements, sizeof(__nv_bfloat16),
                        &merged_bytes) ||
      !checked_multiply(query_elements, sizeof(__nv_bfloat16), &query_bytes) ||
      !checked_multiply(key_value_elements, sizeof(__nv_bfloat16),
                        &key_value_bytes) ||
      !checked_multiply(params->query_width, sizeof(__nv_bfloat16),
                        &query_bias_bytes) ||
      !checked_multiply(params->key_value_width, sizeof(__nv_bfloat16),
                        &key_value_bias_bytes)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "merged QKV shape product overflows uint64_t");
  }

  ResolvedSpan merged{};
  ResolvedSpan query_bias{};
  ResolvedSpan key_bias{};
  ResolvedSpan value_bias{};
  ResolvedSpan query{};
  ResolvedSpan key{};
  ResolvedSpan value{};
  RileyCudaStatus status =
      resolve_span(params->merged, RILEY_CUDA_DTYPE_BF16, merged_bytes,
                   &merged, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->query_bias, RILEY_CUDA_DTYPE_BF16,
                          query_bias_bytes, &query_bias, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_bias, RILEY_CUDA_DTYPE_BF16,
                          key_value_bias_bytes, &key_bias, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_bias, RILEY_CUDA_DTYPE_BF16,
                          key_value_bias_bytes, &value_bias, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->query, RILEY_CUDA_DTYPE_BF16, query_bytes,
                          &query, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key, RILEY_CUDA_DTYPE_BF16, key_value_bytes,
                          &key, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value, RILEY_CUDA_DTYPE_BF16,
                          key_value_bytes, &value, error, kOperation);
  }
  ResolvedSpan rope_cos{};
  ResolvedSpan rope_sin{};
  ResolvedSpan rope_positions{};
  if (status == RILEY_CUDA_STATUS_SUCCESS && with_rope) {
    uint64_t table_elements = 0;
    uint64_t table_bytes = 0;
    uint64_t position_bytes = 0;
    if (!checked_multiply(params->rope_table_position_count,
                          params->rope_head_size / 2, &table_elements) ||
        !checked_multiply(table_elements, sizeof(float), &table_bytes) ||
        !checked_multiply(params->reserved[1], sizeof(uint32_t),
                          &position_bytes)) {
      return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                              "merged QKV RoPE table product overflows uint64_t");
    }
    status = resolve_span(params->rope_cos, RILEY_CUDA_DTYPE_F32, table_bytes,
                          &rope_cos, error, kOperation);
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = resolve_span(params->rope_sin, RILEY_CUDA_DTYPE_F32,
                            table_bytes, &rope_sin, error, kOperation);
    }
    if (status == RILEY_CUDA_STATUS_SUCCESS) {
      status = resolve_span(params->rope_positions, RILEY_CUDA_DTYPE_U32,
                            position_bytes, &rope_positions, error,
                            kOperation);
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan inputs[] = {merged,   query_bias, key_bias, value_bias,
                                 rope_cos, rope_sin,   rope_positions};
  const size_t input_count = with_rope ? 7 : 4;
  const ResolvedSpan writable[] = {query, key, value};
  for (size_t writable_index = 0;
       status == RILEY_CUDA_STATUS_SUCCESS && writable_index < 3;
       ++writable_index) {
    for (size_t input_index = 0;
         status == RILEY_CUDA_STATUS_SUCCESS && input_index < input_count;
         ++input_index) {
      status = reject_overlap(writable[writable_index], inputs[input_index],
                              false, error, kOperation);
    }
    for (size_t prior_index = 0;
         status == RILEY_CUDA_STATUS_SUCCESS && prior_index < writable_index;
         ++prior_index) {
      status = reject_overlap(writable[writable_index],
                              writable[prior_index], false, error,
                              kOperation);
    }
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {merged,   query_bias, key_bias,
                                value_bias, query,    key,
                                value,    rope_cos,   rope_sin,
                                rope_positions};
  const size_t span_count = with_rope ? 10 : 7;
  status = validate_contexts(stream, spans, span_count, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < span_count; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "merged QKV buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS && deferred_launch) {
    auto& deferred = riley_cuda_internal::thread_deferred_qkv_split();
    if (!uses.command_batch() || deferred.armed || merged_elements == 0) {
      status = validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
          "a deferred merged QKV split needs a command batch, a non-empty input and no pending deferred split");
    } else {
      deferred.armed = true;
      deferred.stream = stream->stream;
      deferred.merged = merged.data;
      deferred.query_bias = query_bias.data;
      deferred.key_bias = key_bias.data;
      deferred.value_bias = value_bias.data;
      deferred.cos = rope_cos.data;
      deferred.sin = rope_sin.data;
      deferred.positions = rope_positions.data;
      deferred.query = query.data;
      deferred.key = key.data;
      deferred.value = value.data;
      deferred.row_count = params->row_count;
      deferred.query_width = params->query_width;
      deferred.key_value_width = params->key_value_width;
      deferred.head_size = params->rope_head_size;
      deferred.table_position_count = params->rope_table_position_count;
      deferred.rotated_row_count = params->reserved[1];
    }
  } else if (status == RILEY_CUDA_STATUS_SUCCESS && merged_elements != 0 &&
      with_rope) {
    launch_attempted = true;
    const uint64_t work_items =
        params->row_count * (params->query_width / 2 +
                             params->key_value_width / 2 +
                             params->key_value_width);
    qkv_split_bias_rope_kernel<<<block_count(work_items), kThreads, 0,
                                 stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(merged.data),
        reinterpret_cast<const __nv_bfloat16*>(query_bias.data),
        reinterpret_cast<const __nv_bfloat16*>(key_bias.data),
        reinterpret_cast<const __nv_bfloat16*>(value_bias.data),
        reinterpret_cast<const float*>(rope_cos.data),
        reinterpret_cast<const float*>(rope_sin.data),
        reinterpret_cast<const uint32_t*>(rope_positions.data),
        reinterpret_cast<__nv_bfloat16*>(query.data),
        reinterpret_cast<__nv_bfloat16*>(key.data),
        reinterpret_cast<__nv_bfloat16*>(value.data), params->query_width,
        params->key_value_width, params->rope_head_size,
        params->rope_table_position_count, params->reserved[1], work_items,
        riley_cuda_internal::thread_step_capture_dynamic().counts);
    status = launch_status(error, kOperation);
  } else if (status == RILEY_CUDA_STATUS_SUCCESS && merged_elements != 0) {
    launch_attempted = true;
    qkv_split_bias_kernel<<<block_count(merged_elements), kThreads, 0,
                            stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(merged.data),
        reinterpret_cast<const __nv_bfloat16*>(query_bias.data),
        reinterpret_cast<const __nv_bfloat16*>(key_bias.data),
        reinterpret_cast<const __nv_bfloat16*>(value_bias.data),
        reinterpret_cast<__nv_bfloat16*>(query.data),
        reinterpret_cast<__nv_bfloat16*>(key.data),
        reinterpret_cast<__nv_bfloat16*>(value.data), params->query_width,
        params->key_value_width, merged_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

cudaError_t riley_cuda_internal::enqueue_qkv_rope(cudaStream_t stream,
    const void* q, const void* k, void* qr, void* kr, const void* cos,
    const void* sin, const void* positions, uint64_t q_heads, uint64_t kv_heads,
    uint64_t table_positions) noexcept {
  indexed_rope_kernel<__nv_bfloat16><<<block_count(q_heads * 32), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q), static_cast<const float*>(cos), static_cast<const float*>(sin),
      static_cast<const uint32_t*>(positions), static_cast<__nv_bfloat16*>(qr), table_positions, q_heads, 64, 64, q_heads * 32);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  indexed_rope_kernel<__nv_bfloat16><<<block_count(kv_heads * 32), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(k), static_cast<const float*>(cos), static_cast<const float*>(sin),
      static_cast<const uint32_t*>(positions), static_cast<__nv_bfloat16*>(kr), table_positions, kv_heads, 64, 64, kv_heads * 32);
  return cudaGetLastError();
}

namespace {
__global__ void bound_kv_write_kernel(const __nv_bfloat16* key, const __nv_bfloat16* value,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool, const uint32_t* position,
    uint64_t heads, uint64_t physical) {
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       i < heads*64; i += static_cast<uint64_t>(gridDim.x)*blockDim.x) {
    const uint64_t target = ((physical*heads+i/64)*16+(*position%16))*64+i%64;
    key_pool[target]=key[i]; value_pool[target]=value[i];
  }
}
}
cudaError_t riley_cuda_internal::enqueue_bound_kv_write(cudaStream_t stream,
    const void* key,const void* value,void* key_pool,void* value_pool,const void* position,
    uint64_t heads,uint64_t physical_block) noexcept {
  bound_kv_write_kernel<<<block_count(heads*64),kThreads,0,stream>>>(
      static_cast<const __nv_bfloat16*>(key),static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(key_pool),static_cast<__nv_bfloat16*>(value_pool),
      static_cast<const uint32_t*>(position),heads,physical_block);
  return cudaGetLastError();
}

cudaError_t riley_cuda_internal::enqueue_bound_attention(cudaStream_t stream,const void* query,
    const void* key,const void* value,void* output,const void* metadata,const uint64_t* fields,
    uint64_t position_offset,uint64_t query_heads,uint64_t kv_heads,uint64_t physical_blocks) noexcept {
  const auto* m=static_cast<const uint8_t*>(metadata);
  DeviceBatch batch{reinterpret_cast<const uint32_t*>(m+fields[0]),reinterpret_cast<const uint32_t*>(m+fields[1]),
      reinterpret_cast<const uint16_t*>(m+fields[2]),reinterpret_cast<const uint32_t*>(m+fields[3]),
      reinterpret_cast<const uint32_t*>(m+position_offset),1,1,1,physical_blocks};
  const auto group=query_heads/kv_heads;
  if(group>1 && group<=kRaggedAttentionMaximumWarpsPerBlock) {
    ragged_paged_attention_gqa_shared_kv_kernel<<<dim3(1,static_cast<uint32_t>(kv_heads),1),static_cast<uint32_t>(group*kWarpSize),0,stream>>>(
      static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(output),batch,query_heads,kv_heads,0.125F);
  } else {
    const auto warps=static_cast<uint32_t>(query_heads<kRaggedAttentionMaximumWarpsPerBlock?query_heads:kRaggedAttentionMaximumWarpsPerBlock);
    ragged_paged_attention_grouped_heads_kernel<<<dim3(1,static_cast<uint32_t>((query_heads+warps-1)/warps),1),warps*kWarpSize,0,stream>>>(
      static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(output),batch,query_heads,kv_heads,0.125F);
  }
  return cudaGetLastError();
}

// Metadata: token, position, [0,live_blocks], padded physical ids, valid prefixes,
// then one row slot. Host replay validates the complete mapping before staging.
cudaError_t riley_cuda_internal::enqueue_decode_kv_attention(cudaStream_t stream,
    const void* query,const void* key_source,const void* value_source,void* key,void* value,
    void* output,const void* metadata,uint64_t capacity,uint64_t query_heads,
    uint64_t kv_heads,uint64_t physical_blocks) noexcept {
  const auto* m=static_cast<const uint8_t*>(metadata);
  const uint64_t slot=(16+6*capacity+3)&~uint64_t(3);
  DeviceBatch batch{reinterpret_cast<const uint32_t*>(m+8),reinterpret_cast<const uint32_t*>(m+16),
      reinterpret_cast<const uint16_t*>(m+16+4*capacity),reinterpret_cast<const uint32_t*>(m+slot),
      reinterpret_cast<const uint32_t*>(m+4),1,capacity,1,physical_blocks};
  ragged_paged_kv_cache_write_kernel<<<block_count(kv_heads*64),kThreads,0,stream>>>(
      static_cast<const __nv_bfloat16*>(key_source),static_cast<const __nv_bfloat16*>(value_source),
      static_cast<__nv_bfloat16*>(key),static_cast<__nv_bfloat16*>(value),batch,kv_heads,64,kv_heads*64);
  auto status=cudaGetLastError(); if(status!=cudaSuccess) return status;
  const auto group=query_heads/kv_heads;
  if(group>1 && group<=kRaggedAttentionMaximumWarpsPerBlock) {
    ragged_paged_attention_gqa_shared_kv_kernel<<<dim3(1,static_cast<uint32_t>(kv_heads),1),static_cast<uint32_t>(group*kWarpSize),0,stream>>>(
      static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(output),batch,query_heads,kv_heads,0.125F);
  } else {
    const auto warps=static_cast<uint32_t>(query_heads<kRaggedAttentionMaximumWarpsPerBlock?query_heads:kRaggedAttentionMaximumWarpsPerBlock);
    ragged_paged_attention_grouped_heads_kernel<<<dim3(1,static_cast<uint32_t>((query_heads+warps-1)/warps),1),warps*kWarpSize,0,stream>>>(
      static_cast<const __nv_bfloat16*>(query),static_cast<const __nv_bfloat16*>(key),static_cast<const __nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(output),batch,query_heads,kv_heads,0.125F);
  }
  return cudaGetLastError();
}

cudaError_t riley_cuda_internal::enqueue_decode_argmax(cudaStream_t stream,const void* logits,void* output,uint64_t vocab) noexcept {
  bf16_argmax_kernel<<<1,kArgmaxThreads,0,stream>>>(static_cast<const __nv_bfloat16*>(logits),static_cast<RileyCudaBf16ArgmaxResult*>(output),1,vocab,vocab,nullptr);
  return cudaGetLastError();
}
