#include "ffi_internal.hpp"
#include "fixed37_reduction.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_exclusive_use;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::try_acquire_exclusive_use;
using rustinfer_cuda_internal::validation_error;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaximumBlocks = 65535;
constexpr size_t kMaximumDecodeBuffers = 7;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
constexpr uint64_t kOptimizedHeadSize = 64;
constexpr uint64_t kOptimizedStateStride = kOptimizedHeadSize + 2;
constexpr uint64_t kFixed37TwoPassHeadSize = 64;
constexpr uint64_t kFixed37TwoPassDepthPartialCount =
    rustinfer_cuda_fixed37::chunk_count(kFixed37TwoPassHeadSize);
constexpr uint64_t kFixed37TwoPassMaximumTokenCount = 8192;
constexpr uint64_t kHuggingFaceShortDecodeMaximumTokenCount = 32;
constexpr uint64_t kReviewedHuggingFaceQueryHeadCount = 9;
constexpr uint64_t kReviewedHuggingFaceKeyValueHeadCount = 3;
constexpr uint64_t kReviewedHuggingFaceHeadSize = 64;
constexpr uint64_t kMaximumGridX = 2147483647;
constexpr uint64_t kMaximumGridYOrZ = 65535;

static_assert(sizeof(RustInferCudaKvCacheWriteParams) == 272,
              "KV cache write ABI size changed");
static_assert(sizeof(RustInferCudaDecodeAttentionReferenceParams) == 328,
              "decode reference ABI size changed");
static_assert(sizeof(RustInferCudaDecodeAttentionParams) == 344,
              "decode attention ABI size changed");
static_assert(sizeof(RustInferCudaDecodePartialStateReduceParams) == 176,
              "decode reducer ABI size changed");
static_assert(sizeof(RustInferCudaPagedKvBlockTableV1) == 168,
              "paged block-table ABI size changed");
static_assert(sizeof(RustInferCudaPagedKvCacheWriteParams) == 432,
              "paged KV write ABI size changed");
static_assert(sizeof(RustInferCudaPagedDecodeAttentionReferenceParams) == 480,
              "paged reference decode ABI size changed");
static_assert(sizeof(RustInferCudaPagedDecodeAttentionParams) == 488,
              "paged online decode ABI size changed");
static_assert(kOptimizedHeadSize == 2 * kWarpSize,
              "optimized decode lane ownership changed");

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
};

struct CacheByteCounts {
  uint64_t source;
  uint64_t cache;
};

struct DecodeByteCounts {
  uint64_t query_output;
  uint64_t cache;
  uint64_t scores;
};

struct DecodeTwoPassByteCounts {
  uint64_t query_output;
  uint64_t cache;
};

struct PagedByteCounts {
  uint64_t query_output;
  uint64_t pool;
  uint64_t scores;
  uint64_t block_ids;
  uint64_t valid_tokens;
};

struct PagedTwoPassByteCounts {
  uint64_t query_output;
  uint64_t pool;
  uint64_t block_ids;
  uint64_t valid_tokens;
};

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

bool use_reviewed_hugging_face_short_decode(
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size) noexcept {
  return logical_token_count <= kHuggingFaceShortDecodeMaximumTokenCount &&
         query_head_count == kReviewedHuggingFaceQueryHeadCount &&
         key_value_head_count == kReviewedHuggingFaceKeyValueHeadCount &&
         head_size == kReviewedHuggingFaceHeadSize;
}

RustInferCudaStatus validate_hugging_face_short_workspace_prefix(
    const ResolvedSpan& workspace, uint64_t current_score_bytes,
    uint64_t maximum_score_bytes, uint64_t available_bytes,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (current_score_bytes > maximum_score_bytes) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "short decode current score bytes exceed the reviewed maximum");
  }
  if (maximum_score_bytes > available_bytes ||
      maximum_score_bytes > workspace.used_bytes) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "decode partial-state workspace cannot hold the short score prefix");
  }
  if ((reinterpret_cast<uintptr_t>(workspace.data) %
       alignof(__nv_bfloat16)) != 0) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "decode partial-state workspace is not BF16 aligned");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus typed_bytes(uint64_t element_count, uint64_t element_size,
                                uint64_t* output,
                                RustInferCudaErrorInfo* error,
                                const char* operation) noexcept {
  if (!checked_multiply(element_count, element_size, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "decode byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus cache_byte_counts(
    uint64_t source_token_count, uint64_t maximum_token_count,
    uint64_t key_value_head_count, uint64_t head_size,
    CacheByteCounts* output, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal cache byte counts are null");
  }
  uint64_t source_elements = 0;
  uint64_t cache_elements = 0;
  if (!checked_product3(source_token_count, key_value_head_count, head_size,
                        &source_elements) ||
      !checked_product3(maximum_token_count, key_value_head_count, head_size,
                        &cache_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "KV cache tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status =
      typed_bytes(source_elements, 2, &output->source, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(cache_elements, 2, &output->cache, error, operation);
  }
  return status;
}

RustInferCudaStatus decode_byte_counts(
    uint64_t maximum_token_count, uint64_t logical_token_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, DecodeByteCounts* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal decode byte counts are null");
  }
  uint64_t query_elements = 0;
  uint64_t cache_elements = 0;
  uint64_t score_elements = 0;
  if (!checked_multiply(query_head_count, head_size, &query_elements) ||
      !checked_product3(key_value_head_count, maximum_token_count, head_size,
                        &cache_elements) ||
      !checked_multiply(query_head_count, logical_token_count,
                        &score_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "decode tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status = typed_bytes(
      query_elements, 2, &output->query_output, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(cache_elements, 2, &output->cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(score_elements, 2, &output->scores, error, operation);
  }
  return status;
}

RustInferCudaStatus decode_two_pass_byte_counts(
    uint64_t maximum_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, DecodeTwoPassByteCounts* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "internal two-pass decode byte counts are null");
  }
  uint64_t query_elements = 0;
  uint64_t cache_elements = 0;
  if (!checked_multiply(query_head_count, kFixed37TwoPassHeadSize,
                        &query_elements) ||
      !checked_product3(key_value_head_count, maximum_token_count,
                        kFixed37TwoPassHeadSize, &cache_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "two-pass decode tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status = typed_bytes(
      query_elements, 2, &output->query_output, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(cache_elements, 2, &output->cache, error, operation);
  }
  return status;
}

RustInferCudaStatus partial_state_bytes(
    uint64_t capacity, uint64_t query_head_count, uint64_t head_size,
    uint64_t* output, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  uint64_t stride = 0;
  uint64_t elements = 0;
  if (!checked_add(head_size, 2, &stride) ||
      !checked_product3(capacity, query_head_count, stride, &elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "decode partial-state shape overflows uint64_t");
  }
  return typed_bytes(elements, 4, output, error, operation);
}

RustInferCudaStatus paged_byte_counts(
    const RustInferCudaPagedKvBlockTableV1& table,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, PagedByteCounts* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal paged byte counts are null");
  }
  uint64_t query_elements = 0;
  uint64_t block_elements = 0;
  uint64_t pool_elements = 0;
  uint64_t score_elements = 0;
  if (!checked_multiply(query_head_count, head_size, &query_elements) ||
      !checked_product3(key_value_head_count,
                        RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE, head_size,
                        &block_elements) ||
      !checked_multiply(table.physical_block_count, block_elements,
                        &pool_elements) ||
      !checked_multiply(query_head_count, table.logical_token_count,
                        &score_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged decode tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status = typed_bytes(
      query_elements, 2, &output->query_output, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, 2, &output->pool, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(score_elements, 2, &output->scores, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table.block_count, 4, &output->block_ids, error,
                         operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table.block_count, 2, &output->valid_tokens, error,
                         operation);
  }
  return status;
}

RustInferCudaStatus paged_two_pass_byte_counts(
    const RustInferCudaPagedKvBlockTableV1& table,
    uint64_t query_head_count, uint64_t key_value_head_count,
    PagedTwoPassByteCounts* output, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "internal paged two-pass byte counts are null");
  }
  uint64_t query_elements = 0;
  uint64_t block_elements = 0;
  uint64_t pool_elements = 0;
  if (!checked_multiply(query_head_count, kFixed37TwoPassHeadSize,
                        &query_elements) ||
      !checked_product3(key_value_head_count,
                        RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE,
                        kFixed37TwoPassHeadSize, &block_elements) ||
      !checked_multiply(table.physical_block_count, block_elements,
                        &pool_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged two-pass tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status = typed_bytes(
      query_elements, 2, &output->query_output, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, 2, &output->pool, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table.block_count, 4, &output->block_ids, error,
                         operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table.block_count, 2, &output->valid_tokens, error,
                         operation);
  }
  return status;
}

RustInferCudaStatus resolve_span(const RustInferCudaBufferSpan& span,
                                 RustInferCudaDType dtype,
                                 uint64_t alignment,
                                 uint64_t required_bytes,
                                 ResolvedSpan* output,
                                 RustInferCudaErrorInfo* error,
                                 const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal resolved span is null");
  }
  if (span.struct_size < sizeof(span)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span has an incompatible struct_size");
  }
  if (!reserved_is_zero(span.reserved, 2)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span reserved fields must be zero");
  }
  if (span.dtype != dtype) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span dtype does not match the decode contract");
  }
  if (span.buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span handle is null");
  }
  if (span.byte_offset % alignment != 0 || span.byte_len % alignment != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span offset or length is not dtype-aligned");
  }
  if (span.byte_offset > span.buffer->byte_len ||
      span.byte_len > span.buffer->byte_len - span.byte_offset) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "declared span exceeds the opaque allocation");
  }
  if (required_bytes > span.byte_len) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "required bytes exceed the declared span capacity");
  }
  if (span.buffer->device_data == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "decode span refers to a zero-byte allocation");
  }
  *output = ResolvedSpan{
      span.buffer,
      static_cast<uint8_t*>(span.buffer->device_data) +
          static_cast<size_t>(span.byte_offset),
      span.byte_offset,
      required_bytes,
  };
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

bool overlaps(const ResolvedSpan& left, const ResolvedSpan& right) noexcept {
  if (left.buffer != right.buffer) {
    return false;
  }
  const uint64_t left_end = left.byte_offset + left.used_bytes;
  const uint64_t right_end = right.byte_offset + right.used_bytes;
  return left.byte_offset < right_end && right.byte_offset < left_end;
}

RustInferCudaStatus reject_overlap(const ResolvedSpan& writable,
                                   const ResolvedSpan& other,
                                   RustInferCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(writable, other)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "a writable decode span may not overlap another touched span");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_contexts(RustInferCudaStream* stream,
                                      const ResolvedSpan* spans, size_t count,
                                      RustInferCudaErrorInfo* error,
                                      const char* operation) noexcept {
  if (stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "stream is null");
  }
  if (stream->owner == nullptr ||
      stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "CUDA context owner is missing or poisoned by a prior restoration failure");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!same_context(stream->owner, spans[index].buffer->owner)) {
      return validation_error(
          error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
          "stream and decode spans belong to different context owners");
    }
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

class ExclusiveUses final {
 public:
  explicit ExclusiveUses(RustInferCudaStream* stream) noexcept
      : stream_(stream),
        buffers_{},
        buffer_count_(0),
        acquired_count_(0),
        stream_acquired_(false) {}

  ExclusiveUses(const ExclusiveUses&) = delete;
  ExclusiveUses& operator=(const ExclusiveUses&) = delete;

  bool add(RustInferCudaDeviceBuffer* buffer) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (buffers_[index] == buffer) {
        return true;
      }
    }
    if (buffer == nullptr || buffer_count_ == kMaximumDecodeBuffers) {
      return false;
    }
    buffers_[buffer_count_++] = buffer;
    return true;
  }

  RustInferCudaStatus acquire(RustInferCudaErrorInfo* error,
                              const char* operation) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        release_acquired();
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
            "a decode buffer already has an active asynchronous use");
      }
      ++acquired_count_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      release_acquired();
      return validation_error(
          error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
          "the stream already has an active asynchronous use");
    }
    stream_acquired_ = true;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  bool release_completed() noexcept {
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

 private:
  void release_acquired() noexcept {
    while (acquired_count_ != 0) {
      --acquired_count_;
      (void)release_exclusive_use(buffers_[acquired_count_]->active_uses);
    }
  }

  RustInferCudaStream* stream_;
  RustInferCudaDeviceBuffer* buffers_[kMaximumDecodeBuffers];
  size_t buffer_count_;
  size_t acquired_count_;
  bool stream_acquired_;
};

uint32_t block_count(uint64_t work_items) noexcept {
  const uint64_t needed = ((work_items - 1) / kThreads) + 1;
  return static_cast<uint32_t>(
      needed < kMaximumBlocks ? needed : kMaximumBlocks);
}

RustInferCudaStatus launch_status(RustInferCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RustInferCudaStatus complete_execution(ExclusiveUses* uses,
                                       CurrentContext* scope,
                                       RustInferCudaStream* stream,
                                       RustInferCudaStatus operation_status,
                                       bool launch_attempted,
                                       RustInferCudaErrorInfo* error,
                                       const char* operation) noexcept {
  bool completion_confirmed = !launch_attempted;
  RustInferCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t synchronize_result = cudaStreamSynchronize(stream->stream);
    completion_confirmed = synchronize_result == cudaSuccess;
    if (!completion_confirmed) {
      status = runtime_error(synchronize_result, error,
                             RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                             operation);
    }
  }
  status = scope->leave(status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (completion_confirmed && restoration_confirmed) {
    if (!uses->release_completed()) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            operation,
                            "exclusive-use accounting was corrupted");
    }
  }
  return status;
}

RustInferCudaStatus validate_cache_write_dimensions(
    const RustInferCudaKvCacheWriteParams& params,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (params.source_token_count == 0 || params.maximum_token_count == 0 ||
      params.key_value_head_count == 0 || params.head_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "all KV cache write dimensions must be greater than zero");
  }
  if (params.destination_token_start > params.maximum_token_count ||
      params.source_token_count >
          params.maximum_token_count - params.destination_token_start) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "KV cache destination interval exceeds maximum_token_count");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_decode_dimensions(
    uint64_t maximum_token_count, uint64_t logical_token_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, float scale, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (maximum_token_count == 0 || logical_token_count == 0 ||
      query_head_count == 0 || key_value_head_count == 0 || head_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "all decode dimensions must be greater than zero");
  }
  if (logical_token_count > maximum_token_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "logical_token_count exceeds maximum_token_count");
  }
  if (query_head_count % key_value_head_count != 0) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "key_value_head_count must divide query_head_count");
  }
  if (!std::isfinite(scale) || scale <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "scale must be finite and greater than zero");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_fixed37_axis(
    uint64_t element_count, uint64_t* partial_count,
    uint64_t* shared_bytes, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (element_count == 0 || partial_count == nullptr ||
      shared_bytes == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "the fixed37 reduction axis must be non-zero");
  }
  const uint64_t chunks = rustinfer_cuda_fixed37::chunk_count(element_count);
  if (chunks == 0 || chunks > rustinfer_cuda_fixed37::kMaximumChunkCount) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "the decode reduction axis exceeds the fixed37 chunk-partial capacity");
  }
  *partial_count = chunks;
  *shared_bytes = rustinfer_cuda_fixed37::shared_bytes(element_count);
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus fixed37_two_pass_shared_bytes(
    uint64_t logical_token_count, uint64_t token_partial_count,
    uint64_t* shared_bytes, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (shared_bytes == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "internal two-pass shared byte output is null");
  }
  const uint64_t partial_capacity =
      token_partial_count < kFixed37TwoPassDepthPartialCount
          ? kFixed37TwoPassDepthPartialCount
          : token_partial_count;
  uint64_t score_bytes = 0;
  uint64_t reduction_bytes = 0;
  if (!checked_multiply(logical_token_count, sizeof(float), &score_bytes) ||
      !checked_multiply(partial_capacity, 2 * sizeof(float),
                        &reduction_bytes) ||
      !checked_add(score_bytes, reduction_bytes, shared_bytes)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "fixed37 two-pass shared-memory size overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_reduction_order(
    uint32_t reduction_order, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (reduction_order != RUSTINFER_CUDA_DECODE_REDUCTION_ASCENDING &&
      reduction_order != RUSTINFER_CUDA_DECODE_REDUCTION_DESCENDING) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "decode reduction_order is not supported");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_paged_block_table(
    const RustInferCudaPagedKvBlockTableV1& table,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (table.struct_size < sizeof(table)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block table has an incompatible struct_size");
  }
  if (table.format_version != RUSTINFER_CUDA_PAGED_KV_BLOCK_TABLE_VERSION ||
      table.block_size != RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block table version or block size is unsupported");
  }
  if (table.metadata_kind != RUSTINFER_CUDA_PAGED_KV_METADATA_NONE ||
      table.metadata_version != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged v1 exact execution does not accept a metadata sidecar");
  }
  if (table.reserved0 != 0 || !reserved_is_zero(table.reserved, 3)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block table reserved fields must be zero");
  }
  if (table.logical_token_count == 0 || table.block_count == 0 ||
      table.physical_block_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block-table dimensions must be greater than zero");
  }
  const uint64_t expected_blocks =
      ((table.logical_token_count - 1) /
       RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE) +
      1;
  if (table.block_count != expected_blocks) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block_count does not match logical_token_count");
  }
  if (table.block_count > table.physical_block_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "paged block table exceeds the physical pool");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus resolve_paged_block_table(
    const RustInferCudaPagedKvBlockTableV1& table,
    const PagedByteCounts& bytes, ResolvedSpan* block_ids,
    ResolvedSpan* valid_tokens, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  RustInferCudaStatus status =
      resolve_span(table.block_ids, RUSTINFER_CUDA_DTYPE_U32, 4,
                   bytes.block_ids, block_ids, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(table.valid_tokens, RUSTINFER_CUDA_DTYPE_U16, 2,
                          bytes.valid_tokens, valid_tokens, error, operation);
  }
  return status;
}

__global__ void kv_cache_write_kernel(
    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
    __nv_bfloat16* key_cache, __nv_bfloat16* value_cache,
    uint64_t destination_token_start, uint64_t maximum_token_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t row = index / head_size;
    const uint64_t key_value_head = row % key_value_head_count;
    const uint64_t source_token = row / key_value_head_count;
    const uint64_t destination_token =
        destination_token_start + source_token;
    const uint64_t cache_index =
        (key_value_head * maximum_token_count + destination_token) *
            head_size +
        depth;
    key_cache[cache_index] = key_source[index];
    value_cache[cache_index] = value_source[index];
  }
}

__device__ __forceinline__ bool paged_cache_base(
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    uint64_t logical_token, uint64_t physical_block_count,
    uint64_t key_value_head, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t* output) {
  const uint64_t logical_block =
      logical_token / RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE;
  const uint64_t token_in_block =
      logical_token % RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE;
  const uint64_t physical_block = block_ids[logical_block];
  const uint64_t valid = valid_tokens[logical_block];
  if (output == nullptr || physical_block >= physical_block_count ||
      valid == 0 || valid > RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE ||
      token_in_block >= valid) {
    return false;
  }
  *output =
      ((physical_block * key_value_head_count + key_value_head) *
           RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE +
       token_in_block) *
      head_size;
  return true;
}

__global__ void paged_kv_cache_write_kernel(
    const __nv_bfloat16* key_source, const __nv_bfloat16* value_source,
    __nv_bfloat16* key_pool, __nv_bfloat16* value_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    uint64_t destination_token_start, uint64_t physical_block_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t row = index / head_size;
    const uint64_t key_value_head = row % key_value_head_count;
    const uint64_t logical_token =
        destination_token_start + row / key_value_head_count;
    uint64_t cache_base = 0;
    if (paged_cache_base(block_ids, valid_tokens, logical_token,
                         physical_block_count, key_value_head,
                         key_value_head_count, head_size, &cache_base)) {
      key_pool[cache_base + depth] = key_source[index];
      value_pool[cache_base + depth] = value_source[index];
    }
  }
}

__global__ void paged_decode_qk_reference_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    __nv_bfloat16* scores, uint64_t physical_block_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t score_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < score_element_count; index += stride) {
    const uint64_t logical_token = index % logical_token_count;
    const uint64_t query_head = index / logical_token_count;
    const uint64_t key_value_head = query_head / group_size;
    uint64_t key_base = 0;
    if (!paged_cache_base(block_ids, valid_tokens, logical_token,
                          physical_block_count, key_value_head,
                          key_value_head_count, head_size, &key_base)) {
      scores[index] = __float2bfloat16_rn(CUDART_NAN_F);
      continue;
    }
    const uint64_t query_base = query_head * head_size;
    float accumulator = 0.0F;
    for (uint64_t depth = 0; depth < head_size; ++depth) {
      accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                         __bfloat162float(key_pool[key_base + depth]),
                         accumulator);
    }
    scores[index] = __float2bfloat16_rn(accumulator);
  }
}

__global__ void paged_decode_av_reference_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    __nv_bfloat16* output, uint64_t physical_block_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t output_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < output_element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t query_head = index / head_size;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base = query_head * logical_token_count;
    float accumulator = 0.0F;
    for (uint64_t logical_token = 0; logical_token < logical_token_count;
         ++logical_token) {
      uint64_t value_base = 0;
      if (!paged_cache_base(block_ids, valid_tokens, logical_token,
                            physical_block_count, key_value_head,
                            key_value_head_count, head_size, &value_base)) {
        accumulator = CUDART_NAN_F;
        break;
      }
      accumulator = fmaf(
          __bfloat162float(probabilities[probability_base + logical_token]),
          __bfloat162float(value_pool[value_base + depth]), accumulator);
    }
    output[index] = __float2bfloat16_rn(accumulator);
  }
}

__global__ void decode_qk_reference_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    __nv_bfloat16* scores, uint64_t maximum_token_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t score_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < score_element_count; index += stride) {
    const uint64_t token = index % logical_token_count;
    const uint64_t query_head = index / logical_token_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base = query_head * head_size;
    const uint64_t key_base =
        (key_value_head * maximum_token_count + token) * head_size;
    float accumulator = 0.0F;
    for (uint64_t depth = 0; depth < head_size; ++depth) {
      accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                         __bfloat162float(key_cache[key_base + depth]),
                         accumulator);
    }
    scores[index] = __float2bfloat16_rn(accumulator);
  }
}

__global__ void decode_scale_reference_kernel(__nv_bfloat16* scores,
                                               float scale,
                                               uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    scores[index] =
        __float2bfloat16_rn(__bfloat162float(scores[index]) * scale);
  }
}

__global__ void decode_softmax_reference_kernel(
    __nv_bfloat16* scores, uint64_t logical_token_count,
    uint64_t query_head_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t query_head = first; query_head < query_head_count;
       query_head += stride) {
    const uint64_t base = query_head * logical_token_count;
    float maximum = -CUDART_INF_F;
    bool has_nan = false;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      const float score = __bfloat162float(scores[base + token]);
      has_nan = has_nan || isnan(score);
      maximum = fmaxf(maximum, score);
    }
    if (has_nan) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t token = 0; token < logical_token_count; ++token) {
        scores[base + token] = nan;
      }
      continue;
    }
    if (isinf(maximum)) {
      if (maximum > 0.0F) {
        uint64_t positive_infinity_count = 0;
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          const float score = __bfloat162float(scores[base + token]);
          positive_infinity_count +=
              static_cast<uint64_t>(isinf(score) && score > 0.0F);
        }
        const float tied_probability =
            1.0F / static_cast<float>(positive_infinity_count);
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          const float score = __bfloat162float(scores[base + token]);
          const float probability = isinf(score) && score > 0.0F
                                        ? tied_probability
                                        : 0.0F;
          scores[base + token] = __float2bfloat16_rn(probability);
        }
      } else {
        const __nv_bfloat16 zero = __float2bfloat16_rn(0.0F);
        for (uint64_t token = 0; token < logical_token_count; ++token) {
          scores[base + token] = zero;
        }
      }
      continue;
    }
    float denominator = 0.0F;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      denominator +=
          expf(__bfloat162float(scores[base + token]) - maximum);
    }
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      const float numerator =
          expf(__bfloat162float(scores[base + token]) - maximum);
      scores[base + token] =
          __float2bfloat16_rn(numerator / denominator);
    }
  }
}

__global__ void decode_av_reference_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_cache,
    __nv_bfloat16* output, uint64_t maximum_token_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t output_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < output_element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t query_head = index / head_size;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base = query_head * logical_token_count;
    float accumulator = 0.0F;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      const uint64_t value_index =
          (key_value_head * maximum_token_count + token) * head_size + depth;
      accumulator = fmaf(
          __bfloat162float(probabilities[probability_base + token]),
          __bfloat162float(value_cache[value_index]), accumulator);
    }
    output[index] = __float2bfloat16_rn(accumulator);
  }
}

// PyTorch's BF16 eager AV matmul for the reviewed 9QH/3KVH/D64 one-token
// decode geometry reduces a short K axis as one warp tree. Keeping one output
// element per warp preserves that order and the final BF16 staging boundary.
__global__ void reviewed_hugging_face_decode_av_short_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_cache,
    __nv_bfloat16* output, uint64_t maximum_token_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t logical_token_count) {
  const uint64_t index = blockIdx.x;
  const uint64_t depth = index % head_size;
  const uint64_t query_head = index / head_size;
  if (query_head >= query_head_count) {
    return;
  }
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t token = threadIdx.x;
  float partial = 0.0F;
  if (token < logical_token_count) {
    const uint64_t probability_index = query_head * logical_token_count + token;
    const uint64_t value_index =
        (key_value_head * maximum_token_count + token) * head_size + depth;
    partial = __bfloat162float(probabilities[probability_index]) *
              __bfloat162float(value_cache[value_index]);
  }
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    partial += __shfl_down_sync(0xffffffffU, partial, offset);
  }
  if (threadIdx.x == 0) {
    output[index] = __float2bfloat16_rn(partial);
  }
}

__global__ void reviewed_hugging_face_paged_decode_av_short_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    __nv_bfloat16* output, uint64_t physical_block_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t logical_token_count) {
  const uint64_t index = blockIdx.x;
  const uint64_t depth = index % head_size;
  const uint64_t query_head = index / head_size;
  if (query_head >= query_head_count) {
    return;
  }
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t token = threadIdx.x;
  float partial = 0.0F;
  if (token < logical_token_count) {
    uint64_t value_base = 0;
    if (!paged_cache_base(block_ids, valid_tokens, token,
                          physical_block_count, key_value_head,
                          key_value_head_count, head_size, &value_base)) {
      partial = CUDART_NAN_F;
    } else {
      partial = __bfloat162float(
                    probabilities[query_head * logical_token_count + token]) *
                __bfloat162float(value_pool[value_base + depth]);
    }
  }
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    partial += __shfl_down_sync(0xffffffffU, partial, offset);
  }
  if (threadIdx.x == 0) {
    output[index] = __float2bfloat16_rn(partial);
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_decode_qk_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    __nv_bfloat16* scores, uint64_t maximum_token_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t score_element_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = blockIdx.x; index < score_element_count;
       index += gridDim.x) {
    const uint64_t token = index % logical_token_count;
    const uint64_t query_head = index / logical_token_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base = query_head * head_size;
    const uint64_t key_base =
        (key_value_head * maximum_token_count + token) * head_size;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > head_size) {
        end = head_size;
      }
      float accumulator = 0.0F;
      for (uint64_t depth = begin; depth < end; ++depth) {
        accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                           __bfloat162float(key_cache[key_base + depth]),
                           accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float dot =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      scores[index] = __float2bfloat16_rn(dot);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_paged_decode_qk_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    __nv_bfloat16* scores, uint64_t physical_block_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t score_element_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = blockIdx.x; index < score_element_count;
       index += gridDim.x) {
    const uint64_t logical_token = index % logical_token_count;
    const uint64_t query_head = index / logical_token_count;
    const uint64_t key_value_head = query_head / group_size;
    uint64_t key_base = 0;
    const bool valid = paged_cache_base(
        block_ids, valid_tokens, logical_token, physical_block_count,
        key_value_head, key_value_head_count, head_size, &key_base);
    if (!valid) {
      if (threadIdx.x == 0) {
        scores[index] = __float2bfloat16_rn(CUDART_NAN_F);
      }
      __syncthreads();
      continue;
    }
    const uint64_t query_base = query_head * head_size;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > head_size) {
        end = head_size;
      }
      float accumulator = 0.0F;
      for (uint64_t depth = begin; depth < end; ++depth) {
        accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                           __bfloat162float(key_pool[key_base + depth]),
                           accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float dot =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      scores[index] = __float2bfloat16_rn(dot);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_decode_softmax_kernel(
    __nv_bfloat16* scores, uint64_t logical_token_count,
    uint64_t query_head_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  __shared__ uint32_t has_nan;
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  for (uint64_t query_head = blockIdx.x; query_head < query_head_count;
       query_head += gridDim.x) {
    if (threadIdx.x == 0) {
      has_nan = 0;
    }
    __syncthreads();
    const uint64_t base = query_head * logical_token_count;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > logical_token_count) {
        end = logical_token_count;
      }
      float maximum = -CUDART_INF_F;
      bool local_nan = false;
      for (uint64_t token = begin; token < end; ++token) {
        const float score = __bfloat162float(scores[base + token]);
        local_nan = local_nan || isnan(score);
        maximum = fmaxf(maximum, score);
      }
      if (local_nan) {
        atomicOr(&has_nan, 1U);
      }
      first[chunk] = maximum;
    }
    __syncthreads();
    const float maximum =
        rustinfer_cuda_fixed37::balanced_max(first, second, partial_count);
    if (has_nan != 0 || !isfinite(maximum)) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t token = threadIdx.x; token < logical_token_count;
           token += blockDim.x) {
        scores[base + token] = nan;
      }
      __syncthreads();
      continue;
    }
    __syncthreads();
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > logical_token_count) {
        end = logical_token_count;
      }
      float sum = 0.0F;
      for (uint64_t token = begin; token < end; ++token) {
        sum = __fadd_rn(
            sum, expf(__fsub_rn(__bfloat162float(scores[base + token]),
                                maximum)));
      }
      first[chunk] = sum;
    }
    __syncthreads();
    const float denominator =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    for (uint64_t token = threadIdx.x; token < logical_token_count;
         token += blockDim.x) {
      const float numerator =
          expf(__fsub_rn(__bfloat162float(scores[base + token]), maximum));
      scores[base + token] =
          __float2bfloat16_rn(numerator / denominator);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_decode_av_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_cache,
    __nv_bfloat16* output, uint64_t maximum_token_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t output_element_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = blockIdx.x; index < output_element_count;
       index += gridDim.x) {
    const uint64_t depth = index % head_size;
    const uint64_t query_head = index / head_size;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base = query_head * logical_token_count;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > logical_token_count) {
        end = logical_token_count;
      }
      float accumulator = 0.0F;
      for (uint64_t token = begin; token < end; ++token) {
        const uint64_t value_index =
            (key_value_head * maximum_token_count + token) * head_size + depth;
        accumulator = fmaf(
            __bfloat162float(probabilities[probability_base + token]),
            __bfloat162float(value_cache[value_index]), accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float result =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      output[index] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_paged_decode_av_kernel(
    const __nv_bfloat16* probabilities, const __nv_bfloat16* value_pool,
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    __nv_bfloat16* output, uint64_t physical_block_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t output_element_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = blockIdx.x; index < output_element_count;
       index += gridDim.x) {
    const uint64_t depth = index % head_size;
    const uint64_t query_head = index / head_size;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base = query_head * logical_token_count;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > logical_token_count) {
        end = logical_token_count;
      }
      float accumulator = 0.0F;
      for (uint64_t logical_token = begin; logical_token < end;
           ++logical_token) {
        uint64_t value_base = 0;
        if (!paged_cache_base(
                block_ids, valid_tokens, logical_token, physical_block_count,
                key_value_head, key_value_head_count, head_size, &value_base)) {
          accumulator = CUDART_NAN_F;
          break;
        }
        accumulator = fmaf(
            __bfloat162float(
                probabilities[probability_base + logical_token]),
            __bfloat162float(value_pool[value_base + depth]), accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float result =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      output[index] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

__device__ __forceinline__ float staged_decode_score(float dot_product,
                                                      float scale) {
  const __nv_bfloat16 staged_dot = __float2bfloat16_rn(dot_product);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(staged_dot) * scale));
}

template <bool kPaged>
__device__ __forceinline__ bool fixed37_two_pass_cache_base(
    const uint32_t* block_ids, const uint16_t* valid_tokens,
    uint64_t logical_token, uint64_t maximum_token_count,
    uint64_t physical_block_count, uint64_t key_value_head,
    uint64_t key_value_head_count, uint64_t* output) {
  if constexpr (kPaged) {
    return paged_cache_base(
        block_ids, valid_tokens, logical_token, physical_block_count,
        key_value_head, key_value_head_count, kFixed37TwoPassHeadSize,
        output);
  }
  *output =
      (key_value_head * maximum_token_count + logical_token) *
      kFixed37TwoPassHeadSize;
  return true;
}

template <bool kPaged>
__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_two_pass_decode_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const uint32_t* block_ids,
    const uint16_t* valid_tokens, __nv_bfloat16* output,
    uint64_t maximum_token_count, uint64_t physical_block_count,
    uint64_t logical_token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale,
    uint64_t token_partial_count) {
  extern __shared__ float shared_values[];
  __shared__ uint32_t has_nan;
  float* values = shared_values;
  float* first = values + logical_token_count;
  const uint64_t partial_capacity =
      token_partial_count < kFixed37TwoPassDepthPartialCount
          ? kFixed37TwoPassDepthPartialCount
          : token_partial_count;
  float* second = first + partial_capacity;
  const uint64_t group_size = query_head_count / key_value_head_count;

  for (uint64_t query_head = blockIdx.x; query_head < query_head_count;
       query_head += gridDim.x) {
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base = query_head * kFixed37TwoPassHeadSize;
    if (threadIdx.x == 0) {
      has_nan = 0;
    }
    __syncthreads();

    // Pass one materializes only one shared maximum per logical 37-token
    // chunk. Each score reproduces raw-BF16 then scaled-BF16 staging.
    float chunk_maximum = -CUDART_INF_F;
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      uint64_t key_base = 0;
      const bool valid = fixed37_two_pass_cache_base<kPaged>(
          block_ids, valid_tokens, token, maximum_token_count,
          physical_block_count, key_value_head, key_value_head_count,
          &key_base);
      for (uint64_t chunk = threadIdx.x;
           chunk < kFixed37TwoPassDepthPartialCount;
           chunk += blockDim.x) {
        const uint64_t begin =
            chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > kFixed37TwoPassHeadSize) {
          end = kFixed37TwoPassHeadSize;
        }
        float accumulator = valid ? 0.0F : CUDART_NAN_F;
        if (valid) {
          for (uint64_t depth = begin; depth < end; ++depth) {
            accumulator = fmaf(
                __bfloat162float(query[query_base + depth]),
                __bfloat162float(key[key_base + depth]), accumulator);
          }
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float dot = rustinfer_cuda_fixed37::balanced_sum(
          first, second, kFixed37TwoPassDepthPartialCount);
      if (threadIdx.x == 0) {
        const float score = staged_decode_score(dot, scale);
        if (isnan(score)) {
          has_nan = 1;
        }
        if (token % rustinfer_cuda_fixed37::kChunkElements == 0) {
          chunk_maximum = -CUDART_INF_F;
        }
        chunk_maximum = fmaxf(chunk_maximum, score);
        if (token % rustinfer_cuda_fixed37::kChunkElements ==
                rustinfer_cuda_fixed37::kChunkElements - 1 ||
            token + 1 == logical_token_count) {
          values[token / rustinfer_cuda_fixed37::kChunkElements] =
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
    const float maximum = rustinfer_cuda_fixed37::balanced_max(
        first, second, token_partial_count);
    if (has_nan != 0 || !isfinite(maximum)) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t depth = threadIdx.x;
           depth < kFixed37TwoPassHeadSize; depth += blockDim.x) {
        output[query_base + depth] = nan;
      }
      __syncthreads();
      continue;
    }
    __syncthreads();

    // Pass two reevaluates every score, keeps exp(T) in shared memory, then
    // narrows probabilities to BF16 before the fixed37 AV reduction.
    for (uint64_t token = 0; token < logical_token_count; ++token) {
      uint64_t key_base = 0;
      const bool valid = fixed37_two_pass_cache_base<kPaged>(
          block_ids, valid_tokens, token, maximum_token_count,
          physical_block_count, key_value_head, key_value_head_count,
          &key_base);
      for (uint64_t chunk = threadIdx.x;
           chunk < kFixed37TwoPassDepthPartialCount;
           chunk += blockDim.x) {
        const uint64_t begin =
            chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > kFixed37TwoPassHeadSize) {
          end = kFixed37TwoPassHeadSize;
        }
        float accumulator = valid ? 0.0F : CUDART_NAN_F;
        if (valid) {
          for (uint64_t depth = begin; depth < end; ++depth) {
            accumulator = fmaf(
                __bfloat162float(query[query_base + depth]),
                __bfloat162float(key[key_base + depth]), accumulator);
          }
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float dot = rustinfer_cuda_fixed37::balanced_sum(
          first, second, kFixed37TwoPassDepthPartialCount);
      if (threadIdx.x == 0) {
        values[token] = expf(__fsub_rn(staged_decode_score(dot, scale),
                                       maximum));
      }
      __syncthreads();
    }

    for (uint64_t chunk = threadIdx.x; chunk < token_partial_count;
         chunk += blockDim.x) {
      const uint64_t begin =
          chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
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
    const float denominator = rustinfer_cuda_fixed37::balanced_sum(
        first, second, token_partial_count);
    for (uint64_t token = threadIdx.x; token < logical_token_count;
         token += blockDim.x) {
      const __nv_bfloat16 probability =
          __float2bfloat16_rn(values[token] / denominator);
      values[token] = __bfloat162float(probability);
    }
    __syncthreads();

    for (uint64_t depth = 0; depth < kFixed37TwoPassHeadSize; ++depth) {
      for (uint64_t chunk = threadIdx.x; chunk < token_partial_count;
           chunk += blockDim.x) {
        const uint64_t begin =
            chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > logical_token_count) {
          end = logical_token_count;
        }
        float accumulator = 0.0F;
        for (uint64_t token = begin; token < end; ++token) {
          uint64_t value_base = 0;
          if (!fixed37_two_pass_cache_base<kPaged>(
                  block_ids, valid_tokens, token, maximum_token_count,
                  physical_block_count, key_value_head,
                  key_value_head_count, &value_base)) {
            accumulator = CUDART_NAN_F;
            break;
          }
          accumulator = fmaf(
              values[token],
              __bfloat162float(value[value_base + depth]), accumulator);
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float result = rustinfer_cuda_fixed37::balanced_sum(
          first, second, token_partial_count);
      if (threadIdx.x == 0) {
        output[query_base + depth] = __float2bfloat16_rn(result);
      }
      __syncthreads();
    }
  }
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

__global__ __launch_bounds__(kWarpSize) void decode_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_cache,
    const __nv_bfloat16* value_cache, float* partial_states,
    uint64_t maximum_token_count, uint64_t logical_token_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t tokens_per_partition, float scale) {
  const uint32_t lane = threadIdx.x;
  const uint64_t partition = blockIdx.x;
  const uint64_t query_head = blockIdx.y;
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base = query_head * kOptimizedHeadSize;
  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high =
      __bfloat162float(query[query_base + lane + kWarpSize]);

  const uint64_t token_begin = partition * tokens_per_partition;
  const uint64_t remaining = logical_token_count - token_begin;
  const uint64_t token_count =
      remaining < tokens_per_partition ? remaining : tokens_per_partition;
  const uint64_t token_end = token_begin + token_count;

  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;
  float numerator_low = 0.0F;
  float numerator_high = 0.0F;
  for (uint64_t token = token_begin; token < token_end; ++token) {
    const uint64_t cache_base =
        (key_value_head * maximum_token_count + token) * kOptimizedHeadSize;
    const float key_low = __bfloat162float(key_cache[cache_base + lane]);
    const float key_high =
        __bfloat162float(key_cache[cache_base + lane + kWarpSize]);
    float score = fmaf(query_low, key_low, query_high * key_high);
    score = warp_sum(score);
    score = staged_decode_score(
        __shfl_sync(kFullWarpMask, score, 0), scale);

    float alpha = 0.0F;
    float beta = 0.0F;
    if (lane == 0) {
      update_online_state(score, &maximum, &denominator, &alpha, &beta);
    }
    alpha = __shfl_sync(kFullWarpMask, alpha, 0);
    beta = __shfl_sync(kFullWarpMask, beta, 0);
    numerator_low = update_numerator(
        numerator_low, __bfloat162float(value_cache[cache_base + lane]),
        alpha, beta);
    numerator_high = update_numerator(
        numerator_high,
        __bfloat162float(value_cache[cache_base + lane + kWarpSize]), alpha,
        beta);
  }

  const uint64_t state_base =
      (partition * query_head_count + query_head) *
      kOptimizedStateStride;
  if (lane == 0) {
    partial_states[state_base] = maximum;
    partial_states[state_base + 1] = denominator;
  }
  partial_states[state_base + 2 + lane] = numerator_low;
  partial_states[state_base + 2 + lane + kWarpSize] = numerator_high;
}

__global__ __launch_bounds__(kWarpSize) void
paged_decode_partial_state_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key_pool,
    const __nv_bfloat16* value_pool, const uint32_t* block_ids,
    const uint16_t* valid_tokens, float* partial_states,
    uint64_t physical_block_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale) {
  const uint32_t lane = threadIdx.x;
  const uint64_t logical_block = blockIdx.x;
  const uint64_t query_head = blockIdx.y;
  const uint64_t group_size = query_head_count / key_value_head_count;
  const uint64_t key_value_head = query_head / group_size;
  const uint64_t query_base = query_head * kOptimizedHeadSize;
  const uint64_t physical_block = block_ids[logical_block];
  const uint64_t token_count = valid_tokens[logical_block];
  const bool valid_block =
      physical_block < physical_block_count && token_count != 0 &&
      token_count <= RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE;
  const uint64_t block_head_base =
      (physical_block * key_value_head_count + key_value_head) *
      RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE * kOptimizedHeadSize;

  const float query_low = __bfloat162float(query[query_base + lane]);
  const float query_high =
      __bfloat162float(query[query_base + lane + kWarpSize]);
  float maximum = valid_block ? -CUDART_INF_F : CUDART_NAN_F;
  float denominator = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator_low = valid_block ? 0.0F : CUDART_NAN_F;
  float numerator_high = valid_block ? 0.0F : CUDART_NAN_F;
  if (valid_block) {
    for (uint64_t token_in_block = 0; token_in_block < token_count;
         ++token_in_block) {
      const uint64_t cache_base =
          block_head_base + token_in_block * kOptimizedHeadSize;
      const float key_low =
          __bfloat162float(key_pool[cache_base + lane]);
      const float key_high = __bfloat162float(
          key_pool[cache_base + lane + kWarpSize]);
      float score = fmaf(query_low, key_low, query_high * key_high);
      score = warp_sum(score);
      score = staged_decode_score(
          __shfl_sync(kFullWarpMask, score, 0), scale);

      float alpha = 0.0F;
      float beta = 0.0F;
      if (lane == 0) {
        update_online_state(score, &maximum, &denominator, &alpha, &beta);
      }
      alpha = __shfl_sync(kFullWarpMask, alpha, 0);
      beta = __shfl_sync(kFullWarpMask, beta, 0);
      numerator_low = update_numerator(
          numerator_low, __bfloat162float(value_pool[cache_base + lane]),
          alpha, beta);
      numerator_high = update_numerator(
          numerator_high,
          __bfloat162float(value_pool[cache_base + lane + kWarpSize]), alpha,
          beta);
    }
  }

  const uint64_t state_base =
      (logical_block * query_head_count + query_head) *
      kOptimizedStateStride;
  if (lane == 0) {
    partial_states[state_base] = maximum;
    partial_states[state_base + 1] = denominator;
  }
  partial_states[state_base + 2 + lane] = numerator_low;
  partial_states[state_base + 2 + lane + kWarpSize] = numerator_high;
}

__device__ __forceinline__ void merge_partial_component(
    float other_maximum, float other_denominator, float other_numerator,
    float* maximum, float* denominator, float* numerator) {
  if (other_denominator == 0.0F) {
    return;
  }
  if (*denominator == 0.0F) {
    *maximum = other_maximum;
    *denominator = other_denominator;
    *numerator = other_numerator;
    return;
  }
  if (isnan(*maximum) || isnan(other_maximum) || isnan(*denominator) ||
      isnan(other_denominator)) {
    *maximum = CUDART_NAN_F;
    *denominator = CUDART_NAN_F;
    *numerator = CUDART_NAN_F;
    return;
  }

  const bool left_positive_infinity = isinf(*maximum) && *maximum > 0.0F;
  const bool right_positive_infinity =
      isinf(other_maximum) && other_maximum > 0.0F;
  if (left_positive_infinity || right_positive_infinity) {
    if (left_positive_infinity && right_positive_infinity) {
      *denominator += other_denominator;
      *numerator += other_numerator;
    } else if (right_positive_infinity) {
      *maximum = other_maximum;
      *denominator = other_denominator;
      *numerator = other_numerator;
    }
    return;
  }

  const float next_maximum = fmaxf(*maximum, other_maximum);
  const float left_scale = expf(*maximum - next_maximum);
  const float right_scale = expf(other_maximum - next_maximum);
  *denominator = fmaf(left_scale, *denominator,
                      right_scale * other_denominator);
  if (right_scale == 0.0F) {
    *numerator = left_scale * *numerator;
  } else if (left_scale == 0.0F) {
    *numerator = right_scale * other_numerator;
  } else {
    *numerator =
        fmaf(right_scale, other_numerator, left_scale * *numerator);
  }
  *maximum = next_maximum;
}

__global__ void decode_partial_state_reduce_kernel(
    const float* partial_states, __nv_bfloat16* output,
    uint64_t partial_state_count, uint64_t query_head_count,
    uint64_t head_size, uint32_t reduction_order) {
  const uint64_t state_stride = head_size + 2;
  for (uint64_t query_head = blockIdx.x; query_head < query_head_count;
       query_head += gridDim.x) {
    for (uint64_t depth = threadIdx.x; depth < head_size;
         depth += blockDim.x) {
      float maximum = -CUDART_INF_F;
      float denominator = 0.0F;
      float numerator = 0.0F;
      for (uint64_t ordinal = 0; ordinal < partial_state_count; ++ordinal) {
        const uint64_t partition =
            reduction_order == RUSTINFER_CUDA_DECODE_REDUCTION_ASCENDING
                ? ordinal
                : partial_state_count - 1 - ordinal;
        const uint64_t state_base =
            (partition * query_head_count + query_head) * state_stride;
        merge_partial_component(
            partial_states[state_base], partial_states[state_base + 1],
            partial_states[state_base + 2 + depth], &maximum, &denominator,
            &numerator);
      }
      // Match the prefill online kernel's single final normalization so the
      // one-partition decode path does not introduce an avoidable rounding
      // difference before the BF16 cast.
      const float inverse_denominator =
          isnan(denominator)
              ? CUDART_NAN_F
              : (denominator > 0.0F ? 1.0F / denominator : 0.0F);
      const float normalized = numerator * inverse_denominator;
      output[query_head * head_size + depth] =
          __float2bfloat16_rn(normalized);
    }
  }
}

RustInferCudaStatus resolve_decode_inputs(
    const RustInferCudaBufferSpan& query_span,
    const RustInferCudaBufferSpan& key_cache_span,
    const RustInferCudaBufferSpan& value_cache_span,
    const RustInferCudaBufferSpan& output_span,
    const DecodeByteCounts& bytes, ResolvedSpan* query,
    ResolvedSpan* key_cache, ResolvedSpan* value_cache, ResolvedSpan* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  RustInferCudaStatus status = resolve_span(
      query_span, RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.query_output, query,
      error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(key_cache_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, key_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(value_cache_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, value_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(output_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, output, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *query, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *key_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *value_cache, error, operation);
  }
  return status;
}

RustInferCudaStatus resolve_decode_two_pass_inputs(
    const RustInferCudaBufferSpan& query_span,
    const RustInferCudaBufferSpan& key_cache_span,
    const RustInferCudaBufferSpan& value_cache_span,
    const RustInferCudaBufferSpan& output_span,
    const DecodeTwoPassByteCounts& bytes, ResolvedSpan* query,
    ResolvedSpan* key_cache, ResolvedSpan* value_cache, ResolvedSpan* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  RustInferCudaStatus status = resolve_span(
      query_span, RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.query_output, query,
      error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(key_cache_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, key_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(value_cache_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, value_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(output_span, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, output, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *query, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *key_cache, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(*output, *value_cache, error, operation);
  }
  return status;
}

void launch_partial_state_reducer(const float* partial_states,
                                  __nv_bfloat16* output,
                                  uint64_t partial_state_count,
                                  uint64_t query_head_count,
                                  uint64_t head_size,
                                  uint32_t reduction_order,
                                  cudaStream_t stream) {
  decode_partial_state_reduce_kernel
      <<<block_count(query_head_count), kThreads, 0, stream>>>(
          partial_states, output, partial_state_count, query_head_count,
          head_size, reduction_order);
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_kv_cache_write_execute(
    const RustInferCudaKvCacheWriteParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "write contiguous KV cache";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaKvCacheWriteParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status =
      validate_cache_write_dimensions(*params, error, kOperation);
  CacheByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cache_byte_counts(
        params->source_token_count, params->maximum_token_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan key_source{};
  ResolvedSpan value_source{};
  ResolvedSpan key_cache{};
  ResolvedSpan value_cache{};
  status = resolve_span(params->key_source, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        bytes.source, &key_source, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_source, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.source, &value_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_cache, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, &key_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_cache, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.cache, &value_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_cache, key_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_cache, value_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_cache, value_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_cache, key_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_cache, value_source, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {key_source, value_source, key_cache,
                                value_cache};
  status = validate_contexts(stream, spans, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(key_source.buffer) || !uses.add(value_source.buffer) ||
      !uses.add(key_cache.buffer) || !uses.add(value_cache.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const uint64_t element_count = bytes.source / 2;
    launch_attempted = true;
    kv_cache_write_kernel
        <<<block_count(element_count), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key_source.data),
            reinterpret_cast<const __nv_bfloat16*>(value_source.data),
            reinterpret_cast<__nv_bfloat16*>(key_cache.data),
            reinterpret_cast<__nv_bfloat16*>(value_cache.data),
            params->destination_token_start, params->maximum_token_count,
            params->key_value_head_count, params->head_size, element_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_decode_attention_reference_execute(
    const RustInferCudaDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute reference decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaDecodeAttentionReferenceParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status = validate_decode_dimensions(
      params->maximum_token_count, params->logical_token_count,
      params->query_head_count, params->key_value_head_count,
      params->head_size, params->scale, error, kOperation);
  DecodeByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = decode_byte_counts(
        params->maximum_token_count, params->logical_token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, &bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_cache{};
  ResolvedSpan value_cache{};
  ResolvedSpan score_workspace{};
  ResolvedSpan output{};
  status = resolve_decode_inputs(
      params->query, params->key_cache, params->value_cache, params->output,
      bytes, &query, &key_cache, &value_cache, &output, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->score_workspace,
                          RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.scores,
                          &score_workspace, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, query, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, key_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, value_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, output, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query, key_cache, value_cache,
                                score_workspace, output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_cache.buffer) ||
      !uses.add(value_cache.buffer) || !uses.add(score_workspace.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  const uint64_t score_elements = bytes.scores / 2;
  const uint64_t output_elements = bytes.query_output / 2;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    decode_qk_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data),
            reinterpret_cast<const __nv_bfloat16*>(key_cache.data),
            reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
            params->maximum_token_count, params->logical_token_count,
            params->query_head_count, params->key_value_head_count,
            params->head_size, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
            params->scale, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_softmax_reference_kernel
        <<<block_count(params->query_head_count), kThreads, 0,
           stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
            params->logical_token_count, params->query_head_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_av_reference_kernel
        <<<block_count(output_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(score_workspace.data),
            reinterpret_cast<const __nv_bfloat16*>(value_cache.data),
            reinterpret_cast<__nv_bfloat16*>(output.data),
            params->maximum_token_count, params->logical_token_count,
            params->query_head_count, params->key_value_head_count,
            params->head_size, output_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_decode_attention_reference_execute(
    const RustInferCudaDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 materialized decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaDecodeAttentionReferenceParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status = validate_decode_dimensions(
      params->maximum_token_count, params->logical_token_count,
      params->query_head_count, params->key_value_head_count,
      params->head_size, params->scale, error, kOperation);
  uint64_t depth_partial_count = 0;
  uint64_t depth_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(params->head_size, &depth_partial_count,
                                   &depth_shared_bytes, error, kOperation);
  }
  uint64_t token_partial_count = 0;
  uint64_t token_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(
        params->logical_token_count, &token_partial_count, &token_shared_bytes,
        error, kOperation);
  }
  DecodeByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = decode_byte_counts(
        params->maximum_token_count, params->logical_token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, &bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_cache{};
  ResolvedSpan value_cache{};
  ResolvedSpan score_workspace{};
  ResolvedSpan output{};
  status = resolve_decode_inputs(
      params->query, params->key_cache, params->value_cache, params->output,
      bytes, &query, &key_cache, &value_cache, &output, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->score_workspace,
                          RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.scores,
                          &score_workspace, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, query, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, key_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, value_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(score_workspace, output, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query, key_cache, value_cache,
                                score_workspace, output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_cache.buffer) ||
      !uses.add(value_cache.buffer) || !uses.add(score_workspace.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  const uint64_t score_elements = bytes.scores / 2;
  const uint64_t output_elements = bytes.query_output / 2;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    fixed37_decode_qk_kernel<<<
        rustinfer_cuda_fixed37::block_count(score_elements),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(depth_shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_cache.data),
        reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
        params->maximum_token_count, params->logical_token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, score_elements, depth_partial_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
            params->scale, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    fixed37_decode_softmax_kernel<<<
        rustinfer_cuda_fixed37::block_count(params->query_head_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(token_shared_bytes), stream->stream>>>(
        reinterpret_cast<__nv_bfloat16*>(score_workspace.data),
        params->logical_token_count, params->query_head_count,
        token_partial_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    fixed37_decode_av_kernel<<<
        rustinfer_cuda_fixed37::block_count(output_elements),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(token_shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(score_workspace.data),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->maximum_token_count, params->logical_token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, output_elements, token_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_decode_attention_two_pass_execute(
    const RustInferCudaDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 two-pass decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaDecodeAttentionReferenceParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status = validate_decode_dimensions(
      params->maximum_token_count, params->logical_token_count,
      params->query_head_count, params->key_value_head_count,
      params->head_size, params->scale, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->head_size != kFixed37TwoPassHeadSize) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 two-pass decode supports head_size=64 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->logical_token_count > kFixed37TwoPassMaximumTokenCount) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 two-pass decode supports logical T<=8192 only");
  }
  uint64_t token_partial_count = 0;
  uint64_t unused_reduction_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(
        params->logical_token_count, &token_partial_count,
        &unused_reduction_shared_bytes, error, kOperation);
  }
  uint64_t shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = fixed37_two_pass_shared_bytes(
        params->logical_token_count, token_partial_count, &shared_bytes,
        error, kOperation);
  }
  DecodeTwoPassByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = decode_two_pass_byte_counts(
        params->maximum_token_count, params->query_head_count,
        params->key_value_head_count, &bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_cache{};
  ResolvedSpan value_cache{};
  ResolvedSpan output{};
  status = resolve_decode_two_pass_inputs(
      params->query, params->key_cache, params->value_cache, params->output,
      bytes, &query, &key_cache, &value_cache, &output, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query, key_cache, value_cache, output};
  status = validate_contexts(stream, spans, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_cache.buffer) ||
      !uses.add(value_cache.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    fixed37_two_pass_decode_kernel<false><<<
        rustinfer_cuda_fixed37::block_count(params->query_head_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_cache.data),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data), nullptr,
        nullptr, reinterpret_cast<__nv_bfloat16*>(output.data),
        params->maximum_token_count, 0, params->logical_token_count,
        params->query_head_count, params->key_value_head_count, params->scale,
        token_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_decode_attention_execute(
    const RustInferCudaDecodeAttentionParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute partitioned decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaDecodeAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status = validate_decode_dimensions(
      params->maximum_token_count, params->logical_token_count,
      params->query_head_count, params->key_value_head_count,
      params->head_size, params->scale, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->head_size != kOptimizedHeadSize) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "partitioned decode supports head_size=64 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (params->tokens_per_partition == 0 ||
       params->partial_state_capacity == 0)) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "tokens_per_partition and partial_state_capacity must be greater than zero");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_reduction_order(params->reduction_order, error,
                                      kOperation);
  }
  uint64_t partial_state_count = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    partial_state_count =
        ((params->logical_token_count - 1) / params->tokens_per_partition) +
        1;
    if (partial_state_count > params->partial_state_capacity) {
      status = validation_error(
          error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
          "partial-state capacity is smaller than the required partition count");
    } else if (partial_state_count > kMaximumGridX ||
               params->query_head_count > kMaximumGridYOrZ) {
      status = validation_error(
          error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
          "partitioned decode launch dimensions exceed the CUDA grid contract");
    }
  }

  DecodeByteCounts bytes{};
  uint64_t states_bytes = 0;
  const bool use_hugging_face_short_decode =
      status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_reviewed_hugging_face_short_decode(
          params->logical_token_count, params->query_head_count,
          params->key_value_head_count, params->head_size);
  uint64_t maximum_short_score_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = decode_byte_counts(
        params->maximum_token_count, params->logical_token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, &bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = partial_state_bytes(
        params->partial_state_capacity, params->query_head_count,
        params->head_size, &states_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode &&
      !checked_product3(params->query_head_count,
                        kHuggingFaceShortDecodeMaximumTokenCount,
                        sizeof(__nv_bfloat16),
                        &maximum_short_score_bytes)) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "short decode maximum score workspace size overflows uint64_t");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_cache{};
  ResolvedSpan value_cache{};
  ResolvedSpan partial_states{};
  ResolvedSpan output{};
  status = resolve_decode_inputs(
      params->query, params->key_cache, params->value_cache, params->output,
      bytes, &query, &key_cache, &value_cache, &output, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->partial_states,
                          RUSTINFER_CUDA_DTYPE_F32, 4, states_bytes,
                          &partial_states, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(partial_states, query, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(partial_states, key_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(partial_states, value_cache, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(partial_states, output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    status = validate_hugging_face_short_workspace_prefix(
        partial_states, bytes.scores, maximum_short_score_bytes,
        states_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query, key_cache, value_cache, partial_states,
                                output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_cache.buffer) ||
      !uses.add(value_cache.buffer) || !uses.add(partial_states.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (use_hugging_face_short_decode) {
      const uint64_t score_elements = bytes.scores / sizeof(__nv_bfloat16);
      decode_qk_reference_kernel
          <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(query.data),
              reinterpret_cast<const __nv_bfloat16*>(key_cache.data),
              reinterpret_cast<__nv_bfloat16*>(partial_states.data),
              params->maximum_token_count, params->logical_token_count,
              params->query_head_count, params->key_value_head_count,
              params->head_size, score_elements);
    } else {
      const dim3 grid(static_cast<uint32_t>(partial_state_count),
                      static_cast<uint32_t>(params->query_head_count));
      decode_partial_state_kernel<<<grid, kWarpSize, 0, stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_cache.data),
          reinterpret_cast<const __nv_bfloat16*>(value_cache.data),
          reinterpret_cast<float*>(partial_states.data),
          params->maximum_token_count, params->logical_token_count,
          params->query_head_count, params->key_value_head_count,
          params->tokens_per_partition, params->scale);
    }
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    const uint64_t score_elements = bytes.scores / sizeof(__nv_bfloat16);
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(partial_states.data),
            params->scale, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    decode_softmax_reference_kernel
        <<<block_count(params->query_head_count), kThreads, 0,
           stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(partial_states.data),
            params->logical_token_count, params->query_head_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    const uint64_t output_elements = bytes.query_output / sizeof(__nv_bfloat16);
    reviewed_hugging_face_decode_av_short_kernel
        <<<static_cast<uint32_t>(output_elements), kWarpSize, 0,
           stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(partial_states.data),
            reinterpret_cast<const __nv_bfloat16*>(value_cache.data),
            reinterpret_cast<__nv_bfloat16*>(output.data),
            params->maximum_token_count, params->query_head_count,
            params->key_value_head_count, params->head_size,
            params->logical_token_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      !use_hugging_face_short_decode) {
    launch_partial_state_reducer(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), partial_state_count,
        params->query_head_count, params->head_size, params->reduction_order,
        stream->stream);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_decode_partial_state_reduce_execute(
    const RustInferCudaDecodePartialStateReduceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "reduce decode partial states";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaDecodePartialStateReduceParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }

  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_SUCCESS;
  if (params->partial_state_capacity == 0 ||
      params->query_head_count == 0 || params->head_size == 0) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "all decode reducer dimensions must be greater than zero");
  } else if (params->partial_state_count > params->partial_state_capacity) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "partial_state_count exceeds partial_state_capacity");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_reduction_order(params->reduction_order, error,
                                      kOperation);
  }
  uint64_t states_bytes = 0;
  uint64_t output_elements = 0;
  uint64_t output_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = partial_state_bytes(
        params->partial_state_capacity, params->query_head_count,
        params->head_size, &states_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      !checked_multiply(params->query_head_count, params->head_size,
                        &output_elements)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "decode reducer output shape overflows uint64_t");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, 2, &output_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan partial_states{};
  ResolvedSpan output{};
  status = resolve_span(params->partial_states, RUSTINFER_CUDA_DTYPE_F32, 4,
                        states_bytes, &partial_states, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          output_bytes, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, partial_states, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {partial_states, output};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(partial_states.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    launch_partial_state_reducer(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->partial_state_count, params->query_head_count,
        params->head_size, params->reduction_order, stream->stream);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_paged_kv_cache_write_execute(
    const RustInferCudaPagedKvCacheWriteParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "write paged KV cache";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaPagedKvCacheWriteParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_paged_block_table(params->block_table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (params->source_token_count == 0 ||
       params->key_value_head_count == 0 || params->head_size == 0)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "paged KV write dimensions must be greater than zero");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (params->destination_token_start >
           params->block_table.logical_token_count ||
       params->source_token_count >
           params->block_table.logical_token_count -
               params->destination_token_start)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "paged KV write exceeds the table logical length");
  }

  uint64_t source_elements = 0;
  uint64_t source_bytes = 0;
  PagedByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      !checked_product3(params->source_token_count,
                        params->key_value_head_count, params->head_size,
                        &source_elements)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "paged KV source shape overflows uint64_t");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(source_elements, 2, &source_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = paged_byte_counts(
        params->block_table, params->key_value_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan key_source{};
  ResolvedSpan value_source{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan block_ids{};
  ResolvedSpan valid_tokens{};
  status = resolve_span(params->key_source, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        source_bytes, &key_source, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_source, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          source_bytes, &value_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_paged_block_table(params->block_table, bytes, &block_ids,
                                       &valid_tokens, error, kOperation);
  }
  const ResolvedSpan* const read_spans[] = {
      &key_source, &value_source, &block_ids, &valid_tokens};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 4; ++index) {
    status = reject_overlap(key_pool, *read_spans[index], error, kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 4; ++index) {
    status = reject_overlap(value_pool, *read_spans[index], error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_pool, value_pool, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {key_source, value_source, key_pool,
                                value_pool, block_ids, valid_tokens};
  status = validate_contexts(stream, spans, 6, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(key_source.buffer) || !uses.add(value_source.buffer) ||
      !uses.add(key_pool.buffer) || !uses.add(value_pool.buffer) ||
      !uses.add(block_ids.buffer) || !uses.add(valid_tokens.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    paged_kv_cache_write_kernel
        <<<block_count(source_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key_source.data),
            reinterpret_cast<const __nv_bfloat16*>(value_source.data),
            reinterpret_cast<__nv_bfloat16*>(key_pool.data),
            reinterpret_cast<__nv_bfloat16*>(value_pool.data),
            reinterpret_cast<const uint32_t*>(block_ids.data),
            reinterpret_cast<const uint16_t*>(valid_tokens.data),
            params->destination_token_start,
            params->block_table.physical_block_count,
            params->key_value_head_count, params->head_size,
            source_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_paged_decode_attention_reference_execute(
    const RustInferCudaPagedDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute reference paged decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaPagedDecodeAttentionReferenceParams stable_params =
      *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_paged_block_table(params->block_table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_decode_dimensions(
        params->block_table.logical_token_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, params->scale, error,
        kOperation);
  }
  PagedByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = paged_byte_counts(
        params->block_table, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan scores{};
  ResolvedSpan output{};
  ResolvedSpan block_ids{};
  ResolvedSpan valid_tokens{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        bytes.query_output, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->score_workspace,
                          RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.scores,
                          &scores, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_paged_block_table(params->block_table, bytes, &block_ids,
                                       &valid_tokens, error, kOperation);
  }
  const ResolvedSpan* const inputs[] = {
      &query, &key_pool, &value_pool, &block_ids, &valid_tokens};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(scores, *inputs[index], error, kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, *inputs[index], error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, scores, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query,  key_pool,  value_pool, scores,
                                output, block_ids, valid_tokens};
  status = validate_contexts(stream, spans, 7, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_pool.buffer) ||
      !uses.add(value_pool.buffer) || !uses.add(scores.buffer) ||
      !uses.add(output.buffer) || !uses.add(block_ids.buffer) ||
      !uses.add(valid_tokens.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  const uint64_t score_elements = bytes.scores / 2;
  const uint64_t output_elements = bytes.query_output / 2;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    paged_decode_qk_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(query.data),
            reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
            reinterpret_cast<const uint32_t*>(block_ids.data),
            reinterpret_cast<const uint16_t*>(valid_tokens.data),
            reinterpret_cast<__nv_bfloat16*>(scores.data),
            params->block_table.physical_block_count,
            params->block_table.logical_token_count,
            params->query_head_count, params->key_value_head_count,
            params->head_size, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(scores.data), params->scale,
            score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_softmax_reference_kernel
        <<<block_count(params->query_head_count), kThreads, 0,
           stream->stream>>>(reinterpret_cast<__nv_bfloat16*>(scores.data),
                            params->block_table.logical_token_count,
                            params->query_head_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    paged_decode_av_reference_kernel
        <<<block_count(output_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(scores.data),
            reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
            reinterpret_cast<const uint32_t*>(block_ids.data),
            reinterpret_cast<const uint16_t*>(valid_tokens.data),
            reinterpret_cast<__nv_bfloat16*>(output.data),
            params->block_table.physical_block_count,
            params->block_table.logical_token_count,
            params->query_head_count, params->key_value_head_count,
            params->head_size, output_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_paged_decode_attention_reference_execute(
    const RustInferCudaPagedDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 materialized paged decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaPagedDecodeAttentionReferenceParams stable_params =
      *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_paged_block_table(params->block_table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_decode_dimensions(
        params->block_table.logical_token_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, params->scale, error,
        kOperation);
  }
  uint64_t depth_partial_count = 0;
  uint64_t depth_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(params->head_size, &depth_partial_count,
                                   &depth_shared_bytes, error, kOperation);
  }
  uint64_t token_partial_count = 0;
  uint64_t token_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(
        params->block_table.logical_token_count, &token_partial_count,
        &token_shared_bytes, error, kOperation);
  }
  PagedByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = paged_byte_counts(
        params->block_table, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan scores{};
  ResolvedSpan output{};
  ResolvedSpan block_ids{};
  ResolvedSpan valid_tokens{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        bytes.query_output, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->score_workspace,
                          RUSTINFER_CUDA_DTYPE_BF16, 2, bytes.scores, &scores,
                          error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_paged_block_table(params->block_table, bytes, &block_ids,
                                       &valid_tokens, error, kOperation);
  }
  const ResolvedSpan* const inputs[] = {
      &query, &key_pool, &value_pool, &block_ids, &valid_tokens};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(scores, *inputs[index], error, kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, *inputs[index], error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, scores, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query,  key_pool,  value_pool, scores,
                                output, block_ids, valid_tokens};
  status = validate_contexts(stream, spans, 7, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_pool.buffer) ||
      !uses.add(value_pool.buffer) || !uses.add(scores.buffer) ||
      !uses.add(output.buffer) || !uses.add(block_ids.buffer) ||
      !uses.add(valid_tokens.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  const uint64_t score_elements = bytes.scores / 2;
  const uint64_t output_elements = bytes.query_output / 2;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    fixed37_paged_decode_qk_kernel<<<
        rustinfer_cuda_fixed37::block_count(score_elements),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(depth_shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
        reinterpret_cast<const uint32_t*>(block_ids.data),
        reinterpret_cast<const uint16_t*>(valid_tokens.data),
        reinterpret_cast<__nv_bfloat16*>(scores.data),
        params->block_table.physical_block_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, score_elements,
        depth_partial_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(scores.data), params->scale,
            score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    fixed37_decode_softmax_kernel<<<
        rustinfer_cuda_fixed37::block_count(params->query_head_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(token_shared_bytes), stream->stream>>>(
        reinterpret_cast<__nv_bfloat16*>(scores.data),
        params->block_table.logical_token_count, params->query_head_count,
        token_partial_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    fixed37_paged_decode_av_kernel<<<
        rustinfer_cuda_fixed37::block_count(output_elements),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(token_shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(scores.data),
        reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
        reinterpret_cast<const uint32_t*>(block_ids.data),
        reinterpret_cast<const uint16_t*>(valid_tokens.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->block_table.physical_block_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, output_elements,
        token_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_paged_decode_attention_two_pass_execute(
    const RustInferCudaPagedDecodeAttentionReferenceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 two-pass paged decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaPagedDecodeAttentionReferenceParams stable_params =
      *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_paged_block_table(params->block_table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_decode_dimensions(
        params->block_table.logical_token_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, params->scale, error,
        kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->head_size != kFixed37TwoPassHeadSize) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 two-pass paged decode supports head_size=64 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->block_table.logical_token_count >
          kFixed37TwoPassMaximumTokenCount) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 two-pass paged decode supports logical T<=8192 only");
  }
  uint64_t token_partial_count = 0;
  uint64_t unused_reduction_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_fixed37_axis(
        params->block_table.logical_token_count, &token_partial_count,
        &unused_reduction_shared_bytes, error, kOperation);
  }
  uint64_t shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = fixed37_two_pass_shared_bytes(
        params->block_table.logical_token_count, token_partial_count,
        &shared_bytes, error, kOperation);
  }
  PagedTwoPassByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = paged_two_pass_byte_counts(
        params->block_table, params->query_head_count,
        params->key_value_head_count, &bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan output{};
  ResolvedSpan block_ids{};
  ResolvedSpan valid_tokens{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        bytes.query_output, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->block_table.block_ids,
                          RUSTINFER_CUDA_DTYPE_U32, 4, bytes.block_ids,
                          &block_ids, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->block_table.valid_tokens,
                          RUSTINFER_CUDA_DTYPE_U16, 2, bytes.valid_tokens,
                          &valid_tokens, error, kOperation);
  }
  const ResolvedSpan* const inputs[] = {
      &query, &key_pool, &value_pool, &block_ids, &valid_tokens};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, *inputs[index], error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query,      key_pool, value_pool,
                                output,     block_ids, valid_tokens};
  status = validate_contexts(stream, spans, 6, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_pool.buffer) ||
      !uses.add(value_pool.buffer) || !uses.add(output.buffer) ||
      !uses.add(block_ids.buffer) || !uses.add(valid_tokens.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    fixed37_two_pass_decode_kernel<true><<<
        rustinfer_cuda_fixed37::block_count(params->query_head_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
        reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
        reinterpret_cast<const uint32_t*>(block_ids.data),
        reinterpret_cast<const uint16_t*>(valid_tokens.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), 0,
        params->block_table.physical_block_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->scale, token_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_paged_decode_attention_execute(
    const RustInferCudaPagedDecodeAttentionParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute partitioned paged decode attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaPagedDecodeAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_paged_block_table(params->block_table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_decode_dimensions(
        params->block_table.logical_token_count,
        params->block_table.logical_token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, params->scale, error,
        kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->head_size != kOptimizedHeadSize) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "paged online decode supports head_size=64 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->partial_state_capacity == 0) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "partial_state_capacity must be greater than zero");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->block_table.block_count > params->partial_state_capacity) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "partial-state capacity is smaller than block_count");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (params->block_table.block_count > kMaximumGridX ||
       params->query_head_count > kMaximumGridYOrZ)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "paged online launch dimensions exceed the CUDA grid contract");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_reduction_order(params->reduction_order, error,
                                      kOperation);
  }

  PagedByteCounts bytes{};
  uint64_t states_bytes = 0;
  const bool use_hugging_face_short_decode =
      status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_reviewed_hugging_face_short_decode(
          params->block_table.logical_token_count,
          params->query_head_count, params->key_value_head_count,
          params->head_size);
  uint64_t maximum_short_score_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = paged_byte_counts(
        params->block_table, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = partial_state_bytes(
        params->partial_state_capacity, params->query_head_count,
        params->head_size, &states_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode &&
      !checked_product3(params->query_head_count,
                        kHuggingFaceShortDecodeMaximumTokenCount,
                        sizeof(__nv_bfloat16),
                        &maximum_short_score_bytes)) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "short paged decode maximum score workspace size overflows uint64_t");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan partial_states{};
  ResolvedSpan output{};
  ResolvedSpan block_ids{};
  ResolvedSpan valid_tokens{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16, 2,
                        bytes.query_output, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.pool, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->partial_states,
                          RUSTINFER_CUDA_DTYPE_F32, 4, states_bytes,
                          &partial_states, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16, 2,
                          bytes.query_output, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_paged_block_table(params->block_table, bytes, &block_ids,
                                       &valid_tokens, error, kOperation);
  }
  const ResolvedSpan* const inputs[] = {
      &query, &key_pool, &value_pool, &block_ids, &valid_tokens};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(partial_states, *inputs[index], error,
                            kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, *inputs[index], error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, partial_states, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    status = validate_hugging_face_short_workspace_prefix(
        partial_states, bytes.scores, maximum_short_score_bytes,
        states_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query,          key_pool,  value_pool,
                                partial_states, output,    block_ids,
                                valid_tokens};
  status = validate_contexts(stream, spans, 7, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key_pool.buffer) ||
      !uses.add(value_pool.buffer) || !uses.add(partial_states.buffer) ||
      !uses.add(output.buffer) || !uses.add(block_ids.buffer) ||
      !uses.add(valid_tokens.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "decode buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (use_hugging_face_short_decode) {
      const uint64_t score_elements = bytes.scores / sizeof(__nv_bfloat16);
      paged_decode_qk_reference_kernel
          <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(query.data),
              reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
              reinterpret_cast<const uint32_t*>(block_ids.data),
              reinterpret_cast<const uint16_t*>(valid_tokens.data),
              reinterpret_cast<__nv_bfloat16*>(partial_states.data),
              params->block_table.physical_block_count,
              params->block_table.logical_token_count,
              params->query_head_count, params->key_value_head_count,
              params->head_size, score_elements);
    } else {
      const dim3 grid(static_cast<uint32_t>(params->block_table.block_count),
                      static_cast<uint32_t>(params->query_head_count));
      paged_decode_partial_state_kernel<<<grid, kWarpSize, 0,
                                          stream->stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(query.data),
          reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
          reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
          reinterpret_cast<const uint32_t*>(block_ids.data),
          reinterpret_cast<const uint16_t*>(valid_tokens.data),
          reinterpret_cast<float*>(partial_states.data),
          params->block_table.physical_block_count, params->query_head_count,
          params->key_value_head_count, params->scale);
    }
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    const uint64_t score_elements = bytes.scores / sizeof(__nv_bfloat16);
    decode_scale_reference_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(partial_states.data),
            params->scale, score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    decode_softmax_reference_kernel
        <<<block_count(params->query_head_count), kThreads, 0,
           stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(partial_states.data),
            params->block_table.logical_token_count,
            params->query_head_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      use_hugging_face_short_decode) {
    const uint64_t output_elements = bytes.query_output / sizeof(__nv_bfloat16);
    reviewed_hugging_face_paged_decode_av_short_kernel
        <<<static_cast<uint32_t>(output_elements), kWarpSize, 0,
           stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(partial_states.data),
            reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
            reinterpret_cast<const uint32_t*>(block_ids.data),
            reinterpret_cast<const uint16_t*>(valid_tokens.data),
            reinterpret_cast<__nv_bfloat16*>(output.data),
            params->block_table.physical_block_count,
            params->query_head_count, params->key_value_head_count,
            params->head_size,
            params->block_table.logical_token_count);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      !use_hugging_face_short_decode) {
    launch_partial_state_reducer(
        reinterpret_cast<const float*>(partial_states.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        params->block_table.block_count, params->query_head_count,
        params->head_size, params->reduction_order, stream->stream);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}
