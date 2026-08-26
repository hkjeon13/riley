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
using rustinfer_cuda_internal::command_batch_is_active;
using rustinfer_cuda_internal::command_batch_is_owned_by_current_thread;
using rustinfer_cuda_internal::command_batch_register_use;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_exclusive_use;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::try_acquire_exclusive_use;
using rustinfer_cuda_internal::validation_error;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaximumBlocks = 65535;
constexpr size_t kMaximumBatchBuffers = 9;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
constexpr uint64_t kAttentionHeadSize = 64;
constexpr uint64_t kFixed37RaggedMaximumTokenCount = 8192;
constexpr uint64_t kFixed37RaggedDepthPartialCount =
    rustinfer_cuda_fixed37::chunk_count(kAttentionHeadSize);
constexpr uint64_t kFixed37RaggedMaximumTokenPartialCount =
    rustinfer_cuda_fixed37::chunk_count(
        kFixed37RaggedMaximumTokenCount);
constexpr uint64_t kFixed37RaggedMaximumSharedBytes =
    (kFixed37RaggedMaximumTokenCount +
     2 * kFixed37RaggedMaximumTokenPartialCount) *
    sizeof(float);
constexpr uint64_t kMaximumGridX = 2147483647;
constexpr uint64_t kMaximumGridYOrZ = 65535;

static_assert(sizeof(RustInferCudaIndexedRopeParams) == 320,
              "indexed RoPE ABI size changed");
static_assert(offsetof(RustInferCudaIndexedRopeParams, input) == 8,
              "indexed RoPE input offset changed");
static_assert(offsetof(RustInferCudaIndexedRopeParams, active_row_count) == 248,
              "indexed RoPE dimension offset changed");
static_assert(offsetof(RustInferCudaIndexedRopeParams, reserved) == 288,
              "indexed RoPE reserved tail changed");
static_assert(sizeof(RustInferCudaRowGatherParams) == 208,
              "row gather ABI size changed");
static_assert(offsetof(RustInferCudaRowGatherParams, input_row_count) == 152,
              "row gather dimension offset changed");
static_assert(offsetof(RustInferCudaRowGatherParams, reserved) == 176,
              "row gather reserved tail changed");
static_assert(sizeof(RustInferCudaPackedBatchV1) == 320,
              "packed batch ABI size changed");
static_assert(offsetof(RustInferCudaPackedBatchV1,
                       sequence_block_offsets) == 8,
              "packed batch offsets span changed");
static_assert(offsetof(RustInferCudaPackedBatchV1, sequence_count) == 248,
              "packed batch dimension offset changed");
static_assert(offsetof(RustInferCudaPackedBatchV1, reserved) == 288,
              "packed batch reserved tail changed");
static_assert(sizeof(RustInferCudaRaggedPagedKvCacheWriteParams) == 568,
              "ragged paged KV write ABI size changed");
static_assert(offsetof(RustInferCudaRaggedPagedKvCacheWriteParams, batch) ==
                  200,
              "ragged paged KV write batch offset changed");
static_assert(offsetof(RustInferCudaRaggedPagedKvCacheWriteParams,
                       key_value_head_count) == 520,
              "ragged paged KV write dimension offset changed");
static_assert(sizeof(RustInferCudaRaggedPagedAttentionParams) == 592,
              "ragged paged attention ABI size changed");
static_assert(offsetof(RustInferCudaRaggedPagedAttentionParams, batch) == 200,
              "ragged paged attention batch offset changed");
static_assert(offsetof(RustInferCudaRaggedPagedAttentionParams,
                       output_row_count) == 544,
              "ragged paged attention output-row offset changed");
static_assert(offsetof(RustInferCudaRaggedPagedAttentionParams, scale) == 552,
              "ragged paged attention scale offset changed");
static_assert(sizeof(RustInferCudaFixed37RaggedPagedAttentionParams) == 600,
              "fixed37 ragged paged attention ABI size changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, struct_size) == 0,
    "fixed37 ragged paged attention struct-size offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, reserved0) == 4,
    "fixed37 ragged paged attention reserved0 offset changed");
static_assert(offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, query) ==
                  8,
              "fixed37 ragged paged attention query offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, key_pool) == 56,
    "fixed37 ragged paged attention key-pool offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, value_pool) == 104,
    "fixed37 ragged paged attention value-pool offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, output) == 152,
    "fixed37 ragged paged attention output offset changed");
static_assert(offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, batch) ==
                  200,
              "fixed37 ragged paged attention batch offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams,
             query_head_count) == 520,
    "fixed37 ragged paged attention QH offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams,
             key_value_head_count) == 528,
    "fixed37 ragged paged attention KVH offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, head_size) == 536,
    "fixed37 ragged paged attention head-size offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams,
             output_row_count) == 544,
    "fixed37 ragged paged attention output-row offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams,
             maximum_logical_token_count) == 552,
    "fixed37 ragged paged attention maximum-T offset changed");
static_assert(offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, scale) ==
                  560,
              "fixed37 ragged paged attention scale offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, reserved1) == 564,
    "fixed37 ragged paged attention reserved1 offset changed");
static_assert(
    offsetof(RustInferCudaFixed37RaggedPagedAttentionParams, reserved) == 568,
    "fixed37 ragged paged attention reserved tail changed");
static_assert(kAttentionHeadSize == 2 * kWarpSize,
              "ragged attention lane ownership changed");
static_assert(kFixed37RaggedDepthPartialCount == 2,
              "fixed37 ragged D64 partial shape changed");
static_assert(kFixed37RaggedMaximumTokenPartialCount == 222,
              "fixed37 ragged T8192 partial shape changed");
static_assert(kFixed37RaggedMaximumSharedBytes == 34544,
              "fixed37 ragged maximum shared-memory shape changed");
static_assert(kFixed37RaggedMaximumSharedBytes <= 48 * 1024,
              "fixed37 ragged shared memory exceeds the base CUDA limit");

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
  RustInferCudaDType dtype;
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

bool checked_product4(uint64_t first, uint64_t second, uint64_t third,
                      uint64_t fourth, uint64_t* output) noexcept {
  uint64_t partial = 0;
  return checked_product3(first, second, third, &partial) &&
         checked_multiply(partial, fourth, output);
}

RustInferCudaStatus fixed37_ragged_shared_bytes(
    uint64_t maximum_logical_token_count, uint64_t* token_partial_count,
    uint64_t* shared_bytes, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (token_partial_count == nullptr || shared_bytes == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation,
                          "internal fixed37 ragged shared-memory output is null");
  }
  const uint64_t chunks =
      rustinfer_cuda_fixed37::chunk_count(maximum_logical_token_count);
  if (chunks == 0 ||
      chunks > rustinfer_cuda_fixed37::kMaximumChunkCount) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
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
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "fixed37 ragged shared-memory size overflows uint64_t");
  }
  *token_partial_count = chunks;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
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

uint64_t dtype_size(RustInferCudaDType dtype) noexcept {
  switch (dtype) {
    case RUSTINFER_CUDA_DTYPE_F32:
    case RUSTINFER_CUDA_DTYPE_U32:
      return 4;
    case RUSTINFER_CUDA_DTYPE_BF16:
    case RUSTINFER_CUDA_DTYPE_U16:
      return 2;
    case RUSTINFER_CUDA_DTYPE_U8:
      return 1;
    default:
      return 0;
  }
}

bool arithmetic_dtype(RustInferCudaDType dtype) noexcept {
  return dtype == RUSTINFER_CUDA_DTYPE_F32 ||
         dtype == RUSTINFER_CUDA_DTYPE_BF16;
}

RustInferCudaStatus typed_bytes(uint64_t element_count,
                                RustInferCudaDType dtype, uint64_t* output,
                                RustInferCudaErrorInfo* error,
                                const char* operation) noexcept {
  const uint64_t width = dtype_size(dtype);
  if (width == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "batch span has an unsupported dtype");
  }
  if (!checked_multiply(element_count, width, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "batch byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus resolve_span(const RustInferCudaBufferSpan& span,
                                 RustInferCudaDType expected_dtype,
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
  if (span.dtype != expected_dtype) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span dtype does not match the batch contract");
  }
  const uint64_t alignment = dtype_size(span.dtype);
  if (alignment == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span dtype is invalid");
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
  if (required_bytes != 0 && span.buffer->device_data == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "non-empty span refers to a zero-byte allocation");
  }

  uint8_t* data = nullptr;
  if (span.buffer->device_data != nullptr) {
    data = static_cast<uint8_t*>(span.buffer->device_data) +
           static_cast<size_t>(span.byte_offset);
  }
  *output = ResolvedSpan{span.buffer, data, span.byte_offset, required_bytes,
                         span.dtype};
  return RUSTINFER_CUDA_STATUS_SUCCESS;
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

RustInferCudaStatus reject_overlap(const ResolvedSpan& writable,
                                   const ResolvedSpan& other,
                                   bool exact_alias_allowed,
                                   RustInferCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(writable, other) &&
      !(exact_alias_allowed && exact_alias(writable, other))) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "unsupported writable/touched span overlap");
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
          "stream and batch spans belong to different context owners");
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
        stream_acquired_(false),
        command_batch_(false) {}

  ExclusiveUses(const ExclusiveUses&) = delete;
  ExclusiveUses& operator=(const ExclusiveUses&) = delete;

  bool add(RustInferCudaDeviceBuffer* buffer) noexcept {
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

  RustInferCudaStatus acquire(RustInferCudaErrorInfo* error,
                              const char* operation) noexcept {
    if (command_batch_is_active(stream_)) {
      if (!command_batch_is_owned_by_current_thread(stream_)) {
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
            "an active stream command batch is owned by another thread");
      }
      command_batch_ = true;
      for (size_t index = 0; index < buffer_count_; ++index) {
        const RustInferCudaStatus status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error, operation,
            "a batch buffer already has an active asynchronous use");
        if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
          return status;
        }
      }
      return RUSTINFER_CUDA_STATUS_SUCCESS;
    }
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        release_acquired();
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
            "a batch buffer already has an active asynchronous use");
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

  RustInferCudaStream* stream_;
  RustInferCudaDeviceBuffer* buffers_[kMaximumBatchBuffers];
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

RustInferCudaStatus launch_status(RustInferCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RustInferCudaStatus prior_launch_status(RustInferCudaErrorInfo* error,
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
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  }
  bool completion_confirmed = !launch_attempted;
  RustInferCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t synchronize_result =
        cudaStreamSynchronize(stream->stream);
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

RustInferCudaStatus validate_packed_batch(
    const RustInferCudaPackedBatchV1& batch, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (batch.struct_size < sizeof(batch)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch has an incompatible struct_size");
  }
  if (batch.format_version != RUSTINFER_CUDA_PACKED_BATCH_VERSION ||
      batch.block_size != RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch version or block size is unsupported");
  }
  if (batch.reserved0 != 0 || !reserved_is_zero(batch.reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch reserved fields must be zero");
  }
  if (batch.sequence_count == 0 || batch.block_count == 0 ||
      batch.active_row_count == 0 || batch.physical_block_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch dimensions must be greater than zero");
  }
  constexpr uint64_t kMaximumU32 =
      static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
  if (batch.sequence_count > kMaximumU32 || batch.block_count > kMaximumU32 ||
      batch.active_row_count > kMaximumU32 ||
      batch.physical_block_count > kMaximumU32) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch dimensions exceed U32 metadata range");
  }
  if (batch.block_count > batch.physical_block_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed batch uses more logical blocks than the physical pool");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus resolve_packed_batch(
    const RustInferCudaPackedBatchV1& batch, ResolvedBatch* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal resolved batch is null");
  }
  uint64_t offset_count = 0;
  if (!checked_add(batch.sequence_count, 1, &offset_count)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "packed CSR offset count overflows uint64_t");
  }
  uint64_t offset_bytes = 0;
  uint64_t block_id_bytes = 0;
  uint64_t valid_token_bytes = 0;
  uint64_t row_bytes = 0;
  RustInferCudaStatus status = typed_bytes(
      offset_count, RUSTINFER_CUDA_DTYPE_U32, &offset_bytes, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.block_count, RUSTINFER_CUDA_DTYPE_U32,
                         &block_id_bytes, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.block_count, RUSTINFER_CUDA_DTYPE_U16,
                         &valid_token_bytes, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(batch.active_row_count, RUSTINFER_CUDA_DTYPE_U32,
                         &row_bytes, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.sequence_block_offsets,
                          RUSTINFER_CUDA_DTYPE_U32, offset_bytes,
                          &output->sequence_block_offsets, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.block_ids, RUSTINFER_CUDA_DTYPE_U32,
                          block_id_bytes, &output->block_ids, error,
                          operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.valid_tokens, RUSTINFER_CUDA_DTYPE_U16,
                          valid_token_bytes, &output->valid_tokens, error,
                          operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.row_sequence_slots,
                          RUSTINFER_CUDA_DTYPE_U32, row_bytes,
                          &output->row_sequence_slots, error, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(batch.row_positions, RUSTINFER_CUDA_DTYPE_U32,
                          row_bytes, &output->row_positions, error,
                          operation);
  }
  return status;
}

DeviceBatch device_batch(const RustInferCudaPackedBatchV1& batch,
                         const ResolvedBatch& resolved) noexcept {
  return DeviceBatch{
      reinterpret_cast<const uint32_t*>(
          resolved.sequence_block_offsets.data),
      reinterpret_cast<const uint32_t*>(resolved.block_ids.data),
      reinterpret_cast<const uint16_t*>(resolved.valid_tokens.data),
      reinterpret_cast<const uint32_t*>(resolved.row_sequence_slots.data),
      reinterpret_cast<const uint32_t*>(resolved.row_positions.data),
      batch.sequence_count,
      batch.block_count,
      batch.active_row_count,
      batch.physical_block_count,
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
                                  uint64_t output_element_count) {
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

__device__ __forceinline__ bool resolve_row_cache_base(
    const DeviceBatch& batch, uint64_t row, uint64_t logical_position,
    uint64_t key_value_head, uint64_t key_value_head_count,
    uint64_t head_size, uint64_t* output) {
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
      logical_position / RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE;
  if (logical_block >= block_end - block_begin) {
    return false;
  }
  const uint64_t block_index = block_begin + logical_block;
  const uint64_t token_in_block =
      logical_position % RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE;
  const uint64_t physical_block = batch.block_ids[block_index];
  const uint64_t valid = batch.valid_tokens[block_index];
  if (physical_block >= batch.physical_block_count || valid == 0 ||
      valid > RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE || token_in_block >= valid) {
    return false;
  }
  *output =
      ((physical_block * key_value_head_count + key_value_head) *
           RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE +
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

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset != 0; offset /= 2) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
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
  if (row >= batch.active_row_count) {
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

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
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
  if (row >= batch.active_row_count) {
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
      rustinfer_cuda_fixed37::chunk_count(logical_token_count);
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
          chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
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
    const float dot = rustinfer_cuda_fixed37::balanced_sum(
        first, second, kFixed37RaggedDepthPartialCount);
    if (threadIdx.x == 0) {
      const float score = staged_attention_score(dot, scale);
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
          chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
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
    const float dot = rustinfer_cuda_fixed37::balanced_sum(
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

  for (uint64_t depth = 0; depth < kAttentionHeadSize; ++depth) {
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
    const float result = rustinfer_cuda_fixed37::balanced_sum(
        first, second, token_partial_count);
    if (threadIdx.x == 0) {
      output[output_base + depth] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_indexed_rope_execute(
    const RustInferCudaIndexedRopeParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute row-indexed non-interleaved Llama RoPE";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaIndexedRopeParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype ||
      params->cos.dtype != RUSTINFER_CUDA_DTYPE_F32 ||
      params->sin.dtype != RUSTINFER_CUDA_DTYPE_F32 ||
      params->positions.dtype != RUSTINFER_CUDA_DTYPE_U32) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "indexed RoPE requires matching F32/BF16 tensors, F32 tables, and U32 positions");
  }
  if (params->head_count == 0 || params->head_size == 0 ||
      params->rotary_dimension == 0 ||
      params->rotary_dimension > params->head_size ||
      params->rotary_dimension % 2 != 0) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "head dimensions must be non-zero and rotary_dimension even and <= head_size");
  }
  if (params->active_row_count != 0 &&
      params->table_position_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
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
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "indexed RoPE shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t table_bytes = 0;
  uint64_t position_bytes = 0;
  RustInferCudaStatus status =
      typed_bytes(tensor_elements, params->input.dtype, &tensor_bytes, error,
                  kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(table_elements, RUSTINFER_CUDA_DTYPE_F32,
                         &table_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(params->active_row_count,
                         RUSTINFER_CUDA_DTYPE_U32, &position_bytes, error,
                         kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan input{};
  ResolvedSpan cos{};
  ResolvedSpan sin{};
  ResolvedSpan positions{};
  ResolvedSpan output{};
  status = resolve_span(params->input, params->input.dtype, tensor_bytes,
                        &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->cos, RUSTINFER_CUDA_DTYPE_F32, table_bytes,
                          &cos, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->sin, RUSTINFER_CUDA_DTYPE_F32, table_bytes,
                          &sin, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->positions, RUSTINFER_CUDA_DTYPE_U32,
                          position_bytes, &positions, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, params->input.dtype, tensor_bytes,
                          &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, cos, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, sin, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, positions, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, cos, sin, positions, output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(cos.buffer) ||
      !uses.add(sin.buffer) || !uses.add(positions.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "batch buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->active_row_count == 0) {
    return uses.release_completed()
               ? RUSTINFER_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (params->input.dtype == RUSTINFER_CUDA_DTYPE_F32) {
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

extern "C" RustInferCudaStatus rustinfer_cuda_row_gather_execute(
    const RustInferCudaRowGatherParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute row gather";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaRowGatherParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype ||
      params->row_indices.dtype != RUSTINFER_CUDA_DTYPE_U32) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "row gather requires matching F32/BF16 tensors and U32 indices");
  }
  if (params->input_row_count == 0 || params->column_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "input_row_count and column_count must be greater than zero");
  }
  uint64_t input_elements = 0;
  uint64_t output_elements = 0;
  if (!checked_multiply(params->input_row_count, params->column_count,
                        &input_elements) ||
      !checked_multiply(params->output_row_count, params->column_count,
                        &output_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "row gather matrix shape overflows uint64_t");
  }
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t index_bytes = 0;
  RustInferCudaStatus status =
      typed_bytes(input_elements, params->input.dtype, &input_bytes, error,
                  kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, params->input.dtype, &output_bytes,
                         error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(params->output_row_count,
                         RUSTINFER_CUDA_DTYPE_U32, &index_bytes, error,
                         kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan input{};
  ResolvedSpan row_indices{};
  ResolvedSpan output{};
  status = resolve_span(params->input, params->input.dtype, input_bytes,
                        &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->row_indices, RUSTINFER_CUDA_DTYPE_U32,
                          index_bytes, &row_indices, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, params->input.dtype, output_bytes,
                          &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, row_indices, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, row_indices, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(row_indices.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "batch buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->output_row_count == 0) {
    return uses.release_completed()
               ? RUSTINFER_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (params->input.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      row_gather_kernel<float>
          <<<block_count(output_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const float*>(input.data),
              reinterpret_cast<const uint32_t*>(row_indices.data),
              reinterpret_cast<float*>(output.data), params->input_row_count,
              params->column_count, output_elements);
    } else {
      row_gather_kernel<__nv_bfloat16>
          <<<block_count(output_elements), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data),
              reinterpret_cast<const uint32_t*>(row_indices.data),
              reinterpret_cast<__nv_bfloat16*>(output.data),
              params->input_row_count, params->column_count, output_elements);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_ragged_paged_kv_cache_write_execute(
    const RustInferCudaRaggedPagedKvCacheWriteParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "write ragged paged KV cache";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaRaggedPagedKvCacheWriteParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->key_value_head_count == 0 || params->head_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "KV head dimensions must be greater than zero");
  }

  uint64_t source_elements = 0;
  uint64_t pool_elements = 0;
  if (!checked_product3(params->batch.active_row_count,
                        params->key_value_head_count, params->head_size,
                        &source_elements) ||
      !checked_product4(params->batch.physical_block_count,
                        params->key_value_head_count,
                        RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged paged KV tensor shape overflows uint64_t");
  }
  uint64_t source_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(source_elements, RUSTINFER_CUDA_DTYPE_BF16,
                       &source_bytes, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RUSTINFER_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan key_source{};
  ResolvedSpan value_source{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedBatch batch{};
  status = resolve_span(params->key_source, RUSTINFER_CUDA_DTYPE_BF16,
                        source_bytes, &key_source, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_source, RUSTINFER_CUDA_DTYPE_BF16,
                          source_bytes, &value_source, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  status = reject_overlap(key_pool, key_source, false, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_pool, value_source, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(key_pool, value_pool, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_pool, key_source, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(value_pool, value_source, false, error,
                            kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(key_pool, metadata[index], false, error,
                            kOperation);
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      status = reject_overlap(value_pool, metadata[index], false, error,
                              kOperation);
    }
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      key_source, value_source, key_pool, value_pool,
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    ragged_paged_kv_cache_write_kernel
        <<<block_count(source_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(key_source.data),
            reinterpret_cast<const __nv_bfloat16*>(value_source.data),
            reinterpret_cast<__nv_bfloat16*>(key_pool.data),
            reinterpret_cast<__nv_bfloat16*>(value_pool.data),
            device_batch(params->batch, batch), params->key_value_head_count,
            params->head_size, source_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_ragged_paged_attention_execute(
    const RustInferCudaRaggedPagedAttentionParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute ragged causal paged attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaRaggedPagedAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->query_head_count == 0 ||
      params->key_value_head_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention head counts must be greater than zero");
  }
  if (params->head_size != kAttentionHeadSize) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged paged attention supports head_size=64 only");
  }
  if (params->query_head_count % params->key_value_head_count != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "key_value_head_count must divide query_head_count");
  }
  if (!std::isfinite(params->scale) || params->scale <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention scale must be finite and greater than zero");
  }
  if (params->output_row_count < params->batch.active_row_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "output_row_count is smaller than active_row_count");
  }
  if (params->output_row_count > kMaximumGridX ||
      params->query_head_count > kMaximumGridYOrZ) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
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
                        RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "ragged attention tensor shape overflows uint64_t");
  }
  uint64_t query_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(query_elements, RUSTINFER_CUDA_DTYPE_BF16,
                       &query_bytes, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, RUSTINFER_CUDA_DTYPE_BF16,
                         &output_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RUSTINFER_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan output{};
  ResolvedBatch batch{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16,
                        query_bytes, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16,
                          output_bytes, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = reject_overlap(output, query, false, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key_pool, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value_pool, false, error, kOperation);
  }
  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, metadata[index], false, error,
                            kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      query, key_pool, value_pool, output, batch.sequence_block_offsets,
      batch.block_ids, batch.valid_tokens, batch.row_sequence_slots,
      batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                    static_cast<uint32_t>(params->query_head_count), 1);
    ragged_paged_attention_kernel<<<grid, kWarpSize, 0, stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key_pool.data),
        reinterpret_cast<const __nv_bfloat16*>(value_pool.data),
        reinterpret_cast<__nv_bfloat16*>(output.data),
        device_batch(params->batch, batch), params->query_head_count,
        params->key_value_head_count, params->scale);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_ragged_paged_attention_two_pass_execute(
    const RustInferCudaFixed37RaggedPagedAttentionParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "execute fixed37 two-pass ragged causal paged attention";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaFixed37RaggedPagedAttentionParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status =
      validate_packed_batch(params->batch, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->query_head_count == 0 ||
      params->key_value_head_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention head counts must be greater than zero");
  }
  if (params->head_size != kAttentionHeadSize) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged paged attention supports head_size=64 only");
  }
  if (params->query_head_count % params->key_value_head_count != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "key_value_head_count must divide query_head_count");
  }
  if (params->maximum_logical_token_count == 0) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "maximum_logical_token_count must be greater than zero");
  }
  if (params->maximum_logical_token_count >
      kFixed37RaggedMaximumTokenCount) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged paged attention supports maximum logical T<=8192 only");
  }
  if (!std::isfinite(params->scale) || params->scale <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention scale must be finite and greater than zero");
  }
  if (params->output_row_count < params->batch.active_row_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "output_row_count is smaller than active_row_count");
  }
  if (params->output_row_count > kMaximumGridX ||
      params->query_head_count > kMaximumGridYOrZ) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 ragged attention launch dimensions exceed the CUDA grid contract");
  }

  uint64_t maximum_token_partial_count = 0;
  uint64_t shared_bytes = 0;
  status = fixed37_ragged_shared_bytes(
      params->maximum_logical_token_count, &maximum_token_partial_count,
      &shared_bytes, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
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
                        RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE, params->head_size,
                        &pool_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "fixed37 ragged attention tensor shape overflows uint64_t");
  }
  uint64_t query_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t pool_bytes = 0;
  status = typed_bytes(query_elements, RUSTINFER_CUDA_DTYPE_BF16,
                       &query_bytes, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(output_elements, RUSTINFER_CUDA_DTYPE_BF16,
                         &output_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = typed_bytes(pool_elements, RUSTINFER_CUDA_DTYPE_BF16,
                         &pool_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key_pool{};
  ResolvedSpan value_pool{};
  ResolvedSpan output{};
  ResolvedBatch batch{};
  status = resolve_span(params->query, RUSTINFER_CUDA_DTYPE_BF16,
                        query_bytes, &query, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->key_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &key_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->value_pool, RUSTINFER_CUDA_DTYPE_BF16,
                          pool_bytes, &value_pool, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, RUSTINFER_CUDA_DTYPE_BF16,
                          output_bytes, &output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_packed_batch(params->batch, &batch, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = reject_overlap(output, query, false, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key_pool, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value_pool, false, error, kOperation);
  }
  const ResolvedSpan metadata[] = {
      batch.sequence_block_offsets, batch.block_ids, batch.valid_tokens,
      batch.row_sequence_slots, batch.row_positions};
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < 5; ++index) {
    status = reject_overlap(output, metadata[index], false, error,
                            kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {
      query, key_pool, value_pool, output, batch.sequence_block_offsets,
      batch.block_ids, batch.valid_tokens, batch.row_sequence_slots,
      batch.row_positions};
  status = validate_contexts(stream, spans, 9, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  for (size_t index = 0; index < 9; ++index) {
    if (!uses.add(spans[index].buffer)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "batch buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    const dim3 grid(static_cast<uint32_t>(params->output_row_count),
                    static_cast<uint32_t>(params->query_head_count), 1);
    fixed37_ragged_paged_attention_two_pass_kernel<<<
        grid, rustinfer_cuda_fixed37::kThreadsPerBlock,
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
