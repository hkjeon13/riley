#include "ffi_internal.hpp"

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
constexpr size_t kMaximumDecodeBuffers = 5;
constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
constexpr uint64_t kOptimizedHeadSize = 64;
constexpr uint64_t kOptimizedStateStride = kOptimizedHeadSize + 2;
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
      const float normalized = denominator == 0.0F
                                   ? 0.0F
                                   : numerator / denominator;
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
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
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
