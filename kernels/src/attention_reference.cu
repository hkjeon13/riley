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
constexpr size_t kMaximumAttentionBuffers = 3;
constexpr uint32_t kCausalMaskBf16AsF32Bits = 0xff7f0000U;

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
};

struct GqaByteCounts {
  uint64_t query;
  uint64_t key_value;
  uint64_t scores;
};

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

RustInferCudaStatus bf16_bytes(uint64_t element_count, uint64_t* output,
                              RustInferCudaErrorInfo* error,
                              const char* operation) noexcept {
  if (!checked_multiply(element_count, 2, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "BF16 byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus score_bytes(uint64_t token_count,
                                uint64_t query_head_count, uint64_t* output,
                                RustInferCudaErrorInfo* error,
                                const char* operation) noexcept {
  uint64_t elements = 0;
  if (!checked_product3(query_head_count, token_count, token_count,
                        &elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "attention score shape overflows uint64_t");
  }
  return bf16_bytes(elements, output, error, operation);
}

RustInferCudaStatus gqa_byte_counts(uint64_t token_count,
                                    uint64_t query_head_count,
                                    uint64_t key_value_head_count,
                                    uint64_t head_size, GqaByteCounts* output,
                                    RustInferCudaErrorInfo* error,
                                    const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal GQA byte counts are null");
  }
  uint64_t query_elements = 0;
  uint64_t key_value_elements = 0;
  if (!checked_product3(token_count, query_head_count, head_size,
                        &query_elements) ||
      !checked_product3(token_count, key_value_head_count, head_size,
                        &key_value_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "attention tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status =
      bf16_bytes(query_elements, &output->query, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = bf16_bytes(key_value_elements, &output->key_value, error,
                        operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = score_bytes(token_count, query_head_count, &output->scores, error,
                         operation);
  }
  return status;
}

RustInferCudaStatus resolve_bf16_span(const RustInferCudaBufferSpan& span,
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
  if (span.dtype != RUSTINFER_CUDA_DTYPE_BF16) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "attention spans must use BF16 dtype");
  }
  if (span.buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span handle is null");
  }
  if (span.byte_offset % 2 != 0 || span.byte_len % 2 != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "BF16 span offset or length is not aligned");
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
                            "attention span refers to a zero-byte allocation");
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

RustInferCudaStatus reject_overlap(const ResolvedSpan& output,
                                   const ResolvedSpan& input,
                                   RustInferCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(output, input)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "attention output may not overlap an input span");
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
          "stream and attention spans belong to different context owners");
    }
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

class ExclusiveUses final {
 public:
  explicit ExclusiveUses(RustInferCudaStream* stream) noexcept
      : stream_(stream), buffers_{}, buffer_count_(0), acquired_count_(0),
        stream_acquired_(false) {}

  ExclusiveUses(const ExclusiveUses&) = delete;
  ExclusiveUses& operator=(const ExclusiveUses&) = delete;

  bool add(RustInferCudaDeviceBuffer* buffer) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (buffers_[index] == buffer) {
        return true;
      }
    }
    if (buffer == nullptr || buffer_count_ == kMaximumAttentionBuffers) {
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
            "an attention buffer already has an active asynchronous use");
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
  RustInferCudaDeviceBuffer* buffers_[kMaximumAttentionBuffers];
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

__global__ void qk_gqa_kernel(const __nv_bfloat16* query,
                              const __nv_bfloat16* key,
                              __nv_bfloat16* output, uint64_t token_count,
                              uint64_t query_head_count,
                              uint64_t key_value_head_count,
                              uint64_t head_size,
                              uint64_t score_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < score_element_count; index += stride) {
    const uint64_t key_token = index % token_count;
    const uint64_t row = index / token_count;
    const uint64_t query_token = row % token_count;
    const uint64_t query_head = row / token_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base =
        (query_token * query_head_count + query_head) * head_size;
    const uint64_t key_base =
        (key_token * key_value_head_count + key_value_head) * head_size;
    float accumulator = 0.0F;
    for (uint64_t depth = 0; depth < head_size; ++depth) {
      accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                         __bfloat162float(key[key_base + depth]), accumulator);
    }
    output[index] = __float2bfloat16_rn(accumulator);
  }
}

__global__ void scale_causal_mask_kernel(__nv_bfloat16* scores,
                                         uint64_t token_count, float scale,
                                         uint64_t score_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < score_element_count; index += stride) {
    const uint64_t key_token = index % token_count;
    const uint64_t query_token = (index / token_count) % token_count;
    const __nv_bfloat16 scaled = __float2bfloat16_rn(
        __bfloat162float(scores[index]) * scale);
    const float mask = key_token > query_token
                           ? __uint_as_float(kCausalMaskBf16AsF32Bits)
                           : 0.0F;
    scores[index] =
        __float2bfloat16_rn(__bfloat162float(scaled) + mask);
  }
}

__global__ void causal_softmax_kernel(__nv_bfloat16* scores,
                                      uint64_t token_count,
                                      uint64_t row_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t row = first; row < row_count; row += stride) {
    const uint64_t base = row * token_count;
    float maximum = -CUDART_INF_F;
    bool has_nan = false;
    for (uint64_t column = 0; column < token_count; ++column) {
      const float value = __bfloat162float(scores[base + column]);
      has_nan = has_nan || isnan(value);
      maximum = fmaxf(maximum, value);
    }
    if (has_nan) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t column = 0; column < token_count; ++column) {
        scores[base + column] = nan;
      }
      continue;
    }
    float denominator = 0.0F;
    for (uint64_t column = 0; column < token_count; ++column) {
      denominator +=
          expf(__bfloat162float(scores[base + column]) - maximum);
    }
    for (uint64_t column = 0; column < token_count; ++column) {
      const float numerator =
          expf(__bfloat162float(scores[base + column]) - maximum);
      scores[base + column] = __float2bfloat16_rn(numerator / denominator);
    }
  }
}

__global__ void av_gqa_kernel(const __nv_bfloat16* probabilities,
                              const __nv_bfloat16* value,
                              __nv_bfloat16* output, uint64_t token_count,
                              uint64_t query_head_count,
                              uint64_t key_value_head_count,
                              uint64_t head_size,
                              uint64_t output_element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < output_element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t row = index / head_size;
    const uint64_t query_head = row % query_head_count;
    const uint64_t query_token = row / query_head_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base =
        (query_head * token_count + query_token) * token_count;
    float accumulator = 0.0F;
    for (uint64_t key_token = 0; key_token < token_count; ++key_token) {
      const uint64_t value_index =
          (key_token * key_value_head_count + key_value_head) * head_size +
          depth;
      accumulator =
          fmaf(__bfloat162float(probabilities[probability_base + key_token]),
               __bfloat162float(value[value_index]), accumulator);
    }
    output[index] = __float2bfloat16_rn(accumulator);
  }
}

RustInferCudaStatus validate_gqa_dimensions(
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (token_count == 0 || query_head_count == 0 ||
      key_value_head_count == 0 || head_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "all attention dimensions must be greater than zero");
  }
  if (query_head_count % key_value_head_count != 0) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
        "key_value_head_count must divide query_head_count");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_score_dimensions(
    uint64_t token_count, uint64_t query_head_count,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (token_count == 0 || query_head_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "token_count and query_head_count must be greater than zero");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_qk_gqa_execute(
    const RustInferCudaQkGqaParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute QK GQA";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaQkGqaParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status = validate_gqa_dimensions(
      params->token_count, params->query_head_count,
      params->key_value_head_count, params->head_size, error, kOperation);
  GqaByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = gqa_byte_counts(
        params->token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key{};
  ResolvedSpan output{};
  status = resolve_bf16_span(params->query, bytes.query, &query, error,
                             kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->key, bytes.key_value, &key, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->output, bytes.scores, &output, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, query, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {query, key, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "attention buffer set overflow");
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
    uint64_t score_elements = 0;
    (void)checked_product3(params->query_head_count, params->token_count,
                           params->token_count, &score_elements);
    launch_attempted = true;
    qk_gqa_kernel<<<block_count(score_elements), kThreads, 0, stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), params->token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, score_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_scale_causal_mask_in_place_execute(
    const RustInferCudaScaleCausalMaskParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute attention scale and causal mask";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaScaleCausalMaskParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status = validate_score_dimensions(
      params->token_count, params->query_head_count, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (!std::isfinite(params->scale) || params->scale <= 0.0F)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "scale must be finite and greater than zero");
  }
  uint64_t bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = score_bytes(params->token_count, params->query_head_count, &bytes,
                         error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan scores{};
  status = resolve_bf16_span(params->scores, bytes, &scores, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_contexts(stream, &scores, 1, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(scores.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "attention buffer set overflow");
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
    const uint64_t score_elements = bytes / 2;
    launch_attempted = true;
    scale_causal_mask_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(scores.data),
            params->token_count, params->scale, score_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_causal_softmax_in_place_execute(
    const RustInferCudaCausalSoftmaxParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute causal softmax";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaCausalSoftmaxParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status = validate_score_dimensions(
      params->token_count, params->query_head_count, error, kOperation);
  uint64_t bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = score_bytes(params->token_count, params->query_head_count, &bytes,
                         error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan scores{};
  status = resolve_bf16_span(params->scores, bytes, &scores, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_contexts(stream, &scores, 1, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(scores.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "attention buffer set overflow");
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
    uint64_t row_count = 0;
    (void)checked_multiply(params->query_head_count, params->token_count,
                           &row_count);
    launch_attempted = true;
    causal_softmax_kernel
        <<<block_count(row_count), kThreads, 0, stream->stream>>>(
            reinterpret_cast<__nv_bfloat16*>(scores.data),
            params->token_count, row_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_av_gqa_execute(
    const RustInferCudaAvGqaParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute AV GQA";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RustInferCudaAvGqaParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  RustInferCudaStatus status = validate_gqa_dimensions(
      params->token_count, params->query_head_count,
      params->key_value_head_count, params->head_size, error, kOperation);
  GqaByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = gqa_byte_counts(
        params->token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan probabilities{};
  ResolvedSpan value{};
  ResolvedSpan output{};
  status = resolve_bf16_span(params->probabilities, bytes.scores,
                             &probabilities, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->value, bytes.key_value, &value, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->output, bytes.query, &output, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, probabilities, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {probabilities, value, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(probabilities.buffer) || !uses.add(value.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "attention buffer set overflow");
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
    const uint64_t output_elements = bytes.query / 2;
    launch_attempted = true;
    av_gqa_kernel<<<block_count(output_elements), kThreads, 0, stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(probabilities.data),
        reinterpret_cast<const __nv_bfloat16*>(value.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), params->token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, output_elements);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}
