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

constexpr uint64_t kBf16Bytes = 2;
constexpr uint64_t kTwoPassHeadSize = 64;
constexpr uint64_t kTwoPassDepthPartialCount =
    rustinfer_cuda_fixed37::chunk_count(kTwoPassHeadSize);
constexpr uint64_t kMaximumTwoPassTokenCount = 8192;
constexpr size_t kMaximumBuffers = 4;
constexpr uint32_t kFiniteMinimumBf16AsF32Bits = 0xff7f0000U;

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

struct PrefillByteCounts {
  uint64_t query;
  uint64_t key_value;
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
  if (!checked_multiply(element_count, kBf16Bytes, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "BF16 byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_axis(uint64_t element_count,
                                 uint64_t* partial_count,
                                 uint64_t* shared_bytes,
                                 RustInferCudaErrorInfo* error,
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
        "the reduction axis exceeds the fixed37 chunk-partial capacity");
  }
  *partial_count = chunks;
  *shared_bytes = rustinfer_cuda_fixed37::shared_bytes(element_count);
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

RustInferCudaStatus gqa_byte_counts(
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    GqaByteCounts* output, RustInferCudaErrorInfo* error,
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

RustInferCudaStatus prefill_byte_counts(
    uint64_t batch_count, uint64_t token_count,
    uint64_t query_head_count, uint64_t key_value_head_count,
    uint64_t head_size, PrefillByteCounts* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (output == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          operation, "internal prefill byte counts are null");
  }
  uint64_t batch_tokens = 0;
  uint64_t query_elements = 0;
  uint64_t key_value_elements = 0;
  if (!checked_multiply(batch_count, token_count, &batch_tokens) ||
      !checked_product3(batch_tokens, query_head_count, head_size,
                        &query_elements) ||
      !checked_product3(batch_tokens, key_value_head_count, head_size,
                        &key_value_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "prefill attention tensor shape overflows uint64_t");
  }
  RustInferCudaStatus status =
      bf16_bytes(query_elements, &output->query, error, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = bf16_bytes(key_value_elements, &output->key_value, error,
                        operation);
  }
  return status;
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

RustInferCudaStatus resolve_bf16_span(
    const RustInferCudaBufferSpan& span, uint64_t required_bytes,
    ResolvedSpan* output, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (output == nullptr || span.struct_size < sizeof(span)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span has an incompatible struct_size");
  }
  if (!reserved_is_zero(span.reserved, 2) ||
      span.dtype != RUSTINFER_CUDA_DTYPE_BF16 || span.buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "attention span metadata is invalid or is not BF16");
  }
  if (span.byte_offset % kBf16Bytes != 0 ||
      span.byte_len % kBf16Bytes != 0) {
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
  if (required_bytes != 0 && span.buffer->device_data == nullptr) {
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
  if (left.buffer != right.buffer || left.used_bytes == 0 ||
      right.used_bytes == 0) {
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
  if (stream == nullptr || stream->owner == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "stream or CUDA context owner is null");
  }
  if (stream->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "CUDA context owner is poisoned");
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
    if (buffer == nullptr || buffer_count_ == kMaximumBuffers) {
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
  RustInferCudaDeviceBuffer* buffers_[kMaximumBuffers];
  size_t buffer_count_;
  size_t acquired_count_;
  bool stream_acquired_;
};

RustInferCudaStatus launch_status(RustInferCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RustInferCudaStatus complete_execution(
    ExclusiveUses* uses, CurrentContext* scope, RustInferCudaStream* stream,
    RustInferCudaStatus operation_status, bool launch_attempted,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
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
  if (completion_confirmed && restoration_confirmed &&
      !uses->release_completed()) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          operation,
                          "exclusive-use accounting was corrupted");
  }
  return status;
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_qk_gqa_kernel(const __nv_bfloat16* query,
                           const __nv_bfloat16* key,
                           __nv_bfloat16* output, uint64_t token_count,
                           uint64_t query_head_count,
                           uint64_t key_value_head_count, uint64_t head_size,
                           uint64_t output_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t output_index = blockIdx.x; output_index < output_count;
       output_index += gridDim.x) {
    const uint64_t key_token = output_index % token_count;
    const uint64_t row = output_index / token_count;
    const uint64_t query_token = row % token_count;
    const uint64_t query_head = row / token_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base =
        (query_token * query_head_count + query_head) * head_size;
    const uint64_t key_base =
        (key_token * key_value_head_count + key_value_head) * head_size;
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
                           __bfloat162float(key[key_base + depth]),
                           accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float result =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      output[output_index] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_softmax_kernel(__nv_bfloat16* scores, uint64_t token_count,
                            uint64_t row_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  __shared__ uint32_t has_nan;
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    if (threadIdx.x == 0) {
      has_nan = 0;
    }
    __syncthreads();
    const uint64_t base = row * token_count;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > token_count) {
        end = token_count;
      }
      float maximum = -CUDART_INF_F;
      bool local_nan = false;
      for (uint64_t column = begin; column < end; ++column) {
        const float value = __bfloat162float(scores[base + column]);
        local_nan = local_nan || isnan(value);
        maximum = fmaxf(maximum, value);
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
      for (uint64_t column = threadIdx.x; column < token_count;
           column += blockDim.x) {
        scores[base + column] = nan;
      }
      __syncthreads();
      continue;
    }
    __syncthreads();
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > token_count) {
        end = token_count;
      }
      float sum = 0.0F;
      for (uint64_t column = begin; column < end; ++column) {
        sum = __fadd_rn(
            sum, expf(__fsub_rn(__bfloat162float(scores[base + column]),
                                maximum)));
      }
      first[chunk] = sum;
    }
    __syncthreads();
    const float denominator =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    for (uint64_t column = threadIdx.x; column < token_count;
         column += blockDim.x) {
      const float numerator =
          expf(__fsub_rn(__bfloat162float(scores[base + column]), maximum));
      scores[base + column] =
          __float2bfloat16_rn(numerator / denominator);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_av_gqa_kernel(const __nv_bfloat16* probabilities,
                           const __nv_bfloat16* value,
                           __nv_bfloat16* output, uint64_t token_count,
                           uint64_t query_head_count,
                           uint64_t key_value_head_count, uint64_t head_size,
                           uint64_t output_count, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t output_index = blockIdx.x; output_index < output_count;
       output_index += gridDim.x) {
    const uint64_t depth = output_index % head_size;
    const uint64_t row = output_index / head_size;
    const uint64_t query_head = row % query_head_count;
    const uint64_t query_token = row / query_head_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t probability_base =
        (query_head * token_count + query_token) * token_count;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > token_count) {
        end = token_count;
      }
      float accumulator = 0.0F;
      for (uint64_t key_token = begin; key_token < end; ++key_token) {
        const uint64_t value_index =
            (key_token * key_value_head_count + key_value_head) * head_size +
            depth;
        accumulator = fmaf(
            __bfloat162float(probabilities[probability_base + key_token]),
            __bfloat162float(value[value_index]), accumulator);
      }
      first[chunk] = accumulator;
    }
    __syncthreads();
    const float result =
        rustinfer_cuda_fixed37::balanced_sum(first, second, partial_count);
    if (threadIdx.x == 0) {
      output[output_index] = __float2bfloat16_rn(result);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ bool key_is_visible(
    uint64_t query_token, uint64_t key_token, bool causal_local,
    uint64_t local_window_size) {
  if (key_token > query_token) {
    return false;
  }
  if (!causal_local) {
    return true;
  }
  return local_window_size != 0 &&
         local_window_size > query_token - key_token;
}

__device__ __forceinline__ float stage_score(float dot, float scale,
                                              bool visible) {
  const __nv_bfloat16 raw = __float2bfloat16_rn(dot);
  const __nv_bfloat16 scaled = __float2bfloat16_rn(
      __bfloat162float(raw) * scale);
  const float mask = visible ? 0.0F
                             : __uint_as_float(kFiniteMinimumBf16AsF32Bits);
  return __bfloat162float(
      __float2bfloat16_rn(__bfloat162float(scaled) + mask));
}

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_two_pass_prefill_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, __nv_bfloat16* output,
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, float scale, bool causal_local,
    uint64_t local_window_size, uint64_t row_count,
    uint64_t sequence_partial_count) {
  extern __shared__ float shared_values[];
  __shared__ uint32_t has_nan;
  float* values = shared_values;
  float* first = values + token_count;
  const uint64_t partial_capacity =
      sequence_partial_count < kTwoPassDepthPartialCount
          ? kTwoPassDepthPartialCount
          : sequence_partial_count;
  float* second = first + partial_capacity;
  const uint64_t group_size = query_head_count / key_value_head_count;

  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t query_head = row % query_head_count;
    const uint64_t batch_token = row / query_head_count;
    const uint64_t query_token = batch_token % token_count;
    const uint64_t batch = batch_token / token_count;
    const uint64_t key_value_head = query_head / group_size;
    const uint64_t query_base = row * kTwoPassHeadSize;

    if (threadIdx.x == 0) {
      has_nan = 0;
    }
    __syncthreads();

    // Pass one recomputes no state from a recurrence: it evaluates every QK
    // score in logical key order and records only one max partial per global
    // 37-key chunk. `values` is scratch here and becomes shared exp[S] in the
    // second score pass.
    float chunk_maximum = -CUDART_INF_F;
    for (uint64_t key_token = 0; key_token < token_count; ++key_token) {
      const uint64_t key_base =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
          kTwoPassHeadSize;
      for (uint64_t chunk = threadIdx.x; chunk < kTwoPassDepthPartialCount;
           chunk += blockDim.x) {
        const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > kTwoPassHeadSize) {
          end = kTwoPassHeadSize;
        }
        float accumulator = 0.0F;
        for (uint64_t depth = begin; depth < end; ++depth) {
          accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                             __bfloat162float(key[key_base + depth]),
                             accumulator);
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float dot = rustinfer_cuda_fixed37::balanced_sum(
          first, second, kTwoPassDepthPartialCount);
      if (threadIdx.x == 0) {
        const float score = stage_score(
            dot, scale,
            key_is_visible(query_token, key_token, causal_local,
                           local_window_size));
        if (isnan(score)) {
          atomicOr(&has_nan, 1U);
        }
        if (key_token % rustinfer_cuda_fixed37::kChunkElements == 0) {
          chunk_maximum = -CUDART_INF_F;
        }
        chunk_maximum = fmaxf(chunk_maximum, score);
        if (key_token % rustinfer_cuda_fixed37::kChunkElements ==
                rustinfer_cuda_fixed37::kChunkElements - 1 ||
            key_token + 1 == token_count) {
          values[key_token / rustinfer_cuda_fixed37::kChunkElements] =
              chunk_maximum;
        }
      }
      __syncthreads();
    }

    for (uint64_t chunk = threadIdx.x; chunk < sequence_partial_count;
         chunk += blockDim.x) {
      first[chunk] = values[chunk];
    }
    __syncthreads();
    const float maximum = rustinfer_cuda_fixed37::balanced_max(
        first, second, sequence_partial_count);
    if (has_nan != 0 || !isfinite(maximum)) {
      const __nv_bfloat16 nan = __float2bfloat16_rn(CUDART_NAN_F);
      for (uint64_t depth = threadIdx.x; depth < kTwoPassHeadSize;
           depth += blockDim.x) {
        output[query_base + depth] = nan;
      }
      __syncthreads();
      continue;
    }
    __syncthreads();

    // Pass two evaluates the complete fixed37 QK/staging path again, storing
    // only shared F32 exp[S]. It then reduces the denominator in fixed37 order
    // and narrows every probability to BF16 before fixed37 AV.
    for (uint64_t key_token = 0; key_token < token_count; ++key_token) {
      const uint64_t key_base =
          ((batch * token_count + key_token) * key_value_head_count +
           key_value_head) *
          kTwoPassHeadSize;
      for (uint64_t chunk = threadIdx.x; chunk < kTwoPassDepthPartialCount;
           chunk += blockDim.x) {
        const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > kTwoPassHeadSize) {
          end = kTwoPassHeadSize;
        }
        float accumulator = 0.0F;
        for (uint64_t depth = begin; depth < end; ++depth) {
          accumulator = fmaf(__bfloat162float(query[query_base + depth]),
                             __bfloat162float(key[key_base + depth]),
                             accumulator);
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float dot = rustinfer_cuda_fixed37::balanced_sum(
          first, second, kTwoPassDepthPartialCount);
      if (threadIdx.x == 0) {
        const float score = stage_score(
            dot, scale,
            key_is_visible(query_token, key_token, causal_local,
                           local_window_size));
        values[key_token] = expf(__fsub_rn(score, maximum));
      }
      __syncthreads();
    }

    for (uint64_t chunk = threadIdx.x; chunk < sequence_partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > token_count) {
        end = token_count;
      }
      float sum = 0.0F;
      for (uint64_t key_token = begin; key_token < end; ++key_token) {
        sum = __fadd_rn(sum, values[key_token]);
      }
      first[chunk] = sum;
    }
    __syncthreads();
    const float denominator = rustinfer_cuda_fixed37::balanced_sum(
        first, second, sequence_partial_count);
    for (uint64_t key_token = threadIdx.x; key_token < token_count;
         key_token += blockDim.x) {
      const __nv_bfloat16 probability =
          __float2bfloat16_rn(values[key_token] / denominator);
      values[key_token] = __bfloat162float(probability);
    }
    __syncthreads();

    for (uint64_t depth = 0; depth < kTwoPassHeadSize; ++depth) {
      for (uint64_t chunk = threadIdx.x; chunk < sequence_partial_count;
           chunk += blockDim.x) {
        const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
        uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
        if (end > token_count) {
          end = token_count;
        }
        float accumulator = 0.0F;
        for (uint64_t key_token = begin; key_token < end; ++key_token) {
          const uint64_t value_index =
              ((batch * token_count + key_token) * key_value_head_count +
               key_value_head) *
                  kTwoPassHeadSize +
              depth;
          accumulator = fmaf(values[key_token],
                             __bfloat162float(value[value_index]),
                             accumulator);
        }
        first[chunk] = accumulator;
      }
      __syncthreads();
      const float result = rustinfer_cuda_fixed37::balanced_sum(
          first, second, sequence_partial_count);
      if (threadIdx.x == 0) {
        output[query_base + depth] = __float2bfloat16_rn(result);
      }
      __syncthreads();
    }
  }
}

template <typename Params>
RustInferCudaStatus validate_params_header(
    const Params* params, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "params is null or has an incompatible struct_size");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

template <typename Params>
RustInferCudaStatus validate_params_reserved(
    const Params& params, size_t reserved_count,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (params.reserved0 != 0 ||
      !reserved_is_zero(params.reserved, reserved_count)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "params reserved fields must be zero");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_qk_gqa_execute(
    const RustInferCudaQkGqaParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 QK GQA";
  clear_error(error);
  RustInferCudaStatus status =
      validate_params_header(params, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const RustInferCudaQkGqaParams stable_params = *params;
  params = &stable_params;
  status = validate_params_reserved(*params, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_gqa_dimensions(
      params->token_count, params->query_head_count,
      params->key_value_head_count, params->head_size, error, kOperation);
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_axis(params->head_size, &partial_count, &shared_bytes,
                           error, kOperation);
  }
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
  const ResolvedSpan spans[] = {query, key, output};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_contexts(stream, spans, 3, error, kOperation);
  }
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
    const uint64_t output_count = bytes.scores / kBf16Bytes;
    launch_attempted = true;
    fixed37_qk_gqa_kernel<<<
        rustinfer_cuda_fixed37::block_count(output_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), params->token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, output_count, partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_causal_softmax_in_place_execute(
    const RustInferCudaCausalSoftmaxParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 causal softmax";
  clear_error(error);
  RustInferCudaStatus status =
      validate_params_header(params, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const RustInferCudaCausalSoftmaxParams stable_params = *params;
  params = &stable_params;
  status = validate_params_reserved(*params, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->token_count == 0 || params->query_head_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "token_count and query_head_count must be non-zero");
  }
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  status = validate_axis(params->token_count, &partial_count, &shared_bytes,
                         error, kOperation);
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
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_contexts(stream, &scores, 1, error, kOperation);
  }
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
    fixed37_softmax_kernel<<<
        rustinfer_cuda_fixed37::block_count(row_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<__nv_bfloat16*>(scores.data), params->token_count,
        row_count, partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_av_gqa_execute(
    const RustInferCudaAvGqaParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 AV GQA";
  clear_error(error);
  RustInferCudaStatus status =
      validate_params_header(params, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const RustInferCudaAvGqaParams stable_params = *params;
  params = &stable_params;
  status = validate_params_reserved(*params, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_gqa_dimensions(
      params->token_count, params->query_head_count,
      params->key_value_head_count, params->head_size, error, kOperation);
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_axis(params->token_count, &partial_count, &shared_bytes,
                           error, kOperation);
  }
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
  const ResolvedSpan spans[] = {probabilities, value, output};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_contexts(stream, spans, 3, error, kOperation);
  }
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
    const uint64_t output_count = bytes.query / kBf16Bytes;
    launch_attempted = true;
    fixed37_av_gqa_kernel<<<
        rustinfer_cuda_fixed37::block_count(output_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(probabilities.data),
        reinterpret_cast<const __nv_bfloat16*>(value.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), params->token_count,
        params->query_head_count, params->key_value_head_count,
        params->head_size, output_count, partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_fixed37_prefill_attention_execute(
    const RustInferCudaPrefillAttentionParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 two-pass prefill attention";
  clear_error(error);
  RustInferCudaStatus status =
      validate_params_header(params, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const RustInferCudaPrefillAttentionParams stable_params = *params;
  params = &stable_params;
  status = validate_params_reserved(*params, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_gqa_dimensions(
      params->token_count, params->query_head_count,
      params->key_value_head_count, params->head_size, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && params->batch_count == 0) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation, "batch_count must be non-zero");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->head_size != kTwoPassHeadSize) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "fixed37 two-pass prefill supports head_size=64 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->token_count > kMaximumTwoPassTokenCount) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "fixed37 two-pass prefill supports S<=8192 only");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (!std::isfinite(params->scale) || params->scale <= 0.0F)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "scale must be finite and greater than zero");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->mask_kind != RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL &&
      params->mask_kind != RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL_LOCAL) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "fixed37 two-pass prefill mask kind is unsupported");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->mask_kind == RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL &&
      params->local_window_size != 0) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "causal attention requires local_window_size=0");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      params->mask_kind == RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL_LOCAL &&
      params->local_window_size == 0) {
    status = validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "fixed37 causal-local prefill requires a non-zero local window");
  }
  uint64_t sequence_partial_count = 0;
  uint64_t sequence_reduction_shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_axis(params->token_count, &sequence_partial_count,
                           &sequence_reduction_shared_bytes, error, kOperation);
  }
  const uint64_t partial_capacity =
      sequence_partial_count < kTwoPassDepthPartialCount
          ? kTwoPassDepthPartialCount
          : sequence_partial_count;
  uint64_t reduction_shared_bytes = 0;
  uint64_t score_shared_bytes = 0;
  uint64_t shared_bytes = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (!checked_multiply(partial_capacity, 2 * sizeof(float),
                         &reduction_shared_bytes) ||
       !checked_multiply(params->token_count, sizeof(float),
                         &score_shared_bytes) ||
       score_shared_bytes >
           std::numeric_limits<uint64_t>::max() - reduction_shared_bytes)) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "fixed37 two-pass shared-memory size overflows");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    shared_bytes = score_shared_bytes + reduction_shared_bytes;
  }
  PrefillByteCounts bytes{};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prefill_byte_counts(
        params->batch_count, params->token_count, params->query_head_count,
        params->key_value_head_count, params->head_size, &bytes, error,
        kOperation);
  }
  uint64_t batch_tokens = 0;
  uint64_t row_count = 0;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      (!checked_multiply(params->batch_count, params->token_count,
                         &batch_tokens) ||
       !checked_multiply(batch_tokens, params->query_head_count,
                         &row_count))) {
    status = validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kOperation,
                              "fixed37 two-pass row count overflows uint64_t");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan query{};
  ResolvedSpan key{};
  ResolvedSpan value{};
  ResolvedSpan output{};
  status = resolve_bf16_span(params->query, bytes.query, &query, error,
                             kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->key, bytes.key_value, &key, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->value, bytes.key_value, &value, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_bf16_span(params->output, bytes.query, &output, error,
                               kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, query, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, key, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, value, error, kOperation);
  }
  const ResolvedSpan spans[] = {query, key, value, output};
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = validate_contexts(stream, spans, 4, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(query.buffer) || !uses.add(key.buffer) ||
      !uses.add(value.buffer) || !uses.add(output.buffer)) {
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
    launch_attempted = true;
    fixed37_two_pass_prefill_kernel<<<
        rustinfer_cuda_fixed37::block_count(row_count),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(shared_bytes), stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data),
        reinterpret_cast<const __nv_bfloat16*>(key.data),
        reinterpret_cast<const __nv_bfloat16*>(value.data),
        reinterpret_cast<__nv_bfloat16*>(output.data), params->token_count,
        params->query_head_count, params->key_value_head_count, params->scale,
        params->mask_kind == RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL_LOCAL,
        params->local_window_size, row_count, sequence_partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}
