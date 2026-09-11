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

constexpr size_t kMaximumPrimitiveBuffers = 5;

struct ResolvedSpan {
  RileyCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
  RileyCudaDType dtype;
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
      return 4;
    case RILEY_CUDA_DTYPE_BF16:
      return 2;
    default:
      return 0;
  }
}

bool arithmetic_dtype(RileyCudaDType dtype) noexcept {
  return dtype == RILEY_CUDA_DTYPE_F32 ||
         dtype == RILEY_CUDA_DTYPE_BF16;
}

RileyCudaStatus element_bytes(uint64_t element_count,
                                  RileyCudaDType dtype, uint64_t* output,
                                  RileyCudaErrorInfo* error,
                                  const char* operation) noexcept {
  const uint64_t width = dtype_size(dtype);
  if (width == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "span has an unsupported dtype");
  }
  if (!checked_multiply(element_count, width, output)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "element byte length overflows uint64_t");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus validate_reduction_axis(uint64_t element_count,
                                            uint64_t* partial_count,
                                            uint64_t* shared_bytes,
                                            RileyCudaErrorInfo* error,
                                            const char* operation) noexcept {
  if (element_count == 0 || partial_count == nullptr ||
      shared_bytes == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "the fixed37 reduction axis must be non-zero");
  }
  const uint64_t chunks = riley_cuda_fixed37::chunk_count(element_count);
  // Enforce the bound before computing launch shared-memory bytes. This keeps
  // both the multiplication and the kernel's two partial arrays in contract.
  if (chunks == 0 ||
      chunks > riley_cuda_fixed37::kMaximumChunkCount) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
        "the reduction axis exceeds the fixed37 chunk-partial capacity");
  }
  *partial_count = chunks;
  *shared_bytes = chunks * 2 * sizeof(float);
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus resolve_span(const RileyCudaBufferSpan& span,
                                 uint64_t required_bytes,
                                 ResolvedSpan* output,
                                 RileyCudaErrorInfo* error,
                                 const char* operation) noexcept {
  if (output == nullptr || span.struct_size < sizeof(span)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span has an incompatible struct_size");
  }
  if (!reserved_is_zero(span.reserved, 2)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "buffer span reserved fields must be zero");
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

bool exact_alias(const ResolvedSpan& left,
                 const ResolvedSpan& right) noexcept {
  return left.buffer == right.buffer && left.byte_offset == right.byte_offset &&
         left.used_bytes == right.used_bytes;
}

RileyCudaStatus reject_overlap(const ResolvedSpan& write,
                                   const ResolvedSpan& read,
                                   bool exact_alias_allowed,
                                   RileyCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(write, read) &&
      !(exact_alias_allowed && exact_alias(write, read))) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "unsupported partial or write/input span overlap");
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
          "stream and device spans belong to different context owners");
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
    if (buffer == nullptr || buffer_count_ == kMaximumPrimitiveBuffers) {
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
            "a device buffer already has an active asynchronous use");
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
            "a device buffer already has an active asynchronous use");
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
      const size_t acquired_index = acquired_count_ - 1;
      acquired_count_ = acquired_index;
      valid = release_exclusive_use(buffers_[acquired_index]->active_uses) &&
              valid;
    }
    return valid;
  }

  bool command_batch() const noexcept { return command_batch_; }

 private:
  void release_acquired() noexcept {
    while (acquired_count_ != 0) {
      const size_t acquired_index = acquired_count_ - 1;
      acquired_count_ = acquired_index;
      (void)release_exclusive_use(buffers_[acquired_index]->active_uses);
    }
  }

  RileyCudaStream* stream_;
  RileyCudaDeviceBuffer* buffers_[kMaximumPrimitiveBuffers];
  size_t buffer_count_;
  size_t acquired_count_;
  bool stream_acquired_;
  bool command_batch_;
};

RileyCudaStatus complete_execution(
    ExclusiveUses* uses, CurrentContext* scope, RileyCudaStream* stream,
    RileyCudaStatus operation_status, bool launch_attempted,
    RileyCudaErrorInfo* error, const char* operation) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  }
  bool completion_confirmed = !launch_attempted;
  RileyCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t synchronize_result = cudaStreamSynchronize(stream->stream);
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
  if (completion_confirmed && restoration_confirmed &&
      !uses->release_completed()) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          operation,
                          "exclusive-use accounting was corrupted");
  }
  return status;
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
__global__ __launch_bounds__(riley_cuda_fixed37::kThreadsPerBlock)
void fixed37_rms_norm_kernel(const T* input, const T* weight, T* output,
                             uint64_t row_count, uint64_t hidden_size,
                             float epsilon, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t base = row * hidden_size;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * riley_cuda_fixed37::kChunkElements;
      uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
      if (end > hidden_size) {
        end = hidden_size;
      }
      float sum = 0.0F;
      for (uint64_t column = begin; column < end; ++column) {
        const float value = load_f32(input, base + column);
        sum = fmaf(value, value, sum);
      }
      first[chunk] = sum;
    }
    // The reduction helper requires every logical partial to be visible.
    __syncthreads();
    const float sum_of_squares =
        riley_cuda_fixed37::balanced_sum(first, second, partial_count);
    const float inverse_rms =
        rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float normalized = load_f32(input, base + column) * inverse_rms;
      const float normalized_for_weight = round_to_storage<T>(normalized);
      store_f32(output, base + column,
                normalized_for_weight * load_f32(weight, column));
    }
    __syncthreads();
  }
}

template <typename T>
__global__ __launch_bounds__(riley_cuda_fixed37::kThreadsPerBlock)
void fixed37_residual_rms_norm_kernel(
    const T* left, const T* right, const T* weight, T* residual_output,
    T* normalized_output, uint64_t row_count, uint64_t hidden_size,
    float epsilon, uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t base = row * hidden_size;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * riley_cuda_fixed37::kChunkElements;
      uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
      if (end > hidden_size) {
        end = hidden_size;
      }
      float sum = 0.0F;
      for (uint64_t column = begin; column < end; ++column) {
        const uint64_t index = base + column;
        const float residual = round_to_storage<T>(
            __fadd_rn(load_f32(left, index), load_f32(right, index)));
        store_f32(residual_output, index, residual);
        sum = fmaf(residual, residual, sum);
      }
      first[chunk] = sum;
    }
    // This also makes every residual store visible before normalization.
    __syncthreads();
    const float sum_of_squares =
        riley_cuda_fixed37::balanced_sum(first, second, partial_count);
    const float inverse_rms =
        rsqrtf(sum_of_squares / static_cast<float>(hidden_size) + epsilon);
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const uint64_t index = base + column;
      const float normalized = load_f32(residual_output, index) * inverse_rms;
      const float normalized_for_weight = round_to_storage<T>(normalized);
      store_f32(normalized_output, index,
                normalized_for_weight * load_f32(weight, column));
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(riley_cuda_fixed37::kThreadsPerBlock)
void fixed37_log_softmax_kernel(const __nv_bfloat16* logits, float* output,
                                uint64_t element_count,
                                uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  __shared__ uint32_t has_nan;
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  if (threadIdx.x == 0) {
    has_nan = 0;
  }
  __syncthreads();

  for (uint64_t chunk = threadIdx.x; chunk < partial_count;
       chunk += blockDim.x) {
    const uint64_t begin = chunk * riley_cuda_fixed37::kChunkElements;
    uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
    if (end > element_count) {
      end = element_count;
    }
    float maximum = -CUDART_INF_F;
    bool local_nan = false;
    for (uint64_t index = begin; index < end; ++index) {
      const float value = __bfloat162float(logits[index]);
      local_nan = local_nan || isnan(value);
      // fmaxf gives the specified +0 preference for a +/-0 pair. NaNs are
      // recorded separately so fmaxf's single-NaN suppression cannot hide one.
      maximum = fmaxf(maximum, value);
    }
    if (local_nan) {
      atomicOr(&has_nan, 1U);
    }
    first[chunk] = maximum;
  }
  __syncthreads();
  const float maximum =
      riley_cuda_fixed37::balanced_max(first, second, partial_count);

  // Literal stable log-softmax is undefined for a NaN, a +Inf maximum, or an
  // all--Inf vector. Make that policy deterministic instead of depending on
  // fmaxf/expf incidental NaN handling.
  if (has_nan != 0 || !isfinite(maximum)) {
    for (uint64_t index = threadIdx.x; index < element_count;
         index += blockDim.x) {
      output[index] = CUDART_NAN_F;
    }
    return;
  }

  // Every thread has loaded the max result before `first` is reused for the
  // exponential-sum partials. This matters when the max tree ends in `first`.
  __syncthreads();

  for (uint64_t chunk = threadIdx.x; chunk < partial_count;
       chunk += blockDim.x) {
    const uint64_t begin = chunk * riley_cuda_fixed37::kChunkElements;
    uint64_t end = begin + riley_cuda_fixed37::kChunkElements;
    if (end > element_count) {
      end = element_count;
    }
    float sum = 0.0F;
    for (uint64_t index = begin; index < end; ++index) {
      const float value = __bfloat162float(logits[index]);
      sum = __fadd_rn(sum, expf(__fsub_rn(value, maximum)));
    }
    first[chunk] = sum;
  }
  __syncthreads();
  const float exponential_sum =
      riley_cuda_fixed37::balanced_sum(first, second, partial_count);
  const float logarithm = logf(exponential_sum);
  for (uint64_t index = threadIdx.x; index < element_count;
       index += blockDim.x) {
    const float shifted =
        __fsub_rn(__bfloat162float(logits[index]), maximum);
    output[index] = __fsub_rn(shifted, logarithm);
  }
}

RileyCudaStatus launch_status(RileyCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RILEY_CUDA_ERROR_STAGE_LAUNCH, operation);
}

template <typename T>
void launch_rms_norm(const ResolvedSpan& input, const ResolvedSpan& weight,
                     const ResolvedSpan& output, uint64_t row_count,
                     uint64_t hidden_size, float epsilon,
                     uint64_t partial_count, uint64_t shared_bytes,
                     cudaStream_t stream) {
  const uint32_t blocks = riley_cuda_fixed37::block_count(row_count);
  fixed37_rms_norm_kernel<T>
      <<<blocks, riley_cuda_fixed37::kThreadsPerBlock,
         static_cast<size_t>(shared_bytes), stream>>>(
          reinterpret_cast<const T*>(input.data),
          reinterpret_cast<const T*>(weight.data),
          reinterpret_cast<T*>(output.data), row_count, hidden_size, epsilon,
          partial_count);
}

template <typename T>
void launch_residual_rms_norm(
    const ResolvedSpan& left, const ResolvedSpan& right,
    const ResolvedSpan& weight, const ResolvedSpan& residual_output,
    const ResolvedSpan& normalized_output, uint64_t row_count,
    uint64_t hidden_size, float epsilon, uint64_t partial_count,
    uint64_t shared_bytes, cudaStream_t stream) {
  const uint32_t blocks = riley_cuda_fixed37::block_count(row_count);
  fixed37_residual_rms_norm_kernel<T>
      <<<blocks, riley_cuda_fixed37::kThreadsPerBlock,
         static_cast<size_t>(shared_bytes), stream>>>(
          reinterpret_cast<const T*>(left.data),
          reinterpret_cast<const T*>(right.data),
          reinterpret_cast<const T*>(weight.data),
          reinterpret_cast<T*>(residual_output.data),
          reinterpret_cast<T*>(normalized_output.data), row_count,
          hidden_size, epsilon, partial_count);
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_fixed37_rms_norm_execute(
    const RileyCudaRmsNormParams* params, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 RMSNorm";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaRmsNormParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->weight.dtype != params->input.dtype ||
      params->output.dtype != params->input.dtype) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "input, weight, and output must share F32 or BF16 dtype");
  }
  if (params->hidden_size == 0 || !std::isfinite(params->epsilon) ||
      params->epsilon <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "hidden_size and finite positive epsilon are required");
  }
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  RileyCudaStatus status = validate_reduction_axis(
      params->hidden_size, &partial_count, &shared_bytes, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  uint64_t element_count = 0;
  if (!checked_multiply(params->row_count, params->hidden_size,
                        &element_count)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "RMSNorm shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t weight_bytes = 0;
  status = element_bytes(element_count, params->input.dtype, &tensor_bytes,
                         error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->hidden_size, params->weight.dtype,
                           &weight_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan input{};
  ResolvedSpan weight{};
  ResolvedSpan output{};
  status = resolve_span(params->input, tensor_bytes, &input, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->weight, weight_bytes, &weight, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, tensor_bytes, &output, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, true, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, weight, false, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, weight, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(weight.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
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
    status = launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (params->input.dtype == RILEY_CUDA_DTYPE_F32) {
      launch_rms_norm<float>(input, weight, output, params->row_count,
                             params->hidden_size, params->epsilon,
                             partial_count, shared_bytes, stream->stream);
    } else {
      launch_rms_norm<__nv_bfloat16>(
          input, weight, output, params->row_count, params->hidden_size,
          params->epsilon, partial_count, shared_bytes, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RileyCudaStatus
riley_cuda_fixed37_residual_rms_norm_execute(
    const RileyCudaResidualRmsNormParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 fused residual RMSNorm";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaResidualRmsNormParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->left.dtype) ||
      params->right.dtype != params->left.dtype ||
      params->weight.dtype != params->left.dtype ||
      params->residual_output.dtype != params->left.dtype ||
      params->normalized_output.dtype != params->left.dtype) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "all residual RMSNorm spans must share F32 or BF16 dtype");
  }
  if (params->hidden_size == 0 || !std::isfinite(params->epsilon) ||
      params->epsilon <= 0.0F) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "hidden_size and finite positive epsilon are required");
  }
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  RileyCudaStatus status = validate_reduction_axis(
      params->hidden_size, &partial_count, &shared_bytes, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  uint64_t element_count = 0;
  if (!checked_multiply(params->row_count, params->hidden_size,
                        &element_count)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "residual RMSNorm shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t weight_bytes = 0;
  status = element_bytes(element_count, params->left.dtype, &tensor_bytes,
                         error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->hidden_size, params->weight.dtype,
                           &weight_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan left{};
  ResolvedSpan right{};
  ResolvedSpan weight{};
  ResolvedSpan residual_output{};
  ResolvedSpan normalized_output{};
  status = resolve_span(params->left, tensor_bytes, &left, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->right, tensor_bytes, &right, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->weight, weight_bytes, &weight, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->residual_output, tensor_bytes,
                          &residual_output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->normalized_output, tensor_bytes,
                          &normalized_output, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, left, true, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, right, true, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, weight, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, left, false, error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, right, false, error,
                            kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, weight, false, error,
                            kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, residual_output, false, error,
                            kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {left, right, weight, residual_output,
                                normalized_output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(left.buffer) || !uses.add(right.buffer) ||
      !uses.add(weight.buffer) || !uses.add(residual_output.buffer) ||
      !uses.add(normalized_output.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
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
    status = launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    if (params->left.dtype == RILEY_CUDA_DTYPE_F32) {
      launch_residual_rms_norm<float>(
          left, right, weight, residual_output, normalized_output,
          params->row_count, params->hidden_size, params->epsilon,
          partial_count, shared_bytes, stream->stream);
    } else {
      launch_residual_rms_norm<__nv_bfloat16>(
          left, right, weight, residual_output, normalized_output,
          params->row_count, params->hidden_size, params->epsilon,
          partial_count, shared_bytes, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RileyCudaStatus riley_cuda_fixed37_log_softmax_execute(
    const RileyCudaFixed37LogSoftmaxParams* params,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute fixed37 log-softmax";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  const RileyCudaFixed37LogSoftmaxParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (params->logits.dtype != RILEY_CUDA_DTYPE_BF16 ||
      params->output.dtype != RILEY_CUDA_DTYPE_F32) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "logits must be BF16 and output must be F32");
  }
  uint64_t partial_count = 0;
  uint64_t shared_bytes = 0;
  RileyCudaStatus status = validate_reduction_axis(
      params->element_count, &partial_count, &shared_bytes, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  uint64_t logits_bytes = 0;
  uint64_t output_bytes = 0;
  status = element_bytes(params->element_count, RILEY_CUDA_DTYPE_BF16,
                         &logits_bytes, error, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->element_count, RILEY_CUDA_DTYPE_F32,
                           &output_bytes, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan logits{};
  ResolvedSpan output{};
  status = resolve_span(params->logits, logits_bytes, &logits, error,
                        kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, output_bytes, &output, error,
                          kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, logits, false, error, kOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {logits, output};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(logits.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    fixed37_log_softmax_kernel
        <<<1, riley_cuda_fixed37::kThreadsPerBlock,
           static_cast<size_t>(shared_bytes), stream->stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(logits.data),
            reinterpret_cast<float*>(output.data), params->element_count,
            partial_count);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}
