#include "ffi_internal.hpp"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::try_acquire_exclusive_use;
using rustinfer_cuda_internal::validation_error;

constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaximumBlocks = 65535;
constexpr size_t kMaximumPrimitiveBuffers = 5;
constexpr uint64_t kHuggingFaceSmolLm2HiddenSize = 576;
constexpr uint64_t kHuggingFaceSmolLm2MaximumRows = 8192;
constexpr uint32_t kHuggingFaceSmolLm2EpsilonBits = 0x3727c5acU;

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t used_bytes;
  RustInferCudaDType dtype;
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

bool checked_add(uint64_t left, uint64_t right, uint64_t* output) noexcept {
  if (output == nullptr ||
      right > std::numeric_limits<uint64_t>::max() - left) {
    return false;
  }
  *output = left + right;
  return true;
}

bool is_hugging_face_smollm2_rms_norm_contract(
    RustInferCudaDType dtype, uint64_t row_count, uint64_t hidden_size,
    float epsilon) noexcept {
  uint32_t epsilon_bits = 0;
  static_assert(sizeof(epsilon_bits) == sizeof(epsilon));
  std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
  return dtype == RUSTINFER_CUDA_DTYPE_BF16 &&
         row_count <= kHuggingFaceSmolLm2MaximumRows &&
         hidden_size == kHuggingFaceSmolLm2HiddenSize &&
         epsilon_bits == kHuggingFaceSmolLm2EpsilonBits;
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

RustInferCudaStatus element_bytes(uint64_t element_count,
                                  RustInferCudaDType dtype,
                                  uint64_t* output,
                                  RustInferCudaErrorInfo* error,
                                  const char* operation) noexcept {
  const uint64_t width = dtype_size(dtype);
  if (width == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "span has an unsupported dtype");
  }
  if (!checked_multiply(element_count, width, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "element byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus resolve_span(const RustInferCudaBufferSpan& span,
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

bool exact_alias(const ResolvedSpan& left,
                 const ResolvedSpan& right) noexcept {
  return left.buffer == right.buffer &&
         left.byte_offset == right.byte_offset &&
         left.used_bytes == right.used_bytes;
}

RustInferCudaStatus reject_overlap(const ResolvedSpan& write,
                                   const ResolvedSpan& read,
                                   bool exact_alias_allowed,
                                   RustInferCudaErrorInfo* error,
                                   const char* operation) noexcept {
  if (overlaps(write, read) &&
      !(exact_alias_allowed && exact_alias(write, read))) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "unsupported partial or write/input span overlap");
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
          "stream and device spans belong to different context owners");
    }
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

class ExclusiveUses final {
 public:
  explicit ExclusiveUses(RustInferCudaStream* stream) noexcept
      : stream_(stream), buffers_{}, buffer_count_(0), acquired_count_(0),
        stream_acquired_(false), command_batch_(false) {}

  ExclusiveUses(const ExclusiveUses&) = delete;
  ExclusiveUses& operator=(const ExclusiveUses&) = delete;

  bool add(RustInferCudaDeviceBuffer* buffer) noexcept {
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
            "a device buffer already has an active asynchronous use");
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
            "a device buffer already has an active asynchronous use");
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
  RustInferCudaDeviceBuffer* buffers_[kMaximumPrimitiveBuffers];
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

RustInferCudaStatus complete_execution(ExclusiveUses* uses,
                                       CurrentContext* scope,
                                       RustInferCudaStream* stream,
                                       RustInferCudaStatus operation_status,
                                       bool launch_attempted,
                                       RustInferCudaErrorInfo* error,
                                       const char* operation,
                                       bool defer_in_command_batch = true) noexcept {
  if (uses->command_batch() && defer_in_command_batch) {
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

  // A restoration failure poisons the context in CurrentContext::leave. Even
  // if CUDA work completed, retaining exclusive uses keeps all opaque handles
  // alive and prevents a later close/reuse from turning ambiguity into UAF.
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
  return round_to_storage<T>(left * right);
}

__global__ void reset_embedding_error(
    RustInferCudaEmbeddingErrorReport* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    report->struct_size = sizeof(RustInferCudaEmbeddingErrorReport);
    report->code = RUSTINFER_CUDA_EMBEDDING_ERROR_NONE;
    report->token_position = UINT64_MAX;
    report->token_id = 0;
    report->reserved = 0;
  }
}

template <typename T>
__global__ void embedding_kernel(
    const T* table, const uint32_t* token_ids, T* output,
    const RustInferCudaEmbeddingErrorReport* report, uint64_t hidden_size,
    uint64_t output_elements) {
  // The preceding validation kernel is ordered on this stream. Suppress every
  // write when any token is invalid so CPU and GPU share fail-before-write.
  if (report->token_position != UINT64_MAX) {
    return;
  }
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < output_elements; index += stride) {
    const uint64_t token_position = index / hidden_size;
    const uint64_t hidden_index = index - token_position * hidden_size;
    const uint64_t token_id = token_ids[token_position];
    // Gather is a storage move, so preserve BF16/F32 NaN payload bits rather
    // than round-trip an otherwise untouched value through FP32.
    output[index] = table[token_id * hidden_size + hidden_index];
  }
}

__global__ void validate_embedding_tokens(
    const uint32_t* token_ids, uint64_t token_count,
    uint64_t vocabulary_size, RustInferCudaEmbeddingErrorReport* report) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t position = first; position < token_count; position += stride) {
    if (token_ids[position] >= vocabulary_size) {
      atomicMin(
          reinterpret_cast<unsigned long long*>(&report->token_position),
          static_cast<unsigned long long>(position));
    }
  }
}

__global__ void finalize_embedding_error(
    const uint32_t* token_ids, uint64_t token_count,
    RustInferCudaEmbeddingErrorReport* report) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    if (report->token_position == UINT64_MAX) {
      report->token_position = 0;
      report->token_id = 0;
      report->code = RUSTINFER_CUDA_EMBEDDING_ERROR_NONE;
    } else if (report->token_position < token_count) {
      report->token_id = token_ids[report->token_position];
      report->code = RUSTINFER_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE;
    }
  }
}

template <typename T>
__global__ void rms_norm_kernel(const T* input, const T* weight, T* output,
                                uint64_t row_count, uint64_t hidden_size,
                                float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t base = row * hidden_size;
    float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const float value = load_f32(input, base + column);
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
      const float normalized = load_f32(input, base + column) * inverse_rms;
      // Hugging Face Llama RMSNorm converts the normalized FP32 activation
      // back to the input dtype before multiplying by the dtype-matched
      // learned weight. Preserve that explicit BF16 boundary here.
      const float normalized_for_weight = round_to_storage<T>(normalized);
      store_f32(output, base + column,
                normalized_for_weight * load_f32(weight, column));
    }
    __syncthreads();
  }
}

template <bool kResidual>
__global__ void hugging_face_smollm2_rms_norm_kernel(
    const __nv_bfloat16* left, const __nv_bfloat16* right,
    const __nv_bfloat16* weight, __nv_bfloat16* residual_output,
    __nv_bfloat16* normalized_output, uint64_t row_count) {
  extern __shared__ float partial_sums[];
  __shared__ float inverse_rms_by_row[16];
  const uint64_t row = static_cast<uint64_t>(blockIdx.x) * blockDim.y +
                       threadIdx.y;
  const bool valid_row = row < row_count;
  const uint64_t base = row * kHuggingFaceSmolLm2HiddenSize;

  float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
  if (valid_row) {
    for (uint32_t vector_index = threadIdx.x;
         vector_index < kHuggingFaceSmolLm2HiddenSize / 4;
         vector_index += blockDim.x) {
      const uint64_t column = static_cast<uint64_t>(vector_index) * 4;
      float values[4];
#pragma unroll
      for (uint32_t component = 0; component < 4; ++component) {
        const uint64_t index = base + column + component;
        float value = __bfloat162float(left[index]);
        if constexpr (kResidual) {
          value = __bfloat162float(__float2bfloat16_rn(
              __fadd_rn(value, __bfloat162float(right[index]))));
          residual_output[index] = __float2bfloat16_rn(value);
        }
        values[component] = value;
      }
#pragma unroll
      for (uint32_t component = 0; component < 4; ++component) {
        accumulators[component] = __fadd_rn(
            accumulators[component],
            __fmul_rn(values[component], values[component]));
      }
    }
  }

  float sum = __fadd_rn(accumulators[0], accumulators[1]);
  sum = __fadd_rn(sum, accumulators[2]);
  sum = __fadd_rn(sum, accumulators[3]);
  const uint32_t shared_index =
      threadIdx.x + threadIdx.y * blockDim.x;
  partial_sums[shared_index] = sum;
  for (uint32_t offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
    __syncthreads();
    if (threadIdx.x < offset) {
      sum = __fadd_rn(sum, partial_sums[shared_index + offset]);
      partial_sums[shared_index] = sum;
    }
  }
  __syncthreads();
  for (uint32_t offset = 16; offset != 0; offset >>= 1) {
    sum = __fadd_rn(
        sum, __shfl_down_sync(0xffffffffU, sum, static_cast<int>(offset)));
  }
  if (threadIdx.x == 0) {
    const float mean = __fmul_rn(sum, 1.0F / 576.0F);
    inverse_rms_by_row[threadIdx.y] =
        rsqrtf(__fadd_rn(mean, 1.0e-5F));
  }
  __syncthreads();
  if (!valid_row) {
    return;
  }

  const float inverse_rms = inverse_rms_by_row[threadIdx.y];
  for (uint64_t column = threadIdx.x;
       column < kHuggingFaceSmolLm2HiddenSize; column += blockDim.x) {
    const uint64_t index = base + column;
    float value = 0.0F;
    if constexpr (kResidual) {
      value = __bfloat162float(residual_output[index]);
    } else {
      value = __bfloat162float(left[index]);
    }
    const __nv_bfloat16 staged =
        __float2bfloat16_rn(__fmul_rn(value, inverse_rms));
    normalized_output[index] = __float2bfloat16_rn(__fmul_rn(
        __bfloat162float(weight[column]), __bfloat162float(staged)));
  }
}

template <typename T>
__global__ void residual_add_kernel(const T* left, const T* right, T* output,
                                    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const float sum = load_f32(left, index) + load_f32(right, index);
    store_f32(output, index, sum);
  }
}

template <typename T>
__global__ void residual_rms_norm_kernel(
    const T* left, const T* right, const T* weight, T* residual_output,
    T* normalized_output, uint64_t row_count, uint64_t hidden_size,
    float epsilon) {
  extern __shared__ float partial_sums[];
  for (uint64_t row = blockIdx.x; row < row_count; row += gridDim.x) {
    const uint64_t base = row * hidden_size;
    float sum = 0.0F;
    for (uint64_t column = threadIdx.x; column < hidden_size;
         column += blockDim.x) {
      const uint64_t index = base + column;
      // Match the standalone residual store before RMSNorm observes the
      // activation. BF16 therefore rounds once at the exact same boundary.
      const float residual =
          round_to_storage<T>(load_f32(left, index) + load_f32(right, index));
      store_f32(residual_output, index, residual);
      sum += residual * residual;
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
      const uint64_t index = base + column;
      const float normalized = load_f32(residual_output, index) * inverse_rms;
      const float normalized_for_weight = round_to_storage<T>(normalized);
      store_f32(normalized_output, index,
                normalized_for_weight * load_f32(weight, column));
    }
    __syncthreads();
  }
}

__global__ void row_bias_add_in_place_kernel(__nv_bfloat16* matrix,
                                             const __nv_bfloat16* bias,
                                             uint64_t column_count,
                                             uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t column = index % column_count;
    const float sum = __fadd_rn(__bfloat162float(matrix[index]),
                                __bfloat162float(bias[column]));
    matrix[index] = __float2bfloat16_rn(sum);
  }
}

template <typename T>
__global__ void silu_kernel(const T* input, T* output,
                            uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    const float value = load_f32(input, index);
    store_f32(output, index, value / (1.0F + expf(-value)));
  }
}

template <typename T>
__global__ void gated_multiply_kernel(const T* activated_gate, const T* up,
                                      T* output, uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    store_f32(output, index,
              load_f32(activated_gate, index) * load_f32(up, index));
  }
}

template <typename T>
__global__ void rope_kernel(const T* input, const float* cos,
                            const float* sin, T* output,
                            uint64_t head_count, uint64_t head_size,
                            uint64_t rotary_dimension, uint64_t position_offset,
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
    const uint64_t token = head_linear / head_count;
    const uint64_t base = head_linear * head_size;
    if (unit < half) {
      const uint64_t first_index = base + unit;
      const uint64_t second_index = first_index + half;
      const uint64_t table_index = (position_offset + token) * half + unit;
      const float first_value = load_f32(input, first_index);
      const float second_value = load_f32(input, second_index);
      // The pinned Llama implementation casts its FP32 rotary tables to the
      // query dtype, rounds each elementwise product in that dtype, and only
      // then performs the final add. Model those BF16 boundaries explicitly;
      // the F32 specialization remains algebraically unchanged.
      const float cosine = round_to_storage<T>(cos[table_index]);
      const float sine = round_to_storage<T>(sin[table_index]);
      const float first_cosine = multiply_in_storage<T>(first_value, cosine);
      const float second_sine = multiply_in_storage<T>(second_value, sine);
      const float second_cosine = multiply_in_storage<T>(second_value, cosine);
      const float first_sine = multiply_in_storage<T>(first_value, sine);
      store_f32(output, first_index,
                first_cosine - second_sine);
      store_f32(output, second_index,
                second_cosine + first_sine);
    } else {
      const uint64_t dimension = rotary_dimension + (unit - half);
      // The non-rotary tail is an exact storage copy, including NaN payloads.
      output[base + dimension] = input[base + dimension];
    }
  }
}

__global__ void cast_bf16_to_f32_kernel(const __nv_bfloat16* input,
                                        float* output,
                                        uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    output[index] = __bfloat162float(input[index]);
  }
}

__global__ void cast_f32_to_bf16_kernel(const float* input,
                                        __nv_bfloat16* output,
                                        uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
    output[index] = __float2bfloat16_rn(input[index]);
  }
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

template <typename T>
void launch_embedding(const ResolvedSpan& table, const ResolvedSpan& token_ids,
                      const ResolvedSpan& output,
                      const ResolvedSpan& device_error,
                      uint64_t hidden_size, uint64_t output_elements,
                      cudaStream_t stream) {
  embedding_kernel<T><<<block_count(output_elements), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(table.data),
      reinterpret_cast<const uint32_t*>(token_ids.data),
      reinterpret_cast<T*>(output.data),
      reinterpret_cast<RustInferCudaEmbeddingErrorReport*>(device_error.data),
      hidden_size, output_elements);
}

template <typename T>
void launch_rms_norm(const ResolvedSpan& input, const ResolvedSpan& weight,
                     const ResolvedSpan& output, uint64_t row_count,
                     uint64_t hidden_size, float epsilon,
                     cudaStream_t stream) {
  const uint32_t blocks = static_cast<uint32_t>(
      row_count < kMaximumBlocks ? row_count : kMaximumBlocks);
  rms_norm_kernel<T><<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
      reinterpret_cast<const T*>(input.data),
      reinterpret_cast<const T*>(weight.data),
      reinterpret_cast<T*>(output.data), row_count, hidden_size, epsilon);
}

dim3 hugging_face_smollm2_rms_norm_block(uint64_t row_count) noexcept {
  uint32_t block_height = 1;
  while (block_height < 16 &&
         static_cast<uint64_t>(block_height) * 2 <= row_count) {
    block_height *= 2;
  }
  const uint32_t block_width =
      block_height <= 4 ? 128 : (block_height == 8 ? 64 : 32);
  return dim3(block_width, block_height, 1);
}

void launch_hugging_face_smollm2_rms_norm(
    const ResolvedSpan& input, const ResolvedSpan& weight,
    const ResolvedSpan& output, uint64_t row_count, cudaStream_t stream) {
  const dim3 block = hugging_face_smollm2_rms_norm_block(row_count);
  const uint32_t blocks = static_cast<uint32_t>(
      (row_count + block.y - 1) / block.y);
  const size_t shared_bytes =
      static_cast<size_t>(block.x) * block.y * sizeof(float);
  hugging_face_smollm2_rms_norm_kernel<false>
      <<<blocks, block, shared_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(input.data), nullptr,
          reinterpret_cast<const __nv_bfloat16*>(weight.data), nullptr,
          reinterpret_cast<__nv_bfloat16*>(output.data), row_count);
}

template <typename T>
void launch_residual_add(const ResolvedSpan& left, const ResolvedSpan& right,
                         const ResolvedSpan& output, uint64_t element_count,
                         cudaStream_t stream) {
  residual_add_kernel<T><<<block_count(element_count), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(left.data),
      reinterpret_cast<const T*>(right.data),
      reinterpret_cast<T*>(output.data), element_count);
}

template <typename T>
void launch_residual_rms_norm(
    const ResolvedSpan& left, const ResolvedSpan& right,
    const ResolvedSpan& weight, const ResolvedSpan& residual_output,
    const ResolvedSpan& normalized_output, uint64_t row_count,
    uint64_t hidden_size, float epsilon, cudaStream_t stream) {
  const uint32_t blocks = static_cast<uint32_t>(
      row_count < kMaximumBlocks ? row_count : kMaximumBlocks);
  residual_rms_norm_kernel<T>
      <<<blocks, kThreads, kThreads * sizeof(float), stream>>>(
          reinterpret_cast<const T*>(left.data),
          reinterpret_cast<const T*>(right.data),
          reinterpret_cast<const T*>(weight.data),
          reinterpret_cast<T*>(residual_output.data),
          reinterpret_cast<T*>(normalized_output.data), row_count, hidden_size,
          epsilon);
}

void launch_hugging_face_smollm2_residual_rms_norm(
    const ResolvedSpan& left, const ResolvedSpan& right,
    const ResolvedSpan& weight, const ResolvedSpan& residual_output,
    const ResolvedSpan& normalized_output, uint64_t row_count,
    cudaStream_t stream) {
  const dim3 block = hugging_face_smollm2_rms_norm_block(row_count);
  const uint32_t blocks = static_cast<uint32_t>(
      (row_count + block.y - 1) / block.y);
  const size_t shared_bytes =
      static_cast<size_t>(block.x) * block.y * sizeof(float);
  hugging_face_smollm2_rms_norm_kernel<true>
      <<<blocks, block, shared_bytes, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(left.data),
          reinterpret_cast<const __nv_bfloat16*>(right.data),
          reinterpret_cast<const __nv_bfloat16*>(weight.data),
          reinterpret_cast<__nv_bfloat16*>(residual_output.data),
          reinterpret_cast<__nv_bfloat16*>(normalized_output.data), row_count);
}

void launch_row_bias_add_in_place(const ResolvedSpan& matrix,
                                  const ResolvedSpan& bias,
                                  uint64_t column_count,
                                  uint64_t element_count,
                                  cudaStream_t stream) {
  row_bias_add_in_place_kernel
      <<<block_count(element_count), kThreads, 0, stream>>>(
          reinterpret_cast<__nv_bfloat16*>(matrix.data),
          reinterpret_cast<const __nv_bfloat16*>(bias.data), column_count,
          element_count);
}

template <typename T>
void launch_silu(const ResolvedSpan& input, const ResolvedSpan& output,
                 uint64_t element_count, cudaStream_t stream) {
  silu_kernel<T><<<block_count(element_count), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(input.data),
      reinterpret_cast<T*>(output.data), element_count);
}

template <typename T>
void launch_gated_multiply(const ResolvedSpan& activated_gate,
                           const ResolvedSpan& up,
                           const ResolvedSpan& output, uint64_t element_count,
                           cudaStream_t stream) {
  gated_multiply_kernel<T>
      <<<block_count(element_count), kThreads, 0, stream>>>(
          reinterpret_cast<const T*>(activated_gate.data),
          reinterpret_cast<const T*>(up.data),
          reinterpret_cast<T*>(output.data), element_count);
}

template <typename T>
void launch_rope(const ResolvedSpan& input, const ResolvedSpan& cos,
                 const ResolvedSpan& sin, const ResolvedSpan& output,
                 uint64_t head_count, uint64_t head_size,
                 uint64_t rotary_dimension, uint64_t position_offset,
                 uint64_t work_items, cudaStream_t stream) {
  rope_kernel<T><<<block_count(work_items), kThreads, 0, stream>>>(
      reinterpret_cast<const T*>(input.data),
      reinterpret_cast<const float*>(cos.data),
      reinterpret_cast<const float*>(sin.data),
      reinterpret_cast<T*>(output.data), head_count, head_size,
      rotary_dimension, position_offset, work_items);
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_embedding_execute(
    const RustInferCudaEmbeddingParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute embedding gather";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  // Stabilize every descriptor before touching the caller's host output. This
  // also prevents a malformed raw C caller from corrupting fields mid-call by
  // placing out_report inside its params storage.
  const RustInferCudaEmbeddingParams stable_params = *params;
  params = &stable_params;
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 3)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (params->out_report == nullptr ||
      params->out_report->struct_size < sizeof(*params->out_report)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "out_report is null or has an incompatible struct_size");
  }
  const uint32_t caller_report_size = params->out_report->struct_size;
  std::memset(params->out_report, 0, sizeof(*params->out_report));
  params->out_report->struct_size = caller_report_size;

  if (!arithmetic_dtype(params->table.dtype) ||
      params->output.dtype != params->table.dtype ||
      params->token_ids.dtype != RUSTINFER_CUDA_DTYPE_U32 ||
      params->device_error_scratch.dtype != RUSTINFER_CUDA_DTYPE_U8) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "embedding requires matching F32/BF16 table/output, U32 ids, and U8 scratch");
  }
  if (params->vocabulary_size == 0 || params->hidden_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "vocabulary_size and hidden_size must be non-zero");
  }

  uint64_t table_elements = 0;
  uint64_t output_elements = 0;
  uint64_t table_bytes = 0;
  uint64_t token_bytes = 0;
  uint64_t output_bytes = 0;
  if (!checked_multiply(params->vocabulary_size, params->hidden_size,
                        &table_elements) ||
      !checked_multiply(params->token_count, params->hidden_size,
                        &output_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "embedding shape product overflows uint64_t");
  }
  RustInferCudaStatus status =
      element_bytes(table_elements, params->table.dtype, &table_bytes, error,
                    kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->token_count, params->token_ids.dtype,
                           &token_bytes, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(output_elements, params->output.dtype,
                           &output_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan table{};
  ResolvedSpan token_ids{};
  ResolvedSpan output{};
  ResolvedSpan device_error{};
  status = resolve_span(params->table, table_bytes, &table, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->token_ids, token_bytes, &token_ids, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, output_bytes, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->device_error_scratch,
                          sizeof(RustInferCudaEmbeddingErrorReport),
                          &device_error, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (device_error.byte_offset % alignof(RustInferCudaEmbeddingErrorReport) !=
      0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "embedding device error scratch is not 8-byte aligned");
  }
  status = reject_overlap(output, table, false, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, token_ids, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, device_error, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(device_error, table, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(device_error, token_ids, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {table, token_ids, output, device_error};
  status = validate_contexts(stream, spans, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveUses uses(stream);
  if (!uses.add(table.buffer) || !uses.add(token_ids.buffer) ||
      !uses.add(output.buffer) || !uses.add(device_error.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->token_count == 0) {
    return uses.release_completed()
               ? RUSTINFER_CUDA_STATUS_SUCCESS
               : internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "exclusive-use accounting was corrupted");
  }

  RustInferCudaEmbeddingErrorReport host_report{};
  host_report.struct_size = sizeof(host_report);
  bool launch_attempted = false;
  CurrentContext scope(stream->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prior_launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    reset_embedding_error<<<1, 1, 0, stream->stream>>>(
        reinterpret_cast<RustInferCudaEmbeddingErrorReport*>(
            device_error.data));
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    validate_embedding_tokens
        <<<block_count(params->token_count), kThreads, 0, stream->stream>>>(
            reinterpret_cast<const uint32_t*>(token_ids.data),
            params->token_count, params->vocabulary_size,
            reinterpret_cast<RustInferCudaEmbeddingErrorReport*>(
                device_error.data));
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    if (params->table.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      launch_embedding<float>(table, token_ids, output, device_error,
                              params->hidden_size, output_elements,
                              stream->stream);
    } else {
      launch_embedding<__nv_bfloat16>(
          table, token_ids, output, device_error, params->hidden_size,
          output_elements, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    finalize_embedding_error<<<1, 1, 0, stream->stream>>>(
        reinterpret_cast<const uint32_t*>(token_ids.data), params->token_count,
        reinterpret_cast<RustInferCudaEmbeddingErrorReport*>(
            device_error.data));
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(
        cudaMemcpyAsync(&host_report, device_error.data, sizeof(host_report),
                        cudaMemcpyDeviceToHost, stream->stream),
        error, RUSTINFER_CUDA_ERROR_STAGE_COPY,
        "copy completed embedding error report");
  }
  // host_report is stack-backed and is read immediately below. Complete its
  // asynchronous D2H copy even inside a command batch; release_completed()
  // remains a batch no-op, so the ledger leases stay held until batch end.
  status = complete_execution(&uses, &scope, stream, status,
                              launch_attempted, error, kOperation, false);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  if (host_report.struct_size != sizeof(host_report) ||
      host_report.reserved != 0) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kOperation,
                          "device embedding error report is malformed");
  }
  if (host_report.code == RUSTINFER_CUDA_EMBEDDING_ERROR_NONE) {
    if (host_report.token_position != 0 || host_report.token_id != 0) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            kOperation,
                            "successful embedding error report is inconsistent");
    }
    std::memcpy(params->out_report, &host_report, sizeof(host_report));
    params->out_report->struct_size = caller_report_size;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if (host_report.code !=
          RUSTINFER_CUDA_EMBEDDING_ERROR_TOKEN_OUT_OF_RANGE ||
      host_report.token_position >= params->token_count ||
      host_report.token_id < params->vocabulary_size) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kOperation,
                          "device embedding error report is inconsistent");
  }
  // Publish only a fully validated report. Any malformed device record leaves
  // the caller's pre-cleared NONE report intact so a higher layer cannot
  // misclassify an INTERNAL contract failure as a token OOB error.
  std::memcpy(params->out_report, &host_report, sizeof(host_report));
  params->out_report->struct_size = caller_report_size;
  status = set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
                     RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                     "token id exceeds vocabulary range");
  if (error != nullptr && error->struct_size >= sizeof(*error)) {
    std::snprintf(error->message, sizeof(error->message),
                  "%s: token[%llu]=%llu is outside vocabulary_size=%llu",
                  kOperation,
                  static_cast<unsigned long long>(host_report.token_position),
                  static_cast<unsigned long long>(host_report.token_id),
                  static_cast<unsigned long long>(params->vocabulary_size));
  }
  return status;
}

RustInferCudaStatus execute_rms_norm_impl(
    const RustInferCudaRmsNormParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error, bool hugging_face_smollm2) noexcept {
  const char* kOperation = hugging_face_smollm2
                               ? "execute Hugging Face SmolLM2 RMSNorm"
                               : "execute RMSNorm";
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->weight.dtype != params->input.dtype ||
      params->output.dtype != params->input.dtype) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "input, weight, and output must share F32 or BF16 dtype");
  }
  if (params->hidden_size == 0 || !std::isfinite(params->epsilon) ||
      params->epsilon <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "hidden_size and finite positive epsilon are required");
  }
  if (hugging_face_smollm2 &&
      !is_hugging_face_smollm2_rms_norm_contract(
          params->input.dtype, params->row_count, params->hidden_size,
          params->epsilon)) {
    return set_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
        RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "the Hugging Face SmolLM2 path requires BF16, hidden_size=576, "
        "row_count<=8192, and epsilon=1e-5 exactly");
  }
  uint64_t element_count = 0;
  if (!checked_multiply(params->row_count, params->hidden_size,
                        &element_count)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "RMSNorm shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t weight_bytes = 0;
  RustInferCudaStatus status =
      element_bytes(element_count, params->input.dtype, &tensor_bytes, error,
                    kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->hidden_size, params->weight.dtype,
                           &weight_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan input{};
  ResolvedSpan weight{};
  ResolvedSpan output{};
  status = resolve_span(params->input, tensor_bytes, &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->weight, weight_bytes, &weight, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, tensor_bytes, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, weight, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, weight, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(weight.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->row_count == 0) {
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
    if (hugging_face_smollm2) {
      launch_hugging_face_smollm2_rms_norm(
          input, weight, output, params->row_count, stream->stream);
    } else if (params->input.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      launch_rms_norm<float>(input, weight, output, params->row_count,
                             params->hidden_size, params->epsilon,
                             stream->stream);
    } else {
      launch_rms_norm<__nv_bfloat16>(
          input, weight, output, params->row_count, params->hidden_size,
          params->epsilon, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_rms_norm_execute(
    const RustInferCudaRmsNormParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  return execute_rms_norm_impl(params, stream, error, false);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_hugging_face_smollm2_rms_norm_execute(
    const RustInferCudaRmsNormParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  return execute_rms_norm_impl(params, stream, error, true);
}

extern "C" RustInferCudaStatus rustinfer_cuda_residual_add_execute(
    const RustInferCudaResidualAddParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute residual add";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->left.dtype) ||
      params->right.dtype != params->left.dtype ||
      params->output.dtype != params->left.dtype) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "left, right, and output must share F32 or BF16 dtype");
  }
  uint64_t byte_count = 0;
  RustInferCudaStatus status =
      element_bytes(params->element_count, params->left.dtype, &byte_count,
                    error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan left{};
  ResolvedSpan right{};
  ResolvedSpan output{};
  status = resolve_span(params->left, byte_count, &left, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->right, byte_count, &right, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, byte_count, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, left, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, right, true, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {left, right, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(left.buffer) || !uses.add(right.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->element_count == 0) {
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
    if (params->left.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      launch_residual_add<float>(left, right, output, params->element_count,
                                 stream->stream);
    } else {
      launch_residual_add<__nv_bfloat16>(
          left, right, output, params->element_count, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

RustInferCudaStatus execute_residual_rms_norm_impl(
    const RustInferCudaResidualRmsNormParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error,
    bool hugging_face_smollm2) noexcept {
  const char* kOperation =
      hugging_face_smollm2
          ? "execute Hugging Face SmolLM2 fused residual RMSNorm"
          : "execute fused residual RMSNorm";
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || params->reserved1 != 0 ||
      !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->left.dtype) ||
      params->right.dtype != params->left.dtype ||
      params->weight.dtype != params->left.dtype ||
      params->residual_output.dtype != params->left.dtype ||
      params->normalized_output.dtype != params->left.dtype) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "all residual RMSNorm spans must share F32 or BF16 dtype");
  }
  if (params->hidden_size == 0 || !std::isfinite(params->epsilon) ||
      params->epsilon <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "hidden_size and finite positive epsilon are required");
  }
  if (hugging_face_smollm2 &&
      !is_hugging_face_smollm2_rms_norm_contract(
          params->left.dtype, params->row_count, params->hidden_size,
          params->epsilon)) {
    return set_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
        RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "the Hugging Face SmolLM2 fused path requires BF16, "
        "hidden_size=576, row_count<=8192, and epsilon=1e-5 exactly");
  }
  uint64_t element_count = 0;
  if (!checked_multiply(params->row_count, params->hidden_size,
                        &element_count)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "residual RMSNorm shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t weight_bytes = 0;
  RustInferCudaStatus status =
      element_bytes(element_count, params->left.dtype, &tensor_bytes, error,
                    kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->hidden_size, params->weight.dtype,
                           &weight_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan left{};
  ResolvedSpan right{};
  ResolvedSpan weight{};
  ResolvedSpan residual_output{};
  ResolvedSpan normalized_output{};
  status = resolve_span(params->left, tensor_bytes, &left, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status =
        resolve_span(params->right, tensor_bytes, &right, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status =
        resolve_span(params->weight, weight_bytes, &weight, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->residual_output, tensor_bytes,
                          &residual_output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->normalized_output, tensor_bytes,
                          &normalized_output, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, left, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, right, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(residual_output, weight, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, left, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, right, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status =
        reject_overlap(normalized_output, weight, false, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(normalized_output, residual_output, false, error,
                            kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {left, right, weight, residual_output,
                                normalized_output};
  status = validate_contexts(stream, spans, 5, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(left.buffer) || !uses.add(right.buffer) ||
      !uses.add(weight.buffer) || !uses.add(residual_output.buffer) ||
      !uses.add(normalized_output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->row_count == 0) {
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
    if (hugging_face_smollm2) {
      launch_hugging_face_smollm2_residual_rms_norm(
          left, right, weight, residual_output, normalized_output,
          params->row_count, stream->stream);
    } else if (params->left.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      launch_residual_rms_norm<float>(
          left, right, weight, residual_output, normalized_output,
          params->row_count, params->hidden_size, params->epsilon,
          stream->stream);
    } else {
      launch_residual_rms_norm<__nv_bfloat16>(
          left, right, weight, residual_output, normalized_output,
          params->row_count, params->hidden_size, params->epsilon,
          stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_residual_rms_norm_execute(
    const RustInferCudaResidualRmsNormParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  return execute_residual_rms_norm_impl(params, stream, error, false);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_hugging_face_smollm2_residual_rms_norm_execute(
    const RustInferCudaResidualRmsNormParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  return execute_residual_rms_norm_impl(params, stream, error, true);
}

extern "C" RustInferCudaStatus rustinfer_cuda_row_bias_add_in_place_execute(
    const RustInferCudaRowBiasAddInPlaceParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute row-bias add in place";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (params->matrix.dtype != RUSTINFER_CUDA_DTYPE_BF16 ||
      params->bias.dtype != RUSTINFER_CUDA_DTYPE_BF16) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "matrix and bias must both use BF16 dtype");
  }
  if (params->column_count == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "column_count must be non-zero");
  }

  uint64_t matrix_elements = 0;
  if (!checked_multiply(params->row_count, params->column_count,
                        &matrix_elements)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "row-bias shape product overflows uint64_t");
  }
  uint64_t matrix_bytes = 0;
  uint64_t bias_bytes = 0;
  RustInferCudaStatus status =
      element_bytes(matrix_elements, RUSTINFER_CUDA_DTYPE_BF16,
                    &matrix_bytes, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->column_count, RUSTINFER_CUDA_DTYPE_BF16,
                           &bias_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ResolvedSpan matrix{};
  ResolvedSpan bias{};
  status = resolve_span(params->matrix, matrix_bytes, &matrix, error,
                        kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->bias, bias_bytes, &bias, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(matrix, bias, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const ResolvedSpan spans[] = {matrix, bias};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(matrix.buffer) || !uses.add(bias.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (matrix_elements == 0) {
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
    launch_row_bias_add_in_place(matrix, bias, params->column_count,
                                 matrix_elements, stream->stream);
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_silu_execute(
    const RustInferCudaSiluParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute SiLU";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "input and output must share F32 or BF16 dtype");
  }
  uint64_t byte_count = 0;
  RustInferCudaStatus status =
      element_bytes(params->element_count, params->input.dtype, &byte_count,
                    error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan input{};
  ResolvedSpan output{};
  status = resolve_span(params->input, byte_count, &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, byte_count, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, true, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, output};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->element_count == 0) {
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
      launch_silu<float>(input, output, params->element_count, stream->stream);
    } else {
      launch_silu<__nv_bfloat16>(input, output, params->element_count,
                                 stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_gated_multiply_execute(
    const RustInferCudaGatedMultiplyParams* params,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute activated-gate multiply";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->activated_gate.dtype) ||
      params->up.dtype != params->activated_gate.dtype ||
      params->output.dtype != params->activated_gate.dtype) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "activated_gate, up, and output must share F32 or BF16 dtype");
  }
  uint64_t byte_count = 0;
  RustInferCudaStatus status = element_bytes(
      params->element_count, params->activated_gate.dtype, &byte_count, error,
      kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan activated_gate{};
  ResolvedSpan up{};
  ResolvedSpan output{};
  status = resolve_span(params->activated_gate, byte_count, &activated_gate,
                        error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->up, byte_count, &up, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, byte_count, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, activated_gate, true, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, up, true, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {activated_gate, up, output};
  status = validate_contexts(stream, spans, 3, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(activated_gate.buffer) || !uses.add(up.buffer) ||
      !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->element_count == 0) {
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
    if (params->activated_gate.dtype == RUSTINFER_CUDA_DTYPE_F32) {
      launch_gated_multiply<float>(activated_gate, up, output,
                                   params->element_count, stream->stream);
    } else {
      launch_gated_multiply<__nv_bfloat16>(
          activated_gate, up, output, params->element_count, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_rope_execute(
    const RustInferCudaRopeParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute non-interleaved Llama RoPE";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  if (!arithmetic_dtype(params->input.dtype) ||
      params->output.dtype != params->input.dtype ||
      params->cos.dtype != RUSTINFER_CUDA_DTYPE_F32 ||
      params->sin.dtype != RUSTINFER_CUDA_DTYPE_F32) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "RoPE requires matching F32/BF16 input/output and F32 cos/sin tables");
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
  uint64_t position_end = 0;
  if (!checked_add(params->position_offset, params->token_count,
                   &position_end) ||
      position_end > params->table_position_count) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "requested positions exceed the cos/sin tables");
  }

  const uint64_t half = params->rotary_dimension / 2;
  const uint64_t tail = params->head_size - params->rotary_dimension;
  uint64_t heads_total = 0;
  uint64_t input_elements = 0;
  uint64_t table_elements = 0;
  uint64_t work_units_per_head = 0;
  uint64_t work_items = 0;
  if (!checked_multiply(params->token_count, params->head_count,
                        &heads_total) ||
      !checked_multiply(heads_total, params->head_size, &input_elements) ||
      !checked_multiply(params->table_position_count, half,
                        &table_elements) ||
      !checked_add(half, tail, &work_units_per_head) ||
      !checked_multiply(heads_total, work_units_per_head, &work_items)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "RoPE shape product overflows uint64_t");
  }
  uint64_t tensor_bytes = 0;
  uint64_t table_bytes = 0;
  RustInferCudaStatus status =
      element_bytes(input_elements, params->input.dtype, &tensor_bytes, error,
                    kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(table_elements, RUSTINFER_CUDA_DTYPE_F32,
                           &table_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan input{};
  ResolvedSpan cos{};
  ResolvedSpan sin{};
  ResolvedSpan output{};
  status = resolve_span(params->input, tensor_bytes, &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->cos, table_bytes, &cos, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->sin, table_bytes, &sin, error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, tensor_bytes, &output, error,
                          kOperation);
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
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, cos, sin, output};
  status = validate_contexts(stream, spans, 4, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(cos.buffer) ||
      !uses.add(sin.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->token_count == 0) {
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
      launch_rope<float>(input, cos, sin, output, params->head_count,
                         params->head_size,
                         params->rotary_dimension, params->position_offset,
                         work_items, stream->stream);
    } else {
      launch_rope<__nv_bfloat16>(
          input, cos, sin, output, params->head_count, params->head_size,
          params->rotary_dimension, params->position_offset,
          work_items, stream->stream);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus rustinfer_cuda_cast_execute(
    const RustInferCudaCastParams* params, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute BF16/F32 cast";
  clear_error(error);
  if (params == nullptr || params->struct_size < sizeof(*params)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params is null or has an incompatible struct_size");
  }
  if (params->reserved0 != 0 || !reserved_is_zero(params->reserved, 5)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "params reserved fields must be zero");
  }
  const bool bf16_to_f32 =
      params->input.dtype == RUSTINFER_CUDA_DTYPE_BF16 &&
      params->output.dtype == RUSTINFER_CUDA_DTYPE_F32;
  const bool f32_to_bf16 =
      params->input.dtype == RUSTINFER_CUDA_DTYPE_F32 &&
      params->output.dtype == RUSTINFER_CUDA_DTYPE_BF16;
  if (!bf16_to_f32 && !f32_to_bf16) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "cast supports only BF16-to-F32 or F32-to-BF16");
  }
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  RustInferCudaStatus status =
      element_bytes(params->element_count, params->input.dtype, &input_bytes,
                    error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = element_bytes(params->element_count, params->output.dtype,
                           &output_bytes, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ResolvedSpan input{};
  ResolvedSpan output{};
  status = resolve_span(params->input, input_bytes, &input, error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(params->output, output_bytes, &output, error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = reject_overlap(output, input, false, error, kOperation);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const ResolvedSpan spans[] = {input, output};
  status = validate_contexts(stream, spans, 2, error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(stream);
  if (!uses.add(input.buffer) || !uses.add(output.buffer)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                          kOperation, "primitive buffer set overflow");
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (params->element_count == 0) {
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
    if (bf16_to_f32) {
      cast_bf16_to_f32_kernel
          <<<block_count(params->element_count), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const __nv_bfloat16*>(input.data),
              reinterpret_cast<float*>(output.data), params->element_count);
    } else {
      cast_f32_to_bf16_kernel
          <<<block_count(params->element_count), kThreads, 0, stream->stream>>>(
              reinterpret_cast<const float*>(input.data),
              reinterpret_cast<__nv_bfloat16*>(output.data),
              params->element_count);
    }
    status = launch_status(error, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}
