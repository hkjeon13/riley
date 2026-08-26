#include "ffi_internal.hpp"
#include "fixed37_reduction.cuh"

#include <cuda_bf16.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint64_t kBfloat16Bytes = 2;
constexpr uint64_t kRequiredAlignment = 256;
constexpr size_t kMaximumGemmBuffers = 3;
constexpr const char* kCreateOperation = "create fixed37 GEMM plan";
constexpr const char* kQueryOperation = "query fixed37 GEMM plan";
constexpr const char* kExecuteOperation = "execute fixed37 GEMM";
constexpr const char* kCloseOperation = "close fixed37 GEMM plan";

struct GemmByteLengths {
  uint64_t input;
  uint64_t weight;
  uint64_t output;
};

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  void* data;
  uint64_t byte_offset;
  uint64_t byte_len;
};

}  // namespace

struct RustInferCudaFixed37GemmPlan {
  RustInferCudaFixed37GemmPlan(RustInferCudaContext* owning_context,
                               const RustInferCudaGemmConfig& plan_config,
                               const GemmByteLengths& lengths,
                               uint64_t shared_bytes) noexcept
      : owner(owning_context),
        config(plan_config),
        info{},
        input_bytes(lengths.input),
        weight_bytes(lengths.weight),
        output_bytes(lengths.output),
        active_uses(0) {
    info.struct_size = sizeof(info);
    info.backend = RUSTINFER_CUDA_GEMM_BACKEND_FIXED37;
    info.reduction_version = RUSTINFER_CUDA_FIXED37_REDUCTION_VERSION;
    info.chunk_elements = RUSTINFER_CUDA_FIXED37_CHUNK_ELEMENTS;
    info.accumulator_dtype = RUSTINFER_CUDA_DTYPE_F32;
    info.output_dtype = RUSTINFER_CUDA_DTYPE_BF16;
    info.threads_per_block = rustinfer_cuda_fixed37::kThreadsPerBlock;
    info.deterministic = RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED;
    info.dynamic_shared_memory_bytes = shared_bytes;
    info.workspace_bytes = 0;
    info.m = config.m;
    info.n = config.n;
    info.k = config.k;
  }

  RustInferCudaContext* owner;
  RustInferCudaGemmConfig config;
  RustInferCudaFixed37GemmPlanInfo info;
  uint64_t input_bytes;
  uint64_t weight_bytes;
  uint64_t output_bytes;
  std::atomic<uint32_t> active_uses;
};

namespace {

using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::command_batch_is_active;
using rustinfer_cuda_internal::command_batch_is_owned_by_current_thread;
using rustinfer_cuda_internal::command_batch_register_use;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_child;
using rustinfer_cuda_internal::release_exclusive_use;
using rustinfer_cuda_internal::retain_child;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::try_acquire_exclusive_use;
using rustinfer_cuda_internal::validation_error;

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

RustInferCudaStatus matrix_bytes(uint64_t rows, uint64_t columns,
                                 uint64_t* output,
                                 RustInferCudaErrorInfo* error) noexcept {
  uint64_t elements = 0;
  if (!checked_multiply(rows, columns, &elements) ||
      !checked_multiply(elements, kBfloat16Bytes, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "matrix byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_config(const RustInferCudaGemmConfig* config,
                                    GemmByteLengths* lengths,
                                    uint64_t* shared_bytes,
                                    RustInferCudaErrorInfo* error) noexcept {
  if (config == nullptr || lengths == nullptr || shared_bytes == nullptr ||
      config->struct_size < sizeof(*config)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "config is null or has an incompatible struct_size");
  }
  if (config->flags != 0 || config->reserved0 != 0 ||
      !reserved_is_zero(config->reserved, 3)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "flags and reserved fields must be zero");
  }
  if (config->m == 0 || config->n == 0 || config->k == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "M, N, and K must all be non-zero");
  }
  const uint64_t maximum_dimension =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (config->m > maximum_dimension || config->n > maximum_dimension ||
      config->k > maximum_dimension) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "M, N, and K must fit signed 32-bit dimensions");
  }
  if (config->input_dtype != RUSTINFER_CUDA_DTYPE_BF16 ||
      config->weight_dtype != RUSTINFER_CUDA_DTYPE_BF16 ||
      config->accumulator_dtype != RUSTINFER_CUDA_DTYPE_F32 ||
      config->output_dtype != RUSTINFER_CUDA_DTYPE_BF16) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "only BF16 input/weight/output with F32 accumulation is supported");
  }
  if (config->input_transpose != RUSTINFER_CUDA_GEMM_TRANSPOSE_N ||
      config->weight_transpose != RUSTINFER_CUDA_GEMM_TRANSPOSE_T ||
      config->input_layout != RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->weight_layout != RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->output_layout != RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->epilogue != RUSTINFER_CUDA_GEMM_EPILOGUE_NONE ||
      config->deterministic !=
          RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "only row-major X=N/W=T, epilogue-none, deterministic GEMM is supported");
  }

  const uint64_t chunks = rustinfer_cuda_fixed37::chunk_count(config->k);
  if (chunks == 0 ||
      chunks > rustinfer_cuda_fixed37::kMaximumChunkCount) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "K exceeds the fixed37 chunk-partial capacity");
  }
  *shared_bytes = rustinfer_cuda_fixed37::shared_bytes(config->k);

  RustInferCudaStatus status =
      matrix_bytes(config->m, config->k, &lengths->input, error);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = matrix_bytes(config->n, config->k, &lengths->weight, error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = matrix_bytes(config->m, config->n, &lengths->output, error);
  }
  return status;
}

RustInferCudaStatus resolve_exact_span(
    const RustInferCudaBufferSpan* span, RustInferCudaDType required_dtype,
    uint64_t required_bytes, ResolvedSpan* output,
    RustInferCudaErrorInfo* error, const char* dtype_detail) noexcept {
  if (span == nullptr || output == nullptr ||
      span->struct_size < sizeof(*span)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a span is null or has an incompatible struct_size");
  }
  if (!reserved_is_zero(span->reserved, 2)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "span reserved fields must be zero");
  }
  if (span->dtype != required_dtype) {
    return set_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
                     RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                     kExecuteOperation, dtype_detail);
  }
  if (span->buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a span buffer handle is null");
  }
  if (span->byte_offset % kRequiredAlignment != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "every span byte_offset must be 256-byte aligned");
  }
  if (span->byte_len != required_bytes) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
        "a span byte_len does not exactly match the prepared requirement");
  }
  if (span->byte_offset > span->buffer->byte_len ||
      span->byte_len > span->buffer->byte_len - span->byte_offset) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a declared span exceeds its opaque allocation");
  }
  if (required_bytes != 0 && span->buffer->device_data == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a non-empty span refers to a zero-byte allocation");
  }
  void* data = static_cast<uint8_t*>(span->buffer->device_data) +
               static_cast<size_t>(span->byte_offset);
  *output = ResolvedSpan{span->buffer, data, span->byte_offset,
                         required_bytes};
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

bool spans_overlap(const ResolvedSpan& left,
                   const ResolvedSpan& right) noexcept {
  if (left.buffer != right.buffer || left.byte_len == 0 ||
      right.byte_len == 0) {
    return false;
  }
  const uint64_t left_end = left.byte_offset + left.byte_len;
  const uint64_t right_end = right.byte_offset + right.byte_len;
  return left.byte_offset < right_end && right.byte_offset < left_end;
}

RustInferCudaStatus validate_span_relationships(
    RustInferCudaFixed37GemmPlan* plan, RustInferCudaStream* stream,
    const ResolvedSpan* spans, size_t count,
    RustInferCudaErrorInfo* error) noexcept {
  if (plan->owner == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "plan owner or stream is null");
  }
  if (plan->owner->restoration_failed.load(std::memory_order_acquire) ||
      !same_context(plan->owner, stream->owner)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
        "plan and stream context owners differ or are poisoned");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!same_context(plan->owner, spans[index].buffer->owner)) {
      return validation_error(
          error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
          "plan and device spans belong to different context owners");
    }
    for (size_t other = index + 1; other < count; ++other) {
      if (spans_overlap(spans[index], spans[other])) {
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
            "input, weight, and output spans must not overlap");
      }
    }
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

class ExclusiveFixed37GemmUses final {
 public:
  ExclusiveFixed37GemmUses(RustInferCudaFixed37GemmPlan* plan,
                           RustInferCudaStream* stream) noexcept
      : plan_(plan),
        stream_(stream),
        buffers_{},
        buffer_count_(0),
        acquired_buffers_(0),
        plan_acquired_(false),
        stream_acquired_(false),
        command_batch_(false) {}

  ExclusiveFixed37GemmUses(const ExclusiveFixed37GemmUses&) = delete;
  ExclusiveFixed37GemmUses& operator=(const ExclusiveFixed37GemmUses&) =
      delete;

  bool add(RustInferCudaDeviceBuffer* buffer) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (buffers_[index] == buffer) {
        return true;
      }
    }
    if (buffer == nullptr || buffer_count_ == kMaximumGemmBuffers) {
      return false;
    }
    buffers_[buffer_count_++] = buffer;
    return true;
  }

  RustInferCudaStatus acquire(RustInferCudaErrorInfo* error) noexcept {
    if (command_batch_is_active(stream_)) {
      if (!command_batch_is_owned_by_current_thread(stream_)) {
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
            "an active stream command batch is owned by another thread");
      }
      command_batch_ = true;
      RustInferCudaStatus status = command_batch_register_use(
          stream_, &plan_->active_uses, error, kExecuteOperation,
          "the fixed37 GEMM plan already has an active use");
      if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
        return status;
      }
      for (size_t index = 0; index < buffer_count_; ++index) {
        status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error, kExecuteOperation,
            "a fixed37 GEMM device buffer already has an active use");
        if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
          return status;
        }
      }
      return RUSTINFER_CUDA_STATUS_SUCCESS;
    }
    if (!try_acquire_exclusive_use(plan_->active_uses)) {
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "the fixed37 GEMM plan already has an active use");
    }
    plan_acquired_ = true;
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        if (!release_acquired()) {
          return internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                kExecuteOperation,
                                "exclusive-use rollback was corrupted");
        }
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
            "a fixed37 GEMM device buffer already has an active use");
      }
      ++acquired_buffers_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      if (!release_acquired()) {
        return internal_error(error,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "exclusive-use rollback was corrupted");
      }
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "the stream already has an active use");
    }
    stream_acquired_ = true;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  bool release_completed() noexcept { return release_acquired(); }
  bool command_batch() const noexcept { return command_batch_; }

 private:
  bool release_acquired() noexcept {
    if (command_batch_) {
      return true;
    }
    bool valid = true;
    if (stream_acquired_) {
      valid = release_exclusive_use(stream_->active_uses) && valid;
      stream_acquired_ = false;
    }
    while (acquired_buffers_ != 0) {
      const size_t acquired_index = acquired_buffers_ - 1;
      acquired_buffers_ = acquired_index;
      valid = release_exclusive_use(buffers_[acquired_index]->active_uses) &&
              valid;
    }
    if (plan_acquired_) {
      valid = release_exclusive_use(plan_->active_uses) && valid;
      plan_acquired_ = false;
    }
    return valid;
  }

  RustInferCudaFixed37GemmPlan* plan_;
  RustInferCudaStream* stream_;
  RustInferCudaDeviceBuffer* buffers_[kMaximumGemmBuffers];
  size_t buffer_count_;
  size_t acquired_buffers_;
  bool plan_acquired_;
  bool stream_acquired_;
  bool command_batch_;
};

__global__ __launch_bounds__(rustinfer_cuda_fixed37::kThreadsPerBlock)
void fixed37_gemm_kernel(const __nv_bfloat16* input,
                         const __nv_bfloat16* weight,
                         __nv_bfloat16* output, uint64_t m, uint64_t n,
                         uint64_t k, uint64_t output_elements,
                         uint64_t partial_count) {
  extern __shared__ float shared_partials[];
  float* first = shared_partials;
  float* second = shared_partials + partial_count;
  for (uint64_t output_index = blockIdx.x; output_index < output_elements;
       output_index += gridDim.x) {
    const uint64_t row = output_index / n;
    const uint64_t column = output_index % n;
    const uint64_t input_base = row * k;
    const uint64_t weight_base = column * k;
    for (uint64_t chunk = threadIdx.x; chunk < partial_count;
         chunk += blockDim.x) {
      const uint64_t begin = chunk * rustinfer_cuda_fixed37::kChunkElements;
      uint64_t end = begin + rustinfer_cuda_fixed37::kChunkElements;
      if (end > k) {
        end = k;
      }
      float accumulator = 0.0F;
      for (uint64_t depth = begin; depth < end; ++depth) {
        accumulator = fmaf(__bfloat162float(input[input_base + depth]),
                           __bfloat162float(weight[weight_base + depth]),
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
  (void)m;
}

RustInferCudaStatus complete_execution(
    ExclusiveFixed37GemmUses* uses, CurrentContext* scope,
    RustInferCudaStream* stream, RustInferCudaStatus operation_status,
    bool launch_attempted, RustInferCudaErrorInfo* error) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                        kExecuteOperation);
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
                             kExecuteOperation);
    }
  }
  status = scope->leave(status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                        kExecuteOperation);
  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (completion_confirmed && restoration_confirmed &&
      !uses->release_completed()) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kExecuteOperation,
                          "exclusive-use accounting was corrupted");
  }
  return status;
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_create(
    RustInferCudaContext* context, const RustInferCudaGemmConfig* config,
    RustInferCudaFixed37GemmPlan** out_plan,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation, "out_plan is null");
  }
  *out_plan = nullptr;
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation, "context is null");
  }
  if (context->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                            kCreateOperation,
                            "context is poisoned by a prior restoration failure");
  }

  GemmByteLengths lengths{};
  uint64_t shared_bytes = 0;
  RustInferCudaStatus status =
      validate_config(config, &lengths, &shared_bytes, error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  RustInferCudaGemmConfig normalized_config = *config;
  normalized_config.struct_size = sizeof(normalized_config);
  void* storage = std::calloc(1, sizeof(RustInferCudaFixed37GemmPlan));
  if (storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE, kCreateOperation,
                     "host plan allocation failed");
  }
  auto* plan = new (storage) RustInferCudaFixed37GemmPlan(
      context, normalized_config, lengths, shared_bytes);
  if (!retain_child(context)) {
    plan->~RustInferCudaFixed37GemmPlan();
    std::free(plan);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          kCreateOperation,
                          "context child-resource counter overflow");
  }
  *out_plan = plan;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_info(
    RustInferCudaFixed37GemmPlan* plan,
    RustInferCudaFixed37GemmPlanInfo* out_info,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kQueryOperation,
        "plan or out_info is null, or struct_size is incompatible");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  if (!try_acquire_exclusive_use(plan->active_uses)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                            kQueryOperation,
                            "the fixed37 GEMM plan already has an active use");
  }
  *out_info = plan->info;
  if (!release_exclusive_use(plan->active_uses)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                          kQueryOperation,
                          "plan use accounting was corrupted");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_execute(
    RustInferCudaFixed37GemmPlan* plan,
    const RustInferCudaBufferSpan* input,
    const RustInferCudaBufferSpan* weight,
    const RustInferCudaBufferSpan* output, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation, "the plan is null");
  }

  ResolvedSpan spans[kMaximumGemmBuffers]{};
  RustInferCudaStatus status = resolve_exact_span(
      input, RUSTINFER_CUDA_DTYPE_BF16, plan->input_bytes, &spans[0], error,
      "input span dtype must be BF16");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(weight, RUSTINFER_CUDA_DTYPE_BF16,
                                plan->weight_bytes, &spans[1], error,
                                "weight span dtype must be BF16");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(output, RUSTINFER_CUDA_DTYPE_BF16,
                                plan->output_bytes, &spans[2], error,
                                "output span dtype must be BF16");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_span_relationships(plan, stream, spans,
                                       kMaximumGemmBuffers, error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveFixed37GemmUses uses(plan, stream);
  for (const ResolvedSpan& span : spans) {
    if (!uses.add(span.buffer)) {
      return internal_error(error,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "too many unique fixed37 GEMM device buffers");
    }
  }
  status = uses.acquire(error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  CurrentContext scope(plan->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                       kExecuteOperation);
  bool launch_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = runtime_error(cudaGetLastError(), error,
                           RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                           kExecuteOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const uint64_t output_elements = plan->config.m * plan->config.n;
    const uint64_t partial_count =
        rustinfer_cuda_fixed37::chunk_count(plan->config.k);
    launch_attempted = true;
    fixed37_gemm_kernel<<<
        rustinfer_cuda_fixed37::block_count(output_elements),
        rustinfer_cuda_fixed37::kThreadsPerBlock,
        static_cast<size_t>(plan->info.dynamic_shared_memory_bytes),
        stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),
                          static_cast<const __nv_bfloat16*>(spans[1].data),
                          static_cast<__nv_bfloat16*>(spans[2].data),
                          plan->config.m, plan->config.n, plan->config.k,
                          output_elements, partial_count);
    status = runtime_error(cudaGetLastError(), error,
                           RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                           kExecuteOperation);
  }
  return complete_execution(&uses, &scope, stream, status,
                            launch_attempted, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_fixed37_gemm_plan_close(
    RustInferCudaFixed37GemmPlan** plan,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseOperation, "plan pointer is null");
  }
  if (*plan == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  RustInferCudaFixed37GemmPlan* value = *plan;
  if (value->owner == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            kCloseOperation,
                            "plan context owner is null");
  }
  if (!try_acquire_exclusive_use(value->active_uses)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_CLOSE, kCloseOperation,
        "the fixed37 GEMM plan has an active or permanent use guard");
  }
  RustInferCudaContext* owner = value->owner;
  value->~RustInferCudaFixed37GemmPlan();
  std::free(value);
  *plan = nullptr;
  if (!release_child(owner)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                          kCloseOperation,
                          "context child-resource counter underflow");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}
