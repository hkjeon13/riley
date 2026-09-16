#include "ffi_internal.hpp"

#if defined(RILEY_CUDA_ENABLE_CUBLAS_GEMM_PROBE)

#include <cublas_v2.h>

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
constexpr const char* kCreateOperation = "create direct cuBLAS GEMM probe plan";
constexpr const char* kQueryOperation = "query direct cuBLAS GEMM probe plan";
constexpr const char* kExecuteOperation = "execute direct cuBLAS GEMM probe";
constexpr const char* kCloseOperation = "close direct cuBLAS GEMM probe plan";

struct GemmByteLengths {
  uint64_t input;
  uint64_t weight;
  uint64_t output;
};

struct ResolvedSpan {
  RileyCudaDeviceBuffer* buffer;
  void* data;
  uint64_t byte_offset;
  uint64_t byte_len;
};

}  // namespace

struct RileyCudaCublasGemmProbePlan {
  RileyCudaCublasGemmProbePlan(RileyCudaContext* owning_context,
                                const RileyCudaGemmConfig& plan_config,
                                const GemmByteLengths& lengths) noexcept
      : owner(owning_context),
        config(plan_config),
        info{},
        handle(nullptr),
        input_bytes(lengths.input),
        weight_bytes(lengths.weight),
        output_bytes(lengths.output),
        active_uses(0) {
    info.struct_size = sizeof(info);
    info.backend = RILEY_CUDA_CUBLAS_GEMM_PROBE_BACKEND_CUBLAS;
    info.requested_math_mode = static_cast<int32_t>(CUBLAS_DEFAULT_MATH);
    info.requested_pointer_mode =
        static_cast<int32_t>(CUBLAS_POINTER_MODE_HOST);
    info.requested_atomics_mode =
        static_cast<int32_t>(CUBLAS_ATOMICS_NOT_ALLOWED);
    info.m = config.m;
    info.n = config.n;
    info.k = config.k;
    info.workspace_bytes = 0;
  }

  RileyCudaContext* owner;
  RileyCudaGemmConfig config;
  RileyCudaCublasGemmProbeInfo info;
  cublasHandle_t handle;
  uint64_t input_bytes;
  uint64_t weight_bytes;
  uint64_t output_bytes;
  std::atomic<uint32_t> active_uses;
};

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::driver_error;
using riley_cuda_internal::internal_error;
using riley_cuda_internal::release_child;
using riley_cuda_internal::release_exclusive_use;
using riley_cuda_internal::retain_child;
using riley_cuda_internal::runtime_error;
using riley_cuda_internal::same_context;
using riley_cuda_internal::set_error;
using riley_cuda_internal::try_acquire_exclusive_use;
using riley_cuda_internal::validation_error;

static_assert(sizeof(RileyCudaCublasGemmProbeInfo) == 96,
              "direct cuBLAS probe info ABI size changed");
static_assert(offsetof(RileyCudaCublasGemmProbeInfo, requested_math_mode) ==
                  8,
              "direct cuBLAS probe info math-mode offset changed");
static_assert(offsetof(RileyCudaCublasGemmProbeInfo, runtime_version) == 32,
              "direct cuBLAS probe info version offset changed");
static_assert(offsetof(RileyCudaCublasGemmProbeInfo, m) == 48,
              "direct cuBLAS probe info dimension offset changed");
static_assert(offsetof(RileyCudaCublasGemmProbeInfo, reserved) == 80,
              "direct cuBLAS probe info reserved offset changed");

const char* cublas_status_detail(cublasStatus_t status) noexcept {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBLAS was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBLAS allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBLAS received an invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "the CUDA device architecture is unsupported";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "cuBLAS could not map a resource";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBLAS execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBLAS reported an internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "the requested cuBLAS operation is unsupported";
    default:
      return "cuBLAS reported an unknown error";
  }
}

RileyCudaStatus cublas_probe_error(cublasStatus_t result,
                                    RileyCudaErrorInfo* error,
                                    uint32_t stage,
                                    const char* operation) noexcept {
  if (result == CUBLAS_STATUS_SUCCESS) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaStatus status = RILEY_CUDA_STATUS_CUBLASLT_ERROR;
  if (result == CUBLAS_STATUS_INVALID_VALUE) {
    status = RILEY_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == CUBLAS_STATUS_ALLOC_FAILED) {
    status = RILEY_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == CUBLAS_STATUS_ARCH_MISMATCH ||
             result == CUBLAS_STATUS_NOT_SUPPORTED) {
    status = RILEY_CUDA_STATUS_NOT_SUPPORTED;
  }
  // The public ABI has a single pre-existing math-library error surface. This
  // test-only lane records the original cublasStatus_t in native_code there.
  return set_error(error, status, static_cast<int32_t>(result),
                   RILEY_CUDA_ERROR_DOMAIN_CUBLASLT, stage, operation,
                   cublas_status_detail(result));
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

RileyCudaStatus matrix_bytes(uint64_t rows, uint64_t columns,
                             uint64_t* output,
                             RileyCudaErrorInfo* error) noexcept {
  uint64_t elements = 0;
  if (!checked_multiply(rows, columns, &elements) ||
      !checked_multiply(elements, kBfloat16Bytes, output)) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "matrix byte length overflows uint64_t");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus validate_config(const RileyCudaGemmConfig* config,
                                GemmByteLengths* lengths,
                                RileyCudaErrorInfo* error) noexcept {
  if (config == nullptr || lengths == nullptr ||
      config->struct_size < sizeof(*config)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "config is null or has an incompatible struct_size");
  }
  // The direct cuBLAS probe intentionally has no split-K extension path.
  if (config->flags != 0 || config->reserved0 != 0 ||
      !reserved_is_zero(config->reserved, 3)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "flags and reserved fields must be zero");
  }
  if (config->m == 0 || config->n == 0 || config->k == 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "M, N, and K must all be non-zero");
  }
  const uint64_t maximum_dimension =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (config->m > maximum_dimension || config->n > maximum_dimension ||
      config->k > maximum_dimension) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation,
                            "M, N, and K must fit signed 32-bit dimensions");
  }
  if (config->input_dtype != RILEY_CUDA_DTYPE_BF16 ||
      config->weight_dtype != RILEY_CUDA_DTYPE_BF16 ||
      config->accumulator_dtype != RILEY_CUDA_DTYPE_F32 ||
      config->output_dtype != RILEY_CUDA_DTYPE_BF16) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "only BF16 input/weight/output with F32 accumulation is supported");
  }
  if (config->input_transpose != RILEY_CUDA_GEMM_TRANSPOSE_N ||
      config->weight_transpose != RILEY_CUDA_GEMM_TRANSPOSE_T ||
      config->input_layout != RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->weight_layout != RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->output_layout != RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR ||
      config->epilogue != RILEY_CUDA_GEMM_EPILOGUE_NONE ||
      config->deterministic != RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED) {
    return validation_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kCreateOperation,
        "only row-major X=N/W=T, epilogue-none, deterministic GEMM is supported");
  }

  RileyCudaStatus status =
      matrix_bytes(config->m, config->k, &lengths->input, error);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = matrix_bytes(config->n, config->k, &lengths->weight, error);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = matrix_bytes(config->m, config->n, &lengths->output, error);
  }
  return status;
}

RileyCudaStatus resolve_exact_span(
    const RileyCudaBufferSpan* span, RileyCudaDType required_dtype,
    uint64_t required_bytes, ResolvedSpan* output,
    RileyCudaErrorInfo* error, const char* dtype_detail) noexcept {
  if (span == nullptr || output == nullptr ||
      span->struct_size < sizeof(*span)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a span is null or has an incompatible struct_size");
  }
  if (!reserved_is_zero(span->reserved, 2)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "span reserved fields must be zero");
  }
  if (span->dtype != required_dtype) {
    return set_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT, 0,
                     RILEY_CUDA_ERROR_DOMAIN_VALIDATION,
                     RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
                     dtype_detail);
  }
  if (span->buffer == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation, "a span buffer handle is null");
  }
  if (span->byte_offset % kRequiredAlignment != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "every span byte_offset must be 256-byte aligned");
  }
  if (span->byte_len != required_bytes) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
        "a span byte_len does not exactly match the prepared requirement");
  }
  if (span->byte_offset > span->buffer->byte_len ||
      span->byte_len > span->buffer->byte_len - span->byte_offset) {
    return validation_error(error, RILEY_CUDA_STATUS_OUT_OF_RANGE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a declared span exceeds its opaque allocation");
  }
  if (required_bytes != 0 && span->buffer->device_data == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "a non-empty span refers to a zero-byte allocation");
  }
  void* data = nullptr;
  if (span->buffer->device_data != nullptr) {
    data = static_cast<void*>(
        static_cast<uint8_t*>(span->buffer->device_data) +
        static_cast<size_t>(span->byte_offset));
  }
  *output = ResolvedSpan{span->buffer, data, span->byte_offset,
                         required_bytes};
  return RILEY_CUDA_STATUS_SUCCESS;
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

RileyCudaStatus validate_span_relationships(
    RileyCudaCublasGemmProbePlan* plan, RileyCudaStream* stream,
    const ResolvedSpan* spans, size_t count,
    RileyCudaErrorInfo* error) noexcept {
  if (plan->owner == nullptr || stream == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation, "plan owner or stream is null");
  }
  if (plan->owner->restoration_failed.load(std::memory_order_acquire) ||
      !same_context(plan->owner, stream->owner)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
        "plan and stream context owners differ or are poisoned");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!same_context(plan->owner, spans[index].buffer->owner)) {
      return validation_error(
          error, RILEY_CUDA_STATUS_INVALID_STATE,
          RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
          "plan and device spans belong to different context owners");
    }
    for (size_t other = index + 1; other < count; ++other) {
      if (spans_overlap(spans[index], spans[other])) {
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
            RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
            "input, weight, and output spans must not overlap");
      }
    }
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

class ExclusiveProbeUses final {
 public:
  ExclusiveProbeUses(RileyCudaCublasGemmProbePlan* plan,
                     RileyCudaStream* stream) noexcept
      : plan_(plan),
        stream_(stream),
        buffers_{},
        buffer_count_(0),
        acquired_buffers_(0),
        plan_acquired_(false),
        stream_acquired_(false) {}

  ExclusiveProbeUses(const ExclusiveProbeUses&) = delete;
  ExclusiveProbeUses& operator=(const ExclusiveProbeUses&) = delete;

  bool add(RileyCudaDeviceBuffer* buffer) noexcept {
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

  RileyCudaStatus acquire(RileyCudaErrorInfo* error) noexcept {
    // This test-only path synchronizes every execution, so it cannot register
    // an asynchronous lifetime in a command-batch ledger safely.
    if (command_batch_is_active(stream_)) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "direct cuBLAS probe rejects active command batches");
    }
    if (!try_acquire_exclusive_use(plan_->active_uses)) {
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "the direct cuBLAS probe plan already has an active use");
    }
    plan_acquired_ = true;
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        if (!release_acquired()) {
          return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kExecuteOperation,
                                "exclusive-use rollback was corrupted");
        }
        return validation_error(
            error, RILEY_CUDA_STATUS_INVALID_STATE,
            RILEY_CUDA_ERROR_STAGE_VALIDATION, kExecuteOperation,
            "a direct cuBLAS probe device buffer already has an active use");
      }
      ++acquired_buffers_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      if (!release_acquired()) {
        return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "exclusive-use rollback was corrupted");
      }
      return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                              RILEY_CUDA_ERROR_STAGE_VALIDATION,
                              kExecuteOperation,
                              "the stream already has an active use");
    }
    stream_acquired_ = true;
    return RILEY_CUDA_STATUS_SUCCESS;
  }

  bool release_completed() noexcept { return release_acquired(); }

 private:
  bool release_acquired() noexcept {
    bool valid = true;
    if (stream_acquired_) {
      valid = release_exclusive_use(stream_->active_uses) && valid;
      stream_acquired_ = false;
    }
    while (acquired_buffers_ != 0) {
      --acquired_buffers_;
      valid =
          release_exclusive_use(buffers_[acquired_buffers_]->active_uses) &&
          valid;
    }
    if (plan_acquired_) {
      valid = release_exclusive_use(plan_->active_uses) && valid;
      plan_acquired_ = false;
    }
    return valid;
  }

  RileyCudaCublasGemmProbePlan* plan_;
  RileyCudaStream* stream_;
  RileyCudaDeviceBuffer* buffers_[kMaximumGemmBuffers];
  size_t buffer_count_;
  size_t acquired_buffers_;
  bool plan_acquired_;
  bool stream_acquired_;
};

RileyCudaStatus verify_configured_modes(
    RileyCudaCublasGemmProbePlan* plan,
    RileyCudaErrorInfo* error) noexcept {
  RileyCudaStatus status = cublas_probe_error(
      cublasSetMathMode(plan->handle, CUBLAS_DEFAULT_MATH), error,
      RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = cublas_probe_error(
      cublasSetPointerMode(plan->handle, CUBLAS_POINTER_MODE_HOST), error,
      RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = cublas_probe_error(
      cublasSetAtomicsMode(plan->handle, CUBLAS_ATOMICS_NOT_ALLOWED), error,
      RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  cublasMath_t actual_math = CUBLAS_DEFAULT_MATH;
  cublasPointerMode_t actual_pointer = CUBLAS_POINTER_MODE_HOST;
  cublasAtomicsMode_t actual_atomics = CUBLAS_ATOMICS_NOT_ALLOWED;
  status = cublas_probe_error(cublasGetMathMode(plan->handle, &actual_math),
                              error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                              kCreateOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = cublas_probe_error(
        cublasGetPointerMode(plan->handle, &actual_pointer), error,
        RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = cublas_probe_error(
        cublasGetAtomicsMode(plan->handle, &actual_atomics), error,
        RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  plan->info.actual_math_mode = static_cast<int32_t>(actual_math);
  plan->info.actual_pointer_mode = static_cast<int32_t>(actual_pointer);
  plan->info.actual_atomics_mode = static_cast<int32_t>(actual_atomics);
  if (plan->info.actual_math_mode != plan->info.requested_math_mode ||
      plan->info.actual_pointer_mode != plan->info.requested_pointer_mode ||
      plan->info.actual_atomics_mode != plan->info.requested_atomics_mode) {
    return set_error(
        error, RILEY_CUDA_STATUS_NOT_SUPPORTED, 0,
        RILEY_CUDA_ERROR_DOMAIN_CUBLASLT, RILEY_CUDA_ERROR_STAGE_PREPARE,
        kCreateOperation,
        "cuBLAS did not retain the requested math, pointer, and atomics modes");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus record_environment(RileyCudaCublasGemmProbePlan* plan,
                                   RileyCudaErrorInfo* error) noexcept {
  int capability_major = 0;
  CUresult driver_result = cuDeviceGetAttribute(
      &capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                        kCreateOperation);
  }
  int capability_minor = 0;
  driver_result = cuDeviceGetAttribute(
      &capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                        kCreateOperation);
  }
  if (capability_major < 0 || capability_minor < 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                          kCreateOperation,
                          "CUDA returned a negative compute capability");
  }

  int runtime_version = 0;
  const cudaError_t runtime_result = cudaRuntimeGetVersion(&runtime_version);
  if (runtime_result != cudaSuccess) {
    return runtime_error(runtime_result, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                         kCreateOperation);
  }
  int cublas_version = 0;
  RileyCudaStatus status = cublas_probe_error(
      cublasGetVersion(plan->handle, &cublas_version), error,
      RILEY_CUDA_ERROR_STAGE_PREPARE, kCreateOperation);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (runtime_version < 0 || cublas_version < 0) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                          kCreateOperation,
                          "CUDA returned a negative runtime or cuBLAS version");
  }

  plan->info.compute_capability_major =
      static_cast<uint32_t>(capability_major);
  plan->info.compute_capability_minor =
      static_cast<uint32_t>(capability_minor);
  plan->info.runtime_version = runtime_version;
  plan->info.cublas_version = cublas_version;
  return RILEY_CUDA_STATUS_SUCCESS;
}

RileyCudaStatus prepare_plan(RileyCudaCublasGemmProbePlan* plan,
                             RileyCudaErrorInfo* error) noexcept {
  RileyCudaStatus status = cublas_probe_error(
      cublasCreate(&plan->handle), error, RILEY_CUDA_ERROR_STAGE_PREPARE,
      kCreateOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = verify_configured_modes(plan, error);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = record_environment(plan, error);
  }
  return status;
}

RileyCudaStatus destroy_plan_resources(RileyCudaCublasGemmProbePlan* plan,
                                       RileyCudaErrorInfo* error,
                                       uint32_t stage,
                                       const char* operation) noexcept {
  if (plan->handle != nullptr) {
    const cublasStatus_t result = cublasDestroy(plan->handle);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublas_probe_error(result, error, stage, operation);
    }
    plan->handle = nullptr;
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

bool plan_resources_destroyed(
    const RileyCudaCublasGemmProbePlan* plan) noexcept {
  return plan != nullptr && plan->handle == nullptr;
}

RileyCudaStatus complete_execution(
    ExclusiveProbeUses* uses, CurrentContext* scope, RileyCudaStream* stream,
    RileyCudaStatus operation_status, bool launch_attempted,
    RileyCudaErrorInfo* error) noexcept {
  bool completion_confirmed = !launch_attempted;
  RileyCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t synchronize_result = cudaStreamSynchronize(stream->stream);
    completion_confirmed = synchronize_result == cudaSuccess;
    if (!completion_confirmed) {
      status = runtime_error(synchronize_result, error,
                             RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                             kExecuteOperation);
    }
  }
  status = scope->leave(status, error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                        kExecuteOperation);
  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (completion_confirmed && restoration_confirmed &&
      !uses->release_completed()) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE,
                          kExecuteOperation,
                          "exclusive-use accounting was corrupted");
  }
  return status;
}

}  // namespace

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_create(
    RileyCudaContext* context, const RileyCudaGemmConfig* config,
    RileyCudaCublasGemmProbePlan** out_plan,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_plan == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation, "out_plan is null");
  }
  *out_plan = nullptr;
  if (context == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCreateOperation, "context is null");
  }
  if (context->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_PREPARE,
                            kCreateOperation,
                            "context is poisoned by a prior restoration failure");
  }

  GemmByteLengths lengths{};
  RileyCudaStatus status = validate_config(config, &lengths, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  RileyCudaGemmConfig normalized_config = *config;
  normalized_config.struct_size = sizeof(normalized_config);
  void* storage = std::calloc(1, sizeof(RileyCudaCublasGemmProbePlan));
  if (storage == nullptr) {
    return set_error(error, RILEY_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RILEY_CUDA_ERROR_DOMAIN_INTERNAL,
                     RILEY_CUDA_ERROR_STAGE_CREATE, kCreateOperation,
                     "host plan allocation failed");
  }
  auto* plan = new (storage)
      RileyCudaCublasGemmProbePlan(context, normalized_config, lengths);
  if (!retain_child(context)) {
    plan->~RileyCudaCublasGemmProbePlan();
    std::free(plan);
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_CREATE,
                          kCreateOperation,
                          "context child-resource counter overflow");
  }

  CurrentContext scope(context);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kCreateOperation);
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = prepare_plan(plan, error);
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    // Cleanup runs before leave while this context is still current. Any
    // uncertain native destruction retains the plan and child lease below.
    (void)destroy_plan_resources(plan, nullptr,
                                 RILEY_CUDA_ERROR_STAGE_PREPARE,
                                 "cleanup failed direct cuBLAS GEMM probe plan");
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                       kCreateOperation);

  const bool restoration_confirmed =
      !context->restoration_failed.load(std::memory_order_acquire);
  if (status == RILEY_CUDA_STATUS_SUCCESS && plan->handle != nullptr &&
      restoration_confirmed) {
    *out_plan = plan;
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (plan_resources_destroyed(plan) && restoration_confirmed) {
    if (!release_child(context)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_PREPARE,
                            "cleanup failed direct cuBLAS GEMM probe plan",
                            "context child-resource counter underflow");
    }
    plan->~RileyCudaCublasGemmProbePlan();
    std::free(plan);
  }
  // Ambiguous native destruction or context restoration deliberately retains
  // the unreachable wrapper and context-child lease fail closed.
  return status;
}

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_info(
    RileyCudaCublasGemmProbePlan* plan,
    RileyCudaCublasGemmProbeInfo* out_info,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION, kQueryOperation,
        "plan or out_info is null, or struct_size is incompatible");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  if (!try_acquire_exclusive_use(plan->active_uses)) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_QUERY, kQueryOperation,
                            "the direct cuBLAS probe plan already has an active use");
  }
  *out_info = plan->info;
  if (!release_exclusive_use(plan->active_uses)) {
    return internal_error(error, RILEY_CUDA_ERROR_STAGE_QUERY,
                          kQueryOperation,
                          "plan use accounting was corrupted");
  }
  return RILEY_CUDA_STATUS_SUCCESS;
}

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_execute(
    RileyCudaCublasGemmProbePlan* plan, const RileyCudaBufferSpan* input,
    const RileyCudaBufferSpan* weight, const RileyCudaBufferSpan* output,
    RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation, "the plan is null");
  }
  if (plan->handle == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "the direct cuBLAS probe plan is not prepared");
  }

  ResolvedSpan spans[kMaximumGemmBuffers]{};
  RileyCudaStatus status = resolve_exact_span(
      input, RILEY_CUDA_DTYPE_BF16, plan->input_bytes, &spans[0], error,
      "input span dtype must be BF16");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(weight, RILEY_CUDA_DTYPE_BF16,
                                plan->weight_bytes, &spans[1], error,
                                "weight span dtype must be BF16");
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(output, RILEY_CUDA_DTYPE_BF16,
                                plan->output_bytes, &spans[2], error,
                                "output span dtype must be BF16");
  }
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_span_relationships(plan, stream, spans,
                                       kMaximumGemmBuffers, error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveProbeUses uses(plan, stream);
  for (const ResolvedSpan& span : spans) {
    if (!uses.add(span.buffer)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "too many unique direct cuBLAS probe device buffers");
    }
  }
  status = uses.acquire(error);
  if (status != RILEY_CUDA_STATUS_SUCCESS) {
    return status;
  }

  CurrentContext scope(plan->owner);
  status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                       kExecuteOperation);
  bool launch_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = cublas_probe_error(cublasSetStream(plan->handle, stream->stream),
                                error, RILEY_CUDA_ERROR_STAGE_LAUNCH,
                                kExecuteOperation);
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    // Row-major X[M,K], W[N,K], Y[M,N] are physical column-major
    // Xc[K,M], Wc[K,N], Yc[N,M]. Therefore Yc = Wc^T * Xc.
    launch_attempted = true;
    status = cublas_probe_error(
        cublasGemmEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     static_cast<int>(plan->config.n),
                     static_cast<int>(plan->config.m),
                     static_cast<int>(plan->config.k), &alpha, spans[1].data,
                     CUDA_R_16BF, static_cast<int>(plan->config.k),
                     spans[0].data, CUDA_R_16BF,
                     static_cast<int>(plan->config.k), &beta, spans[2].data,
                     CUDA_R_16BF, static_cast<int>(plan->config.n),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kExecuteOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error);
}

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_close(
    RileyCudaCublasGemmProbePlan** plan,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kCloseOperation, "plan pointer is null");
  }
  if (*plan == nullptr) {
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  RileyCudaCublasGemmProbePlan* value = *plan;
  if (value->owner == nullptr) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                            RILEY_CUDA_ERROR_STAGE_CLOSE, kCloseOperation,
                            "plan context owner is null");
  }
  if (!try_acquire_exclusive_use(value->active_uses)) {
    return validation_error(
        error, RILEY_CUDA_STATUS_INVALID_STATE, RILEY_CUDA_ERROR_STAGE_CLOSE,
        kCloseOperation,
        "the direct cuBLAS probe plan has an active or permanent use guard");
  }

  CurrentContext scope(value->owner);
  RileyCudaStatus status = scope.enter(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                        kCloseOperation);
  bool destruction_attempted = false;
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    destruction_attempted = true;
    status = destroy_plan_resources(value, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                                    kCloseOperation);
  }
  status = scope.leave(status, error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                       kCloseOperation);

  const bool restoration_confirmed =
      !value->owner->restoration_failed.load(std::memory_order_acquire);
  if (status == RILEY_CUDA_STATUS_SUCCESS && plan_resources_destroyed(value) &&
      restoration_confirmed) {
    RileyCudaContext* owner = value->owner;
    value->~RileyCudaCublasGemmProbePlan();
    std::free(value);
    *plan = nullptr;
    if (!release_child(owner)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseOperation,
                            "context child-resource counter underflow");
    }
    return RILEY_CUDA_STATUS_SUCCESS;
  }
  if (!destruction_attempted && restoration_confirmed) {
    if (!release_exclusive_use(value->active_uses)) {
      return internal_error(error, RILEY_CUDA_ERROR_STAGE_CLOSE,
                            kCloseOperation,
                            "plan use accounting was corrupted");
    }
  }
  // A destruction attempt or ambiguous context restoration leaves the plan's
  // exclusive-use guard set forever, preserving its context-child lease.
  return status;
}

#endif  // RILEY_CUDA_ENABLE_CUBLAS_GEMM_PROBE
