#include "ffi_internal.hpp"

#if defined(RILEY_CUDA_ENABLE_CUBLAS_GEMM_PROBE)

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <mma.h>
#include "hf_softmax_diagnostic.cuh"
#include "hf_regular_f32_softmax.cuh"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <initializer_list>
#include <new>

namespace {

bool hf_wide_extended_kernel_arch_qualified() noexcept;

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
        active_uses(0),
        owned_command_batches(false) {
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
  bool owned_command_batches;
  // Pin the independent-M1 topology to the qualified RTX4090 environment.
  bool owned_weight_reuse_qualified = false;
  bool owned_sm80_gateup_rows_qualified = false;
  bool owned_sm80_attention_enabled = false;
  bool owned_sm80_paged_attention_enabled = false;
  bool owned_sm80_decode_mlp_v8_enabled = false;
  bool owned_sm80_full_projection_rows = false;
  bool owned_smol_attention_qualified = false;
  bool owned_attention_extended_qualified = false;
  cublasLtHandle_t decode_lt = nullptr;
  cublasLtMatmulDesc_t decode_op = nullptr;
  cublasLtMatmulPreference_t decode_preference = nullptr;
  cublasLtMatrixLayout_t decode_weight = nullptr;
  cublasLtMatrixLayout_t decode_batch_weight[33]{};
  cublasLtMatrixLayout_t decode_input[33]{};
  cublasLtMatrixLayout_t decode_output[33]{};
  cublasLtMatmulAlgo_t decode_algo{};
  // Cold-only bounded prefill recipes; no descriptors or scratch in dispatch.
  cublasLtHandle_t prefill_lt = nullptr;
  cublasLtMatmulDesc_t prefill_op = nullptr;
  cublasLtMatrixLayout_t prefill_weight = nullptr;
  cublasLtMatrixLayout_t prefill_input[577]{};
  cublasLtMatrixLayout_t prefill_output[577]{};
  cublasLtMatmulAlgo_t prefill_algo{};
  RileyCudaDeviceBuffer* prefill_workspace = nullptr;
  uint64_t prefill_workspace_bytes = 0;
  uint32_t prefill_max_rows = 0;
  bool decode_ready = false;
  uint32_t decode_span_alignment = 0;
  uint32_t sm80_projection_rows = 0;
  RileyCudaDeviceBuffer* sm80_full_projection_workspace = nullptr;
  uint64_t sm80_full_projection_workspace_bytes = 0;
  RileyCudaDeviceBuffer* sm80_projection_workspace = nullptr;
  uint64_t sm80_projection_workspace_bytes = 0;
};

namespace {

using riley_cuda_internal::CurrentContext;
using riley_cuda_internal::clear_error;
using riley_cuda_internal::command_batch_is_active;
using riley_cuda_internal::command_batch_is_owned_by_current_thread;
using riley_cuda_internal::command_batch_register_use;
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

RileyCudaStatus resolve_attention_span(
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
  if (span->byte_offset % (required_dtype == RILEY_CUDA_DTYPE_BF16 ? 2 : required_dtype == RILEY_CUDA_DTYPE_U8 ? 1 : 4) != 0) {
    return validation_error(error, RILEY_CUDA_STATUS_INVALID_ARGUMENT,
                            RILEY_CUDA_ERROR_STAGE_VALIDATION,
                            kExecuteOperation,
                            "every span byte_offset must be element aligned");
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

// The attention owner may reuse completed BF16 QK rows for probabilities.
// Only this exact role/range alias is safe: each tile is fully read into F32
// scratch before its disjoint BF16 rows are overwritten on the same stream.
constexpr bool hf_score_probability_alias(bool enabled, size_t left, size_t right,
                                         uint64_t lo, uint64_t ro,
                                         uint64_t ll, uint64_t rl) noexcept {
  return enabled && left == 8 && right == 11 && lo == ro && ll == rl;
}
static_assert(hf_score_probability_alias(true,8,11,0,0,64,64));
static_assert(!hf_score_probability_alias(false,8,11,0,0,64,64));
static_assert(!hf_score_probability_alias(true,8,11,0,2,64,64));
static_assert(!hf_score_probability_alias(true,8,11,0,0,64,62));
static_assert(!hf_score_probability_alias(true,5,12,0,0,64,64));

RileyCudaStatus validate_span_relationships(
    RileyCudaCublasGemmProbePlan* plan, RileyCudaStream* stream,
    const ResolvedSpan* spans, size_t count,
    RileyCudaErrorInfo* error, bool allow_hf_score_alias = false) noexcept {
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
      if (spans_overlap(spans[index], spans[other]) &&
          !hf_score_probability_alias(allow_hf_score_alias,index,other,
              spans[index].byte_offset,spans[other].byte_offset,
              spans[index].byte_len,spans[other].byte_len)) {
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
        stream_acquired_(false),
        command_batch_(false) {}

  ExclusiveProbeUses(const ExclusiveProbeUses&) = delete;
  ExclusiveProbeUses& operator=(const ExclusiveProbeUses&) = delete;

  bool add(RileyCudaDeviceBuffer* buffer) noexcept {
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (buffers_[index] == buffer) {
        return true;
      }
    }
    if (buffer == nullptr || buffer_count_ == 16) {
      return false;
    }
    buffers_[buffer_count_++] = buffer;
    return true;
  }

  RileyCudaStatus acquire(RileyCudaErrorInfo* error) noexcept {
    if (command_batch_is_active(stream_)) {
      if (!plan_->owned_command_batches) {
        return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kExecuteOperation,
                                "direct cuBLAS probe rejects active command batches");
      }
      if (!command_batch_is_owned_by_current_thread(stream_)) {
        return validation_error(error, RILEY_CUDA_STATUS_INVALID_STATE,
                                RILEY_CUDA_ERROR_STAGE_VALIDATION,
                                kExecuteOperation,
                                "owned cuBLAS command batch belongs to another thread");
      }
      command_batch_ = true;
      RileyCudaStatus status = command_batch_register_use(
          stream_, &plan_->active_uses, error, kExecuteOperation,
          "owned cuBLAS plan already has an active use");
      if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
      for (size_t index = 0; index < buffer_count_; ++index) {
        status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error, kExecuteOperation,
            "owned cuBLAS device buffer already has an active use");
        if (status != RILEY_CUDA_STATUS_SUCCESS) return status;
      }
      // The batch owns the stream lease. All registered resources remain
      // leased, including after partial validation failure, until completion.
      return RILEY_CUDA_STATUS_SUCCESS;
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
  bool command_batch() const noexcept { return command_batch_; }

 private:
  bool release_acquired() noexcept {
    if (command_batch_) return true;
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
  RileyCudaDeviceBuffer* buffers_[16];
  size_t buffer_count_;
  size_t acquired_buffers_;
  bool plan_acquired_;
  bool stream_acquired_;
  bool command_batch_;
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
  // cuBLAS can select a different GEMV reduction on another SM count.
  // Keep unknown environments on the original cuBLAS path. This cold query
  // adds no CUDA call or resource to capture/replay or the execution loop.
  if (plan->owned_command_batches && capability_major==8 &&
      capability_minor==9 && runtime_version==12080 && cublas_version==120804) {
    int multiprocessors=0;
    if(cuDeviceGetAttribute(&multiprocessors,
          CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,plan->owner->device)==CUDA_SUCCESS)
      plan->owned_weight_reuse_qualified=multiprocessors==128;
  }
  const char* workspace_config=std::getenv("CUBLAS_WORKSPACE_CONFIG");
  const char* mps_percentage=std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
  // Candidate220: independently qualified SM80 Gate/Up row batching only.
  if(plan->owned_command_batches && capability_major==8 && capability_minor==0 &&
     runtime_version==12040 && cublas_version==120405 &&
     (!workspace_config || !*workspace_config) && (!mps_percentage || !*mps_percentage)) {
    int multiprocessors=0;
    if(cuDeviceGetAttribute(&multiprocessors,CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
         plan->owner->device)==CUDA_SUCCESS)
      plan->owned_sm80_gateup_rows_qualified=multiprocessors==108;
  }
  plan->owned_smol_attention_qualified=plan->owned_weight_reuse_qualified && plan->config.k==64 &&
      (!workspace_config || !*workspace_config) && (!mps_percentage || !*mps_percentage);
  plan->owned_attention_extended_qualified=plan->owned_weight_reuse_qualified &&
      (!workspace_config || !*workspace_config) && (!mps_percentage || !*mps_percentage) &&
      hf_wide_extended_kernel_arch_qualified();
  return RILEY_CUDA_STATUS_SUCCESS;
}

// Qualified, workspace-free Gate/Up recipe. A shape-dependent heuristic can
// choose sliced-K for small M and change BF16 rounding; fix the reduction here.
RileyCudaStatus prepare_owned_decode_recipe(RileyCudaCublasGemmProbePlan* plan,
                                            RileyCudaErrorInfo* error) noexcept {
  const bool singleton_batch=plan->info.compute_capability_major==8 &&
      plan->info.compute_capability_minor==9 && plan->info.cublas_version==120804;
  const bool merged_rows=plan->info.compute_capability_major==8 &&
      plan->info.compute_capability_minor==0 && plan->info.cublas_version==120405;
  // Separately qualified native-M1 small-K shapes; do not borrow the
  // large-model SIMT or A100 recipes. Unknown environments retain GemmEx.
  const char* workspace_config=std::getenv("CUBLAS_WORKSPACE_CONFIG");
  const char* mps_percentage=std::getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
  const bool small_k_singleton=plan->owned_weight_reuse_qualified &&
      (!workspace_config || !*workspace_config) && (!mps_percentage || !*mps_percentage) &&
      ((plan->config.k==576 && (plan->config.n==192 || plan->config.n==576 ||
        plan->config.n==960 || plan->config.n==1536 || plan->config.n==49152)) ||
       (plan->config.k==1536 && plan->config.n==576));
  const bool original_gate_up=plan->config.n==11008 && plan->config.k==2048;
  if (!plan->owned_command_batches || plan->config.m < 2 ||
      (!original_gate_up && !small_k_singleton) ||
      (!singleton_batch && !merged_rows)) return RILEY_CUDA_STATUS_SUCCESS;
  if(small_k_singleton)plan->decode_span_alignment=16;
  auto checked = [&](cublasStatus_t st) { return cublas_probe_error(st,error,
      RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation); };
  RileyCudaStatus status=checked(cublasLtCreate(&plan->decode_lt));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescCreate(&plan->decode_op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescSetAttribute(plan->decode_op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_weight,CUDA_R_16BF,plan->config.k,plan->config.n,plan->config.k));
  if(singleton_batch) {
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_input[1],CUDA_R_16BF,plan->config.k,1,plan->config.k));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_output[1],CUDA_R_16BF,plan->config.n,1,plan->config.n));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulPreferenceCreate(&plan->decode_preference));
    const uint64_t workspace_cap=1048576;const uint32_t alignment=small_k_singleton?16:256;
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulPreferenceSetAttribute(plan->decode_preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_cap,sizeof(workspace_cap)));
    for(auto attr:{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
      if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulPreferenceSetAttribute(plan->decode_preference,attr,&alignment,sizeof(alignment)));
    cublasLtMatmulHeuristicResult_t selected{};int returned=0;
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoGetHeuristic(plan->decode_lt,plan->decode_op,plan->decode_weight,plan->decode_input[1],plan->decode_output[1],plan->decode_output[1],plan->decode_preference,1,&selected,&returned));
    if(status==RILEY_CUDA_STATUS_SUCCESS && (returned!=1 || selected.state!=CUBLAS_STATUS_SUCCESS || selected.workspaceSize!=0))
      return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"singleton Gate/Up algorithm is unqualified");
    plan->decode_algo=selected.algo;
    const cublasLtMatmulAlgoConfigAttributes_t attributes[5]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME};
    const int expected[5]={13,0,0,1,0};
    for(size_t i=0;i<5 && status==RILEY_CUDA_STATUS_SUCCESS;++i){int value=0;size_t written=0;status=checked(cublasLtMatmulAlgoConfigGetAttribute(&plan->decode_algo,attributes[i],&value,sizeof(value),&written));if(status==RILEY_CUDA_STATUS_SUCCESS && (written!=sizeof(value)||value!=expected[i]))return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"singleton Gate/Up algorithm metadata changed");}
  } else if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoInit(plan->decode_lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,6,&plan->decode_algo));
  const uint32_t tile=singleton_batch?0:18,stages=singleton_batch?0:16,split=1,reduction=CUBLASLT_REDUCTION_SCHEME_NONE;
  if(!singleton_batch && status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->decode_algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile)));
  if(!singleton_batch && status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->decode_algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stages,sizeof(stages)));
  if(!singleton_batch && status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->decode_algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)));
  if(!singleton_batch && status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->decode_algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&reduction,sizeof(reduction)));
  // Cold prepare live row extents, independently of the enclosing bucket.
  for(uint64_t rows=2;rows<=32 && rows<=plan->config.m && status==RILEY_CUDA_STATUS_SUCCESS;++rows) {
    status=checked(cublasLtMatrixLayoutCreate(&plan->decode_input[rows],CUDA_R_16BF,plan->config.k,singleton_batch?1:rows,plan->config.k));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_output[rows],CUDA_R_16BF,plan->config.n,singleton_batch?1:rows,plan->config.n));
    if(singleton_batch && status==RILEY_CUDA_STATUS_SUCCESS) {
      status=checked(cublasLtMatrixLayoutCreate(&plan->decode_batch_weight[rows],CUDA_R_16BF,plan->config.k,plan->config.n,plan->config.k));
      const int32_t count=static_cast<int32_t>(rows);
      const int64_t strides[3]={0,static_cast<int64_t>(plan->config.k),static_cast<int64_t>(plan->config.n)};
      const cublasLtMatrixLayout_t layouts[3]={plan->decode_batch_weight[rows],plan->decode_input[rows],plan->decode_output[rows]};
      for(size_t i=0;i<3 && status==RILEY_CUDA_STATUS_SUCCESS;++i) {
        status=checked(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&count,sizeof(count)));
        if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutSetAttribute(layouts[i],CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&strides[i],sizeof(strides[i])));
      }
    }
    cublasLtMatmulHeuristicResult_t check{};
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoCheck(plan->decode_lt,plan->decode_op,singleton_batch?plan->decode_batch_weight[rows]:plan->decode_weight,plan->decode_input[rows],plan->decode_output[rows],plan->decode_output[rows],&plan->decode_algo,&check));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(check.state);
    if(status==RILEY_CUDA_STATUS_SUCCESS && check.workspaceSize!=0)
      return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"qualified decode recipe requires unexpected workspace");
  }
  plan->decode_ready=status==RILEY_CUDA_STATUS_SUCCESS;
  return status;
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
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=prepare_owned_decode_recipe(plan,error);
  return status;
}

RileyCudaStatus destroy_plan_resources(RileyCudaCublasGemmProbePlan* plan,
                                       RileyCudaErrorInfo* error,
                                       uint32_t stage,
                                       const char* operation) noexcept {
  plan->prefill_max_rows=0;
  if(plan->prefill_workspace!=nullptr) {
    auto st=riley_cuda_device_buffer_close(&plan->prefill_workspace,error);
    if(st!=RILEY_CUDA_STATUS_SUCCESS)return st;
  }
  plan->prefill_workspace_bytes=0;
  plan->decode_ready=false;
  plan->sm80_projection_rows=0;
  plan->owned_sm80_full_projection_rows=false;
  if(plan->sm80_full_projection_workspace!=nullptr){auto st=riley_cuda_device_buffer_close(&plan->sm80_full_projection_workspace,error);if(st!=RILEY_CUDA_STATUS_SUCCESS)return st;}
  plan->sm80_full_projection_workspace_bytes=0;
  if(plan->sm80_projection_workspace!=nullptr) {
    auto st=riley_cuda_device_buffer_close(&plan->sm80_projection_workspace,error);
    if(st!=RILEY_CUDA_STATUS_SUCCESS)return st;
  }
  plan->sm80_projection_workspace_bytes=0;
#define DESTROY_DECODE(member, destroy) \
  if(plan->member!=nullptr) { auto st=destroy(plan->member); \
    if(st!=CUBLAS_STATUS_SUCCESS)return cublas_probe_error(st,error,stage,operation); \
    plan->member=nullptr; }
  for(size_t rows=0;rows<33;++rows) {
    DESTROY_DECODE(decode_output[rows],cublasLtMatrixLayoutDestroy)
    DESTROY_DECODE(decode_input[rows],cublasLtMatrixLayoutDestroy)
    DESTROY_DECODE(decode_batch_weight[rows],cublasLtMatrixLayoutDestroy)
  }
  for(size_t rows=0;rows<577;++rows) {
    DESTROY_DECODE(prefill_output[rows],cublasLtMatrixLayoutDestroy)
    DESTROY_DECODE(prefill_input[rows],cublasLtMatrixLayoutDestroy)
  }
  DESTROY_DECODE(prefill_weight,cublasLtMatrixLayoutDestroy)
  DESTROY_DECODE(prefill_op,cublasLtMatmulDescDestroy)
  DESTROY_DECODE(prefill_lt,cublasLtDestroy)
  DESTROY_DECODE(decode_weight,cublasLtMatrixLayoutDestroy)
  DESTROY_DECODE(decode_preference,cublasLtMatmulPreferenceDestroy)
  DESTROY_DECODE(decode_op,cublasLtMatmulDescDestroy)
  DESTROY_DECODE(decode_lt,cublasLtDestroy)
#undef DESTROY_DECODE
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
  if(plan==nullptr || plan->sm80_full_projection_workspace!=nullptr || plan->prefill_workspace!=nullptr || plan->prefill_lt!=nullptr || plan->prefill_op!=nullptr || plan->prefill_weight!=nullptr || plan->sm80_projection_workspace!=nullptr || plan->handle!=nullptr || plan->decode_lt!=nullptr || plan->decode_op!=nullptr || plan->decode_weight!=nullptr || plan->decode_preference!=nullptr)return false;
  for(size_t rows=0;rows<577;++rows)
    if(plan->prefill_input[rows]!=nullptr || plan->prefill_output[rows]!=nullptr)return false;
  for(size_t rows=0;rows<33;++rows)
    if(plan->decode_input[rows]!=nullptr || plan->decode_output[rows]!=nullptr || plan->decode_batch_weight[rows]!=nullptr)return false;
  return true;
}

RileyCudaStatus complete_execution(
    ExclusiveProbeUses* uses, CurrentContext* scope, RileyCudaStream* stream,
    RileyCudaStatus operation_status, bool launch_attempted,
    RileyCudaErrorInfo* error) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RILEY_CUDA_ERROR_STAGE_SYNCHRONIZE, kExecuteOperation);
  }
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

// SM89/cuBLAS12.8.4 independent-M1 reduction, qualified on 109 actual
// Qwen2.5 projections, 4469 row-count/input-offset conditions and changed-input
// CUDA Graph replay. Gate/Up/LM-head retain 16 K lanes; Down retains 32.
// Reuse only weights and input staging, preserving every FP32 FMA, reduction
// addition and final BF16 rounding. No allocation or library call in the kernel.
// Preserve the proven lane-strided serial FMA and final half-tree. Only
// shared-input staging and the independent row/column tile change.
template<int L,int R,int C> __global__ void owned_decode_weight_reuse_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 int lane=threadIdx.x,col=blockIdx.x*C+threadIdx.y,tid=lane+L*threadIdx.y,row_base=blockIdx.y*R;
 float accum[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int i=tid*8;i<R*256;i+=L*C*8){int row=row_base+i/256,pos=chunk+i%256;auto dst=reinterpret_cast<uint4*>(sx+i);if(row<rows&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(x+row*k+pos);else *dst=make_uint4(0,0,0,0);}
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(w[col*k+chunk+pos])):0;
   #pragma unroll
   for(int r=0;r<R;r++)accum[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(sx[r*256+pos])),accum[r]);
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows)y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));
 }
}
// SM89 component gate:109actual projections,4469row/input-offset conditions,
// unchanged-input and changed-input CUDA Graph replay all byte exact.
// Vectorize only weight-bit staging; every FP32 FMA and half-tree stays fixed.
template<int L,int R,int C> __global__ void owned_decode_weight_staged_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 unsigned short* sw=sx+R*256;
 constexpr int WSTRIDE=272;
 int lane=threadIdx.x,col=blockIdx.x*C+threadIdx.y,tid=lane+L*threadIdx.y,row_base=blockIdx.y*R;
 float accum[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int i=tid*8;i<R*256;i+=L*C*8){int row=row_base+i/256,pos=chunk+i%256;auto dst=reinterpret_cast<uint4*>(sx+i);if(row<rows&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(x+row*k+pos);else *dst=make_uint4(0,0,0,0);}
  // Copy unchanged weight bits with coalesced 16-byte loads. Padding each
  // shared weight row separates adjacent half-warps' bank addresses.
  for(int i=tid*8;i<C*256;i+=L*C*8){
   int c=blockIdx.x*C+i/256,pos=chunk+i%256;
   auto dst=reinterpret_cast<uint4*>(sw+(i/256)*WSTRIDE+i%256);
   if(c<n&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(w+c*k+pos);
   else *dst=make_uint4(0,0,0,0);
  }
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WSTRIDE+pos])):0;
   #pragma unroll
   for(int r=0;r<R;r++)accum[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(sx[r*256+pos])),accum[r]);
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows)y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));
 }
}
template<int L,int R,int C> __global__ void owned_decode_weight_staged_interleaved_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 unsigned short* sw=sx+R*256;
 constexpr int WSTRIDE=272;
 int lane=threadIdx.x,col=blockIdx.y*C+threadIdx.y,tid=lane+L*threadIdx.y,row_base=blockIdx.x*R;
 float accum[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int i=tid*8;i<R*256;i+=L*C*8){int row=row_base+i/256,pos=chunk+i%256;auto dst=reinterpret_cast<uint4*>(sx+i);if(row<rows&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(x+row*k+pos);else *dst=make_uint4(0,0,0,0);}
  // Copy unchanged weight bits with coalesced 16-byte loads. Padding each
  // shared weight row separates adjacent half-warps' bank addresses.
  for(int i=tid*8;i<C*256;i+=L*C*8){
   int c=blockIdx.y*C+i/256,pos=chunk+i%256;
   auto dst=reinterpret_cast<uint4*>(sw+(i/256)*WSTRIDE+i%256);
   if(c<n&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(w+c*k+pos);
   else *dst=make_uint4(0,0,0,0);
  }
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WSTRIDE+pos])):0;
   #pragma unroll
   for(int r=0;r<R;r++)accum[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(sx[r*256+pos])),accum[r]);
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows)y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));
 }
}
template<int L,int R,int C> __global__ void owned_decode_packed4_transpose_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 unsigned short* sw=sx+R*256;
 constexpr int WSTRIDE=272;
 int lane=threadIdx.x,col=blockIdx.x*C+threadIdx.y,tid=lane+L*threadIdx.y,row_base=blockIdx.y*R;
 float accum[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int vector=tid;vector<R*256/8;vector+=L*C){
   int within=vector%4,group=vector/4,rbase=(group/32)*4,pos=(group%32)*8,row=row_base+rbase+within;
   uint4 source=(row<rows&&chunk+pos+7<k)?*reinterpret_cast<const uint4*>(x+row*k+chunk+pos):make_uint4(0,0,0,0);
   unsigned a=source.x,b=source.y,c=source.z,d=source.w;
   // A4x4 transpose of packed BF16 pairs within each four-lane group.
   unsigned t0=__shfl_xor_sync(0xffffffff,(within&1)?a:b,1,4);
   unsigned t1=__shfl_xor_sync(0xffffffff,(within&1)?c:d,1,4);
   if(within&1){a=t0;c=t1;}else{b=t0;d=t1;}
   t0=__shfl_xor_sync(0xffffffff,(within&2)?a:c,2,4);
   t1=__shfl_xor_sync(0xffffffff,(within&2)?b:d,2,4);
   if(within&2){a=t0;b=t1;}else{c=t0;d=t1;}
   int p0=pos+within*2,target=((p0/L)*(R/4)+rbase/4)*(L*4)+(p0%L)*4;
   *reinterpret_cast<uint4*>(sx+target)=make_uint4(__byte_perm(a,b,0x5410),__byte_perm(c,d,0x5410),__byte_perm(a,b,0x7632),__byte_perm(c,d,0x7632));
  }
  // Copy unchanged weight bits with coalesced 16-byte loads. Padding each
  // shared weight row separates adjacent half-warps' bank addresses.
  for(int i=tid*8;i<C*256;i+=L*C*8){
   int c=blockIdx.x*C+i/256,pos=chunk+i%256;
   auto dst=reinterpret_cast<uint4*>(sw+(i/256)*WSTRIDE+i%256);
   if(c<n&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(w+c*k+pos);
   else *dst=make_uint4(0,0,0,0);
  }
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WSTRIDE+pos])):0;
   #pragma unroll
   for(int group=0;group<R/4;group++){
    int address=((pos/L)*(R/4)+group)*(L*4)+(pos%L)*4;
    // One volatile block consumes each shared vector before the next row
    // group, avoiding hoisting all packed vectors into long-lived registers.
    unsigned location=static_cast<unsigned>(__cvta_generic_to_shared(sx+address));
    asm volatile("{\n\t"
      ".reg .b32 lo, hi, bits;\n\t"
      "ld.shared.v2.u32 {lo, hi}, [%4];\n\t"
      "shl.b32 bits, lo, 16;\n\t"
      "fma.rn.f32 %0, %5, bits, %0;\n\t"
      "and.b32 bits, lo, 0xffff0000;\n\t"
      "fma.rn.f32 %1, %5, bits, %1;\n\t"
      "shl.b32 bits, hi, 16;\n\t"
      "fma.rn.f32 %2, %5, bits, %2;\n\t"
      "and.b32 bits, hi, 0xffff0000;\n\t"
      "fma.rn.f32 %3, %5, bits, %3;\n\t"
      "}" : "+f"(accum[group*4]), "+f"(accum[group*4+1]), "+f"(accum[group*4+2]), "+f"(accum[group*4+3]) : "r"(location), "f"(ww) : "memory");
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows)y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));
 }
}

template<int L,int R,int C,bool RowMajor=false> __global__ void owned_decode_column_pair_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 unsigned short* sw=sx+R*256;
 constexpr int WSTRIDE=272;
 int lane=threadIdx.x,col=(RowMajor?blockIdx.y:blockIdx.x)*(2*C)+threadIdx.y,tid=lane+L*threadIdx.y,row_base=(RowMajor?blockIdx.x:blockIdx.y)*R;
 float accum[R]={},accum2[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int vector=tid;vector<R*256/8;vector+=L*C){
   int within=vector%4,group=vector/4,rbase=(group/32)*4,pos=(group%32)*8,row=row_base+rbase+within;
   uint4 source=(row<rows&&chunk+pos+7<k)?*reinterpret_cast<const uint4*>(x+row*k+chunk+pos):make_uint4(0,0,0,0);
   unsigned a=source.x,b=source.y,c=source.z,d=source.w;
   // A4x4 transpose of packed BF16 pairs within each four-lane group.
   unsigned t0=__shfl_xor_sync(0xffffffff,(within&1)?a:b,1,4);
   unsigned t1=__shfl_xor_sync(0xffffffff,(within&1)?c:d,1,4);
   if(within&1){a=t0;c=t1;}else{b=t0;d=t1;}
   t0=__shfl_xor_sync(0xffffffff,(within&2)?a:c,2,4);
   t1=__shfl_xor_sync(0xffffffff,(within&2)?b:d,2,4);
   if(within&2){a=t0;b=t1;}else{c=t0;d=t1;}
   int p0=pos+within*2,target=((p0/L)*(R/4)+rbase/4)*(L*4)+(p0%L)*4;
   *reinterpret_cast<uint4*>(sx+target)=make_uint4(__byte_perm(a,b,0x5410),__byte_perm(c,d,0x5410),__byte_perm(a,b,0x7632),__byte_perm(c,d,0x7632));
  }
  // Copy unchanged weight bits with coalesced 16-byte loads. Padding each
  // shared weight row separates adjacent half-warps' bank addresses.
  for(int i=tid*8;i<2*C*256;i+=L*C*8){
   int c=(RowMajor?blockIdx.y:blockIdx.x)*(2*C)+i/256,pos=chunk+i%256;
   auto dst=reinterpret_cast<uint4*>(sw+(i/256)*WSTRIDE+i%256);
   if(c<n&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(w+c*k+pos);
   else *dst=make_uint4(0,0,0,0);
  }
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WSTRIDE+pos])):0;float ww2=col+C<n?__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+C)*WSTRIDE+pos])):0;
   #pragma unroll
   for(int group=0;group<R/4;group++){
    int address=((pos/L)*(R/4)+group)*(L*4)+(pos%L)*4;
    // One volatile block consumes each shared vector before the next row
    // group, avoiding hoisting all packed vectors into long-lived registers.
    unsigned location=static_cast<unsigned>(__cvta_generic_to_shared(sx+address));
    asm volatile("{\n\t"
      ".reg .b32 lo, hi, bits;\n\t"
      "ld.shared.v2.u32 {lo, hi}, [%8];\n\t"
      "shl.b32 bits, lo, 16;\n\t"
      "fma.rn.f32 %0, %9, bits, %0;\n\t"
      "fma.rn.f32 %4, %10, bits, %4;\n\t"
      "and.b32 bits, lo, 0xffff0000;\n\t"
      "fma.rn.f32 %1, %9, bits, %1;\n\t"
      "fma.rn.f32 %5, %10, bits, %5;\n\t"
      "shl.b32 bits, hi, 16;\n\t"
      "fma.rn.f32 %2, %9, bits, %2;\n\t"
      "fma.rn.f32 %6, %10, bits, %6;\n\t"
      "and.b32 bits, hi, 0xffff0000;\n\t"
      "fma.rn.f32 %3, %9, bits, %3;\n\t"
      "fma.rn.f32 %7, %10, bits, %7;\n\t"
      "}" : "+f"(accum[group*4]), "+f"(accum[group*4+1]), "+f"(accum[group*4+2]), "+f"(accum[group*4+3]),
      "+f"(accum2[group*4]), "+f"(accum2[group*4+1]), "+f"(accum2[group*4+2]), "+f"(accum2[group*4+3]) : "r"(location), "f"(ww), "f"(ww2) : "memory");
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other);float other2=__shfl_down_sync(0xffffffff,accum2[r],offset,L);if(lane+offset<L)accum2[r]=__fadd_rn(accum2[r],other2);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows){y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));if(col+C<n)y[(row_base+r)*n+col+C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum2[r]));}
 }
}


// Four-column native-order reuse; only measured profitable crossover intervals.
template<int L,int R,int C,bool RowMajor=false> __global__ void owned_decode_column_quad_kernel(const unsigned short* __restrict__ w,const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 extern __shared__ __align__(16) unsigned short sx[];
 unsigned short* sw=sx+R*256;
 constexpr int WSTRIDE=272;
 int lane=threadIdx.x,col=(RowMajor?blockIdx.y:blockIdx.x)*(4*C)+threadIdx.y,tid=lane+L*threadIdx.y,row_base=(RowMajor?blockIdx.x:blockIdx.y)*R;
 float accum[R]={},accum2[R]={},accum3[R]={},accum4[R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int vector=tid;vector<R*256/8;vector+=L*C){
   int within=vector%4,group=vector/4,rbase=(group/32)*4,pos=(group%32)*8,row=row_base+rbase+within;
   uint4 source=(row<rows&&chunk+pos+7<k)?*reinterpret_cast<const uint4*>(x+row*k+chunk+pos):make_uint4(0,0,0,0);
   unsigned a=source.x,b=source.y,c=source.z,d=source.w;
   // A4x4 transpose of packed BF16 pairs within each four-lane group.
   unsigned t0=__shfl_xor_sync(0xffffffff,(within&1)?a:b,1,4);
   unsigned t1=__shfl_xor_sync(0xffffffff,(within&1)?c:d,1,4);
   if(within&1){a=t0;c=t1;}else{b=t0;d=t1;}
   t0=__shfl_xor_sync(0xffffffff,(within&2)?a:c,2,4);
   t1=__shfl_xor_sync(0xffffffff,(within&2)?b:d,2,4);
   if(within&2){a=t0;b=t1;}else{c=t0;d=t1;}
   int p0=pos+within*2,target=((p0/L)*(R/4)+rbase/4)*(L*4)+(p0%L)*4;
   *reinterpret_cast<uint4*>(sx+target)=make_uint4(__byte_perm(a,b,0x5410),__byte_perm(c,d,0x5410),__byte_perm(a,b,0x7632),__byte_perm(c,d,0x7632));
  }
  // Copy unchanged weight bits with coalesced 16-byte loads. Padding each
  // shared weight row separates adjacent half-warps' bank addresses.
  for(int i=tid*8;i<4*C*256;i+=L*C*8){
   int c=(RowMajor?blockIdx.y:blockIdx.x)*(4*C)+i/256,pos=chunk+i%256;
   auto dst=reinterpret_cast<uint4*>(sw+(i/256)*WSTRIDE+i%256);
   if(c<n&&pos+7<k)*dst=*reinterpret_cast<const uint4*>(w+c*k+pos);
   else *dst=make_uint4(0,0,0,0);
  }
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){
   float ww=col+0*C<n?__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+0*C)*WSTRIDE+pos])):0;
   float ww2=col+1*C<n?__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+1*C)*WSTRIDE+pos])):0;
   float ww3=col+2*C<n?__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+2*C)*WSTRIDE+pos])):0;
   float ww4=col+3*C<n?__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+3*C)*WSTRIDE+pos])):0;
   #pragma unroll
   for(int group=0;group<R/4;group++){
    int address=((pos/L)*(R/4)+group)*(L*4)+(pos%L)*4;
    unsigned location=static_cast<unsigned>(__cvta_generic_to_shared(sx+address));
    asm volatile(
      "{\n\t"
      ".reg .b32 lo, hi, bits;\n\t"
      "ld.shared.v2.u32 {lo, hi}, [%16];\n\t"
      "shl.b32 bits, lo, 16;\n\t"
      "fma.rn.f32 %0, %17, bits, %0;\n\t"
      "fma.rn.f32 %4, %18, bits, %4;\n\t"
      "fma.rn.f32 %8, %19, bits, %8;\n\t"
      "fma.rn.f32 %12, %20, bits, %12;\n\t"
      "and.b32 bits, lo, 0xffff0000;\n\t"
      "fma.rn.f32 %1, %17, bits, %1;\n\t"
      "fma.rn.f32 %5, %18, bits, %5;\n\t"
      "fma.rn.f32 %9, %19, bits, %9;\n\t"
      "fma.rn.f32 %13, %20, bits, %13;\n\t"
      "shl.b32 bits, hi, 16;\n\t"
      "fma.rn.f32 %2, %17, bits, %2;\n\t"
      "fma.rn.f32 %6, %18, bits, %6;\n\t"
      "fma.rn.f32 %10, %19, bits, %10;\n\t"
      "fma.rn.f32 %14, %20, bits, %14;\n\t"
      "and.b32 bits, hi, 0xffff0000;\n\t"
      "fma.rn.f32 %3, %17, bits, %3;\n\t"
      "fma.rn.f32 %7, %18, bits, %7;\n\t"
      "fma.rn.f32 %11, %19, bits, %11;\n\t"
      "fma.rn.f32 %15, %20, bits, %15;\n\t"
      "}\n\t"
      : "+f"(accum[group*4+0]), "+f"(accum[group*4+1]), "+f"(accum[group*4+2]), "+f"(accum[group*4+3]), "+f"(accum2[group*4+0]), "+f"(accum2[group*4+1]), "+f"(accum2[group*4+2]), "+f"(accum2[group*4+3]), "+f"(accum3[group*4+0]), "+f"(accum3[group*4+1]), "+f"(accum3[group*4+2]), "+f"(accum3[group*4+3]), "+f"(accum4[group*4+0]), "+f"(accum4[group*4+1]), "+f"(accum4[group*4+2]), "+f"(accum4[group*4+3]) : "r"(location), "f"(ww), "f"(ww2), "f"(ww3), "f"(ww4) : "memory");
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int r=0;r<R;r++){
   float other0=__shfl_down_sync(0xffffffff,accum[r],offset,L);if(lane+offset<L)accum[r]=__fadd_rn(accum[r],other0);
   float other1=__shfl_down_sync(0xffffffff,accum2[r],offset,L);if(lane+offset<L)accum2[r]=__fadd_rn(accum2[r],other1);
   float other2=__shfl_down_sync(0xffffffff,accum3[r],offset,L);if(lane+offset<L)accum3[r]=__fadd_rn(accum3[r],other2);
   float other3=__shfl_down_sync(0xffffffff,accum4[r],offset,L);if(lane+offset<L)accum4[r]=__fadd_rn(accum4[r],other3);
  }
 }
 if(lane==0){
  #pragma unroll
  for(int r=0;r<R;r++)if(row_base+r<rows){
   if(col+0*C<n)y[(row_base+r)*n+col+0*C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[r]));
   if(col+1*C<n)y[(row_base+r)*n+col+1*C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum2[r]));
   if(col+2*C<n)y[(row_base+r)*n+col+2*C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum3[r]));
   if(col+3*C<n)y[(row_base+r)*n+col+3*C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum4[r]));
  }
 }
}

// V112: qualified small-row body, preserving original native FMA/reduction.
template<int L,int R,int C>__global__ void owned_small_row_quad_v109(const unsigned short*__restrict__ w,const unsigned short*__restrict__ x,unsigned short*y,int n,int k,int rows){
 static_assert(R==1||R==2,"small rows");constexpr int WS=272;
 extern __shared__ __align__(16) unsigned short data[];unsigned short*sx=data;unsigned short*sw=sx+R*256;
 int lane=threadIdx.x,tid=lane+L*threadIdx.y,row0=blockIdx.x*R,col0=blockIdx.y*(4*C)+threadIdx.y;
 float accum[4][R]={};
 for(int chunk=0;chunk<k;chunk+=256){
  for(int i=tid*8;i<R*256;i+=L*C*8){int row=row0+i/256,pos=chunk+i%256;uint4 v=row<rows&&pos+7<k?*reinterpret_cast<const uint4*>(x+row*k+pos):make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(sx+i)=v;}
  for(int i=tid*8;i<4*C*256;i+=L*C*8){int col=blockIdx.y*(4*C)+i/256,pos=chunk+i%256;uint4 v=col<n&&pos+7<k?*reinterpret_cast<const uint4*>(w+col*k+pos):make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(sw+(i/256)*WS+i%256)=v;}
  __syncthreads();
  for(int pos=lane;pos<256&&chunk+pos<k;pos+=L){
   float weights[4];
   #pragma unroll
   for(int col=0;col<4;col++)weights[col]=__bfloat162float(__ushort_as_bfloat16(sw[(threadIdx.y+col*C)*WS+pos]));
   #pragma unroll
   for(int row=0;row<R;row++){float input=__bfloat162float(__ushort_as_bfloat16(sx[row*256+pos]));
    #pragma unroll
    for(int col=0;col<4;col++)accum[col][row]=__fmaf_rn(weights[col],input,accum[col][row]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int offset=L/2;offset;offset/=2){
  #pragma unroll
  for(int col=0;col<4;col++){
   #pragma unroll
   for(int row=0;row<R;row++){float other=__shfl_down_sync(0xffffffff,accum[col][row],offset,L);if(lane+offset<L)accum[col][row]=__fadd_rn(accum[col][row],other);}
  }
 }
 if(lane==0){
  #pragma unroll
  for(int col=0;col<4;col++){
   #pragma unroll
   for(int row=0;row<R;row++)if(col0+col*C<n&&row0+row<rows)y[(row0+row)*n+col0+col*C]=__bfloat16_as_ushort(__float2bfloat16_rn(accum[col][row]));
  }
 }
}

static int owned_quad_v107_variant(int n,int k,int rows){
 if(rows<3||rows>32)return -1;
 if(n==2560&&k==9728){
  if(rows>=9&&rows<=16)return 1;
  if(rows>=17&&rows<=24)return 1;
  if(rows>=25&&rows<=32)return 0;
  return -1;
 }
 if(n==9728&&k==2560){
  if(rows>=3&&rows<=8)return 1;
  if(rows>=17&&rows<=24)return 1;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 if(n==2560&&k==4096){
  if(rows>=17&&rows<=24)return 1;
  return -1;
 }
 if(n==4096&&k==2560){
  if(rows>=17&&rows<=24)return 1;
  return -1;
 }
 if(n==151936&&k==2560){
  if(rows>=17&&rows<=24)return 1;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 if(n==4096&&k==14336){
  if(rows>=9&&rows<=16)return 1;
  if(rows>=17&&rows<=24)return 3;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 if(n==14336&&k==4096){
  if(rows>=9&&rows<=16)return 3;
  if(rows>=17&&rows<=24)return 0;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 if(n==4096&&k==4096){
  if(rows>=9&&rows<=16)return 1;
  if(rows>=17&&rows<=24)return 1;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 if(n==32768&&k==4096){
  if(rows>=9&&rows<=16)return 2;
  if(rows>=17&&rows<=24)return 0;
  if(rows>=25&&rows<=32)return 3;
  return -1;
 }
 return -1;
}

template<int L>static void launch_owned_quad_v107_t(int variant,const unsigned short*w,const unsigned short*x,unsigned short*y,int n,int k,int rows,cudaStream_t stream){
#define RILEY_QUAD_V107(R,C) owned_decode_column_quad_kernel<L,R,C,true><<<dim3((rows+R-1)/R,(n+4*C-1)/(4*C)),dim3(L,C),(R*256+4*C*272)*2,stream>>>(w,x,y,n,k,rows)
 switch(variant){case 0:RILEY_QUAD_V107(8,4);break;case 1:RILEY_QUAD_V107(8,8);break;case 2:RILEY_QUAD_V107(16,4);break;case 3:RILEY_QUAD_V107(16,8);break;}
#undef RILEY_QUAD_V107
}
static void launch_owned_quad_v107(int variant,const unsigned short*w,const unsigned short*x,unsigned short*y,int n,int k,int rows,cudaStream_t stream){
 const bool lanes32=(n==2560&&k==9728)||(n==4096&&k==14336)||(k==4096&&(n==14336||n==32768));
 if(lanes32)launch_owned_quad_v107_t<32>(variant,w,x,y,n,k,rows,stream);else launch_owned_quad_v107_t<16>(variant,w,x,y,n,k,rows,stream);
}

void launch_owned_decode_weight_reuse(const unsigned short*w,const unsigned short*x,unsigned short*y,int n,int k,int rows,cudaStream_t stream){
 #define RILEY_RUN_OWNED_INTERLEAVED(L,R,C) owned_decode_weight_staged_interleaved_kernel<L,R,C><<<dim3((rows+R-1)/R,(n+C-1)/C),dim3(L,C),(R*256+C*272)*2,stream>>>(w,x,y,n,k,rows)
 #define RILEY_RUN_OWNED_STAGED(L,R,C) owned_decode_weight_staged_kernel<L,R,C><<<dim3((n+C-1)/C,(rows+R-1)/R),dim3(L,C),(R*256+C*272)*2,stream>>>(w,x,y,n,k,rows)
 // Weight spans permit BF16 alignment; use vector loads only when their
 // actual pointer is16-byte aligned. The scalar-load fallback remains valid.
 const bool vector_weight_aligned=(reinterpret_cast<uintptr_t>(w)&uintptr_t(15))==0;
 // Reuse each packed input conversion across two independent output columns.
 // Only measured profitable shapes/counts; preserve each native FMA chain.
 // Q4 large-row tile:16rows x16outputs; each thread shares input across2columns.
 // Row-first CTAs reuse adjacent weight tiles through L2. Exact L16 order.
 if(vector_weight_aligned&&rows>=17&&rows<=32&&
    ((k==2560&&(n==9728||n==151936||n==4096))||(n==2560&&k==4096))){
  owned_decode_column_pair_kernel<16,16,8,true><<<dim3((rows+15)/16,(n+15)/16),dim3(16,8),(16*256+16*272)*2,stream>>>(w,x,y,n,k,rows);
  return;
 }
 int column_pair_lanes=0;
 if(rows<=32){
  if(k==4096&&(n==14336||n==32768)&&rows>=17)column_pair_lanes=32;
  if(n==4096&&k==14336&&rows>=9&&rows<=24)column_pair_lanes=32;
  if(n==9728&&k==2560&&rows>=2&&rows<=8)column_pair_lanes=16;
  if(n==2560&&k==9728&&rows>=17)column_pair_lanes=32;
 }
 if(vector_weight_aligned&&column_pair_lanes){
 #define RILEY_COLUMN_PAIR(L,R) owned_decode_column_pair_kernel<L,R,4><<<dim3((n+7)/8,(rows+R-1)/R),dim3(L,4),(R*256+8*272)*2,stream>>>(w,x,y,n,k,rows)
  if(column_pair_lanes==16){RILEY_COLUMN_PAIR(16,8);}
  else if(rows<=16){RILEY_COLUMN_PAIR(32,16);}else{RILEY_COLUMN_PAIR(32,32);}
 #undef RILEY_COLUMN_PAIR
  return;
 }

 // Packed BF16 shared loads preserve each output's serial FMA/half-tree.
 // Warp-register transpose keeps global/shared staging vectorized as well.
 int packed_lanes=0;
 if(n==14336&&k==4096&&rows>=9)packed_lanes=32;
 if(n==4096&&k==14336&&rows>=9)packed_lanes=32;
 if(n==32768&&k==4096&&rows>=9)packed_lanes=32;
 if(n==4096&&k==4096&&rows>=3)packed_lanes=16;
 if(n==9728&&k==2560&&rows>=3)packed_lanes=16;
 if(n==2560&&k==9728&&rows>=4)packed_lanes=32;
 if(n==151936&&k==2560&&rows>=9)packed_lanes=16;
 if(n==4096&&k==2560&&rows>=2)packed_lanes=16;
 if(n==2560&&k==4096&&rows>=8)packed_lanes=16;
 if(vector_weight_aligned&&packed_lanes){
 #define RILEY_RUN_PACKED(L,R) owned_decode_packed4_transpose_kernel<L,R,8><<<dim3((n+7)/8,(rows+R-1)/R),dim3(L,8),(R*256+8*272)*2,stream>>>(w,x,y,n,k,rows)
  if(packed_lanes==16){if(rows<=8){RILEY_RUN_PACKED(16,8);}else if(rows<=16){RILEY_RUN_PACKED(16,16);}else{RILEY_RUN_PACKED(16,32);}}
  else {if(rows<=8){RILEY_RUN_PACKED(32,8);}else if(rows<=16){RILEY_RUN_PACKED(32,16);}else{RILEY_RUN_PACKED(32,32);}}
 #undef RILEY_RUN_PACKED
  return;
 }

 // Small-N K/V tiles need more resident CTAs. Row-first grid ordering
 // retains each dot product while sharing nearby weight tiles through L2.
 if(vector_weight_aligned&&n==1024&&k==4096&&rows>16){RILEY_RUN_OWNED_INTERLEAVED(16,16,8);return;}
 const bool qwen4b_down=n==2560&&k==9728;
 const bool qwen4b_l16=(k==2560&&(n==9728||n==151936||n==4096||n==1024))||(n==2560&&k==4096);
 if(vector_weight_aligned&&(qwen4b_down||qwen4b_l16)){
  if(qwen4b_down){
   if(rows>16){RILEY_RUN_OWNED_INTERLEAVED(32,16,8);}
   else if(rows<=8){RILEY_RUN_OWNED_STAGED(32,8,8);}else{RILEY_RUN_OWNED_STAGED(32,16,8);}
  }else{
   const bool interleave=rows>16&&((n==1024&&k==2560)||(n==2560&&k==4096));
   if(interleave){RILEY_RUN_OWNED_INTERLEAVED(16,16,8);}
   else if(rows<=8){RILEY_RUN_OWNED_STAGED(16,8,8);}
   else if(rows<=16){RILEY_RUN_OWNED_STAGED(16,16,8);}
   else{RILEY_RUN_OWNED_STAGED(16,32,8);}
  }
  return;
 }

 // Coalesced 16-byte weight staging preserves every FMA and reduction.
 // Keep scalar weights for large row tiles without a measured gain.
 if(vector_weight_aligned&&(((n==14336||n==32768)&&k==4096)||(n==4096&&k==14336))&&rows<=16){
  if(rows<=8){RILEY_RUN_OWNED_STAGED(32,8,8);}else{RILEY_RUN_OWNED_STAGED(32,16,8);}return;
 }
 if(vector_weight_aligned&&k==4096&&((n==4096&&rows<=8)||n==1024)){
  if(rows<=8){RILEY_RUN_OWNED_STAGED(16,8,8);}else if(rows<=16){RILEY_RUN_OWNED_STAGED(16,16,8);}else{RILEY_RUN_OWNED_STAGED(16,32,8);}return;
 }

 // Keep the original Gate/Up and large-Down kernels: their micro gains were
 // absent. LM Head and small Down alone use the qualified staged variant.
 if(vector_weight_aligned&&n==2048&&k==11008&&rows<=8){RILEY_RUN_OWNED_STAGED(32,8,8);return;}
 if(vector_weight_aligned&&n==151936&&k==2048){
  if(rows<=8){RILEY_RUN_OWNED_STAGED(16,8,16);}
  else if(rows<=16){RILEY_RUN_OWNED_STAGED(16,16,16);}
  else{RILEY_RUN_OWNED_STAGED(16,32,16);}
  return;
 }
 #undef RILEY_RUN_OWNED_STAGED
 #undef RILEY_RUN_OWNED_INTERLEAVED
 #define RILEY_RUN_OWNED_REUSE(L,R,C) owned_decode_weight_reuse_kernel<L,R,C><<<dim3((n+C-1)/C,(rows+R-1)/R),dim3(L,C),R*256*2,stream>>>(w,x,y,n,k,rows)
 // Mistral-width SM89 projections: retain the qualified M=1 lane/FMA
 // topology while sharing weights across independent decode requests.
 // Q/K/V/O require L16; Gate/Up/Down/LM require L32 on this runtime.
 if((k==4096&&(n==4096||n==1024))){
  if(rows<=8){RILEY_RUN_OWNED_REUSE(16,8,8);}
  else if(rows<=16){RILEY_RUN_OWNED_REUSE(16,16,8);}
  else{RILEY_RUN_OWNED_REUSE(16,32,8);}
  return;
 }
 if((n==14336&&k==4096)||(n==4096&&k==14336)||(n==32768&&k==4096)){
  if(rows<=8){RILEY_RUN_OWNED_REUSE(32,8,8);}
  else if(rows<=16){RILEY_RUN_OWNED_REUSE(32,16,8);}
  else{RILEY_RUN_OWNED_REUSE(32,32,8);}
  return;
 }
 if(n==11008&&k==2048){if(rows<=8){RILEY_RUN_OWNED_REUSE(16,8,8);}else {RILEY_RUN_OWNED_REUSE(16,16,16);}}
 else if(n==2048&&k==11008){if(rows<=8){RILEY_RUN_OWNED_REUSE(32,8,8);}else {RILEY_RUN_OWNED_REUSE(32,16,8);}}
 else if(n==151936&&k==2048){if(rows<=8){RILEY_RUN_OWNED_REUSE(16,8,16);}else if(rows<=16){RILEY_RUN_OWNED_REUSE(16,16,16);}else{RILEY_RUN_OWNED_REUSE(16,32,16);}}


 #undef RILEY_RUN_OWNED_REUSE
}
}  // namespace

static RileyCudaStatus create_probe_plan(
    RileyCudaContext* context, const RileyCudaGemmConfig* config,
    RileyCudaCublasGemmProbePlan** out_plan,
    RileyCudaErrorInfo* error, bool owned) noexcept {
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
  plan->owned_command_batches=owned;
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

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_create(
    RileyCudaContext* context,const RileyCudaGemmConfig* config,
    RileyCudaCublasGemmProbePlan** out_plan,RileyCudaErrorInfo* error) noexcept {
  return create_probe_plan(context,config,out_plan,error,false);
}
extern "C" RileyCudaStatus riley_cuda_owned_hf_gemm_plan_create(
    RileyCudaContext* context,const RileyCudaGemmConfig* config,
    RileyCudaCublasGemmProbePlan** out_plan,RileyCudaErrorInfo* error) noexcept {
  return create_probe_plan(context,config,out_plan,error,true);
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

// SM89 exact Smol cross-row reuse, qualified against independent native M=1.
// Recipe must come from the independent native-M1 arithmetic screen.
// Every accumulator sees the exact same ordered K indices as that recipe.
template<int L,int Chunk,int Reduction,int R,int C>
__global__ void smol_native_row_reuse_v95(const unsigned short* __restrict__ w,
 const unsigned short* __restrict__ x,unsigned short*y,int n,int k,int rows){
 static_assert(Chunk>0 && L<=32 && R%4==0,"qualified native recipe");
 constexpr int Tile=L*Chunk==576?576:128,WeightStride=Tile+16,Period=L*Chunk;
 extern __shared__ __align__(16) unsigned short data[];
 unsigned short*sx=data;unsigned short*sw=sx+R*Tile;
 float*partial=reinterpret_cast<float*>(sw+C*WeightStride);
 int lane=threadIdx.x,tid=lane+L*threadIdx.y;
 int col=blockIdx.y*C+threadIdx.y,row_base=blockIdx.x*R;
 float accum[R]={};
 for(int base=0;base<k;base+=Tile){
  for(int i=tid*8;i<R*Tile;i+=L*C*8){
   int row=row_base+i/Tile,p=base+i%Tile;
   uint4 value=(row<rows&&p+7<k)?*reinterpret_cast<const uint4*>(x+row*k+p):make_uint4(0,0,0,0);
   *reinterpret_cast<uint4*>(sx+i)=value;
  }
  for(int i=tid*8;i<C*Tile;i+=L*C*8){
   int column=blockIdx.y*C+i/Tile,p=base+i%Tile;
   uint4 value=(column<n&&p+7<k)?*reinterpret_cast<const uint4*>(w+column*k+p):make_uint4(0,0,0,0);
   *reinterpret_cast<uint4*>(sw+(i/Tile)*WeightStride+i%Tile)=value;
  }
  __syncthreads();
  int end=min(k,base+Tile),group=(base/Period)*Period+lane*Chunk;
  if(group+Chunk<=base)group+=Period;
  if(col<n)for(;group<end;group+=Period){
   int begin=max(group,base),limit=min(group+Chunk,end);
   for(int p=begin;p<limit;p++){
    int local=p-base;
    float weight=__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WeightStride+local]));
    #pragma unroll
    for(int r=0;r<R;r++){
     float input=__bfloat162float(__ushort_as_bfloat16(sx[r*Tile+local]));
     accum[r]=__fmaf_rn(weight,input,accum[r]);
    }
   }
  }
  __syncthreads();
 }
 if constexpr(Reduction==1){
  // Padding keeps different output rows from all reading one shared bank.
  #pragma unroll
  for(int r=0;r<R;r++)partial[(threadIdx.y*R+r)*(L+1)+lane]=accum[r];
  __syncthreads();
  for(int r=lane;r<R;r+=L)if(col<n&&row_base+r<rows){
   float value=partial[(threadIdx.y*R+r)*(L+1)];
   #pragma unroll
   for(int i=1;i<L;i++)value=__fadd_rn(value,partial[(threadIdx.y*R+r)*(L+1)+i]);
   y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(1.f,value,0.f)));
  }
 }else{
  if constexpr(Reduction==0){
   #pragma unroll
   for(int off=L/2;off;off/=2){
    #pragma unroll
    for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffffu,accum[r],off,L);if(lane+off<L)accum[r]=__fadd_rn(accum[r],other);}
   }
  }else{
   #pragma unroll
   for(int off=1;off<L;off*=2){
    #pragma unroll
    for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffffu,accum[r],off,L);if(lane%(2*off)==0)accum[r]=__fadd_rn(accum[r],other);}
   }
  }
  if(lane==0&&col<n){
   #pragma unroll
   for(int r=0;r<R;r++)if(row_base+r<rows)y[(row_base+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(1.f,accum[r],0.f)));
  }
 }
}

template<int L,int Chunk,int Reduction,int R,int C>
void launch_smol_native_row_reuse_v95(const unsigned short*w,const unsigned short*x,
 unsigned short*y,int n,int k,int rows,cudaStream_t stream){
 constexpr int Tile=L*Chunk==576?576:128;
 constexpr int bytes=2*(R*Tile+C*(Tile+16))+(Reduction==1?C*R*(L+1)*4:0);
 static_assert(bytes<=48*1024,"prototype avoids opt-in shared memory");
 smol_native_row_reuse_v95<L,Chunk,Reduction,R,C><<<dim3((rows+R-1)/R,(n+C-1)/C),dim3(L,C),bytes,stream>>>(w,x,y,n,k,rows);

}

static bool smol_native_reuse_v95_shape(int n,int k,int rows){
 return rows>=2&&rows<=32&&((k==576&&((n==576&&rows>=6)||(n==192&&rows>=18)||((n==960||n==1536||n==49152)&&rows>=4)))||(n==576&&k==1536&&rows>=12));
}
static void launch_smol_native_reuse_v95(const unsigned short*w,const unsigned short*x,unsigned short*y,int n,int k,int rows,cudaStream_t stream){
 if(k==1536){owned_decode_packed4_transpose_kernel<32,8,8><<<dim3((n+7)/8,(rows+7)/8),dim3(32,8),(8*256+8*272)*2,stream>>>(w,x,y,n,k,rows);return;}
 if(n==49152){
  if(rows>=25)launch_smol_native_row_reuse_v95<8,1,0,32,8>(w,x,y,n,k,rows,stream);
  else if(rows>=9&&rows<=16)launch_smol_native_row_reuse_v95<8,1,0,16,16>(w,x,y,n,k,rows,stream);
  else launch_smol_native_row_reuse_v95<8,1,0,8,8>(w,x,y,n,k,rows,stream);
 }else launch_smol_native_row_reuse_v95<32,18,0,8,8>(w,x,y,n,k,rows,stream);
}

namespace {
template<int R>__global__ void smol_three_projection_v97(const unsigned short*wq,const unsigned short*wk,const unsigned short*wv,const unsigned short*x,unsigned short*yq,unsigned short*yk,unsigned short*yv,int rows){
 constexpr int L=32,C=8,K=576,WS=592;
 extern __shared__ __align__(16) unsigned short data[];unsigned short*sx=data;unsigned short*sw=sx+R*K;
 int lane=threadIdx.x,tid=lane+L*threadIdx.y,row0=blockIdx.x*R,col0=blockIdx.y*C;
 const unsigned short*w=col0<576?wq:(col0<768?wk:wv);unsigned short*y=col0<576?yq:(col0<768?yk:yv);int n=col0<576?576:192;int column0=col0-(col0<576?0:col0<768?576:768);
 for(int i=tid*8;i<R*K;i+=L*C*8){int row=row0+i/K;uint4 value=row<rows?*reinterpret_cast<const uint4*>(x+row*K+i%K):make_uint4(0,0,0,0);*reinterpret_cast<uint4*>(sx+i)=value;}
 for(int i=tid*8;i<C*K;i+=L*C*8){uint4 value=*reinterpret_cast<const uint4*>(w+(column0+i/K)*K+i%K);*reinterpret_cast<uint4*>(sw+(i/K)*WS+i%K)=value;}
 __syncthreads();float accum[R]={};
 #pragma unroll
 for(int j=0;j<18;j++){int p=lane*18+j;float weight=__bfloat162float(__ushort_as_bfloat16(sw[threadIdx.y*WS+p]));
  #pragma unroll
  for(int row=0;row<R;row++)accum[row]=__fmaf_rn(weight,__bfloat162float(__ushort_as_bfloat16(sx[row*K+p])),accum[row]);
 }
 #pragma unroll
 for(int off=16;off;off/=2){
  #pragma unroll
  for(int row=0;row<R;row++){float other=__shfl_down_sync(0xffffffffu,accum[row],off,32);if(lane+off<32)accum[row]=__fadd_rn(accum[row],other);}
 }
 if(lane==0){
  #pragma unroll
  for(int row=0;row<R;row++)if(row0+row<rows)y[(row0+row)*n+column0+threadIdx.y]=__bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(1.f,accum[row],0.f)));
 }
}
template<int R>void launch_hf_smol_qkv_tile(const unsigned short*wq,const unsigned short*wk,const unsigned short*wv,const unsigned short*x,unsigned short*yq,unsigned short*yk,unsigned short*yv,int rows,cudaStream_t stream){
 constexpr int bytes=(R*576+8*592)*2;static_assert(bytes<=48*1024);
 smol_three_projection_v97<R><<<dim3((rows+R-1)/R,120),dim3(32,8),bytes,stream>>>(wq,wk,wv,x,yq,yk,yv,rows);
}
}

namespace {
__device__ unsigned short gate_up_finish_v101(float gate,float up){
 float g=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(1.f,gate,0.f)));
 float u=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(1.f,up,0.f)));
 float activated=__bfloat162float(__float2bfloat16_rn(g/(1.f+expf(-g))));
 return __bfloat16_as_ushort(__float2bfloat16_rn(activated*u));
}
template<int R>__global__ void smol_gate_up_silu_v101(const unsigned short*wg,const unsigned short*wu,const unsigned short*x,unsigned short*y,int rows){
 constexpr int L=32,C=8,K=576,N=1536,WS=592;
 extern __shared__ __align__(16) unsigned short data[];unsigned short*sx=data;unsigned short*sg=sx+R*K;unsigned short*su=sg+C*WS;
 int lane=threadIdx.x,tid=lane+L*threadIdx.y,row0=blockIdx.x*R,col0=blockIdx.y*C;
 for(int i=tid*8;i<R*K;i+=L*C*8){int row=row0+i/K;*reinterpret_cast<uint4*>(sx+i)=row<rows?*reinterpret_cast<const uint4*>(x+row*K+i%K):make_uint4(0,0,0,0);}
 for(int i=tid*8;i<C*K;i+=L*C*8){*reinterpret_cast<uint4*>(sg+(i/K)*WS+i%K)=*reinterpret_cast<const uint4*>(wg+(col0+i/K)*K+i%K);*reinterpret_cast<uint4*>(su+(i/K)*WS+i%K)=*reinterpret_cast<const uint4*>(wu+(col0+i/K)*K+i%K);}
 __syncthreads();float g[R]={},u[R]={};
 #pragma unroll
 for(int j=0;j<18;j++){int p=lane*18+j;float gw=__bfloat162float(__ushort_as_bfloat16(sg[threadIdx.y*WS+p]));float uw=__bfloat162float(__ushort_as_bfloat16(su[threadIdx.y*WS+p]));
  #pragma unroll
  for(int row=0;row<R;row++){float v=__bfloat162float(__ushort_as_bfloat16(sx[row*K+p]));g[row]=__fmaf_rn(gw,v,g[row]);u[row]=__fmaf_rn(uw,v,u[row]);}
 }
 #pragma unroll
 for(int off=16;off;off/=2){
  #pragma unroll
  for(int row=0;row<R;row++){float gg=__shfl_down_sync(0xffffffffu,g[row],off,32);float uu=__shfl_down_sync(0xffffffffu,u[row],off,32);if(lane+off<32){g[row]=__fadd_rn(g[row],gg);u[row]=__fadd_rn(u[row],uu);}}
 }
 if(lane==0){
  #pragma unroll
  for(int row=0;row<R;row++)if(row0+row<rows)y[(row0+row)*N+col0+threadIdx.y]=gate_up_finish_v101(g[row],u[row]);
 }
}
template<int R>void v102_launch(const unsigned short*wg,const unsigned short*wu,const unsigned short*x,unsigned short*y,int rows,cudaStream_t s){constexpr int bytes=(R*576+2*8*592)*2;static_assert(bytes<=48*1024);smol_gate_up_silu_v101<R><<<dim3((rows+R-1)/R,192),dim3(32,8),bytes,s>>>(wg,wu,x,y,rows);}
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

// SM89 component148: unchanged lane-strided M1 FMAs, no shared-input barriers.
template<int L,int R,int C>__global__ void owned_small_row_global_v148(const unsigned short*__restrict__ w,const unsigned short*__restrict__ x,unsigned short*y,int n,int k,int rows){
 int lane=threadIdx.x,col=blockIdx.x*C+threadIdx.y,row0=blockIdx.y*R;float acc[R]={};
 for(int p=lane;p<k;p+=L){float ww=col<n?__bfloat162float(__ushort_as_bfloat16(w[col*k+p])):0.f;
  #pragma unroll
  for(int r=0;r<R;r++)if(row0+r<rows)acc[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(x[(row0+r)*k+p])),acc[r]);
 }
 #pragma unroll
 for(int d=L/2;d;d/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float v=__shfl_down_sync(0xffffffff,acc[r],d,L);if(lane+d<L)acc[r]=__fadd_rn(acc[r],v);}
 }
 if(lane==0&&col<n){
  #pragma unroll
  for(int r=0;r<R;r++)if(row0+r<rows)y[(row0+r)*n+col]=__bfloat16_as_ushort(__float2bfloat16_rn(acc[r]));
 }
}

// SM80/cuBLAS12.4.5 native K/V: lane-strided32 serial FFMA, descending32 tree.
// Reuse each W value across8 requests without changing any row's arithmetic.
template<int R>__global__ void owned_sm80_native_kv_rows(const uint16_t*__restrict__ w,const uint16_t*__restrict__ x,uint16_t*y,int rows){
 int lane=threadIdx.x,col=blockIdx.x*4+threadIdx.y,first=blockIdx.y*R;float a[R]={};
 #pragma unroll 16
 for(int j=0;j<80;j++){int k=lane+j*32;float ww=__bfloat162float(__ushort_as_bfloat16(w[col*2560+k]));
  #pragma unroll
  for(int r=0;r<R;r++)if(first+r<rows)a[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(x[(first+r)*2560+k])),a[r]);
 }
 #pragma unroll
 for(int d=16;d;d/=2){
  #pragma unroll
  for(int r=0;r<R;r++){float other=__shfl_down_sync(0xffffffffu,a[r],d);if(lane<d)a[r]=__fadd_rn(a[r],other);}
 }
 if(lane==0){
  #pragma unroll
  for(int r=0;r<R;r++)if(first+r<rows)y[(first+r)*1024+col]=__bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(1.f,a[r],0.f)));
 }
}

__device__ __forceinline__ void owned_sm80_kv_copy16(void*dst,const void*src,bool valid){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 unsigned sm=static_cast<unsigned>(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(sm),"l"(src),"r"(valid?16:0));
#else
 if(valid)*reinterpret_cast<uint4*>(dst)=*reinterpret_cast<const uint4*>(src);
#endif
}
template<int C,int R>__device__ __forceinline__ void owned_sm80_kv_stage(uint16_t*dst,const uint16_t*w,const uint16_t*x,int rows,int first,int col,int it){for(int i=threadIdx.x+32*threadIdx.y;i<(C+R)*32;i+=C*32){int r=i/32,k=i%32*8;bool valid=it<10&&(r<C||first+r-C<rows);const uint16_t*src=valid?(r<C?w+(col+r)*2560+it*256+k:x+(first+r-C)*2560+it*256+k):w;owned_sm80_kv_copy16(dst+r*256+k,src,valid);}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 asm volatile("cp.async.commit_group;");
#endif
}
template<int C,int R>__global__ void owned_sm80_tiled_kv(const uint16_t*w,const uint16_t*x,uint16_t*y,int rows){

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 constexpr int Stage=(C+R)*256;__shared__ __align__(32)uint16_t sh[Stage*2];int lane=threadIdx.x,col=blockIdx.x*C+threadIdx.y,first=blockIdx.y*R;float a[R]={};owned_sm80_kv_stage<C,R>(sh,w,x,rows,first,blockIdx.x*C,0);owned_sm80_kv_stage<C,R>(sh+Stage,w,x,rows,first,blockIdx.x*C,1);
 for(int it=0;it<10;it++){asm volatile("cp.async.wait_group 1;");__syncthreads();auto*t=sh+it%2*Stage;
 #pragma unroll
 for(int j=0;j<8;j++){int d=lane+j*32;float ww=__bfloat162float(__ushort_as_bfloat16(t[threadIdx.y*256+d]));
 #pragma unroll
 for(int r=0;r<R;r++)if(first+r<rows)a[r]=__fmaf_rn(ww,__bfloat162float(__ushort_as_bfloat16(t[(C+r)*256+d])),a[r]);}
 __syncthreads();owned_sm80_kv_stage<C,R>(t,w,x,rows,first,blockIdx.x*C,it+2);}
 asm volatile("cp.async.wait_group 0;");
 #pragma unroll
 for(int d=16;d;d/=2){
 #pragma unroll
 for(int r=0;r<R;r++){float b=__shfl_down_sync(0xffffffff,a[r],d);if(lane<d)a[r]=__fadd_rn(a[r],b);}}
 if(lane==0){
 #pragma unroll
 for(int r=0;r<R;r++)if(first+r<rows)y[(first+r)*1024+col]=__bfloat16_as_ushort(__float2bfloat16_rn(__fmaf_rn(1.f,a[r],0.f)));}
#endif

}

// Native SM80 Down: split-K2, two K32-interleaved FP32 accumulators per
// split, FP32 sliced sum, BF16 partials, FP32 split sum, BF16 output.
// Four asynchronous K64 stages reuse each weight tile across16 requests.
namespace {
__device__ __forceinline__ void sm80_down_copy16(void*dst,const void*src,bool valid){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
unsigned sm=static_cast<unsigned>(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(sm),"l"(src),"r"(valid?16:0));
#endif
}
template<int Warps> __device__ __forceinline__ void sm80_down_stage(__nv_bfloat16*stage,const __nv_bfloat16*w,const __nv_bfloat16*x,int rows,int col,int first,int split,int iteration){
 constexpr int A=Warps*16*64,B=16*64;int kbase=split*4864+iteration*64;
 for(int i=threadIdx.x;i<(A+B)/8;i+=Warps*32){bool isA=i<A/8;int offset=isA?i*8:(i-A/8)*8,row=offset/64,k=offset%64;bool valid=iteration<76&&(isA||first+row<rows);const __nv_bfloat16*src=valid?(isA?w+(col+row)*9728+kbase+k:x+(first+row)*9728+kbase+k):w;sm80_down_copy16(stage+i*8,src,valid);}
 asm volatile("cp.async.commit_group;");
}
template<int Warps> __global__ void sm80_down_native_sliced(const __nv_bfloat16*w,const __nv_bfloat16*x,__nv_bfloat16*parts,int rows){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 using namespace nvcuda;constexpr int N=Warps*16,A=N*64,Stage=(N+16)*64;
 __shared__ __align__(32) __nv_bfloat16 shared[Stage*4];
 int col=blockIdx.x*N,first=blockIdx.y*16,split=blockIdx.z,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc[2];wmma::fill_fragment(acc[0],0.f);wmma::fill_fragment(acc[1],0.f);
 #pragma unroll
 for(int st=0;st<4;st++)sm80_down_stage<Warps>(shared+st*Stage,w,x,rows,col,first,split,st);
 for(int iteration=0;iteration<76;iteration++){
  asm volatile("cp.async.wait_group 3;");__syncthreads();auto*tile=shared+(iteration%4)*Stage;
  #pragma unroll
  for(int j=0;j<4;j++){wmma::load_matrix_sync(a,tile+warp*16*64+j*16,64);wmma::load_matrix_sync(b,tile+A+j*16,64);wmma::mma_sync(acc[j/2],a,b,acc[j/2]);}
  __syncthreads();sm80_down_stage<Warps>(tile,w,x,rows,col,first,split,iteration+4);
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 for(int i=0;i<acc[0].num_elements;i++)acc[0].x[i]=__fadd_rn(acc[0].x[i],acc[1].x[i]);
 float*out=reinterpret_cast<float*>(shared)+warp*256;wmma::store_matrix_sync(out,acc[0],16,wmma::mem_col_major);__syncwarp();
 for(int i=lane;i<256;i+=32){int r=first+i/16;if(r<rows)parts[(split*rows+r)*2560+col+warp*16+i%16]=__float2bfloat16_rn(out[i]);}

#endif
}
__global__ void sm80_down_reduce(const __nv_bfloat16*p,__nv_bfloat16*y,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(p[i]),__bfloat162float(p[count+i])));}
}


// Decode v9 preserves both K32 accumulator chains in independent CTAs.
template<int W,int R,int P,int S>__device__ __forceinline__ void sm80_v9_down_stage(__nv_bfloat16*dst,const __nv_bfloat16*w,const __nv_bfloat16*x,int rows,int col,int first,int part,int it){
 constexpr int N=W*16,L=32+P;int kbase=(part/2)*4864+(part%2)*32+it*64;
 for(int i=threadIdx.x;i<(N+R)*4;i+=W*32){int row=i/4,k=(i%4)*8;bool a=row<N;int xr=row-N;bool valid=it<76&&(a||first+xr<rows);const __nv_bfloat16*src=valid?(a?w+(col+row)*9728+kbase+k:x+(first+xr)*9728+kbase+k):w;sm80_down_copy16(dst+row*L+k,src,valid);}asm volatile("cp.async.commit_group;");
}
template<int W,int R,int P,int S>__global__ void sm80_v9_down(const __nv_bfloat16*w,const __nv_bfloat16*x,float*parts,int rows){
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 using namespace nvcuda;constexpr int N=W*16,L=32+P,Stage=(N+R)*L,RB=R/16;__shared__ __align__(32)__nv_bfloat16 shared[Stage*S];int col=blockIdx.x*N,first=blockIdx.y*R,part=blockIdx.z,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>c[RB];
 #pragma unroll
 for(int rb=0;rb<RB;rb++)wmma::fill_fragment(c[rb],0.f);
 #pragma unroll
 for(int st=0;st<S;st++)sm80_v9_down_stage<W,R,P,S>(shared+st*Stage,w,x,rows,col,first,part,st);
 for(int it=0;it<76;it++){asm volatile("cp.async.wait_group %0;"::"n"(S-1));__syncthreads();auto*t=shared+(it%S)*Stage;
 #pragma unroll
 for(int j=0;j<2;j++){wmma::load_matrix_sync(a,t+warp*16*L+j*16,L);
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::load_matrix_sync(b,t+(N+rb*16)*L+j*16,L);wmma::mma_sync(c[rb],a,b,c[rb]);}}
 __syncthreads();sm80_v9_down_stage<W,R,P,S>(t,w,x,rows,col,first,part,it+S);}
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 #pragma unroll
 for(int rb=0;rb<RB;rb++){float*out=reinterpret_cast<float*>(shared)+(warp*RB+rb)*256;wmma::store_matrix_sync(out,c[rb],16,wmma::mem_col_major);__syncwarp();for(int i=lane;i<256;i+=32){int row=first+rb*16+i/16;if(row<rows)parts[(part*rows+row)*2560+col+warp*16+i%16]=out[i];}}
#endif
}
__global__ void sm80_v9_down_reduce(const float*p,__nv_bfloat16*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float a=__bfloat162float(__float2bfloat16_rn(__fadd_rn(p[i],p[n+i])));float b=__bfloat162float(__float2bfloat16_rn(__fadd_rn(p[2*n+i],p[3*n+i])));y[i]=__float2bfloat16_rn(__fadd_rn(a,b));}}

// Decode v8: qualified native K32 sliced arithmetic, padded shared strides.
template<int W,int R,int P,int S>__device__ __forceinline__ void sm80_v8_down_stage(__nv_bfloat16*dst,const __nv_bfloat16*w,const __nv_bfloat16*x,int rows,int col,int first,int split,int it){
 constexpr int N=W*16,L=64+P;int kbase=split*4864+it*64;
 for(int i=threadIdx.x;i<(N+R)*8;i+=W*32){int row=i/8,k=(i%8)*8;bool a=row<N;int xr=row-N;bool valid=it<76&&(a||first+xr<rows);const __nv_bfloat16*src=valid?(a?w+(col+row)*9728+kbase+k:x+(first+xr)*9728+kbase+k):w;sm80_down_copy16(dst+row*L+k,src,valid);}
 asm volatile("cp.async.commit_group;");
}
template<int W,int R,int P,int S>__global__ void sm80_v8_down(const __nv_bfloat16*w,const __nv_bfloat16*x,__nv_bfloat16*parts,int rows){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 using namespace nvcuda;constexpr int N=W*16,L=64+P,Stage=(N+R)*L,RB=R/16;
 __shared__ __align__(32)__nv_bfloat16 shared[Stage*S];int col=blockIdx.x*N,first=blockIdx.y*R,split=blockIdx.z,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;
 wmma::fragment<wmma::accumulator,16,16,16,float>c[RB][2];
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::fill_fragment(c[rb][0],0.f);wmma::fill_fragment(c[rb][1],0.f);}
 #pragma unroll
 for(int st=0;st<S;st++)sm80_v8_down_stage<W,R,P,S>(shared+st*Stage,w,x,rows,col,first,split,st);
 for(int it=0;it<76;it++){
  asm volatile("cp.async.wait_group %0;"::"n"(S-1));__syncthreads();auto*t=shared+(it%S)*Stage;
  #pragma unroll
  for(int j=0;j<4;j++){wmma::load_matrix_sync(a,t+warp*16*L+j*16,L);
   #pragma unroll
   for(int rb=0;rb<RB;rb++){wmma::load_matrix_sync(b,t+(N+rb*16)*L+j*16,L);wmma::mma_sync(c[rb][j/2],a,b,c[rb][j/2]);}}
  __syncthreads();sm80_v8_down_stage<W,R,P,S>(t,w,x,rows,col,first,split,it+S);
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 #pragma unroll
 for(int rb=0;rb<RB;rb++){
  for(int i=0;i<c[rb][0].num_elements;i++)c[rb][0].x[i]=__fadd_rn(c[rb][0].x[i],c[rb][1].x[i]);
  float*out=reinterpret_cast<float*>(shared)+(warp*RB+rb)*256;wmma::store_matrix_sync(out,c[rb][0],16,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<256;i+=32){int row=first+rb*16+i/16;if(row<rows)parts[(split*rows+row)*2560+col+warp*16+i%16]=__float2bfloat16_rn(out[i]);}
 }

#endif
}

template<int W,int R,int P,int S>__device__ __forceinline__ void sm80_v8_gate_stage(__nv_bfloat16*dst,const __nv_bfloat16*w,const __nv_bfloat16*x,int rows,int col,int first,int split,int it){
 constexpr int N=W*16,L=64+P;int kbase=it*64;
 for(int i=threadIdx.x;i<(N+R)*8;i+=W*32){int row=i/8,k=(i%8)*8;bool a=row<N;int xr=row-N;bool valid=it<40&&(a||first+xr<rows);const __nv_bfloat16*src=valid?(a?w+(col+row)*2560+kbase+k:x+(first+xr)*2560+kbase+k):w;sm80_down_copy16(dst+row*L+k,src,valid);}
 asm volatile("cp.async.commit_group;");
}
template<int W,int R,int P,int S>__global__ void sm80_v8_gate(const __nv_bfloat16*w,const __nv_bfloat16*x,__nv_bfloat16*parts,int rows){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 using namespace nvcuda;constexpr int N=W*16,L=64+P,Stage=(N+R)*L,RB=R/16;
 __shared__ __align__(32)__nv_bfloat16 shared[Stage*S];int col=blockIdx.x*N,first=blockIdx.y*R,split=blockIdx.z,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;
 wmma::fragment<wmma::accumulator,16,16,16,float>c[RB][2];
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::fill_fragment(c[rb][0],0.f);wmma::fill_fragment(c[rb][1],0.f);}
 #pragma unroll
 for(int st=0;st<S;st++)sm80_v8_gate_stage<W,R,P,S>(shared+st*Stage,w,x,rows,col,first,split,st);
 for(int it=0;it<40;it++){
  asm volatile("cp.async.wait_group %0;"::"n"(S-1));__syncthreads();auto*t=shared+(it%S)*Stage;
  #pragma unroll
  for(int j=0;j<4;j++){wmma::load_matrix_sync(a,t+warp*16*L+j*16,L);
   #pragma unroll
   for(int rb=0;rb<RB;rb++){wmma::load_matrix_sync(b,t+(N+rb*16)*L+j*16,L);wmma::mma_sync(c[rb][j/2],a,b,c[rb][j/2]);}}
  __syncthreads();sm80_v8_gate_stage<W,R,P,S>(t,w,x,rows,col,first,split,it+S);
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 #pragma unroll
 for(int rb=0;rb<RB;rb++){
  for(int i=0;i<c[rb][0].num_elements;i++)c[rb][0].x[i]=__fadd_rn(c[rb][0].x[i],c[rb][1].x[i]);
  float*out=reinterpret_cast<float*>(shared)+(warp*RB+rb)*256;wmma::store_matrix_sync(out,c[rb][0],16,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<256;i+=32){int row=first+rb*16+i/16;if(row<rows)parts[row*9728+col+warp*16+i%16]=__float2bfloat16_rn(__fmaf_rn(1.f,out[i],0.f));}
 }

#endif
}


template<int W,int R,int P,int S>__device__ __forceinline__ void sm80_batched_lm_head_stage(__nv_bfloat16*dst,const __nv_bfloat16*w,const __nv_bfloat16*x,int rows,int col,int first,int split,int it){
 constexpr int N=W*16,L=64+P;int kbase=it*64;
 for(int i=threadIdx.x;i<(N+R)*8;i+=W*32){int row=i/8,k=(i%8)*8;bool a=row<N;int xr=row-N;bool valid=it<40&&(a||first+xr<rows);const __nv_bfloat16*src=valid?(a?w+(col+row)*2560+kbase+k:x+(first+xr)*2560+kbase+k):w;sm80_down_copy16(dst+row*L+k,src,valid);}
 asm volatile("cp.async.commit_group;");
}
template<int W,int R,int P,int S>__global__ void sm80_batched_lm_head(const __nv_bfloat16*w,const __nv_bfloat16*x,__nv_bfloat16*parts,int rows){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 using namespace nvcuda;constexpr int N=W*16,L=64+P,Stage=(N+R)*L,RB=R/16;
 __shared__ __align__(32)__nv_bfloat16 shared[Stage*S];int col=blockIdx.x*N,first=blockIdx.y*R,split=blockIdx.z,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;
 wmma::fragment<wmma::accumulator,16,16,16,float>c[RB][2];
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::fill_fragment(c[rb][0],0.f);wmma::fill_fragment(c[rb][1],0.f);}
 #pragma unroll
 for(int st=0;st<S;st++)sm80_batched_lm_head_stage<W,R,P,S>(shared+st*Stage,w,x,rows,col,first,split,st);
 for(int it=0;it<40;it++){
  asm volatile("cp.async.wait_group %0;"::"n"(S-1));__syncthreads();auto*t=shared+(it%S)*Stage;
  #pragma unroll
  for(int j=0;j<4;j++){wmma::load_matrix_sync(a,t+warp*16*L+j*16,L);
   #pragma unroll
   for(int rb=0;rb<RB;rb++){wmma::load_matrix_sync(b,t+(N+rb*16)*L+j*16,L);wmma::mma_sync(c[rb][j/2],a,b,c[rb][j/2]);}}
  __syncthreads();sm80_batched_lm_head_stage<W,R,P,S>(t,w,x,rows,col,first,split,it+S);
 }
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 #pragma unroll
 for(int rb=0;rb<RB;rb++){
  for(int i=0;i<c[rb][0].num_elements;i++)c[rb][0].x[i]=__fadd_rn(c[rb][0].x[i],c[rb][1].x[i]);
  float*out=reinterpret_cast<float*>(shared)+(warp*RB+rb)*256;wmma::store_matrix_sync(out,c[rb][0],16,wmma::mem_col_major);__syncwarp();
  for(int i=lane;i<256;i+=32){int row=first+rb*16+i/16;if(row<rows)parts[row*151936+col+warp*16+i%16]=__float2bfloat16_rn(__fmaf_rn(1.f,out[i],0.f));}
 }

#endif
}



template<int W,int R,int P,int S>__device__ __forceinline__ void gateup_stage(__nv_bfloat16*dst,const __nv_bfloat16*wg,const __nv_bfloat16*wu,const __nv_bfloat16*x,int rows,int col,int first,int it){
 constexpr int N=W*16,L=64+P;for(int i=threadIdx.x;i<(N+R)*8;i+=W*32){int row=i/8,k=i%8*8;bool weight=row<N,valid=it<40&&(weight||first+row-N<rows);const __nv_bfloat16*src=wg;if(valid)src=weight?((row<N/2?wg:wu)+(col+row%(N/2))*2560+it*64+k):(x+(first+row-N)*2560+it*64+k);sm80_down_copy16(dst+row*L+k,src,valid);}asm volatile("cp.async.commit_group;");
}
template<int W,int R,int P,int S>__global__ void sm80_fused_gateup(const __nv_bfloat16*wg,const __nv_bfloat16*wu,const __nv_bfloat16*x,__nv_bfloat16*y,int rows){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 using namespace nvcuda;constexpr int N=W*16,L=64+P,Stage=(N+R)*L,RB=R/16;__shared__ __align__(32)__nv_bfloat16 shared[Stage*S];auto results=reinterpret_cast<float (*)[RB][256]>(shared);int col=blockIdx.x*(N/2),first=blockIdx.y*R,warp=threadIdx.x/32,lane=threadIdx.x%32;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>c[RB][2];
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::fill_fragment(c[rb][0],0.f);wmma::fill_fragment(c[rb][1],0.f);}
 #pragma unroll
 for(int st=0;st<S;st++)gateup_stage<W,R,P,S>(shared+st*Stage,wg,wu,x,rows,col,first,st);
 for(int it=0;it<40;it++){asm volatile("cp.async.wait_group %0;"::"n"(S-1));__syncthreads();auto*t=shared+it%S*Stage;
 #pragma unroll
 for(int j=0;j<4;j++){wmma::load_matrix_sync(a,t+warp*16*L+j*16,L);
 #pragma unroll
 for(int rb=0;rb<RB;rb++){wmma::load_matrix_sync(b,t+(N+rb*16)*L+j*16,L);wmma::mma_sync(c[rb][j/2],a,b,c[rb][j/2]);}}
 __syncthreads();gateup_stage<W,R,P,S>(t,wg,wu,x,rows,col,first,it+S);}
 asm volatile("cp.async.wait_group 0;");__syncthreads();
 #pragma unroll
 for(int rb=0;rb<RB;rb++){for(int i=0;i<c[rb][0].num_elements;i++)c[rb][0].x[i]=__fadd_rn(c[rb][0].x[i],c[rb][1].x[i]);wmma::store_matrix_sync(results[warp][rb],c[rb][0],16,wmma::mem_col_major);}
 __syncthreads();if(warp<W/2){
 #pragma unroll
 for(int rb=0;rb<RB;rb++)for(int i=lane;i<256;i+=32){int row=first+rb*16+i/16;if(row<rows){float g=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(1.f,results[warp][rb][i],0.f)));float u=__bfloat162float(__float2bfloat16_rn(__fmaf_rn(1.f,results[warp+W/2][rb][i],0.f)));float activated=__bfloat162float(__float2bfloat16_rn(g/(1.f+expf(-g))));y[row*9728+col+warp*16+i%16]=__float2bfloat16_rn(__fmul_rn(activated,u));}}}
#endif
}

extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_execute_rows(
    RileyCudaCublasGemmProbePlan* plan, const RileyCudaBufferSpan* input,
    const RileyCudaBufferSpan* weight, const RileyCudaBufferSpan* output,
    uint64_t rows, uint32_t grouped_decode, RileyCudaStream* stream, RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  // Explicit mode3 owns the separately qualified R2 Gate/Up route.
  const bool merged_sm80_profile=grouped_decode==12;
  const bool prefill_sm80_v7_profile=grouped_decode==11 || merged_sm80_profile;
  if(prefill_sm80_v7_profile)grouped_decode=0;
  const bool native_prefill_sm80_profile=grouped_decode==10;
  const bool global_r2_profile=grouped_decode==4;
  const bool native_kv_sm80_profile=(grouped_decode==8||grouped_decode==9||grouped_decode==10);
  const bool projection_sm80_profile=grouped_decode==7||native_kv_sm80_profile;
  const bool bounded_sm80_profile=grouped_decode==6||projection_sm80_profile;
  const bool smallrows_profile=grouped_decode==5;
  const bool exact_r2_profile=grouped_decode==3||global_r2_profile||smallrows_profile;
  if(exact_r2_profile || bounded_sm80_profile)grouped_decode=2;
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

  if(bounded_sm80_profile && !plan->owned_sm80_gateup_rows_qualified)
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"bounded SM80 Gate/Up requires independently qualified hardware and workspace configuration");
  if(grouped_decode > 2 || (grouped_decode && !plan->owned_command_batches))
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"grouped decode requires an owned plan");
  if(rows==0 || rows>plan->config.m) return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"requested GEMM rows exceed the cold plan");
  if(projection_sm80_profile && rows>=2 && rows<=32 && plan->config.n==2560 &&
      (plan->config.k==4096 || plan->config.k==9728) &&
      (plan->sm80_projection_rows==0 || plan->decode_lt==nullptr || plan->sm80_projection_workspace==nullptr))
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_STATE,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"bounded projection recipe was not cold-prepared");
  if(prefill_sm80_v7_profile && (rows<(merged_sm80_profile?1u:161u) || (merged_sm80_profile && (plan->config.n!=19456 || plan->config.k!=2560 || rows>32)) || rows>plan->prefill_max_rows || plan->prefill_lt==nullptr || plan->prefill_workspace==nullptr))
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified cold prefill row extent required");
  ResolvedSpan spans[kMaximumGemmBuffers]{};
  RileyCudaStatus status = resolve_attention_span(
      input, RILEY_CUDA_DTYPE_BF16, rows*plan->config.k*2, &spans[0], error,
      "input span dtype must be BF16");
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_attention_span(weight, RILEY_CUDA_DTYPE_BF16,
                                plan->weight_bytes, &spans[1], error,
                                "weight span dtype must be BF16");
  }
  if (status == RILEY_CUDA_STATUS_SUCCESS) {
    status = resolve_attention_span(output, RILEY_CUDA_DTYPE_BF16,
                                rows*plan->config.n*2, &spans[2], error,
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
  if(projection_sm80_profile && plan->sm80_projection_workspace!=nullptr && !uses.add(plan->sm80_projection_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"private projection workspace use registration failed");
  if(native_prefill_sm80_profile && plan->owned_sm80_full_projection_rows && !uses.add(plan->sm80_full_projection_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"full projection workspace use registration failed");
  if(prefill_sm80_v7_profile && !uses.add(plan->prefill_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"private prefill workspace registration failed");
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
    // V81 measured Q4 crossover, extending only three profitable shapes.
    // Unknown environments/alignment and every existing R17+ route stay intact.
    const int quad_variant=(grouped_decode==2 && rows>=3 && rows<=32 && plan->owned_attention_extended_qualified &&
       ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(15))==0)
       ?owned_quad_v107_variant(static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),static_cast<int>(rows)):-1;
    // Existing mode3 retains its Gate/Up implementation. Only explicit mode4
    // adds R2 Down/output weight reuse; common guards keep prefill dispatch lean.
    if(prefill_sm80_v7_profile){
      status=cublas_probe_error(cublasLtMatmul(plan->prefill_lt,plan->prefill_op,&alpha,
        spans[1].data,plan->prefill_weight,spans[0].data,plan->prefill_input[rows],&beta,
        spans[2].data,plan->prefill_output[rows],spans[2].data,plan->prefill_output[rows],
        &plan->prefill_algo,plan->prefill_workspace->device_data,plan->prefill_workspace_bytes,stream->stream),
        error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    }else if(native_prefill_sm80_profile && plan->owned_sm80_full_projection_rows && rows>=2 && rows<=32 && plan->config.n==2560 && (plan->config.k==4096||plan->config.k==9728)){
      status=cublas_probe_error(cublasLtMatmul(plan->decode_lt,plan->decode_op,&alpha,spans[1].data,plan->decode_weight,spans[0].data,plan->decode_input[rows],&beta,spans[2].data,plan->decode_output[rows],spans[2].data,plan->decode_output[rows],&plan->decode_algo,plan->sm80_full_projection_workspace->device_data,plan->sm80_full_projection_workspace_bytes,stream->stream),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    }else if(exact_r2_profile && grouped_decode==2 && rows==2 && plan->owned_attention_extended_qualified &&
       ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(15))==0 &&
       ((plan->config.n==9728 && plan->config.k==2560) ||
        (global_r2_profile && plan->config.n==2560 && (plan->config.k==4096 || plan->config.k==9728)))){
      auto*w=static_cast<const unsigned short*>(spans[1].data);auto*x=static_cast<const unsigned short*>(spans[0].data);auto*y=static_cast<unsigned short*>(spans[2].data);
      if(plan->config.n==9728){owned_small_row_quad_v109<16,2,4><<<dim3(1,(9728+15)/16),dim3(16,4),(2*256+16*272)*2,stream->stream>>>(w,x,y,9728,2560,2);}
      else if(plan->config.k==9728){owned_small_row_global_v148<32,2,4><<<dim3((2560+3)/4,1),dim3(32,4),0,stream->stream>>>(w,x,y,2560,9728,2);}
      else{owned_small_row_global_v148<16,2,8><<<dim3((2560+7)/8,1),dim3(16,8),0,stream->stream>>>(w,x,y,2560,4096,2);}
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    }else if(grouped_decode==2 && plan->owned_attention_extended_qualified &&
       ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(15))==0 &&
       ((smallrows_profile && rows==3 && plan->config.n==2560 && (plan->config.k==4096||plan->config.k==9728)) ||
        (smallrows_profile && rows==2 && plan->config.n==4096 && plan->config.k==4096))){
      // Small exact row groups: reuse each W load across actual requests. The
      // original per-lane FMA order, reduction tree and BF16 rounding stay intact.
      auto*w=static_cast<const unsigned short*>(spans[1].data);auto*x=static_cast<const unsigned short*>(spans[0].data);auto*y=static_cast<unsigned short*>(spans[2].data);
      if(rows==2){owned_small_row_global_v148<16,2,8><<<dim3((4096+7)/8,1),dim3(16,8),0,stream->stream>>>(w,x,y,4096,4096,2);}
      else if(plan->config.k==9728){owned_small_row_global_v148<32,3,4><<<dim3((2560+3)/4,1),dim3(32,4),0,stream->stream>>>(w,x,y,2560,9728,3);}
      else{owned_small_row_global_v148<16,3,8><<<dim3((2560+7)/8,1),dim3(16,8),0,stream->stream>>>(w,x,y,2560,4096,3);}
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    }else if(quad_variant>=0){
      launch_owned_quad_v107(quad_variant,static_cast<const unsigned short*>(spans[1].data),static_cast<const unsigned short*>(spans[0].data),static_cast<unsigned short*>(spans[2].data),static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),static_cast<int>(rows),stream->stream);
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    }else if(grouped_decode==2 && plan->owned_attention_extended_qualified &&
       rows>=9 && rows<=16 && plan->config.k==2560 &&
       (plan->config.n==9728 || plan->config.n==151936 || plan->config.n==4096) &&
       ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(15))==0) {
      owned_decode_column_pair_kernel<16,16,8,true><<<dim3((rows+15)/16,(plan->config.n+15)/16),dim3(16,8),(16*256+16*272)*2,stream->stream>>>(
          static_cast<const unsigned short*>(spans[1].data),static_cast<const unsigned short*>(spans[0].data),static_cast<unsigned short*>(spans[2].data),
          static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),static_cast<int>(rows));
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if (grouped_decode==2 && plan->owned_attention_extended_qualified &&
        smol_native_reuse_v95_shape(static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),static_cast<int>(rows)) &&
        ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(15))==0) {
      launch_smol_native_reuse_v95(static_cast<const unsigned short*>(spans[1].data),static_cast<const unsigned short*>(spans[0].data),static_cast<unsigned short*>(spans[2].data),static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),static_cast<int>(rows),stream->stream);
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if (grouped_decode==2 && rows>=2 && rows<=32 &&
        plan->owned_weight_reuse_qualified &&
        ((plan->config.n==11008 && plan->config.k==2048) ||
         (plan->config.n==2048 && plan->config.k==11008) ||
         (plan->config.n==151936 && plan->config.k==2048) ||
         // Shape-specific measured crossover; C1 and unsupported spans
         // retain their original cuBLAS path. Q/K/V retain L16 arithmetic.
         ((reinterpret_cast<uintptr_t>(spans[0].data)&uintptr_t(15))==0 &&
          (reinterpret_cast<uintptr_t>(spans[1].data)&uintptr_t(15))==0 &&
          ((plan->config.k==4096 && plan->config.n==4096 && rows>=3) ||
           (plan->config.k==4096 && plan->config.n==1024 && rows>=4) ||
           (plan->config.k==4096 && (plan->config.n==14336||plan->config.n==32768)) ||
           (plan->config.n==4096 && plan->config.k==14336) ||
           (plan->config.n==9728 && plan->config.k==2560 && rows>=3) ||
           (plan->config.n==2560 && plan->config.k==9728 && rows>=4) ||
           (plan->config.n==151936 && plan->config.k==2560) ||
           (plan->config.n==4096 && plan->config.k==2560 && rows>=2) ||
           (plan->config.n==1024 && plan->config.k==2560 && rows>=3) ||
           (plan->config.n==2560 && plan->config.k==4096 && rows>=4))))) {
      launch_owned_decode_weight_reuse(
          static_cast<const unsigned short*>(spans[1].data),
          static_cast<const unsigned short*>(spans[0].data),
          static_cast<unsigned short*>(spans[2].data),
          static_cast<int>(plan->config.n),static_cast<int>(plan->config.k),
          static_cast<int>(rows),stream->stream);
      const cudaError_t launch_result=cudaGetLastError();
      if(launch_result!=cudaSuccess)status=runtime_error(launch_result,error,
          RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(native_kv_sm80_profile && plan->owned_sm80_gateup_rows_qualified &&
        rows>=2 && rows<=32 && plan->config.n==1024 && plan->config.k==2560) {
      const auto*w=static_cast<const uint16_t*>(spans[1].data);
      const auto*x=static_cast<const uint16_t*>(spans[0].data);
      auto*y=static_cast<uint16_t*>(spans[2].data);
      const auto count=static_cast<unsigned>(rows);
      if(native_prefill_sm80_profile&&rows>=5){
        if(rows<=12)owned_sm80_tiled_kv<8,4><<<dim3(128,(count+3)/4),dim3(32,8),0,stream->stream>>>(w,x,y,count);
        else if(rows<=23)owned_sm80_tiled_kv<4,8><<<dim3(256,(count+7)/8),dim3(32,4),0,stream->stream>>>(w,x,y,count);
        else owned_sm80_tiled_kv<8,8><<<dim3(128,(count+7)/8),dim3(32,8),0,stream->stream>>>(w,x,y,count);
      }else{
      if(rows<=4)owned_sm80_native_kv_rows<2><<<dim3(256,(count+1)/2),dim3(32,4),0,stream->stream>>>(w,x,y,count);
      else if(rows<=8)owned_sm80_native_kv_rows<4><<<dim3(256,(count+3)/4),dim3(32,4),0,stream->stream>>>(w,x,y,count);
      else owned_sm80_native_kv_rows<8><<<dim3(256,(count+7)/8),dim3(32,4),0,stream->stream>>>(w,x,y,count);
      }
      const cudaError_t result=cudaGetLastError();
      if(result!=cudaSuccess)status=runtime_error(result,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(projection_sm80_profile && plan->owned_sm80_gateup_rows_qualified &&
        rows>=2 && rows<=32 && plan->config.n==4096 && plan->config.k==2560) {
      status=cublas_probe_error(cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,4096,
          static_cast<int>(rows),2560,&alpha,spans[1].data,CUDA_R_16BF,2560,
          spans[0].data,CUDA_R_16BF,2560,&beta,spans[2].data,CUDA_R_16BF,4096,
          CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(projection_sm80_profile && rows>=2 && rows<=32 && plan->config.n==151936 && plan->config.k==2560) {
      if(native_prefill_sm80_profile && rows>=17 &&
          ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(31))==0){
        sm80_batched_lm_head<4,32,8,2><<<2374,128,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<__nv_bfloat16*>(spans[2].data),static_cast<int>(rows));
        auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
      }else{
      // Preserve the independently checked M<=16 native-M1 vocabulary reduction.
      for(uint64_t first=0;first<rows && status==RILEY_CUDA_STATUS_SUCCESS;first+=16) {
        const int count=static_cast<int>(std::min<uint64_t>(rows-first,16));
        status=cublas_probe_error(cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,151936,count,2560,
            &alpha,spans[1].data,CUDA_R_16BF,2560,static_cast<const __nv_bfloat16*>(spans[0].data)+first*2560,
            CUDA_R_16BF,2560,&beta,static_cast<__nv_bfloat16*>(spans[2].data)+first*151936,
            CUDA_R_16BF,151936,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
      }
      }
    } else if(native_prefill_sm80_profile && plan->owned_sm80_decode_mlp_v8_enabled &&
        rows>=5 && rows<=32 && plan->config.n==2560 && plan->config.k==9728 &&
        plan->sm80_projection_workspace && plan->sm80_projection_workspace_bytes>=4*32*2560*4 &&
        ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(31))==0){
      auto*parts=static_cast<float*>(plan->sm80_projection_workspace->device_data);
      if(rows<=16)sm80_v9_down<8,16,8,4><<<dim3(20,1,4),256,0,stream->stream>>>(
        static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),parts,static_cast<int>(rows));
      else sm80_v9_down<4,32,16,4><<<dim3(40,1,4),128,0,stream->stream>>>(
        static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),parts,static_cast<int>(rows));
      const int count=static_cast<int>(rows)*2560;
      sm80_v9_down_reduce<<<(count+255)/256,256,0,stream->stream>>>(parts,static_cast<__nv_bfloat16*>(spans[2].data),count);
      auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(native_prefill_sm80_profile && plan->owned_sm80_decode_mlp_v8_enabled &&
        rows>=5 && rows<=32 && plan->config.n==2560 && plan->config.k==9728 &&
        plan->sm80_projection_workspace && plan->sm80_projection_workspace_bytes>=2*32*2560*2 &&
        ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(31))==0){
      auto*parts=static_cast<__nv_bfloat16*>(plan->sm80_projection_workspace->device_data);
      sm80_v8_down<4,16,8,4><<<dim3(40,(static_cast<int>(rows)+15)/16,2),128,0,stream->stream>>>(
        static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),parts,static_cast<int>(rows));
      const int count=static_cast<int>(rows)*2560;
      sm80_down_reduce<<<(count+255)/256,256,0,stream->stream>>>(parts,static_cast<__nv_bfloat16*>(spans[2].data),count);
      auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(native_prefill_sm80_profile && plan->owned_sm80_gateup_rows_qualified &&
        rows>=12 && rows<=32 && plan->config.n==2560 && plan->config.k==9728 &&
        plan->sm80_projection_workspace_bytes>=2*32*2560*2 &&
        ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&31)==0) {
      auto*partial=static_cast<__nv_bfloat16*>(plan->sm80_projection_workspace->device_data);
      sm80_down_native_sliced<2><<<dim3(80,(rows+15)/16,2),64,0,stream->stream>>>(
          static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),partial,static_cast<int>(rows));
      sm80_down_reduce<<<(rows*2560+255)/256,256,0,stream->stream>>>(partial,static_cast<__nv_bfloat16*>(spans[2].data),static_cast<int>(rows*2560));
      auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(projection_sm80_profile && rows>=2 && rows<=32 && plan->config.n==2560 &&
        (plan->config.k==4096 || plan->config.k==9728)) {
      for(uint64_t first=0;first<rows && status==RILEY_CUDA_STATUS_SUCCESS;first+=plan->sm80_projection_rows) {
        const int count=static_cast<int>(std::min<uint64_t>(rows-first,plan->sm80_projection_rows));
        const auto*x=static_cast<const __nv_bfloat16*>(spans[0].data)+first*plan->config.k;
        auto*y=static_cast<__nv_bfloat16*>(spans[2].data)+first*plan->config.n;
        auto st=count==1 ? cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,static_cast<int>(plan->config.n),1,
            static_cast<int>(plan->config.k),&alpha,spans[1].data,CUDA_R_16BF,static_cast<int>(plan->config.k),x,CUDA_R_16BF,
            static_cast<int>(plan->config.k),&beta,y,CUDA_R_16BF,static_cast<int>(plan->config.n),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)
          : cublasLtMatmul(plan->decode_lt,plan->decode_op,&alpha,spans[1].data,plan->decode_weight,x,plan->decode_input[count],
            &beta,y,plan->decode_output[count],y,plan->decode_output[count],&plan->decode_algo,
            plan->sm80_projection_workspace->device_data,plan->sm80_projection_workspace_bytes,stream->stream);
        status=cublas_probe_error(st,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
      }
    } else if(native_prefill_sm80_profile && plan->owned_sm80_decode_mlp_v8_enabled &&
        rows>=17 && rows<=32 && plan->config.n==9728 && plan->config.k==2560 &&
        ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(31))==0){
      sm80_v8_gate<4,16,16,2><<<dim3(152,(static_cast<int>(rows)+15)/16),128,0,stream->stream>>>(
        static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<__nv_bfloat16*>(spans[2].data),static_cast<int>(rows));
      auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if(bounded_sm80_profile && plan->owned_sm80_gateup_rows_qualified &&
        rows>=2 && rows<=32 &&
        ((plan->config.n==9728 && plan->config.k==2560) ||
         (plan->config.n==14336 && plan->config.k==4096))) {
      // Separate projections, bounded M<=16 preserves native-M1 reduction.
      // All buffers/plans remain registered by the original owned batch.
      for(uint64_t first=0;first<rows && status==RILEY_CUDA_STATUS_SUCCESS;first+=16) {
        const int count=static_cast<int>(rows-first<16?rows-first:16);
        status=cublas_probe_error(cublasGemmEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,
            static_cast<int>(plan->config.n),count,static_cast<int>(plan->config.k),
            &alpha,spans[1].data,CUDA_R_16BF,static_cast<int>(plan->config.k),
            static_cast<const __nv_bfloat16*>(spans[0].data)+first*plan->config.k,
            CUDA_R_16BF,static_cast<int>(plan->config.k),&beta,
            static_cast<__nv_bfloat16*>(spans[2].data)+first*plan->config.n,
            CUDA_R_16BF,static_cast<int>(plan->config.n),CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,
            kExecuteOperation);
      }
    } else if (grouped_decode==2 && plan->decode_ready && rows>=2 && rows<=32 &&
        (plan->decode_span_alignment==0 ||
         ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|
           reinterpret_cast<uintptr_t>(spans[2].data))&uintptr_t(plan->decode_span_alignment-1))==0)) {
      status=cublas_probe_error(cublasLtMatmul(plan->decode_lt,plan->decode_op,
          &alpha,spans[1].data,plan->decode_batch_weight[rows]?plan->decode_batch_weight[rows]:plan->decode_weight,spans[0].data,plan->decode_input[rows],
          &beta,spans[2].data,plan->decode_output[rows],spans[2].data,plan->decode_output[rows],
          &plan->decode_algo,nullptr,0,stream->stream),error,
          RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if (grouped_decode==2 && rows>=2 && rows<=32 &&
               plan->info.compute_capability_major==8 &&
               plan->info.compute_capability_minor==9 &&
               plan->info.cublas_version==120804 &&
               plan->config.n==2048 && plan->config.k==11008) {
      // SM89 / cuBLAS 12.8.4 Down projection: strided GEMV retains the
      // independent M=1 reduction. Gate/Up select a different batched kernel
      // and are deliberately excluded. Zero weight stride shares W while
      // input/output strides keep each request's row disjoint.
      status=cublas_probe_error(cublasGemmStridedBatchedEx(
          plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,
          static_cast<int>(plan->config.n),1,static_cast<int>(plan->config.k),
          &alpha,spans[1].data,CUDA_R_16BF,static_cast<int>(plan->config.k),0,
          spans[0].data,CUDA_R_16BF,static_cast<int>(plan->config.k),
          static_cast<long long>(plan->config.k),&beta,spans[2].data,CUDA_R_16BF,
          static_cast<int>(plan->config.n),static_cast<long long>(plan->config.n),
          static_cast<int>(rows),CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
    } else if (grouped_decode) {
      // Group submission only: retain the exact per-request cublasGemmEx
      // dimensions, compute type and algorithm. Strided-batch was rejected
      // after independent BF16 controls found changed output bits.
      for (uint64_t row=0; row<rows && status==RILEY_CUDA_STATUS_SUCCESS; ++row) {
        status = cublas_probe_error(cublasGemmEx(
            plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(plan->config.n), 1, static_cast<int>(plan->config.k),
            &alpha, spans[1].data, CUDA_R_16BF, static_cast<int>(plan->config.k),
            static_cast<const __nv_bfloat16*>(spans[0].data)+row*plan->config.k,
            CUDA_R_16BF, static_cast<int>(plan->config.k), &beta,
            static_cast<__nv_bfloat16*>(spans[2].data)+row*plan->config.n,
            CUDA_R_16BF, static_cast<int>(plan->config.n),
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kExecuteOperation);
      }
    } else {
    status = cublas_probe_error(
        cublasGemmEx(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     static_cast<int>(plan->config.n),
                     static_cast<int>(rows),
                     static_cast<int>(plan->config.k), &alpha, spans[1].data,
                     CUDA_R_16BF, static_cast<int>(plan->config.k),
                     spans[0].data, CUDA_R_16BF,
                     static_cast<int>(plan->config.k), &beta, spans[2].data,
                     CUDA_R_16BF, static_cast<int>(plan->config.n),
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        error, RILEY_CUDA_ERROR_STAGE_LAUNCH, kExecuteOperation);
    }
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error);
}

// Re-register every captured operand for one managed MLP replay. The Rust
// owner compares opaque allocation/stream identities before calling this.
extern "C" RileyCudaStatus riley_cuda_owned_hf_mlp_register_uses(
    RileyCudaCublasGemmProbePlan* gate, RileyCudaCublasGemmProbePlan* down,
    const RileyCudaBufferSpan* buffers, RileyCudaStream* stream,
    RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (gate == nullptr || down == nullptr || buffers == nullptr || stream == nullptr ||
      !gate->owned_command_batches || !down->owned_command_batches ||
      !command_batch_is_owned_by_current_thread(stream) ||
      riley_cuda_internal::thread_has_active_step_capture() ||
      riley_cuda_internal::thread_has_active_graph_capture()) {
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_STATE,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,
        "owned MLP replay requires an owned command batch without nested capture");
  }
  if (gate->owner != down->owner || gate->config.m != down->config.m ||
      (gate->config.n != down->config.k && gate->config.n != 2*down->config.k) || gate->config.k != down->config.n) {
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,
        RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"MLP plan dimensions differ");
  }
  const bool merged=gate->config.n==2*down->config.k;
  const uint64_t bytes[8] = {gate->input_bytes,gate->weight_bytes/(merged?2:1),
      gate->weight_bytes/(merged?2:1),down->weight_bytes,gate->output_bytes,
      down->input_bytes,down->input_bytes,down->output_bytes};
  ResolvedSpan spans[8]{};
  for (size_t i=0;i<8;++i) {
    RileyCudaStatus status=resolve_attention_span(&buffers[i],RILEY_CUDA_DTYPE_BF16,
        bytes[i],&spans[i],error,"MLP operand must be BF16");
    if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  }
  RileyCudaStatus status=validate_span_relationships(gate,stream,spans,8,error);
  if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  ExclusiveProbeUses uses(gate,stream);
  for(const auto& span:spans)if(!uses.add(span.buffer))return internal_error(
      error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"MLP resource set overflow");
  // Captured library nodes retain both merged Gate/Up and Down workspaces.
  if(gate->prefill_workspace && !uses.add(gate->prefill_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"merged MLP workspace resource set overflow");
  // A bounded Down graph retains the private Lt workspace on every replay.
  if(down->sm80_projection_workspace && !uses.add(down->sm80_projection_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"MLP workspace resource set overflow");
  if(down->sm80_full_projection_workspace && !uses.add(down->sm80_full_projection_workspace))
    return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"full MLP workspace resource set overflow");
  status=uses.acquire(error);
  if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
  return command_batch_register_use(stream,&down->active_uses,error,kExecuteOperation,
      "owned MLP down plan already has an active use");
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

namespace {
__global__ void hf_q_gather(const __nv_bfloat16*in,__nv_bfloat16*out,int m,int h,int d){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*h*d){int depth=i%d,row=(i/d)%m,head=i/(d*m);out[i]=in[(row*h+head)*d+depth];}}
// One aligned load supplies all four GQA query heads. Copy BF16 bits only;
// retain the expanded operands and original cuBLAS strides and reduction.
__global__ void hf_sm80_vector_kv_gather(const uint4* __restrict__ kin,const uint4* __restrict__ vin,const uint32_t* __restrict__ ids,uint4* __restrict__ k,uint4* __restrict__ v,int t,uint64_t physical){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<t*8*16){int depth=i%16,token=(i/16)%t,head=i/(16*t);uint64_t id=ids[token/16];
  uint4 kk,vv;
  if(id>=physical){uint32_t nan=__bfloat16_as_ushort(__float2bfloat16_rn(CUDART_NAN_F));nan|=nan<<16;kk=vv=make_uint4(nan,nan,nan,nan);}
  else{uint64_t source=((id*8+head)*16+token%16)*16+depth;kk=kin[source];vv=vin[source];}
  #pragma unroll
  for(int g=0;g<4;g++){int target=((head*4+g)*t+token)*16+depth;k[target]=kk;v[target]=vv;}
 }
}
__global__ void hf_kv_gather(const __nv_bfloat16*kin,const __nv_bfloat16*vin,const uint32_t*ids,__nv_bfloat16*k,__nv_bfloat16*v,int t,uint64_t physical,int h,int kvh,int d){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<t*h*d){int depth=i%d,row=(i/d)%t,head=i/(d*t);uint64_t id=ids[row/16];if(id>=physical){k[i]=v[i]=__float2bfloat16_rn(CUDART_NAN_F);}else{uint64_t j=((id*kvh+head/(h/kvh))*16+row%16)*d+depth;k[i]=kin[j];v[i]=vin[j];}}}
// Tile only independent softmax rows. Full BF16 QK/AV geometry is unchanged.
// Global score/probability tensors exceed INT32_MAX elements at 8k Q32.
__global__ void hf_scale_mask_tile(const __nv_bfloat16*in,float*out,int m,int t,int h,int start,int tile,float scale){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<uint64_t(h)*tile*t){int key=i%t,row=(i/t)%tile,head=i/(uint64_t(tile)*t);uint64_t source=(uint64_t(head)*m+start+row)*t+key;
 out[i]=key>t-m+start+row?-3.3895313892515355e38f:__bfloat162float(__float2bfloat16_rn(__bfloat162float(in[source])*scale));}
}
__global__ void hf_scatter_bf16_tile(const float*in,__nv_bfloat16*out,int m,int t,int h,int start,int tile){
 uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<uint64_t(h)*tile*t){int key=i%t,row=(i/t)%tile,head=i/(uint64_t(tile)*t);uint64_t target=(uint64_t(head)*m+start+row)*t+key;out[target]=__float2bfloat16_rn(in[i]);}
}
__global__ void hf_output_rows(const __nv_bfloat16*in,__nv_bfloat16*out,int m,int h,int d){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<m*h*d){int depth=i%d,head=(i/d)%h,row=i/(h*d);out[i]=in[(head*m+row)*d+depth];}}
}
// Private bounded per-operation, caller-owned workspace, existing use ledger.
extern "C" RileyCudaStatus riley_cuda_cublas_hf_attention_execute(
 RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,
 uint64_t rows,uint64_t tokens,uint64_t physical,uint64_t qh,uint64_t kvh,uint64_t depth,
 RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!plan || !plan->handle || !raw || count!=14 || rows==0 || rows>8192 || tokens<rows || tokens>9216 || physical==0 || physical>UINT32_MAX || rows>plan->config.m || tokens>plan->config.n || (depth!=64&&depth!=128) || depth!=plan->config.k || qh==0 || qh>64 || kvh==0 || kvh>qh || qh%kvh!=0 || qh/kvh>8)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"bounded diagnostic requires BF16 D64/D128 GQA<=8, QH<=64, M<=8192, T<=9216 and 14 spans");
 const uint64_t q=rows*qh*depth*2,kv=tokens*qh*depth*2,s=qh*rows*tokens,tile_s=qh*(rows<128?rows:128)*tokens,pool=physical*kvh*16*depth*2;
 const uint64_t sizes[14]={q,pool,pool,((tokens+15)/16)*4,q,q,kv,kv,s*2,tile_s*4,tile_s*4,s*2,q,16777216};
 ResolvedSpan spans[14]{};RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<14;i++){uint32_t dtype=i==3?RILEY_CUDA_DTYPE_U32:(i==9||i==10)?RILEY_CUDA_DTYPE_F32:i==13?RILEY_CUDA_DTYPE_U8:RILEY_CUDA_DTYPE_BF16;status=resolve_attention_span(raw+i,dtype,sizes[i],spans+i,error,"attention span dtype mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 status=validate_span_relationships(plan,stream,spans,14,error,true);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(auto&span:spans)if(!uses.add(span.buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"attention use ledger overflow");
 status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetStream(plan->handle,stream->stream),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetWorkspace(plan->handle,spans[13].data,sizes[13]),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS){int m=static_cast<int>(rows),t=static_cast<int>(tokens),h=static_cast<int>(qh),kh=static_cast<int>(kvh),d=static_cast<int>(depth);float alpha=1,beta=0,scale=static_cast<float>(1.0/std::sqrt(static_cast<double>(d)));attempted=true;
  // A singleton already has both [row,head,depth] and [head,row,depth]
  // layout. Keep GEMM dimensions/reduction unchanged and avoid identity copies.
  const void* query_operand=m==1?spans[0].data:spans[5].data;
  void* context_operand=m==1?spans[4].data:spans[12].data;
  if(m!=1)hf_q_gather<<<(m*h*d+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<__nv_bfloat16*>(spans[5].data),m,h,d);
  const bool sm80_staging=plan->owned_sm80_attention_enabled && h==32 && kh==8 && d==128 &&
      !(reinterpret_cast<uintptr_t>(spans[1].data)&15) && !(reinterpret_cast<uintptr_t>(spans[2].data)&15) &&
      !(reinterpret_cast<uintptr_t>(spans[6].data)&15) && !(reinterpret_cast<uintptr_t>(spans[7].data)&15);
  if(sm80_staging)hf_sm80_vector_kv_gather<<<(t*8*16+255)/256,256,0,stream->stream>>>(static_cast<const uint4*>(spans[1].data),static_cast<const uint4*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<uint4*>(spans[6].data),static_cast<uint4*>(spans[7].data),t,physical);
  else hf_kv_gather<<<(t*h*d+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<__nv_bfloat16*>(spans[6].data),static_cast<__nv_bfloat16*>(spans[7].data),t,physical,h,kh,d);
  status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,t,m,d,&alpha,spans[6].data,CUDA_R_16BF,d,t*d,query_operand,CUDA_R_16BF,d,m*d,&beta,spans[8].data,CUDA_R_16BF,t,m*t,h,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  if(status==RILEY_CUDA_STATUS_SUCCESS){
   // Singleton decode has no causal-mask exclusions. Fuse only the scale load;
   // QK/PV dimensions, cuBLAS selection, warp reduction and BF16 rounding stay fixed.
   const bool fused_d64 = plan->owned_weight_reuse_qualified && m==1 &&
       h==9 && kh==3 && d==64 && t<=1024;
   const bool fused_sm80=sm80_staging && m==1 && t<=2048;
   for(int start=0;start<m;start+=128){int tile=(m-start)<128?(m-start):128;uint64_t elements=uint64_t(h)*tile*t;
    if(!fused_d64 && !fused_sm80)hf_scale_mask_tile<<<(elements+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[8].data),static_cast<float*>(spans[9].data),m,t,h,start,tile,scale);
    if(t<=2048){int log=0;while((1<<log)<t)log++;int width=(1<<log)<32?(1<<log):32,warps=128/width,wb=log<=7?2:1,blocks=(h*tile+warps*wb-1)/(warps*wb);dim3 threads(width,warps);
#define HF_CASE(L) case L: if(fused_sm80){softmax_warp_forward<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,true><<<blocks,threads,0,stream->stream>>>(static_cast<__nv_bfloat16*>(spans[11].data),static_cast<const __nv_bfloat16*>(spans[8].data),h*tile,t,t);}else if(fused_d64){softmax_warp_forward<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,false,true><<<blocks,threads,0,stream->stream>>>(static_cast<__nv_bfloat16*>(spans[11].data),static_cast<const __nv_bfloat16*>(spans[8].data),h*tile,t,t);}else if(m==1){softmax_warp_forward<float,__nv_bfloat16,float,L,false,false,32><<<blocks,threads,0,stream->stream>>>(static_cast<__nv_bfloat16*>(spans[11].data),static_cast<const float*>(spans[9].data),h*tile,t,t);}else{softmax_warp_forward<float,float,float,L,false,false,32><<<blocks,threads,0,stream->stream>>>(static_cast<float*>(spans[10].data),static_cast<const float*>(spans[9].data),h*tile,t,t);}break;
     switch(log){HF_CASE(0) HF_CASE(1) HF_CASE(2) HF_CASE(3) HF_CASE(4) HF_CASE(5) HF_CASE(6) HF_CASE(7) HF_CASE(8) HF_CASE(9) HF_CASE(10) HF_CASE(11)}
#undef HF_CASE
    }else{
#define HF_REG(N) case N: hf_regular_f32_softmax<N><<<h*tile,1024,0,stream->stream>>>(static_cast<float*>(spans[10].data),static_cast<const float*>(spans[9].data),t);break;
     switch((t+1023)/1024){HF_REG(3) HF_REG(4) HF_REG(5) HF_REG(6) HF_REG(7) HF_REG(8) HF_REG(9)}
#undef HF_REG
    }
    if(m!=1 || t>2048)hf_scatter_bf16_tile<<<(elements+255)/256,256,0,stream->stream>>>(static_cast<const float*>(spans[10].data),static_cast<__nv_bfloat16*>(spans[11].data),m,t,h,start,tile);
   }
   status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_N,CUBLAS_OP_N,d,m,t,&alpha,spans[7].data,CUDA_R_16BF,d,t*d,spans[11].data,CUDA_R_16BF,t,m*t,&beta,context_operand,CUDA_R_16BF,d,m*d,h,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   if(status==RILEY_CUDA_STATUS_SUCCESS && m!=1)hf_output_rows<<<(m*h*d+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[12].data),static_cast<__nv_bfloat16*>(spans[4].data),m,h,d);
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS){auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);}
 }
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}
namespace {
struct HfDecodeChunk { int count,total,tokens[32],starts[32]; };
__global__ void hf_heterogeneous_kv_gather(const __nv_bfloat16* __restrict__ kin,const __nv_bfloat16* __restrict__ vin,const uint32_t* __restrict__ ids,const uint32_t* __restrict__ slots,const uint32_t* __restrict__ offsets,__nv_bfloat16* __restrict__ k,__nv_bfloat16* __restrict__ v,uint64_t physical,uint64_t sequences,uint64_t id_count,const HfDecodeChunk info){
 int i=blockIdx.x*blockDim.x+threadIdx.x,request=blockIdx.y,head=blockIdx.z,t=info.tokens[request],depth=i%128,token=i/128;
 if(token<t){
  {uint64_t slot=slots[request],page=token/16;bool valid=slot<sequences;uint64_t start=valid?offsets[slot]:0,end=valid?offsets[slot+1]:0;
   valid=valid && start<=end && end<=id_count && page<end-start;uint64_t id=valid?ids[start+page]:physical;int target=(info.starts[request]*16+head*t+token)*128+depth;
   if(!valid||id>=physical){k[target]=v[target]=__float2bfloat16_rn(CUDART_NAN_F);}else{uint64_t source=((id*2+head/8)*16+token%16)*128+depth;k[target]=kin[source];v[target]=vin[source];}
  }
 }
}
// Compact GQA operand staging. The unused tail of the existing K scratch
// stores six device pointer arrays: no additional buffers or CUDA allocations.
// Each original per-request cuBLAS call still has batchCount16 and actual T.
__global__ void hf_heterogeneous_compact_kv_gather(const __nv_bfloat16* __restrict__ kin,const __nv_bfloat16* __restrict__ vin,const uint32_t* __restrict__ ids,const uint32_t* __restrict__ slots,const uint32_t* __restrict__ offsets,__nv_bfloat16* __restrict__ k,__nv_bfloat16* __restrict__ v,const __nv_bfloat16* __restrict__ query,__nv_bfloat16* __restrict__ score,__nv_bfloat16* __restrict__ output,void**tables,uint64_t physical,uint64_t sequences,uint64_t id_count,const HfDecodeChunk info){
 int i=blockIdx.x*blockDim.x+threadIdx.x,request=blockIdx.y,head=blockIdx.z,t=info.tokens[request],depth=i%128,token=i/128;
 if(token<t){uint64_t slot=slots[request],page=token/16;bool valid=slot<sequences;uint64_t start=valid?offsets[slot]:0,end=valid?offsets[slot+1]:0;
  valid=valid&&start<=end&&end<=id_count&&page<end-start;uint64_t id=valid?ids[start+page]:physical;int target=(info.starts[request]*2+head*t+token)*128+depth;
  if(!valid||id>=physical){k[target]=v[target]=__float2bfloat16_rn(CUDART_NAN_F);}else{uint64_t source=((id*2+head)*16+token%16)*128+depth;k[target]=kin[source];v[target]=vin[source];}
 }
 if(blockIdx.x==0&&threadIdx.x<8){int qhead=head*8+threadIdx.x,index=request*16+qhead,heads=info.count*16;
  tables[index]=k+(info.starts[request]*2+head*t)*128;
  tables[heads+index]=const_cast<__nv_bfloat16*>(query)+(request*16+qhead)*128;
  tables[heads*2+index]=score+info.starts[request]*16+qhead*t;
  tables[heads*3+index]=v+(info.starts[request]*2+head*t)*128;
  tables[heads*4+index]=score+info.starts[request]*16+qhead*t;
  tables[heads*5+index]=output+(request*16+qhead)*128;
 }
}
__global__ void hf_chunk_scaled_trace(const __nv_bfloat16*in,float*out,int count){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=hf_decode_scaled_bf16_d128(__bfloat162float(in[i]));}
}
extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_prepare_sm80_attention(
    RileyCudaCublasGemmProbePlan* plan,RileyCudaErrorInfo* error) noexcept {
 clear_error(error);
 if(!plan || !plan->owned_sm80_gateup_rows_qualified || plan->config.k!=128 || plan->config.n>9216 || plan->active_uses.load()!=0)
  return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"cold qualified SM80 D128 attention owner required");
 plan->owned_sm80_attention_enabled=true;
 return RILEY_CUDA_STATUS_SUCCESS;
}
extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_prepare_sm80_projections(
    RileyCudaCublasGemmProbePlan* plan,RileyCudaErrorInfo* error) noexcept {
  clear_error(error);
  if(plan==nullptr || !plan->owned_sm80_gateup_rows_qualified || plan->active_uses.load()!=0)
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"cold qualified SM80 owner required");
  if(plan->config.n!=2560 || (plan->config.k!=4096 && plan->config.k!=9728) || plan->config.m<2) return RILEY_CUDA_STATUS_SUCCESS;
  if(cublasLtGetVersion()!=120405)
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"fixed SM80 projections require cuBLASLt 120405");
  if(plan->decode_lt!=nullptr || plan->sm80_projection_workspace!=nullptr)
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_STATE,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"projection recipe already prepared");
  for(const char* name:{"CUBLAS_WORKSPACE_CONFIG","CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}) {
    const char*v=std::getenv(name);if(v&&*v)return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"qualified unmodified workspace and SM environment required");
  }
  CurrentContext scope(plan->owner);auto status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
  auto checked=[&](cublasStatus_t st){return cublas_probe_error(st,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);};
  const bool down=plan->config.k==9728;const int id=down?31:21;
  if(down && status==RILEY_CUDA_STATUS_SUCCESS){cudaFuncAttributes a{};auto e=cudaFuncGetAttributes(&a,sm80_down_native_sliced<2>);if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);else if(a.binaryVersion<80||a.ptxVersion<80)status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"native SM80 Down image required");}

  const uint32_t tile=15,stages=18,split=2,reduction=down?4:2,swizzle=down?1:0,custom=0;
  const uint32_t limit=down?4:8;
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtCreate(&plan->decode_lt));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescCreate(&plan->decode_op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescSetAttribute(plan->decode_op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_weight,CUDA_R_16BF,plan->config.k,plan->config.n,plan->config.k));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoInit(plan->decode_lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,id,&plan->decode_algo));
  const cublasLtMatmulAlgoConfigAttributes_t attrs[]={CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
  const uint32_t vals[]={tile,stages,split,reduction,swizzle,custom};
  for(size_t i=0;i<6 && status==RILEY_CUDA_STATUS_SUCCESS;i++) {
    status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->decode_algo,attrs[i],&vals[i],sizeof(vals[i])));
    uint32_t got=0;size_t written=0;
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigGetAttribute(&plan->decode_algo,attrs[i],&got,sizeof(got),&written));
    if(status==RILEY_CUDA_STATUS_SUCCESS && (written!=sizeof(got)||got!=vals[i]))status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"fixed SM80 algorithm metadata changed");
  }
  for(uint32_t rows=2;rows<=limit && rows<=plan->config.m && status==RILEY_CUDA_STATUS_SUCCESS;rows++) {
    status=checked(cublasLtMatrixLayoutCreate(&plan->decode_input[rows],CUDA_R_16BF,plan->config.k,rows,plan->config.k));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->decode_output[rows],CUDA_R_16BF,plan->config.n,rows,plan->config.n));
    cublasLtMatmulHeuristicResult_t check{};
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoCheck(plan->decode_lt,plan->decode_op,plan->decode_weight,plan->decode_input[rows],plan->decode_output[rows],plan->decode_output[rows],&plan->decode_algo,&check));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(check.state);
    if(status==RILEY_CUDA_STATUS_SUCCESS && check.workspaceSize>4u*1024u*1024u)status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"fixed workspace bound exceeded");
    if(status==RILEY_CUDA_STATUS_SUCCESS)plan->sm80_projection_workspace_bytes=std::max<uint64_t>(plan->sm80_projection_workspace_bytes,check.workspaceSize);
  }
  if(down)plan->sm80_projection_workspace_bytes=std::max<uint64_t>(plan->sm80_projection_workspace_bytes,4*32*2560*4);
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=riley_cuda_device_buffer_create(plan->owner,std::max<uint64_t>(plan->sm80_projection_workspace_bytes,256),&plan->sm80_projection_workspace,error);
  if(status==RILEY_CUDA_STATUS_SUCCESS)plan->sm80_projection_rows=limit;
  return scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
}

// Extend the already pinned native Lt recipe to actual R2..32. Separate cold
// workspace keeps the older bounded recipes and their allocation identities.
static RileyCudaStatus prepare_sm80_full_projection_rows(RileyCudaCublasGemmProbePlan*plan,RileyCudaErrorInfo*error)noexcept{
 if(plan->owned_sm80_full_projection_rows)return RILEY_CUDA_STATUS_SUCCESS;
 if(!plan->owned_sm80_gateup_rows_qualified||!plan->decode_lt||!plan->decode_op||!plan->decode_weight||!plan->sm80_projection_workspace)
  return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"cold pinned projection recipe required");
 CurrentContext scope(plan->owner);auto status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);uint64_t required=256;
 auto checked=[&](cublasStatus_t st){return cublas_probe_error(st,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);};
 for(uint32_t r=2;r<=32&&r<=plan->config.m&&status==RILEY_CUDA_STATUS_SUCCESS;r++){
  if(!plan->decode_input[r])status=checked(cublasLtMatrixLayoutCreate(&plan->decode_input[r],CUDA_R_16BF,plan->config.k,r,plan->config.k));
  if(status==RILEY_CUDA_STATUS_SUCCESS&&!plan->decode_output[r])status=checked(cublasLtMatrixLayoutCreate(&plan->decode_output[r],CUDA_R_16BF,plan->config.n,r,plan->config.n));
  cublasLtMatmulHeuristicResult_t check{};
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoCheck(plan->decode_lt,plan->decode_op,plan->decode_weight,plan->decode_input[r],plan->decode_output[r],plan->decode_output[r],&plan->decode_algo,&check));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(check.state);
  if(status==RILEY_CUDA_STATUS_SUCCESS&&check.workspaceSize>4u*1024u*1024u)status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"full projection workspace exceeds4MiB");
  if(status==RILEY_CUDA_STATUS_SUCCESS)required=std::max<uint64_t>(required,check.workspaceSize);
 }
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=riley_cuda_device_buffer_create(plan->owner,required,&plan->sm80_full_projection_workspace,error);
 if(status==RILEY_CUDA_STATUS_SUCCESS){plan->sm80_full_projection_workspace_bytes=required;plan->owned_sm80_full_projection_rows=true;}
 return scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
}

// Qualified against native M128 for every M64..576, six input states.
// Dispatch only merges original M64..130, whose native families match M128.
extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_prepare_sm80_prefill(
    RileyCudaCublasGemmProbePlan*plan,RileyCudaErrorInfo*error) noexcept {
  clear_error(error);
  if(plan==nullptr || !plan->owned_sm80_gateup_rows_qualified || plan->active_uses.load()!=0)
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"cold qualified SM80 owner required");
  const bool down=plan->config.n==2560 && plan->config.k==9728;
  const bool gate=plan->config.n==9728 && plan->config.k==2560;
  const bool merged=plan->config.n==19456 && plan->config.k==2560 && plan->config.m<=32;
  const bool output=plan->config.n==2560&&plan->config.k==4096;
  if((down||output)&&plan->decode_lt&&plan->sm80_projection_workspace){auto st=prepare_sm80_full_projection_rows(plan,error);if(st!=RILEY_CUDA_STATUS_SUCCESS)return st;}
  const bool query=plan->config.n==4096&&plan->config.k==2560;
  const bool kv=plan->config.n==1024&&plan->config.k==2560;
  if(!down&&!gate&&!query&&!kv&&!output&&!merged)return RILEY_CUDA_STATUS_SUCCESS;
  if(down||gate){
    CurrentContext image_scope(plan->owner);auto image_status=image_scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
    if(image_status==RILEY_CUDA_STATUS_SUCCESS){
      cudaFuncAttributes a{};auto e=cudaFuncGetAttributes(&a,down?reinterpret_cast<const void*>(sm80_v8_down<4,16,8,4>):reinterpret_cast<const void*>(sm80_v8_gate<4,16,16,2>));
      if(e!=cudaSuccess)image_status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
      else if(a.binaryVersion<80||a.ptxVersion<80)image_status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"native SM80 decode MLP v8 image required");
    }
    if(image_status==RILEY_CUDA_STATUS_SUCCESS)for(const void*fn:{reinterpret_cast<const void*>(sm80_fused_gateup<4,16,16,2>),reinterpret_cast<const void*>(sm80_fused_gateup<4,32,8,2>),reinterpret_cast<const void*>(owned_sm80_tiled_kv<8,4>),reinterpret_cast<const void*>(owned_sm80_tiled_kv<4,8>),reinterpret_cast<const void*>(owned_sm80_tiled_kv<8,8>),reinterpret_cast<const void*>(sm80_batched_lm_head<4,32,8,2>)}){
      cudaFuncAttributes attrs{};auto e=cudaFuncGetAttributes(&attrs,fn);
      if(e!=cudaSuccess){image_status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);break;}
      if(attrs.binaryVersion<80||attrs.ptxVersion<80){image_status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"native SM80 batch images required");break;}
    }
    image_status=image_scope.leave(image_status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
    if(image_status!=RILEY_CUDA_STATUS_SUCCESS)return image_status;
    if(down){
      CurrentContext v9_scope(plan->owner);auto st=v9_scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
      if(st==RILEY_CUDA_STATUS_SUCCESS)for(auto fn:{reinterpret_cast<const void*>(sm80_v9_down<8,16,8,4>),reinterpret_cast<const void*>(sm80_v9_down<4,32,16,4>)}){
        cudaFuncAttributes a{};auto e=cudaFuncGetAttributes(&a,fn);
        if(e!=cudaSuccess){st=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);break;}
        if(a.binaryVersion<80||a.ptxVersion<80){st=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"native SM80 decode Down v9 image required");break;}
      }
      st=v9_scope.leave(st,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);if(st!=RILEY_CUDA_STATUS_SUCCESS)return st;
    }
    plan->owned_sm80_decode_mlp_v8_enabled=true;
  }
  if(!merged && plan->config.m<161)return RILEY_CUDA_STATUS_SUCCESS;
  if(plan->prefill_lt!=nullptr || plan->prefill_workspace!=nullptr)
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_STATE,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"prefill recipe already prepared");
  if(cublasLtGetVersion()!=120405)
    return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"qualified cuBLASLt120405 required");
  for(const char*name:{"CUBLAS_WORKSPACE_CONFIG","CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"}){
    const char*v=std::getenv(name);if(v&&*v)return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"unmodified workspace/SM environment required");
  }
  CurrentContext scope(plan->owner);auto status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
  auto checked=[&](cublasStatus_t st){return cublas_probe_error(st,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);};
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtCreate(&plan->prefill_lt));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescCreate(&plan->prefill_op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
  cublasOperation_t trans=CUBLAS_OP_T;
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulDescSetAttribute(plan->prefill_op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->prefill_weight,CUDA_R_16BF,plan->config.k,plan->config.n,plan->config.k));
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoInit(plan->prefill_lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,CUDA_R_16BF,merged?31:(query||kv||output||down)?6:21,&plan->prefill_algo));
  const cublasLtMatmulAlgoConfigAttributes_t attrs[]={CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
  // Pinned native-M128 accumulation families, including split-K reduction.
  // Q: original65..130, merged257..576; KV:94..128, merged161..512;
  // O:65..128, merged161..576. Rust dispatch preserves all other extents.
  const uint32_t vals[]={merged?15u:down?24u:(query||kv)?18u:20u,merged?17u:down?15u:(query||kv)?12u:11u,
    merged?0u:down||output?5u:kv?4u:0u,merged?0u:down||output||kv?4u:0u,merged?0u:down||query||kv?1u:0u,0};
  for(size_t i=0;i<6 && status==RILEY_CUDA_STATUS_SUCCESS;i++){
    status=checked(cublasLtMatmulAlgoConfigSetAttribute(&plan->prefill_algo,attrs[i],&vals[i],sizeof(vals[i])));
    uint32_t got=0;size_t written=0;
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoConfigGetAttribute(&plan->prefill_algo,attrs[i],&got,sizeof(got),&written));
    if(status==RILEY_CUDA_STATUS_SUCCESS && (written!=sizeof(got)||got!=vals[i]))status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"prefill algorithm metadata changed");
  }
  const uint32_t maximum=static_cast<uint32_t>(std::min<uint64_t>(merged?32:kv?512:576,plan->config.m));
  for(uint32_t rows=merged?1:161;rows<=maximum && status==RILEY_CUDA_STATUS_SUCCESS;rows++){
    status=checked(cublasLtMatrixLayoutCreate(&plan->prefill_input[rows],CUDA_R_16BF,plan->config.k,rows,plan->config.k));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatrixLayoutCreate(&plan->prefill_output[rows],CUDA_R_16BF,plan->config.n,rows,plan->config.n));
    cublasLtMatmulHeuristicResult_t check{};
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(cublasLtMatmulAlgoCheck(plan->prefill_lt,plan->prefill_op,plan->prefill_weight,plan->prefill_input[rows],plan->prefill_output[rows],plan->prefill_output[rows],&plan->prefill_algo,&check));
    if(status==RILEY_CUDA_STATUS_SUCCESS)status=checked(check.state);
    if(status==RILEY_CUDA_STATUS_SUCCESS && check.workspaceSize>16u*1024u*1024u)status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"prefill workspace exceeds16MiB");
    if(status==RILEY_CUDA_STATUS_SUCCESS)plan->prefill_workspace_bytes=std::max<uint64_t>(plan->prefill_workspace_bytes,check.workspaceSize);
  }
  plan->prefill_workspace_bytes=std::max<uint64_t>(256,plan->prefill_workspace_bytes);
  if(status==RILEY_CUDA_STATUS_SUCCESS)status=riley_cuda_device_buffer_create(plan->owner,plan->prefill_workspace_bytes,&plan->prefill_workspace,error);
  if(status==RILEY_CUDA_STATUS_SUCCESS)plan->prefill_max_rows=maximum;
  return scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
}

extern "C" int riley_cuda_cublas_bounded_sm80_gateup_qualified(const RileyCudaCublasGemmProbePlan*plan) noexcept {return plan && plan->owned_sm80_gateup_rows_qualified;}
extern "C" int riley_cuda_cublas_hf_decode_chunk_qualified(const RileyCudaCublasGemmProbePlan*plan) noexcept {return plan && (plan->owned_weight_reuse_qualified||plan->owned_sm80_paged_attention_enabled);}
// Eleven disjoint, caller-owned spans. Original per-request QK/PV dimensions
// and BF16 probability rounding are retained; only independent work is fused.
extern "C" RileyCudaStatus riley_cuda_cublas_hf_attention_chunk_execute(
 RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,
 const uint64_t*host_tokens,uint64_t requests,uint64_t physical,uint64_t sequences,uint64_t id_count,
 RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!plan || !plan->handle || !plan->owned_weight_reuse_qualified || !raw || (count!=11 && count!=12) || !host_tokens || requests<2 || requests>8 || physical==0 || physical>UINT32_MAX || sequences==0 || sequences>UINT32_MAX || id_count==0 || id_count>UINT32_MAX || plan->config.k!=128)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified Q16/KV2/D128 decode chunk requires2..8requests and11spans(optional12thscaledtrace)");
 HfDecodeChunk info{};info.count=static_cast<int>(requests);uint64_t total=0;int maximum=0;
 for(uint64_t r=0;r<requests;r++){uint64_t t=host_tokens[r];if(t==0 || t>512 || t>plan->config.n)return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"decode chunk T must be1..512 withinprepared bound");info.tokens[r]=static_cast<int>(t);info.starts[r]=static_cast<int>(total);total+=t;info.total=static_cast<int>(total);if(static_cast<int>(t)>maximum)maximum=static_cast<int>(t);}
 const uint64_t q=requests*16*128*2,pool=physical*2*16*128*2,kv=total*16*128*2,scores=total*16*2;
 const uint64_t sizes[12]={q,pool,pool,id_count*4,q,kv,kv,scores,16777216,requests*4,(sequences+1)*4,total*16*4};ResolvedSpan spans[12]{};
 RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<count;i++){uint32_t dtype=(i==3||i==9||i==10)?RILEY_CUDA_DTYPE_U32:i==8?RILEY_CUDA_DTYPE_U8:i==11?RILEY_CUDA_DTYPE_F32:RILEY_CUDA_DTYPE_BF16;status=resolve_attention_span(raw+i,dtype,sizes[i],spans+i,error,"decode chunk span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 status=validate_span_relationships(plan,stream,spans,count,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(size_t i=0;i<count;i++)if(!uses.add(spans[i].buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"decode chunk use ledger overflow");status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetStream(plan->handle,stream->stream),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetWorkspace(plan->handle,spans[8].data,sizes[8]),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;float alpha=1,beta=0;
  const uint64_t compact_bytes=total*2*128*2,table_bytes=6*requests*16*sizeof(void*);
  const bool compact=total>=128 && (reinterpret_cast<uintptr_t>(spans[5].data)%16)==0 && compact_bytes+table_bytes<=kv;
  static_assert(sizeof(void*)==8,"qualified CUDA pointer width");
  // T>=1 and requests<=8 prove this scratch-tail bound.
  void**tables=reinterpret_cast<void**>(static_cast<__nv_bfloat16*>(spans[5].data)+total*2*128);
  const int table_heads=info.count*16;
  if(compact)hf_heterogeneous_compact_kv_gather<<<dim3((maximum*128+255)/256,info.count,2),256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<const uint32_t*>(spans[9].data),static_cast<const uint32_t*>(spans[10].data),static_cast<__nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<__nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),tables,physical,sequences,id_count,info);
  else hf_heterogeneous_kv_gather<<<dim3((maximum*128+255)/256,info.count,16),256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<const uint32_t*>(spans[9].data),static_cast<const uint32_t*>(spans[10].data),static_cast<__nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[6].data),physical,sequences,id_count,info);
  for(int r=0;r<info.count && status==RILEY_CUDA_STATUS_SUCCESS;r++){int t=info.tokens[r];uint64_t kv_start=uint64_t(info.starts[r])*16*128,score_start=uint64_t(info.starts[r])*16;
   if(compact)status=cublas_probe_error(cublasGemmBatchedEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,t,1,128,&alpha,reinterpret_cast<const void* const*>(tables+r*16),CUDA_R_16BF,128,reinterpret_cast<const void* const*>(tables+table_heads+r*16),CUDA_R_16BF,128,&beta,tables+table_heads*2+r*16,CUDA_R_16BF,t,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   else status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,t,1,128,&alpha,static_cast<const __nv_bfloat16*>(spans[5].data)+kv_start,CUDA_R_16BF,128,t*128,static_cast<const __nv_bfloat16*>(spans[0].data)+r*16*128,CUDA_R_16BF,128,128,&beta,static_cast<__nv_bfloat16*>(spans[7].data)+score_start,CUDA_R_16BF,t,t,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  }
  if(count==12 && status==RILEY_CUDA_STATUS_SUCCESS)hf_chunk_scaled_trace<<<(info.total*16+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<float*>(spans[11].data),info.total*16);
  for(int first=0;first<info.count && status==RILEY_CUDA_STATUS_SUCCESS;){int end=first+1,t=info.tokens[first];while(end<info.count && info.tokens[end]==t)end++;int heads=(end-first)*16,log=0;while((1<<log)<t)log++;int width=(1<<log)<32?(1<<log):32,warps=128/width,wb=log<=7?2:1,blocks=(heads+warps*wb-1)/(warps*wb);dim3 threads(width,warps);auto*p=static_cast<__nv_bfloat16*>(spans[7].data)+uint64_t(info.starts[first])*16;
#define HF_CHUNK_CASE(L) case L:softmax_warp_forward<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,true><<<blocks,threads,0,stream->stream>>>(p,p,heads,t,t);break;
   switch(log){HF_CHUNK_CASE(0) HF_CHUNK_CASE(1) HF_CHUNK_CASE(2) HF_CHUNK_CASE(3) HF_CHUNK_CASE(4) HF_CHUNK_CASE(5) HF_CHUNK_CASE(6) HF_CHUNK_CASE(7) HF_CHUNK_CASE(8) HF_CHUNK_CASE(9)}
#undef HF_CHUNK_CASE
   first=end;
  }
  for(int r=0;r<info.count && status==RILEY_CUDA_STATUS_SUCCESS;r++){int t=info.tokens[r];uint64_t kv_start=uint64_t(info.starts[r])*16*128,score_start=uint64_t(info.starts[r])*16;
   if(compact)status=cublas_probe_error(cublasGemmBatchedEx(plan->handle,CUBLAS_OP_N,CUBLAS_OP_N,128,1,t,&alpha,reinterpret_cast<const void* const*>(tables+table_heads*3+r*16),CUDA_R_16BF,128,reinterpret_cast<const void* const*>(tables+table_heads*4+r*16),CUDA_R_16BF,t,&beta,tables+table_heads*5+r*16,CUDA_R_16BF,128,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   else status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_N,CUBLAS_OP_N,128,1,t,&alpha,static_cast<const __nv_bfloat16*>(spans[6].data)+kv_start,CUDA_R_16BF,128,t*128,static_cast<const __nv_bfloat16*>(spans[7].data)+score_start,CUDA_R_16BF,t,t,&beta,static_cast<__nv_bfloat16*>(spans[4].data)+r*16*128,CUDA_R_16BF,128,128,16,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS){auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);}
 }
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}

// Qualified Q32/KV8 variant preserves original per-request batchCount32.
namespace {
__global__ void hf_heterogeneous_wide_kv_gather(const __nv_bfloat16* __restrict__ kin,const __nv_bfloat16* __restrict__ vin,const uint32_t* __restrict__ ids,const uint32_t* __restrict__ slots,const uint32_t* __restrict__ offsets,__nv_bfloat16* __restrict__ k,__nv_bfloat16* __restrict__ v,uint64_t physical,uint64_t sequences,uint64_t id_count,const HfDecodeChunk info){
 int i=blockIdx.x*blockDim.x+threadIdx.x,request=blockIdx.y,head=blockIdx.z,t=info.tokens[request],depth=i%128,token=i/128;
 if(token<t){
  {uint64_t slot=slots[request],page=token/16;bool valid=slot<sequences;uint64_t start=valid?offsets[slot]:0,end=valid?offsets[slot+1]:0;
   valid=valid && start<=end && end<=id_count && page<end-start;uint64_t id=valid?ids[start+page]:physical;int target=(info.starts[request]*32+head*t+token)*128+depth;
   if(!valid||id>=physical){k[target]=v[target]=__float2bfloat16_rn(CUDART_NAN_F);}else{uint64_t source=((id*8+head/4)*16+token%16)*128+depth;k[target]=kin[source];v[target]=vin[source];}
  }
 }
}
// Compact GQA operand staging. The unused tail of the existing K scratch
// stores six device pointer arrays: no additional buffers or CUDA allocations.
// Each original per-request cuBLAS call still has batchCount32 and actual T.
__global__ void hf_heterogeneous_wide_compact_kv_gather(const __nv_bfloat16* __restrict__ kin,const __nv_bfloat16* __restrict__ vin,const uint32_t* __restrict__ ids,const uint32_t* __restrict__ slots,const uint32_t* __restrict__ offsets,__nv_bfloat16* __restrict__ k,__nv_bfloat16* __restrict__ v,const __nv_bfloat16* __restrict__ query,__nv_bfloat16* __restrict__ score,__nv_bfloat16* __restrict__ output,void**tables,uint64_t physical,uint64_t sequences,uint64_t id_count,const HfDecodeChunk info){
 int i=blockIdx.x*blockDim.x+threadIdx.x,request=blockIdx.y,head=blockIdx.z,t=info.tokens[request],depth=i%128,token=i/128;
 if(token<t){uint64_t slot=slots[request],page=token/16;bool valid=slot<sequences;uint64_t start=valid?offsets[slot]:0,end=valid?offsets[slot+1]:0;
  valid=valid&&start<=end&&end<=id_count&&page<end-start;uint64_t id=valid?ids[start+page]:physical;int target=(info.starts[request]*8+head*t+token)*128+depth;
  if(!valid||id>=physical){k[target]=v[target]=__float2bfloat16_rn(CUDART_NAN_F);}else{uint64_t source=((id*8+head)*16+token%16)*128+depth;k[target]=kin[source];v[target]=vin[source];}
 }
 if(blockIdx.x==0&&threadIdx.x<4){int qhead=head*4+threadIdx.x,index=request*32+qhead,heads=info.count*32;
  tables[index]=k+(info.starts[request]*8+head*t)*128;
  tables[heads+index]=const_cast<__nv_bfloat16*>(query)+(request*32+qhead)*128;
  tables[heads*2+index]=score+info.starts[request]*32+qhead*t;
  tables[heads*3+index]=v+(info.starts[request]*8+head*t)*128;
  tables[heads*4+index]=score+info.starts[request]*32+qhead*t;
  tables[heads*5+index]=output+(request*32+qhead)*128;
 }
}
__host__ __device__ __forceinline__ bool hf_wide_qk_native_serial16(int t){if(t<1||t>512)return false;int bit=(t-1)&63;switch((t-1)>>6){
 case 0:return ((0x5555555540000000ULL>>bit)&1)!=0;
 case 1:return ((0x5555555555555555ULL>>bit)&1)!=0;
 case 2:return ((0x7f7f7f7f7f7f5555ULL>>bit)&1)!=0;
 case 3:return ((0x5555555f7f7f7f7fULL>>bit)&1)!=0;
 case 4:return ((0x5555555555555555ULL>>bit)&1)!=0;
 case 5:return ((0x5555555555555555ULL>>bit)&1)!=0;
 case 6:return ((0x0000000000000001ULL>>bit)&1)!=0;
 case 7:return ((0x0000000000000000ULL>>bit)&1)!=0;
 default:return false;}}
__host__ __device__ __forceinline__ bool hf_wide_pv_native_contiguous4(int t){if(t<1||t>512)return false;int bit=(t-1)&63;switch((t-1)>>6){
 case 0:return ((0xff7f7f7f7f7f0000ULL>>bit)&1)!=0;
 case 1:return ((0xd5ffffffffffffffULL>>bit)&1)!=0;
 case 2:return ((0x7f7f7f7f7f7f5555ULL>>bit)&1)!=0;
 case 3:return ((0x555555557f7f7f7fULL>>bit)&1)!=0;
 case 4:return ((0x5555555555555555ULL>>bit)&1)!=0;
 case 5:return ((0x8080000000000055ULL>>bit)&1)!=0;
 case 6:return ((0x8080808080808080ULL>>bit)&1)!=0;
 case 7:return ((0x8080808080808080ULL>>bit)&1)!=0;
 default:return false;}}

__host__ __device__ __forceinline__ bool hf_wide_qk_native_f8(int t){if(t<1||t>512)return false;int bit=(t-1)&63;switch((t-1)>>6){
 case 0:return ((0x0000000000000000ULL>>bit)&1)!=0;
 case 1:return ((0x0000000000000000ULL>>bit)&1)!=0;
 case 2:return ((0x8000000000000000ULL>>bit)&1)!=0;
 case 3:return ((0xaaaaaaa080808080ULL>>bit)&1)!=0;
 case 4:return ((0xaaaaaaaaaaaaaaaaULL>>bit)&1)!=0;
 case 5:return ((0xaaaaaaaaaaaaaaaaULL>>bit)&1)!=0;
 case 6:return ((0xfffffffffffffffeULL>>bit)&1)!=0;
 case 7:return ((0xffffffffffffffffULL>>bit)&1)!=0;
 default:return false;}}
__host__ __device__ __forceinline__ bool hf_wide_pv_native_wmma(int t){if(t<1||t>512)return false;int bit=(t-1)&63;switch((t-1)>>6){
 case 0:return ((0x0080800000000000ULL>>bit)&1)!=0;
 case 1:return ((0x0000000000000000ULL>>bit)&1)!=0;
 case 2:return ((0x8080808080808080ULL>>bit)&1)!=0;
 case 3:return ((0x8080808080808080ULL>>bit)&1)!=0;
 case 4:return ((0x8080808080808080ULL>>bit)&1)!=0;
 case 5:return ((0x0000808080808080ULL>>bit)&1)!=0;
 case 6:return ((0x0000000000000000ULL>>bit)&1)!=0;
 case 7:return ((0x0000000000000000ULL>>bit)&1)!=0;
 default:return false;}}

// Actual SM86 GEMVX inner-K parameters:16items/lane with step1, or
// step8 for odd T>=387. Preserve both native FMA chains and the8-way tree.
template<bool Compact> __global__ void hf_wide_native_f8_qk(
 const __nv_bfloat16*query,const __nv_bfloat16*keys,__nv_bfloat16*scores,const HfDecodeChunk info){
 int r=blockIdx.z,head=blockIdx.y,t=info.tokens[r];if(!hf_wide_qk_native_f8(t))return;
 int lane=threadIdx.x,col=blockIdx.x*16+threadIdx.y,kvhead=Compact?head/4:head,width=Compact?8:32;
 uint64_t start=uint64_t(info.starts[r])*width*128;bool stride8=t>=387&&(t&1);float acc=0;
 if(col<t)for(int j=0;j<16;j++){int p=stride8?lane+j*8:lane*16+j;acc=__fmaf_rn(__bfloat162float(keys[start+(uint64_t(kvhead)*t+col)*128+p]),__bfloat162float(query[(r*32+head)*128+p]),acc);}
 for(int delta=4;delta;delta/=2){float v=__shfl_down_sync(0xffffffffu,acc,delta,8);if(lane+delta<8)acc=__fadd_rn(acc,v);}
 if(lane==0&&col<t)scores[uint64_t(info.starts[r])*32+head*t+col]=__float2bfloat16_rn(__fmaf_rn(1.f,acc,0.f));
}
// Derived from the same installed PyTorch PersistentSoftmax.cuh BSD source
// as hf_softmax_diagnostic.cuh. Only virtual thread indexing changes; every
// FP32 operation, BF16 scale/store boundary, WARP_BATCH and shuffle tree stays.
template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, bool is_masked, int WARP_SIZE_PARAM, bool scaled_bf16_d128 = false, bool scaled_bf16_d64 = false, bool strided_rows = false>
__device__ __forceinline__ void hf_wide_chunk_softmax_body(output_t *dst, const input_t *src, int batch_size, int stride, int element_count, int virtual_x, int virtual_y, int virtual_warps, const bool *mask = nullptr, const int head_chunk_size = -1, bool is_transformer_mask = false, int first_batch_override = -1)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < WARP_SIZE_PARAM) ? next_power_of_two : WARP_SIZE_PARAM;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = first_batch_override >= 0 ? first_batch_override : (virtual_warps * blockIdx.x + virtual_y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = virtual_x;
    int idx_offset = first_batch * stride + local_idx;

    src += idx_offset;
    dst += idx_offset;

    if (is_transformer_mask) {
        mask += ((first_batch * stride) / head_chunk_size) * stride + local_idx;
    } else {
        mask += idx_offset;
    }
    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                elements[i][it] = src[i*(strided_rows?stride:element_count)+it*WARP_SIZE];
                if constexpr (scaled_bf16_d64) {
                    elements[i][it] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(elements[i][it],0.125f)));
                }
                if constexpr (scaled_bf16_d128) {
                    elements[i][it] = hf_decode_scaled_bf16_d128(elements[i][it]);
                }
            } else {
                elements[i][it] = -CUDART_INF_F;
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        bool is_meaningful_max = false;
        max_value[i] = elements[i][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (is_masked) {
                int idx = it*WARP_SIZE;
                if ((idx + local_idx) < batch_element_count) {
                    if (!is_transformer_mask) {
                        idx += i*(strided_rows?stride:element_count);
                    }
                    if (!mask[idx]) {
                        max_value[i] = (is_meaningful_max && max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
                        is_meaningful_max = true;
                    }
                }
            } else {
                max_value[i] = max_value[i] > elements[i][it] ? max_value[i] : elements[i][it];
            }
        }
        if (is_masked) {
            if (!is_meaningful_max) {
                max_value[i] = -CUDART_INF_F;
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (!is_masked) {
                if (is_log_softmax) {
                    sum[i] += std::exp(elements[i][it] - max_value[i]);
                } else {
                    elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                    sum[i] += elements[i][it];
                }
            } else {
                int idx = it*WARP_SIZE;
                bool valid = (idx + local_idx) < batch_element_count;
                if (!is_transformer_mask) {
                    idx += i*(strided_rows?stride:element_count);
                }
                if (valid) {
                    if (!mask[idx]) {
                        if (is_log_softmax) {
                            sum[i] += std::exp(elements[i][it] - max_value[i]);
                        } else {
                            elements[i][it] = std::exp(elements[i][it] - max_value[i]);
                            sum[i] += elements[i][it];
                        }
                    } else {
                        if (!is_log_softmax) {
                            // Masked values are treated as -infinity, and std::exp(-infinity) is 0.
                            elements[i][it] = 0;
                        }
                    }
                } else {
                    if (!is_log_softmax) {
                        elements[i][it] = 0.;
                    }
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                if (is_log_softmax) {
                    dst[i*(strided_rows?stride:element_count)+it*WARP_SIZE] = elements[i][it] - max_value[i] - sum[i];
                } else if (sum[i] == 0) {
                    dst[i*(strided_rows?stride:element_count)+it*WARP_SIZE] = CUDART_NAN_F;
                } else {
                    dst[i*(strided_rows?stride:element_count)+it*WARP_SIZE] = elements[i][it] / sum[i];
                }
            } else {
                break;
            }
        }
    }
}
__global__ void hf_wide_chunk_softmax(__nv_bfloat16* probability,const HfDecodeChunk info){
 const int r=blockIdx.y,t=info.tokens[r];int log=0;while((1<<log)<t)log++;
 const int width=log<5?(1<<log):32,warps=128/width,wb=log<=7?2:1;
 if(blockIdx.x>=(32+warps*wb-1)/(warps*wb))return;
 const int tid=threadIdx.x+blockDim.x*threadIdx.y;
 auto*p=probability+uint64_t(info.starts[r])*32;
 #define HF_WIDE_SOFTMAX_CASE(L) case L:hf_wide_chunk_softmax_body<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,true>(p,p,32,t,t,tid%width,tid/width,warps);break;
 switch(log){HF_WIDE_SOFTMAX_CASE(0) HF_WIDE_SOFTMAX_CASE(1) HF_WIDE_SOFTMAX_CASE(2) HF_WIDE_SOFTMAX_CASE(3) HF_WIDE_SOFTMAX_CASE(4) HF_WIDE_SOFTMAX_CASE(5) HF_WIDE_SOFTMAX_CASE(6) HF_WIDE_SOFTMAX_CASE(7) HF_WIDE_SOFTMAX_CASE(8) HF_WIDE_SOFTMAX_CASE(9)}
 #undef HF_WIDE_SOFTMAX_CASE
}

struct HfNativePvGroup {int count,rows[8];};
// Native CUTLASS WMMA family only. Group equal-T requests without moving
// queries, KV data, probabilities or outputs. FMA/MMA and BF16 boundaries stay fixed.
template<bool Compact> __global__ void hf_wide_native_wmma_pv(
 const __nv_bfloat16*values,const __nv_bfloat16*prob,__nv_bfloat16*out,const HfDecodeChunk info,const HfNativePvGroup group){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 using namespace nvcuda;int lane=threadIdx.x,baseD=blockIdx.x*16,head=blockIdx.y,r=group.rows[blockIdx.z],t=info.tokens[r];
 int kvhead=Compact?head/4:head,width=Compact?8:32;
 uint64_t vs=uint64_t(info.starts[r])*width*128+uint64_t(kvhead)*t*128,ps=uint64_t(info.starts[r])*32+head*t;
 __align__(32) __shared__ __nv_bfloat16 a[256],b[256];__align__(32) __shared__ float c[256];
 for(int i=lane;i<256;i+=32)b[i]=__ushort_as_bfloat16(0);__syncwarp();
 wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.f);
 for(int baseK=0;baseK<t;baseK+=16){
  int k=lane/2,d=(lane%2)*8;auto dst=reinterpret_cast<uint4*>(a+k*16+d);
  if(baseK+k<t)*dst=*reinterpret_cast<const uint4*>(values+vs+(baseK+k)*128+baseD+d);else *dst=make_uint4(0,0,0,0);
  if(lane<2){auto p=reinterpret_cast<uint4*>(b+lane*8);if(baseK+lane*8+7<t)*p=*reinterpret_cast<const uint4*>(prob+ps+baseK+lane*8);else *p=make_uint4(0,0,0,0);}
  __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major> af;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bf;
  wmma::load_matrix_sync(af,a,16);wmma::load_matrix_sync(bf,b,16);wmma::mma_sync(accum,af,bf,accum);__syncwarp();
 }
 wmma::store_matrix_sync(c,accum,16,wmma::mem_row_major);__syncwarp();
 if(lane<16)out[(r*32+head)*128+baseD+lane]=__float2bfloat16_rn(__fmaf_rn(1.f,c[lane*16],0.f));
#endif
}
bool hf_wide_extended_kernel_arch_qualified() noexcept {
 cudaFuncAttributes a{};auto status=cudaFuncGetAttributes(&a,hf_wide_native_wmma_pv<true>);
 if(status!=cudaSuccess){(void)cudaGetLastError();return false;}
 // A compute_70 PTX image could JIT on SM89 with an empty WMMA body.
 return a.binaryVersion>=80 && a.ptxVersion>=80;
}

// Qualified native-family arithmetic, coalesced across heads and requests.
// QK retains stride16 FMA partials and their ascending serial FP32 sum.
template<bool Compact> __global__ void hf_wide_native_order_qk(
 const __nv_bfloat16* __restrict__ query,const __nv_bfloat16* __restrict__ keys,
 __nv_bfloat16* __restrict__ scores,const HfDecodeChunk info){
 int r=blockIdx.z,head=blockIdx.y,t=info.tokens[r],base=blockIdx.x*8;
 if(!hf_wide_qk_native_serial16(t)||base>=t)return;
 int lane=threadIdx.x,column=base+threadIdx.y,kvhead=Compact?head/4:head;
 int width=Compact?8:32;uint64_t start=uint64_t(info.starts[r])*width*128;
 float value=0;
 if(column<t){
  #pragma unroll
  for(int k=lane;k<128;k+=16){
   float a=__bfloat162float(keys[start+(uint64_t(kvhead)*t+column)*128+k]);
   float b=__bfloat162float(query[(r*32+head)*128+k]);
   value=__fmaf_rn(a,b,value);
  }
 }
 __shared__ float partial[8][16];partial[threadIdx.y][lane]=value;__syncthreads();
 int output=lane+16*threadIdx.y;
 if(output<8&&base+output<t){
  float sum=partial[output][0];
  #pragma unroll
  for(int k=1;k<16;k++)sum=__fadd_rn(sum,partial[output][k]);
  scores[uint64_t(info.starts[r])*32+head*t+base+output]=__float2bfloat16_rn(__fmaf_rn(1.f,sum,0.f));
 }
}
// PV retains contiguous groups of4, stride128, then the descending32-way tree.
// Native-like physical lanes coalesce V columns; only the final4-way tail
// uses warp shuffles, with exactly the same addition parenthesization.
template<bool Compact> __global__ void hf_wide_native_order_pv(
 const __nv_bfloat16* __restrict__ values,const __nv_bfloat16* __restrict__ probabilities,
 __nv_bfloat16* __restrict__ output,const HfDecodeChunk info){
 int r=blockIdx.z,head=blockIdx.y,t=info.tokens[r];
 if(!hf_wide_pv_native_contiguous4(t))return;
 int x=threadIdx.x,lane=threadIdx.y,depth=blockIdx.x*8+x,kvhead=Compact?head/4:head;
 int width=Compact?8:32;uint64_t start=uint64_t(info.starts[r])*width*128;
 uint64_t ps=uint64_t(info.starts[r])*32+head*t;float value=0;
 for(int base=lane*4;base<t;base+=128){
  #pragma unroll
  for(int j=0;j<4;j++)if(base+j<t){
   float a=__bfloat162float(values[start+(uint64_t(kvhead)*t+base+j)*128+depth]);
   float b=__bfloat162float(probabilities[ps+base+j]);value=__fmaf_rn(a,b,value);
  }
 }
 __shared__ float partial[32][8];partial[lane][x]=value;__syncthreads();
 #pragma unroll
 for(int offset=16;offset>=4;offset/=2){
  if(lane<offset)partial[lane][x]=__fadd_rn(partial[lane][x],partial[lane+offset][x]);
  __syncthreads();
 }
 if(lane<4){
  float sum=partial[lane][x],other=__shfl_down_sync(0xffffffffu,sum,16,32);
  if(lane<2)sum=__fadd_rn(sum,other);
  other=__shfl_down_sync(0xffffffffu,sum,8,32);if(lane==0)sum=__fadd_rn(sum,other);
  if(lane==0)output[(r*32+head)*128+depth]=__float2bfloat16_rn(__fmaf_rn(1.f,sum,0.f));
 }
}

}
namespace {
__host__ __device__ static int sm80_paged_qk_family(uint64_t t){
 if(t<24 || t>216)return -1;
 if(t>=130 && t%2==0)return 2;
 if((t>=49 && t%2==1)||t%8==0)return 1;
 return 0;
}
struct HfSm80Group {int count=0,maximum=0,rows[32]{};};
__device__ __forceinline__ uint64_t sm80_paged_id(const uint32_t*ids,const uint32_t*slots,const uint32_t*offsets,int r,int page,uint64_t physical,uint64_t sequences,uint64_t id_count){
 uint64_t slot=slots[r];if(slot>=sequences)return physical;uint64_t start=offsets[slot],end=offsets[slot+1];if(start>end||end>id_count||uint64_t(page)>=end-start)return physical;uint64_t id=ids[start+page];return id<physical?id:physical;
}
// Pinned SM80 native reduction families. All scores and probabilities retain
// their original BF16 stores; queries and KV pages keep their existing layout.
template<bool Serial16> __global__ void sm80_chunk_paged_qk(const __nv_bfloat16*q,const __nv_bfloat16*k,const uint32_t*ids,__nv_bfloat16*score,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const HfDecodeChunk info,const HfSm80Group group){
 int r=group.rows[blockIdx.z],t=info.tokens[r];q+=r*32*128;score+=uint64_t(info.starts[r])*32;
 constexpr int L=Serial16?16:8;int lane=threadIdx.x,token=blockIdx.x*8+threadIdx.y,head=blockIdx.y;float sum=0.f;
 if(token<t){uint64_t id=sm80_paged_id(ids,slots,offsets,r,token/16,physical,sequences,id_count);
 #pragma unroll
 for(int j=0;j<128/L;j++){int d=Serial16?lane+j*16:lane*16+j;float a=id<physical?__bfloat162float(k[((id*8+head/4)*16+token%16)*128+d]):CUDART_NAN_F;float b=__bfloat162float(q[head*128+d]);sum=__fmaf_rn(a,b,sum);}}
 __shared__ float p[8][L];p[threadIdx.y][lane]=sum;__syncthreads();
 if(lane==0&&token<t){float v=p[threadIdx.y][0];if(Serial16){
 #pragma unroll
 for(int j=1;j<L;j++)v=__fadd_rn(v,p[threadIdx.y][j]);
 }else{float*pv=p[threadIdx.y];float a=__fadd_rn(pv[0],pv[4]),b=__fadd_rn(pv[1],pv[5]),c=__fadd_rn(pv[2],pv[6]),d=__fadd_rn(pv[3],pv[7]);v=__fadd_rn(__fadd_rn(a,c),__fadd_rn(b,d));}score[head*t+token]=__float2bfloat16_rn(__fmaf_rn(1.f,v,0.f));}
}
__global__ void sm80_chunk_paged_wmma_qk(const __nv_bfloat16*q,const __nv_bfloat16*k,const uint32_t*ids,__nv_bfloat16*score,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const HfDecodeChunk info,const HfSm80Group group){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 int r=group.rows[blockIdx.z],t=info.tokens[r];q+=r*32*128;score+=uint64_t(info.starts[r])*32;if(blockIdx.x*16>=t)return;
 using namespace nvcuda;int lane=threadIdx.x,head=blockIdx.y,first=blockIdx.x*16;
 __shared__ __align__(32) __nv_bfloat16 stage[4096];
 for(int i=lane;i<256;i+=32){int token=first+i/16,depth=i%16;uint64_t id=token<t?sm80_paged_id(ids,slots,offsets,r,token/16,physical,sequences,id_count):physical;uint4 v=make_uint4(0,0,0,0);if(token<t){if(id<physical)v=reinterpret_cast<const uint4*>(k)[((id*8+head/4)*16+token%16)*16+depth];else v=make_uint4(0x7fff7fff,0x7fff7fff,0x7fff7fff,0x7fff7fff);}reinterpret_cast<uint4*>(stage)[i]=v;
  reinterpret_cast<uint4*>(stage+2048)[i]=i<16?reinterpret_cast<const uint4*>(q)[head*16+i]:make_uint4(0,0,0,0);
 }__syncwarp();
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>acc;wmma::fill_fragment(acc,0.f);
 #pragma unroll
 for(int j=0;j<128;j+=16){wmma::load_matrix_sync(a,stage+j,128);wmma::load_matrix_sync(b,stage+2048+j,128);wmma::mma_sync(acc,a,b,acc);}
 __shared__ __align__(32) float result[256];wmma::store_matrix_sync(result,acc,16,wmma::mem_col_major);__syncwarp();if(lane<16&&first+lane<t)score[head*t+first+lane]=__float2bfloat16_rn(__fmaf_rn(1.f,result[lane],0.f));

#endif
}
__global__ void sm80_chunk_paged_pv(const __nv_bfloat16*values,const uint32_t*ids,const __nv_bfloat16*prob,__nv_bfloat16*out,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const HfDecodeChunk info,const HfSm80Group group){
 int r=group.rows[blockIdx.z],t=info.tokens[r];prob+=uint64_t(info.starts[r])*32;out+=r*32*128;
 int tid=threadIdx.x,head=blockIdx.y,depth=blockIdx.x*32+tid%32,lane=tid/32;float sum=0.f;
 for(int page=0;page<(t+15)/16;page++){uint64_t id=sm80_paged_id(ids,slots,offsets,r,page,physical,sequences,id_count),base=((id*8+head/4)*16)*128+depth;
 #pragma unroll
 for(int j=lane;j<16;j+=2){int token=page*16+j;if(token<t){float a=id<physical?__bfloat162float(values[base+j*128]):CUDART_NAN_F;float b=__bfloat162float(prob[head*t+token]);sum=__fmaf_rn(a,b,sum);}}}
 __shared__ float p[64];p[tid]=sum;__syncthreads();if(tid<32)out[head*128+depth]=__float2bfloat16_rn(__fmaf_rn(1.f,__fadd_rn(p[tid],p[tid+32]),0.f));
}
__global__ void sm80_chunk_paged_wmma_pv(const __nv_bfloat16*values,const uint32_t*ids,const __nv_bfloat16*prob,__nv_bfloat16*out,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const HfDecodeChunk info,const HfSm80Group group){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

 int r=group.rows[blockIdx.z],t=info.tokens[r];prob+=uint64_t(info.starts[r])*32;out+=r*32*128;
 using namespace nvcuda;int head=blockIdx.y,kh=head/4,lane=threadIdx.x,first=blockIdx.x*16;
 __shared__ __align__(32) __nv_bfloat16 stage[512];
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>acc;wmma::fill_fragment(acc,0.f);
 for(int base=0;base<t;base+=16){uint64_t id=sm80_paged_id(ids,slots,offsets,r,base/16,physical,sequences,id_count);
  for(int i=lane;i<32;i+=32){int token=base+i/2,depth=first/8+i%2;uint4 v=make_uint4(0,0,0,0);if(token<t){if(id<physical)v=reinterpret_cast<const uint4*>(values)[((id*8+kh)*16+token%16)*16+depth];else v=make_uint4(0x7fff7fff,0x7fff7fff,0x7fff7fff,0x7fff7fff);}reinterpret_cast<uint4*>(stage)[i]=v;}
  for(int i=lane;i<256;i+=32){int token=base+i%16;stage[256+i]=token<t?prob[head*t+token]:__float2bfloat16_rn(0.f);}__syncwarp();
  wmma::load_matrix_sync(a,stage,16);wmma::load_matrix_sync(b,stage+256,16);wmma::mma_sync(acc,a,b,acc);__syncwarp();
 }__shared__ __align__(32) float result[256];wmma::store_matrix_sync(result,acc,16,wmma::mem_col_major);__syncwarp();if(lane<16)out[head*128+first+lane]=__float2bfloat16_rn(__fmaf_rn(1.f,result[lane],0.f));

#endif
}
template<bool Serial16>__device__ __forceinline__ void sm80_batched_qk_simt(const __nv_bfloat16*q,const __nv_bfloat16*k,const uint32_t*ids,__nv_bfloat16*score,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,int r,int t,float*p){
 constexpr int L=Serial16?16:8;int tid=threadIdx.x,lane=tid%L,token=blockIdx.x*8+tid/L,head=blockIdx.y*4;float sum[4]={};bool active=tid<8*L;
 if(active&&token<t){uint64_t id=sm80_paged_id(ids,slots,offsets,r,token/16,physical,sequences,id_count);
 #pragma unroll
 for(int j=0;j<128/L;j++){int d=Serial16?lane+j*16:lane*16+j;float a=id<physical?__bfloat162float(k[((id*8+head/4)*16+token%16)*128+d]):CUDART_NAN_F;
 #pragma unroll
 for(int h=0;h<4;h++)sum[h]=__fmaf_rn(a,__bfloat162float(q[(head+h)*128+d]),sum[h]);}}
 if(active){
 #pragma unroll
 for(int h=0;h<4;h++)p[h*128+tid]=sum[h];}__syncthreads();
 if(active&&lane==0&&token<t){
 #pragma unroll
 for(int h=0;h<4;h++){float*pv=p+h*128+tid,val=pv[0];if constexpr(Serial16){
 #pragma unroll
 for(int j=1;j<16;j++)val=__fadd_rn(val,pv[j]);
 }else{float a=__fadd_rn(pv[0],pv[4]),b=__fadd_rn(pv[1],pv[5]),c=__fadd_rn(pv[2],pv[6]),d=__fadd_rn(pv[3],pv[7]);val=__fadd_rn(__fadd_rn(a,c),__fadd_rn(b,d));}score[(head+h)*t+token]=__float2bfloat16_rn(__fmaf_rn(1.f,val,0.f));}}
}
__global__ void sm80_batched_qk_dispatch(const __nv_bfloat16*q,const __nv_bfloat16*k,const uint32_t*ids,__nv_bfloat16*score,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const uint32_t*positions){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 int r=blockIdx.z,t=positions[r]+1,kind=sm80_paged_qk_family(t);if(kind<0||blockIdx.x*8>=t)return;q+=r*32*128;score+=uint64_t(r)*512*32;
 if(kind!=2){__shared__ float partial[4*128];if(kind==1)sm80_batched_qk_simt<true>(q,k,ids,score,physical,sequences,id_count,slots,offsets,r,t,partial);else sm80_batched_qk_simt<false>(q,k,ids,score,physical,sequences,id_count,slots,offsets,r,t,partial);return;}
 if(blockIdx.x%2||threadIdx.x>=32)return;int lane=threadIdx.x,head=blockIdx.y*4,first=blockIdx.x*8;using namespace nvcuda;
 __shared__ __align__(32)__nv_bfloat16 stage[4096];
 for(int i=lane;i<256;i+=32){int token=first+i/16,depth=i%16;uint64_t id=token<t?sm80_paged_id(ids,slots,offsets,r,token/16,physical,sequences,id_count):physical;uint4 vec=make_uint4(0,0,0,0);if(token<t)vec=id<physical?reinterpret_cast<const uint4*>(k)[((id*8+head/4)*16+token%16)*16+depth]:make_uint4(0x7fff7fff,0x7fff7fff,0x7fff7fff,0x7fff7fff);reinterpret_cast<uint4*>(stage)[i]=vec;reinterpret_cast<uint4*>(stage+2048)[i]=i<64?reinterpret_cast<const uint4*>(q)[head*16+i]:make_uint4(0,0,0,0);}
 __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>c;wmma::fill_fragment(c,0.f);
 #pragma unroll
 for(int j=0;j<128;j+=16){wmma::load_matrix_sync(a,stage+j,128);wmma::load_matrix_sync(b,stage+2048+j,128);wmma::mma_sync(c,a,b,c);}
 __shared__ __align__(32)float result[256];wmma::store_matrix_sync(result,c,16,wmma::mem_col_major);__syncwarp();if(lane<16&&first+lane<t){
 #pragma unroll
 for(int h=0;h<4;h++)score[(head+h)*t+first+lane]=__float2bfloat16_rn(__fmaf_rn(1.f,result[h*16+lane],0.f));}
#endif
}
__global__ void sm80_batched_pv_dispatch(const __nv_bfloat16*v,const uint32_t*ids,const __nv_bfloat16*prob,__nv_bfloat16*out,uint64_t physical,uint64_t sequences,uint64_t id_count,const uint32_t*slots,const uint32_t*offsets,const uint32_t*positions){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
 int r=blockIdx.z,t=positions[r]+1;if(t<24||t>216)return;prob+=uint64_t(r)*512*32;out+=r*32*128;int tid=threadIdx.x;
 if(t%8!=0){int head=blockIdx.y*2,depth=blockIdx.x*16+tid%16,chain=tid/16;float sum[2]={};
  for(int page=0;page<(t+15)/16;page++){uint64_t id=sm80_paged_id(ids,slots,offsets,r,page,physical,sequences,id_count),base=((id*8+head/4)*16)*128+depth;
   #pragma unroll
   for(int j=chain;j<16;j+=2){int token=page*16+j;if(token<t){float a=id<physical?__bfloat162float(v[base+j*128]):CUDART_NAN_F;
    #pragma unroll
    for(int h=0;h<2;h++)sum[h]=__fmaf_rn(a,__bfloat162float(prob[(head+h)*t+token]),sum[h]);}}}
  #pragma unroll
  for(int h=0;h<2;h++){float other=__shfl_down_sync(0xffffffffu,sum[h],16);if(tid<16)out[(head+h)*128+depth]=__float2bfloat16_rn(__fmaf_rn(1.f,__fadd_rn(sum[h],other),0.f));}return;
 }
 if(blockIdx.y%2)return;using namespace nvcuda;int head=blockIdx.y*2,kh=head/4,first=blockIdx.x*16;
 __shared__ __align__(32)__nv_bfloat16 stage[512];wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::col_major>a;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major>b;wmma::fragment<wmma::accumulator,16,16,16,float>c;wmma::fill_fragment(c,0.f);
 for(int base=0;base<t;base+=16){uint64_t id=sm80_paged_id(ids,slots,offsets,r,base/16,physical,sequences,id_count);int token=base+tid/2,depth=first/8+tid%2;uint4 vec=make_uint4(0,0,0,0);if(token<t)vec=id<physical?reinterpret_cast<const uint4*>(v)[((id*8+kh)*16+token%16)*16+depth]:make_uint4(0x7fff7fff,0x7fff7fff,0x7fff7fff,0x7fff7fff);reinterpret_cast<uint4*>(stage)[tid]=vec;
  for(int i=tid;i<256;i+=32){int token=base+i%16;stage[256+i]=i/16<4&&token<t?prob[(head+i/16)*t+token]:__float2bfloat16_rn(0.f);}__syncwarp();wmma::load_matrix_sync(a,stage,16);wmma::load_matrix_sync(b,stage+256,16);wmma::mma_sync(c,a,b,c);__syncwarp();}
 __shared__ __align__(32)float result[256];wmma::store_matrix_sync(result,c,16,wmma::mem_col_major);__syncwarp();if(tid<16){
 #pragma unroll
 for(int h=0;h<4;h++)out[(head+h)*128+first+tid]=__float2bfloat16_rn(__fmaf_rn(1.f,result[h*16+tid],0.f));}
#endif
}

__global__ void sm80_batched_softmax(__nv_bfloat16*prob,const uint32_t*positions){
 int r=blockIdx.y,t=positions[r]+1;if(t<24||t>216)return;int log=0;while((1<<log)<t)log++;int width=log<5?(1<<log):32,warps=128/width,wb=log<=7?2:1;if(blockIdx.x>=(32+warps*wb-1)/(warps*wb))return;int tid=threadIdx.x;auto*p=prob+uint64_t(r)*512*32;
#define BATCH_SM(L) case L:hf_wide_chunk_softmax_body<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,true>(p,p,32,t,t,tid%width,tid/width,warps);break;
 switch(log){BATCH_SM(5) BATCH_SM(6) BATCH_SM(7) BATCH_SM(8)}
#undef BATCH_SM
}

}

extern "C" RileyCudaStatus riley_cuda_cublas_hf_attention_chunk_wide_execute(
 RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,
 const uint64_t*host_tokens,uint64_t requests,uint64_t physical,uint64_t sequences,uint64_t id_count,
 RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 const bool sm80=plan && plan->owned_sm80_paged_attention_enabled;
 const bool dynamic=raw && count==12 && raw[11].dtype==RILEY_CUDA_DTYPE_U32;
 if(dynamic&&!sm80)return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"dynamic attention requires qualified SM80 paged plan");
 if(!plan || !plan->handle || (!plan->owned_weight_reuse_qualified && !sm80) || !raw || (count!=11 && count!=12) || !host_tokens || requests<2 || requests>32 || physical==0 || physical>UINT32_MAX || sequences==0 || sequences>UINT32_MAX || id_count==0 || id_count>UINT32_MAX || plan->config.k!=128)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified Q32/KV8/D128 decode chunk requires2..32requests and11spans(optional12thscaledtrace)");
 HfDecodeChunk info{};info.count=static_cast<int>(requests);uint64_t total=0;int maximum=0;
 for(uint64_t r=0;r<requests;r++){uint64_t t=host_tokens[r];if(t==0 || t>512 || t>plan->config.n || (sm80 && sm80_paged_qk_family(t)<0))return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"decode chunk T must be1..512 withinprepared bound");info.tokens[r]=static_cast<int>(t);info.starts[r]=static_cast<int>(total);total+=t;info.total=static_cast<int>(total);if(static_cast<int>(t)>maximum)maximum=static_cast<int>(t);}
 if(dynamic)total=requests*512;
 const uint64_t q=requests*32*128*2,pool=physical*8*16*128*2,scores=total*32*2,kv=sm80?scores:total*32*128*2;
 const uint64_t sizes[12]={q,pool,pool,id_count*4,q,kv,kv,scores,16777216,requests*4,(sequences+1)*4,dynamic?requests*4:total*32*4};ResolvedSpan spans[12]{};
 RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<count;i++){uint32_t dtype=(i==3||i==9||i==10||(dynamic&&i==11))?RILEY_CUDA_DTYPE_U32:i==8?RILEY_CUDA_DTYPE_U8:i==11?RILEY_CUDA_DTYPE_F32:RILEY_CUDA_DTYPE_BF16;status=resolve_attention_span(raw+i,dtype,sizes[i],spans+i,error,"decode chunk span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 if(sm80 && ((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[1].data)|reinterpret_cast<uintptr_t>(spans[2].data))&15)!=0)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"SM80 paged query/KV spans require16-byte alignment");
 uint64_t graph_ids=id_count,graph_sequences=sequences;
 if(dynamic && riley_cuda_internal::thread_has_active_step_capture()){
  for(size_t i:{size_t(3),size_t(10)}){
   uint64_t remaining=spans[i].buffer->byte_len-spans[i].byte_offset;
   if(remaining%4||remaining/4>UINT32_MAX||remaining<spans[i].byte_len)return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"captured CSR capacity invalid");
   spans[i].byte_len=remaining;
  }
  graph_ids=spans[3].byte_len/4;graph_sequences=spans[10].byte_len/4-1;
 }
 status=validate_span_relationships(plan,stream,spans,count,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(size_t i=0;i<count;i++)if(!uses.add(spans[i].buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"decode chunk use ledger overflow");status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS && sm80){
  if(dynamic){
   attempted=true;
   auto*q=static_cast<const __nv_bfloat16*>(spans[0].data);auto*k=static_cast<const __nv_bfloat16*>(spans[1].data);auto*v=static_cast<const __nv_bfloat16*>(spans[2].data);auto*ids=static_cast<const uint32_t*>(spans[3].data);auto*out=static_cast<__nv_bfloat16*>(spans[4].data);auto*p=static_cast<__nv_bfloat16*>(spans[7].data);auto*slots=static_cast<const uint32_t*>(spans[9].data);auto*offsets=static_cast<const uint32_t*>(spans[10].data);auto*pos=static_cast<const uint32_t*>(spans[11].data);
   sm80_batched_qk_dispatch<<<dim3(27,8,requests),128,0,stream->stream>>>(q,k,ids,p,physical,graph_sequences,graph_ids,slots,offsets,pos);
   sm80_batched_softmax<<<dim3(8,requests),128,0,stream->stream>>>(p,pos);
   sm80_batched_pv_dispatch<<<dim3(8,16,requests),32,0,stream->stream>>>(v,ids,p,out,physical,graph_sequences,graph_ids,slots,offsets,pos);
   auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   return complete_execution(&uses,&scope,stream,status,attempted,error);
  }
  attempted=true;HfSm80Group qgroups[3]{},pgroups[2]{};
  for(int r=0;r<info.count;r++){int t=info.tokens[r],kind=sm80_paged_qk_family(t);auto&g=qgroups[kind];g.rows[g.count++]=r;g.maximum=std::max(g.maximum,t);auto&p=pgroups[t%8==0];p.rows[p.count++]=r;p.maximum=std::max(p.maximum,t);}
  auto*q=static_cast<const __nv_bfloat16*>(spans[0].data);auto*k=static_cast<const __nv_bfloat16*>(spans[1].data);auto*v=static_cast<const __nv_bfloat16*>(spans[2].data);auto*ids=static_cast<const uint32_t*>(spans[3].data);auto*out=static_cast<__nv_bfloat16*>(spans[4].data);auto*prob=static_cast<__nv_bfloat16*>(spans[7].data);auto*slots=static_cast<const uint32_t*>(spans[9].data);auto*offsets=static_cast<const uint32_t*>(spans[10].data);
#define SM80_QK_ARGS q,k,ids,prob,physical,sequences,id_count,slots,offsets,info
  if(qgroups[0].count)sm80_chunk_paged_qk<false><<<dim3((qgroups[0].maximum+7)/8,32,qgroups[0].count),dim3(8,8),0,stream->stream>>>(SM80_QK_ARGS,qgroups[0]);
  if(qgroups[1].count)sm80_chunk_paged_qk<true><<<dim3((qgroups[1].maximum+7)/8,32,qgroups[1].count),dim3(16,8),0,stream->stream>>>(SM80_QK_ARGS,qgroups[1]);
  if(qgroups[2].count)sm80_chunk_paged_wmma_qk<<<dim3((qgroups[2].maximum+15)/16,32,qgroups[2].count),32,0,stream->stream>>>(SM80_QK_ARGS,qgroups[2]);
#undef SM80_QK_ARGS
  if(count==12)hf_chunk_scaled_trace<<<(total*32+255)/256,256,0,stream->stream>>>(prob,static_cast<float*>(spans[11].data),static_cast<int>(total*32));
  hf_wide_chunk_softmax<<<dim3(8,info.count),128,0,stream->stream>>>(prob,info);
#define SM80_PV_ARGS v,ids,prob,out,physical,sequences,id_count,slots,offsets,info
  if(pgroups[0].count)sm80_chunk_paged_pv<<<dim3(4,32,pgroups[0].count),64,0,stream->stream>>>(SM80_PV_ARGS,pgroups[0]);
  if(pgroups[1].count)sm80_chunk_paged_wmma_pv<<<dim3(8,32,pgroups[1].count),32,0,stream->stream>>>(SM80_PV_ARGS,pgroups[1]);
#undef SM80_PV_ARGS
  auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  return complete_execution(&uses,&scope,stream,status,attempted,error);
 }
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetStream(plan->handle,stream->stream),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS)status=cublas_probe_error(cublasSetWorkspace(plan->handle,spans[8].data,sizes[8]),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;float alpha=1,beta=0;
  const uint64_t compact_bytes=total*8*128*2,table_bytes=6*requests*32*sizeof(void*);
  const bool compact=total>=128 && (reinterpret_cast<uintptr_t>(spans[5].data)%16)==0 && compact_bytes+table_bytes<=kv;
  static_assert(sizeof(void*)==8,"qualified CUDA pointer width");
  // T>=1 and requests<=8 prove this scratch-tail bound.
  void**tables=reinterpret_cast<void**>(static_cast<__nv_bfloat16*>(spans[5].data)+total*8*128);
  const int table_heads=info.count*32;
  if(compact)hf_heterogeneous_wide_compact_kv_gather<<<dim3((maximum*128+255)/256,info.count,8),256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<const uint32_t*>(spans[9].data),static_cast<const uint32_t*>(spans[10].data),static_cast<__nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<__nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),tables,physical,sequences,id_count,info);
  else hf_heterogeneous_wide_kv_gather<<<dim3((maximum*128+255)/256,info.count,32),256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<const uint32_t*>(spans[9].data),static_cast<const uint32_t*>(spans[10].data),static_cast<__nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[6].data),physical,sequences,id_count,info);
  // The existing qualified plan and these span alignments freeze the native
  // family-table assumptions. Other spans keep the original cuBLAS route.
  const bool native_order_inputs=((reinterpret_cast<uintptr_t>(spans[0].data)|reinterpret_cast<uintptr_t>(spans[4].data)|reinterpret_cast<uintptr_t>(spans[5].data)|reinterpret_cast<uintptr_t>(spans[6].data)|reinterpret_cast<uintptr_t>(spans[7].data))&uintptr_t(15))==0;
  bool native_qk=false,native_pv=false;
  const bool extended_inputs=native_order_inputs && plan->owned_attention_extended_qualified;
  bool native_f8=false;for(int r=0;r<info.count;r++)native_f8|=extended_inputs&&hf_wide_qk_native_f8(info.tokens[r]);
  bool native_wmma_done[32]{};
  if(native_order_inputs)for(int r=0;r<info.count;r++){native_qk|=hf_wide_qk_native_serial16(info.tokens[r]);native_pv|=hf_wide_pv_native_contiguous4(info.tokens[r]);}
  if(native_qk){
   if(compact)hf_wide_native_order_qk<true><<<dim3((maximum+7)/8,32,info.count),dim3(16,8),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<const __nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[7].data),info);
   else hf_wide_native_order_qk<false><<<dim3((maximum+7)/8,32,info.count),dim3(16,8),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<const __nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[7].data),info);
  }
  if(native_f8){
   if(compact)hf_wide_native_f8_qk<true><<<dim3((maximum+15)/16,32,info.count),dim3(8,16),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<const __nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[7].data),info);
   else hf_wide_native_f8_qk<false><<<dim3((maximum+15)/16,32,info.count),dim3(8,16),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),static_cast<const __nv_bfloat16*>(spans[5].data),static_cast<__nv_bfloat16*>(spans[7].data),info);
  }
  for(int r=0;r<info.count && status==RILEY_CUDA_STATUS_SUCCESS;r++){int t=info.tokens[r];uint64_t kv_start=uint64_t(info.starts[r])*32*128,score_start=uint64_t(info.starts[r])*32;
   if((native_order_inputs&&hf_wide_qk_native_serial16(t))||(extended_inputs&&hf_wide_qk_native_f8(t)))continue;
   if(compact)status=cublas_probe_error(cublasGemmBatchedEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,t,1,128,&alpha,reinterpret_cast<const void* const*>(tables+r*32),CUDA_R_16BF,128,reinterpret_cast<const void* const*>(tables+table_heads+r*32),CUDA_R_16BF,128,&beta,tables+table_heads*2+r*32,CUDA_R_16BF,t,32,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   else status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_T,CUBLAS_OP_N,t,1,128,&alpha,static_cast<const __nv_bfloat16*>(spans[5].data)+kv_start,CUDA_R_16BF,128,t*128,static_cast<const __nv_bfloat16*>(spans[0].data)+r*32*128,CUDA_R_16BF,128,128,&beta,static_cast<__nv_bfloat16*>(spans[7].data)+score_start,CUDA_R_16BF,t,t,32,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  }
  if(count==12 && status==RILEY_CUDA_STATUS_SUCCESS)hf_chunk_scaled_trace<<<(info.total*32+255)/256,256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<float*>(spans[11].data),info.total*32);
  if(status==RILEY_CUDA_STATUS_SUCCESS)hf_wide_chunk_softmax<<<dim3(8,info.count),dim3(32,4),0,stream->stream>>>(static_cast<__nv_bfloat16*>(spans[7].data),info);
  if(native_pv){
   if(compact)hf_wide_native_order_pv<true><<<dim3(16,32,info.count),dim3(8,32),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),info);
   else hf_wide_native_order_pv<false><<<dim3(16,32,info.count),dim3(8,32),0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),info);
  }
  if(extended_inputs)for(int first=0;first<info.count;first++){
   if(native_wmma_done[first]||!hf_wide_pv_native_wmma(info.tokens[first]))continue;
   HfNativePvGroup group{};for(int r=first;r<info.count&&group.count<8;r++)if(!native_wmma_done[r]&&info.tokens[r]==info.tokens[first])group.rows[group.count++]=r;
   // Four or more equal-T requests amortize native WMMA launch work. Small
   // groups retain the original library path; no arithmetic recipe is changed.
   if(group.count<4)continue;
   if(compact)hf_wide_native_wmma_pv<true><<<dim3(8,32,group.count),32,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),info,group);
   else hf_wide_native_wmma_pv<false><<<dim3(8,32,group.count),32,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[6].data),static_cast<const __nv_bfloat16*>(spans[7].data),static_cast<__nv_bfloat16*>(spans[4].data),info,group);
   for(int i=0;i<group.count;i++)native_wmma_done[group.rows[i]]=true;
  }
  for(int r=0;r<info.count && status==RILEY_CUDA_STATUS_SUCCESS;r++){int t=info.tokens[r];uint64_t kv_start=uint64_t(info.starts[r])*32*128,score_start=uint64_t(info.starts[r])*32;
   if((native_order_inputs&&hf_wide_pv_native_contiguous4(t))||native_wmma_done[r])continue;
   if(compact)status=cublas_probe_error(cublasGemmBatchedEx(plan->handle,CUBLAS_OP_N,CUBLAS_OP_N,128,1,t,&alpha,reinterpret_cast<const void* const*>(tables+table_heads*3+r*32),CUDA_R_16BF,128,reinterpret_cast<const void* const*>(tables+table_heads*4+r*32),CUDA_R_16BF,t,&beta,tables+table_heads*5+r*32,CUDA_R_16BF,128,32,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
   else status=cublas_probe_error(cublasGemmStridedBatchedEx(plan->handle,CUBLAS_OP_N,CUBLAS_OP_N,128,1,t,&alpha,static_cast<const __nv_bfloat16*>(spans[6].data)+kv_start,CUDA_R_16BF,128,t*128,static_cast<const __nv_bfloat16*>(spans[7].data)+score_start,CUDA_R_16BF,t,t,&beta,static_cast<__nv_bfloat16*>(spans[4].data)+r*32*128,CUDA_R_16BF,128,128,32,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
  }
  if(status==RILEY_CUDA_STATUS_SUCCESS){auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);}
 }
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}



namespace {
// Empirically qualified only: V45 adversarial controls and V46 exact stage gate.
static constexpr unsigned char smol_host_recipe[1025][2]={{255,255},{255,255},{1,0},{1,1},{1,1},{1,4},{1,4},{1,4},{0,0},{1,7},{1,5},{1,7},{1,5},{1,0},{1,5},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,6},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,7},{2,0},{2,7},{2,0},{2,7},{2,0},{0,0},{2,0},{4,7},{2,0},{4,7},{2,0},{4,7},{2,0},{0,0},{2,0},{4,7},{2,0},{4,7},{2,0},{4,7},{2,0},{0,0},{2,0},{4,0},{2,0},{4,0},{2,0},{4,0},{2,0},{0,0},{2,0},{4,0},{2,0},{4,0},{2,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3}};
__device__ __constant__ unsigned char smol_device_recipe[1025][2]={{255,255},{255,255},{1,0},{1,1},{1,1},{1,4},{1,4},{1,4},{0,0},{1,7},{1,5},{1,7},{1,5},{1,0},{1,5},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,6},{1,0},{3,0},{1,0},{3,0},{1,0},{3,0},{1,0},{0,0},{1,0},{3,0},{1,0},{3,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{2,0},{0,0},{2,0},{2,7},{2,0},{2,7},{2,0},{2,7},{2,0},{0,0},{2,0},{4,7},{2,0},{4,7},{2,0},{4,7},{2,0},{0,0},{2,0},{4,7},{2,0},{4,7},{2,0},{4,7},{2,0},{0,0},{2,0},{4,0},{2,0},{4,0},{2,0},{4,0},{2,0},{0,0},{2,0},{4,0},{2,0},{4,0},{2,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{4,0},{0,0},{4,0},{0,0},{0,0},{0,0},{4,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,3},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,2},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,0},{0,2},{0,0},{0,2},{0,0},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{0,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,2},{0,8},{0,2},{0,8},{0,2},{0,8},{0,2},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,8},{0,1},{0,8},{0,1},{0,8},{0,1},{0,8},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{0,1},{1,3}};
}
// Exact H9/KV3/D64 singleton arithmetic. Unreviewed/unsupported T remains native.
namespace {
struct SmolRows { int count; int rows[32]; };
__global__ void smol_exact_gather(const __nv_bfloat16* kpool,const __nv_bfloat16* vpool,const uint32_t*ids,const uint32_t*slots,const uint32_t*positions,const uint32_t*offsets,__nv_bfloat16*k,__nv_bfloat16*v,uint64_t physical,uint64_t sequences,uint64_t idcount,SmolRows info){
 int r=blockIdx.z,head=blockIdx.y;uint32_t row=info.rows[r];uint64_t t=uint64_t(positions[row])+1;if(t>1024)return;uint64_t slot=slots[row];bool valid=slot<sequences;uint64_t begin=valid?offsets[slot]:0,end=valid?offsets[slot+1]:0;valid=valid&&begin<=end&&end<=idcount;
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<int(t)*64;i+=gridDim.x*blockDim.x){int token=i/64,d=i%64;bool address_valid=valid&&uint64_t(token/16)<end-begin;uint64_t page=address_valid?ids[begin+token/16]:physical;uint64_t dst=((uint64_t(r)*3+head)*1024+token)*64+d;if(!address_valid||page>=physical){k[dst]=v[dst]=__float2bfloat16_rn(CUDART_NAN_F);continue;}uint64_t src=((page*3+head)*16+token%16)*64+d;k[dst]=kpool[src];v[dst]=vpool[src];}
}
template<int L,int Chunk>__device__ __forceinline__ void smol_exact_qk_body(const __nv_bfloat16*q,const __nv_bfloat16*k,__nv_bfloat16*s,int t,int row,int r,int head){int tid=threadIdx.x,lane=tid%L;
 for(int base_col=blockIdx.x*(128/L);base_col<t;base_col+=gridDim.x*(128/L)){int col=base_col+tid/L;float value=0;if(col<t)for(int base=lane*Chunk;base<64;base+=L*Chunk){
#pragma unroll
 for(int j=0;j<Chunk;j++)if(base+j<64)value=__fmaf_rn(__bfloat162float(k[((r*3+head/3)*1024+col)*64+base+j]),__bfloat162float(q[(row*9+head)*64+base+j]),value);}
 for(int off=L/2;off;off/=2){float x=__shfl_down_sync(0xffffffff,value,off,L);if(lane+off<L)value=__fadd_rn(value,x);}if(lane==0&&col<t)s[(r*9+head)*1024+col]=__float2bfloat16_rn(__fmaf_rn(1.f,value,0.f));}
}
__device__ __forceinline__ void smol_exact_wmma_qk(const __nv_bfloat16*q,const __nv_bfloat16*k,__nv_bfloat16*s,int t,int row,int r,int head){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
 using namespace nvcuda;int warp=threadIdx.x/32,lane=threadIdx.x%32,token0=blockIdx.x*64+warp*16;if(token0>=t)return;
 __align__(32) __shared__ __nv_bfloat16 left[4][256],right[4][256];__align__(32) __shared__ float result[4][256];
 wmma::fragment<wmma::accumulator,16,16,16,float> accum;wmma::fill_fragment(accum,0.f);
 for(int base=0;base<64;base+=16){
  for(int i=lane;i<256;i+=32){int token=token0+i/16,depth=i%16;left[warp][i]=token<t?k[((r*3+head/3)*1024+token)*64+base+depth]:__float2bfloat16_rn(0);right[warp][i]=i/16==0?q[(row*9+head)*64+base+depth]:__float2bfloat16_rn(0);}
  __syncwarp();wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> af;wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bfreg;wmma::load_matrix_sync(af,left[warp],16);wmma::load_matrix_sync(bfreg,right[warp],16);wmma::mma_sync(accum,af,bfreg,accum);__syncwarp();
 }
 wmma::store_matrix_sync(result[warp],accum,16,wmma::mem_row_major);__syncwarp();if(lane<16&&token0+lane<t)s[(r*9+head)*1024+token0+lane]=__float2bfloat16_rn(__fmaf_rn(1.f,result[warp][lane*16],0.f));
#endif
}
__global__ __launch_bounds__(128,12) void smol_exact_qk(const __nv_bfloat16*q,const __nv_bfloat16*k,__nv_bfloat16*s,const uint32_t*positions,SmolRows info){int r=blockIdx.z,head=blockIdx.y,row=info.rows[r];uint64_t t=uint64_t(positions[row])+1;if(t>1024)return;switch(smol_device_recipe[t][0]){case 4:smol_exact_wmma_qk(q,k,s,t,row,r,head);break;case 0:smol_exact_qk_body<4,1>(q,k,s,t,row,r,head);break;case 1:smol_exact_qk_body<4,16>(q,k,s,t,row,r,head);break;case 2:smol_exact_qk_body<8,8>(q,k,s,t,row,r,head);break;case 3:smol_exact_qk_body<16,4>(q,k,s,t,row,r,head);break;}}
template<int L,int Chunk,int Cols>__device__ __forceinline__ void smol_exact_pv_body(const __nv_bfloat16*v,const __nv_bfloat16*p,__nv_bfloat16*out,int t,int row,int r,int head,float*partial){int tid=threadIdx.x,x=tid%Cols,lane=tid/Cols,col=blockIdx.x*Cols+x;if(blockIdx.x*Cols>=64)return;float value=0;if(lane<L&&col<64)for(int base=lane*Chunk;base<t;base+=L*Chunk){
#pragma unroll
 for(int j=0;j<Chunk;j++)if(base+j<t)value=__fmaf_rn(__bfloat162float(v[((r*3+head/3)*1024+base+j)*64+col]),__bfloat162float(p[(r*9+head)*1024+base+j]),value);}
 partial[tid]=value;__syncthreads();for(int off=L/2;off;off/=2){if(lane<off)partial[tid]=__fadd_rn(partial[tid],partial[(lane+off)*Cols+x]);__syncthreads();}if(lane==0&&col<64)out[(row*9+head)*64+col]=__float2bfloat16_rn(__fmaf_rn(1.f,partial[x],0.f));}
// Native GEMVX partition: each K lane owns ceil(T/L) consecutive products.
// Preserve ascending FMA inside each lane and the native descending half-tree.
template<int L,int Cols>__device__ __forceinline__ void smol_exact_balanced_pv(const __nv_bfloat16*v,const __nv_bfloat16*p,__nv_bfloat16*out,int t,int row,int r,int head,float*partial){
 int tid=threadIdx.x,x=tid%Cols,lane=tid/Cols,col=blockIdx.x*Cols+x;if(blockIdx.x*Cols>=64)return;int chunk=(t+L-1)/L,begin=lane*chunk,end=min(t,begin+chunk);float value=0;
 if(lane<L&&col<64)for(int k=begin;k<end;k++)value=__fmaf_rn(__bfloat162float(v[((r*3+head/3)*1024+k)*64+col]),__bfloat162float(p[(r*9+head)*1024+k]),value);
 partial[tid]=value;__syncthreads();for(int off=L/2;off;off/=2){if(lane<off)partial[tid]=__fadd_rn(partial[tid],partial[(lane+off)*Cols+x]);__syncthreads();}if(lane==0&&col<64)out[(row*9+head)*64+col]=__float2bfloat16_rn(__fmaf_rn(1.f,partial[x],0.f));
}
__global__ void smol_exact_pv(const __nv_bfloat16*v,const __nv_bfloat16*p,__nv_bfloat16*out,const uint32_t*positions,SmolRows info){int r=blockIdx.z,head=blockIdx.y,row=info.rows[r];uint64_t t=uint64_t(positions[row])+1;if(t>1024)return;__shared__ float partial[256];switch(smol_device_recipe[t][1]){
 case 7:smol_exact_balanced_pv<4,32>(v,p,out,t,row,r,head,partial);break;case 8:smol_exact_balanced_pv<32,8>(v,p,out,t,row,r,head,partial);break;
 case 0:smol_exact_pv_body<32,4,8>(v,p,out,t,row,r,head,partial);break;case 1:smol_exact_pv_body<32,1,8>(v,p,out,t,row,r,head,partial);break;case 2:smol_exact_pv_body<8,1,32>(v,p,out,t,row,r,head,partial);break;case 3:smol_exact_pv_body<16,1,16>(v,p,out,t,row,r,head,partial);break;case 4:smol_exact_pv_body<2,1,32>(v,p,out,t,row,r,head,partial);break;case 5:smol_exact_pv_body<4,1,32>(v,p,out,t,row,r,head,partial);break;case 6:smol_exact_pv_body<2,16,32>(v,p,out,t,row,r,head,partial);break;}}
__global__ void smol_exact_softmax(__nv_bfloat16*p,const uint32_t*positions,SmolRows info){int r=blockIdx.z,row=info.rows[r];uint64_t t=uint64_t(positions[row])+1;if(t>1024||smol_device_recipe[t][0]==255)return;int log=0;while((1<<log)<t)log++;int width=(1<<log)<32?(1<<log):32,warps=128/width,wb=log<=7?2:1,blocks=(9+warps*wb-1)/(warps*wb);if(blockIdx.x>=blocks)return;int tid=threadIdx.x;auto*base=p+r*9*1024;
#define SMOL_SM(L) case L:hf_wide_chunk_softmax_body<__nv_bfloat16,__nv_bfloat16,float,L,false,false,32,false,true,true>(base,base,9,1024,t,tid%width,tid/width,warps);break;
 switch(log){SMOL_SM(0)SMOL_SM(1)SMOL_SM(2)SMOL_SM(3)SMOL_SM(4)SMOL_SM(5)SMOL_SM(6)SMOL_SM(7)SMOL_SM(8)SMOL_SM(9)SMOL_SM(10)}
#undef SMOL_SM
}
}
extern "C" int riley_cuda_cublas_hf_smol_exact_supported(const RileyCudaCublasGemmProbePlan*plan,uint64_t tokens) noexcept {return plan&&plan->owned_smol_attention_qualified&&tokens>0&&tokens<=1024&&smol_host_recipe[tokens][0]!=255&&(smol_host_recipe[tokens][0]!=4||plan->owned_attention_extended_qualified);}
extern "C" RileyCudaStatus riley_cuda_cublas_hf_smol_exact_execute(RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,const uint32_t*rows,const uint64_t*tokens,uint64_t requests,uint64_t active,uint64_t physical,uint64_t sequences,uint64_t ids,RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);if(!plan||!plan->handle||!plan->owned_smol_attention_qualified||plan->config.k!=64||!raw||count!=11||!rows||!tokens||requests<1||requests>32||active==0||active>8192||active>plan->config.m||physical==0||physical>UINT32_MAX||sequences==0||sequences>UINT32_MAX||ids==0||ids>UINT32_MAX)return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified H9KV3D64 exact attention requiresR1..32 and11spans");
 SmolRows info{};info.count=requests;for(uint64_t r=0;r<requests;r++){if(rows[r]>=active||(r&&rows[r]<=rows[r-1])||!riley_cuda_cublas_hf_smol_exact_supported(plan,tokens[r])||tokens[r]>plan->config.n)return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"sorted distinct active singleton rows and eligible T within prepared bounds required");info.rows[r]=rows[r];}
 const uint64_t q=active*576*2,pool=physical*3*16*64*2,kv=requests*3*1024*64*2,scores=requests*9*1024*2;const uint64_t sizes[11]={q,pool,pool,ids*4,q,kv,kv,scores,active*4,(sequences+1)*4,active*4};ResolvedSpan spans[11]{};RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<11;i++){uint32_t dtype=(i==3||i>=8)?RILEY_CUDA_DTYPE_U32:RILEY_CUDA_DTYPE_BF16;status=resolve_attention_span(raw+i,dtype,sizes[i],spans+i,error,"exact Smol span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 // Captured kernels outlive the active CSR prefix. Bound their dynamic reads
 // by owned allocation capacity, while ordinary calls retain exact active bounds.
 // Expand alias validation too: no future replay may read another live span.
 uint64_t graph_sequences=sequences,graph_ids=ids;
 if(riley_cuda_internal::thread_has_active_step_capture()){
  for(size_t i: {size_t(3),size_t(9)}){
   const uint64_t remaining=spans[i].buffer->byte_len-spans[i].byte_offset;
   if(remaining%4 || remaining/4>UINT32_MAX || remaining<spans[i].byte_len)
    return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"captured CSR allocation capacity must be aligned and bounded");
   spans[i].byte_len=remaining;
  }
  graph_ids=spans[3].byte_len/4;graph_sequences=spans[9].byte_len/4-1;
 }
 status=validate_span_relationships(plan,stream,spans,11,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;ExclusiveProbeUses uses(plan,stream);for(auto&span:spans)if(!uses.add(span.buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"Smol use ledger overflow");status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;auto*k=static_cast<__nv_bfloat16*>(spans[5].data);auto*v=static_cast<__nv_bfloat16*>(spans[6].data);auto*p=static_cast<__nv_bfloat16*>(spans[7].data);auto*positions=static_cast<const uint32_t*>(spans[10].data);
 smol_exact_gather<<<dim3(32,3,requests),256,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[1].data),static_cast<const __nv_bfloat16*>(spans[2].data),static_cast<const uint32_t*>(spans[3].data),static_cast<const uint32_t*>(spans[8].data),positions,static_cast<const uint32_t*>(spans[9].data),k,v,physical,graph_sequences,graph_ids,info);
 smol_exact_qk<<<dim3(8,9,requests),128,0,stream->stream>>>(static_cast<const __nv_bfloat16*>(spans[0].data),k,p,positions,info);smol_exact_softmax<<<dim3(3,1,requests),128,0,stream->stream>>>(p,positions,info);smol_exact_pv<<<dim3(8,9,requests),256,0,stream->stream>>>(v,p,static_cast<__nv_bfloat16*>(spans[4].data),positions,info);auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);}
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}


extern "C" int riley_cuda_cublas_hf_smol_fused_qkv_supported(const RileyCudaCublasGemmProbePlan*plan,uint64_t rows) noexcept {
 return plan&&plan->owned_attention_extended_qualified&&plan->config.n==576&&plan->config.k==576&&rows>=1&&rows<=32&&rows<=plan->config.m;
}
extern "C" RileyCudaStatus riley_cuda_cublas_hf_smol_fused_qkv_execute(RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,uint64_t rows,RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!riley_cuda_cublas_hf_smol_fused_qkv_supported(plan,rows)||!plan->handle||!raw||count!=7)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified Smol fused QKV requires R1..32 and seven BF16 spans");
 const uint64_t sizes[7]={576*576*2,192*576*2,192*576*2,rows*576*2,rows*576*2,rows*192*2,rows*192*2};ResolvedSpan spans[7]{};
 RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<7;i++){status=resolve_attention_span(raw+i,RILEY_CUDA_DTYPE_BF16,sizes[i],spans+i,error,"Smol fused QKV span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 for(size_t i=0;i<4;i++)if(reinterpret_cast<uintptr_t>(spans[i].data)&uintptr_t(15))
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"Smol fused QKV vector sources require16-byte alignment");
 status=validate_span_relationships(plan,stream,spans,7,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(auto&span:spans)if(!uses.add(span.buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"Smol fused QKV use ledger overflow");
 status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;
  auto*wq=static_cast<const unsigned short*>(spans[0].data);auto*wk=static_cast<const unsigned short*>(spans[1].data);auto*wv=static_cast<const unsigned short*>(spans[2].data);auto*x=static_cast<const unsigned short*>(spans[3].data);
  auto*yq=static_cast<unsigned short*>(spans[4].data);auto*yk=static_cast<unsigned short*>(spans[5].data);auto*yv=static_cast<unsigned short*>(spans[6].data);
  if(rows<=2)launch_hf_smol_qkv_tile<1>(wq,wk,wv,x,yq,yk,yv,rows,stream->stream);
  else if(rows<=12)launch_hf_smol_qkv_tile<4>(wq,wk,wv,x,yq,yk,yv,rows,stream->stream);
  else launch_hf_smol_qkv_tile<8>(wq,wk,wv,x,yq,yk,yv,rows,stream->stream);
  auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 }
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}


extern "C" int riley_cuda_cublas_hf_smol_fused_gate_up_supported(const RileyCudaCublasGemmProbePlan*plan,uint64_t rows) noexcept {
 return plan&&plan->owned_attention_extended_qualified&&plan->config.n==1536&&plan->config.k==576&&rows>=1&&rows<=32&&rows<=plan->config.m;
}
extern "C" RileyCudaStatus riley_cuda_cublas_hf_smol_fused_gate_up_execute(RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,uint64_t rows,RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!riley_cuda_cublas_hf_smol_fused_gate_up_supported(plan,rows)||!plan->handle||!raw||count!=4)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified Smol GateUp requires R1..32 and four BF16 spans");
 const uint64_t sizes[4]={1536*576*2,1536*576*2,rows*576*2,rows*1536*2};ResolvedSpan spans[4]{};RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<4;i++){status=resolve_attention_span(raw+i,RILEY_CUDA_DTYPE_BF16,sizes[i],spans+i,error,"Smol GateUp span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 for(size_t i=0;i<3;i++)if(reinterpret_cast<uintptr_t>(spans[i].data)&uintptr_t(15))return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"Smol GateUp vector sources require16-byte alignment");
 status=validate_span_relationships(plan,stream,spans,4,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(auto&span:spans)if(!uses.add(span.buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"Smol GateUp use ledger overflow");
 status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;auto*wg=static_cast<const unsigned short*>(spans[0].data);auto*wu=static_cast<const unsigned short*>(spans[1].data);auto*x=static_cast<const unsigned short*>(spans[2].data);auto*y=static_cast<unsigned short*>(spans[3].data);
  if(rows<=2)v102_launch<1>(wg,wu,x,y,rows,stream->stream);else if(rows<=12)v102_launch<4>(wg,wu,x,y,rows,stream->stream);else v102_launch<8>(wg,wu,x,y,rows,stream->stream);
  auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 }
 return complete_execution(&uses,&scope,stream,status,attempted,error);
}


extern "C" int riley_cuda_cublas_hf_sm80_fused_gate_up_supported(const RileyCudaCublasGemmProbePlan*plan,uint64_t rows) noexcept {
 return plan&&plan->owned_sm80_gateup_rows_qualified&&plan->owned_sm80_decode_mlp_v8_enabled&&plan->config.n==9728&&plan->config.k==2560&&rows>=2&&rows<=32&&rows<=plan->config.m;
}
extern "C" RileyCudaStatus riley_cuda_cublas_hf_sm80_fused_gate_up_execute(RileyCudaCublasGemmProbePlan*plan,const RileyCudaBufferSpan*raw,uint64_t count,uint64_t rows,RileyCudaStream*stream,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!riley_cuda_cublas_hf_sm80_fused_gate_up_supported(plan,rows)||!plan->handle||!raw||count!=4)
  return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"qualified SM80 GateUp requires R2..32 and four BF16 spans");
 const uint64_t sizes[4]={9728*2560*2,9728*2560*2,rows*2560*2,rows*9728*2};ResolvedSpan spans[4]{};RileyCudaStatus status=RILEY_CUDA_STATUS_SUCCESS;
 for(size_t i=0;i<4;i++){status=resolve_attention_span(raw+i,RILEY_CUDA_DTYPE_BF16,sizes[i],spans+i,error,"SM80 GateUp span geometry mismatch");if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;}
 for(size_t i=0;i<3;i++)if(reinterpret_cast<uintptr_t>(spans[i].data)&uintptr_t(31))return validation_error(error,RILEY_CUDA_STATUS_INVALID_ARGUMENT,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"SM80 GateUp vector sources require32-byte alignment");
 status=validate_span_relationships(plan,stream,spans,4,error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;
 ExclusiveProbeUses uses(plan,stream);for(auto&span:spans)if(!uses.add(span.buffer))return internal_error(error,RILEY_CUDA_ERROR_STAGE_VALIDATION,kExecuteOperation,"SM80 GateUp use ledger overflow");
 status=uses.acquire(error);if(status!=RILEY_CUDA_STATUS_SUCCESS)return status;CurrentContext scope(plan->owner);status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);bool attempted=false;
 if(status==RILEY_CUDA_STATUS_SUCCESS){attempted=true;
  auto*wg=static_cast<const __nv_bfloat16*>(spans[0].data);auto*wu=static_cast<const __nv_bfloat16*>(spans[1].data);auto*x=static_cast<const __nv_bfloat16*>(spans[2].data);auto*y=static_cast<__nv_bfloat16*>(spans[3].data);
  if(rows<=16)sm80_fused_gateup<4,16,16,2><<<304,128,0,stream->stream>>>(wg,wu,x,y,rows);
  else sm80_fused_gateup<4,32,8,2><<<304,128,0,stream->stream>>>(wg,wu,x,y,rows);
  auto e=cudaGetLastError();if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_LAUNCH,kExecuteOperation);
 }

 return complete_execution(&uses,&scope,stream,status,attempted,error);
}


extern "C" RileyCudaStatus riley_cuda_cublas_gemm_probe_plan_prepare_sm80_paged_attention(RileyCudaCublasGemmProbePlan*plan,RileyCudaErrorInfo*error) noexcept {
 clear_error(error);
 if(!plan || !plan->owned_sm80_attention_enabled || !plan->owned_sm80_gateup_rows_qualified || plan->config.k!=128 || plan->active_uses.load()!=0)
  return validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"cold qualified SM80 attention owner required");
 CurrentContext scope(plan->owner);auto status=scope.enter(error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
 for(const void*fn:{reinterpret_cast<const void*>(sm80_chunk_paged_wmma_qk),reinterpret_cast<const void*>(sm80_chunk_paged_wmma_pv),reinterpret_cast<const void*>(sm80_batched_qk_dispatch),reinterpret_cast<const void*>(sm80_batched_pv_dispatch),reinterpret_cast<const void*>(sm80_batched_softmax)})if(status==RILEY_CUDA_STATUS_SUCCESS){cudaFuncAttributes a{};auto e=cudaFuncGetAttributes(&a,fn);if(e!=cudaSuccess)status=runtime_error(e,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);else if(a.binaryVersion<80||a.ptxVersion<80)status=validation_error(error,RILEY_CUDA_STATUS_NOT_SUPPORTED,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation,"native SM80 paged image required");}
 if(status==RILEY_CUDA_STATUS_SUCCESS)plan->owned_sm80_paged_attention_enabled=true;
 return scope.leave(status,error,RILEY_CUDA_ERROR_STAGE_PREPARE,kCreateOperation);
}
extern "C" int riley_cuda_cublas_hf_sm80_paged_qualified(const RileyCudaCublasGemmProbePlan*plan,uint64_t tokens) noexcept {return plan && plan->owned_sm80_paged_attention_enabled && (tokens==0 || (tokens<=plan->config.n && sm80_paged_qk_family(tokens)>=0));}

#endif // RILEY_CUDA_ENABLE_CUBLAS_GEMM_PROBE
