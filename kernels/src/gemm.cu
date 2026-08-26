#include "ffi_internal.hpp"

#include <cublasLt.h>

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
constexpr int kMaximumHeuristicResults = 32;
constexpr size_t kMaximumGemmBuffers = 4;

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

struct RustInferCudaGemmPlan {
  RustInferCudaGemmPlan(RustInferCudaContext* owning_context,
                        const RustInferCudaGemmConfig& plan_config,
                        const GemmByteLengths& lengths) noexcept
      : owner(owning_context),
        config(plan_config),
        algorithm_info{},
        handle(nullptr),
        operation(nullptr),
        weight_layout(nullptr),
        input_layout(nullptr),
        output_layout(nullptr),
        preference(nullptr),
        algorithm{},
        input_bytes(lengths.input),
        weight_bytes(lengths.weight),
        output_bytes(lengths.output),
        algorithm_ready(false),
        active_uses(0) {
    algorithm_info.struct_size = sizeof(algorithm_info);
  }

  RustInferCudaContext* owner;
  RustInferCudaGemmConfig config;
  RustInferCudaGemmAlgorithmInfo algorithm_info;
  cublasLtHandle_t handle;
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t weight_layout;
  cublasLtMatrixLayout_t input_layout;
  cublasLtMatrixLayout_t output_layout;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulAlgo_t algorithm;
  uint64_t input_bytes;
  uint64_t weight_bytes;
  uint64_t output_bytes;
  bool algorithm_ready;
  std::atomic<uint32_t> active_uses;
};

namespace {

using rustinfer_cuda_internal::CurrentContext;
using rustinfer_cuda_internal::clear_error;
using rustinfer_cuda_internal::command_batch_is_active;
using rustinfer_cuda_internal::command_batch_is_owned_by_current_thread;
using rustinfer_cuda_internal::command_batch_register_use;
using rustinfer_cuda_internal::driver_error;
using rustinfer_cuda_internal::internal_error;
using rustinfer_cuda_internal::release_child;
using rustinfer_cuda_internal::release_exclusive_use;
using rustinfer_cuda_internal::retain_child;
using rustinfer_cuda_internal::runtime_error;
using rustinfer_cuda_internal::same_context;
using rustinfer_cuda_internal::set_error;
using rustinfer_cuda_internal::try_acquire_exclusive_use;
using rustinfer_cuda_internal::validation_error;

const char* cublaslt_status_detail(cublasStatus_t status) noexcept {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "cuBLASLt was not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "cuBLASLt allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "cuBLASLt received an invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "the CUDA device architecture is unsupported";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "cuBLASLt could not map a resource";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "cuBLASLt execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "cuBLASLt reported an internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "the requested cuBLASLt operation is unsupported";
    default:
      return "cuBLASLt reported an unknown error";
  }
}

RustInferCudaStatus cublaslt_error(cublasStatus_t result,
                                   RustInferCudaErrorInfo* error,
                                   uint32_t stage,
                                   const char* operation) noexcept {
  if (result == CUBLAS_STATUS_SUCCESS) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  RustInferCudaStatus status = RUSTINFER_CUDA_STATUS_CUBLASLT_ERROR;
  if (result == CUBLAS_STATUS_INVALID_VALUE) {
    status = RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT;
  } else if (result == CUBLAS_STATUS_ALLOC_FAILED) {
    status = RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY;
  } else if (result == CUBLAS_STATUS_ARCH_MISMATCH ||
             result == CUBLAS_STATUS_NOT_SUPPORTED) {
    status = RUSTINFER_CUDA_STATUS_NOT_SUPPORTED;
  }
  return set_error(error, status, static_cast<int32_t>(result),
                   RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT, stage, operation,
                   cublaslt_status_detail(result));
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

RustInferCudaStatus matrix_bytes(uint64_t rows, uint64_t columns,
                                 uint64_t* output,
                                 RustInferCudaErrorInfo* error) noexcept {
  uint64_t elements = 0;
  if (!checked_multiply(rows, columns, &elements) ||
      !checked_multiply(elements, kBfloat16Bytes, output)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "validate cuBLASLt GEMM config",
                            "matrix byte length overflows uint64_t");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_config(const RustInferCudaGemmConfig* config,
                                    GemmByteLengths* lengths,
                                    RustInferCudaErrorInfo* error) noexcept {
  if (config == nullptr || lengths == nullptr ||
      config->struct_size < sizeof(*config)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
        "validate cuBLASLt GEMM config",
        "config is null or has an incompatible struct_size");
  }
  if ((config->flags &
       ~RUSTINFER_CUDA_GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K) != 0 ||
      config->reserved0 != 0 ||
      !reserved_is_zero(config->reserved, 3)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "validate cuBLASLt GEMM config",
                            "unknown flags and reserved fields must be zero");
  }
  if (config->m == 0 || config->n == 0 || config->k == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "validate cuBLASLt GEMM config",
                            "M, N, and K must all be non-zero");
  }
  const uint64_t maximum_dimension =
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
  if (config->m > maximum_dimension || config->n > maximum_dimension ||
      config->k > maximum_dimension) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "validate cuBLASLt GEMM config",
                            "M, N, and K must fit cuBLASLt int32 dimensions");
  }
  if (config->input_dtype != RUSTINFER_CUDA_DTYPE_BF16 ||
      config->weight_dtype != RUSTINFER_CUDA_DTYPE_BF16 ||
      config->accumulator_dtype != RUSTINFER_CUDA_DTYPE_F32 ||
      config->output_dtype != RUSTINFER_CUDA_DTYPE_BF16) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
        "validate cuBLASLt GEMM config",
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
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
        "validate cuBLASLt GEMM config",
        "only row-major X=N/W=T, epilogue-none, deterministic GEMM is supported");
  }
  if (config->max_workspace_bytes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "validate cuBLASLt GEMM config",
                            "workspace cap exceeds native size_t");
  }

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

template <typename T>
bool algorithm_config_value(
    const cublasLtMatmulAlgo_t* algorithm,
    cublasLtMatmulAlgoConfigAttributes_t attribute, T* output) noexcept {
  if (algorithm == nullptr || output == nullptr) {
    return false;
  }
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
             algorithm, attribute, output, sizeof(*output), &written) ==
             CUBLAS_STATUS_SUCCESS &&
         written == sizeof(*output);
}

template <typename T>
bool algorithm_capability_value(
    const cublasLtMatmulAlgo_t* algorithm,
    cublasLtMatmulAlgoCapAttributes_t attribute, T* output) noexcept {
  if (algorithm == nullptr || output == nullptr) {
    return false;
  }
  size_t written = 0;
  return cublasLtMatmulAlgoCapGetAttribute(
             algorithm, attribute, output, sizeof(*output), &written) ==
             CUBLAS_STATUS_SUCCESS &&
         written == sizeof(*output);
}

bool algorithm_has_supported_alignment(
    const cublasLtMatmulAlgo_t* algorithm) noexcept {
  uint32_t alignment_a = 0;
  uint32_t alignment_b = 0;
  uint32_t alignment_c = 0;
  uint32_t alignment_d = 0;
  if (!algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
          &alignment_a) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
          &alignment_b) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
          &alignment_c) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
          &alignment_d)) {
    return false;
  }
  const auto is_satisfied = [](uint32_t alignment) noexcept {
    return alignment == 0 || kRequiredAlignment % alignment == 0;
  };
  return is_satisfied(alignment_a) && is_satisfied(alignment_b) &&
         is_satisfied(alignment_c) && is_satisfied(alignment_d);
}

bool deterministic_reduction_configuration(
    const RustInferCudaGemmConfig& config, uint32_t split_k,
    uint32_t scheme) noexcept {
  if (split_k <= 1) {
    return scheme ==
           static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE);
  }
  return (config.flags &
          RUSTINFER_CUDA_GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K) != 0 &&
         scheme ==
         static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE);
}

bool deterministic_candidate(
    RustInferCudaGemmPlan* plan,
    const cublasLtMatmulHeuristicResult_t& candidate,
    cublasLtMatmulAlgo_t* algorithm, size_t* workspace_bytes) noexcept {
  if (plan == nullptr || algorithm == nullptr || workspace_bytes == nullptr ||
      candidate.state != CUBLAS_STATUS_SUCCESS) {
    return false;
  }

  *algorithm = candidate.algo;
  *workspace_bytes = candidate.workspaceSize;

  uint32_t split_k = 0;
  uint32_t reduction_scheme = 0;
  if (!algorithm_config_value(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                              &split_k) ||
      !algorithm_config_value(algorithm,
                              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                              &reduction_scheme)) {
    return false;
  }
  if (deterministic_reduction_configuration(plan->config, split_k,
                                            reduction_scheme)) {
    return true;
  }

  const uint32_t deterministic_split_k = 1;
  const uint32_t deterministic_reduction_scheme =
      static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE);
  if (cublasLtMatmulAlgoConfigSetAttribute(
          algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
          &deterministic_split_k, sizeof(deterministic_split_k)) !=
          CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulAlgoConfigSetAttribute(
          algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
          &deterministic_reduction_scheme,
          sizeof(deterministic_reduction_scheme)) != CUBLAS_STATUS_SUCCESS) {
    return false;
  }

  cublasLtMatmulHeuristicResult_t checked{};
  if (cublasLtMatmulAlgoCheck(
          plan->handle, plan->operation, plan->weight_layout,
          plan->input_layout, plan->output_layout, plan->output_layout,
          algorithm, &checked) != CUBLAS_STATUS_SUCCESS ||
      checked.state != CUBLAS_STATUS_SUCCESS) {
    return false;
  }
  *workspace_bytes = checked.workspaceSize;
  return true;
}

RustInferCudaStatus query_plan_environment(
    RustInferCudaGemmPlan* plan, RustInferCudaErrorInfo* error) noexcept {
  int capability_major = 0;
  CUresult driver_result = cuDeviceGetAttribute(
      &capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error,
                        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                        "prepare cuBLASLt GEMM plan");
  }
  int capability_minor = 0;
  driver_result = cuDeviceGetAttribute(
      &capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error,
                        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                        "prepare cuBLASLt GEMM plan");
  }
  if (capability_major < 0 || capability_minor < 0) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare cuBLASLt GEMM plan",
                          "CUDA returned a negative compute capability");
  }

  int runtime_version = 0;
  const cudaError_t runtime_result = cudaRuntimeGetVersion(&runtime_version);
  if (runtime_result != cudaSuccess) {
    return runtime_error(runtime_result, error,
                         RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                         "prepare cuBLASLt GEMM plan");
  }
  const size_t cublaslt_version = cublasLtGetVersion();
  if (cublaslt_version >
      static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare cuBLASLt GEMM plan",
                          "cuBLASLt version exceeds ABI metadata range");
  }

  plan->algorithm_info.backend = RUSTINFER_CUDA_GEMM_BACKEND_CUBLASLT;
  plan->algorithm_info.deterministic =
      RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED;
  plan->algorithm_info.compute_capability_major =
      static_cast<uint32_t>(capability_major);
  plan->algorithm_info.compute_capability_minor =
      static_cast<uint32_t>(capability_minor);
  plan->algorithm_info.runtime_version = runtime_version;
  plan->algorithm_info.cublaslt_version =
      static_cast<int32_t>(cublaslt_version);
  plan->algorithm_info.m = plan->config.m;
  plan->algorithm_info.n = plan->config.n;
  plan->algorithm_info.k = plan->config.k;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus select_deterministic_algorithm(
    RustInferCudaGemmPlan* plan, RustInferCudaErrorInfo* error) noexcept {
  cublasLtMatmulHeuristicResult_t
      candidates[kMaximumHeuristicResults]{};
  int returned_results = 0;
  RustInferCudaStatus status = cublaslt_error(
      cublasLtMatmulAlgoGetHeuristic(
          plan->handle, plan->operation, plan->weight_layout,
          plan->input_layout, plan->output_layout, plan->output_layout,
          plan->preference, kMaximumHeuristicResults, candidates,
          &returned_results),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (returned_results < 0 ||
      returned_results > kMaximumHeuristicResults) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare cuBLASLt GEMM plan",
                          "cuBLASLt returned an invalid heuristic count");
  }

  for (int index = 0; index < returned_results; ++index) {
    const cublasLtMatmulHeuristicResult_t& candidate = candidates[index];
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;
    if (!deterministic_candidate(plan, candidate, &algorithm,
                                 &workspace_bytes) ||
        workspace_bytes > plan->config.max_workspace_bytes ||
        !algorithm_has_supported_alignment(&algorithm)) {
      continue;
    }

    int32_t algorithm_id = 0;
    uint32_t tile_id = 0;
    uint32_t stages_id = 0;
    uint32_t split_k = 0;
    uint32_t reduction_scheme = 0;
    uint32_t cta_swizzling = 0;
    uint32_t custom_option = 0;
    uint64_t numerical_flags = 0;
    if (!algorithm_config_value(&algorithm, CUBLASLT_ALGO_CONFIG_ID,
                                &algorithm_id) ||
        !algorithm_config_value(&algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID,
                                &tile_id) ||
        !algorithm_config_value(&algorithm,
                                CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                &stages_id) ||
        !algorithm_config_value(&algorithm,
                                CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                &split_k) ||
        !algorithm_config_value(&algorithm,
                                CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                &reduction_scheme) ||
        !algorithm_config_value(&algorithm,
                                CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                &cta_swizzling) ||
        !algorithm_config_value(&algorithm,
                                CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                &custom_option) ||
        !algorithm_capability_value(
            &algorithm, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
            &numerical_flags)) {
      continue;
    }
    if (!deterministic_reduction_configuration(
            plan->config, split_k, reduction_scheme)) {
      continue;
    }

    plan->algorithm = algorithm;
    plan->algorithm_info.algorithm_id = algorithm_id;
    plan->algorithm_info.tile_id = tile_id;
    plan->algorithm_info.stages_id = stages_id;
    plan->algorithm_info.split_k = split_k;
    plan->algorithm_info.reduction_scheme = reduction_scheme;
    plan->algorithm_info.cta_swizzling = cta_swizzling;
    plan->algorithm_info.custom_option = custom_option;
    plan->algorithm_info.workspace_bytes =
        static_cast<uint64_t>(workspace_bytes);
    plan->algorithm_info.numerical_implementation_flags = numerical_flags;
    plan->algorithm_ready = true;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  return set_error(
      error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
      RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT,
      RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan",
      "no deterministic algorithm satisfies the workspace and 256-byte alignment contract");
}

RustInferCudaStatus prepare_plan(RustInferCudaGemmPlan* plan,
                                 RustInferCudaErrorInfo* error) noexcept {
  RustInferCudaStatus status = cublaslt_error(
      cublasLtCreate(&plan->handle), error,
      RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = cublaslt_error(
      cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const cublasOperation_t transpose_weight = CUBLAS_OP_T;
  const cublasOperation_t transpose_input = CUBLAS_OP_N;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  const cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  status = cublaslt_error(
      cublasLtMatmulDescSetAttribute(
          plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_weight,
          sizeof(transpose_weight)),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_input,
            sizeof(transpose_input)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
            sizeof(epilogue)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulDescSetAttribute(
            plan->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
            &pointer_mode, sizeof(pointer_mode)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  // Row-major W[N,K], X[M,K], and Y[M,N] have the same bytes as
  // column-major Wc[K,N], Xc[K,M], and Yc[N,M]. The TN(Wc, Xc) operation
  // therefore produces Y without any packing or byte rearrangement.
  status = cublaslt_error(
      cublasLtMatrixLayoutCreate(&plan->weight_layout, CUDA_R_16BF,
                                 plan->config.k, plan->config.n,
                                 plan->config.k),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatrixLayoutCreate(&plan->input_layout, CUDA_R_16BF,
                                   plan->config.k, plan->config.m,
                                   plan->config.k),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatrixLayoutCreate(&plan->output_layout, CUDA_R_16BF,
                                   plan->config.n, plan->config.m,
                                   plan->config.n),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  cublasLtMatrixLayout_t layouts[] = {
      plan->weight_layout, plan->input_layout, plan->output_layout};
  for (cublasLtMatrixLayout_t layout : layouts) {
    status = cublaslt_error(
        cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare cuBLASLt GEMM plan");
    if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
      return status;
    }
  }

  status = cublaslt_error(
      cublasLtMatmulPreferenceCreate(&plan->preference), error,
      RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  const size_t max_workspace =
      static_cast<size_t>(plan->config.max_workspace_bytes);
  status = cublaslt_error(
      cublasLtMatmulPreferenceSetAttribute(
          plan->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &max_workspace, sizeof(max_workspace)),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare cuBLASLt GEMM plan");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  status = query_plan_environment(plan, error);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = select_deterministic_algorithm(plan, error);
  }
  return status;
}

RustInferCudaStatus destroy_plan_resources(
    RustInferCudaGemmPlan* plan, RustInferCudaErrorInfo* error,
    uint32_t stage, const char* operation) noexcept {
  if (plan->preference != nullptr) {
    const cublasStatus_t result =
        cublasLtMatmulPreferenceDestroy(plan->preference);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->preference = nullptr;
  }
  if (plan->output_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(plan->output_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->output_layout = nullptr;
  }
  if (plan->input_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(plan->input_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->input_layout = nullptr;
  }
  if (plan->weight_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(plan->weight_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->weight_layout = nullptr;
  }
  if (plan->operation != nullptr) {
    const cublasStatus_t result = cublasLtMatmulDescDestroy(plan->operation);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->operation = nullptr;
  }
  if (plan->handle != nullptr) {
    const cublasStatus_t result = cublasLtDestroy(plan->handle);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    plan->handle = nullptr;
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

bool plan_resources_destroyed(const RustInferCudaGemmPlan* plan) noexcept {
  return plan->preference == nullptr && plan->output_layout == nullptr &&
         plan->input_layout == nullptr && plan->weight_layout == nullptr &&
         plan->operation == nullptr && plan->handle == nullptr;
}

RustInferCudaStatus resolve_exact_span(
    const RustInferCudaBufferSpan* span, RustInferCudaDType required_dtype,
    uint64_t required_bytes, ResolvedSpan* output,
    RustInferCudaErrorInfo* error, const char* name) noexcept {
  if (span == nullptr || output == nullptr ||
      span->struct_size < sizeof(*span)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "a span is null or has an incompatible struct_size");
  }
  if (!reserved_is_zero(span->reserved, 2)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "span reserved fields must be zero");
  }
  if (span->dtype != required_dtype) {
    return set_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_VALIDATION,
                     RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                     "execute cuBLASLt GEMM", name);
  }
  if (span->buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "a span buffer handle is null");
  }
  if (span->byte_offset % kRequiredAlignment != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "every span byte_offset must be 256-byte aligned");
  }
  if (span->byte_len != required_bytes) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "a span byte_len does not exactly match the prepared requirement");
  }
  if (span->byte_offset > span->buffer->byte_len ||
      span->byte_len > span->buffer->byte_len - span->byte_offset) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "a declared span exceeds its opaque allocation");
  }
  if (required_bytes != 0 && span->buffer->device_data == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
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
    RustInferCudaGemmPlan* plan, RustInferCudaStream* stream,
    const ResolvedSpan* spans, size_t count,
    RustInferCudaErrorInfo* error) noexcept {
  if (plan->owner == nullptr || stream == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "plan owner or stream is null");
  }
  if (!same_context(plan->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "plan and stream belong to different context owners");
  }
  for (size_t index = 0; index < count; ++index) {
    if (!same_context(plan->owner, spans[index].buffer->owner)) {
      return validation_error(
          error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
          RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
          "execute cuBLASLt GEMM",
          "plan and device spans belong to different context owners");
    }
    for (size_t other = index + 1; other < count; ++other) {
      if (spans_overlap(spans[index], spans[other])) {
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
            "execute cuBLASLt GEMM",
            "input, weight, output, and workspace spans must not overlap");
      }
    }
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

class ExclusiveGemmUses final {
 public:
  ExclusiveGemmUses(RustInferCudaGemmPlan* plan,
                    RustInferCudaStream* stream) noexcept
      : plan_(plan),
        stream_(stream),
        buffers_{},
        buffer_count_(0),
        acquired_buffers_(0),
        plan_acquired_(false),
        stream_acquired_(false),
        command_batch_(false) {}

  ExclusiveGemmUses(const ExclusiveGemmUses&) = delete;
  ExclusiveGemmUses& operator=(const ExclusiveGemmUses&) = delete;

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
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
            "execute cuBLASLt GEMM",
            "an active stream command batch is owned by another thread");
      }
      command_batch_ = true;
      RustInferCudaStatus status = command_batch_register_use(
          stream_, &plan_->active_uses, error, "execute cuBLASLt GEMM",
          "the GEMM plan already has an active use");
      if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
        return status;
      }
      for (size_t index = 0; index < buffer_count_; ++index) {
        status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error,
            "execute cuBLASLt GEMM",
            "a GEMM device buffer already has an active use");
        if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
          return status;
        }
      }
      return RUSTINFER_CUDA_STATUS_SUCCESS;
    }
    if (!try_acquire_exclusive_use(plan_->active_uses)) {
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              "execute cuBLASLt GEMM",
                              "the GEMM plan already has an active use");
    }
    plan_acquired_ = true;
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        if (!release_acquired()) {
          return internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                "execute cuBLASLt GEMM",
                                "exclusive-use rollback was corrupted");
        }
        return validation_error(
            error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
            "execute cuBLASLt GEMM",
            "a GEMM device buffer already has an active use");
      }
      ++acquired_buffers_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      if (!release_acquired()) {
        return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              "execute cuBLASLt GEMM",
                              "exclusive-use rollback was corrupted");
      }
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              "execute cuBLASLt GEMM",
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

  RustInferCudaGemmPlan* plan_;
  RustInferCudaStream* stream_;
  RustInferCudaDeviceBuffer* buffers_[kMaximumGemmBuffers];
  size_t buffer_count_;
  size_t acquired_buffers_;
  bool plan_acquired_;
  bool stream_acquired_;
  bool command_batch_;
};

RustInferCudaStatus complete_execution(
    ExclusiveGemmUses* uses, CurrentContext* scope,
    RustInferCudaStream* stream, RustInferCudaStatus operation_status,
    bool matmul_attempted, RustInferCudaErrorInfo* error) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                        "execute cuBLASLt GEMM");
  }
  bool completion_confirmed = !matmul_attempted;
  RustInferCudaStatus status = operation_status;
  if (matmul_attempted) {
    const cudaError_t synchronize_result =
        cudaStreamSynchronize(stream->stream);
    completion_confirmed = synchronize_result == cudaSuccess;
    if (!completion_confirmed) {
      status = runtime_error(synchronize_result, error,
                             RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                             "execute cuBLASLt GEMM");
    }
  }
  status = scope->leave(status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                        "execute cuBLASLt GEMM");

  const bool restoration_confirmed =
      !stream->owner->restoration_failed.load(std::memory_order_acquire);
  if (completion_confirmed && restoration_confirmed) {
    if (!uses->release_completed()) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE,
                            "execute cuBLASLt GEMM",
                            "exclusive-use accounting was corrupted");
    }
  }
  return status;
}

}  // namespace

extern "C" RustInferCudaStatus rustinfer_cuda_gemm_plan_create(
    RustInferCudaContext* context, const RustInferCudaGemmConfig* config,
    RustInferCudaGemmPlan** out_plan,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (out_plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create cuBLASLt GEMM plan",
                            "out_plan is null");
  }
  *out_plan = nullptr;
  if (context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "create cuBLASLt GEMM plan",
                            "context is null");
  }

  GemmByteLengths lengths{};
  RustInferCudaStatus status = validate_config(config, &lengths, error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  RustInferCudaGemmConfig normalized_config = *config;
  normalized_config.struct_size = sizeof(normalized_config);

  void* storage = std::calloc(1, sizeof(RustInferCudaGemmPlan));
  if (storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                     "create cuBLASLt GEMM plan",
                     "host plan allocation failed");
  }
  auto* plan = new (storage)
      RustInferCudaGemmPlan(context, normalized_config, lengths);
  if (!retain_child(context)) {
    plan->~RustInferCudaGemmPlan();
    std::free(plan);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          "create cuBLASLt GEMM plan",
                          "context child-resource counter overflow");
  }

  CurrentContext scope(context);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                       "prepare cuBLASLt GEMM plan");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_plan(plan, error);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    (void)destroy_plan_resources(plan, nullptr,
                                 RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                                 "cleanup failed cuBLASLt GEMM plan");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                       "prepare cuBLASLt GEMM plan");

  const bool restoration_confirmed =
      !context->restoration_failed.load(std::memory_order_acquire);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && restoration_confirmed) {
    *out_plan = plan;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  if (plan_resources_destroyed(plan) && restoration_confirmed) {
    if (!release_child(context)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                            "cleanup failed cuBLASLt GEMM plan",
                            "context child-resource counter underflow");
    }
    plan->~RustInferCudaGemmPlan();
    std::free(plan);
  }
  // Any ambiguous descriptor destruction or context restoration deliberately
  // retains the unreachable wrapper and context-child lease fail closed.
  return status;
}

extern "C" RustInferCudaStatus rustinfer_cuda_gemm_plan_info(
    RustInferCudaGemmPlan* plan, RustInferCudaGemmAlgorithmInfo* out_info,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
        "query cuBLASLt GEMM plan",
        "plan or out_info is null, or struct_size is incompatible");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  out_info->struct_size = sizeof(*out_info);
  if (!try_acquire_exclusive_use(plan->active_uses)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                            "query cuBLASLt GEMM plan",
                            "the GEMM plan already has an active use");
  }
  *out_info = plan->algorithm_info;
  if (!release_exclusive_use(plan->active_uses)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                          "query cuBLASLt GEMM plan",
                          "plan use accounting was corrupted");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus rustinfer_cuda_gemm_plan_execute(
    RustInferCudaGemmPlan* plan, const RustInferCudaBufferSpan* input,
    const RustInferCudaBufferSpan* weight,
    const RustInferCudaBufferSpan* output,
    const RustInferCudaBufferSpan* workspace, RustInferCudaStream* stream,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "the GEMM plan is null");
  }
  if (!plan->algorithm_ready) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "the GEMM plan is not prepared");
  }

  ResolvedSpan spans[kMaximumGemmBuffers]{};
  RustInferCudaStatus status = resolve_exact_span(
      input, RUSTINFER_CUDA_DTYPE_BF16, plan->input_bytes, &spans[0], error,
      "input span dtype must be BF16");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(
        weight, RUSTINFER_CUDA_DTYPE_BF16, plan->weight_bytes, &spans[1],
        error, "weight span dtype must be BF16");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(
        output, RUSTINFER_CUDA_DTYPE_BF16, plan->output_bytes, &spans[2],
        error, "output span dtype must be BF16");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_exact_span(
        workspace, RUSTINFER_CUDA_DTYPE_U8,
        plan->algorithm_info.workspace_bytes, &spans[3], error,
        "workspace span dtype must be U8");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  status = validate_span_relationships(plan, stream, spans,
                                       kMaximumGemmBuffers, error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  ExclusiveGemmUses uses(plan, stream);
  for (const ResolvedSpan& span : spans) {
    if (!uses.add(span.buffer)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "execute cuBLASLt GEMM",
                            "too many unique GEMM device buffers");
    }
  }
  status = uses.acquire(error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  CurrentContext scope(plan->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                       "execute cuBLASLt GEMM");
  bool matmul_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    void* workspace_data = plan->algorithm_info.workspace_bytes == 0
                               ? nullptr
                               : spans[3].data;
    matmul_attempted = true;
    status = cublaslt_error(
        cublasLtMatmul(
            plan->handle, plan->operation, &alpha, spans[1].data,
            plan->weight_layout, spans[0].data, plan->input_layout, &beta,
            spans[2].data, plan->output_layout, spans[2].data,
            plan->output_layout, &plan->algorithm, workspace_data,
            static_cast<size_t>(plan->algorithm_info.workspace_bytes),
            stream->stream),
        error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
        "execute cuBLASLt GEMM");
  }
  return complete_execution(&uses, &scope, stream, status,
                            matmul_attempted, error);
}

extern "C" RustInferCudaStatus rustinfer_cuda_gemm_plan_close(
    RustInferCudaGemmPlan** plan,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "close cuBLASLt GEMM plan",
                            "plan pointer is null");
  }
  if (*plan == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  RustInferCudaGemmPlan* value = *plan;
  if (value->owner == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close cuBLASLt GEMM plan",
                            "plan context owner is null");
  }
  if (!try_acquire_exclusive_use(value->active_uses)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close cuBLASLt GEMM plan",
                            "the GEMM plan has an active or permanent use guard");
  }

  CurrentContext scope(value->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
      "close cuBLASLt GEMM plan");
  bool destruction_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    destruction_attempted = true;
    status = destroy_plan_resources(value, error,
                                    RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                                    "close cuBLASLt GEMM plan");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                       "close cuBLASLt GEMM plan");

  const bool restoration_confirmed =
      !value->owner->restoration_failed.load(std::memory_order_acquire);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      plan_resources_destroyed(value) && restoration_confirmed) {
    RustInferCudaContext* owner = value->owner;
    value->~RustInferCudaGemmPlan();
    std::free(value);
    *plan = nullptr;
    if (!release_child(owner)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close cuBLASLt GEMM plan",
                            "context child-resource counter underflow");
    }
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  if (!destruction_attempted && restoration_confirmed) {
    if (!release_exclusive_use(value->active_uses)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            "close cuBLASLt GEMM plan",
                            "plan use accounting was corrupted");
    }
  }
  // A destruction attempt or ambiguous context restoration leaves the plan's
  // exclusive-use guard set forever, preserving its context-child lease.
  return status;
}
