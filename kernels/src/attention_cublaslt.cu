#include "ffi_internal.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <math_constants.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

namespace {

constexpr uint64_t kBfloat16Bytes = 2;
constexpr uint64_t kRequiredAlignment = 256;
constexpr uint32_t kThreads = 256;
constexpr uint32_t kMaximumBlocks = 65535;
constexpr size_t kMaximumBuffers = 5;
constexpr int kHeuristicResults = 8;
constexpr uint32_t kCausalMaskBf16AsF32Bits = 0xff7f0000U;
constexpr uint64_t kReviewedQueryHeads = 9;
constexpr uint64_t kReviewedKeyValueHeads = 3;
constexpr uint64_t kReviewedHeadSize = 64;
constexpr uint64_t kReviewedMaximumSequence = 8192;

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

struct MatmulState {
  cublasLtMatmulDesc_t operation;
  cublasLtMatrixLayout_t a_layout;
  cublasLtMatrixLayout_t b_layout;
  cublasLtMatrixLayout_t c_layout;
  cublasLtMatmulPreference_t preference;
  cublasLtMatmulAlgo_t algorithm;
  bool algorithm_ready;
};

struct SelectedAlgorithmProvenance {
  int32_t algorithm_id;
  uint32_t tile_id;
  uint32_t stages_id;
  uint32_t split_k;
  uint32_t reduction_scheme;
  uint32_t cta_swizzling;
  uint32_t custom_option;
  uint64_t workspace_bytes;
  uint64_t numerical_implementation_flags;
};

#include "hf_eager_algorithm_allowlist.inc"

struct ByteCounts {
  uint64_t query;
  uint64_t key_value;
  uint64_t score;
  uint64_t repeated_key_value;
  uint64_t repeated_offset;
  uint64_t workspace;
};

struct ResolvedSpan {
  RustInferCudaDeviceBuffer* buffer;
  uint8_t* data;
  uint64_t byte_offset;
  uint64_t byte_len;
};

bool checked_add(uint64_t left, uint64_t right, uint64_t* output) noexcept {
  if (output == nullptr || right > std::numeric_limits<uint64_t>::max() - left) {
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

bool checked_product(const uint64_t* factors, size_t count,
                     uint64_t* output) noexcept {
  uint64_t result = 1;
  for (size_t index = 0; index < count; ++index) {
    if (!checked_multiply(result, factors[index], &result)) {
      return false;
    }
  }
  *output = result;
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
                   "cuBLASLt attention operation failed");
}

RustInferCudaStatus compute_byte_counts(
    const RustInferCudaHfPrefillAttentionConfig& config, ByteCounts* output,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  const uint64_t query_factors[] = {
      config.batch_count, config.token_count, config.query_head_count,
      config.head_size, kBfloat16Bytes};
  const uint64_t key_value_factors[] = {
      config.batch_count, config.token_count, config.key_value_head_count,
      config.head_size, kBfloat16Bytes};
  const uint64_t score_factors[] = {
      config.query_head_count, config.token_count, config.token_count,
      kBfloat16Bytes};
  const uint64_t repeated_factors[] = {
      config.query_head_count, config.token_count, config.head_size,
      kBfloat16Bytes};
  if (!checked_product(query_factors, 5, &output->query) ||
      !checked_product(key_value_factors, 5, &output->key_value) ||
      !checked_product(score_factors, 4, &output->score) ||
      !checked_product(repeated_factors, 4,
                       &output->repeated_key_value)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "HF attention byte-length arithmetic overflow");
  }
  uint64_t aligned = 0;
  if (!checked_add(output->score, kRequiredAlignment - 1, &aligned)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "HF attention workspace alignment overflow");
  }
  output->repeated_offset =
      (aligned / kRequiredAlignment) * kRequiredAlignment;
  if (!checked_add(output->repeated_offset, output->repeated_key_value,
                   &output->workspace)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "HF attention workspace byte length overflows");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus validate_config(
    const RustInferCudaHfPrefillAttentionConfig* config, ByteCounts* bytes,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "prepare HF cuBLASLt prefill attention";
  if (config == nullptr || bytes == nullptr ||
      config->struct_size < sizeof(*config)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "config is null or has an incompatible struct_size");
  }
  if (config->reserved0 != 0 || !reserved_is_zero(config->reserved, 4)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "config reserved fields must be zero");
  }
  if (config->batch_count != 1) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "the exact HF backend currently supports batch_count=1");
  }
  if (config->token_count == 0 || config->query_head_count == 0 ||
      config->key_value_head_count == 0 || config->head_size == 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "all attention dimensions must be non-zero");
  }
  if (config->query_head_count % config->key_value_head_count != 0) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "key_value_head_count must divide query_head_count");
  }
  if (config->query_head_count != kReviewedQueryHeads ||
      config->key_value_head_count != kReviewedKeyValueHeads ||
      config->head_size != kReviewedHeadSize ||
      config->token_count > kReviewedMaximumSequence) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "the reviewed HF backend requires QH=9, KVH=3, D=64, and S<=8192");
  }
  if (config->query_head_count >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      config->token_count >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      config->head_size >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "attention dimensions exceed cuBLASLt limits");
  }
  if (!std::isfinite(config->scale) || config->scale <= 0.0F) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "scale must be finite and greater than zero");
  }
  if (config->deterministic !=
      RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "deterministic execution must be required");
  }
  if (config->max_cublas_workspace_bytes >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "cuBLASLt workspace cap exceeds size_t");
  }
  return compute_byte_counts(*config, bytes, error, kOperation);
}

template <typename T>
bool algorithm_config_value(const cublasLtMatmulAlgo_t* algorithm,
                            cublasLtMatmulAlgoConfigAttributes_t attribute,
                            T* output) noexcept {
  size_t written = 0;
  return algorithm != nullptr && output != nullptr &&
         cublasLtMatmulAlgoConfigGetAttribute(
             algorithm, attribute, output, sizeof(*output), &written) ==
             CUBLAS_STATUS_SUCCESS &&
         written == sizeof(*output);
}

template <typename T>
bool algorithm_capability_value(
    const cublasLtMatmulAlgo_t* algorithm,
    cublasLtMatmulAlgoCapAttributes_t attribute, T* output) noexcept {
  size_t written = 0;
  return algorithm != nullptr && output != nullptr &&
         cublasLtMatmulAlgoCapGetAttribute(
             algorithm, attribute, output, sizeof(*output), &written) ==
             CUBLAS_STATUS_SUCCESS &&
         written == sizeof(*output);
}

bool algorithm_has_supported_alignment(
    const cublasLtMatmulAlgo_t* algorithm) noexcept {
  uint32_t alignments[4]{};
  if (!algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES,
          &alignments[0]) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
          &alignments[1]) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES,
          &alignments[2]) ||
      !algorithm_capability_value(
          algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES,
          &alignments[3])) {
    return false;
  }
  for (uint32_t alignment : alignments) {
    if (alignment != 0 && kRequiredAlignment % alignment != 0) {
      return false;
    }
  }
  return true;
}

RustInferCudaStatus set_layout_attribute(
    cublasLtMatrixLayout_t layout, cublasLtMatrixLayoutAttribute_t attribute,
    const void* value, size_t size, RustInferCudaErrorInfo* error) noexcept {
  return cublaslt_error(
      cublasLtMatrixLayoutSetAttribute(layout, attribute, value, size), error,
      RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare HF cuBLASLt prefill attention");
}

RustInferCudaStatus prepare_layout(
    cublasLtMatrixLayout_t* layout, uint64_t rows, uint64_t columns,
    int64_t leading_dimension, int32_t batch_count, int64_t batch_stride,
    RustInferCudaErrorInfo* error) noexcept {
  RustInferCudaStatus status = cublaslt_error(
      cublasLtMatrixLayoutCreate(layout, CUDA_R_16BF, rows, columns,
                                 leading_dimension),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare HF cuBLASLt prefill attention");
  const cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = set_layout_attribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                  &order, sizeof(order), error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = set_layout_attribute(*layout,
                                  CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                  &batch_count, sizeof(batch_count), error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = set_layout_attribute(*layout,
                                  CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                                  &batch_stride, sizeof(batch_stride), error);
  }
  return status;
}

RustInferCudaStatus select_first_exact_algorithm(
    cublasLtHandle_t handle, MatmulState* matmul,
    SelectedAlgorithmProvenance* provenance,
    RustInferCudaErrorInfo* error) noexcept {
  if (provenance == nullptr) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare HF cuBLASLt prefill attention",
                          "algorithm provenance output is null");
  }
  cublasLtMatmulHeuristicResult_t candidates[kHeuristicResults]{};
  int returned = 0;
  RustInferCudaStatus status = cublaslt_error(
      cublasLtMatmulAlgoGetHeuristic(
          handle, matmul->operation, matmul->a_layout, matmul->b_layout,
          matmul->c_layout, matmul->c_layout, matmul->preference,
          kHeuristicResults, candidates, &returned),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare HF cuBLASLt prefill attention");
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (returned <= 0 || candidates[0].state != CUBLAS_STATUS_SUCCESS) {
    return set_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                     "prepare HF cuBLASLt prefill attention",
                     "the first cuBLASLt heuristic is unavailable");
  }
  matmul->algorithm = candidates[0].algo;
  if (!algorithm_has_supported_alignment(&matmul->algorithm)) {
    return set_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                     "prepare HF cuBLASLt prefill attention",
                     "the first heuristic violates the 256-byte pointer alignment contract");
  }
  cublasLtMatmulHeuristicResult_t checked{};
  if (cublasLtMatmulAlgoCheck(
          handle, matmul->operation, matmul->a_layout, matmul->b_layout,
          matmul->c_layout, matmul->c_layout, &matmul->algorithm,
          &checked) != CUBLAS_STATUS_SUCCESS ||
      checked.state != CUBLAS_STATUS_SUCCESS) {
    return set_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                     "prepare HF cuBLASLt prefill attention",
                     "the first cuBLASLt heuristic failed algoCheck");
  }
  if (!algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_ID,
                              &provenance->algorithm_id) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_TILE_ID,
                              &provenance->tile_id) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_STAGES_ID,
                              &provenance->stages_id) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                              &provenance->split_k) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                              &provenance->reduction_scheme) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                              &provenance->cta_swizzling) ||
      !algorithm_config_value(&matmul->algorithm,
                              CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                              &provenance->custom_option) ||
      !algorithm_capability_value(
          &matmul->algorithm, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
          &provenance->numerical_implementation_flags)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare HF cuBLASLt prefill attention",
                          "cuBLASLt algorithm metadata query failed");
  }
  provenance->workspace_bytes =
      static_cast<uint64_t>(checked.workspaceSize);
  if (candidates[0].workspaceSize != 0 || checked.workspaceSize != 0 ||
      provenance->split_k > 1 ||
      provenance->reduction_scheme !=
          static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE)) {
    return set_error(error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT,
                     RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                     "prepare HF cuBLASLt prefill attention",
                     "the first heuristic violates the zero-workspace no-split contract");
  }
  matmul->algorithm_ready = true;
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

}  // namespace

struct RustInferCudaHfPrefillAttentionPlan {
  RustInferCudaHfPrefillAttentionPlan(
      RustInferCudaContext* context,
      const RustInferCudaHfPrefillAttentionConfig& plan_config,
      const ByteCounts& plan_bytes) noexcept
      : owner(context),
        config(plan_config),
        info{},
        handle(nullptr),
        qk{},
        av{},
        bytes(plan_bytes),
        active_uses(0) {
    info.struct_size = sizeof(info);
  }

  RustInferCudaContext* owner;
  RustInferCudaHfPrefillAttentionConfig config;
  RustInferCudaHfPrefillAttentionPlanInfo info;
  cublasLtHandle_t handle;
  MatmulState qk;
  MatmulState av;
  ByteCounts bytes;
  std::atomic<uint32_t> active_uses;
};

namespace {

const ReviewedAttentionAlgorithm* reviewed_algorithm_for_token_count(
    uint64_t token_count) noexcept {
  for (const ReviewedTokenClass& token_class : kReviewedTokenClasses) {
    if (token_count >= token_class.first &&
        token_count <= token_class.last &&
        (token_count - token_class.first) % 8 == 0) {
      const size_t index = token_class.algorithm_index;
      if (index >= sizeof(kReviewedAttentionAlgorithms) /
                       sizeof(kReviewedAttentionAlgorithms[0])) {
        return nullptr;
      }
      return &kReviewedAttentionAlgorithms[index];
    }
  }
  return nullptr;
}

bool reviewed_algorithms_match(
    const RustInferCudaHfPrefillAttentionPlanInfo& actual,
    const ReviewedAttentionAlgorithm& expected) noexcept {
  return actual.qk_algorithm_id == expected.qk_algorithm_id &&
         actual.qk_tile_id == expected.qk_tile_id &&
         actual.qk_stages_id == expected.qk_stages_id &&
         actual.qk_split_k == 1 &&
         actual.qk_reduction_scheme ==
             static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE) &&
         actual.qk_cta_swizzling == expected.qk_cta_swizzling &&
         actual.qk_custom_option == expected.qk_custom_option &&
         actual.qk_workspace_bytes == 0 &&
         actual.qk_numerical_implementation_flags ==
             expected.qk_numerical_implementation_flags &&
         actual.av_algorithm_id == expected.av_algorithm_id &&
         actual.av_tile_id == expected.av_tile_id &&
         actual.av_stages_id == expected.av_stages_id &&
         actual.av_split_k == 1 &&
         actual.av_reduction_scheme ==
             static_cast<uint32_t>(CUBLASLT_REDUCTION_SCHEME_NONE) &&
         actual.av_cta_swizzling == expected.av_cta_swizzling &&
         actual.av_custom_option == expected.av_custom_option &&
         actual.av_workspace_bytes == 0 &&
         actual.av_numerical_implementation_flags ==
             expected.av_numerical_implementation_flags;
}

RustInferCudaStatus validate_reviewed_plan_provenance(
    const RustInferCudaHfPrefillAttentionPlan* plan,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation =
      "prepare HF cuBLASLt prefill attention";
  if (plan->info.compute_capability_major != 8 ||
      plan->info.compute_capability_minor != 9 ||
      plan->info.runtime_version != kReviewedRuntimeVersion ||
      plan->info.cublaslt_version != kReviewedCublasLtVersion) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_PREPARE, kOperation,
        "CUDA environment is outside the reviewed cc89/runtime-12080/cuBLASLt-120804 contract");
  }
  const ReviewedAttentionAlgorithm* expected =
      reviewed_algorithm_for_token_count(plan->config.token_count);
  if (expected == nullptr ||
      !reviewed_algorithms_match(plan->info, *expected)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_PREPARE, kOperation,
        "the first QK/AV heuristics are outside the reviewed exact algorithm class");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus prepare_operation(
    RustInferCudaHfPrefillAttentionPlan* plan, MatmulState* matmul,
    bool qk, RustInferCudaErrorInfo* error) noexcept {
  RustInferCudaStatus status = cublaslt_error(
      cublasLtMatmulDescCreate(&matmul->operation, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F),
      error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare HF cuBLASLt prefill attention");
  const cublasOperation_t transpose_a = qk ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t transpose_b = CUBLAS_OP_N;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulDescSetAttribute(
            matmul->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a,
            sizeof(transpose_a)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare HF cuBLASLt prefill attention");
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulDescSetAttribute(
            matmul->operation, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b,
            sizeof(transpose_b)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare HF cuBLASLt prefill attention");
  }
  const uint64_t s = plan->config.token_count;
  const uint64_t qh = plan->config.query_head_count;
  const uint64_t d = plan->config.head_size;
  const int32_t batch = static_cast<int32_t>(qh);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_layout(&matmul->a_layout, d, s,
                            static_cast<int64_t>(d), batch,
                            static_cast<int64_t>(s * d), error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && qk) {
    status = prepare_layout(&matmul->b_layout, d, s,
                            static_cast<int64_t>(qh * d), batch,
                            static_cast<int64_t>(d), error);
  } else if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_layout(&matmul->b_layout, s, s,
                            static_cast<int64_t>(s), batch,
                            static_cast<int64_t>(s * s), error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && qk) {
    status = prepare_layout(&matmul->c_layout, s, s,
                            static_cast<int64_t>(s), batch,
                            static_cast<int64_t>(s * s), error);
  } else if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_layout(&matmul->c_layout, d, s,
                            static_cast<int64_t>(qh * d), batch,
                            static_cast<int64_t>(d), error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulPreferenceCreate(&matmul->preference), error,
        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare HF cuBLASLt prefill attention");
  }
  const size_t cap =
      static_cast<size_t>(plan->config.max_cublas_workspace_bytes);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmulPreferenceSetAttribute(
            matmul->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &cap, sizeof(cap)),
        error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare HF cuBLASLt prefill attention");
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  SelectedAlgorithmProvenance provenance{};
  status = select_first_exact_algorithm(plan->handle, matmul, &provenance,
                                        error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  if (qk) {
    plan->info.qk_algorithm_id = provenance.algorithm_id;
    plan->info.qk_tile_id = provenance.tile_id;
    plan->info.qk_stages_id = provenance.stages_id;
    plan->info.qk_split_k = provenance.split_k;
    plan->info.qk_reduction_scheme = provenance.reduction_scheme;
    plan->info.qk_cta_swizzling = provenance.cta_swizzling;
    plan->info.qk_custom_option = provenance.custom_option;
    plan->info.qk_workspace_bytes = provenance.workspace_bytes;
    plan->info.qk_numerical_implementation_flags =
        provenance.numerical_implementation_flags;
  } else {
    plan->info.av_algorithm_id = provenance.algorithm_id;
    plan->info.av_tile_id = provenance.tile_id;
    plan->info.av_stages_id = provenance.stages_id;
    plan->info.av_split_k = provenance.split_k;
    plan->info.av_reduction_scheme = provenance.reduction_scheme;
    plan->info.av_cta_swizzling = provenance.cta_swizzling;
    plan->info.av_custom_option = provenance.custom_option;
    plan->info.av_workspace_bytes = provenance.workspace_bytes;
    plan->info.av_numerical_implementation_flags =
        provenance.numerical_implementation_flags;
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus prepare_plan(
    RustInferCudaHfPrefillAttentionPlan* plan,
    RustInferCudaErrorInfo* error) noexcept {
  RustInferCudaStatus status = cublaslt_error(
      cublasLtCreate(&plan->handle), error,
      RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
      "prepare HF cuBLASLt prefill attention");
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_operation(plan, &plan->qk, true, error);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = prepare_operation(plan, &plan->av, false, error);
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }

  int capability_major = 0;
  CUresult driver_result = cuDeviceGetAttribute(
      &capability_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error,
                        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                        "prepare HF cuBLASLt prefill attention");
  }
  int capability_minor = 0;
  driver_result = cuDeviceGetAttribute(
      &capability_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      plan->owner->device);
  if (driver_result != CUDA_SUCCESS) {
    return driver_error(driver_result, error,
                        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                        "prepare HF cuBLASLt prefill attention");
  }
  int runtime_version = 0;
  const cudaError_t runtime_result = cudaRuntimeGetVersion(&runtime_version);
  if (runtime_result != cudaSuccess) {
    return runtime_error(runtime_result, error,
                         RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                         "prepare HF cuBLASLt prefill attention");
  }
  const size_t cublaslt_version = cublasLtGetVersion();
  if (capability_major < 0 || capability_minor < 0 ||
      cublaslt_version >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                          "prepare HF cuBLASLt prefill attention",
                          "invalid environment metadata");
  }
  if (capability_major != 8 || capability_minor != 9) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_NOT_SUPPORTED,
        RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
        "prepare HF cuBLASLt prefill attention",
        "the reviewed HF backend requires compute capability 8.9");
  }
  plan->info.backend = RUSTINFER_CUDA_ATTENTION_BACKEND_HF_CUBLASLT;
  plan->info.deterministic = RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED;
  plan->info.compute_capability_major =
      static_cast<uint32_t>(capability_major);
  plan->info.compute_capability_minor =
      static_cast<uint32_t>(capability_minor);
  plan->info.runtime_version = runtime_version;
  plan->info.cublaslt_version = static_cast<int32_t>(cublaslt_version);
  plan->info.workspace_bytes = plan->bytes.workspace;
  plan->info.score_bytes = plan->bytes.score;
  plan->info.repeated_key_value_bytes = plan->bytes.repeated_key_value;
  plan->info.layout_copy_bytes = 2 * plan->bytes.repeated_key_value;
  plan->info.batch_count = plan->config.batch_count;
  plan->info.token_count = plan->config.token_count;
  plan->info.query_head_count = plan->config.query_head_count;
  plan->info.key_value_head_count = plan->config.key_value_head_count;
  plan->info.head_size = plan->config.head_size;
  return validate_reviewed_plan_provenance(plan, error);
}

RustInferCudaStatus destroy_matmul(
    MatmulState* matmul, RustInferCudaErrorInfo* error, uint32_t stage,
    const char* operation) noexcept {
  if (matmul->preference != nullptr) {
    const cublasStatus_t result =
        cublasLtMatmulPreferenceDestroy(matmul->preference);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    matmul->preference = nullptr;
  }
  if (matmul->c_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(matmul->c_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    matmul->c_layout = nullptr;
  }
  if (matmul->b_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(matmul->b_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    matmul->b_layout = nullptr;
  }
  if (matmul->a_layout != nullptr) {
    const cublasStatus_t result =
        cublasLtMatrixLayoutDestroy(matmul->a_layout);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    matmul->a_layout = nullptr;
  }
  if (matmul->operation != nullptr) {
    const cublasStatus_t result =
        cublasLtMatmulDescDestroy(matmul->operation);
    if (result != CUBLAS_STATUS_SUCCESS) {
      return cublaslt_error(result, error, stage, operation);
    }
    matmul->operation = nullptr;
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

RustInferCudaStatus destroy_plan_resources(
    RustInferCudaHfPrefillAttentionPlan* plan,
    RustInferCudaErrorInfo* error, uint32_t stage,
    const char* operation) noexcept {
  RustInferCudaStatus status =
      destroy_matmul(&plan->av, error, stage, operation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = destroy_matmul(&plan->qk, error, stage, operation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && plan->handle != nullptr) {
    status = cublaslt_error(cublasLtDestroy(plan->handle), error, stage,
                            operation);
    if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
      plan->handle = nullptr;
    }
  }
  return status;
}

bool matmul_destroyed(const MatmulState& matmul) noexcept {
  return matmul.operation == nullptr && matmul.a_layout == nullptr &&
         matmul.b_layout == nullptr && matmul.c_layout == nullptr &&
         matmul.preference == nullptr;
}

bool plan_resources_destroyed(
    const RustInferCudaHfPrefillAttentionPlan* plan) noexcept {
  return plan->handle == nullptr && matmul_destroyed(plan->qk) &&
         matmul_destroyed(plan->av);
}

RustInferCudaStatus resolve_span(
    const RustInferCudaBufferSpan* span, uint64_t required_bytes,
    ResolvedSpan* output, RustInferCudaErrorInfo* error,
    const char* operation) noexcept {
  if (span == nullptr || output == nullptr ||
      span->struct_size < sizeof(*span)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "a span is null or has an incompatible struct_size");
  }
  if (!reserved_is_zero(span->reserved, 2) ||
      span->dtype != RUSTINFER_CUDA_DTYPE_BF16 || span->buffer == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "all HF attention spans must be valid BF16 spans");
  }
  if (span->byte_offset % kRequiredAlignment != 0 ||
      span->byte_len != required_bytes) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "span offset or exact byte length violates the prepared contract");
  }
  if (span->byte_offset > span->buffer->byte_len ||
      span->byte_len > span->buffer->byte_len - span->byte_offset ||
      span->buffer->device_data == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_RANGE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, operation,
                            "span exceeds its device allocation");
  }
  *output = ResolvedSpan{
      span->buffer,
      static_cast<uint8_t*>(span->buffer->device_data) +
          static_cast<size_t>(span->byte_offset),
      span->byte_offset, span->byte_len};
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

bool spans_overlap(const ResolvedSpan& left,
                   const ResolvedSpan& right) noexcept {
  if (left.buffer != right.buffer || left.byte_len == 0 ||
      right.byte_len == 0) {
    return false;
  }
  return left.byte_offset < right.byte_offset + right.byte_len &&
         right.byte_offset < left.byte_offset + left.byte_len;
}

class ExclusiveUses final {
 public:
  ExclusiveUses(RustInferCudaHfPrefillAttentionPlan* plan,
                RustInferCudaStream* stream) noexcept
      : plan_(plan), stream_(stream), buffers_{}, buffer_count_(0),
        acquired_buffers_(0), plan_acquired_(false),
        stream_acquired_(false), command_batch_(false) {}

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
    if (command_batch_is_active(stream_)) {
      if (!command_batch_is_owned_by_current_thread(stream_)) {
        return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                operation,
                                "active command batch belongs to another thread");
      }
      command_batch_ = true;
      RustInferCudaStatus status = command_batch_register_use(
          stream_, &plan_->active_uses, error, operation,
          "HF attention plan already has an active use");
      for (size_t index = 0;
           status == RUSTINFER_CUDA_STATUS_SUCCESS && index < buffer_count_;
           ++index) {
        status = command_batch_register_use(
            stream_, &buffers_[index]->active_uses, error, operation,
            "HF attention buffer already has an active use");
      }
      return status;
    }
    if (!try_acquire_exclusive_use(plan_->active_uses)) {
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              operation,
                              "HF attention plan already has an active use");
    }
    plan_acquired_ = true;
    for (size_t index = 0; index < buffer_count_; ++index) {
      if (!try_acquire_exclusive_use(buffers_[index]->active_uses)) {
        if (!release_acquired()) {
          return internal_error(error,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                operation,
                                "exclusive-use rollback was corrupted");
        }
        return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                operation,
                                "HF attention buffer already has an active use");
      }
      ++acquired_buffers_;
    }
    if (!try_acquire_exclusive_use(stream_->active_uses)) {
      if (!release_acquired()) {
        return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              operation,
                              "exclusive-use rollback was corrupted");
      }
      return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                              RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                              operation,
                              "HF attention stream already has an active use");
    }
    stream_acquired_ = true;
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }

  bool command_batch() const noexcept { return command_batch_; }
  bool release_completed() noexcept { return release_acquired(); }

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
      valid = release_exclusive_use(
                  buffers_[acquired_buffers_]->active_uses) && valid;
    }
    if (plan_acquired_) {
      valid = release_exclusive_use(plan_->active_uses) && valid;
      plan_acquired_ = false;
    }
    return valid;
  }

  RustInferCudaHfPrefillAttentionPlan* plan_;
  RustInferCudaStream* stream_;
  RustInferCudaDeviceBuffer* buffers_[kMaximumBuffers];
  size_t buffer_count_;
  size_t acquired_buffers_;
  bool plan_acquired_;
  bool stream_acquired_;
  bool command_batch_;
};

uint32_t block_count(uint64_t work_items) noexcept {
  const uint64_t needed = ((work_items - 1) / kThreads) + 1;
  return static_cast<uint32_t>(
      needed < kMaximumBlocks ? needed : kMaximumBlocks);
}

__global__ void repeat_kv_kernel(
    const __nv_bfloat16* source, __nv_bfloat16* destination,
    uint64_t token_count, uint64_t query_head_count,
    uint64_t key_value_head_count, uint64_t head_size,
    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t group_size = query_head_count / key_value_head_count;
  for (uint64_t index = first; index < element_count; index += stride) {
    const uint64_t depth = index % head_size;
    const uint64_t row = index / head_size;
    const uint64_t token = row % token_count;
    const uint64_t query_head = row / token_count;
    const uint64_t key_value_head = query_head / group_size;
    destination[index] =
        source[(token * key_value_head_count + key_value_head) * head_size +
               depth];
  }
}

__global__ void scale_causal_mask_kernel(
    __nv_bfloat16* scores, uint64_t token_count, float scale,
    uint64_t element_count) {
  const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t index = first; index < element_count; index += stride) {
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

template <int Log2Elements>
__global__ void hf_persistent_softmax_kernel(__nv_bfloat16* scores,
                                             uint64_t row_count,
                                             uint64_t token_count) {
  constexpr int kNextPowerOfTwo = 1 << Log2Elements;
  constexpr int kWarpSize = kNextPowerOfTwo < 32 ? kNextPowerOfTwo : 32;
  constexpr int kWarpIterations = kNextPowerOfTwo / kWarpSize;
  constexpr int kWarpBatch = kNextPowerOfTwo <= 128 ? 2 : 1;
  const uint64_t first_row =
      (static_cast<uint64_t>(blockDim.y) * blockIdx.x + threadIdx.y) *
      kWarpBatch;
  const int lane = threadIdx.x;
  float elements[kWarpBatch][kWarpIterations];

#pragma unroll
  for (int batch = 0; batch < kWarpBatch; ++batch) {
    const uint64_t row = first_row + batch;
#pragma unroll
    for (int iteration = 0; iteration < kWarpIterations; ++iteration) {
      const uint64_t column = lane + iteration * kWarpSize;
      elements[batch][iteration] =
          row < row_count && column < token_count
              ? __bfloat162float(scores[row * token_count + column])
              : -CUDART_INF_F;
    }
  }

  float maximum[kWarpBatch];
#pragma unroll
  for (int batch = 0; batch < kWarpBatch; ++batch) {
    maximum[batch] = elements[batch][0];
#pragma unroll
    for (int iteration = 0; iteration < kWarpIterations; ++iteration) {
      maximum[batch] = maximum[batch] > elements[batch][iteration]
                           ? maximum[batch]
                           : elements[batch][iteration];
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int batch = 0; batch < kWarpBatch; ++batch) {
      const float other =
          __shfl_xor_sync(0xffffffffU, maximum[batch], offset, kWarpSize);
      maximum[batch] =
          maximum[batch] < other ? other : maximum[batch];
    }
  }

  float sum[kWarpBatch]{};
#pragma unroll
  for (int batch = 0; batch < kWarpBatch; ++batch) {
#pragma unroll
    for (int iteration = 0; iteration < kWarpIterations; ++iteration) {
      elements[batch][iteration] =
          expf(elements[batch][iteration] - maximum[batch]);
      sum[batch] += elements[batch][iteration];
    }
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
#pragma unroll
    for (int batch = 0; batch < kWarpBatch; ++batch) {
      sum[batch] +=
          __shfl_xor_sync(0xffffffffU, sum[batch], offset, kWarpSize);
    }
  }

#pragma unroll
  for (int batch = 0; batch < kWarpBatch; ++batch) {
    const uint64_t row = first_row + batch;
    if (row >= row_count) {
      continue;
    }
#pragma unroll
    for (int iteration = 0; iteration < kWarpIterations; ++iteration) {
      const uint64_t column = lane + iteration * kWarpSize;
      if (column < token_count) {
        const float probability = sum[batch] == 0.0F
                                      ? CUDART_NAN_F
                                      : elements[batch][iteration] / sum[batch];
        scores[row * token_count + column] =
            __float2bfloat16_rn(probability);
      }
    }
  }
}

template <int Log2Elements>
void launch_hf_persistent_softmax_template(__nv_bfloat16* scores,
                                           uint64_t row_count,
                                           uint64_t token_count,
                                           cudaStream_t stream) {
  constexpr uint32_t kNextPowerOfTwo = 1U << Log2Elements;
  constexpr uint32_t kWarpSize = kNextPowerOfTwo < 32 ? kNextPowerOfTwo : 32;
  constexpr uint32_t kWarpBatch = kNextPowerOfTwo <= 128 ? 2 : 1;
  constexpr uint32_t kThreadsPerBlock = 128;
  constexpr uint32_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;
  constexpr uint32_t kRowsPerBlock = kWarpsPerBlock * kWarpBatch;
  const uint32_t blocks =
      static_cast<uint32_t>((row_count + kRowsPerBlock - 1) / kRowsPerBlock);
  const dim3 threads(kWarpSize, kWarpsPerBlock, 1);
  hf_persistent_softmax_kernel<Log2Elements>
      <<<blocks, threads, 0, stream>>>(scores, row_count, token_count);
}

bool launch_hf_persistent_softmax(__nv_bfloat16* scores,
                                  uint64_t row_count,
                                  uint64_t token_count,
                                  cudaStream_t stream) {
  uint32_t log2_elements = 0;
  while ((uint64_t{1} << log2_elements) < token_count) {
    ++log2_elements;
  }
  switch (log2_elements) {
    case 0:
      launch_hf_persistent_softmax_template<0>(scores, row_count, token_count,
                                                stream);
      return true;
    case 1:
      launch_hf_persistent_softmax_template<1>(scores, row_count, token_count,
                                                stream);
      return true;
    case 2:
      launch_hf_persistent_softmax_template<2>(scores, row_count, token_count,
                                                stream);
      return true;
    case 3:
      launch_hf_persistent_softmax_template<3>(scores, row_count, token_count,
                                                stream);
      return true;
    case 4:
      launch_hf_persistent_softmax_template<4>(scores, row_count, token_count,
                                                stream);
      return true;
    case 5:
      launch_hf_persistent_softmax_template<5>(scores, row_count, token_count,
                                                stream);
      return true;
    case 6:
      launch_hf_persistent_softmax_template<6>(scores, row_count, token_count,
                                                stream);
      return true;
    case 7:
      launch_hf_persistent_softmax_template<7>(scores, row_count, token_count,
                                                stream);
      return true;
    case 8:
      launch_hf_persistent_softmax_template<8>(scores, row_count, token_count,
                                                stream);
      return true;
    case 9:
      launch_hf_persistent_softmax_template<9>(scores, row_count, token_count,
                                                stream);
      return true;
    case 10:
      launch_hf_persistent_softmax_template<10>(scores, row_count,
                                                 token_count, stream);
      return true;
    case 11:
      launch_hf_persistent_softmax_template<11>(scores, row_count,
                                                 token_count, stream);
      return true;
    default:
      return false;
  }
}

template <bool Maximum>
__device__ __forceinline__ float hf_block_reduce(float value,
                                                 float* shared) {
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t warp = threadIdx.x / 32;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    const float other = __shfl_down_sync(0xffffffffU, value, offset);
    if constexpr (Maximum) {
      value = value < other ? other : value;
    } else {
      value += other;
    }
  }
  __syncthreads();
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32
              ? shared[lane]
              : (Maximum ? -FLT_MAX : 0.0F);
  if (warp == 0) {
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffU, value, offset);
      if constexpr (Maximum) {
        value = value < other ? other : value;
      } else {
        value += other;
      }
    }
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

template <int RegisterCount>
__global__ void hf_regular_softmax_kernel(__nv_bfloat16* scores,
                                          uint64_t token_count) {
  __shared__ float reduction[32];
  __nv_bfloat16 values[RegisterCount];
  const uint64_t row_offset = static_cast<uint64_t>(blockIdx.x) * token_count;
  float thread_maximum = -FLT_MAX;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      values[register_index] = scores[row_offset + column];
      const float value = __bfloat162float(values[register_index]);
      thread_maximum = thread_maximum < value ? value : thread_maximum;
    }
  }
  const float maximum = hf_block_reduce<true>(thread_maximum, reduction);
  float thread_sum = 0.0F;
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      thread_sum +=
          expf(__bfloat162float(values[register_index]) - maximum);
    }
  }
  const float sum = hf_block_reduce<false>(thread_sum, reduction);
#pragma unroll
  for (int register_index = 0; register_index < RegisterCount;
       ++register_index) {
    const uint64_t column =
        threadIdx.x + static_cast<uint64_t>(register_index) * blockDim.x;
    if (column < token_count) {
      scores[row_offset + column] = __float2bfloat16_rn(
          expf(__bfloat162float(values[register_index]) - maximum) / sum);
    }
  }
}

bool launch_hf_regular_softmax(__nv_bfloat16* scores, uint64_t row_count,
                               uint64_t token_count, cudaStream_t stream) {
  const uint64_t register_count = (token_count + 1023) / 1024;
  const dim3 grid(static_cast<uint32_t>(row_count));
  constexpr uint32_t kBlockThreads = 1024;
  switch (register_count) {
    case 3:
      hf_regular_softmax_kernel<3>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    case 4:
      hf_regular_softmax_kernel<4>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    case 5:
      hf_regular_softmax_kernel<5>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    case 6:
      hf_regular_softmax_kernel<6>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    case 7:
      hf_regular_softmax_kernel<7>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    case 8:
      hf_regular_softmax_kernel<8>
          <<<grid, kBlockThreads, 0, stream>>>(scores, token_count);
      return true;
    default:
      return false;
  }
}

RustInferCudaStatus launch_status(RustInferCudaErrorInfo* error,
                                  const char* operation) noexcept {
  return runtime_error(cudaGetLastError(), error,
                       RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, operation);
}

RustInferCudaStatus complete_execution(
    ExclusiveUses* uses, CurrentContext* scope, RustInferCudaStream* stream,
    RustInferCudaStatus operation_status, bool launch_attempted,
    RustInferCudaErrorInfo* error, const char* operation) noexcept {
  if (uses->command_batch()) {
    return scope->leave(operation_status, error,
                        RUSTINFER_CUDA_ERROR_STAGE_SYNCHRONIZE, operation);
  }
  bool completion_confirmed = !launch_attempted;
  RustInferCudaStatus status = operation_status;
  if (launch_attempted) {
    const cudaError_t result = cudaStreamSynchronize(stream->stream);
    completion_confirmed = result == cudaSuccess;
    if (!completion_confirmed) {
      status = runtime_error(result, error,
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

}  // namespace

extern "C" RustInferCudaStatus
rustinfer_cuda_hf_prefill_attention_plan_create(
    RustInferCudaContext* context,
    const RustInferCudaHfPrefillAttentionConfig* config,
    RustInferCudaHfPrefillAttentionPlan** out_plan,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "prepare HF cuBLASLt prefill attention";
  clear_error(error);
  if (out_plan == nullptr || context == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "context or out_plan is null");
  }
  *out_plan = nullptr;
  if (context->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(
        error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
        RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
        "a prior CUDA context-stack restoration failed");
  }
  ByteCounts bytes{};
  RustInferCudaStatus status = validate_config(config, &bytes, error);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  RustInferCudaHfPrefillAttentionConfig normalized = *config;
  normalized.struct_size = sizeof(normalized);
  void* storage = std::calloc(1, sizeof(RustInferCudaHfPrefillAttentionPlan));
  if (storage == nullptr) {
    return set_error(error, RUSTINFER_CUDA_STATUS_OUT_OF_MEMORY, 0,
                     RUSTINFER_CUDA_ERROR_DOMAIN_INTERNAL,
                     RUSTINFER_CUDA_ERROR_STAGE_CREATE, kOperation,
                     "host plan allocation failed");
  }
  auto* plan = new (storage)
      RustInferCudaHfPrefillAttentionPlan(context, normalized, bytes);
  if (!retain_child(context)) {
    plan->~RustInferCudaHfPrefillAttentionPlan();
    std::free(plan);
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CREATE,
                          kOperation,
                          "context child-resource counter overflow");
  }
  CurrentContext scope(context);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE, kOperation);
  bool prepare_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    prepare_attempted = true;
    status = prepare_plan(plan, error);
  }
  const bool entry_rejected_without_context_change =
      !prepare_attempted &&
      status == RUSTINFER_CUDA_STATUS_INVALID_STATE && !scope.active();
  if (prepare_attempted && status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    (void)destroy_plan_resources(plan, nullptr,
                                 RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                                 "cleanup failed HF attention plan");
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                       kOperation);
  if (entry_rejected_without_context_change) {
    const bool released = release_child(context);
    plan->~RustInferCudaHfPrefillAttentionPlan();
    std::free(plan);
    if (!released) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_PREPARE,
                            kOperation,
                            "context child-resource counter underflow");
    }
    return status;
  }
  const bool restored =
      !context->restoration_failed.load(std::memory_order_acquire);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS && restored) {
    *out_plan = plan;
    return status;
  }
  if (plan_resources_destroyed(plan) && restored) {
    (void)release_child(context);
    plan->~RustInferCudaHfPrefillAttentionPlan();
    std::free(plan);
  }
  return status;
}

extern "C" RustInferCudaStatus
rustinfer_cuda_hf_prefill_attention_plan_info(
    RustInferCudaHfPrefillAttentionPlan* plan,
    RustInferCudaHfPrefillAttentionPlanInfo* out_info,
    RustInferCudaErrorInfo* error) noexcept {
  clear_error(error);
  if (plan == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            "query HF attention plan",
                            "plan or out_info is incompatible");
  }
  if (plan->owner == nullptr ||
      plan->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                            "query HF attention plan",
                            "the retained CUDA context cannot be restored");
  }
  if (!try_acquire_exclusive_use(plan->active_uses)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                            "query HF attention plan",
                            "plan already has an active use");
  }
  std::memset(out_info, 0, sizeof(*out_info));
  *out_info = plan->info;
  if (!release_exclusive_use(plan->active_uses)) {
    return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_QUERY,
                          "query HF attention plan",
                          "plan use accounting was corrupted");
  }
  return RUSTINFER_CUDA_STATUS_SUCCESS;
}

extern "C" RustInferCudaStatus
rustinfer_cuda_hf_prefill_attention_plan_execute(
    RustInferCudaHfPrefillAttentionPlan* plan,
    const RustInferCudaBufferSpan* query_span,
    const RustInferCudaBufferSpan* key_span,
    const RustInferCudaBufferSpan* value_span,
    const RustInferCudaBufferSpan* output_span,
    const RustInferCudaBufferSpan* workspace_span,
    RustInferCudaStream* stream, RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "execute HF cuBLASLt prefill attention";
  clear_error(error);
  if (plan == nullptr || stream == nullptr || !plan->qk.algorithm_ready ||
      !plan->av.algorithm_ready || plan->owner == nullptr ||
      !same_context(plan->owner, stream->owner)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "plan or stream is invalid or belongs to another context");
  }
  if (plan->owner->restoration_failed.load(std::memory_order_acquire)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "the retained CUDA context cannot be restored");
  }
  ResolvedSpan spans[kMaximumBuffers]{};
  RustInferCudaStatus status = resolve_span(
      query_span, plan->bytes.query, &spans[0], error, kOperation);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(key_span, plan->bytes.key_value, &spans[1], error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(value_span, plan->bytes.key_value, &spans[2], error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(output_span, plan->bytes.query, &spans[3], error,
                          kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = resolve_span(workspace_span, plan->bytes.workspace, &spans[4],
                          error, kOperation);
  }
  for (size_t index = 0;
       status == RUSTINFER_CUDA_STATUS_SUCCESS && index < kMaximumBuffers;
       ++index) {
    if (!same_context(plan->owner, spans[index].buffer->owner)) {
      status = validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                                RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                kOperation,
                                "span belongs to another CUDA context");
    }
    for (size_t other = index + 1;
         status == RUSTINFER_CUDA_STATUS_SUCCESS && other < kMaximumBuffers;
         ++other) {
      if (spans_overlap(spans[index], spans[other])) {
        status = validation_error(error,
                                  RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                                  RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                                  kOperation,
                                  "HF attention spans must not overlap");
      }
    }
  }
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  ExclusiveUses uses(plan, stream);
  for (const ResolvedSpan& span : spans) {
    if (!uses.add(span.buffer)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_VALIDATION,
                            kOperation, "attention buffer set overflow");
    }
  }
  status = uses.acquire(error, kOperation);
  if (status != RUSTINFER_CUDA_STATUS_SUCCESS) {
    return status;
  }
  CurrentContext scope(plan->owner);
  status = scope.enter(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  bool launch_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = launch_status(error, kOperation);
  }
  auto* scores = reinterpret_cast<__nv_bfloat16*>(spans[4].data);
  auto* repeated = reinterpret_cast<__nv_bfloat16*>(
      spans[4].data + static_cast<size_t>(plan->bytes.repeated_offset));
  const uint64_t repeated_elements = plan->bytes.repeated_key_value / 2;
  const uint64_t score_elements = plan->bytes.score / 2;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    launch_attempted = true;
    repeat_kv_kernel<<<block_count(repeated_elements), kThreads, 0,
                       stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(spans[1].data), repeated,
        plan->config.token_count, plan->config.query_head_count,
        plan->config.key_value_head_count, plan->config.head_size,
        repeated_elements);
    status = launch_status(error, kOperation);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmul(
            plan->handle, plan->qk.operation, &alpha, repeated,
            plan->qk.a_layout, spans[0].data, plan->qk.b_layout, &beta,
            scores, plan->qk.c_layout, scores, plan->qk.c_layout,
            &plan->qk.algorithm, nullptr, 0, stream->stream),
        error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    scale_causal_mask_kernel
        <<<block_count(score_elements), kThreads, 0, stream->stream>>>(
            scores, plan->config.token_count, plan->config.scale,
            score_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    const uint64_t rows =
        plan->config.query_head_count * plan->config.token_count;
    const bool softmax_launched =
        launch_hf_persistent_softmax(scores, rows,
                                     plan->config.token_count,
                                     stream->stream) ||
        launch_hf_regular_softmax(scores, rows, plan->config.token_count,
                                  stream->stream);
    if (!softmax_launched) {
      status = internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH,
                              kOperation,
                              "prepared HF softmax dispatch is unavailable");
    } else {
      status = launch_status(error, kOperation);
    }
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    repeat_kv_kernel<<<block_count(repeated_elements), kThreads, 0,
                       stream->stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(spans[2].data), repeated,
        plan->config.token_count, plan->config.query_head_count,
        plan->config.key_value_head_count, plan->config.head_size,
        repeated_elements);
    status = launch_status(error, kOperation);
  }
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    status = cublaslt_error(
        cublasLtMatmul(
            plan->handle, plan->av.operation, &alpha, repeated,
            plan->av.a_layout, scores, plan->av.b_layout, &beta,
            spans[3].data, plan->av.c_layout, spans[3].data,
            plan->av.c_layout, &plan->av.algorithm, nullptr, 0,
            stream->stream),
        error, RUSTINFER_CUDA_ERROR_STAGE_LAUNCH, kOperation);
  }
  return complete_execution(&uses, &scope, stream, status, launch_attempted,
                            error, kOperation);
}

extern "C" RustInferCudaStatus
rustinfer_cuda_hf_prefill_attention_plan_close(
    RustInferCudaHfPrefillAttentionPlan** plan,
    RustInferCudaErrorInfo* error) noexcept {
  constexpr const char* kOperation = "close HF cuBLASLt prefill attention";
  clear_error(error);
  if (plan == nullptr) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_ARGUMENT,
                            RUSTINFER_CUDA_ERROR_STAGE_VALIDATION, kOperation,
                            "plan pointer is null");
  }
  if (*plan == nullptr) {
    return RUSTINFER_CUDA_STATUS_SUCCESS;
  }
  RustInferCudaHfPrefillAttentionPlan* value = *plan;
  if (!try_acquire_exclusive_use(value->active_uses)) {
    return validation_error(error, RUSTINFER_CUDA_STATUS_INVALID_STATE,
                            RUSTINFER_CUDA_ERROR_STAGE_CLOSE, kOperation,
                            "plan has an active or permanent use guard");
  }
  CurrentContext scope(value->owner);
  RustInferCudaStatus status = scope.enter(
      error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE, kOperation);
  bool destruction_attempted = false;
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS) {
    destruction_attempted = true;
    status = destroy_plan_resources(value, error,
                                    RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                                    kOperation);
  }
  status = scope.leave(status, error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                       kOperation);
  const bool restored =
      !value->owner->restoration_failed.load(std::memory_order_acquire);
  if (status == RUSTINFER_CUDA_STATUS_SUCCESS &&
      plan_resources_destroyed(value) && restored) {
    RustInferCudaContext* owner = value->owner;
    value->~RustInferCudaHfPrefillAttentionPlan();
    std::free(value);
    *plan = nullptr;
    if (!release_child(owner)) {
      return internal_error(error, RUSTINFER_CUDA_ERROR_STAGE_CLOSE,
                            kOperation,
                            "context child-resource counter underflow");
    }
    return status;
  }
  if (!destruction_attempted && restored) {
    (void)release_exclusive_use(value->active_uses);
  }
  return status;
}
