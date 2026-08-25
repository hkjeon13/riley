#include "rustinfer_cuda.h"

#include <stddef.h>

_Static_assert(RUSTINFER_CUDA_ABI_VERSION == 1,
               "PR 06 additions must preserve ABI v1");
_Static_assert(sizeof(void*) * 8 == RUSTINFER_CUDA_ABI_POINTER_WIDTH,
               "rustinfer CUDA ABI requires 64-bit pointers");
_Static_assert(sizeof(RustInferCudaDType) == 4,
               "dtype discriminant width changed");
_Static_assert(RUSTINFER_CUDA_DTYPE_F32 == 1 &&
                   RUSTINFER_CUDA_DTYPE_BF16 == 2 &&
                   RUSTINFER_CUDA_DTYPE_U32 == 3 &&
                   RUSTINFER_CUDA_DTYPE_U8 == 4,
               "dtype discriminants changed");
_Static_assert(sizeof(RustInferCudaErrorInfo) == 272,
               "error-info ABI size changed");
_Static_assert(sizeof(RustInferCudaBufferSpan) == 48,
               "buffer-span ABI size changed");
_Static_assert(offsetof(RustInferCudaBufferSpan, buffer) == 8,
               "buffer-span handle offset changed");
_Static_assert(offsetof(RustInferCudaBufferSpan, byte_offset) == 16,
               "buffer-span offset field changed");
_Static_assert(offsetof(RustInferCudaBufferSpan, reserved) == 32,
               "buffer-span reserved tail changed");
_Static_assert(sizeof(RustInferCudaEmbeddingErrorReport) == 32,
               "embedding-report ABI size changed");
_Static_assert(offsetof(RustInferCudaEmbeddingErrorReport, token_position) == 8,
               "embedding-report token position changed");
_Static_assert(sizeof(RustInferCudaEmbeddingParams) == 256,
               "embedding-params ABI size changed");
_Static_assert(offsetof(RustInferCudaEmbeddingParams, out_report) == 200,
               "embedding-params output report offset changed");
_Static_assert(offsetof(RustInferCudaEmbeddingParams, token_count) == 208,
               "embedding-params dimension offset changed");
_Static_assert(sizeof(RustInferCudaRmsNormParams) == 208,
               "RMSNorm-params ABI size changed");
_Static_assert(offsetof(RustInferCudaRmsNormParams, epsilon) == 168,
               "RMSNorm epsilon offset changed");
_Static_assert(sizeof(RustInferCudaResidualAddParams) == 200,
               "residual-add params ABI size changed");
_Static_assert(sizeof(RustInferCudaSiluParams) == 152,
               "SiLU params ABI size changed");
_Static_assert(sizeof(RustInferCudaGatedMultiplyParams) == 200,
               "gated-multiply params ABI size changed");
_Static_assert(sizeof(RustInferCudaRopeParams) == 288,
               "RoPE params ABI size changed");
_Static_assert(offsetof(RustInferCudaRopeParams, token_count) == 200,
               "RoPE dimension offset changed");
_Static_assert(offsetof(RustInferCudaRopeParams, reserved) == 248,
               "RoPE reserved tail changed");
_Static_assert(sizeof(RustInferCudaCastParams) == 152,
               "cast params ABI size changed");
_Static_assert(RUSTINFER_CUDA_STATUS_CUBLASLT_ERROR == 10 &&
                   RUSTINFER_CUDA_STATUS_NOT_SUPPORTED == 11,
               "GEMM status discriminants changed");
_Static_assert(RUSTINFER_CUDA_ERROR_DOMAIN_CUBLASLT == 5,
               "cuBLASLt error domain changed");
_Static_assert(RUSTINFER_CUDA_GEMM_TRANSPOSE_N == 0 &&
                   RUSTINFER_CUDA_GEMM_TRANSPOSE_T == 1 &&
                   RUSTINFER_CUDA_GEMM_LAYOUT_ROW_MAJOR == 1 &&
                   RUSTINFER_CUDA_GEMM_EPILOGUE_NONE == 0 &&
                   RUSTINFER_CUDA_GEMM_DETERMINISTIC_REQUIRED == 1 &&
                   RUSTINFER_CUDA_GEMM_BACKEND_CUBLASLT == 1,
               "GEMM ABI discriminants changed");
_Static_assert(sizeof(RustInferCudaGemmConfig) == 112,
               "GEMM config ABI size changed");
_Static_assert(offsetof(RustInferCudaGemmConfig, m) == 8,
               "GEMM config dimension offset changed");
_Static_assert(offsetof(RustInferCudaGemmConfig, input_dtype) == 32,
               "GEMM config dtype offset changed");
_Static_assert(offsetof(RustInferCudaGemmConfig, max_workspace_bytes) == 80,
               "GEMM config workspace offset changed");
_Static_assert(offsetof(RustInferCudaGemmConfig, reserved) == 88,
               "GEMM config reserved tail changed");
_Static_assert(sizeof(RustInferCudaGemmAlgorithmInfo) == 112,
               "GEMM algorithm-info ABI size changed");
_Static_assert(offsetof(RustInferCudaGemmAlgorithmInfo, workspace_bytes) == 40,
               "GEMM algorithm-info workspace offset changed");
_Static_assert(
    offsetof(RustInferCudaGemmAlgorithmInfo,
             numerical_implementation_flags) == 48,
    "GEMM algorithm-info numerical flags offset changed");
_Static_assert(offsetof(RustInferCudaGemmAlgorithmInfo, m) == 72,
               "GEMM algorithm-info dimension offset changed");
_Static_assert(offsetof(RustInferCudaGemmAlgorithmInfo, reserved) == 96,
               "GEMM algorithm-info reserved tail changed");

// Referencing every additive entry point makes incompatible C declarations a
// compile error without requiring a CUDA device or executing native code.
static RustInferCudaStatus (*const embedding_symbol)(
    const RustInferCudaEmbeddingParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_embedding_execute;
static RustInferCudaStatus (*const rms_norm_symbol)(
    const RustInferCudaRmsNormParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_rms_norm_execute;
static RustInferCudaStatus (*const residual_add_symbol)(
    const RustInferCudaResidualAddParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_residual_add_execute;
static RustInferCudaStatus (*const silu_symbol)(
    const RustInferCudaSiluParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_silu_execute;
static RustInferCudaStatus (*const gated_multiply_symbol)(
    const RustInferCudaGatedMultiplyParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_gated_multiply_execute;
static RustInferCudaStatus (*const rope_symbol)(
    const RustInferCudaRopeParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_rope_execute;
static RustInferCudaStatus (*const cast_symbol)(
    const RustInferCudaCastParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_cast_execute;
static RustInferCudaStatus (*const gemm_plan_create_symbol)(
    RustInferCudaContext*, const RustInferCudaGemmConfig*,
    RustInferCudaGemmPlan**,
    RustInferCudaErrorInfo*) = rustinfer_cuda_gemm_plan_create;
static RustInferCudaStatus (*const gemm_plan_info_symbol)(
    RustInferCudaGemmPlan*, RustInferCudaGemmAlgorithmInfo*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_gemm_plan_info;
static RustInferCudaStatus (*const gemm_plan_execute_symbol)(
    RustInferCudaGemmPlan*, const RustInferCudaBufferSpan*,
    const RustInferCudaBufferSpan*, const RustInferCudaBufferSpan*,
    const RustInferCudaBufferSpan*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_gemm_plan_execute;
static RustInferCudaStatus (*const gemm_plan_close_symbol)(
    RustInferCudaGemmPlan**,
    RustInferCudaErrorInfo*) = rustinfer_cuda_gemm_plan_close;

// Keep the otherwise compile-only references observably used under strict
// warning configurations.
const void* rustinfer_cuda_abi_symbol_references[] = {
    (const void*)&embedding_symbol,      (const void*)&rms_norm_symbol,
    (const void*)&residual_add_symbol,   (const void*)&silu_symbol,
    (const void*)&gated_multiply_symbol, (const void*)&rope_symbol,
    (const void*)&cast_symbol,           (const void*)&gemm_plan_create_symbol,
    (const void*)&gemm_plan_info_symbol,
    (const void*)&gemm_plan_execute_symbol,
    (const void*)&gemm_plan_close_symbol,
};
