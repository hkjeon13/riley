#include "rustinfer_cuda.h"

#include <stddef.h>

_Static_assert(RUSTINFER_CUDA_ABI_VERSION == 1,
               "PR 09 additions must preserve ABI v1");
_Static_assert(sizeof(void*) * 8 == RUSTINFER_CUDA_ABI_POINTER_WIDTH,
               "rustinfer CUDA ABI requires 64-bit pointers");
_Static_assert(sizeof(RustInferCudaDType) == 4,
               "dtype discriminant width changed");
_Static_assert(RUSTINFER_CUDA_DTYPE_F32 == 1 &&
                   RUSTINFER_CUDA_DTYPE_BF16 == 2 &&
                   RUSTINFER_CUDA_DTYPE_U32 == 3 &&
                   RUSTINFER_CUDA_DTYPE_U8 == 4 &&
                   RUSTINFER_CUDA_DTYPE_U16 == 5,
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
_Static_assert(sizeof(RustInferCudaQkGqaParams) == 216,
               "QK GQA params ABI size changed");
_Static_assert(offsetof(RustInferCudaQkGqaParams, query) == 8,
               "QK GQA query offset changed");
_Static_assert(offsetof(RustInferCudaQkGqaParams, token_count) == 152,
               "QK GQA dimension offset changed");
_Static_assert(offsetof(RustInferCudaQkGqaParams, reserved) == 184,
               "QK GQA reserved tail changed");
_Static_assert(sizeof(RustInferCudaScaleCausalMaskParams) == 112,
               "scale/mask params ABI size changed");
_Static_assert(offsetof(RustInferCudaScaleCausalMaskParams, scores) == 8,
               "scale/mask scores offset changed");
_Static_assert(offsetof(RustInferCudaScaleCausalMaskParams, scale) == 72,
               "scale/mask scalar offset changed");
_Static_assert(offsetof(RustInferCudaScaleCausalMaskParams, reserved) == 80,
               "scale/mask reserved tail changed");
_Static_assert(sizeof(RustInferCudaCausalSoftmaxParams) == 112,
               "causal-softmax params ABI size changed");
_Static_assert(offsetof(RustInferCudaCausalSoftmaxParams, scores) == 8,
               "causal-softmax scores offset changed");
_Static_assert(offsetof(RustInferCudaCausalSoftmaxParams, reserved) == 72,
               "causal-softmax reserved tail changed");
_Static_assert(sizeof(RustInferCudaAvGqaParams) == 216,
               "AV GQA params ABI size changed");
_Static_assert(offsetof(RustInferCudaAvGqaParams, probabilities) == 8,
               "AV GQA probabilities offset changed");
_Static_assert(offsetof(RustInferCudaAvGqaParams, token_count) == 152,
               "AV GQA dimension offset changed");
_Static_assert(offsetof(RustInferCudaAvGqaParams, reserved) == 184,
               "AV GQA reserved tail changed");
_Static_assert(RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL == 1 &&
                   RUSTINFER_CUDA_ATTENTION_MASK_CAUSAL_LOCAL == 2,
               "prefill attention mask discriminants changed");
_Static_assert(sizeof(RustInferCudaPrefillAttentionParams) == 288,
               "prefill attention params ABI size changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams, query) == 8,
               "prefill attention query offset changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams, output) == 152,
               "prefill attention output offset changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams, batch_count) ==
                   200,
               "prefill attention dimension offset changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams, scale) == 240,
               "prefill attention scale offset changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams,
                        local_window_size) == 248,
               "prefill attention local-window offset changed");
_Static_assert(offsetof(RustInferCudaPrefillAttentionParams, reserved) == 256,
               "prefill attention reserved tail changed");
_Static_assert(sizeof(RustInferCudaKvCacheWriteParams) == 272,
               "KV cache write params ABI size changed");
_Static_assert(offsetof(RustInferCudaKvCacheWriteParams, key_source) == 8,
               "KV cache write source offset changed");
_Static_assert(offsetof(RustInferCudaKvCacheWriteParams, key_cache) == 104,
               "KV cache write cache offset changed");
_Static_assert(
    offsetof(RustInferCudaKvCacheWriteParams, source_token_count) == 200,
    "KV cache write dimension offset changed");
_Static_assert(offsetof(RustInferCudaKvCacheWriteParams, reserved) == 240,
               "KV cache write reserved tail changed");
_Static_assert(sizeof(RustInferCudaDecodeAttentionReferenceParams) == 328,
               "decode reference params ABI size changed");
_Static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, query) == 8,
    "decode reference query offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, output) == 200,
    "decode reference output offset changed");
_Static_assert(offsetof(RustInferCudaDecodeAttentionReferenceParams,
                        maximum_token_count) == 248,
               "decode reference dimension offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, scale) == 288,
    "decode reference scale offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodeAttentionReferenceParams, reserved) == 296,
    "decode reference reserved tail changed");
_Static_assert(RUSTINFER_CUDA_DECODE_REDUCTION_ASCENDING == 1 &&
                   RUSTINFER_CUDA_DECODE_REDUCTION_DESCENDING == 2,
               "decode reduction-order discriminants changed");
_Static_assert(RUSTINFER_CUDA_DECODE_PARTIAL_STATE_VERSION == 1,
               "decode partial-state ABI version changed");
_Static_assert(sizeof(RustInferCudaDecodeAttentionParams) == 344,
               "decode attention params ABI size changed");
_Static_assert(offsetof(RustInferCudaDecodeAttentionParams, query) == 8,
               "decode attention query offset changed");
_Static_assert(offsetof(RustInferCudaDecodeAttentionParams, output) == 200,
               "decode attention output offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodeAttentionParams, maximum_token_count) == 248,
    "decode attention dimension offset changed");
_Static_assert(offsetof(RustInferCudaDecodeAttentionParams, scale) == 304,
               "decode attention scale offset changed");
_Static_assert(offsetof(RustInferCudaDecodeAttentionParams, reserved) == 312,
               "decode attention reserved tail changed");
_Static_assert(sizeof(RustInferCudaDecodePartialStateReduceParams) == 176,
               "decode reducer params ABI size changed");
_Static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams, partial_states) == 8,
    "decode reducer partial-state offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams,
             partial_state_count) == 104,
    "decode reducer dimension offset changed");
_Static_assert(offsetof(RustInferCudaDecodePartialStateReduceParams,
                        reduction_order) == 136,
               "decode reducer order offset changed");
_Static_assert(
    offsetof(RustInferCudaDecodePartialStateReduceParams, reserved) == 144,
    "decode reducer reserved tail changed");
_Static_assert(RUSTINFER_CUDA_PAGED_KV_BLOCK_TABLE_VERSION == 1 &&
                   RUSTINFER_CUDA_PAGED_KV_BLOCK_SIZE == 16 &&
                   RUSTINFER_CUDA_PAGED_KV_METADATA_NONE == 0,
               "paged KV constants changed");
_Static_assert(sizeof(RustInferCudaPagedKvBlockTableV1) == 168,
               "paged block-table ABI size changed");
_Static_assert(offsetof(RustInferCudaPagedKvBlockTableV1, block_ids) == 8,
               "paged block-table ids offset changed");
_Static_assert(
    offsetof(RustInferCudaPagedKvBlockTableV1, logical_token_count) == 104,
    "paged block-table logical length offset changed");
_Static_assert(offsetof(RustInferCudaPagedKvBlockTableV1, block_size) == 128,
               "paged block-table block-size offset changed");
_Static_assert(offsetof(RustInferCudaPagedKvBlockTableV1, reserved) == 144,
               "paged block-table reserved tail changed");
_Static_assert(sizeof(RustInferCudaPagedKvCacheWriteParams) == 432,
               "paged KV write ABI size changed");
_Static_assert(
    offsetof(RustInferCudaPagedKvCacheWriteParams, block_table) == 200,
    "paged KV write table offset changed");
_Static_assert(
    offsetof(RustInferCudaPagedKvCacheWriteParams, source_token_count) == 368,
    "paged KV write dimension offset changed");
_Static_assert(offsetof(RustInferCudaPagedKvCacheWriteParams, reserved) == 400,
               "paged KV write reserved tail changed");
_Static_assert(sizeof(RustInferCudaPagedDecodeAttentionReferenceParams) == 480,
               "paged reference decode ABI size changed");
_Static_assert(offsetof(RustInferCudaPagedDecodeAttentionReferenceParams,
                        block_table) == 248,
               "paged reference table offset changed");
_Static_assert(offsetof(RustInferCudaPagedDecodeAttentionReferenceParams,
                        query_head_count) == 416,
               "paged reference dimension offset changed");
_Static_assert(offsetof(RustInferCudaPagedDecodeAttentionReferenceParams,
                        reserved) == 448,
               "paged reference reserved tail changed");
_Static_assert(sizeof(RustInferCudaPagedDecodeAttentionParams) == 488,
               "paged online decode ABI size changed");
_Static_assert(
    offsetof(RustInferCudaPagedDecodeAttentionParams, block_table) == 248,
    "paged online table offset changed");
_Static_assert(
    offsetof(RustInferCudaPagedDecodeAttentionParams, query_head_count) == 416,
    "paged online dimension offset changed");
_Static_assert(offsetof(RustInferCudaPagedDecodeAttentionParams, reserved) ==
                   456,
               "paged online reserved tail changed");
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
static RustInferCudaStatus (*const qk_gqa_symbol)(
    const RustInferCudaQkGqaParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_qk_gqa_execute;
static RustInferCudaStatus (*const scale_causal_mask_symbol)(
    const RustInferCudaScaleCausalMaskParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) =
    rustinfer_cuda_scale_causal_mask_in_place_execute;
static RustInferCudaStatus (*const causal_softmax_symbol)(
    const RustInferCudaCausalSoftmaxParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) =
    rustinfer_cuda_causal_softmax_in_place_execute;
static RustInferCudaStatus (*const av_gqa_symbol)(
    const RustInferCudaAvGqaParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_av_gqa_execute;
static RustInferCudaStatus (*const prefill_attention_symbol)(
    const RustInferCudaPrefillAttentionParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_prefill_attention_execute;
static RustInferCudaStatus (*const kv_cache_write_symbol)(
    const RustInferCudaKvCacheWriteParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_kv_cache_write_execute;
static RustInferCudaStatus (*const decode_attention_reference_symbol)(
    const RustInferCudaDecodeAttentionReferenceParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) =
    rustinfer_cuda_decode_attention_reference_execute;
static RustInferCudaStatus (*const decode_attention_symbol)(
    const RustInferCudaDecodeAttentionParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_decode_attention_execute;
static RustInferCudaStatus (*const decode_partial_state_reduce_symbol)(
    const RustInferCudaDecodePartialStateReduceParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) =
    rustinfer_cuda_decode_partial_state_reduce_execute;
static RustInferCudaStatus (*const paged_kv_cache_write_symbol)(
    const RustInferCudaPagedKvCacheWriteParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_paged_kv_cache_write_execute;
static RustInferCudaStatus (*const paged_decode_attention_reference_symbol)(
    const RustInferCudaPagedDecodeAttentionReferenceParams*,
    RustInferCudaStream*, RustInferCudaErrorInfo*) =
    rustinfer_cuda_paged_decode_attention_reference_execute;
static RustInferCudaStatus (*const paged_decode_attention_symbol)(
    const RustInferCudaPagedDecodeAttentionParams*, RustInferCudaStream*,
    RustInferCudaErrorInfo*) = rustinfer_cuda_paged_decode_attention_execute;
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
    (const void*)&cast_symbol,           (const void*)&qk_gqa_symbol,
    (const void*)&scale_causal_mask_symbol,
    (const void*)&causal_softmax_symbol, (const void*)&av_gqa_symbol,
    (const void*)&prefill_attention_symbol,
    (const void*)&kv_cache_write_symbol,
    (const void*)&decode_attention_reference_symbol,
    (const void*)&decode_attention_symbol,
    (const void*)&decode_partial_state_reduce_symbol,
    (const void*)&paged_kv_cache_write_symbol,
    (const void*)&paged_decode_attention_reference_symbol,
    (const void*)&paged_decode_attention_symbol,
    (const void*)&gemm_plan_create_symbol,
    (const void*)&gemm_plan_info_symbol,
    (const void*)&gemm_plan_execute_symbol,
    (const void*)&gemm_plan_close_symbol,
};
