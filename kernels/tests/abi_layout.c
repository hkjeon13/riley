#include "riley_cuda.h"

#include <stddef.h>

_Static_assert(RILEY_CUDA_ABI_VERSION == 1,
               "additive CUDA entry points must preserve ABI v1");
_Static_assert(sizeof(void*) * 8 == RILEY_CUDA_ABI_POINTER_WIDTH,
               "riley CUDA ABI requires 64-bit pointers");
_Static_assert(sizeof(RileyCudaDType) == 4,
               "dtype discriminant width changed");
_Static_assert(RILEY_CUDA_DTYPE_F32 == 1 &&
                   RILEY_CUDA_DTYPE_BF16 == 2 &&
                   RILEY_CUDA_DTYPE_U32 == 3 &&
                   RILEY_CUDA_DTYPE_U8 == 4 &&
                   RILEY_CUDA_DTYPE_U16 == 5,
               "dtype discriminants changed");
_Static_assert(sizeof(RileyCudaErrorInfo) == 272,
               "error-info ABI size changed");
_Static_assert(offsetof(RileyCudaErrorInfo, struct_size) == 0,
               "error-info struct-size offset changed");
_Static_assert(offsetof(RileyCudaErrorInfo, native_code) == 4,
               "error-info native-code offset changed");
_Static_assert(offsetof(RileyCudaErrorInfo, domain) == 8,
               "error-info domain offset changed");
_Static_assert(offsetof(RileyCudaErrorInfo, stage) == 12,
               "error-info stage offset changed");
_Static_assert(offsetof(RileyCudaErrorInfo, message) == 16,
               "error-info message offset changed");
_Static_assert(sizeof(RileyCudaGraphCaptureMode) == 4,
               "graph-capture-mode ABI width changed");
_Static_assert(RILEY_CUDA_GRAPH_CAPTURE_MODE_INVALID == 0 &&
                   RILEY_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL == 1,
               "graph-capture-mode ABI discriminants changed");
_Static_assert(sizeof(RileyCudaGraphCaptureCapability) == 4,
               "graph-capture-capability ABI width changed");
_Static_assert(RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNKNOWN == 0 &&
                   RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_UNSUPPORTED == 1 &&
                   RILEY_CUDA_GRAPH_CAPTURE_CAPABILITY_SUPPORTED == 2,
               "graph-capture-capability ABI discriminants changed");
_Static_assert(sizeof(RileyCudaGraphStage) == 4,
               "graph-stage ABI width changed");
_Static_assert(RILEY_CUDA_GRAPH_STAGE_NONE == 0 &&
                   RILEY_CUDA_GRAPH_STAGE_CAPTURE_BEGIN == 1 &&
                   RILEY_CUDA_GRAPH_STAGE_CAPTURE_ENQUEUE == 2 &&
                   RILEY_CUDA_GRAPH_STAGE_CAPTURE_END == 3 &&
                   RILEY_CUDA_GRAPH_STAGE_CAPTURE_ABORT == 4 &&
                   RILEY_CUDA_GRAPH_STAGE_INSTANTIATE == 5 &&
                   RILEY_CUDA_GRAPH_STAGE_UPDATE == 6 &&
                   RILEY_CUDA_GRAPH_STAGE_LAUNCH == 7 &&
                   RILEY_CUDA_GRAPH_STAGE_COMPLETION == 8 &&
                   RILEY_CUDA_GRAPH_STAGE_CLOSE == 9,
               "graph-stage ABI discriminants changed");
_Static_assert(sizeof(RileyCudaGraphErrorInfo) == 56,
               "graph-error-info ABI size changed");
_Static_assert(_Alignof(RileyCudaGraphErrorInfo) == 8,
               "graph-error-info ABI alignment changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, struct_size) == 0,
               "graph-error-info struct-size offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, graph_stage) == 4,
               "graph-error-info stage offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, capture_id) == 8,
               "graph-error-info capture-id offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, exec_id) == 16,
               "graph-error-info exec-id offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, submission_started) == 24,
               "graph-error-info submission flag offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, completion_known) == 25,
               "graph-error-info completion flag offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, resource_release_known) == 26,
               "graph-error-info release flag offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, poisoned) == 27,
               "graph-error-info poisoned flag offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, reserved0) == 28,
               "graph-error-info reserved0 offset changed");
_Static_assert(offsetof(RileyCudaGraphErrorInfo, reserved) == 32,
               "graph-error-info reserved tail offset changed");
_Static_assert(sizeof(RileyCudaNvidiaDeviceSnapshot) == 320,
               "NVIDIA device snapshot ABI size changed");
_Static_assert(offsetof(RileyCudaNvidiaDeviceSnapshot, name) == 64,
               "NVIDIA device snapshot name offset changed");
_Static_assert(sizeof(RileyCudaNvidiaEnvironmentSnapshot) == 10352,
               "NVIDIA environment snapshot ABI size changed");
_Static_assert(
    offsetof(RileyCudaNvidiaEnvironmentSnapshot, driver_version) == 32,
    "NVIDIA environment driver-version offset changed");
_Static_assert(offsetof(RileyCudaNvidiaEnvironmentSnapshot, devices) == 112,
               "NVIDIA environment devices offset changed");
_Static_assert(RILEY_CUDA_NVIDIA_ENVIRONMENT_MAX_DEVICES == 32 &&
                   RILEY_CUDA_NVIDIA_PERSISTENCE_DISABLED == 0 &&
                   RILEY_CUDA_NVIDIA_PERSISTENCE_ENABLED == 1 &&
                   RILEY_CUDA_ERROR_DOMAIN_NVML == 6,
               "NVML ABI constants changed");
_Static_assert(sizeof(RileyCudaBufferSpan) == 48,
               "buffer-span ABI size changed");
_Static_assert(offsetof(RileyCudaBufferSpan, buffer) == 8,
               "buffer-span handle offset changed");
_Static_assert(offsetof(RileyCudaBufferSpan, byte_offset) == 16,
               "buffer-span offset field changed");
_Static_assert(offsetof(RileyCudaBufferSpan, reserved) == 32,
               "buffer-span reserved tail changed");
_Static_assert(sizeof(RileyCudaEmbeddingErrorReport) == 32,
               "embedding-report ABI size changed");
_Static_assert(offsetof(RileyCudaEmbeddingErrorReport, token_position) == 8,
               "embedding-report token position changed");
_Static_assert(sizeof(RileyCudaEmbeddingParams) == 256,
               "embedding-params ABI size changed");
_Static_assert(offsetof(RileyCudaEmbeddingParams, out_report) == 200,
               "embedding-params output report offset changed");
_Static_assert(offsetof(RileyCudaEmbeddingParams, token_count) == 208,
               "embedding-params dimension offset changed");
_Static_assert(sizeof(RileyCudaRmsNormParams) == 208,
               "RMSNorm-params ABI size changed");
_Static_assert(offsetof(RileyCudaRmsNormParams, epsilon) == 168,
               "RMSNorm epsilon offset changed");
_Static_assert(sizeof(RileyCudaFixed37LogSoftmaxParams) == 152,
               "fixed37 log-softmax params ABI size changed");
_Static_assert(offsetof(RileyCudaFixed37LogSoftmaxParams, logits) == 8,
               "fixed37 log-softmax logits offset changed");
_Static_assert(offsetof(RileyCudaFixed37LogSoftmaxParams, output) == 56,
               "fixed37 log-softmax output offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37LogSoftmaxParams, element_count) == 104,
    "fixed37 log-softmax dimension offset changed");
_Static_assert(sizeof(RileyCudaResidualAddParams) == 200,
               "residual-add params ABI size changed");
_Static_assert(sizeof(RileyCudaResidualRmsNormParams) == 304,
               "residual-RMSNorm params ABI size changed");
_Static_assert(
    offsetof(RileyCudaResidualRmsNormParams, residual_output) == 152,
    "residual-RMSNorm residual output offset changed");
_Static_assert(offsetof(RileyCudaResidualRmsNormParams, row_count) == 248,
               "residual-RMSNorm dimension offset changed");
_Static_assert(offsetof(RileyCudaResidualRmsNormParams, epsilon) == 264,
               "residual-RMSNorm epsilon offset changed");
_Static_assert(sizeof(RileyCudaRowBiasAddInPlaceParams) == 152,
               "row-bias params ABI size changed");
_Static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, matrix) == 8,
               "row-bias matrix offset changed");
_Static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, row_count) == 104,
               "row-bias dimension offset changed");
_Static_assert(offsetof(RileyCudaRowBiasAddInPlaceParams, reserved) == 120,
               "row-bias reserved tail changed");
_Static_assert(sizeof(RileyCudaSiluParams) == 152,
               "SiLU params ABI size changed");
_Static_assert(sizeof(RileyCudaGatedMultiplyParams) == 200,
               "gated-multiply params ABI size changed");
_Static_assert(sizeof(RileyCudaRopeTableParams) == 152,
               "RoPE table params ABI size changed");
_Static_assert(offsetof(RileyCudaRopeTableParams, element_count) == 104,
               "RoPE table dimension offset changed");
_Static_assert(sizeof(RileyCudaRopeParams) == 288,
               "RoPE params ABI size changed");
_Static_assert(offsetof(RileyCudaRopeParams, token_count) == 200,
               "RoPE dimension offset changed");
_Static_assert(offsetof(RileyCudaRopeParams, reserved) == 248,
               "RoPE reserved tail changed");
_Static_assert(sizeof(RileyCudaIndexedRopeParams) == 320,
               "indexed RoPE params ABI size changed");
_Static_assert(offsetof(RileyCudaIndexedRopeParams, input) == 8,
               "indexed RoPE input offset changed");
_Static_assert(offsetof(RileyCudaIndexedRopeParams, positions) == 152,
               "indexed RoPE positions offset changed");
_Static_assert(
    offsetof(RileyCudaIndexedRopeParams, active_row_count) == 248,
    "indexed RoPE dimension offset changed");
_Static_assert(offsetof(RileyCudaIndexedRopeParams, reserved) == 288,
               "indexed RoPE reserved tail changed");
_Static_assert(sizeof(RileyCudaCastParams) == 152,
               "cast params ABI size changed");
_Static_assert(sizeof(RileyCudaRowGatherParams) == 208,
               "row gather params ABI size changed");
_Static_assert(offsetof(RileyCudaRowGatherParams, input) == 8,
               "row gather input offset changed");
_Static_assert(offsetof(RileyCudaRowGatherParams, row_indices) == 56,
               "row gather indices offset changed");
_Static_assert(offsetof(RileyCudaRowGatherParams, input_row_count) == 152,
               "row gather dimension offset changed");
_Static_assert(offsetof(RileyCudaRowGatherParams, reserved) == 176,
               "row gather reserved tail changed");
_Static_assert(sizeof(RileyCudaBf16ArgmaxResult) == 8,
               "BF16 argmax result ABI size changed");
_Static_assert(offsetof(RileyCudaBf16ArgmaxResult, status) == 4,
               "BF16 argmax result status offset changed");
_Static_assert(RILEY_CUDA_BF16_ARGMAX_STATUS_SUCCESS == 0 &&
                   RILEY_CUDA_BF16_ARGMAX_STATUS_NON_FINITE == 1 &&
                   RILEY_CUDA_BF16_ARGMAX_INVALID_TOKEN_ID == UINT32_MAX,
               "BF16 argmax result constants changed");
_Static_assert(sizeof(RileyCudaBf16ArgmaxParams) == 152,
               "BF16 argmax params ABI size changed");
_Static_assert(offsetof(RileyCudaBf16ArgmaxParams, logits) == 8,
               "BF16 argmax logits offset changed");
_Static_assert(offsetof(RileyCudaBf16ArgmaxParams, results) == 56,
               "BF16 argmax results offset changed");
_Static_assert(offsetof(RileyCudaBf16ArgmaxParams, row_count) == 104,
               "BF16 argmax dimension offset changed");
_Static_assert(offsetof(RileyCudaBf16ArgmaxParams, reserved) == 120,
               "BF16 argmax reserved tail changed");
_Static_assert(sizeof(RileyCudaQkGqaParams) == 216,
               "QK GQA params ABI size changed");
_Static_assert(offsetof(RileyCudaQkGqaParams, query) == 8,
               "QK GQA query offset changed");
_Static_assert(offsetof(RileyCudaQkGqaParams, token_count) == 152,
               "QK GQA dimension offset changed");
_Static_assert(offsetof(RileyCudaQkGqaParams, reserved) == 184,
               "QK GQA reserved tail changed");
_Static_assert(sizeof(RileyCudaScaleCausalMaskParams) == 112,
               "scale/mask params ABI size changed");
_Static_assert(offsetof(RileyCudaScaleCausalMaskParams, scores) == 8,
               "scale/mask scores offset changed");
_Static_assert(offsetof(RileyCudaScaleCausalMaskParams, scale) == 72,
               "scale/mask scalar offset changed");
_Static_assert(offsetof(RileyCudaScaleCausalMaskParams, reserved) == 80,
               "scale/mask reserved tail changed");
_Static_assert(sizeof(RileyCudaCausalSoftmaxParams) == 112,
               "causal-softmax params ABI size changed");
_Static_assert(offsetof(RileyCudaCausalSoftmaxParams, scores) == 8,
               "causal-softmax scores offset changed");
_Static_assert(offsetof(RileyCudaCausalSoftmaxParams, reserved) == 72,
               "causal-softmax reserved tail changed");
_Static_assert(sizeof(RileyCudaAvGqaParams) == 216,
               "AV GQA params ABI size changed");
_Static_assert(offsetof(RileyCudaAvGqaParams, probabilities) == 8,
               "AV GQA probabilities offset changed");
_Static_assert(offsetof(RileyCudaAvGqaParams, token_count) == 152,
               "AV GQA dimension offset changed");
_Static_assert(offsetof(RileyCudaAvGqaParams, reserved) == 184,
               "AV GQA reserved tail changed");
_Static_assert(RILEY_CUDA_ATTENTION_MASK_CAUSAL == 1 &&
                   RILEY_CUDA_ATTENTION_MASK_CAUSAL_LOCAL == 2,
               "prefill attention mask discriminants changed");
_Static_assert(sizeof(RileyCudaPrefillAttentionParams) == 288,
               "prefill attention params ABI size changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams, query) == 8,
               "prefill attention query offset changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams, output) == 152,
               "prefill attention output offset changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams, batch_count) ==
                   200,
               "prefill attention dimension offset changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams, scale) == 240,
               "prefill attention scale offset changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams,
                        local_window_size) == 248,
               "prefill attention local-window offset changed");
_Static_assert(offsetof(RileyCudaPrefillAttentionParams, reserved) == 256,
               "prefill attention reserved tail changed");
_Static_assert(RILEY_CUDA_ATTENTION_BACKEND_HF_CUBLASLT == 3,
               "HF prefill attention backend discriminant changed");
_Static_assert(sizeof(RileyCudaHfPrefillAttentionConfig) == 96,
               "HF prefill attention config ABI size changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionConfig, batch_count) == 8,
    "HF prefill attention config batch offset changed");
_Static_assert(offsetof(RileyCudaHfPrefillAttentionConfig,
                        max_cublas_workspace_bytes) == 56,
               "HF prefill attention config workspace cap offset changed");
_Static_assert(offsetof(RileyCudaHfPrefillAttentionConfig, reserved) == 64,
               "HF prefill attention config reserved tail changed");
_Static_assert(sizeof(RileyCudaHfPrefillAttentionPlanInfo) == 216,
               "HF prefill attention plan-info ABI size changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionPlanInfo, qk_workspace_bytes) ==
        40,
    "HF prefill attention plan-info QK workspace offset changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionPlanInfo, av_workspace_bytes) ==
        88,
    "HF prefill attention plan-info AV workspace offset changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionPlanInfo, workspace_bytes) == 128,
    "HF prefill attention plan-info workspace offset changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionPlanInfo, batch_count) == 160,
    "HF prefill attention plan-info batch offset changed");
_Static_assert(
    offsetof(RileyCudaHfPrefillAttentionPlanInfo, reserved) == 200,
    "HF prefill attention plan-info reserved tail changed");
_Static_assert(sizeof(RileyCudaKvCacheWriteParams) == 272,
               "KV cache write params ABI size changed");
_Static_assert(offsetof(RileyCudaKvCacheWriteParams, key_source) == 8,
               "KV cache write source offset changed");
_Static_assert(offsetof(RileyCudaKvCacheWriteParams, key_cache) == 104,
               "KV cache write cache offset changed");
_Static_assert(
    offsetof(RileyCudaKvCacheWriteParams, source_token_count) == 200,
    "KV cache write dimension offset changed");
_Static_assert(offsetof(RileyCudaKvCacheWriteParams, reserved) == 240,
               "KV cache write reserved tail changed");
_Static_assert(sizeof(RileyCudaDecodeAttentionReferenceParams) == 328,
               "decode reference params ABI size changed");
_Static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, query) == 8,
    "decode reference query offset changed");
_Static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, output) == 200,
    "decode reference output offset changed");
_Static_assert(offsetof(RileyCudaDecodeAttentionReferenceParams,
                        maximum_token_count) == 248,
               "decode reference dimension offset changed");
_Static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, scale) == 288,
    "decode reference scale offset changed");
_Static_assert(
    offsetof(RileyCudaDecodeAttentionReferenceParams, reserved) == 296,
    "decode reference reserved tail changed");
_Static_assert(RILEY_CUDA_DECODE_REDUCTION_ASCENDING == 1 &&
                   RILEY_CUDA_DECODE_REDUCTION_DESCENDING == 2,
               "decode reduction-order discriminants changed");
_Static_assert(RILEY_CUDA_DECODE_PARTIAL_STATE_VERSION == 1,
               "decode partial-state ABI version changed");
_Static_assert(sizeof(RileyCudaDecodeAttentionParams) == 344,
               "decode attention params ABI size changed");
_Static_assert(offsetof(RileyCudaDecodeAttentionParams, query) == 8,
               "decode attention query offset changed");
_Static_assert(offsetof(RileyCudaDecodeAttentionParams, output) == 200,
               "decode attention output offset changed");
_Static_assert(
    offsetof(RileyCudaDecodeAttentionParams, maximum_token_count) == 248,
    "decode attention dimension offset changed");
_Static_assert(offsetof(RileyCudaDecodeAttentionParams, scale) == 304,
               "decode attention scale offset changed");
_Static_assert(offsetof(RileyCudaDecodeAttentionParams, reserved) == 312,
               "decode attention reserved tail changed");
_Static_assert(sizeof(RileyCudaDecodePartialStateReduceParams) == 176,
               "decode reducer params ABI size changed");
_Static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams, partial_states) == 8,
    "decode reducer partial-state offset changed");
_Static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams,
             partial_state_count) == 104,
    "decode reducer dimension offset changed");
_Static_assert(offsetof(RileyCudaDecodePartialStateReduceParams,
                        reduction_order) == 136,
               "decode reducer order offset changed");
_Static_assert(
    offsetof(RileyCudaDecodePartialStateReduceParams, reserved) == 144,
    "decode reducer reserved tail changed");
_Static_assert(RILEY_CUDA_PAGED_KV_BLOCK_TABLE_VERSION == 1 &&
                   RILEY_CUDA_PAGED_KV_BLOCK_SIZE == 16 &&
                   RILEY_CUDA_PAGED_KV_METADATA_NONE == 0,
               "paged KV constants changed");
_Static_assert(sizeof(RileyCudaPagedKvBlockTableV1) == 168,
               "paged block-table ABI size changed");
_Static_assert(offsetof(RileyCudaPagedKvBlockTableV1, block_ids) == 8,
               "paged block-table ids offset changed");
_Static_assert(
    offsetof(RileyCudaPagedKvBlockTableV1, logical_token_count) == 104,
    "paged block-table logical length offset changed");
_Static_assert(offsetof(RileyCudaPagedKvBlockTableV1, block_size) == 128,
               "paged block-table block-size offset changed");
_Static_assert(offsetof(RileyCudaPagedKvBlockTableV1, reserved) == 144,
               "paged block-table reserved tail changed");
_Static_assert(sizeof(RileyCudaPagedKvCacheWriteParams) == 432,
               "paged KV write ABI size changed");
_Static_assert(
    offsetof(RileyCudaPagedKvCacheWriteParams, block_table) == 200,
    "paged KV write table offset changed");
_Static_assert(
    offsetof(RileyCudaPagedKvCacheWriteParams, source_token_count) == 368,
    "paged KV write dimension offset changed");
_Static_assert(offsetof(RileyCudaPagedKvCacheWriteParams, reserved) == 400,
               "paged KV write reserved tail changed");
_Static_assert(sizeof(RileyCudaPagedDecodeAttentionReferenceParams) == 480,
               "paged reference decode ABI size changed");
_Static_assert(offsetof(RileyCudaPagedDecodeAttentionReferenceParams,
                        block_table) == 248,
               "paged reference table offset changed");
_Static_assert(offsetof(RileyCudaPagedDecodeAttentionReferenceParams,
                        query_head_count) == 416,
               "paged reference dimension offset changed");
_Static_assert(offsetof(RileyCudaPagedDecodeAttentionReferenceParams,
                        reserved) == 448,
               "paged reference reserved tail changed");
_Static_assert(sizeof(RileyCudaPagedDecodeAttentionParams) == 488,
               "paged online decode ABI size changed");
_Static_assert(
    offsetof(RileyCudaPagedDecodeAttentionParams, block_table) == 248,
    "paged online table offset changed");
_Static_assert(
    offsetof(RileyCudaPagedDecodeAttentionParams, query_head_count) == 416,
    "paged online dimension offset changed");
_Static_assert(offsetof(RileyCudaPagedDecodeAttentionParams, reserved) ==
                   456,
               "paged online reserved tail changed");
_Static_assert(RILEY_CUDA_PACKED_BATCH_VERSION == 1,
               "packed batch ABI version changed");
_Static_assert(sizeof(RileyCudaPackedBatchV1) == 320,
               "packed batch ABI size changed");
_Static_assert(
    offsetof(RileyCudaPackedBatchV1, sequence_block_offsets) == 8,
    "packed batch CSR offsets span changed");
_Static_assert(offsetof(RileyCudaPackedBatchV1, row_positions) == 200,
               "packed batch row positions span changed");
_Static_assert(offsetof(RileyCudaPackedBatchV1, sequence_count) == 248,
               "packed batch dimension offset changed");
_Static_assert(offsetof(RileyCudaPackedBatchV1, block_size) == 280,
               "packed batch block-size offset changed");
_Static_assert(offsetof(RileyCudaPackedBatchV1, reserved) == 288,
               "packed batch reserved tail changed");
_Static_assert(sizeof(RileyCudaRaggedPagedKvCacheWriteParams) == 568,
               "ragged paged KV write ABI size changed");
_Static_assert(
    offsetof(RileyCudaRaggedPagedKvCacheWriteParams, batch) == 200,
    "ragged paged KV write batch offset changed");
_Static_assert(offsetof(RileyCudaRaggedPagedKvCacheWriteParams,
                        key_value_head_count) == 520,
               "ragged paged KV write dimension offset changed");
_Static_assert(
    offsetof(RileyCudaRaggedPagedKvCacheWriteParams, reserved) == 536,
    "ragged paged KV write reserved tail changed");
_Static_assert(sizeof(RileyCudaRaggedPagedAttentionParams) == 592,
               "ragged paged attention ABI size changed");
_Static_assert(
    offsetof(RileyCudaRaggedPagedAttentionParams, batch) == 200,
    "ragged paged attention batch offset changed");
_Static_assert(offsetof(RileyCudaRaggedPagedAttentionParams,
                        query_head_count) == 520,
               "ragged paged attention dimension offset changed");
_Static_assert(offsetof(RileyCudaRaggedPagedAttentionParams,
                        output_row_count) == 544,
               "ragged paged attention output-row offset changed");
_Static_assert(offsetof(RileyCudaRaggedPagedAttentionParams, scale) == 552,
               "ragged paged attention scale offset changed");
_Static_assert(
    offsetof(RileyCudaRaggedPagedAttentionParams, reserved) == 560,
    "ragged paged attention reserved tail changed");
_Static_assert(sizeof(RileyCudaFixed37RaggedPagedAttentionParams) == 600,
               "fixed37 ragged paged attention ABI size changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, struct_size) == 0,
    "fixed37 ragged paged attention struct-size offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved0) == 4,
    "fixed37 ragged paged attention reserved0 offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, query) == 8,
    "fixed37 ragged paged attention query offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, key_pool) == 56,
    "fixed37 ragged paged attention key-pool offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, value_pool) == 104,
    "fixed37 ragged paged attention value-pool offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, output) == 152,
    "fixed37 ragged paged attention output offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, batch) == 200,
    "fixed37 ragged paged attention batch offset changed");
_Static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
                        query_head_count) == 520,
               "fixed37 ragged paged attention QH offset changed");
_Static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
                        key_value_head_count) == 528,
               "fixed37 ragged paged attention KVH offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, head_size) == 536,
    "fixed37 ragged paged attention head-size offset changed");
_Static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
                        output_row_count) == 544,
               "fixed37 ragged paged attention output-row offset changed");
_Static_assert(offsetof(RileyCudaFixed37RaggedPagedAttentionParams,
                        maximum_logical_token_count) == 552,
               "fixed37 ragged paged attention maximum-T offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, scale) == 560,
    "fixed37 ragged paged attention scale offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved1) == 564,
    "fixed37 ragged paged attention reserved1 offset changed");
_Static_assert(
    offsetof(RileyCudaFixed37RaggedPagedAttentionParams, reserved) == 568,
    "fixed37 ragged paged attention reserved tail changed");
_Static_assert(RILEY_CUDA_STATUS_CUBLASLT_ERROR == 10 &&
                   RILEY_CUDA_STATUS_NOT_SUPPORTED == 11,
               "GEMM status discriminants changed");
_Static_assert(RILEY_CUDA_ERROR_DOMAIN_CUBLASLT == 5,
               "cuBLASLt error domain changed");
_Static_assert(RILEY_CUDA_GEMM_TRANSPOSE_N == 0 &&
                   RILEY_CUDA_GEMM_TRANSPOSE_T == 1 &&
                   RILEY_CUDA_GEMM_LAYOUT_ROW_MAJOR == 1 &&
                   RILEY_CUDA_GEMM_EPILOGUE_NONE == 0 &&
                   RILEY_CUDA_GEMM_DETERMINISTIC_REQUIRED == 1 &&
                   RILEY_CUDA_GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K == 1 &&
                   RILEY_CUDA_GEMM_FLAG_ALLOW_INPLACE_SPLIT_K == 2 &&
                   RILEY_CUDA_GEMM_BACKEND_CUBLASLT == 1 &&
                   RILEY_CUDA_GEMM_BACKEND_FIXED37 == 2 &&
                   RILEY_CUDA_FIXED37_REDUCTION_VERSION == 1 &&
                   RILEY_CUDA_FIXED37_CHUNK_ELEMENTS == 37 &&
                   RILEY_CUDA_FIXED37_MAX_CHUNK_COUNT == 4096,
               "GEMM ABI discriminants changed");
_Static_assert(sizeof(RileyCudaGemmConfig) == 112,
               "GEMM config ABI size changed");
_Static_assert(offsetof(RileyCudaGemmConfig, flags) == 4,
               "GEMM config flags offset changed");
_Static_assert(offsetof(RileyCudaGemmConfig, m) == 8,
               "GEMM config dimension offset changed");
_Static_assert(offsetof(RileyCudaGemmConfig, input_dtype) == 32,
               "GEMM config dtype offset changed");
_Static_assert(offsetof(RileyCudaGemmConfig, max_workspace_bytes) == 80,
               "GEMM config workspace offset changed");
_Static_assert(offsetof(RileyCudaGemmConfig, reserved) == 88,
               "GEMM config reserved tail changed");
_Static_assert(sizeof(RileyCudaGemmAlgorithmInfo) == 112,
               "GEMM algorithm-info ABI size changed");
_Static_assert(offsetof(RileyCudaGemmAlgorithmInfo, workspace_bytes) == 40,
               "GEMM algorithm-info workspace offset changed");
_Static_assert(
    offsetof(RileyCudaGemmAlgorithmInfo,
             numerical_implementation_flags) == 48,
    "GEMM algorithm-info numerical flags offset changed");
_Static_assert(offsetof(RileyCudaGemmAlgorithmInfo, m) == 72,
               "GEMM algorithm-info dimension offset changed");
_Static_assert(offsetof(RileyCudaGemmAlgorithmInfo, reserved) == 96,
               "GEMM algorithm-info reserved tail changed");
_Static_assert(sizeof(RileyCudaFixed37GemmPlanInfo) == 96,
               "fixed37 GEMM plan-info ABI size changed");
_Static_assert(
    offsetof(RileyCudaFixed37GemmPlanInfo,
             dynamic_shared_memory_bytes) == 32,
    "fixed37 GEMM plan-info shared-memory offset changed");
_Static_assert(offsetof(RileyCudaFixed37GemmPlanInfo, m) == 48,
               "fixed37 GEMM plan-info dimension offset changed");
_Static_assert(offsetof(RileyCudaFixed37GemmPlanInfo, reserved) == 72,
               "fixed37 GEMM plan-info reserved tail changed");

// Referencing every additive entry point makes incompatible C declarations a
// compile error without requiring a CUDA device or executing native code.
static RileyCudaStatus (*const nvidia_environment_probe_symbol)(
    RileyCudaNvidiaEnvironmentSnapshot*, RileyCudaErrorInfo*) =
    riley_cuda_nvidia_environment_probe;
static RileyCudaStatus (*const graph_capture_begin_symbol)(
    RileyCudaStream*, RileyCudaGraphCaptureMode, RileyCudaGraphCapture**,
    RileyCudaGraphErrorInfo*, RileyCudaErrorInfo*) =
    riley_cuda_graph_capture_begin;
static RileyCudaStatus (*const graph_capture_abort_symbol)(
    RileyCudaGraphCapture**, RileyCudaGraphErrorInfo*, RileyCudaErrorInfo*) =
    riley_cuda_graph_capture_abort;
static RileyCudaStatus (*const context_defer_to_active_capture_symbol)(
    RileyCudaContext**, RileyCudaErrorInfo*) =
    riley_cuda_context_defer_to_active_capture;
static RileyCudaStatus (*const stream_defer_to_active_capture_symbol)(
    RileyCudaStream**, RileyCudaErrorInfo*) =
    riley_cuda_stream_defer_to_active_capture;
static RileyCudaStatus (*const event_defer_to_active_capture_symbol)(
    RileyCudaEvent**, RileyCudaErrorInfo*) =
    riley_cuda_event_defer_to_active_capture;
static RileyCudaStatus (*const device_buffer_defer_to_active_capture_symbol)(
    RileyCudaDeviceBuffer**, RileyCudaErrorInfo*) =
    riley_cuda_device_buffer_defer_to_active_capture;
static RileyCudaStatus (*const pinned_host_buffer_defer_to_active_capture_symbol)(
    RileyCudaPinnedHostBuffer**, RileyCudaErrorInfo*) =
    riley_cuda_pinned_host_buffer_defer_to_active_capture;
static RileyCudaStatus (*const command_batch_copy_h2d_symbol)(
    RileyCudaDeviceBuffer*, uint64_t, RileyCudaPinnedHostBuffer*, uint64_t,
    uint64_t, RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_command_batch_copy_h2d_async;
static RileyCudaStatus (*const embedding_symbol)(
    const RileyCudaEmbeddingParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_embedding_execute;
static RileyCudaStatus (*const rms_norm_symbol)(
    const RileyCudaRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_rms_norm_execute;
static RileyCudaStatus (*const hugging_face_smollm2_rms_norm_symbol)(
    const RileyCudaRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_hugging_face_smollm2_rms_norm_execute;
static RileyCudaStatus (*const fixed37_rms_norm_symbol)(
    const RileyCudaRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_fixed37_rms_norm_execute;
static RileyCudaStatus (*const residual_add_symbol)(
    const RileyCudaResidualAddParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_residual_add_execute;
static RileyCudaStatus (*const residual_rms_norm_symbol)(
    const RileyCudaResidualRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_residual_rms_norm_execute;
static RileyCudaStatus (*const
                                hugging_face_smollm2_residual_rms_norm_symbol)(
    const RileyCudaResidualRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_hugging_face_smollm2_residual_rms_norm_execute;
static RileyCudaStatus (*const fixed37_residual_rms_norm_symbol)(
    const RileyCudaResidualRmsNormParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_fixed37_residual_rms_norm_execute;
static RileyCudaStatus (*const fixed37_log_softmax_symbol)(
    const RileyCudaFixed37LogSoftmaxParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_fixed37_log_softmax_execute;
static RileyCudaStatus (*const row_bias_add_symbol)(
    const RileyCudaRowBiasAddInPlaceParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_row_bias_add_in_place_execute;
static RileyCudaStatus (*const silu_symbol)(
    const RileyCudaSiluParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_silu_execute;
static RileyCudaStatus (*const gated_multiply_symbol)(
    const RileyCudaGatedMultiplyParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_gated_multiply_execute;
static RileyCudaStatus (*const rope_table_symbol)(
    const RileyCudaRopeTableParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_rope_table_execute;
static RileyCudaStatus (*const rope_symbol)(
    const RileyCudaRopeParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_rope_execute;
static RileyCudaStatus (*const indexed_rope_symbol)(
    const RileyCudaIndexedRopeParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_indexed_rope_execute;
static RileyCudaStatus (*const cast_symbol)(
    const RileyCudaCastParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_cast_execute;
static RileyCudaStatus (*const row_gather_symbol)(
    const RileyCudaRowGatherParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_row_gather_execute;
static RileyCudaStatus (*const bf16_argmax_symbol)(
    const RileyCudaBf16ArgmaxParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_bf16_argmax_execute;
static RileyCudaStatus (*const qk_gqa_symbol)(
    const RileyCudaQkGqaParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_qk_gqa_execute;
static RileyCudaStatus (*const scale_causal_mask_symbol)(
    const RileyCudaScaleCausalMaskParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_scale_causal_mask_in_place_execute;
static RileyCudaStatus (*const causal_softmax_symbol)(
    const RileyCudaCausalSoftmaxParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_causal_softmax_in_place_execute;
static RileyCudaStatus (*const av_gqa_symbol)(
    const RileyCudaAvGqaParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_av_gqa_execute;
static RileyCudaStatus (*const fixed37_qk_gqa_symbol)(
    const RileyCudaQkGqaParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_fixed37_qk_gqa_execute;
static RileyCudaStatus (*const fixed37_causal_softmax_symbol)(
    const RileyCudaCausalSoftmaxParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_fixed37_causal_softmax_in_place_execute;
static RileyCudaStatus (*const fixed37_av_gqa_symbol)(
    const RileyCudaAvGqaParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_fixed37_av_gqa_execute;
static RileyCudaStatus (*const prefill_attention_symbol)(
    const RileyCudaPrefillAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_prefill_attention_execute;
static RileyCudaStatus (*const fixed37_prefill_attention_symbol)(
    const RileyCudaPrefillAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_fixed37_prefill_attention_execute;
static RileyCudaStatus (*const hf_prefill_attention_plan_create_symbol)(
    RileyCudaContext*, const RileyCudaHfPrefillAttentionConfig*,
    RileyCudaHfPrefillAttentionPlan**, RileyCudaErrorInfo*) =
    riley_cuda_hf_prefill_attention_plan_create;
static RileyCudaStatus (*const hf_prefill_attention_plan_info_symbol)(
    RileyCudaHfPrefillAttentionPlan*,
    RileyCudaHfPrefillAttentionPlanInfo*, RileyCudaErrorInfo*) =
    riley_cuda_hf_prefill_attention_plan_info;
static RileyCudaStatus (*const hf_prefill_attention_plan_execute_symbol)(
    RileyCudaHfPrefillAttentionPlan*, const RileyCudaBufferSpan*,
    const RileyCudaBufferSpan*, const RileyCudaBufferSpan*,
    const RileyCudaBufferSpan*, const RileyCudaBufferSpan*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_hf_prefill_attention_plan_execute;
static RileyCudaStatus (*const hf_prefill_attention_plan_close_symbol)(
    RileyCudaHfPrefillAttentionPlan**, RileyCudaErrorInfo*) =
    riley_cuda_hf_prefill_attention_plan_close;
static RileyCudaStatus (*const
                                hf_prefill_attention_plan_defer_to_active_capture_symbol)(
    RileyCudaHfPrefillAttentionPlan**, RileyCudaErrorInfo*) =
    riley_cuda_hf_prefill_attention_plan_defer_to_active_capture;
static RileyCudaStatus (*const kv_cache_write_symbol)(
    const RileyCudaKvCacheWriteParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_kv_cache_write_execute;
static RileyCudaStatus (*const decode_attention_reference_symbol)(
    const RileyCudaDecodeAttentionReferenceParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_decode_attention_reference_execute;
static RileyCudaStatus (*const fixed37_decode_attention_reference_symbol)(
    const RileyCudaDecodeAttentionReferenceParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_fixed37_decode_attention_reference_execute;
static RileyCudaStatus (*const fixed37_decode_attention_two_pass_symbol)(
    const RileyCudaDecodeAttentionReferenceParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_fixed37_decode_attention_two_pass_execute;
static RileyCudaStatus (*const decode_attention_symbol)(
    const RileyCudaDecodeAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_decode_attention_execute;
static RileyCudaStatus (*const decode_partial_state_reduce_symbol)(
    const RileyCudaDecodePartialStateReduceParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_decode_partial_state_reduce_execute;
static RileyCudaStatus (*const paged_kv_cache_write_symbol)(
    const RileyCudaPagedKvCacheWriteParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_paged_kv_cache_write_execute;
static RileyCudaStatus (*const paged_decode_attention_reference_symbol)(
    const RileyCudaPagedDecodeAttentionReferenceParams*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_paged_decode_attention_reference_execute;
static RileyCudaStatus (*const
                                fixed37_paged_decode_attention_reference_symbol)(
    const RileyCudaPagedDecodeAttentionReferenceParams*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_paged_decode_attention_reference_execute;
static RileyCudaStatus (*const
                                fixed37_paged_decode_attention_two_pass_symbol)(
    const RileyCudaPagedDecodeAttentionReferenceParams*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_paged_decode_attention_two_pass_execute;
static RileyCudaStatus (*const paged_decode_attention_symbol)(
    const RileyCudaPagedDecodeAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_paged_decode_attention_execute;
static RileyCudaStatus (*const ragged_paged_kv_cache_write_symbol)(
    const RileyCudaRaggedPagedKvCacheWriteParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_ragged_paged_kv_cache_write_execute;
static RileyCudaStatus (*const ragged_paged_attention_symbol)(
    const RileyCudaRaggedPagedAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_ragged_paged_attention_execute;
static RileyCudaStatus (*const ragged_paged_attention_grouped_heads_symbol)(
    const RileyCudaRaggedPagedAttentionParams*, RileyCudaStream*,
    RileyCudaErrorInfo*) =
    riley_cuda_ragged_paged_attention_grouped_heads_execute;
static RileyCudaStatus (*const
                                fixed37_ragged_paged_attention_two_pass_symbol)(
    const RileyCudaFixed37RaggedPagedAttentionParams*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_ragged_paged_attention_two_pass_execute;
static RileyCudaStatus (*const gemm_plan_create_symbol)(
    RileyCudaContext*, const RileyCudaGemmConfig*,
    RileyCudaGemmPlan**,
    RileyCudaErrorInfo*) = riley_cuda_gemm_plan_create;
static RileyCudaStatus (*const gemm_plan_create_anchored_symbol)(
    RileyCudaContext*, const RileyCudaGemmConfig*, RileyCudaGemmPlan*,
    RileyCudaGemmPlan**,
    RileyCudaErrorInfo*) = riley_cuda_gemm_plan_create_anchored;
static RileyCudaStatus (*const gemm_plan_info_symbol)(
    RileyCudaGemmPlan*, RileyCudaGemmAlgorithmInfo*,
    RileyCudaErrorInfo*) = riley_cuda_gemm_plan_info;
static RileyCudaStatus (*const gemm_plan_execute_symbol)(
    RileyCudaGemmPlan*, const RileyCudaBufferSpan*,
    const RileyCudaBufferSpan*, const RileyCudaBufferSpan*,
    const RileyCudaBufferSpan*, RileyCudaStream*,
    RileyCudaErrorInfo*) = riley_cuda_gemm_plan_execute;
static RileyCudaStatus (*const gemm_plan_close_symbol)(
    RileyCudaGemmPlan**,
    RileyCudaErrorInfo*) = riley_cuda_gemm_plan_close;
static RileyCudaStatus (*const gemm_plan_defer_to_active_capture_symbol)(
    RileyCudaGemmPlan**, RileyCudaErrorInfo*) =
    riley_cuda_gemm_plan_defer_to_active_capture;
static RileyCudaStatus (*const fixed37_gemm_plan_create_symbol)(
    RileyCudaContext*, const RileyCudaGemmConfig*,
    RileyCudaFixed37GemmPlan**, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_gemm_plan_create;
static RileyCudaStatus (*const fixed37_gemm_plan_info_symbol)(
    RileyCudaFixed37GemmPlan*, RileyCudaFixed37GemmPlanInfo*,
    RileyCudaErrorInfo*) = riley_cuda_fixed37_gemm_plan_info;
static RileyCudaStatus (*const fixed37_gemm_plan_execute_symbol)(
    RileyCudaFixed37GemmPlan*, const RileyCudaBufferSpan*,
    const RileyCudaBufferSpan*, const RileyCudaBufferSpan*,
    RileyCudaStream*, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_gemm_plan_execute;
static RileyCudaStatus (*const fixed37_gemm_plan_close_symbol)(
    RileyCudaFixed37GemmPlan**, RileyCudaErrorInfo*) =
    riley_cuda_fixed37_gemm_plan_close;

// Keep the otherwise compile-only references observably used under strict
// warning configurations.
const void* riley_cuda_abi_symbol_references[] = {
    (const void*)&nvidia_environment_probe_symbol,
    (const void*)&graph_capture_begin_symbol,
    (const void*)&graph_capture_abort_symbol,
    (const void*)&context_defer_to_active_capture_symbol,
    (const void*)&stream_defer_to_active_capture_symbol,
    (const void*)&event_defer_to_active_capture_symbol,
    (const void*)&device_buffer_defer_to_active_capture_symbol,
    (const void*)&pinned_host_buffer_defer_to_active_capture_symbol,
    (const void*)&command_batch_copy_h2d_symbol,
    (const void*)&embedding_symbol,      (const void*)&rms_norm_symbol,
    (const void*)&hugging_face_smollm2_rms_norm_symbol,
    (const void*)&fixed37_rms_norm_symbol,
    (const void*)&residual_add_symbol,
    (const void*)&residual_rms_norm_symbol,
    (const void*)&hugging_face_smollm2_residual_rms_norm_symbol,
    (const void*)&fixed37_residual_rms_norm_symbol,
    (const void*)&fixed37_log_softmax_symbol,
    (const void*)&silu_symbol,
    (const void*)&row_bias_add_symbol,   (const void*)&gated_multiply_symbol,
    (const void*)&rope_table_symbol,     (const void*)&rope_symbol,
    (const void*)&indexed_rope_symbol,
    (const void*)&cast_symbol,           (const void*)&row_gather_symbol,
    (const void*)&bf16_argmax_symbol,
    (const void*)&qk_gqa_symbol,
    (const void*)&fixed37_qk_gqa_symbol,
    (const void*)&scale_causal_mask_symbol,
    (const void*)&causal_softmax_symbol,
    (const void*)&fixed37_causal_softmax_symbol,
    (const void*)&av_gqa_symbol, (const void*)&fixed37_av_gqa_symbol,
    (const void*)&prefill_attention_symbol,
    (const void*)&fixed37_prefill_attention_symbol,
    (const void*)&hf_prefill_attention_plan_create_symbol,
    (const void*)&hf_prefill_attention_plan_info_symbol,
    (const void*)&hf_prefill_attention_plan_execute_symbol,
    (const void*)&hf_prefill_attention_plan_close_symbol,
    (const void*)&hf_prefill_attention_plan_defer_to_active_capture_symbol,
    (const void*)&kv_cache_write_symbol,
    (const void*)&decode_attention_reference_symbol,
    (const void*)&fixed37_decode_attention_reference_symbol,
    (const void*)&fixed37_decode_attention_two_pass_symbol,
    (const void*)&decode_attention_symbol,
    (const void*)&decode_partial_state_reduce_symbol,
    (const void*)&paged_kv_cache_write_symbol,
    (const void*)&paged_decode_attention_reference_symbol,
    (const void*)&fixed37_paged_decode_attention_reference_symbol,
    (const void*)&fixed37_paged_decode_attention_two_pass_symbol,
    (const void*)&paged_decode_attention_symbol,
    (const void*)&ragged_paged_kv_cache_write_symbol,
    (const void*)&ragged_paged_attention_symbol,
    (const void*)&ragged_paged_attention_grouped_heads_symbol,
    (const void*)&fixed37_ragged_paged_attention_two_pass_symbol,
    (const void*)&gemm_plan_create_symbol,
    (const void*)&gemm_plan_create_anchored_symbol,
    (const void*)&gemm_plan_info_symbol,
    (const void*)&gemm_plan_execute_symbol,
    (const void*)&gemm_plan_close_symbol,
    (const void*)&gemm_plan_defer_to_active_capture_symbol,
    (const void*)&fixed37_gemm_plan_create_symbol,
    (const void*)&fixed37_gemm_plan_info_symbol,
    (const void*)&fixed37_gemm_plan_execute_symbol,
    (const void*)&fixed37_gemm_plan_close_symbol,
};
