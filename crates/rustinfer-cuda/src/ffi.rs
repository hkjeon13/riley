use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::mem::{offset_of, size_of};
use std::ptr::{self, NonNull};

use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};

const STATUS_SUCCESS: i32 = 0;
const STATUS_INVALID_ARGUMENT: i32 = 1;
const STATUS_INVALID_DEVICE: i32 = 2;
const STATUS_OUT_OF_RANGE: i32 = 3;
const STATUS_NOT_READY: i32 = 4;
const STATUS_OUT_OF_MEMORY: i32 = 5;
const STATUS_DRIVER_ERROR: i32 = 6;
const STATUS_RUNTIME_ERROR: i32 = 7;
const STATUS_INVALID_STATE: i32 = 8;
const STATUS_CUBLASLT_ERROR: i32 = 10;
const STATUS_NOT_SUPPORTED: i32 = 11;

const DOMAIN_VALIDATION: u32 = 1;
const DOMAIN_DRIVER: u32 = 2;
const DOMAIN_RUNTIME: u32 = 3;
const DOMAIN_CUBLASLT: u32 = 5;

const STAGE_INITIALIZE: u32 = 1;
const STAGE_CREATE: u32 = 3;
const STAGE_LAUNCH: u32 = 4;
const STAGE_SYNCHRONIZE: u32 = 5;
const STAGE_QUERY: u32 = 6;
const STAGE_RECORD: u32 = 7;
const STAGE_COPY: u32 = 8;
const STAGE_CLOSE: u32 = 9;
const STAGE_PREPARE: u32 = 10;

const ERROR_MESSAGE_CAPACITY: usize = 256;
const DEVICE_NAME_CAPACITY: usize = 256;
const ERROR_INFO_SIZE: u32 = 272;
const DEVICE_PROPERTIES_SIZE: u32 = 320;
const ALLOCATION_STATS_SIZE: u32 = 40;
#[cfg(feature = "cuda-test-fault-injection")]
const TEST_MEMORY_FAULT_STATS_SIZE: u32 = 64;
const BUFFER_SPAN_SIZE: u32 = 48;
const EMBEDDING_ERROR_REPORT_SIZE: u32 = 32;
const EMBEDDING_PARAMS_SIZE: u32 = 256;
const RMS_NORM_PARAMS_SIZE: u32 = 208;
const FIXED37_LOG_SOFTMAX_PARAMS_SIZE: u32 = 152;
const RESIDUAL_ADD_PARAMS_SIZE: u32 = 200;
const RESIDUAL_RMS_NORM_PARAMS_SIZE: u32 = 304;
const ROW_BIAS_ADD_IN_PLACE_PARAMS_SIZE: u32 = 152;
const SILU_PARAMS_SIZE: u32 = 152;
const GATED_MULTIPLY_PARAMS_SIZE: u32 = 200;
const ROPE_PARAMS_SIZE: u32 = 288;
const INDEXED_ROPE_PARAMS_SIZE: u32 = 320;
const CAST_PARAMS_SIZE: u32 = 152;
const ROW_GATHER_PARAMS_SIZE: u32 = 208;
const QK_GQA_PARAMS_SIZE: u32 = 216;
const SCALE_CAUSAL_MASK_PARAMS_SIZE: u32 = 112;
const CAUSAL_SOFTMAX_PARAMS_SIZE: u32 = 112;
const AV_GQA_PARAMS_SIZE: u32 = 216;
const PREFILL_ATTENTION_PARAMS_SIZE: u32 = 288;
const KV_CACHE_WRITE_PARAMS_SIZE: u32 = 272;
const DECODE_ATTENTION_REFERENCE_PARAMS_SIZE: u32 = 328;
const DECODE_ATTENTION_PARAMS_SIZE: u32 = 344;
const DECODE_PARTIAL_STATE_REDUCE_PARAMS_SIZE: u32 = 176;
const PAGED_KV_BLOCK_TABLE_V1_SIZE: u32 = 168;
const PAGED_KV_CACHE_WRITE_PARAMS_SIZE: u32 = 432;
const PAGED_DECODE_ATTENTION_REFERENCE_PARAMS_SIZE: u32 = 480;
const PAGED_DECODE_ATTENTION_PARAMS_SIZE: u32 = 488;
const PACKED_BATCH_V1_SIZE: u32 = 320;
const RAGGED_PAGED_KV_CACHE_WRITE_PARAMS_SIZE: u32 = 568;
const RAGGED_PAGED_ATTENTION_PARAMS_SIZE: u32 = 592;
const GEMM_CONFIG_SIZE: u32 = 112;
const GEMM_ALGORITHM_INFO_SIZE: u32 = 112;
const FIXED37_GEMM_PLAN_INFO_SIZE: u32 = 96;

pub(super) const DTYPE_F32: i32 = 1;
pub(super) const DTYPE_BF16: i32 = 2;
pub(super) const DTYPE_U32: i32 = 3;
pub(super) const DTYPE_U8: i32 = 4;
pub(super) const DTYPE_U16: i32 = 5;

pub(super) const PREFILL_MASK_CAUSAL: u32 = 1;
pub(super) const PREFILL_MASK_CAUSAL_LOCAL: u32 = 2;
pub(super) const DECODE_REDUCTION_LOGICAL_ASCENDING: u32 = 1;
pub(super) const DECODE_REDUCTION_LOGICAL_DESCENDING: u32 = 2;
pub(super) const PACKED_BATCH_VERSION: u32 = 1;

const GEMM_TRANSPOSE_N: u32 = 0;
const GEMM_TRANSPOSE_T: u32 = 1;
const GEMM_LAYOUT_ROW_MAJOR: u32 = 1;
const GEMM_EPILOGUE_NONE: u32 = 0;
const GEMM_DETERMINISTIC_REQUIRED: u32 = 1;

#[repr(C)]
struct ErrorInfo {
    struct_size: u32,
    native_code: i32,
    domain: u32,
    stage: u32,
    message: [c_char; ERROR_MESSAGE_CAPACITY],
}

impl ErrorInfo {
    fn new() -> Self {
        Self {
            struct_size: ERROR_INFO_SIZE,
            native_code: 0,
            domain: 0,
            stage: 0,
            message: [0; ERROR_MESSAGE_CAPACITY],
        }
    }
}

#[repr(C)]
struct RawDeviceProperties {
    struct_size: u32,
    ordinal: i32,
    total_memory_bytes: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    multiprocessor_count: u32,
    warp_size: u32,
    max_threads_per_block: u32,
    driver_version: i32,
    runtime_version: i32,
    reserved: [u32; 5],
    name: [c_char; DEVICE_NAME_CAPACITY],
}

impl RawDeviceProperties {
    fn new() -> Self {
        Self {
            struct_size: DEVICE_PROPERTIES_SIZE,
            ordinal: 0,
            total_memory_bytes: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            multiprocessor_count: 0,
            warp_size: 0,
            max_threads_per_block: 0,
            driver_version: 0,
            runtime_version: 0,
            reserved: [0; 5],
            name: [0; DEVICE_NAME_CAPACITY],
        }
    }
}

#[repr(C)]
struct RawAllocationStats {
    struct_size: u32,
    reserved: u32,
    device_live_bytes: u64,
    device_live_allocations: u64,
    pinned_host_live_bytes: u64,
    pinned_host_live_allocations: u64,
}

impl RawAllocationStats {
    fn new() -> Self {
        Self {
            struct_size: ALLOCATION_STATS_SIZE,
            reserved: 0,
            device_live_bytes: 0,
            device_live_allocations: 0,
            pinned_host_live_bytes: 0,
            pinned_host_live_allocations: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeAllocationStats {
    pub(super) device_live_bytes: u64,
    pub(super) device_live_allocations: u64,
    pub(super) pinned_host_live_bytes: u64,
    pub(super) pinned_host_live_allocations: u64,
}

#[cfg(feature = "cuda-test-fault-injection")]
#[repr(C)]
struct RawTestMemoryFaultStats {
    struct_size: u32,
    armed_fault: u32,
    faults_fired: u64,
    device_free_attempts: u64,
    pinned_free_attempts: u64,
    copy_use_release_attempts: u64,
    reserved: [u64; 3],
}

#[cfg(feature = "cuda-test-fault-injection")]
impl RawTestMemoryFaultStats {
    const fn new() -> Self {
        Self {
            struct_size: TEST_MEMORY_FAULT_STATS_SIZE,
            armed_fault: 0,
            faults_fired: 0,
            device_free_attempts: 0,
            pinned_free_attempts: 0,
            copy_use_release_attempts: 0,
            reserved: [0; 3],
        }
    }
}

#[cfg(feature = "cuda-test-fault-injection")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeTestMemoryFaultStats {
    pub(super) armed_fault: u32,
    pub(super) faults_fired: u64,
    pub(super) device_free_attempts: u64,
    pub(super) pinned_free_attempts: u64,
    pub(super) copy_use_release_attempts: u64,
}

#[derive(Debug)]
pub(super) struct NativeDeviceProperties {
    pub(super) ordinal: i32,
    pub(super) name: String,
    pub(super) total_memory_bytes: u64,
    pub(super) compute_capability_major: u32,
    pub(super) compute_capability_minor: u32,
    pub(super) multiprocessor_count: u32,
    pub(super) warp_size: u32,
    pub(super) max_threads_per_block: u32,
    pub(super) driver_version: i32,
    pub(super) runtime_version: i32,
}

#[repr(C)]
struct RawContext {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawStream {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawEvent {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawSmokeBuffer {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawDeviceBuffer {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawPinnedHostBuffer {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawCopy {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawGemmPlan {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawFixed37GemmPlan {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub(super) struct RawBufferSpan {
    struct_size: u32,
    dtype: i32,
    buffer: *mut RawDeviceBuffer,
    byte_offset: u64,
    byte_len: u64,
    reserved: [u64; 2],
}

#[repr(C)]
struct RawEmbeddingErrorReport {
    struct_size: u32,
    code: u32,
    token_position: u64,
    token_id: u64,
    reserved: u64,
}

impl RawEmbeddingErrorReport {
    const fn new() -> Self {
        Self {
            struct_size: EMBEDDING_ERROR_REPORT_SIZE,
            code: 0,
            token_position: 0,
            token_id: 0,
            reserved: 0,
        }
    }
}

#[repr(C)]
struct RawEmbeddingParams {
    struct_size: u32,
    reserved0: u32,
    table: RawBufferSpan,
    token_ids: RawBufferSpan,
    output: RawBufferSpan,
    device_error_scratch: RawBufferSpan,
    out_report: *mut RawEmbeddingErrorReport,
    token_count: u64,
    vocabulary_size: u64,
    hidden_size: u64,
    reserved: [u64; 3],
}

#[repr(C)]
struct RawRmsNormParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    weight: RawBufferSpan,
    output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawFixed37LogSoftmaxParams {
    struct_size: u32,
    reserved0: u32,
    logits: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawResidualAddParams {
    struct_size: u32,
    reserved0: u32,
    left: RawBufferSpan,
    right: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawResidualRmsNormParams {
    struct_size: u32,
    reserved0: u32,
    left: RawBufferSpan,
    right: RawBufferSpan,
    weight: RawBufferSpan,
    residual_output: RawBufferSpan,
    normalized_output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawRowBiasAddInPlaceParams {
    struct_size: u32,
    reserved0: u32,
    matrix: RawBufferSpan,
    bias: RawBufferSpan,
    row_count: u64,
    column_count: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawSiluParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawGatedMultiplyParams {
    struct_size: u32,
    reserved0: u32,
    activated_gate: RawBufferSpan,
    up: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawRopeParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    cos: RawBufferSpan,
    sin: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    head_count: u64,
    head_size: u64,
    rotary_dimension: u64,
    table_position_count: u64,
    position_offset: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawIndexedRopeParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    cos: RawBufferSpan,
    sin: RawBufferSpan,
    positions: RawBufferSpan,
    output: RawBufferSpan,
    active_row_count: u64,
    head_count: u64,
    head_size: u64,
    rotary_dimension: u64,
    table_position_count: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawCastParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawRowGatherParams {
    struct_size: u32,
    reserved0: u32,
    input: RawBufferSpan,
    row_indices: RawBufferSpan,
    output: RawBufferSpan,
    input_row_count: u64,
    output_row_count: u64,
    column_count: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawQkGqaParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawScaleCausalMaskParams {
    struct_size: u32,
    reserved0: u32,
    scores: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    scale: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawCausalSoftmaxParams {
    struct_size: u32,
    reserved0: u32,
    scores: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    reserved: [u64; 5],
}

#[repr(C)]
struct RawAvGqaParams {
    struct_size: u32,
    reserved0: u32,
    probabilities: RawBufferSpan,
    value: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawPrefillAttentionParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key: RawBufferSpan,
    value: RawBufferSpan,
    output: RawBufferSpan,
    batch_size: u64,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    mask_kind: u32,
    local_window: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawKvCacheWriteParams {
    struct_size: u32,
    reserved0: u32,
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    source_token_count: u64,
    destination_token_start: u64,
    maximum_token_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawDecodeAttentionReferenceParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    score_workspace: RawBufferSpan,
    output: RawBufferSpan,
    maximum_token_count: u64,
    logical_token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawDecodeAttentionParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    maximum_token_count: u64,
    logical_token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    tokens_per_partition: u64,
    partial_state_capacity: u64,
    scale: f32,
    reduction_order: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawDecodePartialStateReduceParams {
    struct_size: u32,
    reserved0: u32,
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    partial_state_count: u64,
    partial_state_capacity: u64,
    query_head_count: u64,
    head_size: u64,
    reduction_order: u32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawPagedKvBlockTableV1 {
    struct_size: u32,
    format_version: u32,
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    logical_token_count: u64,
    block_count: u64,
    physical_block_count: u64,
    block_size: u32,
    metadata_kind: u32,
    metadata_version: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C)]
struct RawPagedKvCacheWriteParams {
    struct_size: u32,
    reserved0: u32,
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    block_table: RawPagedKvBlockTableV1,
    source_token_count: u64,
    destination_token_start: u64,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawPagedDecodeAttentionReferenceParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    score_workspace: RawBufferSpan,
    output: RawBufferSpan,
    block_table: RawPagedKvBlockTableV1,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawPagedDecodeAttentionParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    block_table: RawPagedKvBlockTableV1,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    partial_state_capacity: u64,
    scale: f32,
    reduction_order: u32,
    reserved: [u64; 4],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawPackedBatchV1 {
    struct_size: u32,
    format_version: u32,
    sequence_block_offsets: RawBufferSpan,
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    row_sequence_slots: RawBufferSpan,
    row_positions: RawBufferSpan,
    sequence_count: u64,
    block_count: u64,
    active_row_count: u64,
    physical_block_count: u64,
    block_size: u32,
    reserved0: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawRaggedPagedKvCacheWriteParams {
    struct_size: u32,
    reserved0: u32,
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    batch: RawPackedBatchV1,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawRaggedPagedAttentionParams {
    struct_size: u32,
    reserved0: u32,
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    output: RawBufferSpan,
    batch: RawPackedBatchV1,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    output_row_count: u64,
    scale: f32,
    reserved1: u32,
    reserved: [u64; 4],
}

#[repr(C)]
struct RawGemmConfig {
    struct_size: u32,
    flags: u32,
    m: u64,
    n: u64,
    k: u64,
    input_dtype: i32,
    weight_dtype: i32,
    accumulator_dtype: i32,
    output_dtype: i32,
    input_transpose: u32,
    weight_transpose: u32,
    input_layout: u32,
    weight_layout: u32,
    output_layout: u32,
    epilogue: u32,
    deterministic: u32,
    reserved0: u32,
    max_workspace_bytes: u64,
    reserved: [u64; 3],
}

impl RawGemmConfig {
    const fn new(m: u64, n: u64, k: u64, max_workspace_bytes: u64) -> Self {
        Self {
            struct_size: GEMM_CONFIG_SIZE,
            flags: 0,
            m,
            n,
            k,
            input_dtype: DTYPE_BF16,
            weight_dtype: DTYPE_BF16,
            accumulator_dtype: DTYPE_F32,
            output_dtype: DTYPE_BF16,
            input_transpose: GEMM_TRANSPOSE_N,
            weight_transpose: GEMM_TRANSPOSE_T,
            input_layout: GEMM_LAYOUT_ROW_MAJOR,
            weight_layout: GEMM_LAYOUT_ROW_MAJOR,
            output_layout: GEMM_LAYOUT_ROW_MAJOR,
            epilogue: GEMM_EPILOGUE_NONE,
            deterministic: GEMM_DETERMINISTIC_REQUIRED,
            reserved0: 0,
            max_workspace_bytes,
            reserved: [0; 3],
        }
    }
}

#[repr(C)]
struct RawGemmAlgorithmInfo {
    struct_size: u32,
    backend: u32,
    algorithm_id: i32,
    tile_id: u32,
    stages_id: u32,
    split_k: u32,
    reduction_scheme: u32,
    cta_swizzling: u32,
    custom_option: u32,
    deterministic: u32,
    workspace_bytes: u64,
    numerical_implementation_flags: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    runtime_version: i32,
    cublaslt_version: i32,
    m: u64,
    n: u64,
    k: u64,
    reserved: [u64; 2],
}

impl RawGemmAlgorithmInfo {
    const fn new() -> Self {
        Self {
            struct_size: GEMM_ALGORITHM_INFO_SIZE,
            backend: 0,
            algorithm_id: 0,
            tile_id: 0,
            stages_id: 0,
            split_k: 0,
            reduction_scheme: 0,
            cta_swizzling: 0,
            custom_option: 0,
            deterministic: 0,
            workspace_bytes: 0,
            numerical_implementation_flags: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            runtime_version: 0,
            cublaslt_version: 0,
            m: 0,
            n: 0,
            k: 0,
            reserved: [0; 2],
        }
    }
}

#[repr(C)]
struct RawFixed37GemmPlanInfo {
    struct_size: u32,
    backend: u32,
    reduction_version: u32,
    chunk_elements: u32,
    accumulator_dtype: i32,
    output_dtype: i32,
    threads_per_block: u32,
    deterministic: u32,
    dynamic_shared_memory_bytes: u64,
    workspace_bytes: u64,
    m: u64,
    n: u64,
    k: u64,
    reserved: [u64; 3],
}

impl RawFixed37GemmPlanInfo {
    const fn new() -> Self {
        Self {
            struct_size: FIXED37_GEMM_PLAN_INFO_SIZE,
            backend: 0,
            reduction_version: 0,
            chunk_elements: 0,
            accumulator_dtype: 0,
            output_dtype: 0,
            threads_per_block: 0,
            deterministic: 0,
            dynamic_shared_memory_bytes: 0,
            workspace_bytes: 0,
            m: 0,
            n: 0,
            k: 0,
            reserved: [0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeGemmAlgorithmInfo {
    pub(super) backend: u32,
    pub(super) algorithm_id: i32,
    pub(super) tile_id: u32,
    pub(super) stages_id: u32,
    pub(super) split_k: u32,
    pub(super) reduction_scheme: u32,
    pub(super) cta_swizzling: u32,
    pub(super) custom_option: u32,
    pub(super) deterministic: u32,
    pub(super) workspace_bytes: u64,
    pub(super) numerical_implementation_flags: u64,
    pub(super) compute_capability_major: u32,
    pub(super) compute_capability_minor: u32,
    pub(super) runtime_version: i32,
    pub(super) cublaslt_version: i32,
    pub(super) m: u64,
    pub(super) n: u64,
    pub(super) k: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeFixed37GemmPlanInfo {
    pub(super) backend: u32,
    pub(super) reduction_version: u32,
    pub(super) chunk_elements: u32,
    pub(super) accumulator_dtype: i32,
    pub(super) output_dtype: i32,
    pub(super) threads_per_block: u32,
    pub(super) deterministic: u32,
    pub(super) dynamic_shared_memory_bytes: u64,
    pub(super) workspace_bytes: u64,
    pub(super) m: u64,
    pub(super) n: u64,
    pub(super) k: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct NativeEmbeddingReport {
    pub(super) code: u32,
    pub(super) token_position: u64,
    pub(super) token_id: u64,
}

pub(super) struct NativeEmbeddingCompletion {
    pub(super) report: NativeEmbeddingReport,
    pub(super) result: CudaResult<()>,
}

unsafe extern "C" {
    fn rustinfer_cuda_abi_version() -> u32;
    fn rustinfer_cuda_build_info() -> *const c_char;
    fn rustinfer_cuda_device_count(out_count: *mut u32, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_device_properties(
        ordinal: i32,
        out_properties: *mut RawDeviceProperties,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_create(
        ordinal: i32,
        out_context: *mut *mut RawContext,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_synchronize(context: *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_context_memory_info(
        context: *mut RawContext,
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_allocation_stats(
        context: *mut RawContext,
        out_stats: *mut RawAllocationStats,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_context_close(context: *mut *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_stream_create(
        context: *mut RawContext,
        out_stream: *mut *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_query(
        stream: *mut RawStream,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_synchronize(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_stream_command_batch_begin(
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_command_batch_end(
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_wait_event(
        stream: *mut RawStream,
        event: *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_stream_close(stream: *mut *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_event_create(
        context: *mut RawContext,
        out_event: *mut *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_record(
        event: *mut RawEvent,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_query(
        event: *mut RawEvent,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_synchronize(event: *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_event_elapsed_ms(
        start: *mut RawEvent,
        end: *mut RawEvent,
        out_elapsed_ms: *mut f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_event_close(event: *mut *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_device_buffer_create(
        context: *mut RawContext,
        byte_len: u64,
        out_buffer: *mut *mut RawDeviceBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_device_buffer_close(
        buffer: *mut *mut RawDeviceBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_pinned_host_buffer_create(
        context: *mut RawContext,
        byte_len: u64,
        out_buffer: *mut *mut RawPinnedHostBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_pinned_host_buffer_write(
        buffer: *mut RawPinnedHostBuffer,
        destination_offset: u64,
        source: *const u8,
        source_len: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_pinned_host_buffer_read(
        buffer: *mut RawPinnedHostBuffer,
        source_offset: u64,
        destination: *mut u8,
        destination_len: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_pinned_host_buffer_close(
        buffer: *mut *mut RawPinnedHostBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_copy_h2d_async(
        destination: *mut RawDeviceBuffer,
        destination_offset: u64,
        source: *mut RawPinnedHostBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: *mut RawStream,
        out_copy: *mut *mut RawCopy,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_copy_d2h_async(
        destination: *mut RawPinnedHostBuffer,
        destination_offset: u64,
        source: *mut RawDeviceBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: *mut RawStream,
        out_copy: *mut *mut RawCopy,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_copy_query(
        copy: *mut RawCopy,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_copy_synchronize(
        copy: *mut RawCopy,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_copy_close(copy: *mut *mut RawCopy, error: *mut ErrorInfo) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn rustinfer_cuda_test_memory_fault_reset(
        context: *mut RawContext,
        error: *mut ErrorInfo,
    ) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn rustinfer_cuda_test_memory_fault_arm(
        context: *mut RawContext,
        fault: u32,
        error: *mut ErrorInfo,
    ) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn rustinfer_cuda_test_memory_fault_stats(
        context: *mut RawContext,
        out_stats: *mut RawTestMemoryFaultStats,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_embedding_execute(
        params: *const RawEmbeddingParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_rms_norm_execute(
        params: *const RawRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_rms_norm_execute(
        params: *const RawRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_residual_add_execute(
        params: *const RawResidualAddParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_residual_rms_norm_execute(
        params: *const RawResidualRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_residual_rms_norm_execute(
        params: *const RawResidualRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_log_softmax_execute(
        params: *const RawFixed37LogSoftmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_row_bias_add_in_place_execute(
        params: *const RawRowBiasAddInPlaceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_silu_execute(
        params: *const RawSiluParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_gated_multiply_execute(
        params: *const RawGatedMultiplyParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_rope_execute(
        params: *const RawRopeParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_indexed_rope_execute(
        params: *const RawIndexedRopeParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_cast_execute(
        params: *const RawCastParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_row_gather_execute(
        params: *const RawRowGatherParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_qk_gqa_execute(
        params: *const RawQkGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_scale_causal_mask_in_place_execute(
        params: *const RawScaleCausalMaskParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_causal_softmax_in_place_execute(
        params: *const RawCausalSoftmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_av_gqa_execute(
        params: *const RawAvGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_prefill_attention_execute(
        params: *const RawPrefillAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_kv_cache_write_execute(
        params: *const RawKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_decode_attention_reference_execute(
        params: *const RawDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_decode_attention_execute(
        params: *const RawDecodeAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_decode_partial_state_reduce_execute(
        params: *const RawDecodePartialStateReduceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_paged_kv_cache_write_execute(
        params: *const RawPagedKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_paged_decode_attention_reference_execute(
        params: *const RawPagedDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_paged_decode_attention_execute(
        params: *const RawPagedDecodeAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_ragged_paged_kv_cache_write_execute(
        params: *const RawRaggedPagedKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_ragged_paged_attention_execute(
        params: *const RawRaggedPagedAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_gemm_plan_create(
        context: *mut RawContext,
        config: *const RawGemmConfig,
        out_plan: *mut *mut RawGemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_gemm_plan_info(
        plan: *mut RawGemmPlan,
        out_info: *mut RawGemmAlgorithmInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_gemm_plan_execute(
        plan: *mut RawGemmPlan,
        input: *const RawBufferSpan,
        weight: *const RawBufferSpan,
        output: *const RawBufferSpan,
        workspace: *const RawBufferSpan,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_gemm_plan_close(plan: *mut *mut RawGemmPlan, error: *mut ErrorInfo) -> i32;
    fn rustinfer_cuda_fixed37_gemm_plan_create(
        context: *mut RawContext,
        config: *const RawGemmConfig,
        out_plan: *mut *mut RawFixed37GemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_gemm_plan_info(
        plan: *mut RawFixed37GemmPlan,
        out_info: *mut RawFixed37GemmPlanInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_gemm_plan_execute(
        plan: *mut RawFixed37GemmPlan,
        input: *const RawBufferSpan,
        weight: *const RawBufferSpan,
        output: *const RawBufferSpan,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_fixed37_gemm_plan_close(
        plan: *mut *mut RawFixed37GemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_buffer_create(
        context: *mut RawContext,
        element_count: u64,
        out_buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_fill_launch(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        value: f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_copy_to_host(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        host_output: *mut f32,
        host_element_capacity: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_buffer_close(
        buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn rustinfer_cuda_smoke_invalid_launch(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
}

pub(super) fn abi_version() -> u32 {
    // SAFETY: the statically linked metadata function takes no arguments and
    // returns a fixed-width value defined by the checked C header.
    unsafe { rustinfer_cuda_abi_version() }
}

pub(super) fn build_info() -> CudaResult<String> {
    // SAFETY: the native ABI returns null or a process-lifetime C string.
    let pointer = unsafe { rustinfer_cuda_build_info() };
    if pointer.is_null() {
        return Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Initialize,
            0,
            "read native build info",
            "native build-info pointer is null",
        ));
    }
    // SAFETY: null was rejected and the ABI promises NUL termination and
    // process lifetime; the bytes are copied before returning.
    let value = unsafe { CStr::from_ptr(pointer) };
    value.to_str().map(str::to_owned).map_err(|error| {
        CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Initialize,
            0,
            "read native build info",
            format!("native build info is not UTF-8: {error}"),
        )
    })
}

pub(super) fn device_count() -> CudaResult<u32> {
    let mut count = 0;
    let mut error = ErrorInfo::new();
    // SAFETY: both output pointers refer to initialized, writable values for
    // the duration of the synchronous C call.
    let status = unsafe { rustinfer_cuda_device_count(&mut count, &mut error) };
    status_result(status, "enumerate CUDA devices", &error)?;
    Ok(count)
}

pub(super) fn diagnose_null_device_count() -> CudaResult<()> {
    let mut error = ErrorInfo::new();
    // SAFETY: null is intentionally supplied to exercise the documented ABI
    // validation path; no memory is dereferenced by contract.
    let status = unsafe { rustinfer_cuda_device_count(ptr::null_mut(), &mut error) };
    status_result(status, "diagnose null device-count output", &error)
}

pub(super) fn device_properties(ordinal: i32) -> CudaResult<NativeDeviceProperties> {
    let mut properties = RawDeviceProperties::new();
    let mut error = ErrorInfo::new();
    // SAFETY: properties and error are correctly sized repr(C) caller buffers.
    let status = unsafe { rustinfer_cuda_device_properties(ordinal, &mut properties, &mut error) };
    status_result(status, "query CUDA device properties", &error)?;
    Ok(NativeDeviceProperties {
        ordinal: properties.ordinal,
        name: c_array_to_string(&properties.name),
        total_memory_bytes: properties.total_memory_bytes,
        compute_capability_major: properties.compute_capability_major,
        compute_capability_minor: properties.compute_capability_minor,
        multiprocessor_count: properties.multiprocessor_count,
        warp_size: properties.warp_size,
        max_threads_per_block: properties.max_threads_per_block,
        driver_version: properties.driver_version,
        runtime_version: properties.runtime_version,
    })
}

pub(super) struct ContextHandle {
    pointer: Option<NonNull<RawContext>>,
}

// SAFETY: native operations push/pop the retained primary context on each
// calling thread, and mutation/destruction requires unique Rust ownership.
unsafe impl Send for ContextHandle {}
// SAFETY: shared context methods are thread-safe CUDA calls; close requires
// `&mut self` and the safe layer prevents it while child Arc references exist.
unsafe impl Sync for ContextHandle {}

impl ContextHandle {
    pub(super) fn create(ordinal: i32) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: pointer and error are writable caller buffers; native returns
        // either null or one newly owned context handle.
        let status = unsafe { rustinfer_cuda_context_create(ordinal, &mut pointer, &mut error) };
        status_result(status, "create CUDA context", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("create CUDA context", "native context handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawContext {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn synchronize(&self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the handle is owned and kept alive by self for this call.
        let status = unsafe { rustinfer_cuda_context_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA context", &error)
    }

    pub(super) fn memory_info(&self) -> CudaResult<(u64, u64)> {
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the live context and both output buffers remain valid for the
        // synchronous native call.
        let status = unsafe {
            rustinfer_cuda_context_memory_info(
                self.as_ptr(),
                &mut free_bytes,
                &mut total_bytes,
                &mut error,
            )
        };
        status_result(status, "query CUDA memory info", &error)?;
        Ok((free_bytes, total_bytes))
    }

    pub(super) fn allocation_stats(&self) -> CudaResult<NativeAllocationStats> {
        let mut allocation_snapshot = RawAllocationStats::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the live context and correctly sized repr(C) output remain
        // valid for the complete synchronous native snapshot.
        let call_status = unsafe {
            rustinfer_cuda_context_allocation_stats(
                self.as_ptr(),
                &mut allocation_snapshot,
                &mut error,
            )
        };
        status_result(call_status, "query CUDA allocation stats", &error)?;
        Ok(NativeAllocationStats {
            device_live_bytes: allocation_snapshot.device_live_bytes,
            device_live_allocations: allocation_snapshot.device_live_allocations,
            pinned_host_live_bytes: allocation_snapshot.pinned_host_live_bytes,
            pinned_host_live_allocations: allocation_snapshot.pinned_host_live_allocations,
        })
    }

    #[cfg(feature = "cuda-test-fault-injection")]
    pub(super) fn reset_memory_fault_injection(&self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: this build links the test-only native ABI and the context is
        // kept alive for the synchronous session reset.
        let status = unsafe { rustinfer_cuda_test_memory_fault_reset(self.as_ptr(), &mut error) };
        status_result(status, "reset CUDA memory fault injector", &error)
    }

    #[cfg(feature = "cuda-test-fault-injection")]
    pub(super) fn arm_memory_fault(&self, fault: u32) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the native boundary validates the test-only fault id and the
        // live context/session identity.
        let status =
            unsafe { rustinfer_cuda_test_memory_fault_arm(self.as_ptr(), fault, &mut error) };
        status_result(status, "arm CUDA memory fault injector", &error)
    }

    #[cfg(feature = "cuda-test-fault-injection")]
    pub(super) fn memory_fault_stats(&self) -> CudaResult<NativeTestMemoryFaultStats> {
        let mut stats = RawTestMemoryFaultStats::new();
        let mut error = ErrorInfo::new();
        // SAFETY: stats is a correctly sized repr(C) output and both it and the
        // live context remain valid for the synchronous snapshot.
        let status = unsafe {
            rustinfer_cuda_test_memory_fault_stats(self.as_ptr(), &mut stats, &mut error)
        };
        status_result(status, "query CUDA memory fault injector", &error)?;
        Ok(NativeTestMemoryFaultStats {
            armed_fault: stats.armed_fault,
            faults_fired: stats.faults_fired,
            device_free_attempts: stats.device_free_attempts,
            pinned_free_attempts: stats.pinned_free_attempts,
            copy_use_release_attempts: stats.copy_use_release_attempts,
        })
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is this handle's unique owned pointer; the native close
        // contract nulls it only after consuming the resource.
        let status = unsafe { rustinfer_cuda_context_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA context", &error)
    }
}

impl Drop for ContextHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct StreamHandle {
    pointer: Option<NonNull<RawStream>>,
}

// SAFETY: CUDA streams may move between host threads because every operation
// restores the current CUDA context. The safe wrapper intentionally is !Sync.
unsafe impl Send for StreamHandle {}

impl StreamHandle {
    pub(super) fn create(context: &ContextHandle) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and output/error are writable buffers.
        let status =
            unsafe { rustinfer_cuda_stream_create(context.as_ptr(), &mut pointer, &mut error) };
        status_result(status, "create CUDA stream", &error)?;
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| missing_output("create CUDA stream", "native stream handle is null"))?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawStream {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn query(&mut self) -> CudaResult<bool> {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the stream and output buffers remain live for the call.
        let status =
            unsafe { rustinfer_cuda_stream_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA stream", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle.
        let status = unsafe { rustinfer_cuda_stream_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA stream", &error)
    }

    pub(super) fn command_batch_begin(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle. The native
        // boundary validates lifecycle state before enabling command batching.
        let status =
            unsafe { rustinfer_cuda_stream_command_batch_begin(self.as_ptr(), &mut error) };
        status_result(status, "begin CUDA stream command batch", &error)
    }

    pub(super) fn command_batch_end(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle. The native end
        // call owns completion and any fail-closed post-error lifecycle state.
        let status = unsafe { rustinfer_cuda_stream_command_batch_end(self.as_ptr(), &mut error) };
        status_result(status, "end CUDA stream command batch", &error)
    }

    pub(super) fn wait_event(&mut self, event: &EventHandle) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both native handles remain alive and the native ABI validates
        // that they belong to the same context.
        let status =
            unsafe { rustinfer_cuda_stream_wait_event(self.as_ptr(), event.as_ptr(), &mut error) };
        status_result(status, "wait for CUDA event", &error)
    }

    pub(super) fn diagnose_invalid_launch(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the function intentionally issues a configuration-invalid
        // launch against this live stream and clears the launch error.
        let status = unsafe { rustinfer_cuda_smoke_invalid_launch(self.as_ptr(), &mut error) };
        status_result(status, "diagnose invalid CUDA launch", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned and native nulls it only on consume.
        let status = unsafe { rustinfer_cuda_stream_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA stream", &error)
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct EventHandle {
    pointer: Option<NonNull<RawEvent>>,
}

// SAFETY: events may move between host threads with context push/pop; the safe
// wrapper is intentionally !Sync and serializes mutable operations.
unsafe impl Send for EventHandle {}

impl EventHandle {
    pub(super) fn create(context: &ContextHandle) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and output/error are writable buffers.
        let status =
            unsafe { rustinfer_cuda_event_create(context.as_ptr(), &mut pointer, &mut error) };
        status_result(status, "create CUDA event", &error)?;
        let pointer = NonNull::new(pointer)
            .ok_or_else(|| missing_output("create CUDA event", "native event handle is null"))?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawEvent {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn record(&mut self, stream: &mut StreamHandle) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both handles remain live and the ABI validates their owner.
        let status =
            unsafe { rustinfer_cuda_event_record(self.as_ptr(), stream.as_ptr(), &mut error) };
        status_result(status, "record CUDA event", &error)
    }

    pub(super) fn query(&mut self) -> CudaResult<bool> {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: event and output remain live for the synchronous call.
        let status =
            unsafe { rustinfer_cuda_event_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA event", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live event handle.
        let status = unsafe { rustinfer_cuda_event_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA event", &error)
    }

    pub(super) fn elapsed_ms(&self, end: &Self) -> CudaResult<f32> {
        let mut elapsed = 0.0;
        let mut error = ErrorInfo::new();
        // SAFETY: both event handles and output remain valid; native validates
        // context identity and recording/completion state.
        let status = unsafe {
            rustinfer_cuda_event_elapsed_ms(self.as_ptr(), end.as_ptr(), &mut elapsed, &mut error)
        };
        status_result(status, "measure CUDA event elapsed time", &error)?;
        Ok(elapsed)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned and native nulls it only on consume.
        let status = unsafe { rustinfer_cuda_event_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA event", &error)
    }
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct DeviceBufferHandle {
    pointer: Option<NonNull<RawDeviceBuffer>>,
}

// SAFETY: the opaque allocation may move between host threads. All access is
// serialized by exclusive safe-wrapper borrows and native active-copy guards.
unsafe impl Send for DeviceBufferHandle {}

impl DeviceBufferHandle {
    pub(super) fn create(context: &ContextHandle, byte_len: u64) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and native initializes one owned opaque
        // output or leaves it null on failure.
        let status = unsafe {
            rustinfer_cuda_device_buffer_create(
                context.as_ptr(),
                byte_len,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "allocate CUDA device buffer", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output(
                "allocate CUDA device buffer",
                "native device-buffer handle is null",
            )
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawDeviceBuffer {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn span(&self, dtype: i32, byte_offset: u64, byte_len: u64) -> RawBufferSpan {
        RawBufferSpan {
            struct_size: BUFFER_SPAN_SIZE,
            dtype,
            buffer: self.as_ptr(),
            byte_offset,
            byte_len,
            reserved: [0; 2],
        }
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned; native either retains it before a
        // destructive attempt or consumes and nulls it single-shot.
        let status = unsafe { rustinfer_cuda_device_buffer_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA device buffer", &error)
    }
}

impl Drop for DeviceBufferHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct PinnedHostBufferHandle {
    pointer: Option<NonNull<RawPinnedHostBuffer>>,
}

// SAFETY: the allocation may move between host threads. The safe layer is
// !Sync and native active-copy guards reject CPU access during DMA.
unsafe impl Send for PinnedHostBufferHandle {}

impl PinnedHostBufferHandle {
    pub(super) fn create(context: &ContextHandle, byte_len: u64) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context stays alive and output/error are writable caller
        // buffers for the synchronous creation call.
        let status = unsafe {
            rustinfer_cuda_pinned_host_buffer_create(
                context.as_ptr(),
                byte_len,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "allocate CUDA pinned host buffer", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output(
                "allocate CUDA pinned host buffer",
                "native pinned-host handle is null",
            )
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawPinnedHostBuffer {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn write(&mut self, destination_offset: u64, source: &[u8]) -> CudaResult<()> {
        let source_len = u64::try_from(source.len()).map_err(|_| {
            CudaError::out_of_range(
                "write CUDA pinned host buffer",
                "source length does not fit the fixed-width native ABI",
            )
        })?;
        let mut error = ErrorInfo::new();
        // SAFETY: source is immutably borrowed through the complete synchronous
        // CPU copy and the opaque destination remains uniquely borrowed.
        let status = unsafe {
            rustinfer_cuda_pinned_host_buffer_write(
                self.as_ptr(),
                destination_offset,
                source.as_ptr(),
                source_len,
                &mut error,
            )
        };
        status_result(status, "write CUDA pinned host buffer", &error)
    }

    pub(super) fn read(&mut self, source_offset: u64, destination: &mut [u8]) -> CudaResult<()> {
        let destination_len = u64::try_from(destination.len()).map_err(|_| {
            CudaError::out_of_range(
                "read CUDA pinned host buffer",
                "destination length does not fit the fixed-width native ABI",
            )
        })?;
        let mut error = ErrorInfo::new();
        // SAFETY: destination is exclusively borrowed and valid for the
        // complete synchronous CPU copy; native validates range and busy state.
        let status = unsafe {
            rustinfer_cuda_pinned_host_buffer_read(
                self.as_ptr(),
                source_offset,
                destination.as_mut_ptr(),
                destination_len,
                &mut error,
            )
        };
        status_result(status, "read CUDA pinned host buffer", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned; native active-copy validation occurs
        // before any destructive attempt and single-shot close updates raw.
        let status = unsafe { rustinfer_cuda_pinned_host_buffer_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA pinned host buffer", &error)
    }
}

impl Drop for PinnedHostBufferHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct CopyCompletion {
    pub(super) complete: bool,
    pub(super) result: CudaResult<()>,
}

pub(super) struct CopyHandle {
    pointer: Option<NonNull<RawCopy>>,
}

// SAFETY: a pending token and all three referenced resources move together
// behind exclusive safe Rust borrows. The token is deliberately not Sync.
unsafe impl Send for CopyHandle {}

impl CopyHandle {
    pub(super) fn h2d(
        destination: &DeviceBufferHandle,
        destination_offset: u64,
        source: &PinnedHostBufferHandle,
        source_offset: u64,
        byte_len: u64,
        stream: &StreamHandle,
    ) -> CudaResult<Option<Self>> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: every opaque resource remains exclusively borrowed by the
        // safe pending owner; native validates context/range/use identity.
        let status = unsafe {
            rustinfer_cuda_copy_h2d_async(
                destination.as_ptr(),
                destination_offset,
                source.as_ptr(),
                source_offset,
                byte_len,
                stream.as_ptr(),
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "enqueue CUDA host-to-device copy", &error)?;
        Self::from_submit_output(pointer, byte_len, "enqueue CUDA host-to-device copy")
    }

    pub(super) fn d2h(
        destination: &PinnedHostBufferHandle,
        destination_offset: u64,
        source: &DeviceBufferHandle,
        source_offset: u64,
        byte_len: u64,
        stream: &StreamHandle,
    ) -> CudaResult<Option<Self>> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: every opaque resource stays alive behind the pending Rust
        // borrows; native validates ownership and establishes stream ordering.
        let status = unsafe {
            rustinfer_cuda_copy_d2h_async(
                destination.as_ptr(),
                destination_offset,
                source.as_ptr(),
                source_offset,
                byte_len,
                stream.as_ptr(),
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "enqueue CUDA device-to-host copy", &error)?;
        Self::from_submit_output(pointer, byte_len, "enqueue CUDA device-to-host copy")
    }

    fn from_submit_output(
        pointer: *mut RawCopy,
        byte_len: u64,
        operation: &'static str,
    ) -> CudaResult<Option<Self>> {
        match (NonNull::new(pointer), byte_len) {
            (None, 0) => Ok(None),
            (None, _) => Err(missing_output(operation, "native copy token is null")),
            (Some(pointer), 0) => {
                let mut unexpected = Self {
                    pointer: Some(pointer),
                };
                let _ = unexpected.close();
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Copy,
                    0,
                    operation,
                    "native returned a token for a zero-byte copy",
                ))
            }
            (Some(pointer), _) => Ok(Some(Self {
                pointer: Some(pointer),
            })),
        }
    }

    fn as_ptr(&self) -> *mut RawCopy {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn query(&mut self) -> CopyCompletion {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the owned token remains live, and out_complete/error are
        // writable for the complete native call.
        let status = unsafe { rustinfer_cuda_copy_query(self.as_ptr(), &mut complete, &mut error) };
        copy_completion(status, complete, "query CUDA copy", &error)
    }

    pub(super) fn synchronize(&mut self) -> CopyCompletion {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the owned token and all resources retained by native active
        // use remain valid until out_complete confirms release.
        let status =
            unsafe { rustinfer_cuda_copy_synchronize(self.as_ptr(), &mut complete, &mut error) };
        copy_completion(status, complete, "synchronize CUDA copy", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw uniquely owns the token. Native keeps it non-null unless
        // completion is confirmed and all active-use counters are released.
        let status = unsafe { rustinfer_cuda_copy_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA copy", &error)
    }
}

impl Drop for CopyHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn embedding_execute(
    table: RawBufferSpan,
    token_ids: RawBufferSpan,
    output: RawBufferSpan,
    device_error_scratch: RawBufferSpan,
    token_count: u64,
    vocabulary_size: u64,
    hidden_size: u64,
    stream: &mut StreamHandle,
) -> NativeEmbeddingCompletion {
    let mut report = RawEmbeddingErrorReport::new();
    let params = RawEmbeddingParams {
        struct_size: EMBEDDING_PARAMS_SIZE,
        reserved0: 0,
        table,
        token_ids,
        output,
        device_error_scratch,
        out_report: &mut report,
        token_count,
        vocabulary_size,
        hidden_size,
        reserved: [0; 3],
    };
    let mut error = ErrorInfo::new();
    // SAFETY: every repr(C) descriptor and the report/error outputs remain
    // alive for this synchronously completing native call. Safe wrappers keep
    // all referenced opaque buffers and the stream borrowed throughout.
    let status = unsafe { rustinfer_cuda_embedding_execute(&params, stream.as_ptr(), &mut error) };
    NativeEmbeddingCompletion {
        report: NativeEmbeddingReport {
            code: report.code,
            token_position: report.token_position,
            token_id: report.token_id,
        },
        result: status_result(status, "execute CUDA embedding", &error),
    }
}

pub(super) fn rms_norm_execute(
    input: RawBufferSpan,
    weight: RawBufferSpan,
    output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRmsNormParams {
        struct_size: RMS_NORM_PARAMS_SIZE,
        reserved0: 0,
        input,
        weight,
        output,
        row_count,
        hidden_size,
        epsilon,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status("execute CUDA RMSNorm", stream, |stream, error| {
        // SAFETY: params and both output buffers remain live for the complete
        // synchronous C call; the safe wrapper retains every opaque handle.
        unsafe { rustinfer_cuda_rms_norm_execute(&params, stream, error) }
    })
}

pub(super) fn fixed37_rms_norm_execute(
    input: RawBufferSpan,
    weight: RawBufferSpan,
    output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRmsNormParams {
        struct_size: RMS_NORM_PARAMS_SIZE,
        reserved0: 0,
        input,
        weight,
        output,
        row_count,
        hidden_size,
        epsilon,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status("execute fixed37 CUDA RMSNorm", stream, |stream, error| {
        // SAFETY: the descriptor and every borrowed opaque resource remain
        // live for the synchronously completing native sibling call.
        unsafe { rustinfer_cuda_fixed37_rms_norm_execute(&params, stream, error) }
    })
}

pub(super) fn residual_add_execute(
    left: RawBufferSpan,
    right: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawResidualAddParams {
        struct_size: RESIDUAL_ADD_PARAMS_SIZE,
        reserved0: 0,
        left,
        right,
        output,
        element_count,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA residual add", stream, |stream, error| {
        // SAFETY: params and the borrowed opaque resources outlive the
        // synchronously completing native operation.
        unsafe { rustinfer_cuda_residual_add_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn residual_rms_norm_execute(
    left: RawBufferSpan,
    right: RawBufferSpan,
    weight: RawBufferSpan,
    residual_output: RawBufferSpan,
    normalized_output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawResidualRmsNormParams {
        struct_size: RESIDUAL_RMS_NORM_PARAMS_SIZE,
        reserved0: 0,
        left,
        right,
        weight,
        residual_output,
        normalized_output,
        row_count,
        hidden_size,
        epsilon,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA fused residual RMSNorm",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and all exclusively borrowed opaque
            // resources outlive this synchronously completing native call.
            unsafe { rustinfer_cuda_residual_rms_norm_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_residual_rms_norm_execute(
    left: RawBufferSpan,
    right: RawBufferSpan,
    weight: RawBufferSpan,
    residual_output: RawBufferSpan,
    normalized_output: RawBufferSpan,
    row_count: u64,
    hidden_size: u64,
    epsilon: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawResidualRmsNormParams {
        struct_size: RESIDUAL_RMS_NORM_PARAMS_SIZE,
        reserved0: 0,
        left,
        right,
        weight,
        residual_output,
        normalized_output,
        row_count,
        hidden_size,
        epsilon,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute fixed37 CUDA fused residual RMSNorm",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and all opaque resources
            // remain exclusively borrowed through synchronous completion.
            unsafe { rustinfer_cuda_fixed37_residual_rms_norm_execute(&params, stream, error) }
        },
    )
}

pub(super) fn fixed37_log_softmax_execute(
    logits: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawFixed37LogSoftmaxParams {
        struct_size: FIXED37_LOG_SOFTMAX_PARAMS_SIZE,
        reserved0: 0,
        logits,
        output,
        element_count,
        reserved: [0; 5],
    };
    primitive_status(
        "execute fixed37 CUDA log-softmax",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and disjoint typed spans remain live for
            // the synchronously completing native operation.
            unsafe { rustinfer_cuda_fixed37_log_softmax_execute(&params, stream, error) }
        },
    )
}

pub(super) fn row_bias_add_in_place_execute(
    matrix: RawBufferSpan,
    bias: RawBufferSpan,
    row_count: u64,
    column_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRowBiasAddInPlaceParams {
        struct_size: ROW_BIAS_ADD_IN_PLACE_PARAMS_SIZE,
        reserved0: 0,
        matrix,
        bias,
        row_count,
        column_count,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA row-bias add in place",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and both borrowed opaque buffers remain live
            // for the synchronously completing native operation.
            unsafe { rustinfer_cuda_row_bias_add_in_place_execute(&params, stream, error) }
        },
    )
}

pub(super) fn silu_execute(
    input: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawSiluParams {
        struct_size: SILU_PARAMS_SIZE,
        reserved0: 0,
        input,
        output,
        element_count,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA SiLU", stream, |stream, error| {
        // SAFETY: params and the borrowed opaque resources outlive the
        // synchronously completing native operation.
        unsafe { rustinfer_cuda_silu_execute(&params, stream, error) }
    })
}

pub(super) fn gated_multiply_execute(
    activated_gate: RawBufferSpan,
    up: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawGatedMultiplyParams {
        struct_size: GATED_MULTIPLY_PARAMS_SIZE,
        reserved0: 0,
        activated_gate,
        up,
        output,
        element_count,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA gated multiply", stream, |stream, error| {
        // SAFETY: params and the borrowed opaque resources outlive the
        // synchronously completing native operation.
        unsafe { rustinfer_cuda_gated_multiply_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rope_execute(
    input: RawBufferSpan,
    cos: RawBufferSpan,
    sin: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    head_count: u64,
    head_size: u64,
    rotary_dimension: u64,
    table_position_count: u64,
    position_offset: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRopeParams {
        struct_size: ROPE_PARAMS_SIZE,
        reserved0: 0,
        input,
        cos,
        sin,
        output,
        token_count,
        head_count,
        head_size,
        rotary_dimension,
        table_position_count,
        position_offset,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA RoPE", stream, |stream, error| {
        // SAFETY: params and the borrowed opaque resources outlive the
        // synchronously completing native operation.
        unsafe { rustinfer_cuda_rope_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn indexed_rope_execute(
    input: RawBufferSpan,
    cos: RawBufferSpan,
    sin: RawBufferSpan,
    positions: RawBufferSpan,
    output: RawBufferSpan,
    active_row_count: u64,
    head_count: u64,
    head_size: u64,
    rotary_dimension: u64,
    table_position_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawIndexedRopeParams {
        struct_size: INDEXED_ROPE_PARAMS_SIZE,
        reserved0: 0,
        input,
        cos,
        sin,
        positions,
        output,
        active_row_count,
        head_count,
        head_size,
        rotary_dimension,
        table_position_count,
        reserved: [0; 4],
    };
    primitive_status("execute CUDA indexed RoPE", stream, |stream, error| {
        // SAFETY: params and every borrowed opaque resource remain live for
        // the synchronously completing native operation.
        unsafe { rustinfer_cuda_indexed_rope_execute(&params, stream, error) }
    })
}

pub(super) fn cast_execute(
    input: RawBufferSpan,
    output: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawCastParams {
        struct_size: CAST_PARAMS_SIZE,
        reserved0: 0,
        input,
        output,
        element_count,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA cast", stream, |stream, error| {
        // SAFETY: params and the borrowed opaque resources outlive the
        // synchronously completing native operation.
        unsafe { rustinfer_cuda_cast_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn row_gather_execute(
    input: RawBufferSpan,
    row_indices: RawBufferSpan,
    output: RawBufferSpan,
    input_row_count: u64,
    output_row_count: u64,
    column_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRowGatherParams {
        struct_size: ROW_GATHER_PARAMS_SIZE,
        reserved0: 0,
        input,
        row_indices,
        output,
        input_row_count,
        output_row_count,
        column_count,
        reserved: [0; 4],
    };
    primitive_status("execute CUDA row gather", stream, |stream, error| {
        // SAFETY: params and every borrowed opaque resource remain live for
        // the synchronously completing native operation.
        unsafe { rustinfer_cuda_row_gather_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn qk_gqa_execute(
    query: RawBufferSpan,
    key: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawQkGqaParams {
        struct_size: QK_GQA_PARAMS_SIZE,
        reserved0: 0,
        query,
        key,
        output,
        token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        reserved: [0; 4],
    };
    primitive_status("execute CUDA QK GQA", stream, |stream, error| {
        // SAFETY: the fixed-layout descriptor and every borrowed native handle
        // remain live for the synchronously completing native call.
        unsafe { rustinfer_cuda_qk_gqa_execute(&params, stream, error) }
    })
}

pub(super) fn scale_causal_mask_in_place_execute(
    scores: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawScaleCausalMaskParams {
        struct_size: SCALE_CAUSAL_MASK_PARAMS_SIZE,
        reserved0: 0,
        scores,
        token_count,
        query_head_count,
        scale,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA attention scale and causal mask",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and exclusively borrowed score buffer
            // remain live for the synchronously completing native call.
            unsafe { rustinfer_cuda_scale_causal_mask_in_place_execute(&params, stream, error) }
        },
    )
}

pub(super) fn causal_softmax_in_place_execute(
    scores: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawCausalSoftmaxParams {
        struct_size: CAUSAL_SOFTMAX_PARAMS_SIZE,
        reserved0: 0,
        scores,
        token_count,
        query_head_count,
        reserved: [0; 5],
    };
    primitive_status("execute CUDA causal softmax", stream, |stream, error| {
        // SAFETY: the descriptor and exclusively borrowed score buffer
        // remain live for the synchronously completing native call.
        unsafe { rustinfer_cuda_causal_softmax_in_place_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn av_gqa_execute(
    probabilities: RawBufferSpan,
    value: RawBufferSpan,
    output: RawBufferSpan,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawAvGqaParams {
        struct_size: AV_GQA_PARAMS_SIZE,
        reserved0: 0,
        probabilities,
        value,
        output,
        token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        reserved: [0; 4],
    };
    primitive_status("execute CUDA AV GQA", stream, |stream, error| {
        // SAFETY: the fixed-layout descriptor and every borrowed native handle
        // remain live for the synchronously completing native call.
        unsafe { rustinfer_cuda_av_gqa_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_attention_execute(
    query: RawBufferSpan,
    key: RawBufferSpan,
    value: RawBufferSpan,
    output: RawBufferSpan,
    batch_size: u64,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    mask_kind: u32,
    local_window: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawPrefillAttentionParams {
        struct_size: PREFILL_ATTENTION_PARAMS_SIZE,
        reserved0: 0,
        query,
        key,
        value,
        output,
        batch_size,
        token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        scale,
        mask_kind,
        local_window,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA online prefill attention",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and all opaque resources remain live for the
            // synchronously completing native call.
            unsafe { rustinfer_cuda_prefill_attention_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kv_cache_write_execute(
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    source_token_count: u64,
    destination_token_start: u64,
    maximum_token_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawKvCacheWriteParams {
        struct_size: KV_CACHE_WRITE_PARAMS_SIZE,
        reserved0: 0,
        key_source,
        value_source,
        key_cache,
        value_cache,
        source_token_count,
        destination_token_start,
        maximum_token_count,
        key_value_head_count,
        head_size,
        reserved: [0; 4],
    };
    primitive_status("write CUDA contiguous KV cache", stream, |stream, error| {
        // SAFETY: the descriptor and every borrowed opaque resource remain
        // live for the synchronously completing native operation.
        unsafe { rustinfer_cuda_kv_cache_write_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_reference_execute(
    query: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    score_workspace: RawBufferSpan,
    output: RawBufferSpan,
    maximum_token_count: u64,
    logical_token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawDecodeAttentionReferenceParams {
        struct_size: DECODE_ATTENTION_REFERENCE_PARAMS_SIZE,
        reserved0: 0,
        query,
        key_cache,
        value_cache,
        score_workspace,
        output,
        maximum_token_count,
        logical_token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        scale,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA materialized decode attention",
        stream,
        |stream, error| {
            // SAFETY: all fixed-layout descriptors and opaque resources live
            // through the synchronous native execution.
            unsafe { rustinfer_cuda_decode_attention_reference_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_attention_execute(
    query: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    maximum_token_count: u64,
    logical_token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    tokens_per_partition: u64,
    partial_state_capacity: u64,
    scale: f32,
    reduction_order: u32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawDecodeAttentionParams {
        struct_size: DECODE_ATTENTION_PARAMS_SIZE,
        reserved0: 0,
        query,
        key_cache,
        value_cache,
        partial_states,
        output,
        maximum_token_count,
        logical_token_count,
        query_head_count,
        key_value_head_count,
        head_size,
        tokens_per_partition,
        partial_state_capacity,
        scale,
        reduction_order,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA chunked decode attention",
        stream,
        |stream, error| {
            // SAFETY: all fixed-layout descriptors and opaque resources live
            // through the synchronous native execution.
            unsafe { rustinfer_cuda_decode_attention_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_partial_state_reduce_execute(
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    partial_state_count: u64,
    partial_state_capacity: u64,
    query_head_count: u64,
    head_size: u64,
    reduction_order: u32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawDecodePartialStateReduceParams {
        struct_size: DECODE_PARTIAL_STATE_REDUCE_PARAMS_SIZE,
        reserved0: 0,
        partial_states,
        output,
        partial_state_count,
        partial_state_capacity,
        query_head_count,
        head_size,
        reduction_order,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "reduce CUDA decode partial states",
        stream,
        |stream, error| {
            // SAFETY: the packed F32 state and BF16 output remain borrowed for
            // the synchronously completing native reducer.
            unsafe { rustinfer_cuda_decode_partial_state_reduce_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn raw_paged_block_table_v1(
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    format_version: u32,
    logical_token_count: u64,
    block_count: u64,
    physical_block_count: u64,
    block_size: u32,
) -> RawPagedKvBlockTableV1 {
    RawPagedKvBlockTableV1 {
        struct_size: PAGED_KV_BLOCK_TABLE_V1_SIZE,
        format_version,
        block_ids,
        valid_tokens,
        logical_token_count,
        block_count,
        physical_block_count,
        block_size,
        metadata_kind: 0,
        metadata_version: 0,
        reserved0: 0,
        reserved: [0; 3],
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn paged_kv_cache_write_execute(
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    format_version: u32,
    logical_token_count: u64,
    block_count: u64,
    physical_block_count: u64,
    block_size: u32,
    source_token_count: u64,
    destination_token_start: u64,
    key_value_head_count: u64,
    head_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawPagedKvCacheWriteParams {
        struct_size: PAGED_KV_CACHE_WRITE_PARAMS_SIZE,
        reserved0: 0,
        key_source,
        value_source,
        key_pool,
        value_pool,
        block_table: raw_paged_block_table_v1(
            block_ids,
            valid_tokens,
            format_version,
            logical_token_count,
            block_count,
            physical_block_count,
            block_size,
        ),
        source_token_count,
        destination_token_start,
        key_value_head_count,
        head_size,
        reserved: [0; 4],
    };
    primitive_status("write CUDA paged KV cache", stream, |stream, error| {
        // SAFETY: the descriptor and all borrowed resources remain live for
        // the synchronously completing native operation.
        unsafe { rustinfer_cuda_paged_kv_cache_write_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn paged_decode_attention_reference_execute(
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    score_workspace: RawBufferSpan,
    output: RawBufferSpan,
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    format_version: u32,
    logical_token_count: u64,
    block_count: u64,
    physical_block_count: u64,
    block_size: u32,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawPagedDecodeAttentionReferenceParams {
        struct_size: PAGED_DECODE_ATTENTION_REFERENCE_PARAMS_SIZE,
        reserved0: 0,
        query,
        key_pool,
        value_pool,
        score_workspace,
        output,
        block_table: raw_paged_block_table_v1(
            block_ids,
            valid_tokens,
            format_version,
            logical_token_count,
            block_count,
            physical_block_count,
            block_size,
        ),
        query_head_count,
        key_value_head_count,
        head_size,
        scale,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA materialized paged decode attention",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and borrowed resources live
            // through synchronous completion.
            unsafe {
                rustinfer_cuda_paged_decode_attention_reference_execute(&params, stream, error)
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn paged_decode_attention_execute(
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    partial_states: RawBufferSpan,
    output: RawBufferSpan,
    block_ids: RawBufferSpan,
    valid_tokens: RawBufferSpan,
    format_version: u32,
    logical_token_count: u64,
    block_count: u64,
    physical_block_count: u64,
    block_size: u32,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    partial_state_capacity: u64,
    scale: f32,
    reduction_order: u32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawPagedDecodeAttentionParams {
        struct_size: PAGED_DECODE_ATTENTION_PARAMS_SIZE,
        reserved0: 0,
        query,
        key_pool,
        value_pool,
        partial_states,
        output,
        block_table: raw_paged_block_table_v1(
            block_ids,
            valid_tokens,
            format_version,
            logical_token_count,
            block_count,
            physical_block_count,
            block_size,
        ),
        query_head_count,
        key_value_head_count,
        head_size,
        partial_state_capacity,
        scale,
        reduction_order,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA paged online decode attention",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and borrowed resources live
            // through synchronous completion.
            unsafe { rustinfer_cuda_paged_decode_attention_execute(&params, stream, error) }
        },
    )
}

#[derive(Clone, Copy)]
pub(super) struct PackedBatchRawV1 {
    pub(super) sequence_block_offsets: RawBufferSpan,
    pub(super) block_ids: RawBufferSpan,
    pub(super) valid_tokens: RawBufferSpan,
    pub(super) row_sequence_slots: RawBufferSpan,
    pub(super) row_positions: RawBufferSpan,
    pub(super) sequence_count: u64,
    pub(super) block_count: u64,
    pub(super) active_row_count: u64,
    pub(super) physical_block_count: u64,
    pub(super) block_size: u32,
}

fn raw_packed_batch_v1(batch: &PackedBatchRawV1) -> RawPackedBatchV1 {
    RawPackedBatchV1 {
        struct_size: PACKED_BATCH_V1_SIZE,
        format_version: PACKED_BATCH_VERSION,
        sequence_block_offsets: batch.sequence_block_offsets,
        block_ids: batch.block_ids,
        valid_tokens: batch.valid_tokens,
        row_sequence_slots: batch.row_sequence_slots,
        row_positions: batch.row_positions,
        sequence_count: batch.sequence_count,
        block_count: batch.block_count,
        active_row_count: batch.active_row_count,
        physical_block_count: batch.physical_block_count,
        block_size: batch.block_size,
        reserved0: 0,
        reserved: [0; 4],
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ragged_paged_kv_cache_write_execute(
    key_source: RawBufferSpan,
    value_source: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    batch: &PackedBatchRawV1,
    key_value_head_count: u64,
    head_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRaggedPagedKvCacheWriteParams {
        struct_size: RAGGED_PAGED_KV_CACHE_WRITE_PARAMS_SIZE,
        reserved0: 0,
        key_source,
        value_source,
        key_pool,
        value_pool,
        batch: raw_packed_batch_v1(batch),
        key_value_head_count,
        head_size,
        reserved: [0; 4],
    };
    primitive_status(
        "write CUDA ragged paged KV cache",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and all exclusively borrowed
            // buffers remain live through synchronous completion.
            unsafe { rustinfer_cuda_ragged_paged_kv_cache_write_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ragged_paged_attention_execute(
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    output: RawBufferSpan,
    batch: &PackedBatchRawV1,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    output_row_count: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRaggedPagedAttentionParams {
        struct_size: RAGGED_PAGED_ATTENTION_PARAMS_SIZE,
        reserved0: 0,
        query,
        key_pool,
        value_pool,
        output,
        batch: raw_packed_batch_v1(batch),
        query_head_count,
        key_value_head_count,
        head_size,
        output_row_count,
        scale,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute CUDA ragged paged attention",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and all exclusively
            // borrowed buffers remain live through synchronous completion.
            unsafe { rustinfer_cuda_ragged_paged_attention_execute(&params, stream, error) }
        },
    )
}

/// Runs the existing staged-BF16 primitives for every dense batch while
/// reusing one `[QH,S,S]` score/probability workspace.
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_attention_reference_execute(
    query: RawBufferSpan,
    key: RawBufferSpan,
    value: RawBufferSpan,
    output: RawBufferSpan,
    workspace: RawBufferSpan,
    batch_size: u64,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    const OPERATION: &str = "execute CUDA materialized prefill attention";
    let query_batch_bytes =
        raw_bf16_product(OPERATION, &[token_count, query_head_count, head_size])?;
    let key_value_batch_bytes =
        raw_bf16_product(OPERATION, &[token_count, key_value_head_count, head_size])?;

    for batch in 0..batch_size {
        let query_offset = batch.checked_mul(query_batch_bytes).ok_or_else(|| {
            CudaError::out_of_range(OPERATION, "query batch offset overflows u64")
        })?;
        let key_value_offset = batch.checked_mul(key_value_batch_bytes).ok_or_else(|| {
            CudaError::out_of_range(OPERATION, "key/value batch offset overflows u64")
        })?;
        let query_batch = raw_subspan(query, query_offset, query_batch_bytes, OPERATION)?;
        let key_batch = raw_subspan(key, key_value_offset, key_value_batch_bytes, OPERATION)?;
        let value_batch = raw_subspan(value, key_value_offset, key_value_batch_bytes, OPERATION)?;
        let output_batch = raw_subspan(output, query_offset, query_batch_bytes, OPERATION)?;

        qk_gqa_execute(
            query_batch,
            key_batch,
            workspace,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            stream,
        )?;
        scale_causal_mask_in_place_execute(
            workspace,
            token_count,
            query_head_count,
            scale,
            stream,
        )?;
        causal_softmax_in_place_execute(workspace, token_count, query_head_count, stream)?;
        av_gqa_execute(
            workspace,
            value_batch,
            output_batch,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            stream,
        )?;
    }
    Ok(())
}

fn raw_bf16_product(operation: &'static str, factors: &[u64]) -> CudaResult<u64> {
    factors
        .iter()
        .copied()
        .chain([2])
        .try_fold(1_u64, |product, factor| {
            product.checked_mul(factor).ok_or_else(|| {
                CudaError::out_of_range(operation, "BF16 byte-length arithmetic overflow")
            })
        })
}

fn raw_subspan(
    mut span: RawBufferSpan,
    relative_offset: u64,
    byte_len: u64,
    operation: &'static str,
) -> CudaResult<RawBufferSpan> {
    let relative_end = relative_offset
        .checked_add(byte_len)
        .ok_or_else(|| CudaError::out_of_range(operation, "batch subspan range overflows u64"))?;
    if relative_end > span.byte_len {
        return Err(CudaError::out_of_range(
            operation,
            "batch subspan exceeds the declared parent span",
        ));
    }
    span.byte_offset = span
        .byte_offset
        .checked_add(relative_offset)
        .ok_or_else(|| CudaError::out_of_range(operation, "batch byte offset overflows u64"))?;
    span.byte_len = byte_len;
    Ok(span)
}

fn primitive_status(
    operation: &'static str,
    stream: &mut StreamHandle,
    call: impl FnOnce(*mut RawStream, *mut ErrorInfo) -> i32,
) -> CudaResult<()> {
    let mut error = ErrorInfo::new();
    let status = call(stream.as_ptr(), &mut error);
    status_result(status, operation, &error)
}

pub(super) struct GemmPlanHandle {
    pointer: Option<NonNull<RawGemmPlan>>,
}

// SAFETY: the opaque plan can move between host threads because each native
// call restores its retained CUDA context. The safe wrapper serializes plan
// access with `&mut self` and deliberately does not make this handle Sync.
unsafe impl Send for GemmPlanHandle {}

impl GemmPlanHandle {
    pub(super) fn create(
        context: &ContextHandle,
        m: u64,
        n: u64,
        k: u64,
        max_workspace_bytes: u64,
    ) -> CudaResult<Self> {
        let config = RawGemmConfig::new(m, n, k, max_workspace_bytes);
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: the context is retained by the native plan on success, the
        // fixed-layout config remains live, and both outputs are writable for
        // this synchronously completing preparation call.
        let status = unsafe {
            rustinfer_cuda_gemm_plan_create(context.as_ptr(), &config, &mut pointer, &mut error)
        };
        status_result(status, "prepare CUDA GEMM plan", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("prepare CUDA GEMM plan", "native GEMM plan handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawGemmPlan {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn info(&self) -> CudaResult<NativeGemmAlgorithmInfo> {
        let mut info = RawGemmAlgorithmInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the owned plan and correctly sized output buffers remain
        // live for the complete native metadata snapshot.
        let status = unsafe { rustinfer_cuda_gemm_plan_info(self.as_ptr(), &mut info, &mut error) };
        status_result(status, "query CUDA GEMM plan metadata", &error)?;
        if info.struct_size != GEMM_ALGORITHM_INFO_SIZE || info.reserved != [0; 2] {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Prepare,
                0,
                "query CUDA GEMM plan metadata",
                "native GEMM metadata has an incompatible struct_size or non-zero reserved field",
            ));
        }
        Ok(NativeGemmAlgorithmInfo {
            backend: info.backend,
            algorithm_id: info.algorithm_id,
            tile_id: info.tile_id,
            stages_id: info.stages_id,
            split_k: info.split_k,
            reduction_scheme: info.reduction_scheme,
            cta_swizzling: info.cta_swizzling,
            custom_option: info.custom_option,
            deterministic: info.deterministic,
            workspace_bytes: info.workspace_bytes,
            numerical_implementation_flags: info.numerical_implementation_flags,
            compute_capability_major: info.compute_capability_major,
            compute_capability_minor: info.compute_capability_minor,
            runtime_version: info.runtime_version,
            cublaslt_version: info.cublaslt_version,
            m: info.m,
            n: info.n,
            k: info.k,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: RawBufferSpan,
        weight: RawBufferSpan,
        output: RawBufferSpan,
        workspace: RawBufferSpan,
        stream: &mut StreamHandle,
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the safe layer exclusively borrows the plan, output,
        // workspace, and stream and keeps the immutable inputs live. Native
        // synchronizes the explicit stream before returning and retains every
        // active-use guard if completion or context restoration is ambiguous.
        let status = unsafe {
            rustinfer_cuda_gemm_plan_execute(
                self.as_ptr(),
                &input,
                &weight,
                &output,
                &workspace,
                stream.as_ptr(),
                &mut error,
            )
        };
        status_result(status, "execute CUDA GEMM plan", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw uniquely owns the native plan. Native leaves it non-null
        // after any ambiguous destruction or permanent-use failure and nulls
        // it only after descriptor teardown and context restoration complete.
        let status = unsafe { rustinfer_cuda_gemm_plan_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA GEMM plan", &error)
    }
}

impl Drop for GemmPlanHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct Fixed37GemmPlanHandle {
    pointer: Option<NonNull<RawFixed37GemmPlan>>,
}

// SAFETY: the opaque plan may move between host threads. Native restores its
// retained context per call, while the safe wrapper remains !Sync and requires
// exclusive execution access.
unsafe impl Send for Fixed37GemmPlanHandle {}

impl Fixed37GemmPlanHandle {
    pub(super) fn create(
        context: &ContextHandle,
        m: u64,
        n: u64,
        k: u64,
        max_workspace_bytes: u64,
    ) -> CudaResult<Self> {
        let config = RawGemmConfig::new(m, n, k, max_workspace_bytes);
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: native retains the context on success and initializes the
        // owned output handle or leaves it null on failure.
        let status = unsafe {
            rustinfer_cuda_fixed37_gemm_plan_create(
                context.as_ptr(),
                &config,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "prepare fixed37 CUDA GEMM plan", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output(
                "prepare fixed37 CUDA GEMM plan",
                "native fixed37 GEMM plan handle is null",
            )
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawFixed37GemmPlan {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn info(&self) -> CudaResult<NativeFixed37GemmPlanInfo> {
        let mut info = RawFixed37GemmPlanInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the plan and fixed-size output remain live for the metadata
        // snapshot; native serializes this query with execution and close.
        let status =
            unsafe { rustinfer_cuda_fixed37_gemm_plan_info(self.as_ptr(), &mut info, &mut error) };
        status_result(status, "query fixed37 CUDA GEMM metadata", &error)?;
        if info.struct_size != FIXED37_GEMM_PLAN_INFO_SIZE || info.reserved != [0; 3] {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Prepare,
                0,
                "query fixed37 CUDA GEMM metadata",
                "native fixed37 GEMM metadata has an incompatible struct_size or reserved tail",
            ));
        }
        Ok(NativeFixed37GemmPlanInfo {
            backend: info.backend,
            reduction_version: info.reduction_version,
            chunk_elements: info.chunk_elements,
            accumulator_dtype: info.accumulator_dtype,
            output_dtype: info.output_dtype,
            threads_per_block: info.threads_per_block,
            deterministic: info.deterministic,
            dynamic_shared_memory_bytes: info.dynamic_shared_memory_bytes,
            workspace_bytes: info.workspace_bytes,
            m: info.m,
            n: info.n,
            k: info.k,
        })
    }

    pub(super) fn execute(
        &mut self,
        input: RawBufferSpan,
        weight: RawBufferSpan,
        output: RawBufferSpan,
        stream: &mut StreamHandle,
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the safe owner keeps the plan, stream, and all disjoint
        // buffers exclusively borrowed until native synchronization returns.
        let status = unsafe {
            rustinfer_cuda_fixed37_gemm_plan_execute(
                self.as_ptr(),
                &input,
                &weight,
                &output,
                stream.as_ptr(),
                &mut error,
            )
        };
        status_result(status, "execute fixed37 CUDA GEMM plan", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned; native consumes and nulls only after
        // its context-child lease can be released safely.
        let status = unsafe { rustinfer_cuda_fixed37_gemm_plan_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close fixed37 CUDA GEMM plan", &error)
    }
}

impl Drop for Fixed37GemmPlanHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub(super) struct SmokeHandle {
    pointer: Option<NonNull<RawSmokeBuffer>>,
}

// SAFETY: diagnostic buffers move only with their borrowed !Sync stream in the
// safe layer; context lifetime is retained separately by Arc.
unsafe impl Send for SmokeHandle {}

impl SmokeHandle {
    pub(super) fn create(context: &ContextHandle, element_count: u64) -> CudaResult<Self> {
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: context is live and native returns one owned opaque handle.
        let status = unsafe {
            rustinfer_cuda_smoke_buffer_create(
                context.as_ptr(),
                element_count,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "create CUDA smoke buffer", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("create CUDA smoke buffer", "native smoke handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawSmokeBuffer {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn launch(&mut self, stream: &mut StreamHandle, value: f32) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both handles remain live and native validates context/launch.
        let status = unsafe {
            rustinfer_cuda_smoke_fill_launch(self.as_ptr(), stream.as_ptr(), value, &mut error)
        };
        status_result(status, "launch CUDA smoke fill", &error)
    }

    pub(super) fn copy_to_host(
        &mut self,
        stream: &mut StreamHandle,
        output: &mut [f32],
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        let capacity = u64::try_from(output.len()).map_err(|_| {
            CudaError::out_of_range(
                "copy CUDA smoke buffer",
                "host output length does not fit u64",
            )
        })?;
        // SAFETY: output remains exclusively borrowed for the complete native
        // enqueue+stream-synchronize call, and both handles remain live.
        let status = unsafe {
            rustinfer_cuda_smoke_copy_to_host(
                self.as_ptr(),
                stream.as_ptr(),
                output.as_mut_ptr(),
                capacity,
                &mut error,
            )
        };
        status_result(status, "copy CUDA smoke buffer", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: raw is uniquely owned; native synchronizes any in-flight
        // operation before consuming and nulling the handle.
        let status = unsafe { rustinfer_cuda_smoke_buffer_close(&mut raw, &mut error) };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA smoke buffer", &error)
    }
}

impl Drop for SmokeHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn query_result(
    status: i32,
    complete: u8,
    operation: &'static str,
    error: &ErrorInfo,
) -> CudaResult<bool> {
    if status == STATUS_NOT_READY {
        return Ok(false);
    }
    status_result(status, operation, error)?;
    Ok(complete != 0)
}

fn copy_completion(
    status: i32,
    complete: u8,
    operation: &'static str,
    error: &ErrorInfo,
) -> CopyCompletion {
    let result = if status == STATUS_NOT_READY {
        Ok(())
    } else {
        status_result(status, operation, error)
    };
    CopyCompletion {
        complete: status != STATUS_NOT_READY && complete != 0,
        result,
    }
}

fn status_result(status: i32, operation: &'static str, error: &ErrorInfo) -> CudaResult<()> {
    if status == STATUS_SUCCESS {
        return Ok(());
    }
    let kind = match status {
        STATUS_INVALID_ARGUMENT => CudaErrorKind::InvalidArgument,
        STATUS_INVALID_DEVICE => CudaErrorKind::InvalidDevice,
        STATUS_OUT_OF_RANGE => CudaErrorKind::OutOfRange,
        STATUS_NOT_READY => CudaErrorKind::NotReady,
        STATUS_OUT_OF_MEMORY => CudaErrorKind::OutOfMemory,
        STATUS_DRIVER_ERROR => CudaErrorKind::Driver,
        STATUS_RUNTIME_ERROR | STATUS_CUBLASLT_ERROR => CudaErrorKind::Runtime,
        STATUS_INVALID_STATE => CudaErrorKind::InvalidState,
        STATUS_NOT_SUPPORTED => CudaErrorKind::NotSupported,
        _ => CudaErrorKind::Internal,
    };
    let domain = match error.domain {
        DOMAIN_VALIDATION => CudaErrorDomain::Validation,
        DOMAIN_DRIVER => CudaErrorDomain::Driver,
        DOMAIN_RUNTIME => CudaErrorDomain::Runtime,
        DOMAIN_CUBLASLT => CudaErrorDomain::CuBlasLt,
        _ => CudaErrorDomain::Internal,
    };
    let stage = match error.stage {
        STAGE_INITIALIZE => CudaErrorStage::Initialize,
        STAGE_CREATE => CudaErrorStage::Create,
        STAGE_PREPARE => CudaErrorStage::Prepare,
        STAGE_LAUNCH => CudaErrorStage::Launch,
        STAGE_SYNCHRONIZE => CudaErrorStage::Synchronize,
        STAGE_QUERY => CudaErrorStage::Query,
        STAGE_RECORD => CudaErrorStage::Record,
        STAGE_COPY => CudaErrorStage::Copy,
        STAGE_CLOSE => CudaErrorStage::Close,
        _ => CudaErrorStage::Validation,
    };
    let message = c_array_to_string(&error.message);
    Err(CudaError::new(
        kind,
        domain,
        stage,
        error.native_code,
        operation,
        if message.is_empty() {
            format!("native ABI returned undocumented status {status}")
        } else {
            message
        },
    ))
}

fn missing_output(operation: &'static str, message: &'static str) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Create,
        0,
        operation,
        message,
    )
}

fn c_array_to_string<const N: usize>(bytes: &[c_char; N]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .map(|byte| u8::from_ne_bytes(byte.to_ne_bytes()))
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

const _: () = assert!(size_of::<ErrorInfo>() == 272);
const _: () = assert!(offset_of!(ErrorInfo, message) == 16);
const _: () = assert!(size_of::<RawDeviceProperties>() == 320);
const _: () = assert!(offset_of!(RawDeviceProperties, name) == 64);
const _: () = assert!(size_of::<RawAllocationStats>() == 40);
const _: () = assert!(offset_of!(RawAllocationStats, device_live_bytes) == 8);
const _: () = assert!(offset_of!(RawAllocationStats, pinned_host_live_allocations) == 32);
const _: () = assert!(size_of::<RawBufferSpan>() == 48);
const _: () = assert!(offset_of!(RawBufferSpan, buffer) == 8);
const _: () = assert!(offset_of!(RawBufferSpan, reserved) == 32);
const _: () = assert!(size_of::<RawEmbeddingErrorReport>() == 32);
const _: () = assert!(offset_of!(RawEmbeddingErrorReport, token_position) == 8);
const _: () = assert!(size_of::<RawEmbeddingParams>() == 256);
const _: () = assert!(offset_of!(RawEmbeddingParams, out_report) == 200);
const _: () = assert!(size_of::<RawRmsNormParams>() == 208);
const _: () = assert!(offset_of!(RawRmsNormParams, epsilon) == 168);
const _: () = assert!(size_of::<RawFixed37LogSoftmaxParams>() == 152);
const _: () = assert!(offset_of!(RawFixed37LogSoftmaxParams, logits) == 8);
const _: () = assert!(offset_of!(RawFixed37LogSoftmaxParams, output) == 56);
const _: () = assert!(offset_of!(RawFixed37LogSoftmaxParams, element_count) == 104);
const _: () = assert!(size_of::<RawResidualAddParams>() == 200);
const _: () = assert!(size_of::<RawResidualRmsNormParams>() == 304);
const _: () = assert!(offset_of!(RawResidualRmsNormParams, residual_output) == 152);
const _: () = assert!(offset_of!(RawResidualRmsNormParams, row_count) == 248);
const _: () = assert!(offset_of!(RawResidualRmsNormParams, epsilon) == 264);
const _: () = assert!(size_of::<RawRowBiasAddInPlaceParams>() == 152);
const _: () = assert!(offset_of!(RawRowBiasAddInPlaceParams, matrix) == 8);
const _: () = assert!(offset_of!(RawRowBiasAddInPlaceParams, row_count) == 104);
const _: () = assert!(offset_of!(RawRowBiasAddInPlaceParams, reserved) == 120);
const _: () = assert!(size_of::<RawSiluParams>() == 152);
const _: () = assert!(size_of::<RawGatedMultiplyParams>() == 200);
const _: () = assert!(size_of::<RawRopeParams>() == 288);
const _: () = assert!(offset_of!(RawRopeParams, position_offset) == 240);
const _: () = assert!(size_of::<RawIndexedRopeParams>() == 320);
const _: () = assert!(offset_of!(RawIndexedRopeParams, input) == 8);
const _: () = assert!(offset_of!(RawIndexedRopeParams, active_row_count) == 248);
const _: () = assert!(offset_of!(RawIndexedRopeParams, reserved) == 288);
const _: () = assert!(size_of::<RawCastParams>() == 152);
const _: () = assert!(size_of::<RawRowGatherParams>() == 208);
const _: () = assert!(offset_of!(RawRowGatherParams, input_row_count) == 152);
const _: () = assert!(offset_of!(RawRowGatherParams, reserved) == 176);
const _: () = assert!(size_of::<RawQkGqaParams>() == 216);
const _: () = assert!(offset_of!(RawQkGqaParams, token_count) == 152);
const _: () = assert!(offset_of!(RawQkGqaParams, reserved) == 184);
const _: () = assert!(size_of::<RawScaleCausalMaskParams>() == 112);
const _: () = assert!(offset_of!(RawScaleCausalMaskParams, scale) == 72);
const _: () = assert!(offset_of!(RawScaleCausalMaskParams, reserved) == 80);
const _: () = assert!(size_of::<RawCausalSoftmaxParams>() == 112);
const _: () = assert!(offset_of!(RawCausalSoftmaxParams, reserved) == 72);
const _: () = assert!(size_of::<RawAvGqaParams>() == 216);
const _: () = assert!(offset_of!(RawAvGqaParams, token_count) == 152);
const _: () = assert!(offset_of!(RawAvGqaParams, reserved) == 184);
const _: () = assert!(size_of::<RawPrefillAttentionParams>() == 288);
const _: () = assert!(offset_of!(RawPrefillAttentionParams, query) == 8);
const _: () = assert!(offset_of!(RawPrefillAttentionParams, batch_size) == 200);
const _: () = assert!(offset_of!(RawPrefillAttentionParams, scale) == 240);
const _: () = assert!(offset_of!(RawPrefillAttentionParams, local_window) == 248);
const _: () = assert!(offset_of!(RawPrefillAttentionParams, reserved) == 256);
const _: () = assert!(size_of::<RawKvCacheWriteParams>() == 272);
const _: () = assert!(offset_of!(RawKvCacheWriteParams, key_source) == 8);
const _: () = assert!(offset_of!(RawKvCacheWriteParams, source_token_count) == 200);
const _: () = assert!(offset_of!(RawKvCacheWriteParams, reserved) == 240);
const _: () = assert!(size_of::<RawDecodeAttentionReferenceParams>() == 328);
const _: () = assert!(offset_of!(RawDecodeAttentionReferenceParams, query) == 8);
const _: () = assert!(offset_of!(RawDecodeAttentionReferenceParams, maximum_token_count) == 248);
const _: () = assert!(offset_of!(RawDecodeAttentionReferenceParams, scale) == 288);
const _: () = assert!(offset_of!(RawDecodeAttentionReferenceParams, reserved) == 296);
const _: () = assert!(size_of::<RawDecodeAttentionParams>() == 344);
const _: () = assert!(offset_of!(RawDecodeAttentionParams, query) == 8);
const _: () = assert!(offset_of!(RawDecodeAttentionParams, maximum_token_count) == 248);
const _: () = assert!(offset_of!(RawDecodeAttentionParams, scale) == 304);
const _: () = assert!(offset_of!(RawDecodeAttentionParams, reserved) == 312);
const _: () = assert!(size_of::<RawDecodePartialStateReduceParams>() == 176);
const _: () = assert!(offset_of!(RawDecodePartialStateReduceParams, partial_states) == 8);
const _: () = assert!(offset_of!(RawDecodePartialStateReduceParams, partial_state_count) == 104);
const _: () = assert!(offset_of!(RawDecodePartialStateReduceParams, reduction_order) == 136);
const _: () = assert!(offset_of!(RawDecodePartialStateReduceParams, reserved) == 144);
const _: () = assert!(size_of::<RawPagedKvBlockTableV1>() == 168);
const _: () = assert!(offset_of!(RawPagedKvBlockTableV1, block_ids) == 8);
const _: () = assert!(offset_of!(RawPagedKvBlockTableV1, logical_token_count) == 104);
const _: () = assert!(offset_of!(RawPagedKvBlockTableV1, block_size) == 128);
const _: () = assert!(offset_of!(RawPagedKvBlockTableV1, reserved) == 144);
const _: () = assert!(size_of::<RawPagedKvCacheWriteParams>() == 432);
const _: () = assert!(offset_of!(RawPagedKvCacheWriteParams, block_table) == 200);
const _: () = assert!(offset_of!(RawPagedKvCacheWriteParams, source_token_count) == 368);
const _: () = assert!(offset_of!(RawPagedKvCacheWriteParams, reserved) == 400);
const _: () = assert!(size_of::<RawPagedDecodeAttentionReferenceParams>() == 480);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionReferenceParams, block_table) == 248);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionReferenceParams, query_head_count) == 416);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionReferenceParams, scale) == 440);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionReferenceParams, reserved) == 448);
const _: () = assert!(size_of::<RawPagedDecodeAttentionParams>() == 488);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionParams, block_table) == 248);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionParams, query_head_count) == 416);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionParams, scale) == 448);
const _: () = assert!(offset_of!(RawPagedDecodeAttentionParams, reserved) == 456);
const _: () = assert!(size_of::<RawPackedBatchV1>() == 320);
const _: () = assert!(offset_of!(RawPackedBatchV1, sequence_block_offsets) == 8);
const _: () = assert!(offset_of!(RawPackedBatchV1, sequence_count) == 248);
const _: () = assert!(offset_of!(RawPackedBatchV1, reserved) == 288);
const _: () = assert!(size_of::<RawRaggedPagedKvCacheWriteParams>() == 568);
const _: () = assert!(offset_of!(RawRaggedPagedKvCacheWriteParams, batch) == 200);
const _: () = assert!(offset_of!(RawRaggedPagedKvCacheWriteParams, key_value_head_count) == 520);
const _: () = assert!(offset_of!(RawRaggedPagedKvCacheWriteParams, reserved) == 536);
const _: () = assert!(size_of::<RawRaggedPagedAttentionParams>() == 592);
const _: () = assert!(offset_of!(RawRaggedPagedAttentionParams, batch) == 200);
const _: () = assert!(offset_of!(RawRaggedPagedAttentionParams, query_head_count) == 520);
const _: () = assert!(offset_of!(RawRaggedPagedAttentionParams, output_row_count) == 544);
const _: () = assert!(offset_of!(RawRaggedPagedAttentionParams, scale) == 552);
const _: () = assert!(offset_of!(RawRaggedPagedAttentionParams, reserved) == 560);
const _: () = assert!(size_of::<RawGemmConfig>() == 112);
const _: () = assert!(offset_of!(RawGemmConfig, m) == 8);
const _: () = assert!(offset_of!(RawGemmConfig, input_dtype) == 32);
const _: () = assert!(offset_of!(RawGemmConfig, max_workspace_bytes) == 80);
const _: () = assert!(size_of::<RawGemmAlgorithmInfo>() == 112);
const _: () = assert!(offset_of!(RawGemmAlgorithmInfo, workspace_bytes) == 40);
const _: () = assert!(offset_of!(RawGemmAlgorithmInfo, numerical_implementation_flags) == 48);
const _: () = assert!(offset_of!(RawGemmAlgorithmInfo, m) == 72);
const _: () = assert!(size_of::<RawFixed37GemmPlanInfo>() == 96);
const _: () = assert!(offset_of!(RawFixed37GemmPlanInfo, dynamic_shared_memory_bytes) == 32);
const _: () = assert!(offset_of!(RawFixed37GemmPlanInfo, m) == 48);
const _: () = assert!(offset_of!(RawFixed37GemmPlanInfo, reserved) == 72);
