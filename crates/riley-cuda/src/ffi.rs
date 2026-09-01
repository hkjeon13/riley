use std::ffi::{CStr, c_char};
use std::marker::PhantomData;
use std::mem::{offset_of, size_of};
use std::ptr::{self, NonNull};

use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::graph::{
    CudaGraphFailureInfo, CudaGraphStage, RawGraphErrorInfo, decode_graph_failure_info,
};

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
const DOMAIN_NVML: u32 = 6;

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
#[cfg(feature = "nvml")]
const NVIDIA_DRIVER_VERSION_CAPACITY: usize = 80;
#[cfg(feature = "nvml")]
const NVIDIA_ENVIRONMENT_MAX_DEVICES: usize = 32;
#[cfg(feature = "nvml")]
const NVIDIA_CLOCK_NOT_AVAILABLE: u32 = u32::MAX;
#[cfg(feature = "nvml")]
const NVIDIA_PERSISTENCE_DISABLED: u32 = 0;
#[cfg(feature = "nvml")]
const NVIDIA_PERSISTENCE_ENABLED: u32 = 1;
const ERROR_INFO_SIZE: u32 = 272;
const DEVICE_PROPERTIES_SIZE: u32 = 320;
#[cfg(feature = "nvml")]
const NVIDIA_DEVICE_SNAPSHOT_SIZE: u32 = 320;
#[cfg(feature = "nvml")]
const NVIDIA_ENVIRONMENT_SNAPSHOT_SIZE: u32 = 10_352;
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
const ROPE_TABLE_PARAMS_SIZE: u32 = 152;
const ROPE_PARAMS_SIZE: u32 = 288;
const INDEXED_ROPE_PARAMS_SIZE: u32 = 320;
const CAST_PARAMS_SIZE: u32 = 152;
const ROW_GATHER_PARAMS_SIZE: u32 = 208;
const BF16_ARGMAX_PARAMS_SIZE: u32 = 152;
const QK_GQA_PARAMS_SIZE: u32 = 216;
const SCALE_CAUSAL_MASK_PARAMS_SIZE: u32 = 112;
const CAUSAL_SOFTMAX_PARAMS_SIZE: u32 = 112;
const AV_GQA_PARAMS_SIZE: u32 = 216;
const PREFILL_ATTENTION_PARAMS_SIZE: u32 = 288;
const HF_PREFILL_ATTENTION_CONFIG_SIZE: u32 = 96;
const HF_PREFILL_ATTENTION_PLAN_INFO_SIZE: u32 = 216;
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
const FIXED37_RAGGED_PAGED_ATTENTION_PARAMS_SIZE: u32 = 600;
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
const GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K: u32 = 1;
const GEMM_FLAG_ALLOW_INPLACE_SPLIT_K: u32 = 2;

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

#[cfg(feature = "nvml")]
#[repr(C)]
#[derive(Clone, Copy)]
struct RawNvidiaDeviceSnapshot {
    struct_size: u32,
    index: u32,
    total_memory_bytes: u64,
    used_memory_bytes: u64,
    temperature_c: u32,
    persistence_mode: u32,
    power_limit_milliwatts: u32,
    application_graphics_clock_mhz: u32,
    application_memory_clock_mhz: u32,
    compute_process_count: u32,
    reserved: [u64; 2],
    name: [c_char; DEVICE_NAME_CAPACITY],
}

#[cfg(feature = "nvml")]
impl RawNvidiaDeviceSnapshot {
    const fn new() -> Self {
        Self {
            struct_size: NVIDIA_DEVICE_SNAPSHOT_SIZE,
            index: 0,
            total_memory_bytes: 0,
            used_memory_bytes: 0,
            temperature_c: 0,
            persistence_mode: NVIDIA_PERSISTENCE_DISABLED,
            power_limit_milliwatts: 0,
            application_graphics_clock_mhz: NVIDIA_CLOCK_NOT_AVAILABLE,
            application_memory_clock_mhz: NVIDIA_CLOCK_NOT_AVAILABLE,
            compute_process_count: 0,
            reserved: [0; 2],
            name: [0; DEVICE_NAME_CAPACITY],
        }
    }
}

#[cfg(feature = "nvml")]
#[repr(C)]
struct RawNvidiaEnvironmentSnapshot {
    struct_size: u32,
    cuda_driver_api_version: i32,
    device_count: u32,
    compute_process_count: u32,
    reserved: [u64; 2],
    driver_version: [c_char; NVIDIA_DRIVER_VERSION_CAPACITY],
    devices: [RawNvidiaDeviceSnapshot; NVIDIA_ENVIRONMENT_MAX_DEVICES],
}

#[cfg(feature = "nvml")]
impl RawNvidiaEnvironmentSnapshot {
    const fn new() -> Self {
        Self {
            struct_size: NVIDIA_ENVIRONMENT_SNAPSHOT_SIZE,
            cuda_driver_api_version: 0,
            device_count: 0,
            compute_process_count: 0,
            reserved: [0; 2],
            driver_version: [0; NVIDIA_DRIVER_VERSION_CAPACITY],
            devices: [RawNvidiaDeviceSnapshot::new(); NVIDIA_ENVIRONMENT_MAX_DEVICES],
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

#[cfg(feature = "nvml")]
#[derive(Debug)]
pub(super) struct NativeNvidiaDeviceSnapshot {
    pub(super) index: u32,
    pub(super) name: String,
    pub(super) total_memory_bytes: u64,
    pub(super) used_memory_bytes: u64,
    pub(super) temperature_c: u32,
    pub(super) persistence_mode: u32,
    pub(super) power_limit_milliwatts: u32,
    pub(super) application_graphics_clock_mhz: Option<u32>,
    pub(super) application_memory_clock_mhz: Option<u32>,
    pub(super) compute_process_count: u32,
}

#[cfg(feature = "nvml")]
#[derive(Debug)]
pub(super) struct NativeNvidiaEnvironmentSnapshot {
    pub(super) driver_version: String,
    pub(super) cuda_driver_api_version: i32,
    pub(super) compute_process_count: u32,
    pub(super) devices: Vec<NativeNvidiaDeviceSnapshot>,
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
struct RawGraphCapture {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawGraph {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawGraphExec {
    _private: [u8; 0],
    _not_send_sync: PhantomData<*mut ()>,
}

#[repr(C)]
struct RawGraphLaunch {
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

#[repr(C)]
struct RawHfPrefillAttentionPlan {
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
struct RawRopeTableParams {
    struct_size: u32,
    reserved0: u32,
    angles_cos: RawBufferSpan,
    sin: RawBufferSpan,
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
struct RawBf16ArgmaxParams {
    struct_size: u32,
    reserved0: u32,
    logits: RawBufferSpan,
    results: RawBufferSpan,
    row_count: u64,
    vocabulary_size: u64,
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
struct RawHfPrefillAttentionConfig {
    struct_size: u32,
    reserved0: u32,
    batch_count: u64,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    deterministic: u32,
    max_cublas_workspace_bytes: u64,
    reserved: [u64; 4],
}

impl RawHfPrefillAttentionConfig {
    const fn new(
        batch_count: u64,
        token_count: u64,
        query_head_count: u64,
        key_value_head_count: u64,
        head_size: u64,
        scale: f32,
        max_cublas_workspace_bytes: u64,
    ) -> Self {
        Self {
            struct_size: HF_PREFILL_ATTENTION_CONFIG_SIZE,
            reserved0: 0,
            batch_count,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            scale,
            deterministic: GEMM_DETERMINISTIC_REQUIRED,
            max_cublas_workspace_bytes,
            reserved: [0; 4],
        }
    }
}

#[repr(C)]
struct RawHfPrefillAttentionPlanInfo {
    struct_size: u32,
    backend: u32,
    qk_algorithm_id: i32,
    qk_tile_id: u32,
    qk_stages_id: u32,
    qk_split_k: u32,
    qk_reduction_scheme: u32,
    qk_cta_swizzling: u32,
    qk_custom_option: u32,
    qk_reserved0: u32,
    qk_workspace_bytes: u64,
    qk_numerical_implementation_flags: u64,
    av_algorithm_id: i32,
    av_tile_id: u32,
    av_stages_id: u32,
    av_split_k: u32,
    av_reduction_scheme: u32,
    av_cta_swizzling: u32,
    av_custom_option: u32,
    av_reserved0: u32,
    av_workspace_bytes: u64,
    av_numerical_implementation_flags: u64,
    deterministic: u32,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    runtime_version: i32,
    cublaslt_version: i32,
    reserved0: u32,
    workspace_bytes: u64,
    score_bytes: u64,
    repeated_key_value_bytes: u64,
    layout_copy_bytes: u64,
    batch_count: u64,
    token_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    reserved: [u64; 2],
}

impl RawHfPrefillAttentionPlanInfo {
    const fn new() -> Self {
        Self {
            struct_size: HF_PREFILL_ATTENTION_PLAN_INFO_SIZE,
            backend: 0,
            qk_algorithm_id: 0,
            qk_tile_id: 0,
            qk_stages_id: 0,
            qk_split_k: 0,
            qk_reduction_scheme: 0,
            qk_cta_swizzling: 0,
            qk_custom_option: 0,
            qk_reserved0: 0,
            qk_workspace_bytes: 0,
            qk_numerical_implementation_flags: 0,
            av_algorithm_id: 0,
            av_tile_id: 0,
            av_stages_id: 0,
            av_split_k: 0,
            av_reduction_scheme: 0,
            av_cta_swizzling: 0,
            av_custom_option: 0,
            av_reserved0: 0,
            av_workspace_bytes: 0,
            av_numerical_implementation_flags: 0,
            deterministic: 0,
            compute_capability_major: 0,
            compute_capability_minor: 0,
            runtime_version: 0,
            cublaslt_version: 0,
            reserved0: 0,
            workspace_bytes: 0,
            score_bytes: 0,
            repeated_key_value_bytes: 0,
            layout_copy_bytes: 0,
            batch_count: 0,
            token_count: 0,
            query_head_count: 0,
            key_value_head_count: 0,
            head_size: 0,
            reserved: [0; 2],
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NativeHfPrefillAttentionPlanInfo {
    pub(super) backend: u32,
    pub(super) qk_algorithm_id: i32,
    pub(super) qk_tile_id: u32,
    pub(super) qk_stages_id: u32,
    pub(super) qk_split_k: u32,
    pub(super) qk_reduction_scheme: u32,
    pub(super) qk_cta_swizzling: u32,
    pub(super) qk_custom_option: u32,
    pub(super) qk_workspace_bytes: u64,
    pub(super) qk_numerical_implementation_flags: u64,
    pub(super) av_algorithm_id: i32,
    pub(super) av_tile_id: u32,
    pub(super) av_stages_id: u32,
    pub(super) av_split_k: u32,
    pub(super) av_reduction_scheme: u32,
    pub(super) av_cta_swizzling: u32,
    pub(super) av_custom_option: u32,
    pub(super) av_workspace_bytes: u64,
    pub(super) av_numerical_implementation_flags: u64,
    pub(super) deterministic: u32,
    pub(super) compute_capability_major: u32,
    pub(super) compute_capability_minor: u32,
    pub(super) runtime_version: i32,
    pub(super) cublaslt_version: i32,
    pub(super) workspace_bytes: u64,
    pub(super) score_bytes: u64,
    pub(super) repeated_key_value_bytes: u64,
    pub(super) layout_copy_bytes: u64,
    pub(super) batch_count: u64,
    pub(super) token_count: u64,
    pub(super) query_head_count: u64,
    pub(super) key_value_head_count: u64,
    pub(super) head_size: u64,
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
struct RawFixed37RaggedPagedAttentionParams {
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
    maximum_logical_token_count: u64,
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
    const fn new(flags: u32, m: u64, n: u64, k: u64, max_workspace_bytes: u64) -> Self {
        Self {
            struct_size: GEMM_CONFIG_SIZE,
            flags,
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
    fn riley_cuda_abi_version() -> u32;
    fn riley_cuda_build_info() -> *const c_char;
    fn riley_cuda_device_count(out_count: *mut u32, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_device_properties(
        ordinal: i32,
        out_properties: *mut RawDeviceProperties,
        error: *mut ErrorInfo,
    ) -> i32;
    #[cfg(feature = "nvml")]
    fn riley_cuda_nvidia_environment_probe(
        out_snapshot: *mut RawNvidiaEnvironmentSnapshot,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_context_create(
        ordinal: i32,
        out_context: *mut *mut RawContext,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_context_synchronize(context: *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_context_memory_info(
        context: *mut RawContext,
        out_free_bytes: *mut u64,
        out_total_bytes: *mut u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_context_allocation_stats(
        context: *mut RawContext,
        out_stats: *mut RawAllocationStats,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_context_close(context: *mut *mut RawContext, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_context_defer_to_active_capture(
        context: *mut *mut RawContext,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_stream_create(
        context: *mut RawContext,
        out_stream: *mut *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_stream_query(
        stream: *mut RawStream,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_stream_synchronize(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_stream_command_batch_begin(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_stream_command_batch_end(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_graph_capture_begin(
        stream: *mut RawStream,
        mode: u32,
        out_capture: *mut *mut RawGraphCapture,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_capture_abort(
        capture: *mut *mut RawGraphCapture,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_capture_begin_fill_f32(
        stream: *mut RawStream,
        buffer: *mut RawDeviceBuffer,
        element_count: u64,
        mode: u32,
        out_capture: *mut *mut RawGraphCapture,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_capture_enqueue_fill_f32(
        capture: *mut RawGraphCapture,
        value: f32,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_capture_end(
        capture: *mut *mut RawGraphCapture,
        out_graph: *mut *mut RawGraph,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_instantiate(
        graph: *mut *mut RawGraph,
        out_exec: *mut *mut RawGraphExec,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_exec_launch(
        exec: *mut RawGraphExec,
        stream: *mut RawStream,
        out_launch: *mut *mut RawGraphLaunch,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_launch_complete(
        launch: *mut *mut RawGraphLaunch,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_close(
        graph: *mut *mut RawGraph,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_graph_exec_close(
        exec: *mut *mut RawGraphExec,
        out_graph_error: *mut RawGraphErrorInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_stream_wait_event(
        stream: *mut RawStream,
        event: *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_stream_close(stream: *mut *mut RawStream, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_stream_defer_to_active_capture(
        stream: *mut *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_event_create(
        context: *mut RawContext,
        out_event: *mut *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_event_record(
        event: *mut RawEvent,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_event_query(
        event: *mut RawEvent,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_event_synchronize(event: *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_event_elapsed_ms(
        start: *mut RawEvent,
        end: *mut RawEvent,
        out_elapsed_ms: *mut f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_event_close(event: *mut *mut RawEvent, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_event_defer_to_active_capture(
        event: *mut *mut RawEvent,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_device_buffer_create(
        context: *mut RawContext,
        byte_len: u64,
        out_buffer: *mut *mut RawDeviceBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_device_buffer_close(
        buffer: *mut *mut RawDeviceBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_device_buffer_defer_to_active_capture(
        buffer: *mut *mut RawDeviceBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_pinned_host_buffer_create(
        context: *mut RawContext,
        byte_len: u64,
        out_buffer: *mut *mut RawPinnedHostBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_pinned_host_buffer_write(
        buffer: *mut RawPinnedHostBuffer,
        destination_offset: u64,
        source: *const u8,
        source_len: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_pinned_host_buffer_read(
        buffer: *mut RawPinnedHostBuffer,
        source_offset: u64,
        destination: *mut u8,
        destination_len: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_pinned_host_buffer_close(
        buffer: *mut *mut RawPinnedHostBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_pinned_host_buffer_defer_to_active_capture(
        buffer: *mut *mut RawPinnedHostBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_copy_h2d_async(
        destination: *mut RawDeviceBuffer,
        destination_offset: u64,
        source: *mut RawPinnedHostBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: *mut RawStream,
        out_copy: *mut *mut RawCopy,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_command_batch_copy_h2d_async(
        destination: *mut RawDeviceBuffer,
        destination_offset: u64,
        source: *mut RawPinnedHostBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_copy_d2h_async(
        destination: *mut RawPinnedHostBuffer,
        destination_offset: u64,
        source: *mut RawDeviceBuffer,
        source_offset: u64,
        byte_len: u64,
        stream: *mut RawStream,
        out_copy: *mut *mut RawCopy,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_copy_query(
        copy: *mut RawCopy,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_copy_synchronize(
        copy: *mut RawCopy,
        out_complete: *mut u8,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_copy_close(copy: *mut *mut RawCopy, error: *mut ErrorInfo) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn riley_cuda_test_memory_fault_reset(context: *mut RawContext, error: *mut ErrorInfo) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn riley_cuda_test_memory_fault_arm(
        context: *mut RawContext,
        fault: u32,
        error: *mut ErrorInfo,
    ) -> i32;
    #[cfg(feature = "cuda-test-fault-injection")]
    fn riley_cuda_test_memory_fault_stats(
        context: *mut RawContext,
        out_stats: *mut RawTestMemoryFaultStats,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_embedding_execute(
        params: *const RawEmbeddingParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_rms_norm_execute(
        params: *const RawRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hugging_face_smollm2_rms_norm_execute(
        params: *const RawRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_rms_norm_execute(
        params: *const RawRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_residual_add_execute(
        params: *const RawResidualAddParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_residual_rms_norm_execute(
        params: *const RawResidualRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hugging_face_smollm2_residual_rms_norm_execute(
        params: *const RawResidualRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_residual_rms_norm_execute(
        params: *const RawResidualRmsNormParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_log_softmax_execute(
        params: *const RawFixed37LogSoftmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_row_bias_add_in_place_execute(
        params: *const RawRowBiasAddInPlaceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_silu_execute(
        params: *const RawSiluParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gated_multiply_execute(
        params: *const RawGatedMultiplyParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_rope_table_execute(
        params: *const RawRopeTableParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_rope_execute(
        params: *const RawRopeParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_indexed_rope_execute(
        params: *const RawIndexedRopeParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_cast_execute(
        params: *const RawCastParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_row_gather_execute(
        params: *const RawRowGatherParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_bf16_argmax_execute(
        params: *const RawBf16ArgmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_qk_gqa_execute(
        params: *const RawQkGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_scale_causal_mask_in_place_execute(
        params: *const RawScaleCausalMaskParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_causal_softmax_in_place_execute(
        params: *const RawCausalSoftmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_av_gqa_execute(
        params: *const RawAvGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_qk_gqa_execute(
        params: *const RawQkGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_causal_softmax_in_place_execute(
        params: *const RawCausalSoftmaxParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_av_gqa_execute(
        params: *const RawAvGqaParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_prefill_attention_execute(
        params: *const RawPrefillAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_prefill_attention_execute(
        params: *const RawPrefillAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_kv_cache_write_execute(
        params: *const RawKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_decode_attention_reference_execute(
        params: *const RawDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_decode_attention_reference_execute(
        params: *const RawDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_decode_attention_two_pass_execute(
        params: *const RawDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_decode_attention_execute(
        params: *const RawDecodeAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_decode_partial_state_reduce_execute(
        params: *const RawDecodePartialStateReduceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_paged_kv_cache_write_execute(
        params: *const RawPagedKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_paged_decode_attention_reference_execute(
        params: *const RawPagedDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_paged_decode_attention_reference_execute(
        params: *const RawPagedDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_paged_decode_attention_two_pass_execute(
        params: *const RawPagedDecodeAttentionReferenceParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_paged_decode_attention_execute(
        params: *const RawPagedDecodeAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_ragged_paged_kv_cache_write_execute(
        params: *const RawRaggedPagedKvCacheWriteParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_ragged_paged_attention_execute(
        params: *const RawRaggedPagedAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_ragged_paged_attention_grouped_heads_execute(
        params: *const RawRaggedPagedAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hf_prefill_attention_plan_create(
        context: *mut RawContext,
        config: *const RawHfPrefillAttentionConfig,
        out_plan: *mut *mut RawHfPrefillAttentionPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hf_prefill_attention_plan_info(
        plan: *mut RawHfPrefillAttentionPlan,
        out_info: *mut RawHfPrefillAttentionPlanInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hf_prefill_attention_plan_execute(
        plan: *mut RawHfPrefillAttentionPlan,
        query: *const RawBufferSpan,
        key: *const RawBufferSpan,
        value: *const RawBufferSpan,
        output: *const RawBufferSpan,
        workspace: *const RawBufferSpan,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hf_prefill_attention_plan_close(
        plan: *mut *mut RawHfPrefillAttentionPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_hf_prefill_attention_plan_defer_to_active_capture(
        plan: *mut *mut RawHfPrefillAttentionPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_ragged_paged_attention_two_pass_execute(
        params: *const RawFixed37RaggedPagedAttentionParams,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gemm_plan_create(
        context: *mut RawContext,
        config: *const RawGemmConfig,
        out_plan: *mut *mut RawGemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gemm_plan_create_anchored(
        context: *mut RawContext,
        config: *const RawGemmConfig,
        anchor_plan: *mut RawGemmPlan,
        out_plan: *mut *mut RawGemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gemm_plan_info(
        plan: *mut RawGemmPlan,
        out_info: *mut RawGemmAlgorithmInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gemm_plan_execute(
        plan: *mut RawGemmPlan,
        input: *const RawBufferSpan,
        weight: *const RawBufferSpan,
        output: *const RawBufferSpan,
        workspace: *const RawBufferSpan,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_gemm_plan_close(plan: *mut *mut RawGemmPlan, error: *mut ErrorInfo) -> i32;
    fn riley_cuda_gemm_plan_defer_to_active_capture(
        plan: *mut *mut RawGemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_gemm_plan_create(
        context: *mut RawContext,
        config: *const RawGemmConfig,
        out_plan: *mut *mut RawFixed37GemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_gemm_plan_info(
        plan: *mut RawFixed37GemmPlan,
        out_info: *mut RawFixed37GemmPlanInfo,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_gemm_plan_execute(
        plan: *mut RawFixed37GemmPlan,
        input: *const RawBufferSpan,
        weight: *const RawBufferSpan,
        output: *const RawBufferSpan,
        stream: *mut RawStream,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_fixed37_gemm_plan_close(
        plan: *mut *mut RawFixed37GemmPlan,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_smoke_buffer_create(
        context: *mut RawContext,
        element_count: u64,
        out_buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_smoke_fill_launch(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        value: f32,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_smoke_copy_to_host(
        buffer: *mut RawSmokeBuffer,
        stream: *mut RawStream,
        host_output: *mut f32,
        host_element_capacity: u64,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_smoke_buffer_close(
        buffer: *mut *mut RawSmokeBuffer,
        error: *mut ErrorInfo,
    ) -> i32;
    fn riley_cuda_smoke_invalid_launch(stream: *mut RawStream, error: *mut ErrorInfo) -> i32;
}

pub(super) fn abi_version() -> u32 {
    // SAFETY: the statically linked metadata function takes no arguments and
    // returns a fixed-width value defined by the checked C header.
    unsafe { riley_cuda_abi_version() }
}

pub(super) fn build_info() -> CudaResult<String> {
    // SAFETY: the native ABI returns null or a process-lifetime C string.
    let pointer = unsafe { riley_cuda_build_info() };
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

#[cfg(feature = "nvml")]
pub(super) fn nvidia_environment_snapshot() -> CudaResult<NativeNvidiaEnvironmentSnapshot> {
    let mut snapshot = RawNvidiaEnvironmentSnapshot::new();
    let mut error = ErrorInfo::new();
    // SAFETY: snapshot and error are initialized, correctly sized repr(C)
    // caller buffers kept alive for the complete synchronous probe.
    let status = unsafe { riley_cuda_nvidia_environment_probe(&mut snapshot, &mut error) };
    status_result(status, "probe NVIDIA environment", &error)?;

    decode_nvidia_environment_snapshot(&snapshot)
}

#[cfg(feature = "nvml")]
fn decode_nvidia_environment_snapshot(
    snapshot: &RawNvidiaEnvironmentSnapshot,
) -> CudaResult<NativeNvidiaEnvironmentSnapshot> {
    if snapshot.struct_size != NVIDIA_ENVIRONMENT_SNAPSHOT_SIZE {
        return Err(invalid_nvidia_snapshot(format!(
            "native environment struct_size is {}, expected {NVIDIA_ENVIRONMENT_SNAPSHOT_SIZE}",
            snapshot.struct_size
        )));
    }
    if snapshot.reserved != [0; 2] {
        return Err(invalid_nvidia_snapshot(
            "native environment reserved fields are non-zero",
        ));
    }
    if snapshot.cuda_driver_api_version <= 0 {
        return Err(invalid_nvidia_snapshot(
            "native CUDA Driver API version is non-positive",
        ));
    }

    let driver_version = fixed_c_string_to_utf8(&snapshot.driver_version, "NVIDIA driver version")?;
    if driver_version.is_empty() {
        return Err(invalid_nvidia_snapshot(
            "native NVIDIA driver version is empty",
        ));
    }

    let device_count = usize::try_from(snapshot.device_count)
        .map_err(|_| invalid_nvidia_snapshot("native NVIDIA device count does not fit usize"))?;
    if device_count > NVIDIA_ENVIRONMENT_MAX_DEVICES {
        return Err(invalid_nvidia_snapshot(format!(
            "native NVIDIA device count {device_count} exceeds capacity {NVIDIA_ENVIRONMENT_MAX_DEVICES}"
        )));
    }

    let mut devices = Vec::with_capacity(device_count);
    let mut aggregate_process_count = 0_u32;
    for (expected_index, raw) in (0_u32..).zip(&snapshot.devices[..device_count]) {
        let device = decode_nvidia_device_snapshot(raw, expected_index)?;
        aggregate_process_count = aggregate_process_count
            .checked_add(device.compute_process_count)
            .ok_or_else(|| {
                invalid_nvidia_snapshot("native aggregate compute-process count overflowed")
            })?;
        devices.push(device);
    }
    if aggregate_process_count != snapshot.compute_process_count {
        return Err(invalid_nvidia_snapshot(format!(
            "native aggregate compute-process count is {}, but devices sum to {aggregate_process_count}",
            snapshot.compute_process_count
        )));
    }

    Ok(NativeNvidiaEnvironmentSnapshot {
        driver_version,
        cuda_driver_api_version: snapshot.cuda_driver_api_version,
        compute_process_count: snapshot.compute_process_count,
        devices,
    })
}

#[cfg(feature = "nvml")]
fn decode_nvidia_device_snapshot(
    raw: &RawNvidiaDeviceSnapshot,
    expected_index: u32,
) -> CudaResult<NativeNvidiaDeviceSnapshot> {
    if raw.struct_size != NVIDIA_DEVICE_SNAPSHOT_SIZE {
        return Err(invalid_nvidia_snapshot(format!(
            "native device {} struct_size is {}, expected {NVIDIA_DEVICE_SNAPSHOT_SIZE}",
            raw.index, raw.struct_size
        )));
    }
    if raw.index != expected_index {
        return Err(invalid_nvidia_snapshot(format!(
            "native device index {} is out of order; expected {expected_index}",
            raw.index
        )));
    }
    if raw.reserved != [0; 2] {
        return Err(invalid_nvidia_snapshot(format!(
            "native device {} reserved fields are non-zero",
            raw.index
        )));
    }
    if raw.total_memory_bytes == 0 || raw.used_memory_bytes > raw.total_memory_bytes {
        return Err(invalid_nvidia_snapshot(format!(
            "native device {} returned inconsistent memory totals",
            raw.index
        )));
    }
    if !matches!(
        raw.persistence_mode,
        NVIDIA_PERSISTENCE_DISABLED | NVIDIA_PERSISTENCE_ENABLED
    ) {
        return Err(invalid_nvidia_snapshot(format!(
            "native device {} returned unknown persistence mode {}",
            raw.index, raw.persistence_mode
        )));
    }
    let name = fixed_c_string_to_utf8(&raw.name, "NVIDIA device name")?;
    if name.is_empty() {
        return Err(invalid_nvidia_snapshot(format!(
            "native device {} name is empty",
            raw.index
        )));
    }
    Ok(NativeNvidiaDeviceSnapshot {
        index: raw.index,
        name,
        total_memory_bytes: raw.total_memory_bytes,
        used_memory_bytes: raw.used_memory_bytes,
        temperature_c: raw.temperature_c,
        persistence_mode: raw.persistence_mode,
        power_limit_milliwatts: raw.power_limit_milliwatts,
        application_graphics_clock_mhz: (raw.application_graphics_clock_mhz
            != NVIDIA_CLOCK_NOT_AVAILABLE)
            .then_some(raw.application_graphics_clock_mhz),
        application_memory_clock_mhz: (raw.application_memory_clock_mhz
            != NVIDIA_CLOCK_NOT_AVAILABLE)
            .then_some(raw.application_memory_clock_mhz),
        compute_process_count: raw.compute_process_count,
    })
}

#[cfg(feature = "nvml")]
pub(super) fn diagnose_null_nvidia_environment_snapshot() -> CudaResult<()> {
    let mut error = ErrorInfo::new();
    // SAFETY: null is intentionally supplied to exercise validation before
    // NVML initialization; the native contract never dereferences it.
    let status = unsafe { riley_cuda_nvidia_environment_probe(ptr::null_mut(), &mut error) };
    status_result(status, "diagnose null NVIDIA environment output", &error)
}

pub(super) fn device_count() -> CudaResult<u32> {
    let mut count = 0;
    let mut error = ErrorInfo::new();
    // SAFETY: both output pointers refer to initialized, writable values for
    // the duration of the synchronous C call.
    let status = unsafe { riley_cuda_device_count(&mut count, &mut error) };
    status_result(status, "enumerate CUDA devices", &error)?;
    Ok(count)
}

pub(super) fn diagnose_null_device_count() -> CudaResult<()> {
    let mut error = ErrorInfo::new();
    // SAFETY: null is intentionally supplied to exercise the documented ABI
    // validation path; no memory is dereferenced by contract.
    let status = unsafe { riley_cuda_device_count(ptr::null_mut(), &mut error) };
    status_result(status, "diagnose null device-count output", &error)
}

pub(super) fn device_properties(ordinal: i32) -> CudaResult<NativeDeviceProperties> {
    let mut properties = RawDeviceProperties::new();
    let mut error = ErrorInfo::new();
    // SAFETY: properties and error are correctly sized repr(C) caller buffers.
    let status = unsafe { riley_cuda_device_properties(ordinal, &mut properties, &mut error) };
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
        let status = unsafe { riley_cuda_context_create(ordinal, &mut pointer, &mut error) };
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
        let status = unsafe { riley_cuda_context_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA context", &error)
    }

    pub(super) fn memory_info(&self) -> CudaResult<(u64, u64)> {
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the live context and both output buffers remain valid for the
        // synchronous native call.
        let status = unsafe {
            riley_cuda_context_memory_info(
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
            riley_cuda_context_allocation_stats(self.as_ptr(), &mut allocation_snapshot, &mut error)
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
        let status = unsafe { riley_cuda_test_memory_fault_reset(self.as_ptr(), &mut error) };
        status_result(status, "reset CUDA memory fault injector", &error)
    }

    #[cfg(feature = "cuda-test-fault-injection")]
    pub(super) fn arm_memory_fault(&self, fault: u32) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the native boundary validates the test-only fault id and the
        // live context/session identity.
        let status = unsafe { riley_cuda_test_memory_fault_arm(self.as_ptr(), fault, &mut error) };
        status_result(status, "arm CUDA memory fault injector", &error)
    }

    #[cfg(feature = "cuda-test-fault-injection")]
    pub(super) fn memory_fault_stats(&self) -> CudaResult<NativeTestMemoryFaultStats> {
        let mut stats = RawTestMemoryFaultStats::new();
        let mut error = ErrorInfo::new();
        // SAFETY: stats is a correctly sized repr(C) output and both it and the
        // live context remain valid for the synchronous snapshot.
        let status =
            unsafe { riley_cuda_test_memory_fault_stats(self.as_ptr(), &mut stats, &mut error) };
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
        // SAFETY: raw is this handle's unique owned pointer. During a safe
        // Rust graph capture, transfer that ownership to native abort cleanup
        // instead of issuing a CUDA context release from the capture thread.
        // The ordinary C close entry point remains retryable for raw callers.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_context_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_context_close(&mut raw, &mut error)
            }
        };
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
            unsafe { riley_cuda_stream_create(context.as_ptr(), &mut pointer, &mut error) };
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
        let status = unsafe { riley_cuda_stream_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA stream", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle.
        let status = unsafe { riley_cuda_stream_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA stream", &error)
    }

    pub(super) fn command_batch_begin(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle. The native
        // boundary validates lifecycle state before enabling command batching.
        let status = unsafe { riley_cuda_stream_command_batch_begin(self.as_ptr(), &mut error) };
        status_result(status, "begin CUDA stream command batch", &error)
    }

    pub(super) fn command_batch_end(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live stream handle. The native end
        // call owns completion and any fail-closed post-error lifecycle state.
        let status = unsafe { riley_cuda_stream_command_batch_end(self.as_ptr(), &mut error) };
        status_result(status, "end CUDA stream command batch", &error)
    }

    pub(super) fn begin_graph_capture(&mut self, mode: u32) -> CudaResult<GraphCaptureHandle> {
        const OPERATION: &str = "begin CUDA Graph capture";
        let mut capture = ptr::null_mut::<RawGraphCapture>();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the stream; every output points to a
        // correctly sized local record. A non-null output transfers exactly
        // one native capture owner to this FFI boundary, including the rare
        // deferred-error case that must be aborted before returning an error.
        let status = unsafe {
            riley_cuda_graph_capture_begin(
                self.as_ptr(),
                mode,
                &mut capture,
                &mut graph_error,
                &mut error,
            )
        };
        let decoded = decode_graph_failure_info(&graph_error);
        let pointer = NonNull::new(capture);

        if status == STATUS_SUCCESS {
            if let (Some(pointer), Ok(graph_failure)) = (pointer, decoded.as_ref()) {
                if graph_capture_begin_success_metadata_is_valid(&graph_error, graph_failure) {
                    return Ok(GraphCaptureHandle {
                        pointer: Some(pointer),
                    });
                }
            }
        }

        // Never return a decoder/status/contract error while silently dropping
        // a capture owner. CUDA may have entered capture while surfacing an
        // earlier asynchronous failure; aborting here either restores the
        // stream or intentionally strands the native lease fail-closed.
        let cleanup = pointer.map(|pointer| {
            let mut owner = GraphCaptureHandle {
                pointer: Some(pointer),
            };
            owner.abort()
        });
        let metadata_error = decoded.err();
        let native_error = if status == STATUS_SUCCESS {
            None
        } else {
            Some(
                status_result(status, OPERATION, &error)
                    .expect_err("a non-success native status must decode as an error"),
            )
        };
        if let Some(cleanup_error) = cleanup.and_then(Result::err) {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Close,
                cleanup_error.native_code(),
                OPERATION,
                format!(
                    "native begin did not yield an acceptable owner and abort recovery also failed: {cleanup_error}"
                ),
            ));
        }
        if let Some(metadata_error) = metadata_error {
            return Err(metadata_error);
        }
        if let Some(native_error) = native_error {
            return Err(native_error);
        }
        Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Prepare,
            0,
            OPERATION,
            "native graph capture returned success without a valid owned capture handle",
        ))
    }

    pub(super) fn begin_graph_fill_capture(
        &mut self,
        buffer: &DeviceBufferHandle,
        element_count: u64,
        mode: u32,
    ) -> CudaResult<GraphCaptureHandle> {
        const OPERATION: &str = "begin CUDA Graph f32 fill capture";
        let mut capture = ptr::null_mut::<RawGraphCapture>();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the safe graph owner exclusively borrows the stream and
        // device buffer for the entire graph lifetime. Native validates their
        // exact context, fixed byte range, and persistent graph-use leases
        // before it can enter CUDA capture.
        let status = unsafe {
            riley_cuda_graph_capture_begin_fill_f32(
                self.as_ptr(),
                buffer.as_ptr(),
                element_count,
                mode,
                &mut capture,
                &mut graph_error,
                &mut error,
            )
        };
        let decoded = decode_graph_failure_info(&graph_error);
        let pointer = NonNull::new(capture);

        if status == STATUS_SUCCESS {
            if let (Some(pointer), Ok(graph_failure)) = (pointer, decoded.as_ref()) {
                if graph_capture_begin_success_metadata_is_valid(&graph_error, graph_failure) {
                    return Ok(GraphCaptureHandle {
                        pointer: Some(pointer),
                    });
                }
            }
        }

        // A failed begin can still have entered capture after surfacing a
        // deferred CUDA error. Never let the buffer/stream borrow unwind while
        // silently abandoning that native owner: one abort attempt either
        // restores the native leases or deliberately retains them fail-closed.
        let cleanup = pointer.map(|pointer| {
            let mut owner = GraphCaptureHandle {
                pointer: Some(pointer),
            };
            owner.abort()
        });
        let metadata_error = decoded.err();
        let native_error = if status == STATUS_SUCCESS {
            None
        } else {
            Some(
                status_result(status, OPERATION, &error)
                    .expect_err("a non-success native status must decode as an error"),
            )
        };
        if let Some(cleanup_error) = cleanup.and_then(Result::err) {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Close,
                cleanup_error.native_code(),
                OPERATION,
                format!(
                    "native fill-capture begin did not yield an acceptable owner and abort recovery also failed: {cleanup_error}"
                ),
            ));
        }
        if let Some(metadata_error) = metadata_error {
            return Err(metadata_error);
        }
        if let Some(native_error) = native_error {
            return Err(native_error);
        }
        Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Prepare,
            0,
            OPERATION,
            "native graph fill capture returned success without a valid owned capture handle",
        ))
    }

    pub(super) fn wait_event(&mut self, event: &EventHandle) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: both native handles remain alive and the native ABI validates
        // that they belong to the same context.
        let status =
            unsafe { riley_cuda_stream_wait_event(self.as_ptr(), event.as_ptr(), &mut error) };
        status_result(status, "wait for CUDA event", &error)
    }

    pub(super) fn diagnose_invalid_launch(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the function intentionally issues a configuration-invalid
        // launch against this live stream and clears the launch error.
        let status = unsafe { riley_cuda_smoke_invalid_launch(self.as_ptr(), &mut error) };
        status_result(status, "diagnose invalid CUDA launch", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: the active-capture path transfers this unique owner to the
        // native capture's post-end cleanup queue; ordinary raw C close keeps
        // its existing retryable semantics.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_stream_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_stream_close(&mut raw, &mut error)
            }
        };
        self.pointer = NonNull::new(raw);
        status_result(status, "close CUDA stream", &error)
    }
}

impl Drop for StreamHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Private result for a one-shot native graph transition.
///
/// Native graph metadata is the only evidence that capture-owned deferred
/// Rust context leases may be released. Keep that evidence separate from the
/// public `CudaResult`: CUDA can report a non-success status after all native
/// resource release work is nonetheless known complete.
pub(super) struct GraphTransition<T> {
    pub(super) result: CudaResult<T>,
    pub(super) resource_release_known: bool,
    /// Whether native consumed the in/out owner. A non-null owner returned
    /// from a pre-CUDA validation failure remains retryable by its enclosing
    /// safe guard; an owner consumed by a CUDA lifecycle attempt never is.
    pub(super) owner_consumed: bool,
}

impl<T> GraphTransition<T> {
    fn consumed(result: CudaResult<T>, resource_release_known: bool) -> Self {
        Self {
            result,
            resource_release_known,
            owner_consumed: true,
        }
    }

    fn retained(result: CudaResult<T>) -> Self {
        Self {
            result,
            resource_release_known: false,
            owner_consumed: false,
        }
    }
}

/// One native capture owner. This remains private because the public graph
/// guard supplies the stream/buffer borrows and thread confinement required to
/// use it.
#[derive(Debug)]
pub(super) struct GraphCaptureHandle {
    pointer: Option<NonNull<RawGraphCapture>>,
}

impl GraphCaptureHandle {
    pub(super) fn enqueue_fill(&mut self, value: f32) -> CudaResult<()> {
        const OPERATION: &str = "enqueue CUDA Graph f32 fill";
        let Some(pointer) = self.pointer else {
            return Err(graph_owner_missing(OPERATION));
        };
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the public GraphFillCapture keeps this capture owner, its
        // exact stream, and its fixed device buffer exclusively borrowed.
        let status = unsafe {
            riley_cuda_graph_capture_enqueue_fill_f32(
                pointer.as_ptr(),
                value,
                &mut graph_error,
                &mut error,
            )
        };
        let graph_failure = decode_graph_failure_info(&graph_error)?;
        if !graph_capture_enqueue_metadata_is_valid(&graph_error, &graph_failure) {
            return Err(malformed_graph_metadata(OPERATION));
        }
        status_result(status, OPERATION, &error)
    }

    /// Ends an active capture and transfers a fully-known graph owner out of
    /// it. The input pointer is taken before FFI because a native end attempt
    /// is one-shot. Native validation before CUDA entry is allowed to return
    /// the raw owner unchanged; restore that owner so the enclosing safe guard
    /// can abort it during Drop rather than silently stranding a live capture.
    pub(super) fn end(&mut self) -> GraphTransition<GraphHandle> {
        const OPERATION: &str = "end CUDA Graph capture";
        let Some(pointer) = self.pointer.take() else {
            return GraphTransition::consumed(Err(graph_owner_missing(OPERATION)), false);
        };
        let mut raw = pointer.as_ptr();
        let mut graph = ptr::null_mut::<RawGraph>();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: this uniquely owns the capture. Native consumes it after an
        // end attempt and returns a graph only after ownership transfer is
        // known, otherwise retaining its native leases fail-closed.
        let status = unsafe {
            riley_cuda_graph_capture_end(&mut raw, &mut graph, &mut graph_error, &mut error)
        };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return GraphTransition::retained(if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native capture end returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            });
        }
        let graph_failure = match decode_graph_failure_info(&graph_error) {
            Ok(value) => value,
            Err(error) => return GraphTransition::consumed(Err(error), false),
        };
        if !graph_capture_end_metadata_is_valid(&graph_error, &graph_failure) {
            return GraphTransition::consumed(Err(malformed_graph_metadata(OPERATION)), false);
        }
        let resource_release_known = graph_resources_released(&graph_failure);
        if status != STATUS_SUCCESS {
            return GraphTransition::consumed(
                close_unreturned_graph(
                    graph,
                    non_success_status_error::<()>(status, OPERATION, &error)
                        .expect_err("a non-success native status must decode as an error"),
                ),
                resource_release_known,
            );
        }
        if !resource_release_known {
            return GraphTransition::consumed(
                close_unreturned_graph(graph, malformed_graph_metadata(OPERATION)),
                false,
            );
        }
        let Some(pointer) = NonNull::new(graph) else {
            return GraphTransition::consumed(
                Err(missing_output(
                    OPERATION,
                    "native captured graph handle is null",
                )),
                false,
            );
        };
        GraphTransition::consumed(
            Ok(GraphHandle {
                pointer: Some(pointer),
            }),
            true,
        )
    }

    /// Ends an active capture exactly once without exposing its graph. This
    /// takes the Rust pointer before FFI so a native CUDA lifecycle attempt can
    /// never cause Drop to retry it. A documented pre-attempt validation error
    /// restores the raw owner and remains retryable by the enclosing guard.
    pub(super) fn abort(&mut self) -> CudaResult<()> {
        self.abort_with_transition().result
    }

    pub(super) fn abort_with_transition(&mut self) -> GraphTransition<()> {
        const OPERATION: &str = "abort CUDA Graph capture";
        let Some(pointer) = self.pointer.take() else {
            return GraphTransition::consumed(Ok(()), true);
        };
        let mut raw = pointer.as_ptr();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: this handle is uniquely owned. The native ABI consumes the
        // in/out owner after an end attempt; a documented pre-attempt failure
        // leaves it non-null and is restored below.
        let status =
            unsafe { riley_cuda_graph_capture_abort(&mut raw, &mut graph_error, &mut error) };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return GraphTransition::retained(if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native capture abort returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            });
        }
        let graph_failure = match decode_graph_failure_info(&graph_error) {
            Ok(value) => value,
            Err(error) => return GraphTransition::consumed(Err(error), false),
        };
        if !graph_capture_abort_metadata_is_valid(&graph_error, &graph_failure, status) {
            return GraphTransition::consumed(Err(malformed_graph_metadata(OPERATION)), false);
        }
        GraphTransition::consumed(
            status_result(status, OPERATION, &error),
            graph_resources_released(&graph_failure),
        )
    }
}

impl Drop for GraphCaptureHandle {
    fn drop(&mut self) {
        let _ = self.abort();
    }
}

/// Captured graph owner. It is consumed by instantiate or one-shot close.
pub(super) struct GraphHandle {
    pointer: Option<NonNull<RawGraph>>,
}

impl GraphHandle {
    pub(super) fn instantiate(&mut self) -> CudaResult<GraphExecHandle> {
        const OPERATION: &str = "instantiate CUDA Graph";
        let Some(pointer) = self.pointer.take() else {
            return Err(graph_owner_missing(OPERATION));
        };
        let mut raw = pointer.as_ptr();
        let mut exec = ptr::null_mut::<RawGraphExec>();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the safe CapturedGraph uniquely owns the graph and keeps its
        // retained stream/buffer borrows alive. Native consumes graph ownership
        // after a CUDA instantiate attempt; a pre-attempt validation error
        // returns it unchanged and is restored below.
        let status = unsafe {
            riley_cuda_graph_instantiate(&mut raw, &mut exec, &mut graph_error, &mut error)
        };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native graph instantiate returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            };
        }
        let graph_failure = match decode_graph_failure_info(&graph_error) {
            Ok(value) => value,
            Err(error) => return close_unreturned_graph_exec(exec, error),
        };
        if !graph_instantiate_metadata_is_valid(&graph_error, &graph_failure) {
            return close_unreturned_graph_exec(exec, malformed_graph_metadata(OPERATION));
        }
        if status != STATUS_SUCCESS {
            return close_unreturned_graph_exec(
                exec,
                non_success_status_error::<()>(status, OPERATION, &error)
                    .expect_err("a non-success native status must decode as an error"),
            );
        }
        if graph_failure.poisoned() {
            return close_unreturned_graph_exec(exec, malformed_graph_metadata(OPERATION));
        }
        let pointer = NonNull::new(exec)
            .ok_or_else(|| missing_output(OPERATION, "native graph exec handle is null"))?;
        Ok(GraphExecHandle {
            pointer: Some(pointer),
        })
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "close CUDA Graph";
        let Some(pointer) = self.pointer.take() else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: graph close is one-shot after CUDA destruction is attempted.
        // A documented validation failure leaves the owner unchanged and is
        // restored below for the enclosing safe owner to clean up on Drop.
        let status = unsafe { riley_cuda_graph_close(&mut raw, &mut graph_error, &mut error) };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native graph close returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            };
        }
        let graph_failure = decode_graph_failure_info(&graph_error)?;
        if !graph_close_metadata_is_valid(&graph_error, &graph_failure, false)
            || (status == STATUS_SUCCESS
                && (!graph_resources_released(&graph_failure)
                    || graph_failure.submission_started()
                    || graph_failure.completion_known()))
        {
            return Err(malformed_graph_metadata(OPERATION));
        }
        status_result(status, OPERATION, &error)
    }
}

impl Drop for GraphHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Instantiated CUDA Graph owner. Launch is separate so an in-flight launch
/// can borrow both this owner and the capture stream until completion.
pub(super) struct GraphExecHandle {
    pointer: Option<NonNull<RawGraphExec>>,
}

impl GraphExecHandle {
    pub(super) fn launch(&mut self, stream: &mut StreamHandle) -> CudaResult<GraphLaunchHandle> {
        const OPERATION: &str = "launch CUDA Graph exec";
        let Some(pointer) = self.pointer else {
            return Err(graph_owner_missing(OPERATION));
        };
        let mut launch = ptr::null_mut::<RawGraphLaunch>();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: GraphExec and the exact captured stream remain uniquely
        // borrowed by GraphLaunch. Native rejects any foreign stream before
        // issuing cudaGraphLaunch.
        let status = unsafe {
            riley_cuda_graph_exec_launch(
                pointer.as_ptr(),
                stream.as_ptr(),
                &mut launch,
                &mut graph_error,
                &mut error,
            )
        };
        let graph_failure = match decode_graph_failure_info(&graph_error) {
            Ok(value) => value,
            Err(error) => return settle_unreturned_graph_launch(launch, error),
        };
        if !graph_exec_launch_metadata_is_valid(&graph_error, &graph_failure) {
            return settle_unreturned_graph_launch(launch, malformed_graph_metadata(OPERATION));
        }
        if status != STATUS_SUCCESS {
            return settle_unreturned_graph_launch(
                launch,
                non_success_status_error::<()>(status, OPERATION, &error)
                    .expect_err("a non-success native status must decode as an error"),
            );
        }
        if graph_failure.poisoned() || !graph_failure.submission_started() {
            return settle_unreturned_graph_launch(launch, malformed_graph_metadata(OPERATION));
        }
        let Some(pointer) = NonNull::new(launch) else {
            // A successful submission without its required completion owner is
            // an ABI violation. Never let the safe wrapper issue another
            // launch or close against an exec whose in-flight state is unknown.
            self.pointer = None;
            return Err(missing_output(
                OPERATION,
                "native graph launch owner is null",
            ));
        };
        Ok(GraphLaunchHandle {
            pointer: Some(pointer),
        })
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "close CUDA Graph exec";
        let Some(pointer) = self.pointer.take() else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: graph-exec close is one-shot after CUDA destruction is
        // attempted. A documented pre-attempt validation rejection leaves the
        // owner unchanged and is restored below.
        let status = unsafe { riley_cuda_graph_exec_close(&mut raw, &mut graph_error, &mut error) };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native graph exec close returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            };
        }
        let graph_failure = decode_graph_failure_info(&graph_error)?;
        if !graph_close_metadata_is_valid(&graph_error, &graph_failure, true)
            || (status == STATUS_SUCCESS
                && (!graph_resources_released(&graph_failure)
                    || graph_failure.submission_started()
                    || graph_failure.completion_known()))
        {
            return Err(malformed_graph_metadata(OPERATION));
        }
        status_result(status, OPERATION, &error)
    }
}

impl Drop for GraphExecHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// A consuming safe transition cannot return a native graph alongside an
/// error. If native did produce a valid output owner after a deferred failure,
/// close it synchronously so its permanent stream/buffer leases do not become
/// an accidental invisible leak.
fn close_unreturned_graph(
    raw_graph: *mut RawGraph,
    transition_error: CudaError,
) -> CudaResult<GraphHandle> {
    let Some(pointer) = NonNull::new(raw_graph) else {
        return Err(transition_error);
    };
    let mut graph = GraphHandle {
        pointer: Some(pointer),
    };
    match graph.close() {
        Ok(()) => Err(transition_error),
        Err(close_error) => Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Close,
            close_error.native_code(),
            "end CUDA Graph capture",
            format!(
                "native capture end did not yield a safe graph result ({transition_error}) and mandatory graph-close recovery also failed ({close_error}); native leases are retained fail-closed"
            ),
        )),
    }
}

/// Mirrors [`close_unreturned_graph`] for a graph exec returned with an
/// otherwise failing instantiate transition.
fn close_unreturned_graph_exec(
    raw_exec: *mut RawGraphExec,
    transition_error: CudaError,
) -> CudaResult<GraphExecHandle> {
    let Some(pointer) = NonNull::new(raw_exec) else {
        return Err(transition_error);
    };
    let mut exec = GraphExecHandle {
        pointer: Some(pointer),
    };
    match exec.close() {
        Ok(()) => Err(transition_error),
        Err(close_error) => Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Close,
            close_error.native_code(),
            "instantiate CUDA Graph",
            format!(
                "native graph instantiate did not yield a safe exec result ({transition_error}) and mandatory exec-close recovery also failed ({close_error}); native leases are retained fail-closed"
            ),
        )),
    }
}

/// One completion owner for a submitted graph launch.
pub(super) struct GraphLaunchHandle {
    pointer: Option<NonNull<RawGraphLaunch>>,
}

impl GraphLaunchHandle {
    pub(super) fn complete(&mut self) -> CudaResult<()> {
        const OPERATION: &str = "complete CUDA Graph launch";
        let Some(pointer) = self.pointer.take() else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut graph_error = RawGraphErrorInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: this is the unique native completion owner. Native consumes
        // it after one CUDA completion attempt and retains graph leases on an
        // ambiguous result. A pre-attempt validation failure returns it
        // unchanged and is restored below.
        let status =
            unsafe { riley_cuda_graph_launch_complete(&mut raw, &mut graph_error, &mut error) };
        if let Some(pointer) = NonNull::new(raw) {
            self.pointer = Some(pointer);
            return if status == STATUS_SUCCESS {
                Err(CudaError::new(
                    CudaErrorKind::Internal,
                    CudaErrorDomain::Internal,
                    CudaErrorStage::Close,
                    0,
                    OPERATION,
                    "native graph completion returned success while retaining its input owner",
                ))
            } else {
                non_success_status_error(status, OPERATION, &error)
            };
        }
        let graph_failure = decode_graph_failure_info(&graph_error)?;
        if !graph_launch_complete_metadata_is_valid(&graph_error, &graph_failure)
            || (status == STATUS_SUCCESS
                && (!graph_failure.submission_started()
                    || !graph_failure.completion_known()
                    || !graph_resources_released(&graph_failure)))
        {
            return Err(malformed_graph_metadata(OPERATION));
        }
        status_result(status, OPERATION, &error)
    }
}

impl Drop for GraphLaunchHandle {
    fn drop(&mut self) {
        let _ = self.complete();
    }
}

/// A safe `GraphExec::launch` cannot return a completion owner alongside an
/// error. If native reports one anyway, settle it synchronously before
/// returning so an otherwise recoverable launch error cannot strand the exec
/// and its stream/buffer leases forever. A failed settlement is deliberately
/// reported as an internal close error; native then keeps the exec fail-closed.
fn settle_unreturned_graph_launch(
    raw_launch: *mut RawGraphLaunch,
    launch_error: CudaError,
) -> CudaResult<GraphLaunchHandle> {
    let Some(pointer) = NonNull::new(raw_launch) else {
        return Err(launch_error);
    };
    let mut launch = GraphLaunchHandle {
        pointer: Some(pointer),
    };
    match launch.complete() {
        Ok(()) => Err(launch_error),
        Err(completion_error) => Err(CudaError::new(
            CudaErrorKind::Internal,
            CudaErrorDomain::Internal,
            CudaErrorStage::Close,
            completion_error.native_code(),
            "launch CUDA Graph exec",
            format!(
                "native graph launch failed ({launch_error}) and mandatory completion recovery also failed ({completion_error}); the graph exec is retained fail-closed"
            ),
        )),
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
        let status = unsafe { riley_cuda_event_create(context.as_ptr(), &mut pointer, &mut error) };
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
        let status = unsafe { riley_cuda_event_record(self.as_ptr(), stream.as_ptr(), &mut error) };
        status_result(status, "record CUDA event", &error)
    }

    pub(super) fn query(&mut self) -> CudaResult<bool> {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: event and output remain live for the synchronous call.
        let status = unsafe { riley_cuda_event_query(self.as_ptr(), &mut complete, &mut error) };
        query_result(status, complete, "query CUDA event", &error)
    }

    pub(super) fn synchronize(&mut self) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: self uniquely owns the live event handle.
        let status = unsafe { riley_cuda_event_synchronize(self.as_ptr(), &mut error) };
        status_result(status, "synchronize CUDA event", &error)
    }

    pub(super) fn elapsed_ms(&self, end: &Self) -> CudaResult<f32> {
        let mut elapsed = 0.0;
        let mut error = ErrorInfo::new();
        // SAFETY: both event handles and output remain valid; native validates
        // context identity and recording/completion state.
        let status = unsafe {
            riley_cuda_event_elapsed_ms(self.as_ptr(), end.as_ptr(), &mut elapsed, &mut error)
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
        // SAFETY: see StreamHandle::close; graph-active safe owners move to
        // native cleanup rather than calling CUDA from the capture body.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_event_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_event_close(&mut raw, &mut error)
            }
        };
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
            riley_cuda_device_buffer_create(context.as_ptr(), byte_len, &mut pointer, &mut error)
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

    pub(super) fn copy_from_pinned_in_command_batch(
        &self,
        destination_offset: u64,
        source: &PinnedHostBufferHandle,
        source_offset: u64,
        byte_len: u64,
        stream: &mut StreamHandle,
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the safe command-batch guard exclusively borrows the stream;
        // both opaque buffers remain borrowed for this submission. Native
        // validation registers their active-use counters in the owning batch
        // before enqueuing the fixed-direction H2D operation.
        let status = unsafe {
            riley_cuda_command_batch_copy_h2d_async(
                self.as_ptr(),
                destination_offset,
                source.as_ptr(),
                source_offset,
                byte_len,
                stream.as_ptr(),
                &mut error,
            )
        };
        status_result(
            status,
            "enqueue command-batch pinned host-to-device copy",
            &error,
        )
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: graph-active safe Drop transfers this raw owner to the
        // capture's native post-end queue. Outside capture, ordinary close
        // preserves its single-shot/ambiguous-destruction contract.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_device_buffer_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_device_buffer_close(&mut raw, &mut error)
            }
        };
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
            riley_cuda_pinned_host_buffer_create(
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
            riley_cuda_pinned_host_buffer_write(
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
            riley_cuda_pinned_host_buffer_read(
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
        // SAFETY: native validates deferred transfer before consuming the raw
        // owner. Direct close retains its existing active-copy and retry
        // behavior for non-capture callers.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_pinned_host_buffer_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_pinned_host_buffer_close(&mut raw, &mut error)
            }
        };
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
            riley_cuda_copy_h2d_async(
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
            riley_cuda_copy_d2h_async(
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
        let status = unsafe { riley_cuda_copy_query(self.as_ptr(), &mut complete, &mut error) };
        copy_completion(status, complete, "query CUDA copy", &error)
    }

    pub(super) fn synchronize(&mut self) -> CopyCompletion {
        let mut complete = 0;
        let mut error = ErrorInfo::new();
        // SAFETY: the owned token and all resources retained by native active
        // use remain valid until out_complete confirms release.
        let status =
            unsafe { riley_cuda_copy_synchronize(self.as_ptr(), &mut complete, &mut error) };
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
        let status = unsafe { riley_cuda_copy_close(&mut raw, &mut error) };
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
    let status = unsafe { riley_cuda_embedding_execute(&params, stream.as_ptr(), &mut error) };
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
        unsafe { riley_cuda_rms_norm_execute(&params, stream, error) }
    })
}

pub(super) fn hugging_face_smollm2_rms_norm_execute(
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
    primitive_status(
        "execute Hugging Face SmolLM2 CUDA RMSNorm",
        stream,
        |stream, error| {
            // SAFETY: the reviewed descriptor and every borrowed opaque
            // resource remain live through synchronous native completion.
            unsafe { riley_cuda_hugging_face_smollm2_rms_norm_execute(&params, stream, error) }
        },
    )
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
        unsafe { riley_cuda_fixed37_rms_norm_execute(&params, stream, error) }
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
        unsafe { riley_cuda_residual_add_execute(&params, stream, error) }
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
            unsafe { riley_cuda_residual_rms_norm_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn hugging_face_smollm2_residual_rms_norm_execute(
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
        "execute Hugging Face SmolLM2 fused residual CUDA RMSNorm",
        stream,
        |stream, error| {
            // SAFETY: the reviewed descriptor and all exclusively borrowed
            // resources outlive this synchronously completing native call.
            unsafe {
                riley_cuda_hugging_face_smollm2_residual_rms_norm_execute(&params, stream, error)
            }
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
            unsafe { riley_cuda_fixed37_residual_rms_norm_execute(&params, stream, error) }
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
            unsafe { riley_cuda_fixed37_log_softmax_execute(&params, stream, error) }
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
            unsafe { riley_cuda_row_bias_add_in_place_execute(&params, stream, error) }
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
        unsafe { riley_cuda_silu_execute(&params, stream, error) }
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
        unsafe { riley_cuda_gated_multiply_execute(&params, stream, error) }
    })
}

pub(super) fn rope_table_execute(
    angles_cos: RawBufferSpan,
    sin: RawBufferSpan,
    element_count: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawRopeTableParams {
        struct_size: ROPE_TABLE_PARAMS_SIZE,
        reserved0: 0,
        angles_cos,
        sin,
        element_count,
        reserved: [0; 5],
    };
    primitive_status("prepare CUDA RoPE tables", stream, |stream, error| {
        // SAFETY: params outlives submission, while native active-use leases
        // retain both buffers through synchronous completion or command-batch
        // completion.
        unsafe { riley_cuda_rope_table_execute(&params, stream, error) }
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
        unsafe { riley_cuda_rope_execute(&params, stream, error) }
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
        unsafe { riley_cuda_indexed_rope_execute(&params, stream, error) }
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
        unsafe { riley_cuda_cast_execute(&params, stream, error) }
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
        unsafe { riley_cuda_row_gather_execute(&params, stream, error) }
    })
}

pub(super) fn bf16_argmax_execute(
    logits: RawBufferSpan,
    results: RawBufferSpan,
    row_count: u64,
    vocabulary_size: u64,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawBf16ArgmaxParams {
        struct_size: BF16_ARGMAX_PARAMS_SIZE,
        reserved0: 0,
        logits,
        results,
        row_count,
        vocabulary_size,
        reserved: [0; 4],
    };
    primitive_status(
        "execute deterministic CUDA BF16 argmax",
        stream,
        |stream, error| {
            // SAFETY: params and both borrowed opaque resources remain live
            // for the synchronously completing native operation.
            unsafe { riley_cuda_bf16_argmax_execute(&params, stream, error) }
        },
    )
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
        unsafe { riley_cuda_qk_gqa_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_qk_gqa_execute(
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
    primitive_status("execute fixed37 CUDA QK GQA", stream, |stream, error| {
        // SAFETY: the descriptor and borrowed native handles remain live until
        // the synchronous fixed37 operation completes.
        unsafe { riley_cuda_fixed37_qk_gqa_execute(&params, stream, error) }
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
            unsafe { riley_cuda_scale_causal_mask_in_place_execute(&params, stream, error) }
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
        unsafe { riley_cuda_causal_softmax_in_place_execute(&params, stream, error) }
    })
}

pub(super) fn fixed37_causal_softmax_in_place_execute(
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
    primitive_status(
        "execute fixed37 CUDA causal softmax",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and exclusively borrowed score buffer remain
            // live until the synchronous native operation completes.
            unsafe { riley_cuda_fixed37_causal_softmax_in_place_execute(&params, stream, error) }
        },
    )
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
        unsafe { riley_cuda_av_gqa_execute(&params, stream, error) }
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_av_gqa_execute(
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
    primitive_status("execute fixed37 CUDA AV GQA", stream, |stream, error| {
        // SAFETY: the descriptor and borrowed native handles remain live until
        // the synchronous fixed37 operation completes.
        unsafe { riley_cuda_fixed37_av_gqa_execute(&params, stream, error) }
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
            unsafe { riley_cuda_prefill_attention_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_prefill_attention_execute(
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
        "execute fixed37 CUDA two-pass prefill attention",
        stream,
        |stream, error| {
            // SAFETY: the descriptor and all opaque resources remain live until
            // the synchronous native operation completes.
            unsafe { riley_cuda_fixed37_prefill_attention_execute(&params, stream, error) }
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
        unsafe { riley_cuda_kv_cache_write_execute(&params, stream, error) }
    })
}

const fn unused_bf16_span() -> RawBufferSpan {
    RawBufferSpan {
        struct_size: BUFFER_SPAN_SIZE,
        dtype: DTYPE_BF16,
        buffer: ptr::null_mut(),
        byte_offset: 0,
        byte_len: 0,
        reserved: [0; 2],
    }
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
            unsafe { riley_cuda_decode_attention_reference_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_decode_attention_reference_execute(
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
        "execute fixed37 CUDA materialized decode attention",
        stream,
        |stream, error| {
            // SAFETY: the reused reference descriptor and every opaque resource
            // remain live through synchronous fixed37 native completion.
            unsafe { riley_cuda_fixed37_decode_attention_reference_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_decode_attention_two_pass_execute(
    query: RawBufferSpan,
    key_cache: RawBufferSpan,
    value_cache: RawBufferSpan,
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
        score_workspace: unused_bf16_span(),
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
        "execute fixed37 CUDA two-pass decode attention",
        stream,
        |stream, error| {
            // SAFETY: native deliberately ignores score_workspace and keeps
            // every real borrowed resource live through synchronous completion.
            unsafe { riley_cuda_fixed37_decode_attention_two_pass_execute(&params, stream, error) }
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
            unsafe { riley_cuda_decode_attention_execute(&params, stream, error) }
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
            unsafe { riley_cuda_decode_partial_state_reduce_execute(&params, stream, error) }
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
        unsafe { riley_cuda_paged_kv_cache_write_execute(&params, stream, error) }
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
            unsafe { riley_cuda_paged_decode_attention_reference_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_paged_decode_attention_reference_execute(
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
        "execute fixed37 CUDA materialized paged decode attention",
        stream,
        |stream, error| {
            // SAFETY: the reused reference and page-table descriptors and every
            // borrowed resource remain live through synchronous completion.
            unsafe {
                riley_cuda_fixed37_paged_decode_attention_reference_execute(&params, stream, error)
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_paged_decode_attention_two_pass_execute(
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
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
        score_workspace: unused_bf16_span(),
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
        "execute fixed37 CUDA two-pass paged decode attention",
        stream,
        |stream, error| {
            // SAFETY: native ignores the zero workspace placeholder; the page
            // table and all real data buffers remain borrowed synchronously.
            unsafe {
                riley_cuda_fixed37_paged_decode_attention_two_pass_execute(&params, stream, error)
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
            unsafe { riley_cuda_paged_decode_attention_execute(&params, stream, error) }
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
            unsafe { riley_cuda_ragged_paged_kv_cache_write_execute(&params, stream, error) }
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
            unsafe { riley_cuda_ragged_paged_attention_execute(&params, stream, error) }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ragged_paged_attention_grouped_heads_execute(
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
        "execute CUDA grouped-head ragged paged attention",
        stream,
        |stream, error| {
            // SAFETY: the fixed-layout descriptor and all exclusively
            // borrowed buffers remain live through synchronous completion.
            unsafe {
                riley_cuda_ragged_paged_attention_grouped_heads_execute(&params, stream, error)
            }
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fixed37_ragged_paged_attention_two_pass_execute(
    query: RawBufferSpan,
    key_pool: RawBufferSpan,
    value_pool: RawBufferSpan,
    output: RawBufferSpan,
    batch: &PackedBatchRawV1,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    output_row_count: u64,
    maximum_logical_token_count: u64,
    scale: f32,
    stream: &mut StreamHandle,
) -> CudaResult<()> {
    let params = RawFixed37RaggedPagedAttentionParams {
        struct_size: FIXED37_RAGGED_PAGED_ATTENTION_PARAMS_SIZE,
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
        maximum_logical_token_count,
        scale,
        reserved1: 0,
        reserved: [0; 4],
    };
    primitive_status(
        "execute fixed37 CUDA ragged paged attention",
        stream,
        |stream, error| {
            // SAFETY: the safe caller keeps every borrowed tensor and packed
            // metadata resource live through submission. Native command-batch
            // mode retains registered resources until batch completion.
            unsafe {
                riley_cuda_fixed37_ragged_paged_attention_two_pass_execute(&params, stream, error)
            }
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

/// Runs fixed37 QK/softmax/AV for every dense batch while reusing one
/// `[QH,S,S]` BF16 workspace. Score scale and finite-minimum causal masking
/// intentionally reuse the canonical elementwise staging primitive.
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_attention_fixed37_materialized_execute(
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
    const OPERATION: &str = "execute fixed37 CUDA materialized prefill attention";
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

        fixed37_qk_gqa_execute(
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
        fixed37_causal_softmax_in_place_execute(workspace, token_count, query_head_count, stream)?;
        fixed37_av_gqa_execute(
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

pub(super) struct HfPrefillAttentionPlanHandle {
    pointer: Option<NonNull<RawHfPrefillAttentionPlan>>,
}

// SAFETY: native calls restore the retained CUDA context and serialize plan
// use. The safe owner requires `&mut self` for execution and remains !Sync.
unsafe impl Send for HfPrefillAttentionPlanHandle {}

impl HfPrefillAttentionPlanHandle {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn create(
        context: &ContextHandle,
        batch_count: u64,
        token_count: u64,
        query_head_count: u64,
        key_value_head_count: u64,
        head_size: u64,
        scale: f32,
        max_cublas_workspace_bytes: u64,
    ) -> CudaResult<Self> {
        let config = RawHfPrefillAttentionConfig::new(
            batch_count,
            token_count,
            query_head_count,
            key_value_head_count,
            head_size,
            scale,
            max_cublas_workspace_bytes,
        );
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: native copies the fixed-layout config synchronously and
        // retains the context iff it returns a non-null owning plan.
        let status = unsafe {
            riley_cuda_hf_prefill_attention_plan_create(
                context.as_ptr(),
                &config,
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "prepare HF cuBLASLt prefill attention", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output(
                "prepare HF cuBLASLt prefill attention",
                "native attention plan handle is null",
            )
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    fn as_ptr(&self) -> *mut RawHfPrefillAttentionPlan {
        self.pointer.map_or(ptr::null_mut(), NonNull::as_ptr)
    }

    pub(super) fn info(&self) -> CudaResult<NativeHfPrefillAttentionPlanInfo> {
        let mut info = RawHfPrefillAttentionPlanInfo::new();
        let mut error = ErrorInfo::new();
        // SAFETY: the plan remains owned and the output is correctly sized.
        let status = unsafe {
            riley_cuda_hf_prefill_attention_plan_info(self.as_ptr(), &mut info, &mut error)
        };
        status_result(status, "query HF prefill attention plan", &error)?;
        if info.struct_size != HF_PREFILL_ATTENTION_PLAN_INFO_SIZE
            || info.reserved0 != 0
            || info.qk_reserved0 != 0
            || info.av_reserved0 != 0
            || info.reserved != [0; 2]
        {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Prepare,
                0,
                "query HF prefill attention plan",
                "native attention metadata violates its ABI contract",
            ));
        }
        Ok(NativeHfPrefillAttentionPlanInfo {
            backend: info.backend,
            qk_algorithm_id: info.qk_algorithm_id,
            qk_tile_id: info.qk_tile_id,
            qk_stages_id: info.qk_stages_id,
            qk_split_k: info.qk_split_k,
            qk_reduction_scheme: info.qk_reduction_scheme,
            qk_cta_swizzling: info.qk_cta_swizzling,
            qk_custom_option: info.qk_custom_option,
            qk_workspace_bytes: info.qk_workspace_bytes,
            qk_numerical_implementation_flags: info.qk_numerical_implementation_flags,
            av_algorithm_id: info.av_algorithm_id,
            av_tile_id: info.av_tile_id,
            av_stages_id: info.av_stages_id,
            av_split_k: info.av_split_k,
            av_reduction_scheme: info.av_reduction_scheme,
            av_cta_swizzling: info.av_cta_swizzling,
            av_custom_option: info.av_custom_option,
            av_workspace_bytes: info.av_workspace_bytes,
            av_numerical_implementation_flags: info.av_numerical_implementation_flags,
            deterministic: info.deterministic,
            compute_capability_major: info.compute_capability_major,
            compute_capability_minor: info.compute_capability_minor,
            runtime_version: info.runtime_version,
            cublaslt_version: info.cublaslt_version,
            workspace_bytes: info.workspace_bytes,
            score_bytes: info.score_bytes,
            repeated_key_value_bytes: info.repeated_key_value_bytes,
            layout_copy_bytes: info.layout_copy_bytes,
            batch_count: info.batch_count,
            token_count: info.token_count,
            query_head_count: info.query_head_count,
            key_value_head_count: info.key_value_head_count,
            head_size: info.head_size,
        })
    }

    pub(super) fn execute(
        &mut self,
        query: RawBufferSpan,
        key: RawBufferSpan,
        value: RawBufferSpan,
        output: RawBufferSpan,
        workspace: RawBufferSpan,
        stream: &mut StreamHandle,
    ) -> CudaResult<()> {
        let mut error = ErrorInfo::new();
        // SAFETY: the safe layer uniquely borrows writable spans, plan, and
        // stream until native confirms synchronous completion.
        let status = unsafe {
            riley_cuda_hf_prefill_attention_plan_execute(
                self.as_ptr(),
                &query,
                &key,
                &value,
                &output,
                &workspace,
                stream.as_ptr(),
                &mut error,
            )
        };
        status_result(status, "execute HF cuBLASLt prefill attention", &error)
    }

    pub(super) fn close(&mut self) -> CudaResult<()> {
        let Some(pointer) = self.pointer else {
            return Ok(());
        };
        let mut raw = pointer.as_ptr();
        let mut error = ErrorInfo::new();
        // SAFETY: a live safe capture transfers this unique plan to native
        // post-end cleanup. Raw callers still use the unchanged close ABI.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_hf_prefill_attention_plan_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_hf_prefill_attention_plan_close(&mut raw, &mut error)
            }
        };
        self.pointer = NonNull::new(raw);
        status_result(status, "close HF cuBLASLt prefill attention", &error)
    }
}

impl Drop for HfPrefillAttentionPlanHandle {
    fn drop(&mut self) {
        let _ = self.close();
    }
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
        flags: u32,
    ) -> CudaResult<Self> {
        if flags & !(GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K | GEMM_FLAG_ALLOW_INPLACE_SPLIT_K) != 0 {
            return Err(CudaError::invalid_argument(
                "prepare CUDA GEMM plan",
                "unknown GEMM reduction-policy flags",
            ));
        }
        let config = RawGemmConfig::new(flags, m, n, k, max_workspace_bytes);
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: the context is retained by the native plan on success, the
        // fixed-layout config remains live, and both outputs are writable for
        // this synchronously completing preparation call.
        let status = unsafe {
            riley_cuda_gemm_plan_create(context.as_ptr(), &config, &mut pointer, &mut error)
        };
        status_result(status, "prepare CUDA GEMM plan", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output("prepare CUDA GEMM plan", "native GEMM plan handle is null")
        })?;
        Ok(Self {
            pointer: Some(pointer),
        })
    }

    pub(super) fn create_anchored(
        context: &ContextHandle,
        m: u64,
        n: u64,
        k: u64,
        max_workspace_bytes: u64,
        flags: u32,
        anchor: &Self,
    ) -> CudaResult<Self> {
        if flags & !(GEMM_FLAG_ALLOW_OUTPUT_TYPE_SPLIT_K | GEMM_FLAG_ALLOW_INPLACE_SPLIT_K) != 0 {
            return Err(CudaError::invalid_argument(
                "prepare anchored CUDA GEMM plan",
                "unknown GEMM reduction-policy flags",
            ));
        }
        let config = RawGemmConfig::new(flags, m, n, k, max_workspace_bytes);
        let anchor = anchor.pointer.ok_or_else(|| {
            CudaError::invalid_state(
                "prepare anchored CUDA GEMM plan",
                "the anchor GEMM plan has already been closed",
            )
        })?;
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: both plans are owned by this safe layer. Native borrows the
        // anchor only for this synchronous preparation call, validates context
        // and immutable GEMM compatibility, then copies its opaque algorithm
        // into the returned child without retaining the anchor pointer.
        let status = unsafe {
            riley_cuda_gemm_plan_create_anchored(
                context.as_ptr(),
                &config,
                anchor.as_ptr(),
                &mut pointer,
                &mut error,
            )
        };
        status_result(status, "prepare anchored CUDA GEMM plan", &error)?;
        let pointer = NonNull::new(pointer).ok_or_else(|| {
            missing_output(
                "prepare anchored CUDA GEMM plan",
                "native anchored GEMM plan handle is null",
            )
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
        let status = unsafe { riley_cuda_gemm_plan_info(self.as_ptr(), &mut info, &mut error) };
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
            riley_cuda_gemm_plan_execute(
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
        // SAFETY: native capture cleanup takes this unique owner only after
        // validating that it can later close it under the exact capture owner.
        // Ordinary C close remains the retryable raw API.
        let status = unsafe {
            if crate::graph::has_active_graph_capture() {
                riley_cuda_gemm_plan_defer_to_active_capture(&mut raw, &mut error)
            } else {
                riley_cuda_gemm_plan_close(&mut raw, &mut error)
            }
        };
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
        let config = RawGemmConfig::new(0, m, n, k, max_workspace_bytes);
        let mut pointer = ptr::null_mut();
        let mut error = ErrorInfo::new();
        // SAFETY: native retains the context on success and initializes the
        // owned output handle or leaves it null on failure.
        let status = unsafe {
            riley_cuda_fixed37_gemm_plan_create(context.as_ptr(), &config, &mut pointer, &mut error)
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
            unsafe { riley_cuda_fixed37_gemm_plan_info(self.as_ptr(), &mut info, &mut error) };
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
            riley_cuda_fixed37_gemm_plan_execute(
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
        let status = unsafe { riley_cuda_fixed37_gemm_plan_close(&mut raw, &mut error) };
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
            riley_cuda_smoke_buffer_create(
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
            riley_cuda_smoke_fill_launch(self.as_ptr(), stream.as_ptr(), value, &mut error)
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
            riley_cuda_smoke_copy_to_host(
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
        let status = unsafe { riley_cuda_smoke_buffer_close(&mut raw, &mut error) };
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
        DOMAIN_NVML => CudaErrorDomain::Nvml,
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

/// Preserves the native error for an operation that documented a retryable
/// pre-CUDA validation failure by returning its in/out owner unchanged.
fn non_success_status_error<T>(
    status: i32,
    operation: &'static str,
    error: &ErrorInfo,
) -> CudaResult<T> {
    debug_assert_ne!(status, STATUS_SUCCESS);
    match status_result(status, operation, error) {
        Ok(()) => unreachable!("a non-success native status must decode as an error"),
        Err(error) => Err(error),
    }
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

fn graph_capture_begin_success_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::CaptureBegin))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_none()
        && !decoded.submission_started()
        && !decoded.completion_known()
        && !decoded.resource_release_known()
        && !decoded.poisoned()
}

fn graph_capture_abort_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
    status: i32,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::CaptureAbort))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_none()
        && !decoded.submission_started()
        && !decoded.completion_known()
        && (status != STATUS_SUCCESS || (decoded.resource_release_known() && !decoded.poisoned()))
}

fn graph_capture_enqueue_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::CaptureEnqueue))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_none()
}

fn graph_capture_end_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::CaptureEnd))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_none()
}

fn graph_instantiate_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::Instantiate))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_some()
}

fn graph_exec_launch_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::Launch))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_some()
}

fn graph_launch_complete_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::Completion))
        && decoded.capture_id().is_some()
        && decoded.exec_id().is_some()
}

fn graph_close_metadata_is_valid(
    raw: &RawGraphErrorInfo,
    decoded: &CudaGraphFailureInfo,
    requires_exec_id: bool,
) -> bool {
    raw.struct_size() == RawGraphErrorInfo::ABI_SIZE
        && matches!(decoded.stage(), Some(CudaGraphStage::Close))
        && decoded.capture_id().is_some()
        && (decoded.exec_id().is_some() == requires_exec_id)
}

fn graph_resources_released(decoded: &CudaGraphFailureInfo) -> bool {
    decoded.resource_release_known() && !decoded.poisoned()
}

fn graph_owner_missing(operation: &'static str) -> CudaError {
    CudaError::new(
        CudaErrorKind::InvalidState,
        CudaErrorDomain::Rust,
        CudaErrorStage::Validation,
        0,
        operation,
        "the safe CUDA Graph owner was already consumed or intentionally abandoned",
    )
}

fn malformed_graph_metadata(operation: &'static str) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Validation,
        0,
        operation,
        "native CUDA Graph operation returned malformed lifecycle metadata",
    )
}

#[cfg(feature = "nvml")]
fn invalid_nvidia_snapshot(message: impl Into<String>) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Query,
        0,
        "probe NVIDIA environment",
        message,
    )
}

#[cfg(feature = "nvml")]
fn fixed_c_string_to_utf8<const N: usize>(
    bytes: &[c_char; N],
    field: &'static str,
) -> CudaResult<String> {
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or_else(|| invalid_nvidia_snapshot(format!("native {field} is not NUL-terminated")))?;
    let value: Vec<u8> = bytes[..nul]
        .iter()
        .copied()
        .map(|byte| u8::from_ne_bytes(byte.to_ne_bytes()))
        .collect();
    String::from_utf8(value)
        .map_err(|error| invalid_nvidia_snapshot(format!("native {field} is not UTF-8: {error}")))
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
#[cfg(feature = "nvml")]
const _: () = assert!(size_of::<RawNvidiaDeviceSnapshot>() == 320);
#[cfg(feature = "nvml")]
const _: () = assert!(offset_of!(RawNvidiaDeviceSnapshot, name) == 64);
#[cfg(feature = "nvml")]
const _: () = assert!(size_of::<RawNvidiaEnvironmentSnapshot>() == 10_352);
#[cfg(feature = "nvml")]
const _: () = assert!(offset_of!(RawNvidiaEnvironmentSnapshot, driver_version) == 32);
#[cfg(feature = "nvml")]
const _: () = assert!(offset_of!(RawNvidiaEnvironmentSnapshot, devices) == 112);
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
const _: () = assert!(size_of::<RawRopeTableParams>() == 152);
const _: () = assert!(offset_of!(RawRopeTableParams, element_count) == 104);
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
const _: () = assert!(size_of::<RawBf16ArgmaxParams>() == 152);
const _: () = assert!(offset_of!(RawBf16ArgmaxParams, logits) == 8);
const _: () = assert!(offset_of!(RawBf16ArgmaxParams, results) == 56);
const _: () = assert!(offset_of!(RawBf16ArgmaxParams, row_count) == 104);
const _: () = assert!(offset_of!(RawBf16ArgmaxParams, reserved) == 120);
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
const _: () = assert!(size_of::<RawHfPrefillAttentionConfig>() == 96);
const _: () = assert!(offset_of!(RawHfPrefillAttentionConfig, batch_count) == 8);
const _: () = assert!(offset_of!(RawHfPrefillAttentionConfig, max_cublas_workspace_bytes) == 56);
const _: () = assert!(size_of::<RawHfPrefillAttentionPlanInfo>() == 216);
const _: () = assert!(offset_of!(RawHfPrefillAttentionPlanInfo, qk_workspace_bytes) == 40);
const _: () = assert!(offset_of!(RawHfPrefillAttentionPlanInfo, av_workspace_bytes) == 88);
const _: () = assert!(offset_of!(RawHfPrefillAttentionPlanInfo, workspace_bytes) == 128);
const _: () = assert!(offset_of!(RawHfPrefillAttentionPlanInfo, batch_count) == 160);
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
const _: () = assert!(size_of::<RawFixed37RaggedPagedAttentionParams>() == 600);
const _: () = assert!(offset_of!(RawFixed37RaggedPagedAttentionParams, batch) == 200);
const _: () = assert!(offset_of!(RawFixed37RaggedPagedAttentionParams, query_head_count) == 520);
const _: () = assert!(offset_of!(RawFixed37RaggedPagedAttentionParams, output_row_count) == 544);
const _: () = assert!(
    offset_of!(
        RawFixed37RaggedPagedAttentionParams,
        maximum_logical_token_count
    ) == 552
);
const _: () = assert!(offset_of!(RawFixed37RaggedPagedAttentionParams, scale) == 560);
const _: () = assert!(offset_of!(RawFixed37RaggedPagedAttentionParams, reserved) == 568);
const _: () = assert!(size_of::<RawGemmConfig>() == 112);
const _: () = assert!(offset_of!(RawGemmConfig, flags) == 4);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_graph_abort_rejects_poisoned_release_evidence() {
        let poisoned = RawGraphErrorInfo::capture_abort_for_test(1, 1);
        let poisoned_decoded = decode_graph_failure_info(&poisoned)
            .expect("well-formed poisoned capture-abort evidence must decode");
        assert!(
            !graph_capture_abort_metadata_is_valid(&poisoned, &poisoned_decoded, STATUS_SUCCESS),
            "success must not accept an abort record that leaves the owner poisoned"
        );

        let released = RawGraphErrorInfo::capture_abort_for_test(1, 0);
        let released_decoded = decode_graph_failure_info(&released)
            .expect("well-formed released capture-abort evidence must decode");
        assert!(
            graph_capture_abort_metadata_is_valid(&released, &released_decoded, STATUS_SUCCESS),
            "success requires known release with no poison flag"
        );
    }
}
