//! Safe ownership boundary for the optional native CUDA host runtime.
//!
//! With `cuda` disabled, initialization returns an actionable error without
//! loading or probing CUDA. With it enabled, all raw pointers and unsafe calls
//! remain confined to the private FFI module.

mod attention;
mod batch;
mod decode;
mod environment;
mod error;
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod ffi;
mod gemm;
mod graph;
#[cfg(any(feature = "cuda", test))]
mod hf_eager_allowlist;
mod memory;
mod prefill;
mod primitives;
mod runtime;

pub use attention::{
    AvGqaParams, CausalSoftmaxInPlaceParams, QkGqaParams, ScaleCausalMaskInPlaceParams, av_gqa,
    causal_softmax_in_place, fixed37_av_gqa, fixed37_causal_softmax_in_place, fixed37_qk_gqa,
    qk_gqa, scale_causal_mask_in_place,
};
pub use batch::{
    FIXED37_RAGGED_MAX_LOGICAL_TOKENS, IndexedRopeParams, PACKED_BATCH_BLOCK_SIZE,
    PACKED_BATCH_VERSION, PackedBatchHostV1, PackedBatchV1, RaggedPagedAttentionParams,
    RaggedPagedKvCacheWriteParams, RowGatherParams, fixed37_ragged_paged_attention,
    grouped_ragged_paged_attention, indexed_rope, ragged_paged_attention,
    ragged_paged_kv_cache_write, row_gather,
};
pub use decode::{
    DECODE_PARTIAL_STATE_VERSION, DecodeAttentionBackend, DecodeAttentionBackendAvailability,
    DecodeAttentionCapability, DecodeAttentionNoWorkspaceParams, DecodeAttentionParams,
    DecodeAttentionPreference, DecodeAttentionRequest, DecodeAttentionSelectionReason,
    DecodeAttentionSelectionTrace, DecodePartialReductionOrder, DecodePartialState,
    DecodePartialStateError, DecodePartialStateLayout, DecodePartialStateReduceParams,
    KvCacheAppendParams, PAGED_KV_BLOCK_SIZE, PAGED_KV_BLOCK_TABLE_VERSION,
    PagedDecodeAttentionNoWorkspaceParams, PagedDecodeAttentionParams, PagedDecodeAttentionRequest,
    PagedKvBlockTableHostV1, PagedKvBlockTableV1, PagedKvCacheAppendParams,
    PreparedDecodeAttention, PreparedPagedDecodeAttention, decode_partial_states_reduce,
    kv_cache_append, paged_kv_cache_append,
};
pub use environment::{
    NVML_ENABLED, NvidiaDeviceSnapshot, NvidiaEnvironmentSnapshot, NvidiaPersistenceMode,
    diagnose_null_nvidia_environment_output, probe_nvidia_environment,
};
pub use error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
pub use gemm::{
    CudaFixed37GemmMetadata, CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaGemmReductionPolicy,
    CudaPreparedFixed37Gemm, CudaPreparedGemm, FIXED37_CHUNK_ELEMENTS, FIXED37_MAX_CHUNK_COUNT,
    FIXED37_MAX_REDUCTION_ELEMENTS, FIXED37_REDUCTION_VERSION, Fixed37GemmParams, GemmParams,
};
pub use graph::{
    Bf16EmbeddingStatusD2HStatus, CapturedGraph, CudaGraphCaptureCapability, CudaGraphCaptureMode,
    CudaGraphCaptureOperation, CudaGraphFailureInfo, CudaGraphLifecycle, CudaGraphLifecycleState,
    CudaGraphStage, GraphCapture, GraphExec, GraphFillCapture, GraphLaunch,
    OwnedCapturedBf16ArgmaxGraph, OwnedCapturedBf16EmbeddingStatusD2HGraph,
    OwnedCapturedBf16RowGatherArgmaxD2HGraph, OwnedCapturedBf16RowGatherArgmaxGraph,
    OwnedCapturedBf16RowGatherGraph, OwnedCapturedCanonicalGemmBf16Graph,
    OwnedCapturedCanonicalRmsNormBf16Graph, OwnedCapturedCanonicalRmsNormGemmBf16Graph,
    OwnedCapturedGatedMultiplyBf16Graph, OwnedCapturedGraph,
    OwnedCapturedGroupedRaggedPagedAttentionBf16Graph, OwnedCapturedH2DGraph,
    OwnedCapturedIndexedRopeBf16Graph, OwnedCapturedRaggedPagedKvCacheWriteBf16Graph,
    OwnedCapturedResidualAddBf16Graph, OwnedCapturedSiluBf16Graph, OwnedGraphBf16ArgmaxCapture,
    OwnedGraphBf16ArgmaxCaptureBeginError, OwnedGraphBf16ArgmaxExec, OwnedGraphBf16ArgmaxLaunch,
    OwnedGraphBf16ArgmaxResources, OwnedGraphBf16EmbeddingStatusD2HCapture,
    OwnedGraphBf16EmbeddingStatusD2HCaptureBeginError, OwnedGraphBf16EmbeddingStatusD2HCompletion,
    OwnedGraphBf16EmbeddingStatusD2HExec, OwnedGraphBf16EmbeddingStatusD2HLaunch,
    OwnedGraphBf16EmbeddingStatusD2HResources, OwnedGraphBf16RowGatherArgmaxCapture,
    OwnedGraphBf16RowGatherArgmaxCaptureBeginError, OwnedGraphBf16RowGatherArgmaxD2HCapture,
    OwnedGraphBf16RowGatherArgmaxD2HCaptureBeginError, OwnedGraphBf16RowGatherArgmaxD2HCompletion,
    OwnedGraphBf16RowGatherArgmaxD2HExec, OwnedGraphBf16RowGatherArgmaxD2HLaunch,
    OwnedGraphBf16RowGatherArgmaxD2HResources, OwnedGraphBf16RowGatherArgmaxExec,
    OwnedGraphBf16RowGatherArgmaxLaunch, OwnedGraphBf16RowGatherArgmaxResources,
    OwnedGraphBf16RowGatherCapture, OwnedGraphBf16RowGatherCaptureBeginError,
    OwnedGraphBf16RowGatherExec, OwnedGraphBf16RowGatherLaunch, OwnedGraphBf16RowGatherResources,
    OwnedGraphCanonicalGemmBf16Capture, OwnedGraphCanonicalGemmBf16CaptureBeginError,
    OwnedGraphCanonicalGemmBf16Exec, OwnedGraphCanonicalGemmBf16Launch,
    OwnedGraphCanonicalGemmBf16Resources, OwnedGraphCanonicalRmsNormBf16Capture,
    OwnedGraphCanonicalRmsNormBf16CaptureBeginError, OwnedGraphCanonicalRmsNormBf16Exec,
    OwnedGraphCanonicalRmsNormBf16Launch, OwnedGraphCanonicalRmsNormBf16Resources,
    OwnedGraphCanonicalRmsNormGemmBf16Capture, OwnedGraphCanonicalRmsNormGemmBf16CaptureBeginError,
    OwnedGraphCanonicalRmsNormGemmBf16Exec, OwnedGraphCanonicalRmsNormGemmBf16Launch,
    OwnedGraphCanonicalRmsNormGemmBf16Resources, OwnedGraphExec, OwnedGraphFillCapture,
    OwnedGraphFillCaptureBeginError, OwnedGraphFillResources, OwnedGraphGatedMultiplyBf16Capture,
    OwnedGraphGatedMultiplyBf16CaptureBeginError, OwnedGraphGatedMultiplyBf16Exec,
    OwnedGraphGatedMultiplyBf16Launch, OwnedGraphGatedMultiplyBf16Resources,
    OwnedGraphGroupedRaggedPagedAttentionBf16Capture,
    OwnedGraphGroupedRaggedPagedAttentionBf16CaptureBeginError,
    OwnedGraphGroupedRaggedPagedAttentionBf16Exec, OwnedGraphGroupedRaggedPagedAttentionBf16Launch,
    OwnedGraphGroupedRaggedPagedAttentionBf16Resources, OwnedGraphH2DCapture,
    OwnedGraphH2DCaptureBeginError, OwnedGraphH2DExec, OwnedGraphH2DLaunch, OwnedGraphH2DResources,
    OwnedGraphIndexedRopeBf16Capture, OwnedGraphIndexedRopeBf16CaptureBeginError,
    OwnedGraphIndexedRopeBf16Exec, OwnedGraphIndexedRopeBf16Launch,
    OwnedGraphIndexedRopeBf16Resources, OwnedGraphLaunch,
    OwnedGraphRaggedPagedKvCacheWriteBf16Capture,
    OwnedGraphRaggedPagedKvCacheWriteBf16CaptureBeginError,
    OwnedGraphRaggedPagedKvCacheWriteBf16Exec, OwnedGraphRaggedPagedKvCacheWriteBf16Launch,
    OwnedGraphRaggedPagedKvCacheWriteBf16Resources, OwnedGraphResidualAddBf16Capture,
    OwnedGraphResidualAddBf16CaptureBeginError, OwnedGraphResidualAddBf16Exec,
    OwnedGraphResidualAddBf16Launch, OwnedGraphResidualAddBf16Resources, OwnedGraphSiluBf16Capture,
    OwnedGraphSiluBf16CaptureBeginError, OwnedGraphSiluBf16Exec, OwnedGraphSiluBf16Launch,
    OwnedGraphSiluBf16Resources,
};
pub use memory::{
    CudaAllocationStats, CudaDeviceBuffer, CudaPendingD2H, CudaPendingH2D, CudaPinnedHostBuffer,
};
#[cfg(feature = "cuda-test-fault-injection")]
pub use memory::{CudaMemoryFault, CudaMemoryFaultStats};
pub use prefill::{
    AttentionBackend, AttentionBackendAvailability, AttentionCapability, AttentionLayout,
    AttentionMask, AttentionMode, AttentionPreference, AttentionReductionProfile,
    AttentionScoreMaterialization, AttentionSelectionReason, AttentionSelectionTrace,
    OnlineSoftmaxError, OnlineSoftmaxState, PrefillAttentionParams, PrefillAttentionRequest,
    PreparedPrefillAttention,
};
pub use primitives::{
    BF16_ARGMAX_INVALID_TOKEN_ID, BF16_ARGMAX_RESULT_U32_WORDS, BF16_ARGMAX_STATUS_NON_FINITE,
    BF16_ARGMAX_STATUS_SUCCESS, Bf16ArgmaxParams, Bf16ArgmaxResult, CastParams, CudaBufferSpan,
    CudaBufferSpanMut, CudaDType, EmbeddingError, EmbeddingParams, Fixed37LogSoftmaxParams,
    GatedMultiplyParams, HUGGING_FACE_SMOLLM2_RMS_NORM_EPSILON_BITS,
    HUGGING_FACE_SMOLLM2_RMS_NORM_HIDDEN_SIZE, HUGGING_FACE_SMOLLM2_RMS_NORM_MAX_ROWS,
    ResidualAddParams, ResidualRmsNormParams, RmsNormParams, RopeParams, RopeTableParams,
    RowBiasAddInPlaceParams, SiluParams, cast, deterministic_bf16_argmax, embedding,
    fixed37_log_softmax, fixed37_residual_rms_norm, fixed37_rms_norm, gated_multiply,
    hugging_face_smollm2_residual_rms_norm, hugging_face_smollm2_rms_norm, residual_add,
    residual_rms_norm, rms_norm, rope, rope_table, row_bias_add_in_place, silu,
};
pub use runtime::{
    CudaCommandBatch, CudaCommandStream, CudaContext, CudaDevice, CudaEvent, CudaExecutionStream,
    CudaKernel, CudaPendingFill, CudaRuntime, CudaStream, DeviceProperties,
};

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "native CUDA C ABI and host-runtime boundary";

/// Whether this build includes the native CUDA feature.
pub const CUDA_ENABLED: bool = cfg!(feature = "cuda");

/// Normalized `CMAKE_CUDA_ARCHITECTURES` used for the native archive.
///
/// Plain numeric entries contain real and virtual code, while `-real` and
/// `-virtual` preserve their `CMake` meanings. When [`CUDA_ENABLED`] is false,
/// this reports the targets configured for a corresponding CUDA build without
/// claiming that the native archive is linked.
pub const CUDA_COMPILED_ARCHITECTURES: &str = env!("RILEY_CUDA_COMPILED_ARCHITECTURES");

/// ABI version expected by this Rust wrapper.
pub const EXPECTED_ABI_VERSION: u32 = 1;

/// Verifies and returns the linked native CUDA ABI version.
///
/// This metadata call never initializes or queries a device.
///
/// # Errors
///
/// Returns a native-contract error when the linked ABI version differs from
/// [`EXPECTED_ABI_VERSION`].
#[cfg(feature = "cuda")]
pub fn abi_version() -> riley_core::Result<u32> {
    let actual = ffi::abi_version();
    if actual == EXPECTED_ABI_VERSION {
        Ok(actual)
    } else {
        Err(riley_core::Error::native_contract(
            "riley-cuda",
            format!(
                "ABI mismatch: Rust expects {EXPECTED_ABI_VERSION}, native library reports {actual}"
            ),
        ))
    }
}

/// Returns compiler and ABI metadata from the linked native library.
///
/// This metadata call never initializes or queries a device.
///
/// # Errors
///
/// Returns a native-contract error if the native build string is invalid.
#[cfg(feature = "cuda")]
pub fn build_info() -> riley_core::Result<String> {
    abi_version()?;
    ffi::build_info()
        .map_err(|error| riley_core::Error::native_contract("riley-cuda", error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{CUDA_ENABLED, EXPECTED_ABI_VERSION};

    #[test]
    fn feature_flag_is_exposed_without_loading_cuda() {
        assert_eq!(CUDA_ENABLED, cfg!(feature = "cuda"));
        assert_eq!(EXPECTED_ABI_VERSION, 1);
    }
}
