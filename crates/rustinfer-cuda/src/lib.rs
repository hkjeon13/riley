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
    RaggedPagedKvCacheWriteParams, RowGatherParams, fixed37_ragged_paged_attention, indexed_rope,
    ragged_paged_attention, ragged_paged_kv_cache_write, row_gather,
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
    CastParams, CudaBufferSpan, CudaBufferSpanMut, CudaDType, EmbeddingError, EmbeddingParams,
    Fixed37LogSoftmaxParams, GatedMultiplyParams, ResidualAddParams, ResidualRmsNormParams,
    RmsNormParams, RopeParams, RowBiasAddInPlaceParams, SiluParams, cast, embedding,
    fixed37_log_softmax, fixed37_residual_rms_norm, fixed37_rms_norm, gated_multiply, residual_add,
    residual_rms_norm, rms_norm, rope, row_bias_add_in_place, silu,
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
pub const CUDA_COMPILED_ARCHITECTURES: &str = env!("RUSTINFER_CUDA_COMPILED_ARCHITECTURES");

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
pub fn abi_version() -> rustinfer_core::Result<u32> {
    let actual = ffi::abi_version();
    if actual == EXPECTED_ABI_VERSION {
        Ok(actual)
    } else {
        Err(rustinfer_core::Error::native_contract(
            "rustinfer-cuda",
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
pub fn build_info() -> rustinfer_core::Result<String> {
    abi_version()?;
    ffi::build_info().map_err(|error| {
        rustinfer_core::Error::native_contract("rustinfer-cuda", error.to_string())
    })
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
