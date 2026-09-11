//! Model, tensor, and backend orchestration boundary.

pub mod generation;
mod kernel;
pub mod llama;
pub mod paged_kv;
pub mod reference;
pub mod rng;
pub mod sampling;

#[cfg(any(feature = "cuda", test))]
mod cuda_weights;

#[cfg(feature = "cuda")]
pub use cuda_weights::{
    CudaUploadedTensor, CudaUploadedWeight, CudaUploadedWeights, CudaWeightUploadError,
    CudaWeightUploadResult,
};

pub use kernel::{
    KernelCapability, KernelImplementation, KernelKey, KernelOrigin, KernelPreference,
    KernelRegistry, KernelRegistryError, OpId, ROW_BIAS_ADD_IN_PLACE_BF16_KEY,
};

#[cfg(feature = "cuda")]
pub use riley_cuda::{
    AttentionBackend, AttentionPreference, AttentionSelectionTrace, CudaContext, CudaError,
    CudaEvent, CudaRuntime, CudaStream,
};

/// Static build information that does not initialize a CUDA device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildInfo {
    /// Whether the optional native CUDA ABI is linked.
    pub cuda_enabled: bool,
    /// The native ABI version when CUDA is linked.
    pub cuda_abi_version: Option<u32>,
    /// The model layer role bound into this runtime.
    pub model_role: &'static str,
    /// The tensor layer role bound into this runtime.
    pub tensor_role: &'static str,
}

/// Returns build information without creating a CUDA context or calling a device.
///
/// # Errors
///
/// Returns a native-contract error when a CUDA-enabled build reports an
/// incompatible C ABI version.
pub fn build_info() -> riley_core::Result<BuildInfo> {
    #[cfg(feature = "cuda")]
    let cuda_abi_version = Some(riley_cuda::abi_version()?);
    #[cfg(not(feature = "cuda"))]
    let cuda_abi_version = None;

    Ok(BuildInfo {
        cuda_enabled: cfg!(feature = "cuda"),
        cuda_abi_version,
        model_role: riley_model::CRATE_ROLE,
        tensor_role: riley_tensor::CRATE_ROLE,
    })
}

#[cfg(test)]
mod tests {
    use super::build_info;

    #[test]
    fn cpu_build_does_not_claim_cuda() {
        if cfg!(not(feature = "cuda")) {
            let info = build_info().expect("CPU build info must be available");
            assert!(!info.cuda_enabled);
            assert_eq!(info.cuda_abi_version, None);
        }
    }
}
