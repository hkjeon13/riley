//! Safe ownership boundary for the optional native CUDA host runtime.
//!
//! With `cuda` disabled, initialization returns an actionable error without
//! loading or probing CUDA. With it enabled, all raw pointers and unsafe calls
//! remain confined to the private FFI module.

mod error;
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod ffi;
mod gemm;
mod memory;
mod primitives;
mod runtime;

pub use error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
pub use gemm::{CudaGemmAlgorithmMetadata, CudaGemmConfig, CudaPreparedGemm, GemmParams};
pub use memory::{
    CudaAllocationStats, CudaDeviceBuffer, CudaPendingD2H, CudaPendingH2D, CudaPinnedHostBuffer,
};
pub use primitives::{
    CastParams, CudaBufferSpan, CudaBufferSpanMut, CudaDType, EmbeddingError, EmbeddingParams,
    GatedMultiplyParams, ResidualAddParams, RmsNormParams, RopeParams, SiluParams, cast, embedding,
    gated_multiply, residual_add, rms_norm, rope, silu,
};
pub use runtime::{
    CudaContext, CudaDevice, CudaEvent, CudaKernel, CudaPendingFill, CudaRuntime, CudaStream,
    DeviceProperties,
};

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "native CUDA C ABI and host-runtime boundary";

/// Whether this build includes the native CUDA feature.
pub const CUDA_ENABLED: bool = cfg!(feature = "cuda");

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
