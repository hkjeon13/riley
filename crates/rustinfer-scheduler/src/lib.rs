//! Admission and batching state boundary; scheduling arrives in PR 13.

/// Returns lower-layer build information without starting a scheduler.
///
/// # Errors
///
/// Propagates a native-contract error from a CUDA-enabled runtime.
pub fn runtime_build_info() -> rustinfer_core::Result<rustinfer_runtime::BuildInfo> {
    rustinfer_runtime::build_info()
}
