//! Binary-facing build information and the bounded PR 14 serving surface.

#[cfg(feature = "bench")]
pub mod benchmark;
#[cfg(feature = "server")]
pub mod domain;
#[cfg(feature = "server")]
pub mod engine;
#[cfg(feature = "server")]
pub mod http;
#[cfg(feature = "server")]
pub mod openai;
#[cfg(feature = "server")]
pub mod service;

/// Returns a stable, human-readable version line without starting a server.
///
/// # Errors
///
/// Propagates a native-contract error from a CUDA-enabled runtime.
pub fn version_line() -> rustinfer_core::Result<String> {
    let runtime = rustinfer_scheduler::runtime_build_info()?;
    let cuda_abi = runtime
        .cuda_abi_version
        .map_or_else(|| "none".to_owned(), |version| version.to_string());
    Ok(format!(
        "rustinfer {} (server={}, cuda={}, cuda_abi={cuda_abi})",
        env!("CARGO_PKG_VERSION"),
        cfg!(feature = "server"),
        runtime.cuda_enabled
    ))
}

#[cfg(test)]
mod tests {
    use super::version_line;

    #[test]
    fn version_line_is_available_without_cuda() {
        if cfg!(not(feature = "cuda")) {
            assert_eq!(
                version_line().expect("version line must be available"),
                format!(
                    "rustinfer {} (server={}, cuda=false, cuda_abi=none)",
                    env!("CARGO_PKG_VERSION"),
                    cfg!(feature = "server")
                )
            );
        }
    }
}
