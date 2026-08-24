//! Optional, safe Rust wrapper around the native CUDA C ABI.
//!
//! Enabling `cuda` performs AOT compilation and linking only. The PR 02 ABI
//! contains no device calls, context creation, allocation, or inference.

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "native CUDA C ABI boundary";

/// Whether this build includes the native CUDA feature.
pub const CUDA_ENABLED: bool = cfg!(feature = "cuda");

/// ABI version expected by this Rust wrapper.
pub const EXPECTED_ABI_VERSION: u32 = 1;

#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
mod native {
    use std::ffi::{CStr, c_char};

    use rustinfer_core::{Error, Result};

    use super::EXPECTED_ABI_VERSION;

    unsafe extern "C" {
        fn rustinfer_cuda_abi_version() -> u32;
        fn rustinfer_cuda_build_info() -> *const c_char;
    }

    /// Verifies and returns the linked native CUDA ABI version.
    ///
    /// This calls a host-only function and never initializes a CUDA device.
    ///
    /// # Errors
    ///
    /// Returns a native-contract error when the linked ABI version differs
    /// from [`EXPECTED_ABI_VERSION`].
    pub fn abi_version() -> Result<u32> {
        // SAFETY: the symbol is provided by the statically linked library and
        // takes no pointers or arguments. The C header fixes its return type.
        let actual = unsafe { rustinfer_cuda_abi_version() };
        if actual == EXPECTED_ABI_VERSION {
            Ok(actual)
        } else {
            Err(Error::native_contract(
                "rustinfer-cuda",
                format!(
                    "ABI mismatch: Rust expects {EXPECTED_ABI_VERSION}, native library reports {actual}"
                ),
            ))
        }
    }

    /// Returns compile-time information from the linked native library.
    ///
    /// The returned value describes the compiler/ABI only. It is not a runtime
    /// device capability query and does not initialize CUDA.
    ///
    /// # Errors
    ///
    /// Returns a native-contract error when the native symbol returns null or
    /// returns bytes that are not valid UTF-8.
    pub fn build_info() -> Result<String> {
        // SAFETY: the linked ABI promises either null or a pointer to a
        // process-lifetime, NUL-terminated string. We check null and copy the
        // bytes into owned Rust memory before returning.
        let pointer = unsafe { rustinfer_cuda_build_info() };
        if pointer.is_null() {
            return Err(Error::native_contract(
                "rustinfer-cuda",
                "build-info symbol returned a null pointer",
            ));
        }
        // SAFETY: null was rejected above and the native ABI guarantees a
        // process-lifetime NUL-terminated byte string.
        let value = unsafe { CStr::from_ptr(pointer) };
        value.to_str().map(str::to_owned).map_err(|error| {
            Error::native_contract(
                "rustinfer-cuda",
                format!("build-info symbol returned invalid UTF-8: {error}"),
            )
        })
    }
}

#[cfg(feature = "cuda")]
pub use native::{abi_version, build_info};

#[cfg(test)]
mod tests {
    use super::{CUDA_ENABLED, EXPECTED_ABI_VERSION};

    #[test]
    fn feature_flag_is_exposed_without_loading_cuda() {
        assert_eq!(CUDA_ENABLED, cfg!(feature = "cuda"));
        assert_eq!(EXPECTED_ABI_VERSION, 1);
    }
}
