use std::error;
use std::fmt;

/// Result type for CUDA host-runtime operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// Stable high-level classification independent of CUDA's numeric codes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaErrorKind {
    /// The crate was built without CUDA support or the runtime is unavailable.
    Unavailable,
    /// An argument violates the safe or native API contract.
    InvalidArgument,
    /// A device ordinal does not exist.
    InvalidDevice,
    /// A size, count, or launch dimension is outside the supported range.
    OutOfRange,
    /// An asynchronous resource has not completed yet.
    NotReady,
    /// Host or device allocation failed.
    OutOfMemory,
    /// The requested operation or exact contract is unsupported.
    NotSupported,
    /// The CUDA Driver API returned an error.
    Driver,
    /// A CUDA runtime or execution-library API returned an error.
    Runtime,
    /// Resource ownership or lifecycle state is invalid.
    InvalidState,
    /// The native boundary reported an internal contract failure.
    Internal,
}

/// Origin of a CUDA host-runtime error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaErrorDomain {
    /// Rust-side validation before entering native code.
    Rust,
    /// Native fixed-width ABI validation.
    Validation,
    /// CUDA Driver API.
    Driver,
    /// CUDA Runtime API.
    Runtime,
    /// cuBLASLt execution and algorithm-selection API.
    CuBlasLt,
    /// NVIDIA Management Library (NVML).
    Nvml,
    /// Native implementation invariant.
    Internal,
}

/// Lifecycle stage at which an error was observed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CudaErrorStage {
    /// Compile-time feature or runtime initialization.
    Initialize,
    /// Argument and ownership validation.
    Validation,
    /// Resource creation.
    Create,
    /// Immutable execution-plan preparation and algorithm selection.
    Prepare,
    /// CUDA kernel launch and immediate launch checking.
    Launch,
    /// Asynchronous completion and late error checking.
    Synchronize,
    /// Non-blocking completion query.
    Query,
    /// Stream/event dependency recording.
    Record,
    /// Host/device data copy.
    Copy,
    /// Explicit or best-effort resource destruction.
    Close,
}

/// Detailed, owned CUDA failure returned by the safe API.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaError {
    kind: CudaErrorKind,
    domain: CudaErrorDomain,
    stage: CudaErrorStage,
    native_code: i32,
    operation: &'static str,
    message: String,
}

impl CudaError {
    pub(crate) fn new(
        kind: CudaErrorKind,
        domain: CudaErrorDomain,
        stage: CudaErrorStage,
        native_code: i32,
        operation: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            domain,
            stage,
            native_code,
            operation,
            message: message.into(),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn unavailable(operation: &'static str) -> Self {
        Self::new(
            CudaErrorKind::Unavailable,
            CudaErrorDomain::Rust,
            CudaErrorStage::Initialize,
            0,
            operation,
            "rustinfer-cuda was compiled without the `cuda` feature; rebuild with `--features cuda` on a host with the CUDA toolkit",
        )
    }

    #[cfg(not(feature = "nvml"))]
    pub(crate) fn nvml_unavailable(operation: &'static str) -> Self {
        Self::new(
            CudaErrorKind::Unavailable,
            CudaErrorDomain::Rust,
            CudaErrorStage::Initialize,
            0,
            operation,
            "rustinfer-cuda was compiled without the `nvml` feature; rebuild the development/calibration binary with `--features nvml` on a host with the NVIDIA Management Library",
        )
    }

    pub(crate) fn invalid_state(operation: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            CudaErrorKind::InvalidState,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            message,
        )
    }

    pub(crate) fn invalid_argument(operation: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            CudaErrorKind::InvalidArgument,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            message,
        )
    }

    pub(crate) fn out_of_range(operation: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            CudaErrorKind::OutOfRange,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            message,
        )
    }

    pub(crate) fn invalid_device(operation: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            CudaErrorKind::InvalidDevice,
            CudaErrorDomain::Rust,
            CudaErrorStage::Validation,
            0,
            operation,
            message,
        )
    }

    pub(crate) fn host_allocation(operation: &'static str, message: impl Into<String>) -> Self {
        Self::new(
            CudaErrorKind::OutOfMemory,
            CudaErrorDomain::Rust,
            CudaErrorStage::Copy,
            0,
            operation,
            message,
        )
    }

    /// Stable high-level classification.
    #[must_use]
    pub const fn kind(&self) -> CudaErrorKind {
        self.kind
    }

    /// Subsystem that produced the error.
    #[must_use]
    pub const fn domain(&self) -> CudaErrorDomain {
        self.domain
    }

    /// Lifecycle stage at which the error was observed.
    #[must_use]
    pub const fn stage(&self) -> CudaErrorStage {
        self.stage
    }

    /// Original CUDA code, or zero for Rust/validation errors.
    #[must_use]
    pub const fn native_code(&self) -> i32 {
        self.native_code
    }

    /// Stable operation label for diagnostics and tests.
    #[must_use]
    pub const fn operation(&self) -> &'static str {
        self.operation
    }

    /// Detailed owned diagnostic from the native caller buffer.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "CUDA {:?} error during {} ({:?}/{:?}, native code {}): {}",
            self.kind, self.operation, self.domain, self.stage, self.native_code, self.message
        )
    }
}

impl error::Error for CudaError {}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage};

    #[test]
    fn feature_off_error_is_actionable() {
        let error = CudaError::unavailable("CudaRuntime::initialize");
        assert_eq!(error.kind(), CudaErrorKind::Unavailable);
        assert_eq!(error.domain(), CudaErrorDomain::Rust);
        assert_eq!(error.stage(), CudaErrorStage::Initialize);
        assert!(error.to_string().contains("without the `cuda` feature"));
    }
}
