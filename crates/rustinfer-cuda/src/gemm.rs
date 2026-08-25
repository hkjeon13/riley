use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

use crate::error::{CudaError, CudaResult};
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{
    ContextInner, CudaContext, CudaExecutionStream, CudaStream, ensure_same_context,
    execution_stream_mut,
};

#[cfg(feature = "cuda")]
use crate::error::{CudaErrorDomain, CudaErrorKind, CudaErrorStage};
#[cfg(feature = "cuda")]
use crate::ffi;

const BF16_BYTES: u64 = 2;
const GEMM_OFFSET_ALIGNMENT: u64 = 256;
const MAX_CUBLASLT_DIMENSION: u64 = 2_147_483_647;
#[cfg(feature = "cuda")]
const NATIVE_CUBLASLT_BACKEND_ID: u32 = 1;
#[cfg(feature = "cuda")]
const DETERMINISTIC_REQUIRED: u32 = 1;
#[cfg(feature = "cuda")]
const REDUCTION_SCHEME_NONE: u32 = 0;

/// Exact dense GEMM contract accepted by the PR 06 CUDA adapter.
///
/// The logical operation is row-major `Y[M, N] = X[M, K] * W[N, K]^T`.
/// Inputs, weights, and output are BF16, accumulation is F32, the epilogue is
/// disabled, and deterministic algorithm selection is mandatory. Those
/// properties are intentionally not configurable in this initial boundary.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaGemmConfig {
    m: u64,
    n: u64,
    k: u64,
    max_workspace_bytes: u64,
    input_bytes: u64,
    weight_bytes: u64,
    output_bytes: u64,
}

impl CudaGemmConfig {
    /// Creates a checked deterministic BF16/F32 GEMM configuration.
    ///
    /// `max_workspace_bytes` is a preparation-time cap. The exact selected
    /// workspace requirement is reported by [`CudaGemmAlgorithmMetadata`].
    ///
    /// # Errors
    ///
    /// Returns invalid-argument for a zero dimension and out-of-range if a
    /// required matrix byte length or the native workspace size overflows.
    pub fn new(m: u64, n: u64, k: u64, max_workspace_bytes: u64) -> CudaResult<Self> {
        const OPERATION: &str = "CudaGemmConfig::new";
        if m == 0 || n == 0 || k == 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "M, N, and K must all be non-zero",
            ));
        }
        if m > MAX_CUBLASLT_DIMENSION || n > MAX_CUBLASLT_DIMENSION || k > MAX_CUBLASLT_DIMENSION {
            return Err(CudaError::out_of_range(
                OPERATION,
                "M, N, and K must each fit the cuBLASLt signed 32-bit dimension range",
            ));
        }

        let max_native_workspace = u64::try_from(usize::MAX).unwrap_or(u64::MAX);
        if max_workspace_bytes > max_native_workspace {
            return Err(CudaError::out_of_range(
                OPERATION,
                "workspace cap exceeds the native size_t range",
            ));
        }

        let input_bytes = checked_bf16_matrix_bytes(m, k, "input")?;
        let weight_bytes = checked_bf16_matrix_bytes(n, k, "weight")?;
        let output_bytes = checked_bf16_matrix_bytes(m, n, "output")?;
        Ok(Self {
            m,
            n,
            k,
            max_workspace_bytes,
            input_bytes,
            weight_bytes,
            output_bytes,
        })
    }

    /// Logical output row count.
    #[must_use]
    pub const fn m(self) -> u64 {
        self.m
    }

    /// Logical output column count.
    #[must_use]
    pub const fn n(self) -> u64 {
        self.n
    }

    /// Reduction dimension.
    #[must_use]
    pub const fn k(self) -> u64 {
        self.k
    }

    /// Maximum temporary bytes allowed during deterministic algorithm choice.
    #[must_use]
    pub const fn max_workspace_bytes(self) -> u64 {
        self.max_workspace_bytes
    }

    /// Exact required bytes for row-major BF16 `X[M, K]`.
    #[must_use]
    pub const fn input_bytes(self) -> u64 {
        self.input_bytes
    }

    /// Exact required bytes for row-major BF16 `W[N, K]`.
    #[must_use]
    pub const fn weight_bytes(self) -> u64 {
        self.weight_bytes
    }

    /// Exact required bytes for row-major BF16 `Y[M, N]`.
    #[must_use]
    pub const fn output_bytes(self) -> u64 {
        self.output_bytes
    }

    /// Input storage type fixed by this adapter.
    #[must_use]
    pub const fn input_dtype(self) -> CudaDType {
        CudaDType::BF16
    }

    /// Weight storage type fixed by this adapter.
    #[must_use]
    pub const fn weight_dtype(self) -> CudaDType {
        CudaDType::BF16
    }

    /// Accumulation type fixed by this adapter.
    #[must_use]
    pub const fn accumulator_dtype(self) -> CudaDType {
        CudaDType::F32
    }

    /// Output storage type fixed by this adapter.
    #[must_use]
    pub const fn output_dtype(self) -> CudaDType {
        CudaDType::BF16
    }

    /// Whether algorithm selection is required to be deterministic.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        true
    }
}

fn checked_bf16_matrix_bytes(
    rows: u64,
    columns: u64,
    matrix_name: &'static str,
) -> CudaResult<u64> {
    rows.checked_mul(columns)
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or_else(|| {
            CudaError::out_of_range(
                "CudaGemmConfig::new",
                format!("{matrix_name} BF16 matrix byte length overflows u64"),
            )
        })
}

/// Immutable provenance for the cuBLASLt algorithm selected at preparation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaGemmAlgorithmMetadata {
    backend_id: u32,
    algorithm_id: i32,
    tile_id: u32,
    stages_id: u32,
    split_k: u32,
    reduction_scheme: u32,
    cta_swizzling: u32,
    custom_option: u32,
    deterministic: bool,
    workspace_bytes: u64,
    numerical_implementation_flags: u64,
    compute_capability_major: u32,
    compute_capability_minor: u32,
    runtime_version: i32,
    cublaslt_version: i32,
    m: u64,
    n: u64,
    k: u64,
}

impl CudaGemmAlgorithmMetadata {
    /// Stable backend implementation identifier for cuBLASLt.
    pub const CUBLASLT_BACKEND_ID: u32 = 1;

    /// Native backend implementation identifier.
    #[must_use]
    pub const fn backend_id(self) -> u32 {
        self.backend_id
    }

    /// cuBLASLt algorithm configuration identifier.
    #[must_use]
    pub const fn algorithm_id(self) -> i32 {
        self.algorithm_id
    }

    /// cuBLASLt tile identifier.
    #[must_use]
    pub const fn tile_id(self) -> u32 {
        self.tile_id
    }

    /// cuBLASLt pipeline-stages identifier.
    #[must_use]
    pub const fn stages_id(self) -> u32 {
        self.stages_id
    }

    /// Selected split-K count.
    #[must_use]
    pub const fn split_k(self) -> u32 {
        self.split_k
    }

    /// cuBLASLt reduction-scheme identifier.
    #[must_use]
    pub const fn reduction_scheme(self) -> u32 {
        self.reduction_scheme
    }

    /// Selected CTA swizzling value.
    #[must_use]
    pub const fn cta_swizzling(self) -> u32 {
        self.cta_swizzling
    }

    /// Selected backend-specific custom option.
    #[must_use]
    pub const fn custom_option(self) -> u32 {
        self.custom_option
    }

    /// Whether the selected algorithm satisfies the deterministic contract.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Exact U8 workspace bytes required by every execution of this plan.
    #[must_use]
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }

    /// cuBLASLt numerical implementation capability flags.
    #[must_use]
    pub const fn numerical_implementation_flags(self) -> u64 {
        self.numerical_implementation_flags
    }

    /// Compute capability used during algorithm selection.
    #[must_use]
    pub const fn compute_capability(self) -> (u32, u32) {
        (self.compute_capability_major, self.compute_capability_minor)
    }

    /// CUDA Runtime version used during algorithm selection.
    #[must_use]
    pub const fn runtime_version(self) -> i32 {
        self.runtime_version
    }

    /// cuBLASLt version used during algorithm selection.
    #[must_use]
    pub const fn cublaslt_version(self) -> i32 {
        self.cublaslt_version
    }

    /// Prepared `(M, N, K)` dimensions recorded by native code.
    #[must_use]
    pub const fn dimensions(self) -> (u64, u64, u64) {
        (self.m, self.n, self.k)
    }
}

/// Borrowed buffers for one synchronous prepared GEMM execution.
///
/// Every span length must exactly match the prepared shape. All byte offsets
/// must be 256-byte aligned, all provided buffers must be distinct and owned
/// by the plan's context, and workspace must be U8. `workspace` may be `None`
/// only when the selected workspace requirement is zero.
#[derive(Debug)]
pub struct GemmParams<'a> {
    /// Row-major BF16 `X[M, K]`.
    pub input: CudaBufferSpan<'a>,
    /// Row-major BF16 `W[N, K]` consumed logically transposed.
    pub weight: CudaBufferSpan<'a>,
    /// Row-major BF16 `Y[M, N]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Exact selected U8 workspace, or none when zero bytes were selected.
    pub workspace: Option<CudaBufferSpanMut<'a>>,
}

/// Owning immutable cuBLASLt execution plan.
///
/// A plan retains its CUDA context and can move between host threads, but is
/// deliberately `!Sync`. Execution requires `&mut self` and returns only after
/// the explicit stream has synchronized, so all borrowed buffers can be reused
/// immediately after `Ok(())`.
pub struct CudaPreparedGemm {
    #[cfg(feature = "cuda")]
    native: ffi::GemmPlanHandle,
    // Native must close before releasing the context lease.
    context: Arc<ContextInner>,
    config: CudaGemmConfig,
    algorithm: CudaGemmAlgorithmMetadata,
    poisoned: bool,
    _not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for CudaPreparedGemm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaPreparedGemm")
            .field("device_ordinal", &self.context.ordinal)
            .field("config", &self.config)
            .field("algorithm", &self.algorithm)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl CudaContext {
    /// Prepares one deterministic cuBLASLt algorithm and all of its immutable
    /// descriptors for allocation-free repeated execution.
    ///
    /// # Errors
    ///
    /// Returns an actionable unavailable error when CUDA support is disabled,
    /// or a translated validation, capability, allocation, driver, runtime, or
    /// cuBLASLt preparation error.
    pub fn prepare_gemm(&self, config: CudaGemmConfig) -> CudaResult<CudaPreparedGemm> {
        #[cfg(feature = "cuda")]
        {
            let native = ffi::GemmPlanHandle::create(
                &self.inner.native,
                config.m,
                config.n,
                config.k,
                config.max_workspace_bytes,
            )?;
            let algorithm = CudaGemmAlgorithmMetadata::from_native(config, native.info()?)?;
            Ok(CudaPreparedGemm {
                native,
                context: Arc::clone(&self.inner),
                config,
                algorithm,
                poisoned: false,
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = config;
            Err(CudaError::unavailable("CudaContext::prepare_gemm"))
        }
    }
}

impl CudaPreparedGemm {
    /// Exact logical and storage contract used to prepare this plan.
    #[must_use]
    pub const fn config(&self) -> CudaGemmConfig {
        self.config
    }

    /// Immutable selected algorithm and environment provenance.
    #[must_use]
    pub const fn algorithm_metadata(&self) -> CudaGemmAlgorithmMetadata {
        self.algorithm
    }

    /// Device ordinal retained by this plan.
    #[must_use]
    pub fn device_ordinal(&self) -> u32 {
        self.context.ordinal
    }

    /// Whether a prior native execution failed and permanently disabled reuse
    /// of this safe plan wrapper.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Executes the prepared GEMM and synchronizes the same explicit stream.
    ///
    /// The successful repeated path performs no host or device allocation.
    /// A native execution error poisons this wrapper conservatively; native
    /// active-use guards independently prevent buffer, stream, or plan reuse
    /// when completion or context restoration could not be confirmed.
    ///
    /// # Errors
    ///
    /// Returns before entering native code for dtype, exact-size, alignment,
    /// context, alias, workspace, busy-buffer, or poisoned-plan violations.
    /// Native launch, cuBLASLt, synchronization, and restoration failures are
    /// translated with their status domain and lifecycle stage.
    pub fn execute<S: CudaExecutionStream + ?Sized>(
        &mut self,
        params: &mut GemmParams<'_>,
        stream: &mut S,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaPreparedGemm::execute";
        let stream = execution_stream_mut(stream);
        if self.poisoned {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the GEMM plan was poisoned by a prior native execution failure",
            ));
        }
        self.validate_execution(params, stream)?;

        #[cfg(feature = "cuda")]
        {
            let workspace = match params.workspace.as_ref() {
                Some(workspace) => workspace.raw(),
                None => params.output.buffer().native_handle().span(
                    ffi::DTYPE_U8,
                    params.output.byte_offset(),
                    0,
                ),
            };
            let result = self.native.execute(
                params.input.raw(),
                params.weight.raw(),
                params.output.raw(),
                workspace,
                &mut stream.native,
            );
            if result.is_err() {
                self.poisoned = true;
            }
            result
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (params, stream);
            Err(CudaError::unavailable("CudaPreparedGemm::execute"))
        }
    }

    /// Explicitly closes the prepared plan.
    ///
    /// A poisoned wrapper refuses to report a successful explicit close. Its
    /// private native owner still performs best-effort fail-closed cleanup on
    /// drop; ambiguous native resources remain retained rather than becoming
    /// reachable through a stale handle.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for a poisoned plan, unavailable without CUDA, or
    /// a translated native descriptor/context destruction error.
    pub fn close(self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            if this.poisoned {
                return Err(CudaError::invalid_state(
                    "CudaPreparedGemm::close",
                    "a poisoned GEMM plan cannot be explicitly reused or reported as cleanly closed",
                ));
            }
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaPreparedGemm::close"))
        }
    }

    fn validate_execution(&self, params: &GemmParams<'_>, stream: &CudaStream) -> CudaResult<()> {
        const OPERATION: &str = "CudaPreparedGemm::execute";
        ensure_same_context(&self.context, &stream.context, OPERATION)?;

        validate_span(
            params.input.buffer(),
            params.input.dtype(),
            params.input.byte_offset(),
            params.input.byte_len(),
            CudaDType::BF16,
            self.config.input_bytes,
            "input",
        )?;
        validate_span(
            params.weight.buffer(),
            params.weight.dtype(),
            params.weight.byte_offset(),
            params.weight.byte_len(),
            CudaDType::BF16,
            self.config.weight_bytes,
            "weight",
        )?;
        validate_span(
            params.output.buffer(),
            params.output.dtype(),
            params.output.byte_offset(),
            params.output.byte_len(),
            CudaDType::BF16,
            self.config.output_bytes,
            "output",
        )?;

        let workspace_bytes = self.algorithm.workspace_bytes;
        match params.workspace.as_ref() {
            Some(workspace) => validate_span(
                workspace.buffer(),
                workspace.dtype(),
                workspace.byte_offset(),
                workspace.byte_len(),
                CudaDType::U8,
                workspace_bytes,
                "workspace",
            )?,
            None if workspace_bytes == 0 => {}
            None => {
                return Err(CudaError::invalid_argument(
                    OPERATION,
                    format!(
                        "the selected algorithm requires an exact {workspace_bytes}-byte U8 workspace"
                    ),
                ));
            }
        }

        let required_buffers = [
            ("input", params.input.buffer()),
            ("weight", params.weight.buffer()),
            ("output", params.output.buffer()),
        ];
        for (_, buffer) in required_buffers {
            ensure_same_context(&self.context, buffer.context_owner(), OPERATION)?;
        }
        ensure_distinct_buffers(&required_buffers, OPERATION)?;

        if let Some(workspace) = params.workspace.as_ref() {
            let workspace_buffer = workspace.buffer();
            ensure_same_context(&self.context, workspace_buffer.context_owner(), OPERATION)?;
            for (name, buffer) in required_buffers {
                if ptr::eq(buffer, workspace_buffer) {
                    return Err(CudaError::invalid_argument(
                        OPERATION,
                        format!("workspace aliases the {name} device-buffer handle"),
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaGemmAlgorithmMetadata {
    fn from_native(
        config: CudaGemmConfig,
        native: ffi::NativeGemmAlgorithmInfo,
    ) -> CudaResult<Self> {
        if native.backend != NATIVE_CUBLASLT_BACKEND_ID {
            return Err(native_metadata_error(format!(
                "native GEMM backend id {} does not identify cuBLASLt",
                native.backend
            )));
        }
        if native.deterministic != DETERMINISTIC_REQUIRED {
            return Err(native_metadata_error(
                "native GEMM algorithm is not marked deterministic",
            ));
        }
        if native.split_k > 1 || native.reduction_scheme != REDUCTION_SCHEME_NONE {
            return Err(native_metadata_error(format!(
                "native GEMM algorithm violates the deterministic reduction contract: split_k={}, reduction_scheme={}",
                native.split_k, native.reduction_scheme
            )));
        }
        if (native.m, native.n, native.k) != (config.m, config.n, config.k) {
            return Err(native_metadata_error(format!(
                "native GEMM metadata dimensions ({}, {}, {}) differ from prepared ({}, {}, {})",
                native.m, native.n, native.k, config.m, config.n, config.k
            )));
        }
        if native.workspace_bytes > config.max_workspace_bytes {
            return Err(native_metadata_error(format!(
                "native GEMM workspace requirement {} exceeds configured cap {}",
                native.workspace_bytes, config.max_workspace_bytes
            )));
        }
        Ok(Self {
            backend_id: native.backend,
            algorithm_id: native.algorithm_id,
            tile_id: native.tile_id,
            stages_id: native.stages_id,
            split_k: native.split_k,
            reduction_scheme: native.reduction_scheme,
            cta_swizzling: native.cta_swizzling,
            custom_option: native.custom_option,
            deterministic: true,
            workspace_bytes: native.workspace_bytes,
            numerical_implementation_flags: native.numerical_implementation_flags,
            compute_capability_major: native.compute_capability_major,
            compute_capability_minor: native.compute_capability_minor,
            runtime_version: native.runtime_version,
            cublaslt_version: native.cublaslt_version,
            m: native.m,
            n: native.n,
            k: native.k,
        })
    }
}

#[cfg(feature = "cuda")]
fn native_metadata_error(message: impl Into<String>) -> CudaError {
    CudaError::new(
        CudaErrorKind::Internal,
        CudaErrorDomain::Internal,
        CudaErrorStage::Prepare,
        0,
        "CudaContext::prepare_gemm",
        message,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_span(
    buffer: &CudaDeviceBuffer,
    actual_dtype: CudaDType,
    byte_offset: u64,
    actual_bytes: u64,
    required_dtype: CudaDType,
    required_bytes: u64,
    name: &'static str,
) -> CudaResult<()> {
    const OPERATION: &str = "CudaPreparedGemm::execute";
    if actual_dtype != required_dtype {
        return Err(CudaError::invalid_argument(
            OPERATION,
            format!("{name} dtype must be {required_dtype}, got {actual_dtype}"),
        ));
    }
    if actual_bytes != required_bytes {
        return Err(CudaError::invalid_argument(
            OPERATION,
            format!("{name} span must contain exactly {required_bytes} bytes, got {actual_bytes}"),
        ));
    }
    if byte_offset % GEMM_OFFSET_ALIGNMENT != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            format!("{name} byte offset {byte_offset} is not {GEMM_OFFSET_ALIGNMENT}-byte aligned"),
        ));
    }
    buffer.ensure_idle_for_operation(OPERATION)
}

fn ensure_distinct_buffers(
    buffers: &[(&'static str, &CudaDeviceBuffer); 3],
    operation: &'static str,
) -> CudaResult<()> {
    for left_index in 0..buffers.len() {
        for right_index in left_index + 1..buffers.len() {
            let (left_name, left) = buffers[left_index];
            let (right_name, right) = buffers[right_index];
            if ptr::eq(left, right) {
                return Err(CudaError::invalid_argument(
                    operation,
                    format!("{left_name} and {right_name} alias the same device-buffer handle"),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CudaGemmConfig, checked_bf16_matrix_bytes};
    use crate::{CudaDType, CudaErrorKind};

    #[test]
    fn config_pins_exact_dtype_and_checked_byte_arithmetic() {
        let config = CudaGemmConfig::new(3, 5, 7, 4096).expect("valid odd shape");
        assert_eq!((config.m(), config.n(), config.k()), (3, 5, 7));
        assert_eq!(config.input_bytes(), 42);
        assert_eq!(config.weight_bytes(), 70);
        assert_eq!(config.output_bytes(), 30);
        assert_eq!(config.max_workspace_bytes(), 4096);
        assert_eq!(config.input_dtype(), CudaDType::BF16);
        assert_eq!(config.weight_dtype(), CudaDType::BF16);
        assert_eq!(config.accumulator_dtype(), CudaDType::F32);
        assert_eq!(config.output_dtype(), CudaDType::BF16);
        assert!(config.deterministic());
    }

    #[test]
    fn config_rejects_zero_and_out_of_native_range_dimensions() {
        let zero = CudaGemmConfig::new(0, 1, 1, 0).expect_err("zero M must fail");
        assert_eq!(zero.kind(), CudaErrorKind::InvalidArgument);

        let native_dimension_overflow =
            CudaGemmConfig::new(super::MAX_CUBLASLT_DIMENSION + 1, 1, 1, 0)
                .expect_err("cuBLASLt dimensions must fit i32");
        assert_eq!(native_dimension_overflow.kind(), CudaErrorKind::OutOfRange);
    }

    #[test]
    fn bf16_matrix_byte_arithmetic_rejects_u64_overflow() {
        let overflow = checked_bf16_matrix_bytes(u64::MAX, 2, "test")
            .expect_err("matrix byte arithmetic must be checked");
        assert_eq!(overflow.kind(), CudaErrorKind::OutOfRange);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn feature_off_gemm_error_is_actionable() {
        let error = crate::CudaError::unavailable("CudaContext::prepare_gemm");
        assert_eq!(error.kind(), CudaErrorKind::Unavailable);
        assert!(error.message().contains("--features cuda"));
    }
}
