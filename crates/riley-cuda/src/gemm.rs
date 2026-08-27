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
/// Version of the fixed-contiguous-37-balanced reduction contract.
pub const FIXED37_REDUCTION_VERSION: u32 = 1;
/// Elements traversed by each ascending local reduction chunk.
pub const FIXED37_CHUNK_ELEMENTS: u32 = 37;
/// Maximum number of chunk partials accepted by the additive backend.
pub const FIXED37_MAX_CHUNK_COUNT: u32 = 4096;
/// Largest logical reduction axis accepted by the additive backend.
pub const FIXED37_MAX_REDUCTION_ELEMENTS: u64 =
    (FIXED37_CHUNK_ELEMENTS as u64) * (FIXED37_MAX_CHUNK_COUNT as u64);
#[cfg(feature = "cuda")]
const NATIVE_CUBLASLT_BACKEND_ID: u32 = 1;
#[cfg(feature = "cuda")]
const NATIVE_FIXED37_BACKEND_ID: u32 = 2;
#[cfg(feature = "cuda")]
const DETERMINISTIC_REQUIRED: u32 = 1;
#[cfg(any(feature = "cuda", test))]
const REDUCTION_SCHEME_NONE: u32 = 0;
#[cfg(any(feature = "cuda", test))]
const REDUCTION_SCHEME_INPLACE: u32 = 1;
#[cfg(any(feature = "cuda", test))]
const REDUCTION_SCHEME_OUTPUT_TYPE: u32 = 4;

#[cfg(any(feature = "cuda", test))]
const fn is_deterministic_reduction_configuration(
    policy: CudaGemmReductionPolicy,
    split_k: u32,
    scheme: u32,
) -> bool {
    if split_k <= 1 {
        return scheme == REDUCTION_SCHEME_NONE;
    }
    match policy {
        CudaGemmReductionPolicy::StrictNoSplitV1 => false,
        CudaGemmReductionPolicy::AllowOutputTypeSplitKV1 => scheme == REDUCTION_SCHEME_OUTPUT_TYPE,
        CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1 => {
            scheme == REDUCTION_SCHEME_INPLACE || scheme == REDUCTION_SCHEME_OUTPUT_TYPE
        }
    }
}

/// Reviewed deterministic cuBLASLt reduction policy selected during prepare.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum CudaGemmReductionPolicy {
    /// Require split-K at most one with the `NONE` reduction scheme.
    #[default]
    StrictNoSplitV1,
    /// Also permit split-K greater than one with `OUTPUT_TYPE` reduction.
    AllowOutputTypeSplitKV1,
    /// Also permit reviewed `INPLACE` and `OUTPUT_TYPE` split-K reductions.
    ///
    /// cuBLASLt's `INPLACE` scheme uses output-type storage plus workspace
    /// counters to guarantee sequentiality. `OUTPUT_TYPE` stores output-type
    /// partials in workspace and reduces them in a separate deterministic
    /// step. This policy preserves either scheme when returned by the reviewed
    /// first heuristic instead of rewriting it to no-split.
    AllowInPlaceAndOutputTypeSplitKV1,
}

impl CudaGemmReductionPolicy {
    /// Stable diagnostic identifier.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::StrictNoSplitV1 => "strict-no-split-v1",
            Self::AllowOutputTypeSplitKV1 => "allow-output-type-split-k-v1",
            Self::AllowInPlaceAndOutputTypeSplitKV1 => "allow-in-place-and-output-type-split-k-v1",
        }
    }

    #[cfg(feature = "cuda")]
    const fn abi_flags(self) -> u32 {
        match self {
            Self::StrictNoSplitV1 => 0,
            Self::AllowOutputTypeSplitKV1 => 1,
            Self::AllowInPlaceAndOutputTypeSplitKV1 => 3,
        }
    }
}

/// Exact dense GEMM contract accepted by the PR 06 CUDA adapter.
///
/// The logical operation is row-major `Y[M, N] = X[M, K] * W[N, K]^T`.
/// Inputs, weights, and output are BF16, accumulation is F32, the epilogue is
/// disabled, and deterministic algorithm selection is mandatory. The only
/// selectable detail is the reviewed reduction policy; strict no-split is the
/// fail-closed default.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaGemmConfig {
    m: u64,
    n: u64,
    k: u64,
    max_workspace_bytes: u64,
    reduction_policy: CudaGemmReductionPolicy,
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
            reduction_policy: CudaGemmReductionPolicy::StrictNoSplitV1,
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

    /// Returns a copy resolved to the reviewed deterministic reduction policy.
    #[must_use]
    pub const fn with_reduction_policy(mut self, policy: CudaGemmReductionPolicy) -> Self {
        self.reduction_policy = policy;
        self
    }

    /// Deterministic reduction policy used during native algorithm selection.
    #[must_use]
    pub const fn reduction_policy(self) -> CudaGemmReductionPolicy {
        self.reduction_policy
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
                config.reduction_policy.abi_flags(),
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

    /// Prepares an exact-M child GEMM that preserves an existing plan's
    /// cuBLASLt reduction topology.
    ///
    /// This is intended for shape-bucketed execution where changing only M
    /// must not permit cuBLASLt to select a numerically different heuristic.
    /// The anchor must belong to this context and use the same GEMM contract
    /// other than M. Native validates the copied opaque algorithm against the
    /// child descriptors and fails closed without a heuristic fallback if that
    /// algorithm cannot execute for the requested shape.
    ///
    /// # Errors
    ///
    /// Returns invalid-state for a foreign or poisoned anchor; returns
    /// not-supported when the anchor algorithm is not valid for `config`.
    pub fn prepare_gemm_anchored(
        &self,
        config: CudaGemmConfig,
        anchor: &CudaPreparedGemm,
    ) -> CudaResult<CudaPreparedGemm> {
        const OPERATION: &str = "CudaContext::prepare_gemm_anchored";
        ensure_same_context(&self.inner, &anchor.context, OPERATION)?;
        if anchor.poisoned {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the anchor GEMM plan was poisoned by a prior native execution failure",
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let native = ffi::GemmPlanHandle::create_anchored(
                &self.inner.native,
                config.m,
                config.n,
                config.k,
                config.max_workspace_bytes,
                config.reduction_policy.abi_flags(),
                &anchor.native,
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
            Err(CudaError::unavailable(OPERATION))
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
            OPERATION,
            params.input.buffer(),
            params.input.dtype(),
            params.input.byte_offset(),
            params.input.byte_len(),
            CudaDType::BF16,
            self.config.input_bytes,
            "input",
        )?;
        validate_span(
            OPERATION,
            params.weight.buffer(),
            params.weight.dtype(),
            params.weight.byte_offset(),
            params.weight.byte_len(),
            CudaDType::BF16,
            self.config.weight_bytes,
            "weight",
        )?;
        validate_span(
            OPERATION,
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
                OPERATION,
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

/// Immutable provenance for the custom fixed37 GEMM implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CudaFixed37GemmMetadata {
    backend_id: u32,
    reduction_version: u32,
    chunk_elements: u32,
    accumulator_dtype: CudaDType,
    output_dtype: CudaDType,
    threads_per_block: u32,
    deterministic: bool,
    dynamic_shared_memory_bytes: u64,
    workspace_bytes: u64,
    m: u64,
    n: u64,
    k: u64,
}

impl CudaFixed37GemmMetadata {
    /// Stable native backend identifier for this custom implementation.
    pub const BACKEND_ID: u32 = 2;

    /// Native backend identifier.
    #[must_use]
    pub const fn backend_id(self) -> u32 {
        self.backend_id
    }

    /// Fixed reduction contract version.
    #[must_use]
    pub const fn reduction_version(self) -> u32 {
        self.reduction_version
    }

    /// Logical elements in every full local chunk.
    #[must_use]
    pub const fn chunk_elements(self) -> u32 {
        self.chunk_elements
    }

    /// Accumulation dtype, always F32.
    #[must_use]
    pub const fn accumulator_dtype(self) -> CudaDType {
        self.accumulator_dtype
    }

    /// Output dtype, always BF16.
    #[must_use]
    pub const fn output_dtype(self) -> CudaDType {
        self.output_dtype
    }

    /// CUDA threads in each custom kernel block.
    #[must_use]
    pub const fn threads_per_block(self) -> u32 {
        self.threads_per_block
    }

    /// Whether execution has a fixed deterministic reduction order.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Exact per-block dynamic shared-memory scratch.
    #[must_use]
    pub const fn dynamic_shared_memory_bytes(self) -> u64 {
        self.dynamic_shared_memory_bytes
    }

    /// Caller workspace bytes, always zero for the custom backend.
    #[must_use]
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }

    /// Prepared `(M, N, K)` dimensions.
    #[must_use]
    pub const fn dimensions(self) -> (u64, u64, u64) {
        (self.m, self.n, self.k)
    }
}

/// Borrowed buffers for one synchronous custom fixed37 GEMM execution.
///
/// Span sizes exactly match the prepared BF16 matrices, offsets are 256-byte
/// aligned, and the three allocations must be distinct. The implementation
/// never accepts a workspace and never falls back to canonical GEMM.
#[derive(Debug)]
pub struct Fixed37GemmParams<'a> {
    /// Row-major BF16 `X[M, K]`.
    pub input: CudaBufferSpan<'a>,
    /// Row-major BF16 `W[N, K]` consumed logically transposed.
    pub weight: CudaBufferSpan<'a>,
    /// Row-major BF16 `Y[M, N]`.
    pub output: CudaBufferSpanMut<'a>,
}

/// Owning custom fixed-contiguous-37-balanced-v1 GEMM plan.
///
/// This is deliberately a sibling of [`CudaPreparedGemm`], not a runtime
/// selector. It owns its additive native plan and performs no cuBLASLt query,
/// workspace allocation, or canonical fallback.
pub struct CudaPreparedFixed37Gemm {
    #[cfg(feature = "cuda")]
    native: ffi::Fixed37GemmPlanHandle,
    context: Arc<ContextInner>,
    config: CudaGemmConfig,
    metadata: CudaFixed37GemmMetadata,
    poisoned: bool,
    _not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for CudaPreparedFixed37Gemm {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaPreparedFixed37Gemm")
            .field("device_ordinal", &self.context.ordinal)
            .field("config", &self.config)
            .field("metadata", &self.metadata)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl CudaContext {
    /// Prepares the additive fixed37 GEMM implementation.
    ///
    /// # Errors
    ///
    /// Returns not-supported when `K` needs more than 4096 fixed chunks,
    /// unavailable without CUDA, or a translated native creation/contract
    /// error. No CUDA kernel or model is run during preparation.
    pub fn prepare_fixed37_gemm(
        &self,
        config: CudaGemmConfig,
    ) -> CudaResult<CudaPreparedFixed37Gemm> {
        #[cfg(feature = "cuda")]
        {
            if config.reduction_policy != CudaGemmReductionPolicy::StrictNoSplitV1 {
                return Err(fixed37_not_supported(
                    "CudaContext::prepare_fixed37_gemm",
                    "fixed37 has its own exact reduction contract and rejects cuBLASLt split-K policy flags",
                ));
            }
            if config.k > FIXED37_MAX_REDUCTION_ELEMENTS {
                return Err(fixed37_not_supported(
                    "CudaContext::prepare_fixed37_gemm",
                    format!(
                        "K={} requires more than {} fixed {}-element chunks",
                        config.k, FIXED37_MAX_CHUNK_COUNT, FIXED37_CHUNK_ELEMENTS
                    ),
                ));
            }
            let native = ffi::Fixed37GemmPlanHandle::create(
                &self.inner.native,
                config.m,
                config.n,
                config.k,
                config.max_workspace_bytes,
            )?;
            let metadata = CudaFixed37GemmMetadata::from_native(config, native.info()?)?;
            Ok(CudaPreparedFixed37Gemm {
                native,
                context: Arc::clone(&self.inner),
                config,
                metadata,
                poisoned: false,
                _not_sync: PhantomData,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = config;
            Err(CudaError::unavailable("CudaContext::prepare_fixed37_gemm"))
        }
    }
}

impl CudaPreparedFixed37Gemm {
    /// Exact logical and storage contract used to prepare this plan.
    #[must_use]
    pub const fn config(&self) -> CudaGemmConfig {
        self.config
    }

    /// Immutable fixed-reduction implementation metadata.
    #[must_use]
    pub const fn metadata(&self) -> CudaFixed37GemmMetadata {
        self.metadata
    }

    /// Device ordinal retained by this plan.
    #[must_use]
    pub fn device_ordinal(&self) -> u32 {
        self.context.ordinal
    }

    /// Whether a prior native execution error disabled safe reuse.
    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Executes the custom GEMM and synchronizes the explicit stream.
    ///
    /// # Errors
    ///
    /// Returns before native execution for dtype, size, alignment, context,
    /// alias, busy-buffer, or poisoned-plan violations. Any native execution
    /// failure poisons this safe wrapper conservatively.
    pub fn execute<S: CudaExecutionStream + ?Sized>(
        &mut self,
        params: &mut Fixed37GemmParams<'_>,
        stream: &mut S,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaPreparedFixed37Gemm::execute";
        let stream = execution_stream_mut(stream);
        if self.poisoned {
            return Err(CudaError::invalid_state(
                OPERATION,
                "the fixed37 GEMM plan was poisoned by a prior native execution failure",
            ));
        }
        self.validate_execution(params, stream)?;
        #[cfg(feature = "cuda")]
        {
            let result = self.native.execute(
                params.input.raw(),
                params.weight.raw(),
                params.output.raw(),
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
            Err(CudaError::unavailable(OPERATION))
        }
    }

    /// Explicitly closes the custom native plan.
    ///
    /// # Errors
    ///
    /// A poisoned plan cannot be reported as cleanly closed. Drop retains the
    /// existing fail-closed best-effort native cleanup behavior.
    pub fn close(self) -> CudaResult<()> {
        #[cfg(feature = "cuda")]
        {
            let mut this = self;
            if this.poisoned {
                return Err(CudaError::invalid_state(
                    "CudaPreparedFixed37Gemm::close",
                    "a poisoned fixed37 GEMM plan cannot be reported as cleanly closed",
                ));
            }
            this.native.close()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(CudaError::unavailable("CudaPreparedFixed37Gemm::close"))
        }
    }

    fn validate_execution(
        &self,
        params: &Fixed37GemmParams<'_>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "CudaPreparedFixed37Gemm::execute";
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        validate_span(
            OPERATION,
            params.input.buffer(),
            params.input.dtype(),
            params.input.byte_offset(),
            params.input.byte_len(),
            CudaDType::BF16,
            self.config.input_bytes,
            "input",
        )?;
        validate_span(
            OPERATION,
            params.weight.buffer(),
            params.weight.dtype(),
            params.weight.byte_offset(),
            params.weight.byte_len(),
            CudaDType::BF16,
            self.config.weight_bytes,
            "weight",
        )?;
        validate_span(
            OPERATION,
            params.output.buffer(),
            params.output.dtype(),
            params.output.byte_offset(),
            params.output.byte_len(),
            CudaDType::BF16,
            self.config.output_bytes,
            "output",
        )?;
        let buffers = [
            ("input", params.input.buffer()),
            ("weight", params.weight.buffer()),
            ("output", params.output.buffer()),
        ];
        for (_, buffer) in buffers {
            ensure_same_context(&self.context, buffer.context_owner(), OPERATION)?;
        }
        ensure_distinct_buffers(&buffers, OPERATION)
    }
}

#[cfg(feature = "cuda")]
impl CudaFixed37GemmMetadata {
    fn from_native(
        config: CudaGemmConfig,
        native: ffi::NativeFixed37GemmPlanInfo,
    ) -> CudaResult<Self> {
        const OPERATION: &str = "CudaContext::prepare_fixed37_gemm";
        let expected_chunks = config.k.div_ceil(u64::from(FIXED37_CHUNK_ELEMENTS));
        let expected_shared_bytes = expected_chunks * 2 * 4;
        if native.backend != NATIVE_FIXED37_BACKEND_ID
            || native.reduction_version != FIXED37_REDUCTION_VERSION
            || native.chunk_elements != FIXED37_CHUNK_ELEMENTS
            || native.accumulator_dtype != ffi::DTYPE_F32
            || native.output_dtype != ffi::DTYPE_BF16
            || native.threads_per_block != 256
            || native.deterministic != DETERMINISTIC_REQUIRED
            || native.workspace_bytes != 0
            || native.dynamic_shared_memory_bytes != expected_shared_bytes
            || (native.m, native.n, native.k) != (config.m, config.n, config.k)
        {
            return Err(CudaError::new(
                CudaErrorKind::Internal,
                CudaErrorDomain::Internal,
                CudaErrorStage::Prepare,
                0,
                OPERATION,
                format!("native fixed37 metadata violates the prepared contract: {native:?}"),
            ));
        }
        Ok(Self {
            backend_id: native.backend,
            reduction_version: native.reduction_version,
            chunk_elements: native.chunk_elements,
            accumulator_dtype: CudaDType::F32,
            output_dtype: CudaDType::BF16,
            threads_per_block: native.threads_per_block,
            deterministic: true,
            dynamic_shared_memory_bytes: native.dynamic_shared_memory_bytes,
            workspace_bytes: 0,
            m: native.m,
            n: native.n,
            k: native.k,
        })
    }
}

#[cfg(feature = "cuda")]
fn fixed37_not_supported(operation: &'static str, message: impl Into<String>) -> CudaError {
    CudaError::new(
        CudaErrorKind::NotSupported,
        CudaErrorDomain::Rust,
        CudaErrorStage::Validation,
        0,
        operation,
        message,
    )
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
        if !is_deterministic_reduction_configuration(
            config.reduction_policy,
            native.split_k,
            native.reduction_scheme,
        ) {
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
    operation: &'static str,
    buffer: &CudaDeviceBuffer,
    actual_dtype: CudaDType,
    byte_offset: u64,
    actual_bytes: u64,
    required_dtype: CudaDType,
    required_bytes: u64,
    name: &'static str,
) -> CudaResult<()> {
    if actual_dtype != required_dtype {
        return Err(CudaError::invalid_argument(
            operation,
            format!("{name} dtype must be {required_dtype}, got {actual_dtype}"),
        ));
    }
    if actual_bytes != required_bytes {
        return Err(CudaError::invalid_argument(
            operation,
            format!("{name} span must contain exactly {required_bytes} bytes, got {actual_bytes}"),
        ));
    }
    if byte_offset % GEMM_OFFSET_ALIGNMENT != 0 {
        return Err(CudaError::invalid_argument(
            operation,
            format!("{name} byte offset {byte_offset} is not {GEMM_OFFSET_ALIGNMENT}-byte aligned"),
        ));
    }
    buffer.ensure_idle_for_operation(operation)
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
    use super::{
        CudaGemmConfig, CudaGemmReductionPolicy, FIXED37_CHUNK_ELEMENTS, FIXED37_MAX_CHUNK_COUNT,
        FIXED37_MAX_REDUCTION_ELEMENTS, FIXED37_REDUCTION_VERSION, checked_bf16_matrix_bytes,
        is_deterministic_reduction_configuration,
    };
    use crate::{CudaDType, CudaErrorKind};

    #[test]
    fn config_pins_exact_dtype_and_checked_byte_arithmetic() {
        let config = CudaGemmConfig::new(3, 5, 7, 4096).expect("valid odd shape");
        assert_eq!((config.m(), config.n(), config.k()), (3, 5, 7));
        assert_eq!(config.input_bytes(), 42);
        assert_eq!(config.weight_bytes(), 70);
        assert_eq!(config.output_bytes(), 30);
        assert_eq!(config.max_workspace_bytes(), 4096);
        assert_eq!(
            config.reduction_policy(),
            CudaGemmReductionPolicy::StrictNoSplitV1
        );
        let relaxed =
            config.with_reduction_policy(CudaGemmReductionPolicy::AllowOutputTypeSplitKV1);
        assert_eq!(
            relaxed.reduction_policy(),
            CudaGemmReductionPolicy::AllowOutputTypeSplitKV1
        );
        assert_eq!(
            relaxed.reduction_policy().id(),
            "allow-output-type-split-k-v1"
        );
        let heuristic = config
            .with_reduction_policy(CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1);
        assert_eq!(
            heuristic.reduction_policy(),
            CudaGemmReductionPolicy::AllowInPlaceAndOutputTypeSplitKV1
        );
        assert_eq!(
            heuristic.reduction_policy().id(),
            "allow-in-place-and-output-type-split-k-v1"
        );
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

    #[test]
    fn deterministic_cublaslt_reduction_contract_is_policy_scoped() {
        use CudaGemmReductionPolicy::{
            AllowInPlaceAndOutputTypeSplitKV1, AllowOutputTypeSplitKV1, StrictNoSplitV1,
        };

        for accepted in [(0, 0), (1, 0)] {
            assert!(
                is_deterministic_reduction_configuration(StrictNoSplitV1, accepted.0, accepted.1),
                "strict configuration {accepted:?} was rejected"
            );
        }
        for rejected in [(0, 4), (1, 4), (2, 0), (2, 1), (2, 2), (2, 4), (2, 7)] {
            assert!(
                !is_deterministic_reduction_configuration(StrictNoSplitV1, rejected.0, rejected.1),
                "strict policy accepted {rejected:?}"
            );
        }
        for accepted in [(0, 0), (1, 0), (2, 4), (42, 4)] {
            assert!(
                is_deterministic_reduction_configuration(
                    AllowOutputTypeSplitKV1,
                    accepted.0,
                    accepted.1
                ),
                "output-type split-K configuration {accepted:?} was rejected"
            );
        }
        for rejected in [(0, 4), (1, 4), (2, 0), (2, 1), (2, 2), (2, 7)] {
            assert!(
                !is_deterministic_reduction_configuration(
                    AllowOutputTypeSplitKV1,
                    rejected.0,
                    rejected.1
                ),
                "output-type split-K policy accepted {rejected:?}"
            );
        }
        for accepted in [(0, 0), (1, 0), (2, 1), (3, 4), (42, 1), (42, 4)] {
            assert!(
                is_deterministic_reduction_configuration(
                    AllowInPlaceAndOutputTypeSplitKV1,
                    accepted.0,
                    accepted.1
                ),
                "reviewed heuristic split-K configuration {accepted:?} was rejected"
            );
        }
        for rejected in [(0, 1), (0, 4), (1, 1), (1, 4), (2, 0), (2, 2), (2, 7)] {
            assert!(
                !is_deterministic_reduction_configuration(
                    AllowInPlaceAndOutputTypeSplitKV1,
                    rejected.0,
                    rejected.1
                ),
                "reviewed heuristic split-K policy accepted {rejected:?}"
            );
        }
    }

    #[test]
    fn fixed37_constants_and_order_sensitive_witness_are_pinned() {
        assert_eq!(FIXED37_REDUCTION_VERSION, 1);
        assert_eq!(FIXED37_CHUNK_ELEMENTS, 37);
        assert_eq!(FIXED37_MAX_CHUNK_COUNT, 4096);
        assert_eq!(FIXED37_MAX_REDUCTION_ELEMENTS, 151_552);

        // A flat logical-order fold loses each unit after 2^24, while the
        // fixed37 local fold first creates a 37.0 chunk partial. The adjacent
        // merge therefore retains 36.0 after F32 round-to-nearest-even.
        let mut values = [0.0_f32; 74];
        values[0] = 16_777_216.0;
        values[37..].fill(1.0);
        let canonical_flat = values
            .iter()
            .copied()
            .fold(0.0_f32, |sum, value| sum + value);
        let mut partials = values
            .chunks(37)
            .map(|chunk| {
                chunk
                    .iter()
                    .copied()
                    .fold(0.0_f32, |sum, value| sum + value)
            })
            .collect::<Vec<_>>();
        while partials.len() > 1 {
            let mut merged = partials
                .chunks_exact(2)
                .map(|pair| pair[0] + pair[1])
                .collect::<Vec<_>>();
            if partials.len() % 2 != 0 {
                merged.push(*partials.last().expect("non-empty partials"));
            }
            partials = merged;
        }
        assert_eq!(canonical_flat.to_bits(), 16_777_216.0_f32.to_bits());
        assert_eq!(partials[0].to_bits(), 16_777_252.0_f32.to_bits());
        assert_ne!(canonical_flat.to_bits(), partials[0].to_bits());
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn feature_off_gemm_error_is_actionable() {
        let error = crate::CudaError::unavailable("CudaContext::prepare_gemm");
        assert_eq!(error.kind(), CudaErrorKind::Unavailable);
        assert!(error.message().contains("--features cuda"));
    }
}
