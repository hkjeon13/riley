//! Prepared native CUDA backends for dense full-sequence prefill attention.

use std::error;
use std::fmt;
use std::sync::Arc;

use crate::CUDA_COMPILED_ARCHITECTURES;
use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::gemm::{
    FIXED37_CHUNK_ELEMENTS, FIXED37_MAX_REDUCTION_ELEMENTS, FIXED37_REDUCTION_VERSION,
};
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{ContextInner, CudaContext, CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;

const BF16_BYTES: u64 = 2;
const MINIMUM_HARDWARE_COMPUTE_CAPABILITY: (u32, u32) = (8, 0);
const ONLINE_HEAD_SIZE: u64 = 64;
const REFERENCE_IMPLEMENTATION_ID: &str = "rustinfer.cuda.materialized-gqa-prefill.bf16";
const ONLINE_IMPLEMENTATION_ID: &str = "rustinfer.cuda.online-gqa-prefill.bf16.d64";
const FIXED37_MATERIALIZED_IMPLEMENTATION_ID: &str =
    "rustinfer.cuda.fixed37.materialized-gqa-prefill.bf16";
const FIXED37_TWO_PASS_IMPLEMENTATION_ID: &str =
    "rustinfer.cuda.fixed37.two-pass-gqa-prefill.bf16.d64.s8192";
const FIXED37_MAX_TWO_PASS_SEQUENCE: u64 = 8192;
const MAXIMUM_ONLINE_GRID_BATCH: u64 = 65_535;
const MAXIMUM_ONLINE_GRID_HEADS: u64 = 65_535;
const MAXIMUM_ONLINE_SEQUENCE_TILES: u64 = i32::MAX as u64;
const IMPLEMENTATION_VERSION: &str = "1";
const NATIVE_DEPENDENCY: &str = concat!(
    "rustinfer_cuda_native@abi1+cuda-architectures=",
    env!("RUSTINFER_CUDA_COMPILED_ARCHITECTURES"),
    "+cudart"
);

/// Attention execution family selected at cold preparation time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionMode {
    /// Full-sequence prompt processing without a key/value cache.
    Prefill,
}

/// Reduction order fixed into a prepared prefill implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionReductionProfile {
    /// Existing implementation-defined canonical reduction order.
    CanonicalV1,
    /// Ascending 37-element F32 left folds followed by an adjacent balanced
    /// binary tree with odd carry.
    FixedContiguous37BalancedV1,
}

/// Dense device layout accepted by the PR08 backends.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionLayout {
    /// Contiguous `[batch, sequence, head, depth]` row-major storage.
    DenseBshd,
}

/// Built-in causal masks accepted by prefill attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionMask {
    /// Attend to the current and every preceding token.
    Causal,
    /// Attend to at most `window` current-or-preceding tokens.
    ///
    /// A zero window defines a fully masked row for canonical online attention,
    /// whose output is all zeros. The fixed-contiguous-37-balanced-v1 profile
    /// rejects zero during cold selection because its literal finite-mask
    /// two-pass contract has no empty online state.
    CausalLocal {
        /// Number of visible tokens including the current token.
        window: u64,
    },
}

/// Production selection policy for one immutable prefill plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionPreference {
    /// Require the staged-BF16 native materialized reference.
    Reference,
    /// Prefer the online native backend and fall back during cold selection.
    Optimized,
}

/// Native implementation fixed into a prepared prefill plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionBackend {
    /// Four-stage QK, mask, softmax, and AV with a caller score workspace.
    MaterializedReference,
    /// Fused online-softmax CUDA attention without an HBM score matrix.
    Online,
    /// Materialized fixed37 QK, softmax, and AV with canonical score staging.
    Fixed37Materialized,
    /// No-HBM two-score-pass fixed37 attention for `D=64`, `S<=8192`.
    Fixed37TwoPass,
}

/// Why cold selection fixed a particular backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AttentionSelectionReason {
    /// The caller explicitly required the reference backend.
    ExplicitReference,
    /// The optimized backend was available and satisfied every capability.
    OptimizedCapabilityMatch,
    /// The optimized backend was not present in the supplied availability set.
    OptimizedUnavailableFallback,
    /// The linked CUDA code set cannot execute on the selected GPU.
    UnsupportedComputeCapabilityFallback,
    /// The requested head size was not supported by the online backend.
    UnsupportedHeadSizeFallback,
    /// CUDA Graph capture was requested but is not supported.
    UnsupportedGraphCaptureFallback,
    /// More than one key partition was requested without a merge backend.
    UnsupportedPartialMergeFallback,
    /// The request exceeds the fixed native online launch-grid contract.
    UnsupportedLaunchGeometryFallback,
    /// The two-pass fixed37 backend supports at most `S=8192`.
    UnsupportedSequenceLengthFallback,
    /// A zero local window is deliberately unsupported by fixed37 prefill.
    UnsupportedZeroLocalWindowFallback,
}

/// Which score representation an implementation writes to HBM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionScoreMaterialization {
    /// No complete score or probability matrix is written to global memory.
    None,
    /// A full BF16 score matrix is transformed in place into BF16 probabilities.
    FullStagedBf16,
}

/// Static capability declaration for a built-in native backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct AttentionCapability {
    implementation_id: &'static str,
    implementation_version: &'static str,
    native_dependency: &'static str,
    mode: AttentionMode,
    layout: AttentionLayout,
    input_dtype: CudaDType,
    accumulator_dtype: CudaDType,
    output_dtype: CudaDType,
    minimum_hardware_compute_capability: (u32, u32),
    compiled_architectures: &'static str,
    head_size: Option<u64>,
    causal: bool,
    causal_local: bool,
    minimum_local_window_size: Option<u64>,
    variable_sequence: bool,
    non_contiguous: bool,
    cuda_graph_capture: bool,
    online_reduction: bool,
    partial_state_merge: bool,
    score_materialization: AttentionScoreMaterialization,
    reduction_profile: AttentionReductionProfile,
    reduction_version: Option<u32>,
    reduction_chunk_elements: Option<u64>,
    maximum_reduction_elements: Option<u64>,
}

impl AttentionCapability {
    /// Stable implementation identifier used by traces and benchmarks.
    #[must_use]
    pub const fn implementation_id(self) -> &'static str {
        self.implementation_id
    }

    /// Backend contract version, independent of the crate version.
    #[must_use]
    pub const fn implementation_version(self) -> &'static str {
        self.implementation_version
    }

    /// Native runtime dependency required by the implementation.
    #[must_use]
    pub const fn native_dependency(self) -> &'static str {
        self.native_dependency
    }

    /// Supported attention execution family.
    #[must_use]
    pub const fn mode(self) -> AttentionMode {
        self.mode
    }

    /// Supported dense layout.
    #[must_use]
    pub const fn layout(self) -> AttentionLayout {
        self.layout
    }

    /// Required query, key, and value scalar type.
    #[must_use]
    pub const fn input_dtype(self) -> CudaDType {
        self.input_dtype
    }

    /// Reduction and dot-product accumulator type.
    #[must_use]
    pub const fn accumulator_dtype(self) -> CudaDType {
        self.accumulator_dtype
    }

    /// Attention output scalar type.
    #[must_use]
    pub const fn output_dtype(self) -> CudaDType {
        self.output_dtype
    }

    /// Effective hardware floor and lowest compiled `(major, minor)` target.
    ///
    /// This is provenance rather than a complete compatibility predicate:
    /// real-only code is not forward compatible across major versions. Use
    /// [`Self::supports_compute_capability`] for selection decisions.
    #[must_use]
    pub fn minimum_compute_capability(self) -> (u32, u32) {
        self.minimum_hardware_compute_capability
            .max(minimum_architecture(self.compiled_architectures))
    }

    /// Normalized `CMake` CUDA architecture set compiled into the native archive.
    #[must_use]
    pub const fn compiled_architectures(self) -> &'static str {
        self.compiled_architectures
    }

    /// Whether the compiled real/virtual code can execute on `actual`.
    #[must_use]
    pub fn supports_compute_capability(self, actual: (u32, u32)) -> bool {
        compute_capability_at_least(actual, self.minimum_hardware_compute_capability)
            && architecture_set_supports(self.compiled_architectures, actual)
    }

    /// Required exact head size, or `None` for any positive head size.
    #[must_use]
    pub const fn head_size(self) -> Option<u64> {
        self.head_size
    }

    /// Whether full causal masking is supported.
    #[must_use]
    pub const fn supports_causal(self) -> bool {
        self.causal
    }

    /// Whether causal local-window masking is supported.
    #[must_use]
    pub const fn supports_causal_local(self) -> bool {
        self.causal_local
    }

    /// Minimum supported causal-local window, or `None` when local masking is
    /// unsupported. Canonical online accepts zero; fixed two-pass starts at one.
    #[must_use]
    pub const fn minimum_local_window_size(self) -> Option<u64> {
        self.minimum_local_window_size
    }

    /// Whether sequence length is a runtime dimension.
    #[must_use]
    pub const fn supports_variable_sequence(self) -> bool {
        self.variable_sequence
    }

    /// Whether non-contiguous Q/K/V/output views are accepted.
    #[must_use]
    pub const fn supports_non_contiguous(self) -> bool {
        self.non_contiguous
    }

    /// Whether execution can occur during CUDA Graph capture.
    #[must_use]
    pub const fn supports_cuda_graph_capture(self) -> bool {
        self.cuda_graph_capture
    }

    /// Whether softmax is accumulated online without revisiting global scores.
    #[must_use]
    pub const fn uses_online_reduction(self) -> bool {
        self.online_reduction
    }

    /// Whether multiple partial `(m, l, n)` states can be merged natively.
    #[must_use]
    pub const fn supports_partial_state_merge(self) -> bool {
        self.partial_state_merge
    }

    /// Full-score HBM materialization contract.
    #[must_use]
    pub const fn score_materialization(self) -> AttentionScoreMaterialization {
        self.score_materialization
    }

    /// Reduction order implemented by this backend.
    #[must_use]
    pub const fn reduction_profile(self) -> AttentionReductionProfile {
        self.reduction_profile
    }

    /// Fixed reduction contract version, or `None` for canonical order.
    #[must_use]
    pub const fn reduction_version(self) -> Option<u32> {
        self.reduction_version
    }

    /// Logical elements per fixed chunk, or `None` for canonical order.
    #[must_use]
    pub const fn reduction_chunk_elements(self) -> Option<u64> {
        self.reduction_chunk_elements
    }

    /// Largest fixed reduction axis, or `None` for canonical order.
    #[must_use]
    pub const fn maximum_reduction_elements(self) -> Option<u64> {
        self.maximum_reduction_elements
    }
}

/// Which statically linked backends may participate in cold selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)]
pub struct AttentionBackendAvailability {
    reference: bool,
    online: bool,
    fixed37_materialized: bool,
    fixed37_two_pass: bool,
}

impl AttentionBackendAvailability {
    /// Creates an explicit availability snapshot for deterministic selection.
    #[must_use]
    pub const fn new(reference: bool, online: bool) -> Self {
        Self {
            reference,
            online,
            fixed37_materialized: false,
            fixed37_two_pass: false,
        }
    }

    /// Adds the independently linked fixed37 materialized and two-pass paths.
    #[must_use]
    pub const fn with_fixed37(mut self, materialized: bool, two_pass: bool) -> Self {
        self.fixed37_materialized = materialized;
        self.fixed37_two_pass = two_pass;
        self
    }

    /// Backends linked into the current crate feature set.
    #[must_use]
    pub const fn linked() -> Self {
        Self::new(cfg!(feature = "cuda"), cfg!(feature = "cuda"))
            .with_fixed37(cfg!(feature = "cuda"), cfg!(feature = "cuda"))
    }

    /// Whether the materialized native reference is available.
    #[must_use]
    pub const fn reference(self) -> bool {
        self.reference
    }

    /// Whether the online native implementation is available.
    #[must_use]
    pub const fn online(self) -> bool {
        self.online
    }

    /// Whether the materialized fixed37 sibling set is linked.
    #[must_use]
    pub const fn fixed37_materialized(self) -> bool {
        self.fixed37_materialized
    }

    /// Whether the no-HBM fixed37 two-pass implementation is linked.
    #[must_use]
    pub const fn fixed37_two_pass(self) -> bool {
        self.fixed37_two_pass
    }
}

/// Immutable dimensions and execution requirements used for cold selection.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrefillAttentionRequest {
    mode: AttentionMode,
    layout: AttentionLayout,
    batch_size: u64,
    sequence_length: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    mask: AttentionMask,
    graph_capture: bool,
    key_partition_count: u32,
}

impl PrefillAttentionRequest {
    /// Builds the dense BSHD contract for one fixed prefill shape.
    #[must_use]
    pub const fn new(
        batch_size: u64,
        sequence_length: u64,
        query_head_count: u64,
        key_value_head_count: u64,
        head_size: u64,
        scale: f32,
        mask: AttentionMask,
    ) -> Self {
        Self {
            mode: AttentionMode::Prefill,
            layout: AttentionLayout::DenseBshd,
            batch_size,
            sequence_length,
            query_head_count,
            key_value_head_count,
            head_size,
            scale,
            mask,
            graph_capture: false,
            key_partition_count: 1,
        }
    }

    /// Requests or disables CUDA Graph capture capability.
    #[must_use]
    pub const fn with_graph_capture(mut self, requested: bool) -> Self {
        self.graph_capture = requested;
        self
    }

    /// Requests a future split-K partition count; PR08 implementations accept one.
    #[must_use]
    pub const fn with_key_partition_count(mut self, count: u32) -> Self {
        self.key_partition_count = count;
        self
    }

    /// Attention execution family.
    #[must_use]
    pub const fn mode(self) -> AttentionMode {
        self.mode
    }

    /// Dense input and output layout.
    #[must_use]
    pub const fn layout(self) -> AttentionLayout {
        self.layout
    }

    /// Number of independent dense sequences.
    #[must_use]
    pub const fn batch_size(self) -> u64 {
        self.batch_size
    }

    /// Query, key, and value sequence length.
    #[must_use]
    pub const fn sequence_length(self) -> u64 {
        self.sequence_length
    }

    /// Number of query/output heads.
    #[must_use]
    pub const fn query_head_count(self) -> u64 {
        self.query_head_count
    }

    /// Number of key/value heads.
    #[must_use]
    pub const fn key_value_head_count(self) -> u64 {
        self.key_value_head_count
    }

    /// Elements in one attention head.
    #[must_use]
    pub const fn head_size(self) -> u64 {
        self.head_size
    }

    /// Positive finite score scale.
    #[must_use]
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// Requested causal mask.
    #[must_use]
    pub const fn mask(self) -> AttentionMask {
        self.mask
    }

    /// Whether CUDA Graph capture support is required.
    #[must_use]
    pub const fn graph_capture_requested(self) -> bool {
        self.graph_capture
    }

    /// Requested number of independent key partitions.
    #[must_use]
    pub const fn key_partition_count(self) -> u32 {
        self.key_partition_count
    }
}

/// Immutable cold-selection evidence for one prepared attention path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionSelectionTrace {
    reason: AttentionSelectionReason,
    implementation_id: &'static str,
    implementation_version: &'static str,
    native_dependency: &'static str,
    compiled_architectures: &'static str,
    device_ordinal: u32,
    compute_capability: (u32, u32),
    score_materialization: AttentionScoreMaterialization,
    materialized_score_bytes: u64,
    workspace_bytes: u64,
    layout_copy_bytes: u64,
    reduction_profile: AttentionReductionProfile,
    dynamic_shared_memory_bytes: u64,
}

impl AttentionSelectionTrace {
    /// Deterministic selection or fallback reason.
    #[must_use]
    pub const fn reason(self) -> AttentionSelectionReason {
        self.reason
    }

    /// Selected stable implementation identifier.
    #[must_use]
    pub const fn implementation_id(self) -> &'static str {
        self.implementation_id
    }

    /// Selected implementation contract version.
    #[must_use]
    pub const fn implementation_version(self) -> &'static str {
        self.implementation_version
    }

    /// Selected native runtime dependency.
    #[must_use]
    pub const fn native_dependency(self) -> &'static str {
        self.native_dependency
    }

    /// Normalized native CUDA architecture set used for compatibility checks.
    #[must_use]
    pub const fn compiled_architectures(self) -> &'static str {
        self.compiled_architectures
    }

    /// Device ordinal whose context owner is bound into the prepared plan.
    #[must_use]
    pub const fn device_ordinal(self) -> u32 {
        self.device_ordinal
    }

    /// Actual device compute capability observed during cold selection.
    #[must_use]
    pub const fn compute_capability(self) -> (u32, u32) {
        self.compute_capability
    }

    /// Selected full-score materialization strategy.
    #[must_use]
    pub const fn score_materialization(self) -> AttentionScoreMaterialization {
        self.score_materialization
    }

    /// Logical bytes written as full score/probability matrices per execution.
    #[must_use]
    pub const fn materialized_score_bytes(self) -> u64 {
        self.materialized_score_bytes
    }

    /// Caller-owned device workspace required by the selected backend.
    #[must_use]
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }

    /// Bytes copied solely to satisfy the selected layout contract.
    #[must_use]
    pub const fn layout_copy_bytes(self) -> u64 {
        self.layout_copy_bytes
    }

    /// Immutable reduction profile selected before execution.
    #[must_use]
    pub const fn reduction_profile(self) -> AttentionReductionProfile {
        self.reduction_profile
    }

    /// Maximum dynamic shared memory used by one selected attention kernel.
    #[must_use]
    pub const fn dynamic_shared_memory_bytes(self) -> u64 {
        self.dynamic_shared_memory_bytes
    }
}

/// Q/K/V, output, and optional caller-owned workspace for one execution.
#[derive(Debug)]
pub struct PrefillAttentionParams<'a> {
    /// BF16 contiguous `[B,S,QH,D]` query view.
    pub query: CudaBufferSpan<'a>,
    /// BF16 contiguous `[B,S,KVH,D]` key view.
    pub key: CudaBufferSpan<'a>,
    /// BF16 contiguous `[B,S,KVH,D]` value view.
    pub value: CudaBufferSpan<'a>,
    /// BF16 contiguous `[B,S,QH,D]` output view.
    pub output: CudaBufferSpanMut<'a>,
    /// Reference-only BF16 `[QH,S,S]` scratch, reused across batches.
    pub workspace: Option<CudaBufferSpanMut<'a>>,
}

/// One backend and execution contract fixed before any hot execution.
#[derive(Clone)]
pub struct PreparedPrefillAttention {
    context: Arc<ContextInner>,
    request: PrefillAttentionRequest,
    backend: AttentionBackend,
    capability: AttentionCapability,
    trace: AttentionSelectionTrace,
    reduction_profile: AttentionReductionProfile,
}

impl fmt::Debug for PreparedPrefillAttention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedPrefillAttention")
            .field("device_ordinal", &self.context.ordinal)
            .field("request", &self.request)
            .field("backend", &self.backend)
            .field("capability", &self.capability)
            .field("trace", &self.trace)
            .field("reduction_profile", &self.reduction_profile)
            .finish()
    }
}

impl PreparedPrefillAttention {
    /// Selects a compatible backend without allocating or executing CUDA work.
    ///
    /// Optimized fallback occurs only in this cold method. [`Self::execute`]
    /// never retries a native failure with another backend.
    ///
    /// # Errors
    ///
    /// Returns for invalid dimensions/scalars, arithmetic overflow, unavailable
    /// required backends, or a request unsupported by both native paths.
    pub fn select(
        context: &CudaContext,
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_with_reduction_profile(
            context,
            request,
            preference,
            AttentionReductionProfile::CanonicalV1,
            availability,
        )
    }

    /// Selects a compatible backend within one immutable reduction profile.
    ///
    /// The fixed profile may fall back only from fixed37 two-pass to fixed37
    /// materialized attention during this cold call. It never selects either
    /// canonical backend.
    ///
    /// # Errors
    ///
    /// Returns when the requested profile has no compatible linked backend;
    /// canonical cross-profile fallback is deliberately forbidden.
    pub fn select_with_reduction_profile(
        context: &CudaContext,
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        let compute_capability = context.compute_capability();
        Self::select_for_compute_capability_and_reduction_profile(
            context,
            compute_capability,
            request,
            preference,
            reduction_profile,
            availability,
        )
    }

    #[cfg(test)]
    fn select_for_compute_capability(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_for_compute_capability_and_reduction_profile(
            context,
            compute_capability,
            request,
            preference,
            AttentionReductionProfile::CanonicalV1,
            availability,
        )
    }

    fn select_for_compute_capability_and_reduction_profile(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_request(request)?;

        match reduction_profile {
            AttentionReductionProfile::CanonicalV1 => Self::select_canonical(
                context,
                compute_capability,
                request,
                preference,
                availability,
            ),
            AttentionReductionProfile::FixedContiguous37BalancedV1 => Self::select_fixed37(
                context,
                compute_capability,
                request,
                preference,
                availability,
            ),
        }
    }

    fn select_canonical(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        match preference {
            AttentionPreference::Reference => {
                if !availability.reference {
                    return Err(not_supported(
                        "select_prefill_attention",
                        "the explicitly requested materialized reference backend is unavailable",
                    ));
                }
                require_reference_support(request, compute_capability)?;
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    AttentionBackend::MaterializedReference,
                    AttentionSelectionReason::ExplicitReference,
                    AttentionReductionProfile::CanonicalV1,
                )
            }
            AttentionPreference::Optimized => {
                let online_reason = if !availability.online {
                    AttentionSelectionReason::OptimizedUnavailableFallback
                } else if !ONLINE_CAPABILITY.supports_compute_capability(compute_capability) {
                    AttentionSelectionReason::UnsupportedComputeCapabilityFallback
                } else if request.head_size != ONLINE_HEAD_SIZE {
                    AttentionSelectionReason::UnsupportedHeadSizeFallback
                } else if request.graph_capture {
                    AttentionSelectionReason::UnsupportedGraphCaptureFallback
                } else if request.key_partition_count != 1 {
                    AttentionSelectionReason::UnsupportedPartialMergeFallback
                } else if !online_launch_geometry_supported(request) {
                    AttentionSelectionReason::UnsupportedLaunchGeometryFallback
                } else {
                    return prepare_selection(
                        context,
                        compute_capability,
                        request,
                        AttentionBackend::Online,
                        AttentionSelectionReason::OptimizedCapabilityMatch,
                        AttentionReductionProfile::CanonicalV1,
                    );
                };

                if !availability.reference {
                    return Err(not_supported(
                        "select_prefill_attention",
                        format!(
                            "optimized backend was rejected ({online_reason:?}) and the native reference is unavailable"
                        ),
                    ));
                }
                require_reference_support(request, compute_capability).map_err(|reference_error| {
                    not_supported(
                        "select_prefill_attention",
                        format!(
                            "optimized backend was rejected ({online_reason:?}) and the native reference cannot satisfy the request: {}",
                            reference_error.message()
                        ),
                    )
                })?;
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    AttentionBackend::MaterializedReference,
                    online_reason,
                    AttentionReductionProfile::CanonicalV1,
                )
            }
        }
    }

    fn select_fixed37(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<Self> {
        const PROFILE: AttentionReductionProfile =
            AttentionReductionProfile::FixedContiguous37BalancedV1;
        match preference {
            AttentionPreference::Reference => {
                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_prefill_attention",
                        "the fixed37 materialized backend is unavailable; canonical fallback is forbidden",
                    ));
                }
                require_fixed37_materialized_support(request, compute_capability)?;
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    AttentionBackend::Fixed37Materialized,
                    AttentionSelectionReason::ExplicitReference,
                    PROFILE,
                )
            }
            AttentionPreference::Optimized => {
                let two_pass_reason = if !availability.fixed37_two_pass {
                    AttentionSelectionReason::OptimizedUnavailableFallback
                } else if !FIXED37_TWO_PASS_CAPABILITY
                    .supports_compute_capability(compute_capability)
                {
                    AttentionSelectionReason::UnsupportedComputeCapabilityFallback
                } else if request.head_size != ONLINE_HEAD_SIZE {
                    AttentionSelectionReason::UnsupportedHeadSizeFallback
                } else if request.sequence_length > FIXED37_MAX_TWO_PASS_SEQUENCE {
                    AttentionSelectionReason::UnsupportedSequenceLengthFallback
                } else if matches!(request.mask, AttentionMask::CausalLocal { window: 0 }) {
                    AttentionSelectionReason::UnsupportedZeroLocalWindowFallback
                } else if request.graph_capture {
                    AttentionSelectionReason::UnsupportedGraphCaptureFallback
                } else if request.key_partition_count != 1 {
                    AttentionSelectionReason::UnsupportedPartialMergeFallback
                } else {
                    return prepare_selection(
                        context,
                        compute_capability,
                        request,
                        AttentionBackend::Fixed37TwoPass,
                        AttentionSelectionReason::OptimizedCapabilityMatch,
                        PROFILE,
                    );
                };

                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_prefill_attention",
                        format!(
                            "fixed37 two-pass was rejected ({two_pass_reason:?}), fixed37 materialized is unavailable, and canonical fallback is forbidden"
                        ),
                    ));
                }
                require_fixed37_materialized_support(request, compute_capability).map_err(
                    |materialized_error| {
                        not_supported(
                            "select_prefill_attention",
                            format!(
                                "fixed37 two-pass was rejected ({two_pass_reason:?}) and fixed37 materialized cannot satisfy the request: {}; canonical fallback is forbidden",
                                materialized_error.message()
                            ),
                        )
                    },
                )?;
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    AttentionBackend::Fixed37Materialized,
                    two_pass_reason,
                    PROFILE,
                )
            }
        }
    }

    /// Fixed request used during cold selection.
    #[must_use]
    pub const fn request(&self) -> PrefillAttentionRequest {
        self.request
    }

    /// Selected native implementation.
    #[must_use]
    pub const fn backend(&self) -> AttentionBackend {
        self.backend
    }

    /// Capability declaration of the selected implementation.
    #[must_use]
    pub const fn capability(&self) -> AttentionCapability {
        self.capability
    }

    /// Immutable selection and memory trace.
    #[must_use]
    pub const fn selection_trace(&self) -> AttentionSelectionTrace {
        self.trace
    }

    /// Required caller-owned workspace bytes.
    #[must_use]
    pub const fn workspace_bytes(&self) -> u64 {
        self.trace.workspace_bytes
    }

    /// Immutable reduction profile selected for every reduction in this plan.
    #[must_use]
    pub const fn reduction_profile(&self) -> AttentionReductionProfile {
        self.reduction_profile
    }

    /// Executes exactly the backend selected during [`Self::select`].
    ///
    /// Native launch or synchronization failure is returned directly. The
    /// method deliberately performs no post-launch fallback.
    ///
    /// # Errors
    ///
    /// Returns for dtype, capacity, context, workspace, launch, or completion
    /// failure.
    pub fn execute(
        &self,
        params: &mut PrefillAttentionParams<'_>,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "prefill_attention";
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        validate_execute_params(self, params, stream)?;

        #[cfg(feature = "cuda")]
        match self.backend {
            AttentionBackend::MaterializedReference => {
                let workspace = params.workspace.as_mut().ok_or_else(|| {
                    CudaError::invalid_argument(OPERATION, "reference workspace is missing")
                })?;
                ffi::prefill_attention_reference_execute(
                    params.query.raw(),
                    params.key.raw(),
                    params.value.raw(),
                    params.output.raw(),
                    workspace.raw(),
                    self.request.batch_size,
                    self.request.sequence_length,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            AttentionBackend::Online => ffi::prefill_attention_execute(
                params.query.raw(),
                params.key.raw(),
                params.value.raw(),
                params.output.raw(),
                self.request.batch_size,
                self.request.sequence_length,
                self.request.query_head_count,
                self.request.key_value_head_count,
                self.request.head_size,
                self.request.scale,
                mask_kind(self.request.mask),
                local_window(self.request.mask),
                &mut stream.native,
            ),
            AttentionBackend::Fixed37Materialized => {
                let workspace = params.workspace.as_mut().ok_or_else(|| {
                    CudaError::invalid_argument(
                        OPERATION,
                        "fixed37 materialized workspace is missing",
                    )
                })?;
                ffi::prefill_attention_fixed37_materialized_execute(
                    params.query.raw(),
                    params.key.raw(),
                    params.value.raw(),
                    params.output.raw(),
                    workspace.raw(),
                    self.request.batch_size,
                    self.request.sequence_length,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            AttentionBackend::Fixed37TwoPass => ffi::fixed37_prefill_attention_execute(
                params.query.raw(),
                params.key.raw(),
                params.value.raw(),
                params.output.raw(),
                self.request.batch_size,
                self.request.sequence_length,
                self.request.query_head_count,
                self.request.key_value_head_count,
                self.request.head_size,
                self.request.scale,
                mask_kind(self.request.mask),
                local_window(self.request.mask),
                &mut stream.native,
            ),
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = params;
            Err(CudaError::unavailable(OPERATION))
        }
    }
}

/// Stable CPU online-softmax state for reference merge tests.
///
/// NaN scores or values are rejected before mutation. Negative infinity is a
/// masked element. Positive-infinity scores are supported: only values tied at
/// positive infinity contribute, with equal weight.
#[derive(Clone, Debug, PartialEq)]
pub struct OnlineSoftmaxState {
    maximum: f32,
    normalizer: f32,
    numerator: Vec<f32>,
}

impl OnlineSoftmaxState {
    /// Creates an empty `(m=-inf, l=0, n=0)` state.
    ///
    /// # Errors
    ///
    /// Returns when `value_width` is zero or its accumulator allocation cannot
    /// be reserved.
    pub fn new(value_width: usize) -> Result<Self, OnlineSoftmaxError> {
        if value_width == 0 {
            return Err(OnlineSoftmaxError::ZeroValueWidth);
        }
        let mut numerator = Vec::new();
        numerator
            .try_reserve_exact(value_width)
            .map_err(|_| OnlineSoftmaxError::AllocationFailed { value_width })?;
        numerator.resize(value_width, 0.0);
        Ok(Self {
            maximum: f32::NEG_INFINITY,
            normalizer: 0.0,
            numerator,
        })
    }

    /// Number of values accumulated in `n` for each score.
    #[must_use]
    pub fn value_width(&self) -> usize {
        self.numerator.len()
    }

    /// Current maximum score, or negative infinity while empty.
    #[must_use]
    pub const fn maximum(&self) -> f32 {
        self.maximum
    }

    /// Current shifted exponential denominator.
    #[must_use]
    pub const fn normalizer(&self) -> f32 {
        self.normalizer
    }

    /// Whether no unmasked score has been accumulated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.normalizer == 0.0
    }

    /// Adds one score/value pair to this online state.
    ///
    /// # Errors
    ///
    /// Returns before mutation for width mismatch or any NaN input.
    pub fn accumulate(&mut self, score: f32, value: &[f32]) -> Result<(), OnlineSoftmaxError> {
        self.validate_value(score, value)?;
        if score == f32::NEG_INFINITY {
            return Ok(());
        }
        if self.is_empty() || score == f32::INFINITY && self.maximum != f32::INFINITY {
            self.maximum = score;
            self.normalizer = 1.0;
            self.numerator.copy_from_slice(value);
            return Ok(());
        }
        if self.maximum == f32::INFINITY {
            if score == f32::INFINITY {
                self.normalizer += 1.0;
                for (numerator, &item) in self.numerator.iter_mut().zip(value) {
                    *numerator += item;
                }
            }
            return Ok(());
        }

        let merged_maximum = self.maximum.max(score);
        let old_scale = (self.maximum - merged_maximum).exp();
        let item_scale = (score - merged_maximum).exp();
        self.normalizer = old_scale.mul_add(self.normalizer, item_scale);
        for (numerator, &item) in self.numerator.iter_mut().zip(value) {
            *numerator = old_scale.mul_add(*numerator, item_scale * item);
        }
        self.maximum = merged_maximum;
        Ok(())
    }

    /// Merges another disjoint score partition without revisiting its scores.
    ///
    /// # Errors
    ///
    /// Returns before mutation when the value widths differ.
    pub fn merge(&mut self, other: &Self) -> Result<(), OnlineSoftmaxError> {
        if self.value_width() != other.value_width() {
            return Err(OnlineSoftmaxError::ValueWidthMismatch {
                expected: self.value_width(),
                actual: other.value_width(),
            });
        }
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            self.maximum = other.maximum;
            self.normalizer = other.normalizer;
            self.numerator.copy_from_slice(&other.numerator);
            return Ok(());
        }
        if self.maximum == f32::INFINITY || other.maximum == f32::INFINITY {
            match (
                self.maximum == f32::INFINITY,
                other.maximum == f32::INFINITY,
            ) {
                (true, true) => {
                    self.normalizer += other.normalizer;
                    for (left, &right) in self.numerator.iter_mut().zip(&other.numerator) {
                        *left += right;
                    }
                }
                (false, true) => {
                    self.maximum = f32::INFINITY;
                    self.normalizer = other.normalizer;
                    self.numerator.copy_from_slice(&other.numerator);
                }
                (true, false) => {}
                (false, false) => unreachable!(),
            }
            return Ok(());
        }

        let merged_maximum = self.maximum.max(other.maximum);
        let left_scale = (self.maximum - merged_maximum).exp();
        let right_scale = (other.maximum - merged_maximum).exp();
        self.normalizer = left_scale.mul_add(self.normalizer, right_scale * other.normalizer);
        for (left, &right) in self.numerator.iter_mut().zip(&other.numerator) {
            *left = left_scale.mul_add(*left, right_scale * right);
        }
        self.maximum = merged_maximum;
        Ok(())
    }

    /// Writes `n/l`, or all zeros for a fully masked row.
    ///
    /// # Errors
    ///
    /// Returns before writing when the destination width differs.
    pub fn finalize(&self, output: &mut [f32]) -> Result<(), OnlineSoftmaxError> {
        if output.len() != self.value_width() {
            return Err(OnlineSoftmaxError::ValueWidthMismatch {
                expected: self.value_width(),
                actual: output.len(),
            });
        }
        if self.is_empty() {
            output.fill(0.0);
        } else {
            for (destination, &numerator) in output.iter_mut().zip(&self.numerator) {
                *destination = numerator / self.normalizer;
            }
        }
        Ok(())
    }

    fn validate_value(&self, score: f32, value: &[f32]) -> Result<(), OnlineSoftmaxError> {
        if value.len() != self.value_width() {
            return Err(OnlineSoftmaxError::ValueWidthMismatch {
                expected: self.value_width(),
                actual: value.len(),
            });
        }
        if score.is_nan() {
            return Err(OnlineSoftmaxError::NaNScore);
        }
        if let Some(index) = value.iter().position(|item| item.is_nan()) {
            return Err(OnlineSoftmaxError::NaNValue { index });
        }
        Ok(())
    }
}

/// Stable validation failure from [`OnlineSoftmaxState`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OnlineSoftmaxError {
    /// Value vectors must contain at least one component.
    ZeroValueWidth,
    /// The requested numerator accumulator could not be reserved.
    AllocationFailed {
        /// Number of F32 accumulator elements requested.
        value_width: usize,
    },
    /// A value vector or output had the wrong width.
    ValueWidthMismatch {
        /// Width fixed by the state.
        expected: usize,
        /// Width supplied by the caller.
        actual: usize,
    },
    /// NaN scores are rejected rather than silently poisoning the state.
    NaNScore,
    /// A value vector contained NaN.
    NaNValue {
        /// First NaN component.
        index: usize,
    },
}

impl fmt::Display for OnlineSoftmaxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroValueWidth => formatter.write_str("online softmax value width is zero"),
            Self::AllocationFailed { value_width } => write!(
                formatter,
                "online softmax could not reserve {value_width} F32 accumulator elements"
            ),
            Self::ValueWidthMismatch { expected, actual } => write!(
                formatter,
                "online softmax value width requires {expected}, got {actual}"
            ),
            Self::NaNScore => formatter.write_str("online softmax score is NaN"),
            Self::NaNValue { index } => {
                write!(formatter, "online softmax value at index {index} is NaN")
            }
        }
    }
}

impl error::Error for OnlineSoftmaxError {}

const REFERENCE_CAPABILITY: AttentionCapability = AttentionCapability {
    implementation_id: REFERENCE_IMPLEMENTATION_ID,
    implementation_version: IMPLEMENTATION_VERSION,
    native_dependency: NATIVE_DEPENDENCY,
    mode: AttentionMode::Prefill,
    layout: AttentionLayout::DenseBshd,
    input_dtype: CudaDType::BF16,
    accumulator_dtype: CudaDType::F32,
    output_dtype: CudaDType::BF16,
    minimum_hardware_compute_capability: MINIMUM_HARDWARE_COMPUTE_CAPABILITY,
    compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
    head_size: None,
    causal: true,
    causal_local: false,
    minimum_local_window_size: None,
    variable_sequence: true,
    non_contiguous: false,
    cuda_graph_capture: false,
    online_reduction: false,
    partial_state_merge: false,
    score_materialization: AttentionScoreMaterialization::FullStagedBf16,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const ONLINE_CAPABILITY: AttentionCapability = AttentionCapability {
    implementation_id: ONLINE_IMPLEMENTATION_ID,
    implementation_version: IMPLEMENTATION_VERSION,
    native_dependency: NATIVE_DEPENDENCY,
    mode: AttentionMode::Prefill,
    layout: AttentionLayout::DenseBshd,
    input_dtype: CudaDType::BF16,
    accumulator_dtype: CudaDType::F32,
    output_dtype: CudaDType::BF16,
    minimum_hardware_compute_capability: MINIMUM_HARDWARE_COMPUTE_CAPABILITY,
    compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
    head_size: Some(ONLINE_HEAD_SIZE),
    causal: true,
    causal_local: true,
    minimum_local_window_size: Some(0),
    variable_sequence: true,
    non_contiguous: false,
    cuda_graph_capture: false,
    online_reduction: true,
    partial_state_merge: false,
    score_materialization: AttentionScoreMaterialization::None,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const FIXED37_MATERIALIZED_CAPABILITY: AttentionCapability = AttentionCapability {
    implementation_id: FIXED37_MATERIALIZED_IMPLEMENTATION_ID,
    implementation_version: IMPLEMENTATION_VERSION,
    native_dependency: NATIVE_DEPENDENCY,
    mode: AttentionMode::Prefill,
    layout: AttentionLayout::DenseBshd,
    input_dtype: CudaDType::BF16,
    accumulator_dtype: CudaDType::F32,
    output_dtype: CudaDType::BF16,
    minimum_hardware_compute_capability: MINIMUM_HARDWARE_COMPUTE_CAPABILITY,
    compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
    head_size: None,
    causal: true,
    causal_local: false,
    minimum_local_window_size: None,
    variable_sequence: true,
    non_contiguous: false,
    cuda_graph_capture: false,
    online_reduction: false,
    partial_state_merge: false,
    score_materialization: AttentionScoreMaterialization::FullStagedBf16,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_REDUCTION_ELEMENTS),
};

const FIXED37_TWO_PASS_CAPABILITY: AttentionCapability = AttentionCapability {
    implementation_id: FIXED37_TWO_PASS_IMPLEMENTATION_ID,
    implementation_version: IMPLEMENTATION_VERSION,
    native_dependency: NATIVE_DEPENDENCY,
    mode: AttentionMode::Prefill,
    layout: AttentionLayout::DenseBshd,
    input_dtype: CudaDType::BF16,
    accumulator_dtype: CudaDType::F32,
    output_dtype: CudaDType::BF16,
    minimum_hardware_compute_capability: MINIMUM_HARDWARE_COMPUTE_CAPABILITY,
    compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
    head_size: Some(ONLINE_HEAD_SIZE),
    causal: true,
    causal_local: true,
    minimum_local_window_size: Some(1),
    variable_sequence: true,
    non_contiguous: false,
    cuda_graph_capture: false,
    // This is a two-score-pass fixed tree, not an online recurrence.
    online_reduction: false,
    partial_state_merge: false,
    score_materialization: AttentionScoreMaterialization::None,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_TWO_PASS_SEQUENCE),
};

fn prepare_selection(
    context: &CudaContext,
    compute_capability: (u32, u32),
    request: PrefillAttentionRequest,
    backend: AttentionBackend,
    reason: AttentionSelectionReason,
    reduction_profile: AttentionReductionProfile,
) -> CudaResult<PreparedPrefillAttention> {
    let capability = match backend {
        AttentionBackend::MaterializedReference => REFERENCE_CAPABILITY,
        AttentionBackend::Online => ONLINE_CAPABILITY,
        AttentionBackend::Fixed37Materialized => FIXED37_MATERIALIZED_CAPABILITY,
        AttentionBackend::Fixed37TwoPass => FIXED37_TWO_PASS_CAPABILITY,
    };
    let workspace_bytes = match backend {
        AttentionBackend::MaterializedReference | AttentionBackend::Fixed37Materialized => {
            reference_workspace_bytes(request)?
        }
        AttentionBackend::Online | AttentionBackend::Fixed37TwoPass => 0,
    };
    let materialized_score_bytes = match backend {
        AttentionBackend::MaterializedReference | AttentionBackend::Fixed37Materialized => request
            .batch_size
            .checked_mul(workspace_bytes)
            .ok_or_else(|| {
                CudaError::out_of_range(
                    "select_prefill_attention",
                    "materialized attention byte count overflows u64",
                )
            })?,
        AttentionBackend::Online | AttentionBackend::Fixed37TwoPass => 0,
    };
    let dynamic_shared_memory_bytes = match backend {
        AttentionBackend::MaterializedReference | AttentionBackend::Online => 0,
        AttentionBackend::Fixed37Materialized => {
            fixed37_reduction_shared_bytes(request.head_size.max(request.sequence_length))?
        }
        AttentionBackend::Fixed37TwoPass => fixed37_two_pass_shared_bytes(request.sequence_length)?,
    };
    Ok(PreparedPrefillAttention {
        context: Arc::clone(&context.inner),
        request,
        backend,
        capability,
        trace: AttentionSelectionTrace {
            reason,
            implementation_id: capability.implementation_id,
            implementation_version: capability.implementation_version,
            native_dependency: capability.native_dependency,
            compiled_architectures: capability.compiled_architectures,
            device_ordinal: context.device_ordinal(),
            compute_capability,
            score_materialization: capability.score_materialization,
            materialized_score_bytes,
            workspace_bytes,
            layout_copy_bytes: 0,
            reduction_profile,
            dynamic_shared_memory_bytes,
        },
        reduction_profile,
    })
}

fn fixed37_reduction_shared_bytes(element_count: u64) -> CudaResult<u64> {
    let chunks = element_count
        .checked_add(u64::from(FIXED37_CHUNK_ELEMENTS) - 1)
        .ok_or_else(|| {
            CudaError::out_of_range(
                "select_prefill_attention",
                "fixed37 chunk-count arithmetic overflow",
            )
        })?
        / u64::from(FIXED37_CHUNK_ELEMENTS);
    checked_product("select_prefill_attention", &[chunks, 2, 4])
}

fn fixed37_two_pass_shared_bytes(sequence_length: u64) -> CudaResult<u64> {
    let scores = checked_product("select_prefill_attention", &[sequence_length, 4])?;
    scores
        .checked_add(fixed37_reduction_shared_bytes(
            sequence_length.max(ONLINE_HEAD_SIZE),
        )?)
        .ok_or_else(|| {
            CudaError::out_of_range(
                "select_prefill_attention",
                "fixed37 two-pass shared-memory arithmetic overflow",
            )
        })
}

fn validate_request(request: PrefillAttentionRequest) -> CudaResult<()> {
    const OPERATION: &str = "select_prefill_attention";
    for (name, value) in [
        ("batch_size", request.batch_size),
        ("sequence_length", request.sequence_length),
        ("query_head_count", request.query_head_count),
        ("key_value_head_count", request.key_value_head_count),
        ("head_size", request.head_size),
    ] {
        if value == 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("{name} must be greater than zero"),
            ));
        }
    }
    if request.query_head_count % request.key_value_head_count != 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "key_value_head_count must divide query_head_count",
        ));
    }
    if !request.scale.is_finite() || request.scale <= 0.0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "scale must be finite and greater than zero",
        ));
    }
    if request.key_partition_count == 0 {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "key_partition_count must be greater than zero",
        ));
    }
    required_tensor_bytes(request, request.query_head_count)?;
    required_tensor_bytes(request, request.key_value_head_count)?;
    Ok(())
}

fn require_reference_support(
    request: PrefillAttentionRequest,
    compute_capability: (u32, u32),
) -> CudaResult<()> {
    if !REFERENCE_CAPABILITY.supports_compute_capability(compute_capability) {
        return Err(not_supported(
            "select_prefill_attention",
            format!(
                "compute capability {}.{} does not satisfy the prefill hardware floor {}.{} and compiled CUDA architectures {CUDA_COMPILED_ARCHITECTURES}",
                compute_capability.0,
                compute_capability.1,
                MINIMUM_HARDWARE_COMPUTE_CAPABILITY.0,
                MINIMUM_HARDWARE_COMPUTE_CAPABILITY.1,
            ),
        ));
    }
    if !matches!(request.mask, AttentionMask::Causal) {
        return Err(not_supported(
            "select_prefill_attention",
            "materialized reference supports full causal masking only",
        ));
    }
    if request.graph_capture {
        return Err(not_supported(
            "select_prefill_attention",
            "materialized reference synchronizes its stream and cannot be graph captured",
        ));
    }
    if request.key_partition_count != 1 {
        return Err(not_supported(
            "select_prefill_attention",
            "materialized reference does not support partial-state merge",
        ));
    }
    Ok(())
}

fn require_fixed37_materialized_support(
    request: PrefillAttentionRequest,
    compute_capability: (u32, u32),
) -> CudaResult<()> {
    if !FIXED37_MATERIALIZED_CAPABILITY.supports_compute_capability(compute_capability) {
        return Err(not_supported(
            "select_prefill_attention",
            format!(
                "compute capability {}.{} cannot execute the fixed37 materialized CUDA targets {CUDA_COMPILED_ARCHITECTURES}",
                compute_capability.0, compute_capability.1
            ),
        ));
    }
    if !matches!(request.mask, AttentionMask::Causal) {
        return Err(not_supported(
            "select_prefill_attention",
            "fixed37 materialized attention supports full causal masking only",
        ));
    }
    if request.head_size > FIXED37_MAX_REDUCTION_ELEMENTS {
        return Err(not_supported(
            "select_prefill_attention",
            format!(
                "fixed37 QK depth {} exceeds the reduction-axis limit {FIXED37_MAX_REDUCTION_ELEMENTS}",
                request.head_size
            ),
        ));
    }
    if request.sequence_length > FIXED37_MAX_REDUCTION_ELEMENTS {
        return Err(not_supported(
            "select_prefill_attention",
            format!(
                "fixed37 softmax/AV sequence {} exceeds the reduction-axis limit {FIXED37_MAX_REDUCTION_ELEMENTS}",
                request.sequence_length
            ),
        ));
    }
    if request.graph_capture {
        return Err(not_supported(
            "select_prefill_attention",
            "fixed37 materialized attention synchronizes its stream and cannot be graph captured",
        ));
    }
    if request.key_partition_count != 1 {
        return Err(not_supported(
            "select_prefill_attention",
            "fixed37 materialized attention does not support partial-state merge",
        ));
    }
    Ok(())
}

fn validate_execute_params(
    prepared: &PreparedPrefillAttention,
    params: &PrefillAttentionParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "prefill_attention";
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key", params.key.dtype()),
        ("value", params.value.dtype()),
        ("output", params.output.dtype()),
    ] {
        if dtype != CudaDType::BF16 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("{name} must be bf16, got {dtype}"),
            ));
        }
    }

    let query_bytes = required_tensor_bytes(prepared.request, prepared.request.query_head_count)?;
    let key_value_bytes =
        required_tensor_bytes(prepared.request, prepared.request.key_value_head_count)?;
    require_capacity(OPERATION, "query", params.query.byte_len(), query_bytes)?;
    require_capacity(OPERATION, "key", params.key.byte_len(), key_value_bytes)?;
    require_capacity(OPERATION, "value", params.value.byte_len(), key_value_bytes)?;
    require_capacity(OPERATION, "output", params.output.byte_len(), query_bytes)?;

    validate_resource(OPERATION, stream, params.query.buffer())?;
    validate_resource(OPERATION, stream, params.key.buffer())?;
    validate_resource(OPERATION, stream, params.value.buffer())?;
    validate_resource(OPERATION, stream, params.output.buffer())?;

    match prepared.backend {
        AttentionBackend::MaterializedReference | AttentionBackend::Fixed37Materialized => {
            let workspace = params.workspace.as_ref().ok_or_else(|| {
                CudaError::invalid_argument(
                    OPERATION,
                    "materialized attention workspace is missing",
                )
            })?;
            if workspace.dtype() != CudaDType::BF16 {
                return Err(CudaError::invalid_argument(
                    OPERATION,
                    format!("workspace must be bf16, got {}", workspace.dtype()),
                ));
            }
            require_capacity(
                OPERATION,
                "workspace",
                workspace.byte_len(),
                prepared.workspace_bytes(),
            )?;
            validate_resource(OPERATION, stream, workspace.buffer())?;
        }
        AttentionBackend::Online | AttentionBackend::Fixed37TwoPass => {
            if params.workspace.is_some() {
                return Err(CudaError::invalid_argument(
                    OPERATION,
                    "non-materialized attention requires no workspace; pass None",
                ));
            }
        }
    }
    Ok(())
}

fn validate_resource(
    operation: &'static str,
    stream: &CudaStream,
    buffer: &CudaDeviceBuffer,
) -> CudaResult<()> {
    ensure_same_context(buffer.context_owner(), &stream.context, operation)?;
    buffer.ensure_idle_for_operation(operation)
}

fn require_capacity(
    operation: &'static str,
    name: &'static str,
    actual: u64,
    required: u64,
) -> CudaResult<()> {
    if actual < required {
        Err(CudaError::out_of_range(
            operation,
            format!("{name} requires {required} bytes, span exposes {actual}"),
        ))
    } else {
        Ok(())
    }
}

fn required_tensor_bytes(request: PrefillAttentionRequest, head_count: u64) -> CudaResult<u64> {
    checked_product(
        "select_prefill_attention",
        &[
            request.batch_size,
            request.sequence_length,
            head_count,
            request.head_size,
            BF16_BYTES,
        ],
    )
}

fn reference_workspace_bytes(request: PrefillAttentionRequest) -> CudaResult<u64> {
    checked_product(
        "select_prefill_attention",
        &[
            request.query_head_count,
            request.sequence_length,
            request.sequence_length,
            BF16_BYTES,
        ],
    )
}

fn checked_product(operation: &'static str, factors: &[u64]) -> CudaResult<u64> {
    factors.iter().try_fold(1_u64, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            CudaError::out_of_range(operation, "prefill byte-length arithmetic overflow")
        })
    })
}

const fn compute_capability_at_least(actual: (u32, u32), minimum: (u32, u32)) -> bool {
    actual.0 > minimum.0 || actual.0 == minimum.0 && actual.1 >= minimum.1
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArchitectureEmission {
    Real,
    Virtual,
    RealAndVirtual,
}

fn architecture_target(token: &str) -> Option<((u32, u32), ArchitectureEmission)> {
    let (numeric, emission) = if let Some(numeric) = token.strip_suffix("-real") {
        (numeric, ArchitectureEmission::Real)
    } else if let Some(numeric) = token.strip_suffix("-virtual") {
        (numeric, ArchitectureEmission::Virtual)
    } else {
        (token, ArchitectureEmission::RealAndVirtual)
    };
    let encoded = numeric.parse::<u32>().ok()?;
    Some(((encoded / 10, encoded % 10), emission))
}

fn minimum_architecture(architectures: &str) -> (u32, u32) {
    architectures
        .split(';')
        .filter_map(architecture_target)
        .map(|(target, _)| target)
        .min()
        // build.rs validates and canonicalizes this compile-time string. Keep
        // the capability fail-closed if that invariant is ever broken.
        .unwrap_or((u32::MAX, u32::MAX))
}

fn architecture_set_supports(architectures: &str, actual: (u32, u32)) -> bool {
    architectures.split(';').any(|token| {
        architecture_target(token).is_some_and(|(target, emission)| match emission {
            // CUDA cubins are binary compatible with later minor revisions in
            // the same major family, but not with a different major family.
            ArchitectureEmission::Real => actual.0 == target.0 && actual.1 >= target.1,
            // PTX is forward compatible. A plain CMake target emits both real
            // and virtual code and therefore follows the same broad predicate.
            ArchitectureEmission::Virtual | ArchitectureEmission::RealAndVirtual => {
                compute_capability_at_least(actual, target)
            }
        })
    })
}

const fn online_launch_geometry_supported(request: PrefillAttentionRequest) -> bool {
    let final_tile = if request.sequence_length % 8 == 0 {
        0
    } else {
        1
    };
    let sequence_tiles = request.sequence_length / 8 + final_tile;
    request.batch_size <= MAXIMUM_ONLINE_GRID_BATCH
        && request.query_head_count <= MAXIMUM_ONLINE_GRID_HEADS
        && sequence_tiles <= MAXIMUM_ONLINE_SEQUENCE_TILES
}

#[cfg(feature = "cuda")]
const fn mask_kind(mask: AttentionMask) -> u32 {
    match mask {
        AttentionMask::Causal => ffi::PREFILL_MASK_CAUSAL,
        AttentionMask::CausalLocal { .. } => ffi::PREFILL_MASK_CAUSAL_LOCAL,
    }
}

#[cfg(feature = "cuda")]
const fn local_window(mask: AttentionMask) -> u64 {
    match mask {
        AttentionMask::Causal => 0,
        AttentionMask::CausalLocal { window } => window,
    }
}

fn not_supported(operation: &'static str, message: impl Into<String>) -> CudaError {
    CudaError::new(
        CudaErrorKind::NotSupported,
        CudaErrorDomain::Rust,
        CudaErrorStage::Prepare,
        0,
        operation,
        message,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "cuda"))]
    fn request(mask: AttentionMask) -> PrefillAttentionRequest {
        PrefillAttentionRequest::new(2, 128, 8, 2, 64, 0.125, mask)
    }

    #[cfg(not(feature = "cuda"))]
    fn test_context(ordinal: u32) -> CudaContext {
        CudaContext {
            inner: Arc::new(ContextInner {
                ordinal,
                compute_capability: minimum_architecture(CUDA_COMPILED_ARCHITECTURES),
            }),
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn select_for_test(
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<PreparedPrefillAttention> {
        let context = test_context(7);
        PreparedPrefillAttention::select_for_compute_capability(
            &context,
            minimum_architecture(CUDA_COMPILED_ARCHITECTURES),
            request,
            preference,
            availability,
        )
    }

    #[cfg(not(feature = "cuda"))]
    fn select_fixed37_for_test(
        request: PrefillAttentionRequest,
        preference: AttentionPreference,
        availability: AttentionBackendAvailability,
    ) -> CudaResult<PreparedPrefillAttention> {
        let context = test_context(7);
        PreparedPrefillAttention::select_for_compute_capability_and_reduction_profile(
            &context,
            minimum_architecture(CUDA_COMPILED_ARCHITECTURES),
            request,
            preference,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn compiled_architecture_compatibility_distinguishes_real_and_virtual_code() {
        assert!(architecture_set_supports("80-real", (8, 0)));
        assert!(architecture_set_supports("80-real", (8, 9)));
        assert!(!architecture_set_supports("80-real", (9, 0)));
        assert!(!architecture_set_supports("90-real", (8, 9)));
        assert!(architecture_set_supports("80-virtual", (9, 0)));
        assert!(architecture_set_supports("80", (10, 0)));
        assert!(architecture_set_supports("90-real;89-real", (8, 9)));
        assert_eq!(minimum_architecture("90-real;89-real"), (8, 9));
        assert!(!ONLINE_CAPABILITY.supports_compute_capability((7, 9)));
        assert!(compute_capability_at_least(
            ONLINE_CAPABILITY.minimum_compute_capability(),
            MINIMUM_HARDWARE_COMPUTE_CAPABILITY
        ));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn prepared_selection_is_bound_to_the_exact_context_owner() {
        let selected_context = test_context(0);
        let other_owner_same_ordinal = test_context(0);
        let prepared = PreparedPrefillAttention::select_for_compute_capability(
            &selected_context,
            minimum_architecture(CUDA_COMPILED_ARCHITECTURES),
            request(AttentionMask::Causal),
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap();

        assert!(Arc::ptr_eq(&prepared.context, &selected_context.inner));
        ensure_same_context(&prepared.context, &selected_context.inner, "test").unwrap();
        let error = ensure_same_context(&prepared.context, &other_owner_same_ordinal.inner, "test")
            .expect_err("a plan must not execute through another context owner");
        assert_eq!(error.kind(), CudaErrorKind::InvalidState);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn compiled_architecture_mismatch_fails_closed_for_both_backends() {
        let context = test_context(0);
        let error = PreparedPrefillAttention::select_for_compute_capability(
            &context,
            (1, 0),
            request(AttentionMask::Causal),
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .expect_err("no configured CUDA target can execute on compute capability 1.0");
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(error.message().contains(CUDA_COMPILED_ARCHITECTURES));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn online_selection_records_zero_materialization_and_workspace() {
        let prepared = select_for_test(
            request(AttentionMask::Causal),
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(prepared.backend(), AttentionBackend::Online);
        assert_eq!(prepared.workspace_bytes(), 0);
        let trace = prepared.selection_trace();
        assert_eq!(
            trace.reason(),
            AttentionSelectionReason::OptimizedCapabilityMatch
        );
        assert_eq!(
            trace.score_materialization(),
            AttentionScoreMaterialization::None
        );
        assert_eq!(trace.materialized_score_bytes(), 0);
        assert_eq!(trace.layout_copy_bytes(), 0);
        assert_eq!(trace.device_ordinal(), 7);
        assert_eq!(
            trace.compute_capability(),
            minimum_architecture(CUDA_COMPILED_ARCHITECTURES)
        );
        assert_eq!(trace.compiled_architectures(), CUDA_COMPILED_ARCHITECTURES);
        assert!(
            trace
                .native_dependency()
                .contains(CUDA_COMPILED_ARCHITECTURES)
        );
        let capability = prepared.capability();
        assert_eq!(capability.input_dtype(), CudaDType::BF16);
        assert_eq!(capability.accumulator_dtype(), CudaDType::F32);
        assert_eq!(capability.output_dtype(), CudaDType::BF16);
        assert_eq!(capability.head_size(), Some(64));
        assert_eq!(
            capability.compiled_architectures(),
            CUDA_COMPILED_ARCHITECTURES
        );
        assert!(capability.supports_compute_capability(trace.compute_capability()));
        assert!(capability.uses_online_reduction());
        assert_eq!(capability.minimum_local_window_size(), Some(0));
        assert!(!capability.supports_cuda_graph_capture());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn unavailable_online_falls_back_cold_to_materialized_reference() {
        let prepared = select_for_test(
            request(AttentionMask::Causal),
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, false),
        )
        .unwrap();
        assert_eq!(prepared.backend(), AttentionBackend::MaterializedReference);
        assert_eq!(prepared.workspace_bytes(), 8 * 128 * 128 * 2);
        let trace = prepared.selection_trace();
        assert_eq!(
            trace.reason(),
            AttentionSelectionReason::OptimizedUnavailableFallback
        );
        assert_eq!(trace.materialized_score_bytes(), 2 * 8 * 128 * 128 * 2);
        assert_eq!(
            trace.score_materialization(),
            AttentionScoreMaterialization::FullStagedBf16
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn unsupported_online_head_size_falls_back_but_local_mask_fails_explicitly() {
        let unsupported_head =
            PrefillAttentionRequest::new(1, 7, 6, 2, 32, 0.125, AttentionMask::Causal);
        let prepared = select_for_test(
            unsupported_head,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(prepared.backend(), AttentionBackend::MaterializedReference);
        assert_eq!(
            prepared.selection_trace().reason(),
            AttentionSelectionReason::UnsupportedHeadSizeFallback
        );

        let local = request(AttentionMask::CausalLocal { window: 0 });
        let selected = select_for_test(
            local,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(selected.backend(), AttentionBackend::Online);
        let error = select_for_test(
            local,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, false),
        )
        .unwrap_err();
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(error.message().contains("full causal"));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn selector_rejects_invalid_shapes_scalars_and_future_execution_modes() {
        let invalid_requests = [
            PrefillAttentionRequest::new(0, 1, 1, 1, 64, 0.125, AttentionMask::Causal),
            PrefillAttentionRequest::new(1, 1, 6, 4, 64, 0.125, AttentionMask::Causal),
            PrefillAttentionRequest::new(1, 1, 1, 1, 64, f32::NAN, AttentionMask::Causal),
        ];
        for invalid in invalid_requests {
            let error = select_for_test(
                invalid,
                AttentionPreference::Optimized,
                AttentionBackendAvailability::new(true, true),
            )
            .unwrap_err();
            assert_eq!(error.kind(), CudaErrorKind::InvalidArgument);
        }

        for unsupported in [
            request(AttentionMask::Causal).with_graph_capture(true),
            request(AttentionMask::Causal).with_key_partition_count(2),
        ] {
            let error = select_for_test(
                unsupported,
                AttentionPreference::Optimized,
                AttentionBackendAvailability::new(true, true),
            )
            .unwrap_err();
            assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn online_selection_does_not_require_quadratic_reference_workspace() {
        let online_only =
            PrefillAttentionRequest::new(1, 4_000_000_000, 1, 1, 64, 0.125, AttentionMask::Causal);
        let selected = select_for_test(
            online_only,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(false, true),
        )
        .expect("online selection must not calculate an S-squared workspace");
        assert_eq!(selected.backend(), AttentionBackend::Online);
        assert_eq!(selected.workspace_bytes(), 0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn online_launch_geometry_is_rejected_during_cold_selection() {
        let too_many_batches = PrefillAttentionRequest::new(
            MAXIMUM_ONLINE_GRID_BATCH + 1,
            1,
            1,
            1,
            64,
            0.125,
            AttentionMask::Causal,
        );
        let fallback = select_for_test(
            too_many_batches,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(fallback.backend(), AttentionBackend::MaterializedReference);
        assert_eq!(
            fallback.selection_trace().reason(),
            AttentionSelectionReason::UnsupportedLaunchGeometryFallback
        );

        let too_many_tiles = PrefillAttentionRequest::new(
            1,
            MAXIMUM_ONLINE_SEQUENCE_TILES * 8 + 1,
            1,
            1,
            64,
            0.125,
            AttentionMask::CausalLocal { window: 1 },
        );
        let error = select_for_test(
            too_many_tiles,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true),
        )
        .unwrap_err();
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(
            error
                .message()
                .contains("UnsupportedLaunchGeometryFallback")
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_two_pass_selection_pins_profile_capability_and_shared_memory() {
        let fixed_request = PrefillAttentionRequest::new(
            1,
            FIXED37_MAX_TWO_PASS_SEQUENCE,
            8,
            2,
            ONLINE_HEAD_SIZE,
            0.125,
            AttentionMask::Causal,
        );
        let prepared = select_fixed37_for_test(
            fixed_request,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
        )
        .expect("the maximum fixed37 two-pass shape must select without allocating");

        assert_eq!(prepared.backend(), AttentionBackend::Fixed37TwoPass);
        assert_eq!(
            prepared.reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(prepared.workspace_bytes(), 0);
        let trace = prepared.selection_trace();
        assert_eq!(trace.workspace_bytes(), 0);
        assert_eq!(trace.materialized_score_bytes(), 0);
        assert_eq!(trace.dynamic_shared_memory_bytes(), 34_544);
        assert_eq!(
            trace.reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert!(trace.implementation_id().contains("fixed37.two-pass"));
        let capability = prepared.capability();
        assert_eq!(
            capability.reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            capability.reduction_version(),
            Some(FIXED37_REDUCTION_VERSION)
        );
        assert_eq!(
            capability.reduction_chunk_elements(),
            Some(u64::from(FIXED37_CHUNK_ELEMENTS))
        );
        assert_eq!(
            capability.maximum_reduction_elements(),
            Some(FIXED37_MAX_TWO_PASS_SEQUENCE)
        );
        assert!(!capability.uses_online_reduction());
        assert_eq!(capability.minimum_local_window_size(), Some(1));
        assert_eq!(
            capability.score_materialization(),
            AttentionScoreMaterialization::None
        );

        for (sequence_length, expected_shared_bytes) in [(1, 20), (36, 160), (37, 164), (38, 168)] {
            let short = PrefillAttentionRequest::new(
                1,
                sequence_length,
                1,
                1,
                ONLINE_HEAD_SIZE,
                0.125,
                AttentionMask::Causal,
            );
            let short = select_fixed37_for_test(
                short,
                AttentionPreference::Optimized,
                AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
            )
            .expect("short fixed37 two-pass shape must select");
            assert_eq!(
                short.selection_trace().dynamic_shared_memory_bytes(),
                expected_shared_bytes,
                "the shared partial arrays must hold both D64 and S reductions"
            );
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_fallback_stays_inside_profile_and_reports_exact_workspace() {
        let non_d64 = PrefillAttentionRequest::new(2, 128, 8, 2, 74, 0.125, AttentionMask::Causal);
        let prepared = select_fixed37_for_test(
            non_d64,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
        )
        .expect("unsupported two-pass D must fall back to fixed37 materialized");
        assert_eq!(prepared.backend(), AttentionBackend::Fixed37Materialized);
        assert_eq!(prepared.workspace_bytes(), 2 * 8 * 128 * 128);
        assert_eq!(
            prepared.selection_trace().materialized_score_bytes(),
            2 * 2 * 8 * 128 * 128
        );
        assert_eq!(
            prepared.selection_trace().reason(),
            AttentionSelectionReason::UnsupportedHeadSizeFallback
        );
        assert!(
            prepared
                .selection_trace()
                .implementation_id()
                .contains("fixed37.materialized")
        );

        let error = select_fixed37_for_test(
            non_d64,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true).with_fixed37(false, true),
        )
        .expect_err("canonical availability must never rescue a fixed37 request");
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(error.message().contains("canonical fallback is forbidden"));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_local_masks_keep_global_chunks_and_zero_window_fails_closed() {
        for window in [1, 37, u64::MAX] {
            let local = PrefillAttentionRequest::new(
                1,
                75,
                8,
                2,
                ONLINE_HEAD_SIZE,
                0.125,
                AttentionMask::CausalLocal { window },
            );
            let prepared = select_fixed37_for_test(
                local,
                AttentionPreference::Optimized,
                AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
            )
            .expect("non-zero fixed37 local windows must use two-pass");
            assert_eq!(prepared.backend(), AttentionBackend::Fixed37TwoPass);
        }

        let local_zero = PrefillAttentionRequest::new(
            1,
            75,
            8,
            2,
            ONLINE_HEAD_SIZE,
            0.125,
            AttentionMask::CausalLocal { window: 0 },
        );
        let error = select_fixed37_for_test(
            local_zero,
            AttentionPreference::Optimized,
            AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
        )
        .expect_err("fixed37 local-window zero must fail instead of inventing uniform output");
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(error.message().contains("full causal masking only"));
        assert!(error.message().contains("canonical fallback is forbidden"));
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_axis_limits_and_explicit_materialized_selection_are_fail_closed() {
        let explicit = select_fixed37_for_test(
            request(AttentionMask::Causal),
            AttentionPreference::Reference,
            AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
        )
        .expect("explicit fixed37 reference means fixed37 materialized");
        assert_eq!(explicit.backend(), AttentionBackend::Fixed37Materialized);

        for unsupported in [
            PrefillAttentionRequest::new(
                1,
                FIXED37_MAX_REDUCTION_ELEMENTS + 1,
                1,
                1,
                64,
                0.125,
                AttentionMask::Causal,
            ),
            PrefillAttentionRequest::new(
                1,
                1,
                1,
                1,
                FIXED37_MAX_REDUCTION_ELEMENTS + 1,
                0.125,
                AttentionMask::Causal,
            ),
        ] {
            let error = select_fixed37_for_test(
                unsupported,
                AttentionPreference::Reference,
                AttentionBackendAvailability::new(true, true).with_fixed37(true, true),
            )
            .expect_err("a fixed37 reduction axis over the limit must fail at selection");
            assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        }
    }

    #[test]
    fn online_state_matches_single_and_partitioned_stable_softmax() {
        let scores = [-1000.0_f32, -1.0, 0.0, 1.0, 1000.0];
        let values = [
            [1.0_f32, -2.0],
            [2.0, -1.0],
            [3.0, 0.0],
            [4.0, 1.0],
            [5.0, 2.0],
        ];
        let mut whole = OnlineSoftmaxState::new(2).unwrap();
        for (&score, value) in scores.iter().zip(&values) {
            whole.accumulate(score, value).unwrap();
        }
        let mut left = OnlineSoftmaxState::new(2).unwrap();
        let mut middle = OnlineSoftmaxState::new(2).unwrap();
        let mut right = OnlineSoftmaxState::new(2).unwrap();
        for (&score, value) in scores[..2].iter().zip(&values[..2]) {
            left.accumulate(score, value).unwrap();
        }
        middle.accumulate(scores[2], &values[2]).unwrap();
        for (&score, value) in scores[3..].iter().zip(&values[3..]) {
            right.accumulate(score, value).unwrap();
        }
        let mut expected = [0.0; 2];
        whole.finalize(&mut expected).unwrap();

        let mut left_associative = left.clone();
        left_associative.merge(&middle).unwrap();
        left_associative.merge(&right).unwrap();

        let mut middle_right = middle.clone();
        middle_right.merge(&right).unwrap();
        let mut right_associative = left.clone();
        right_associative.merge(&middle_right).unwrap();

        let mut permuted = right.clone();
        permuted.merge(&left).unwrap();
        permuted.merge(&middle).unwrap();

        for merged in [left_associative, right_associative, permuted] {
            let mut actual = [0.0; 2];
            merged.finalize(&mut actual).unwrap();
            assert_close(&actual, &expected, 1.0e-6);
            assert_close(&actual, &[5.0, 2.0], 1.0e-6);
        }

        let all_masked = OnlineSoftmaxState::new(2).unwrap();
        let mut empty_on_left = all_masked.clone();
        empty_on_left.merge(&whole).unwrap();
        let mut empty_on_right = whole.clone();
        empty_on_right.merge(&all_masked).unwrap();
        for merged in [empty_on_left, empty_on_right] {
            let mut actual = [0.0; 2];
            merged.finalize(&mut actual).unwrap();
            assert_close(&actual, &expected, 1.0e-6);
        }

        let mut empty_with_empty = all_masked.clone();
        empty_with_empty.merge(&all_masked).unwrap();
        let mut empty_output = [1.0, 1.0];
        empty_with_empty.finalize(&mut empty_output).unwrap();
        assert_close(&empty_output, &[0.0, 0.0], f32::EPSILON);
    }

    #[test]
    fn online_state_allocation_errors_are_structured() {
        assert_eq!(
            OnlineSoftmaxState::new(0),
            Err(OnlineSoftmaxError::ZeroValueWidth)
        );
        let error = OnlineSoftmaxState::new(usize::MAX)
            .expect_err("an impossible accumulator width must fail without panicking");
        assert_eq!(
            error,
            OnlineSoftmaxError::AllocationFailed {
                value_width: usize::MAX
            }
        );
        assert!(error.to_string().contains("could not reserve"));
    }

    #[test]
    fn online_state_defines_empty_infinity_and_nan_behavior() {
        let mut empty = OnlineSoftmaxState::new(2).unwrap();
        empty.accumulate(f32::NEG_INFINITY, &[9.0, 9.0]).unwrap();
        let mut output = [7.0, 7.0];
        empty.finalize(&mut output).unwrap();
        assert_close(&output, &[0.0, 0.0], f32::EPSILON);

        let mut infinities = OnlineSoftmaxState::new(2).unwrap();
        infinities.accumulate(1.0, &[100.0, 100.0]).unwrap();
        infinities.accumulate(f32::INFINITY, &[2.0, 4.0]).unwrap();
        infinities.accumulate(f32::INFINITY, &[4.0, 8.0]).unwrap();
        infinities.finalize(&mut output).unwrap();
        assert_close(&output, &[3.0, 6.0], f32::EPSILON);

        let snapshot = infinities.clone();
        assert_eq!(
            infinities.accumulate(f32::NAN, &[1.0, 2.0]),
            Err(OnlineSoftmaxError::NaNScore)
        );
        assert_eq!(infinities, snapshot);
        assert_eq!(
            infinities.accumulate(0.0, &[f32::NAN, 2.0]),
            Err(OnlineSoftmaxError::NaNValue { index: 0 })
        );
        assert_eq!(infinities, snapshot);
    }
}
