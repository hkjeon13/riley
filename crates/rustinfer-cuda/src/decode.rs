//! Prepared query-length-one decode attention and contiguous KV-cache writes.

use std::error;
use std::fmt;
use std::sync::Arc;

use crate::CUDA_COMPILED_ARCHITECTURES;
use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::memory::CudaDeviceBuffer;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{ContextInner, CudaContext, CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const ONLINE_HEAD_SIZE: u64 = 64;
const DEFAULT_TOKENS_PER_PARTITION: u64 = 128;
const MINIMUM_HARDWARE_COMPUTE_CAPABILITY: (u32, u32) = (8, 0);
const MAXIMUM_GRID_X: u64 = i32::MAX as u64;
const MAXIMUM_GRID_Y: u64 = 65_535;
const REFERENCE_IMPLEMENTATION_ID: &str = "rustinfer.cuda.materialized-gqa-decode.bf16";
const ONLINE_IMPLEMENTATION_ID: &str = "rustinfer.cuda.chunked-online-gqa-decode.bf16.d64";
const IMPLEMENTATION_VERSION: &str = "1";
const NATIVE_DEPENDENCY: &str = concat!(
    "rustinfer_cuda_native@abi1+cuda-architectures=",
    env!("RUSTINFER_CUDA_COMPILED_ARCHITECTURES"),
    "+cudart"
);

/// Version of the packed decode-partial-state storage contract.
pub const DECODE_PARTIAL_STATE_VERSION: u32 = 1;

/// One cache write from dense token-major projections into head-major storage.
#[derive(Debug)]
pub struct KvCacheAppendParams<'a> {
    /// BF16 source key rows `[T,KVH,D]`, after `RoPE`.
    pub key_source: CudaBufferSpan<'a>,
    /// BF16 source value rows `[T,KVH,D]`.
    pub value_source: CudaBufferSpan<'a>,
    /// BF16 destination key cache `[KVH,M,D]`.
    pub key_cache: CudaBufferSpanMut<'a>,
    /// BF16 destination value cache `[KVH,M,D]`.
    pub value_cache: CudaBufferSpanMut<'a>,
    /// Number of source tokens `T`; it must be non-zero.
    pub source_token_count: u64,
    /// First logical cache position to overwrite.
    pub destination_token_start: u64,
    /// Fixed cache capacity `M`.
    pub maximum_token_count: u64,
    /// Number of key/value heads `KVH`.
    pub key_value_head_count: u64,
    /// Elements per key/value head `D`.
    pub head_size: u64,
}

/// Writes K and V into the contiguous head-major cache in one native call.
///
/// The copy is bit-preserving; no scalar conversion or arithmetic is applied.
/// All validation happens before either cache is modified. Logical-length
/// ownership remains with the caller and should be committed only after every
/// layer write in a request succeeds.
///
/// # Errors
///
/// Returns for invalid dimensions, range arithmetic, dtype, capacity, context,
/// overlap, launch, or synchronization failure.
pub fn kv_cache_append(
    params: &mut KvCacheAppendParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "kv_cache_append";
    for (name, value) in [
        ("source_token_count", params.source_token_count),
        ("maximum_token_count", params.maximum_token_count),
        ("key_value_head_count", params.key_value_head_count),
        ("head_size", params.head_size),
    ] {
        require_nonzero(OPERATION, name, value)?;
    }
    if params.destination_token_start > params.maximum_token_count
        || params.source_token_count > params.maximum_token_count - params.destination_token_start
    {
        return Err(CudaError::out_of_range(
            OPERATION,
            "cache destination range exceeds maximum_token_count",
        ));
    }
    for (name, dtype) in [
        ("key_source", params.key_source.dtype()),
        ("value_source", params.value_source.dtype()),
        ("key_cache", params.key_cache.dtype()),
        ("value_cache", params.value_cache.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let source_bytes = checked_bytes(
        OPERATION,
        &[
            params.source_token_count,
            params.key_value_head_count,
            params.head_size,
            BF16_BYTES,
        ],
    )?;
    let cache_bytes = checked_bytes(
        OPERATION,
        &[
            params.key_value_head_count,
            params.maximum_token_count,
            params.head_size,
            BF16_BYTES,
        ],
    )?;
    require_capacity(
        OPERATION,
        "key_source",
        params.key_source.byte_len(),
        source_bytes,
    )?;
    require_capacity(
        OPERATION,
        "value_source",
        params.value_source.byte_len(),
        source_bytes,
    )?;
    require_capacity(
        OPERATION,
        "key_cache",
        params.key_cache.byte_len(),
        cache_bytes,
    )?;
    require_capacity(
        OPERATION,
        "value_cache",
        params.value_cache.byte_len(),
        cache_bytes,
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.key_source.buffer(),
            params.value_source.buffer(),
            params.key_cache.buffer(),
            params.value_cache.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::kv_cache_write_execute(
            params.key_source.raw(),
            params.value_source.raw(),
            params.key_cache.raw(),
            params.value_cache.raw(),
            params.source_token_count,
            params.destination_token_start,
            params.maximum_token_count,
            params.key_value_head_count,
            params.head_size,
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// Cold selection policy for query-length-one decode attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeAttentionPreference {
    /// Require the materialized staged-BF16 reference.
    Reference,
    /// Prefer chunked online attention, with cold reference fallback.
    Optimized,
}

/// Backend fixed into a prepared decode plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeAttentionBackend {
    /// Materializes BF16 `[QH,T]` scores and probabilities.
    MaterializedReference,
    /// Produces and merges packed F32 partial states.
    ChunkedOnline,
}

/// Why cold selection chose its backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeAttentionSelectionReason {
    /// The caller explicitly required the reference.
    ExplicitReference,
    /// Every optimized capability matched.
    OptimizedCapabilityMatch,
    /// The optimized implementation was not linked.
    OptimizedUnavailableFallback,
    /// The optimized kernel supports D64 only.
    UnsupportedHeadSizeFallback,
    /// The fixed partition/grid contract cannot represent the request.
    UnsupportedLaunchGeometryFallback,
}

/// Merge order for packed partial states.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePartialReductionOrder {
    /// Merge logical range slots from zero upward.
    LogicalAscending,
    /// Merge slots in reverse, primarily for associativity verification.
    LogicalDescending,
}

#[cfg(feature = "cuda")]
const fn reduction_order_code(order: DecodePartialReductionOrder) -> u32 {
    match order {
        DecodePartialReductionOrder::LogicalAscending => ffi::DECODE_REDUCTION_LOGICAL_ASCENDING,
        DecodePartialReductionOrder::LogicalDescending => ffi::DECODE_REDUCTION_LOGICAL_DESCENDING,
    }
}

/// Which native decode implementations may participate in cold selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeAttentionBackendAvailability {
    reference: bool,
    chunked_online: bool,
}

impl DecodeAttentionBackendAvailability {
    /// Creates an explicit availability snapshot.
    #[must_use]
    pub const fn new(reference: bool, chunked_online: bool) -> Self {
        Self {
            reference,
            chunked_online,
        }
    }

    /// Native backends linked by the current feature set.
    #[must_use]
    pub const fn linked() -> Self {
        Self::new(cfg!(feature = "cuda"), cfg!(feature = "cuda"))
    }

    /// Whether the materialized reference is linked.
    #[must_use]
    pub const fn reference(self) -> bool {
        self.reference
    }

    /// Whether the chunked online backend is linked.
    #[must_use]
    pub const fn chunked_online(self) -> bool {
        self.chunked_online
    }
}

/// Fixed request dimensions for query-length-one, batch-one decode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecodeAttentionRequest {
    maximum_sequence_length: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
    tokens_per_partition: u64,
}

impl DecodeAttentionRequest {
    /// Creates a request using the default 128-token optimized partition.
    #[must_use]
    pub const fn new(
        maximum_sequence_length: u64,
        query_head_count: u64,
        key_value_head_count: u64,
        head_size: u64,
        scale: f32,
    ) -> Self {
        Self {
            maximum_sequence_length,
            query_head_count,
            key_value_head_count,
            head_size,
            scale,
            tokens_per_partition: DEFAULT_TOKENS_PER_PARTITION,
        }
    }

    /// Overrides the cold partition size, primarily for validation and tuning.
    #[must_use]
    pub const fn with_tokens_per_partition(mut self, tokens: u64) -> Self {
        self.tokens_per_partition = tokens;
        self
    }

    /// Fixed cache capacity.
    #[must_use]
    pub const fn maximum_sequence_length(self) -> u64 {
        self.maximum_sequence_length
    }

    /// Number of query heads.
    #[must_use]
    pub const fn query_head_count(self) -> u64 {
        self.query_head_count
    }

    /// Number of key/value heads.
    #[must_use]
    pub const fn key_value_head_count(self) -> u64 {
        self.key_value_head_count
    }

    /// Head width.
    #[must_use]
    pub const fn head_size(self) -> u64 {
        self.head_size
    }

    /// Positive finite score scale.
    #[must_use]
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// Fixed optimized logical range size.
    #[must_use]
    pub const fn tokens_per_partition(self) -> u64 {
        self.tokens_per_partition
    }
}

/// Static contract declared by a decode backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeAttentionCapability {
    implementation_id: &'static str,
    head_size: Option<u64>,
    accumulator_dtype: CudaDType,
    materializes_scores: bool,
    partial_state_merge: bool,
}

impl DecodeAttentionCapability {
    /// Stable implementation identity.
    #[must_use]
    pub const fn implementation_id(self) -> &'static str {
        self.implementation_id
    }

    /// Required exact head width, if any.
    #[must_use]
    pub const fn head_size(self) -> Option<u64> {
        self.head_size
    }

    /// Accumulator scalar type.
    #[must_use]
    pub const fn accumulator_dtype(self) -> CudaDType {
        self.accumulator_dtype
    }

    /// Whether a complete score/probability row is written to HBM.
    #[must_use]
    pub const fn materializes_scores(self) -> bool {
        self.materializes_scores
    }

    /// Whether packed partial states are supported.
    #[must_use]
    pub const fn supports_partial_state_merge(self) -> bool {
        self.partial_state_merge
    }
}

const REFERENCE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: REFERENCE_IMPLEMENTATION_ID,
    head_size: None,
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: false,
};

const ONLINE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: ONLINE_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: false,
    partial_state_merge: true,
};

/// Immutable selection, workspace, and provenance evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeAttentionSelectionTrace {
    reason: DecodeAttentionSelectionReason,
    implementation_id: &'static str,
    implementation_version: &'static str,
    native_dependency: &'static str,
    compiled_architectures: &'static str,
    device_ordinal: u32,
    compute_capability: (u32, u32),
    workspace_dtype: CudaDType,
    workspace_bytes: u64,
    materialized_score_bytes: u64,
    partial_state_bytes: u64,
    partial_state_capacity: u64,
    tokens_per_partition: u64,
}

impl DecodeAttentionSelectionTrace {
    #[must_use]
    pub const fn reason(self) -> DecodeAttentionSelectionReason {
        self.reason
    }
    #[must_use]
    pub const fn implementation_id(self) -> &'static str {
        self.implementation_id
    }
    #[must_use]
    pub const fn implementation_version(self) -> &'static str {
        self.implementation_version
    }
    #[must_use]
    pub const fn native_dependency(self) -> &'static str {
        self.native_dependency
    }
    #[must_use]
    pub const fn compiled_architectures(self) -> &'static str {
        self.compiled_architectures
    }
    #[must_use]
    pub const fn device_ordinal(self) -> u32 {
        self.device_ordinal
    }
    #[must_use]
    pub const fn compute_capability(self) -> (u32, u32) {
        self.compute_capability
    }
    #[must_use]
    pub const fn workspace_dtype(self) -> CudaDType {
        self.workspace_dtype
    }
    #[must_use]
    pub const fn workspace_bytes(self) -> u64 {
        self.workspace_bytes
    }
    #[must_use]
    pub const fn materialized_score_bytes(self) -> u64 {
        self.materialized_score_bytes
    }
    #[must_use]
    pub const fn partial_state_bytes(self) -> u64 {
        self.partial_state_bytes
    }
    #[must_use]
    pub const fn partial_state_capacity(self) -> u64 {
        self.partial_state_capacity
    }
    #[must_use]
    pub const fn tokens_per_partition(self) -> u64 {
        self.tokens_per_partition
    }
}

/// Packed F32 `[partition,QH,D+2]` workspace geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodePartialStateLayout {
    partition_capacity: u64,
    query_head_count: u64,
    head_size: u64,
    state_stride_elements: u64,
    byte_len: u64,
}

impl DecodePartialStateLayout {
    /// Creates a checked version-1 packed layout.
    ///
    /// # Errors
    ///
    /// Returns for zero dimensions or fixed-width arithmetic overflow.
    pub fn new(partition_capacity: u64, query_head_count: u64, head_size: u64) -> CudaResult<Self> {
        const OPERATION: &str = "DecodePartialStateLayout::new";
        require_nonzero(OPERATION, "partition_capacity", partition_capacity)?;
        require_nonzero(OPERATION, "query_head_count", query_head_count)?;
        require_nonzero(OPERATION, "head_size", head_size)?;
        let state_stride_elements = head_size.checked_add(2).ok_or_else(|| {
            CudaError::out_of_range(OPERATION, "partial-state stride overflows u64")
        })?;
        let byte_len = checked_bytes(
            OPERATION,
            &[
                partition_capacity,
                query_head_count,
                state_stride_elements,
                F32_BYTES,
            ],
        )?;
        Ok(Self {
            partition_capacity,
            query_head_count,
            head_size,
            state_stride_elements,
            byte_len,
        })
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        DECODE_PARTIAL_STATE_VERSION
    }
    #[must_use]
    pub const fn partition_capacity(self) -> u64 {
        self.partition_capacity
    }
    #[must_use]
    pub const fn query_head_count(self) -> u64 {
        self.query_head_count
    }
    #[must_use]
    pub const fn head_size(self) -> u64 {
        self.head_size
    }
    /// F32 elements per `(partition,query_head)` state: `m,l,n[D]`.
    #[must_use]
    pub const fn state_stride_elements(self) -> u64 {
        self.state_stride_elements
    }
    #[must_use]
    pub const fn byte_len(self) -> u64 {
        self.byte_len
    }

    /// F32 element offset of one state slot.
    ///
    /// # Errors
    ///
    /// Returns when either index is outside the declared layout or arithmetic
    /// overflows.
    pub fn state_offset_elements(self, partition: u64, query_head: u64) -> CudaResult<u64> {
        const OPERATION: &str = "DecodePartialStateLayout::state_offset_elements";
        if partition >= self.partition_capacity || query_head >= self.query_head_count {
            return Err(CudaError::out_of_range(
                OPERATION,
                "partial-state index exceeds the packed layout",
            ));
        }
        partition
            .checked_mul(self.query_head_count)
            .and_then(|value| value.checked_add(query_head))
            .and_then(|value| value.checked_mul(self.state_stride_elements))
            .ok_or_else(|| CudaError::out_of_range(OPERATION, "state offset overflows u64"))
    }
}

/// Device views used by one prepared decode execution.
#[derive(Debug)]
pub struct DecodeAttentionParams<'a> {
    /// BF16 query `[QH,D]` at the current position.
    pub query: CudaBufferSpan<'a>,
    /// BF16 head-major key cache `[KVH,M,D]`.
    pub key_cache: CudaBufferSpan<'a>,
    /// BF16 head-major value cache `[KVH,M,D]`.
    pub value_cache: CudaBufferSpan<'a>,
    /// BF16 output `[QH,D]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Reference capacity `QH*M*2` (active BF16 layout `[QH,T]`) or packed F32
    /// online states, according to [`PreparedDecodeAttention::workspace_dtype`].
    pub workspace: CudaBufferSpanMut<'a>,
}

/// One immutable backend selection bound to its CUDA context owner.
#[derive(Clone)]
pub struct PreparedDecodeAttention {
    context: Arc<ContextInner>,
    request: DecodeAttentionRequest,
    backend: DecodeAttentionBackend,
    capability: DecodeAttentionCapability,
    trace: DecodeAttentionSelectionTrace,
}

impl fmt::Debug for PreparedDecodeAttention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedDecodeAttention")
            .field("device_ordinal", &self.context.ordinal)
            .field("request", &self.request)
            .field("backend", &self.backend)
            .field("capability", &self.capability)
            .field("trace", &self.trace)
            .finish()
    }
}

impl PreparedDecodeAttention {
    /// Cold-selects one backend without allocating or launching CUDA work.
    ///
    /// # Errors
    ///
    /// Returns for invalid dimensions, arithmetic overflow, incompatible AOT
    /// architecture, or unavailable required backends.
    pub fn select(
        context: &CudaContext,
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_for_compute_capability(
            context,
            context.compute_capability(),
            request,
            preference,
            availability,
        )
    }

    fn select_for_compute_capability(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_request(request)?;
        require_architecture_support(compute_capability)?;
        match preference {
            DecodeAttentionPreference::Reference => {
                if !availability.reference {
                    return Err(not_supported(
                        "select_decode_attention",
                        "the explicitly requested materialized reference is unavailable",
                    ));
                }
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::MaterializedReference,
                    DecodeAttentionSelectionReason::ExplicitReference,
                )
            }
            DecodeAttentionPreference::Optimized => {
                let fallback_reason = if !availability.chunked_online {
                    DecodeAttentionSelectionReason::OptimizedUnavailableFallback
                } else if request.head_size != ONLINE_HEAD_SIZE {
                    DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback
                } else if !optimized_geometry_supported(request)? {
                    DecodeAttentionSelectionReason::UnsupportedLaunchGeometryFallback
                } else {
                    validate_optimized_workspace(request)?;
                    return prepare_selection(
                        context,
                        compute_capability,
                        request,
                        DecodeAttentionBackend::ChunkedOnline,
                        DecodeAttentionSelectionReason::OptimizedCapabilityMatch,
                    );
                };
                if !availability.reference {
                    return Err(not_supported(
                        "select_decode_attention",
                        format!(
                            "optimized decode was rejected ({fallback_reason:?}) and the reference is unavailable"
                        ),
                    ));
                }
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::MaterializedReference,
                    fallback_reason,
                )
            }
        }
    }

    #[must_use]
    pub const fn request(&self) -> DecodeAttentionRequest {
        self.request
    }
    #[must_use]
    pub const fn backend(&self) -> DecodeAttentionBackend {
        self.backend
    }
    #[must_use]
    pub const fn capability(&self) -> DecodeAttentionCapability {
        self.capability
    }
    #[must_use]
    pub const fn selection_trace(&self) -> DecodeAttentionSelectionTrace {
        self.trace
    }
    #[must_use]
    pub const fn workspace_bytes(&self) -> u64 {
        self.trace.workspace_bytes
    }
    #[must_use]
    pub const fn workspace_dtype(&self) -> CudaDType {
        self.trace.workspace_dtype
    }
    #[must_use]
    pub const fn partial_state_capacity(&self) -> u64 {
        self.trace.partial_state_capacity
    }
    #[must_use]
    pub const fn tokens_per_partition(&self) -> u64 {
        self.trace.tokens_per_partition
    }

    /// Executes exactly the cold-selected backend for logical prefix `T`.
    ///
    /// `logical_token_count` includes the current token already appended to the
    /// cache. No allocation or backend fallback occurs here.
    ///
    /// # Errors
    ///
    /// Returns for logical-length, dtype, capacity, ownership, native launch,
    /// or synchronization failure.
    pub fn execute(
        &self,
        logical_token_count: u64,
        params: &mut DecodeAttentionParams<'_>,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "decode_attention";
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        require_nonzero(OPERATION, "logical_token_count", logical_token_count)?;
        if logical_token_count > self.request.maximum_sequence_length {
            return Err(CudaError::out_of_range(
                OPERATION,
                "logical_token_count exceeds the prepared cache capacity",
            ));
        }
        validate_execute_params(self, params, stream)?;

        #[cfg(feature = "cuda")]
        match self.backend {
            DecodeAttentionBackend::MaterializedReference => {
                ffi::decode_attention_reference_execute(
                    params.query.raw(),
                    params.key_cache.raw(),
                    params.value_cache.raw(),
                    params.workspace.raw(),
                    params.output.raw(),
                    self.request.maximum_sequence_length,
                    logical_token_count,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            DecodeAttentionBackend::ChunkedOnline => ffi::decode_attention_execute(
                params.query.raw(),
                params.key_cache.raw(),
                params.value_cache.raw(),
                params.workspace.raw(),
                params.output.raw(),
                self.request.maximum_sequence_length,
                logical_token_count,
                self.request.query_head_count,
                self.request.key_value_head_count,
                self.request.head_size,
                self.request.tokens_per_partition,
                self.trace.partial_state_capacity,
                self.request.scale,
                reduction_order_code(DecodePartialReductionOrder::LogicalAscending),
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

/// Standalone packed-state reducer parameters.
#[derive(Debug)]
pub struct DecodePartialStateReduceParams<'a> {
    /// F32 `[capacity,QH,D+2]` packed states.
    pub partial_states: CudaBufferSpan<'a>,
    /// BF16 `[QH,D]` normalized output.
    pub output: CudaBufferSpanMut<'a>,
    /// Number of logical range slots to merge; zero produces all-zero output.
    pub partial_state_count: u64,
    /// Allocated range capacity represented by `partial_states`.
    pub partial_state_capacity: u64,
    /// Number of independent query heads.
    pub query_head_count: u64,
    /// Value width `D`; unlike the optimized producer, the reducer is generic.
    pub head_size: u64,
    /// Deterministic logical slot traversal.
    pub order: DecodePartialReductionOrder,
}

/// Reduces already-produced packed states and normalizes exactly once.
///
/// # Errors
///
/// Returns for invalid state geometry, dtype, capacity, context, launch, or
/// synchronization failure.
pub fn decode_partial_states_reduce(
    params: &mut DecodePartialStateReduceParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "decode_partial_states_reduce";
    require_nonzero(
        OPERATION,
        "partial_state_capacity",
        params.partial_state_capacity,
    )?;
    require_nonzero(OPERATION, "query_head_count", params.query_head_count)?;
    require_nonzero(OPERATION, "head_size", params.head_size)?;
    if params.partial_state_count > params.partial_state_capacity {
        return Err(CudaError::out_of_range(
            OPERATION,
            "partial_state_count exceeds partial_state_capacity",
        ));
    }
    require_dtype(
        OPERATION,
        "partial_states",
        params.partial_states.dtype(),
        CudaDType::F32,
    )?;
    require_dtype(OPERATION, "output", params.output.dtype(), CudaDType::BF16)?;
    let layout = DecodePartialStateLayout::new(
        params.partial_state_capacity,
        params.query_head_count,
        params.head_size,
    )?;
    let output_bytes = checked_bytes(
        OPERATION,
        &[params.query_head_count, params.head_size, BF16_BYTES],
    )?;
    require_capacity(
        OPERATION,
        "partial_states",
        params.partial_states.byte_len(),
        layout.byte_len(),
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len(), output_bytes)?;
    validate_resources(
        OPERATION,
        stream,
        &[params.partial_states.buffer(), params.output.buffer()],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::decode_partial_state_reduce_execute(
            params.partial_states.raw(),
            params.output.raw(),
            params.partial_state_count,
            params.partial_state_capacity,
            params.query_head_count,
            params.head_size,
            reduction_order_code(params.order),
            &mut stream.native,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = params;
        Err(CudaError::unavailable(OPERATION))
    }
}

/// CPU reference representation of `(m,l,n)` for one logical KV range.
///
/// This owned helper is for tests and reference composition, never the decode
/// hot path. Its packed form is exactly `[m,l,n[0],...,n[D-1]]` F32.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodePartialState {
    max_score: f32,
    exp_sum: f32,
    weighted_value_sum: Vec<f32>,
}

impl DecodePartialState {
    /// Creates the empty `(-inf,0,0)` state.
    ///
    /// # Errors
    ///
    /// Returns for zero width or host accumulator allocation failure.
    pub fn new(value_width: usize) -> Result<Self, DecodePartialStateError> {
        if value_width == 0 {
            return Err(DecodePartialStateError::ZeroValueWidth);
        }
        let mut weighted_value_sum = Vec::new();
        weighted_value_sum
            .try_reserve_exact(value_width)
            .map_err(|_| DecodePartialStateError::AllocationFailed { value_width })?;
        weighted_value_sum.resize(value_width, 0.0);
        Ok(Self {
            max_score: f32::NEG_INFINITY,
            exp_sum: 0.0,
            weighted_value_sum,
        })
    }

    /// Decodes one exact packed ABI row.
    ///
    /// # Errors
    ///
    /// Returns for an invalid row length, NaN, negative denominator, or host
    /// accumulator allocation failure.
    pub fn from_packed(packed: &[f32]) -> Result<Self, DecodePartialStateError> {
        if packed.len() < 3 {
            return Err(DecodePartialStateError::InvalidPackedLength {
                actual: packed.len(),
            });
        }
        let value_width = packed.len() - 2;
        let max_score = packed[0];
        let exp_sum = packed[1];
        if max_score.is_nan() {
            return Err(DecodePartialStateError::NaNScore);
        }
        if exp_sum.is_nan() || exp_sum < 0.0 {
            return Err(DecodePartialStateError::InvalidExpSum);
        }
        if let Some(index) = packed[2..].iter().position(|value| value.is_nan()) {
            return Err(DecodePartialStateError::NaNValue { index });
        }
        let mut state = Self::new(value_width)?;
        state.max_score = max_score;
        state.exp_sum = exp_sum;
        state.weighted_value_sum.copy_from_slice(&packed[2..]);
        Ok(state)
    }

    #[must_use]
    pub fn value_width(&self) -> usize {
        self.weighted_value_sum.len()
    }
    #[must_use]
    pub const fn max_score(&self) -> f32 {
        self.max_score
    }
    #[must_use]
    pub const fn exp_sum(&self) -> f32 {
        self.exp_sum
    }
    #[must_use]
    pub fn weighted_value_sum(&self) -> &[f32] {
        &self.weighted_value_sum
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.exp_sum == 0.0
    }

    /// Adds one unnormalized score/value pair.
    ///
    /// # Errors
    ///
    /// Returns before mutation for width mismatch or NaN input.
    pub fn accumulate(&mut self, score: f32, value: &[f32]) -> Result<(), DecodePartialStateError> {
        self.validate_value(score, value)?;
        if score == f32::NEG_INFINITY {
            return Ok(());
        }
        if self.is_empty() || score == f32::INFINITY && self.max_score != f32::INFINITY {
            self.max_score = score;
            self.exp_sum = 1.0;
            self.weighted_value_sum.copy_from_slice(value);
            return Ok(());
        }
        if self.max_score == f32::INFINITY {
            if score == f32::INFINITY {
                self.exp_sum += 1.0;
                for (sum, &item) in self.weighted_value_sum.iter_mut().zip(value) {
                    *sum += item;
                }
            }
            return Ok(());
        }
        let maximum = self.max_score.max(score);
        let old_scale = (self.max_score - maximum).exp();
        let item_scale = (score - maximum).exp();
        self.exp_sum = old_scale.mul_add(self.exp_sum, item_scale);
        for (sum, &item) in self.weighted_value_sum.iter_mut().zip(value) {
            *sum = old_scale.mul_add(*sum, item_scale * item);
        }
        self.max_score = maximum;
        Ok(())
    }

    /// Associatively merges another disjoint logical range.
    ///
    /// # Errors
    ///
    /// Returns before mutation when value widths differ.
    pub fn merge(&mut self, other: &Self) -> Result<(), DecodePartialStateError> {
        self.require_width(other.value_width())?;
        if other.is_empty() {
            return Ok(());
        }
        if self.is_empty() {
            self.max_score = other.max_score;
            self.exp_sum = other.exp_sum;
            self.weighted_value_sum
                .copy_from_slice(&other.weighted_value_sum);
            return Ok(());
        }
        if self.max_score == f32::INFINITY || other.max_score == f32::INFINITY {
            match (
                self.max_score == f32::INFINITY,
                other.max_score == f32::INFINITY,
            ) {
                (true, true) => {
                    self.exp_sum += other.exp_sum;
                    for (left, &right) in self
                        .weighted_value_sum
                        .iter_mut()
                        .zip(&other.weighted_value_sum)
                    {
                        *left += right;
                    }
                }
                (false, true) => {
                    self.max_score = other.max_score;
                    self.exp_sum = other.exp_sum;
                    self.weighted_value_sum
                        .copy_from_slice(&other.weighted_value_sum);
                }
                (true, false) => {}
                (false, false) => unreachable!(),
            }
            return Ok(());
        }
        let maximum = self.max_score.max(other.max_score);
        let left_scale = (self.max_score - maximum).exp();
        let right_scale = (other.max_score - maximum).exp();
        self.exp_sum = left_scale.mul_add(self.exp_sum, right_scale * other.exp_sum);
        for (left, &right) in self
            .weighted_value_sum
            .iter_mut()
            .zip(&other.weighted_value_sum)
        {
            *left = left_scale.mul_add(*left, right_scale * right);
        }
        self.max_score = maximum;
        Ok(())
    }

    /// Writes normalized `n/l`, or zero for an empty state.
    ///
    /// # Errors
    ///
    /// Returns before writing when the destination width differs.
    pub fn finalize(&self, output: &mut [f32]) -> Result<(), DecodePartialStateError> {
        self.require_width(output.len())?;
        if self.is_empty() {
            output.fill(0.0);
        } else {
            for (output, &numerator) in output.iter_mut().zip(&self.weighted_value_sum) {
                *output = numerator / self.exp_sum;
            }
        }
        Ok(())
    }

    /// Writes the exact packed version-1 row without normalization.
    ///
    /// # Errors
    ///
    /// Returns before writing when the destination length is not exactly
    /// `D+2`.
    pub fn write_packed(&self, output: &mut [f32]) -> Result<(), DecodePartialStateError> {
        let expected = self.value_width() + 2;
        if output.len() != expected {
            return Err(DecodePartialStateError::PackedLengthMismatch {
                expected,
                actual: output.len(),
            });
        }
        output[0] = self.max_score;
        output[1] = self.exp_sum;
        output[2..].copy_from_slice(&self.weighted_value_sum);
        Ok(())
    }

    fn validate_value(&self, score: f32, value: &[f32]) -> Result<(), DecodePartialStateError> {
        self.require_width(value.len())?;
        if score.is_nan() {
            return Err(DecodePartialStateError::NaNScore);
        }
        if let Some(index) = value.iter().position(|item| item.is_nan()) {
            return Err(DecodePartialStateError::NaNValue { index });
        }
        Ok(())
    }

    fn require_width(&self, actual: usize) -> Result<(), DecodePartialStateError> {
        if actual == self.value_width() {
            Ok(())
        } else {
            Err(DecodePartialStateError::ValueWidthMismatch {
                expected: self.value_width(),
                actual,
            })
        }
    }
}

/// CPU packed-state contract failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodePartialStateError {
    ZeroValueWidth,
    AllocationFailed { value_width: usize },
    ValueWidthMismatch { expected: usize, actual: usize },
    InvalidPackedLength { actual: usize },
    PackedLengthMismatch { expected: usize, actual: usize },
    InvalidExpSum,
    NaNScore,
    NaNValue { index: usize },
}

impl fmt::Display for DecodePartialStateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroValueWidth => {
                formatter.write_str("decode partial value width must be non-zero")
            }
            Self::AllocationFailed { value_width } => write!(
                formatter,
                "could not reserve {value_width} F32 decode partial accumulators"
            ),
            Self::ValueWidthMismatch { expected, actual } => {
                write!(formatter, "expected value width {expected}, got {actual}")
            }
            Self::InvalidPackedLength { actual } => write!(
                formatter,
                "packed decode partial row must contain m, l, and at least one n value; got {actual} elements"
            ),
            Self::PackedLengthMismatch { expected, actual } => write!(
                formatter,
                "packed decode partial row requires {expected} elements, got {actual}"
            ),
            Self::InvalidExpSum => {
                formatter.write_str("decode partial exp_sum must be non-negative and not NaN")
            }
            Self::NaNScore => formatter.write_str("decode partial score must not be NaN"),
            Self::NaNValue { index } => {
                write!(
                    formatter,
                    "decode partial value at index {index} must not be NaN"
                )
            }
        }
    }
}

impl error::Error for DecodePartialStateError {}

fn prepare_selection(
    context: &CudaContext,
    compute_capability: (u32, u32),
    request: DecodeAttentionRequest,
    backend: DecodeAttentionBackend,
    reason: DecodeAttentionSelectionReason,
) -> CudaResult<PreparedDecodeAttention> {
    let (
        capability,
        workspace_dtype,
        workspace_bytes,
        materialized_score_bytes,
        partial_state_bytes,
        partial_state_capacity,
        tokens_per_partition,
    ) = match backend {
        DecodeAttentionBackend::MaterializedReference => {
            let bytes = checked_bytes(
                "select_decode_attention",
                &[
                    request.query_head_count,
                    request.maximum_sequence_length,
                    BF16_BYTES,
                ],
            )?;
            (REFERENCE_CAPABILITY, CudaDType::BF16, bytes, bytes, 0, 0, 0)
        }
        DecodeAttentionBackend::ChunkedOnline => {
            let capacity = partition_count(
                request.maximum_sequence_length,
                request.tokens_per_partition,
            )?;
            let layout = DecodePartialStateLayout::new(
                capacity,
                request.query_head_count,
                request.head_size,
            )?;
            (
                ONLINE_CAPABILITY,
                CudaDType::F32,
                layout.byte_len(),
                0,
                layout.byte_len(),
                capacity,
                request.tokens_per_partition,
            )
        }
    };
    Ok(PreparedDecodeAttention {
        context: Arc::clone(&context.inner),
        request,
        backend,
        capability,
        trace: DecodeAttentionSelectionTrace {
            reason,
            implementation_id: capability.implementation_id,
            implementation_version: IMPLEMENTATION_VERSION,
            native_dependency: NATIVE_DEPENDENCY,
            compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
            device_ordinal: context.device_ordinal(),
            compute_capability,
            workspace_dtype,
            workspace_bytes,
            materialized_score_bytes,
            partial_state_bytes,
            partial_state_capacity,
            tokens_per_partition,
        },
    })
}

fn validate_request(request: DecodeAttentionRequest) -> CudaResult<()> {
    const OPERATION: &str = "select_decode_attention";
    for (name, value) in [
        ("maximum_sequence_length", request.maximum_sequence_length),
        ("query_head_count", request.query_head_count),
        ("key_value_head_count", request.key_value_head_count),
        ("head_size", request.head_size),
    ] {
        require_nonzero(OPERATION, name, value)?;
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
    checked_bytes(
        OPERATION,
        &[request.query_head_count, request.head_size, BF16_BYTES],
    )?;
    checked_bytes(
        OPERATION,
        &[
            request.key_value_head_count,
            request.maximum_sequence_length,
            request.head_size,
            BF16_BYTES,
        ],
    )?;
    Ok(())
}

fn validate_optimized_workspace(request: DecodeAttentionRequest) -> CudaResult<()> {
    let capacity = partition_count(
        request.maximum_sequence_length,
        request.tokens_per_partition,
    )?;
    DecodePartialStateLayout::new(capacity, request.query_head_count, request.head_size)?;
    Ok(())
}

fn validate_execute_params(
    prepared: &PreparedDecodeAttention,
    params: &DecodeAttentionParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "decode_attention";
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_cache", params.key_cache.dtype()),
        ("value_cache", params.value_cache.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    require_dtype(
        OPERATION,
        "workspace",
        params.workspace.dtype(),
        prepared.workspace_dtype(),
    )?;
    let query_bytes = checked_bytes(
        OPERATION,
        &[
            prepared.request.query_head_count,
            prepared.request.head_size,
            BF16_BYTES,
        ],
    )?;
    let cache_bytes = checked_bytes(
        OPERATION,
        &[
            prepared.request.key_value_head_count,
            prepared.request.maximum_sequence_length,
            prepared.request.head_size,
            BF16_BYTES,
        ],
    )?;
    require_capacity(OPERATION, "query", params.query.byte_len(), query_bytes)?;
    require_capacity(
        OPERATION,
        "key_cache",
        params.key_cache.byte_len(),
        cache_bytes,
    )?;
    require_capacity(
        OPERATION,
        "value_cache",
        params.value_cache.byte_len(),
        cache_bytes,
    )?;
    require_capacity(OPERATION, "output", params.output.byte_len(), query_bytes)?;
    require_capacity(
        OPERATION,
        "workspace",
        params.workspace.byte_len(),
        prepared.workspace_bytes(),
    )?;
    validate_resources(
        OPERATION,
        stream,
        &[
            params.query.buffer(),
            params.key_cache.buffer(),
            params.value_cache.buffer(),
            params.output.buffer(),
            params.workspace.buffer(),
        ],
    )
}

fn optimized_geometry_supported(request: DecodeAttentionRequest) -> CudaResult<bool> {
    let partitions = partition_count(
        request.maximum_sequence_length,
        request.tokens_per_partition,
    )?;
    Ok(partitions <= MAXIMUM_GRID_X && request.query_head_count <= MAXIMUM_GRID_Y)
}

fn partition_count(token_count: u64, tokens_per_partition: u64) -> CudaResult<u64> {
    if tokens_per_partition == 0 {
        return Err(CudaError::invalid_argument(
            "select_decode_attention",
            "tokens_per_partition must be greater than zero",
        ));
    }
    Ok(token_count / tokens_per_partition + u64::from(token_count % tokens_per_partition != 0))
}

fn require_architecture_support(actual: (u32, u32)) -> CudaResult<()> {
    if compute_capability_at_least(actual, MINIMUM_HARDWARE_COMPUTE_CAPABILITY)
        && architecture_set_supports(CUDA_COMPILED_ARCHITECTURES, actual)
    {
        Ok(())
    } else {
        Err(not_supported(
            "select_decode_attention",
            format!(
                "compute capability {}.{} cannot execute decode hardware floor {}.{} and compiled architectures {CUDA_COMPILED_ARCHITECTURES}",
                actual.0,
                actual.1,
                MINIMUM_HARDWARE_COMPUTE_CAPABILITY.0,
                MINIMUM_HARDWARE_COMPUTE_CAPABILITY.1
            ),
        ))
    }
}

#[derive(Clone, Copy)]
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

fn architecture_set_supports(architectures: &str, actual: (u32, u32)) -> bool {
    architectures.split(';').any(|token| {
        architecture_target(token).is_some_and(|(target, emission)| match emission {
            ArchitectureEmission::Real => actual.0 == target.0 && actual.1 >= target.1,
            ArchitectureEmission::Virtual | ArchitectureEmission::RealAndVirtual => {
                compute_capability_at_least(actual, target)
            }
        })
    })
}

const fn compute_capability_at_least(actual: (u32, u32), minimum: (u32, u32)) -> bool {
    actual.0 > minimum.0 || actual.0 == minimum.0 && actual.1 >= minimum.1
}

fn checked_bytes(operation: &'static str, factors: &[u64]) -> CudaResult<u64> {
    factors.iter().try_fold(1_u64, |product, &factor| {
        product.checked_mul(factor).ok_or_else(|| {
            CudaError::out_of_range(operation, "decode byte-length arithmetic overflow")
        })
    })
}

fn require_nonzero(operation: &'static str, name: &'static str, value: u64) -> CudaResult<()> {
    if value == 0 {
        Err(CudaError::invalid_argument(
            operation,
            format!("{name} must be greater than zero"),
        ))
    } else {
        Ok(())
    }
}

fn require_dtype(
    operation: &'static str,
    name: &'static str,
    actual: CudaDType,
    expected: CudaDType,
) -> CudaResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(CudaError::invalid_argument(
            operation,
            format!("{name} must be {expected}, got {actual}"),
        ))
    }
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

fn validate_resources(
    operation: &'static str,
    stream: &CudaStream,
    buffers: &[&CudaDeviceBuffer],
) -> CudaResult<()> {
    for buffer in buffers {
        ensure_same_context(buffer.context_owner(), &stream.context, operation)?;
        buffer.ensure_idle_for_operation(operation)?;
    }
    Ok(())
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
    fn test_context() -> CudaContext {
        let target = architecture_target(
            CUDA_COMPILED_ARCHITECTURES
                .split(';')
                .next()
                .expect("build.rs requires one architecture"),
        )
        .expect("build.rs normalizes architecture syntax")
        .0;
        CudaContext {
            inner: Arc::new(ContextInner {
                ordinal: 0,
                compute_capability: target,
            }),
        }
    }

    #[test]
    fn packed_layout_pins_m_l_n_stride_and_offsets() {
        let layout = DecodePartialStateLayout::new(3, 9, 64).unwrap();
        assert_eq!(layout.version(), 1);
        assert_eq!(layout.state_stride_elements(), 66);
        assert_eq!(layout.byte_len(), 3 * 9 * 66 * 4);
        assert_eq!(layout.state_offset_elements(2, 8).unwrap(), 26 * 66);
        assert_eq!(
            layout.state_offset_elements(3, 0).unwrap_err().kind(),
            CudaErrorKind::OutOfRange
        );
    }

    #[test]
    fn cpu_partial_state_round_trips_and_merges_without_partial_normalization() {
        let scores = [-100.0_f32, -1.0, 0.0, 3.0, 100.0];
        let values = [
            [1.0_f32, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 0.0],
        ];
        let mut whole = DecodePartialState::new(2).unwrap();
        for (&score, value) in scores.iter().zip(&values) {
            whole.accumulate(score, value).unwrap();
        }
        let mut left = DecodePartialState::new(2).unwrap();
        let mut right = DecodePartialState::new(2).unwrap();
        for (&score, value) in scores[..3].iter().zip(&values[..3]) {
            left.accumulate(score, value).unwrap();
        }
        for (&score, value) in scores[3..].iter().zip(&values[3..]) {
            right.accumulate(score, value).unwrap();
        }
        left.merge(&right).unwrap();
        let mut expected = [0.0; 2];
        let mut actual = [0.0; 2];
        whole.finalize(&mut expected).unwrap();
        left.finalize(&mut actual).unwrap();
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }

        let mut packed = [0.0; 4];
        left.write_packed(&mut packed).unwrap();
        assert_eq!(DecodePartialState::from_packed(&packed).unwrap(), left);
    }

    #[test]
    fn cpu_partial_state_defines_empty_infinity_and_pre_mutation_errors() {
        let mut state = DecodePartialState::new(2).unwrap();
        state.accumulate(f32::NEG_INFINITY, &[9.0, 9.0]).unwrap();
        let mut output = [1.0; 2];
        state.finalize(&mut output).unwrap();
        assert!(output.iter().all(|value| value.abs() <= f32::EPSILON));
        state.accumulate(f32::INFINITY, &[2.0, 4.0]).unwrap();
        state.accumulate(0.0, &[99.0, 99.0]).unwrap();
        state.accumulate(f32::INFINITY, &[4.0, 8.0]).unwrap();
        state.finalize(&mut output).unwrap();
        assert!((output[0] - 3.0).abs() <= f32::EPSILON);
        assert!((output[1] - 6.0).abs() <= f32::EPSILON);
        let snapshot = state.clone();
        assert_eq!(
            state.accumulate(f32::NAN, &[1.0, 2.0]),
            Err(DecodePartialStateError::NaNScore)
        );
        assert_eq!(state, snapshot);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn cold_selection_records_linear_workspace_and_fallback() {
        let context = test_context();
        let request = DecodeAttentionRequest::new(4096, 9, 3, 64, 0.125);
        let online = PreparedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            request,
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(online.backend(), DecodeAttentionBackend::ChunkedOnline);
        assert_eq!(online.partial_state_capacity(), 32);
        assert_eq!(online.workspace_bytes(), 32 * 9 * 66 * 4);
        assert_eq!(online.workspace_dtype(), CudaDType::F32);

        let fallback = PreparedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            request,
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::new(true, false),
        )
        .unwrap();
        assert_eq!(
            fallback.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(fallback.workspace_bytes(), 4096 * 9 * 2);
        assert_eq!(fallback.workspace_dtype(), CudaDType::BF16);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn selection_rejects_invalid_requests_and_falls_back_for_non_d64() {
        let context = test_context();
        for request in [
            DecodeAttentionRequest::new(0, 9, 3, 64, 0.125),
            DecodeAttentionRequest::new(8, 8, 3, 64, 0.125),
            DecodeAttentionRequest::new(8, 8, 1, 64, f32::NAN),
        ] {
            assert_eq!(
                PreparedDecodeAttention::select_for_compute_capability(
                    &context,
                    context.compute_capability(),
                    request,
                    DecodeAttentionPreference::Optimized,
                    DecodeAttentionBackendAvailability::new(true, true),
                )
                .unwrap_err()
                .kind(),
                CudaErrorKind::InvalidArgument
            );
        }
        assert_eq!(
            PreparedDecodeAttention::select_for_compute_capability(
                &context,
                context.compute_capability(),
                DecodeAttentionRequest::new(8, 8, 1, 64, 0.125).with_tokens_per_partition(0),
                DecodeAttentionPreference::Optimized,
                DecodeAttentionBackendAvailability::new(true, true),
            )
            .unwrap_err()
            .kind(),
            CudaErrorKind::InvalidArgument
        );
        let fallback = PreparedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            DecodeAttentionRequest::new(8, 8, 1, 32, 0.125),
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(
            fallback.selection_trace().reason(),
            DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn explicit_reference_ignores_optimized_only_partition_geometry() {
        let context = test_context();
        let request = DecodeAttentionRequest::new(8, 8, 1, 64, 0.125).with_tokens_per_partition(0);
        let prepared = PreparedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            request,
            DecodeAttentionPreference::Reference,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(
            prepared.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(prepared.workspace_bytes(), 8 * 8 * 2);
        assert_eq!(prepared.tokens_per_partition(), 0);
    }
}
