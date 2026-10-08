//! Prepared query-length-one decode attention and contiguous KV-cache writes.

use std::error;
use std::fmt;
use std::sync::Arc;

use crate::CUDA_COMPILED_ARCHITECTURES;
use crate::error::{CudaError, CudaErrorDomain, CudaErrorKind, CudaErrorStage, CudaResult};
use crate::gemm::{
    FIXED37_CHUNK_ELEMENTS, FIXED37_MAX_REDUCTION_ELEMENTS, FIXED37_REDUCTION_VERSION,
};
use crate::memory::CudaDeviceBuffer;
use crate::prefill::AttentionReductionProfile;
use crate::primitives::{CudaBufferSpan, CudaBufferSpanMut, CudaDType};
use crate::runtime::{ContextInner, CudaContext, CudaStream, ensure_same_context};

#[cfg(feature = "cuda")]
use crate::ffi;

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const ONLINE_HEAD_SIZE: u64 = 64;
const HUGGING_FACE_SHORT_DECODE_MAX_TOKENS: u64 = 32;
const REVIEWED_HF_QUERY_HEAD_COUNT: u64 = 9;
const REVIEWED_HF_KEY_VALUE_HEAD_COUNT: u64 = 3;
const FIXED37_MAX_TWO_PASS_TOKENS: u64 = 8192;
const DEFAULT_TOKENS_PER_PARTITION: u64 = 128;
const MINIMUM_HARDWARE_COMPUTE_CAPABILITY: (u32, u32) = (8, 0);
const MAXIMUM_GRID_X: u64 = i32::MAX as u64;
const MAXIMUM_GRID_Y: u64 = 65_535;
const REFERENCE_IMPLEMENTATION_ID: &str = "riley.cuda.materialized-gqa-decode.bf16";
const ONLINE_IMPLEMENTATION_ID: &str = "riley.cuda.chunked-online-gqa-decode.bf16.d64";
const REVIEWED_HF_HYBRID_IMPLEMENTATION_ID: &str =
    "riley.cuda.reviewed-9qh-3kvh-hf-short-materialized-then-chunked-online.bf16.d64.t32";
const PAGED_REFERENCE_IMPLEMENTATION_ID: &str =
    "riley.cuda.paged-materialized-gqa-decode.bf16.block16";
const PAGED_ONLINE_IMPLEMENTATION_ID: &str =
    "riley.cuda.paged-block-online-gqa-decode.bf16.d64.block16";
const REVIEWED_HF_PAGED_HYBRID_IMPLEMENTATION_ID: &str = "riley.cuda.reviewed-9qh-3kvh-paged-hf-short-materialized-then-block-online.bf16.d64.t32.block16";
const FIXED37_REFERENCE_IMPLEMENTATION_ID: &str = "riley.cuda.fixed37.materialized-gqa-decode.bf16";
const FIXED37_PAGED_REFERENCE_IMPLEMENTATION_ID: &str =
    "riley.cuda.fixed37.paged-materialized-gqa-decode.bf16.block16";
const FIXED37_TWO_PASS_IMPLEMENTATION_ID: &str =
    "riley.cuda.fixed37.two-pass-gqa-decode.bf16.d64.t8192";
const FIXED37_PAGED_TWO_PASS_IMPLEMENTATION_ID: &str =
    "riley.cuda.fixed37.paged-two-pass-gqa-decode.bf16.d64.t8192.block16";
const IMPLEMENTATION_VERSION: &str = "1";
const REVIEWED_HF_HYBRID_IMPLEMENTATION_VERSION: &str = "2";
const NATIVE_DEPENDENCY: &str = concat!(
    "riley_cuda_native@abi1+cuda-architectures=",
    env!("RILEY_CUDA_COMPILED_ARCHITECTURES"),
    "+cudart"
);

/// Version of the packed decode-partial-state storage contract.
pub const DECODE_PARTIAL_STATE_VERSION: u32 = 1;
/// Version of the exact paged KV block-table contract.
pub const PAGED_KV_BLOCK_TABLE_VERSION: u32 = 1;
/// Fixed number of logical tokens in every PR 10 physical KV block.
pub const PAGED_KV_BLOCK_SIZE: u64 = 16;
const _: () = assert!(F32_BYTES % BF16_BYTES == 0);
#[cfg(feature = "cuda")]
#[allow(clippy::cast_possible_truncation)] // Guarded by the adjacent const assertion.
const PAGED_KV_BLOCK_SIZE_ABI: u32 = PAGED_KV_BLOCK_SIZE as u32;
#[cfg(feature = "cuda")]
const _: () = assert!(PAGED_KV_BLOCK_SIZE <= u32::MAX as u64);

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

/// Allocation-free host mirror of one exact paged block table.
///
/// Physical IDs are stored in logical-block order and may be arbitrarily
/// shuffled. The constructor rejects duplicate or out-of-pool IDs and pins the
/// valid-token count of every logical block.
#[derive(Clone, Copy, Debug)]
pub struct PagedKvBlockTableHostV1<'a> {
    format_version: u32,
    block_ids: &'a [u32],
    valid_tokens: &'a [u16],
    logical_token_count: u64,
    physical_block_count: u64,
    block_count: u64,
}

impl<'a> PagedKvBlockTableHostV1<'a> {
    /// Creates a version-1 host table.
    ///
    /// # Errors
    ///
    /// Returns before CUDA execution for malformed lengths, invalid valid-token
    /// counts, duplicate or out-of-pool physical IDs, or arithmetic overflow.
    pub fn new(
        block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        logical_token_count: u64,
        physical_block_count: u64,
    ) -> CudaResult<Self> {
        Self::from_versioned_parts(
            PAGED_KV_BLOCK_TABLE_VERSION,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
        )
    }

    /// Creates a version-1 host table using caller-owned duplicate scratch.
    ///
    /// This is the allocation-free linear-time validation path for a prepared
    /// paged owner. `duplicate_scratch` must contain at least one byte per
    /// physical block; its prior contents are ignored and may be overwritten.
    ///
    /// # Errors
    ///
    /// Returns for insufficient scratch or the same malformed V1 fields as
    /// [`Self::new`].
    pub fn new_with_duplicate_scratch(
        block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        logical_token_count: u64,
        physical_block_count: u64,
        duplicate_scratch: &mut [u8],
    ) -> CudaResult<Self> {
        validate_paged_block_table_host_with_scratch(
            PAGED_KV_BLOCK_TABLE_VERSION,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
            duplicate_scratch,
        )?;
        Self::from_validated_parts(
            PAGED_KV_BLOCK_TABLE_VERSION,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
        )
    }

    /// Decodes an explicitly versioned table, rejecting unknown versions.
    ///
    /// # Errors
    ///
    /// Returns for an unsupported version or any invalid v1 field.
    pub fn from_versioned_parts(
        format_version: u32,
        block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        logical_token_count: u64,
        physical_block_count: u64,
    ) -> CudaResult<Self> {
        validate_paged_block_table_host(
            format_version,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
        )?;
        Self::from_validated_parts(
            format_version,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
        )
    }

    fn from_validated_parts(
        format_version: u32,
        block_ids: &'a [u32],
        valid_tokens: &'a [u16],
        logical_token_count: u64,
        physical_block_count: u64,
    ) -> CudaResult<Self> {
        let block_count = u64::try_from(block_ids.len()).map_err(|_| {
            CudaError::out_of_range(
                "PagedKvBlockTableHostV1::from_versioned_parts",
                "host block-id length does not fit u64",
            )
        })?;
        Ok(Self {
            format_version,
            block_ids,
            valid_tokens,
            logical_token_count,
            physical_block_count,
            block_count,
        })
    }

    #[must_use]
    pub const fn format_version(self) -> u32 {
        self.format_version
    }

    #[must_use]
    pub const fn block_ids(self) -> &'a [u32] {
        self.block_ids
    }

    #[must_use]
    pub const fn valid_tokens(self) -> &'a [u16] {
        self.valid_tokens
    }

    #[must_use]
    pub const fn logical_token_count(self) -> u64 {
        self.logical_token_count
    }

    #[must_use]
    pub const fn physical_block_count(self) -> u64 {
        self.physical_block_count
    }

    #[must_use]
    pub const fn block_count(self) -> u64 {
        self.block_count
    }
}

/// Paired host mirror and device arrays for [`PagedKvBlockTableHostV1`].
#[derive(Debug)]
pub struct PagedKvBlockTableV1<'a> {
    host: PagedKvBlockTableHostV1<'a>,
    device_block_ids: CudaBufferSpan<'a>,
    device_valid_tokens: CudaBufferSpan<'a>,
}

impl<'a> PagedKvBlockTableV1<'a> {
    /// Binds validated host metadata to pre-uploaded U32/U16 device arrays.
    ///
    /// The caller owns synchronization of the mirrored contents. This method
    /// validates dtype and capacity; execution additionally validates CUDA
    /// context ownership and idle state.
    ///
    /// # Errors
    ///
    /// Returns for an incompatible dtype or undersized device span.
    pub fn new(
        host: PagedKvBlockTableHostV1<'a>,
        device_block_ids: CudaBufferSpan<'a>,
        device_valid_tokens: CudaBufferSpan<'a>,
    ) -> CudaResult<Self> {
        const OPERATION: &str = "PagedKvBlockTableV1::new";
        require_dtype(
            OPERATION,
            "device_block_ids",
            device_block_ids.dtype(),
            CudaDType::U32,
        )?;
        require_dtype(
            OPERATION,
            "device_valid_tokens",
            device_valid_tokens.dtype(),
            CudaDType::U16,
        )?;
        require_capacity(
            OPERATION,
            "device_block_ids",
            device_block_ids.byte_len(),
            checked_bytes(
                OPERATION,
                &[host.block_count(), CudaDType::U32.size_bytes()],
            )?,
        )?;
        require_capacity(
            OPERATION,
            "device_valid_tokens",
            device_valid_tokens.byte_len(),
            checked_bytes(
                OPERATION,
                &[host.block_count(), CudaDType::U16.size_bytes()],
            )?,
        )?;
        Ok(Self {
            host,
            device_block_ids,
            device_valid_tokens,
        })
    }

    #[must_use]
    pub const fn host(&self) -> PagedKvBlockTableHostV1<'a> {
        self.host
    }

    #[must_use]
    pub const fn device_block_ids(&self) -> CudaBufferSpan<'a> {
        self.device_block_ids
    }

    #[must_use]
    pub const fn device_valid_tokens(&self) -> CudaBufferSpan<'a> {
        self.device_valid_tokens
    }
}

/// One dense-to-paged BF16 K/V scatter.
#[derive(Debug)]
pub struct PagedKvCacheAppendParams<'a> {
    /// BF16 `[T,KVH,D]` keys after `RoPE`.
    pub key_source: CudaBufferSpan<'a>,
    /// BF16 `[T,KVH,D]` values.
    pub value_source: CudaBufferSpan<'a>,
    /// BF16 `[physical_block,KVH,16,D]` key pool.
    pub key_pool: CudaBufferSpanMut<'a>,
    /// BF16 `[physical_block,KVH,16,D]` value pool.
    pub value_pool: CudaBufferSpanMut<'a>,
    /// Post-write logical address translation.
    pub block_table: PagedKvBlockTableV1<'a>,
    /// Number of dense source tokens.
    pub source_token_count: u64,
    /// First logical destination token.
    pub destination_token_start: u64,
    /// Number of K/V heads.
    pub key_value_head_count: u64,
    /// Elements per head.
    pub head_size: u64,
}

/// Scatters paired K/V rows into a fixed-block paged cache.
///
/// Validation completes before either pool is modified. Logical commit remains
/// the caller's responsibility after all layer writes succeed.
///
/// # Errors
///
/// Returns for invalid table metadata, range, dtype, capacity, ownership,
/// overlap, launch, or synchronization failure.
pub fn paged_kv_cache_append(
    params: &mut PagedKvCacheAppendParams<'_>,
    stream: &mut CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "paged_kv_cache_append";
    require_nonzero(OPERATION, "source_token_count", params.source_token_count)?;
    require_nonzero(
        OPERATION,
        "key_value_head_count",
        params.key_value_head_count,
    )?;
    require_nonzero(OPERATION, "head_size", params.head_size)?;
    let host = params.block_table.host();
    if params.destination_token_start > host.logical_token_count()
        || params.source_token_count > host.logical_token_count() - params.destination_token_start
    {
        return Err(CudaError::out_of_range(
            OPERATION,
            "paged cache destination range exceeds logical_token_count",
        ));
    }
    for (name, dtype) in [
        ("key_source", params.key_source.dtype()),
        ("value_source", params.value_source.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
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
    let pool_bytes = paged_pool_bytes(
        OPERATION,
        host.physical_block_count(),
        params.key_value_head_count,
        params.head_size,
    )?;
    for (name, actual, required) in [
        ("key_source", params.key_source.byte_len(), source_bytes),
        ("value_source", params.value_source.byte_len(), source_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.key_source.buffer(),
            params.value_source.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
            params.block_table.device_block_ids.buffer(),
            params.block_table.device_valid_tokens.buffer(),
        ],
    )?;

    #[cfg(feature = "cuda")]
    {
        ffi::paged_kv_cache_write_execute(
            params.key_source.raw(),
            params.value_source.raw(),
            params.key_pool.raw(),
            params.value_pool.raw(),
            params.block_table.device_block_ids.raw(),
            params.block_table.device_valid_tokens.raw(),
            host.format_version(),
            host.logical_token_count(),
            host.block_count(),
            host.physical_block_count(),
            PAGED_KV_BLOCK_SIZE_ABI,
            params.source_token_count,
            params.destination_token_start,
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
    /// Produces and merges packed F32 partial states. The reviewed
    /// 9QH/3KVH/D64 plan is an explicitly versioned hybrid: logical `T<=32`
    /// reuses the aligned workspace prefix for HF-eager materialized scores,
    /// while `T>=33` remains the ordinary online path.
    ChunkedOnline,
    /// Materializes staged BF16 scores and uses fixed-contiguous-37-balanced-v1
    /// for every D/T reduction.
    Fixed37Materialized,
    /// Recomputes scores twice and keeps only per-row F32 scratch in shared
    /// memory; supported for `D=64` and logical `T<=8192`.
    Fixed37TwoPass,
}

/// Why cold selection chose its backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DecodeAttentionSelectionReason {
    /// The caller explicitly required the reference.
    ExplicitReference,
    /// Every optimized capability matched.
    OptimizedCapabilityMatch,
    /// The reviewed 9QH/3KVH/D64 geometry selected the versioned
    /// HF-short/online hybrid production path.
    ReviewedHuggingFaceShortExactHybrid,
    /// The optimized implementation was not linked.
    OptimizedUnavailableFallback,
    /// The optimized kernel supports D64 only.
    UnsupportedHeadSizeFallback,
    /// The fixed partition/grid contract cannot represent the request.
    UnsupportedLaunchGeometryFallback,
    /// The fixed37 no-HBM backend supports at most logical `T=8192`.
    UnsupportedSequenceLengthFallback,
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
#[allow(clippy::struct_excessive_bools)]
pub struct DecodeAttentionBackendAvailability {
    reference: bool,
    chunked_online: bool,
    fixed37_materialized: bool,
    fixed37_two_pass: bool,
}

impl DecodeAttentionBackendAvailability {
    /// Creates an explicit availability snapshot.
    #[must_use]
    pub const fn new(reference: bool, chunked_online: bool) -> Self {
        Self {
            reference,
            chunked_online,
            fixed37_materialized: false,
            fixed37_two_pass: false,
        }
    }

    /// Adds the independently linked fixed37 materialized sibling.
    #[must_use]
    pub const fn with_fixed37_materialized(mut self, available: bool) -> Self {
        self.fixed37_materialized = available;
        self
    }

    /// Adds both independently linked fixed37 decode siblings.
    #[must_use]
    pub const fn with_fixed37(mut self, materialized: bool, two_pass: bool) -> Self {
        self.fixed37_materialized = materialized;
        self.fixed37_two_pass = two_pass;
        self
    }

    /// Native backends linked by the current feature set.
    #[must_use]
    pub const fn linked() -> Self {
        Self::new(cfg!(feature = "cuda"), cfg!(feature = "cuda"))
            .with_fixed37(cfg!(feature = "cuda"), cfg!(feature = "cuda"))
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

    /// Whether the fixed37 materialized contiguous/paged sibling is linked.
    #[must_use]
    pub const fn fixed37_materialized(self) -> bool {
        self.fixed37_materialized
    }

    /// Whether the fixed37 no-HBM contiguous/paged sibling is linked.
    #[must_use]
    pub const fn fixed37_two_pass(self) -> bool {
        self.fixed37_two_pass
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

/// Fixed dimensions for one query-length-one decode over a paged KV pool.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PagedDecodeAttentionRequest {
    maximum_sequence_length: u64,
    physical_block_count: u64,
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
    scale: f32,
}

impl PagedDecodeAttentionRequest {
    #[must_use]
    pub const fn new(
        maximum_sequence_length: u64,
        physical_block_count: u64,
        query_head_count: u64,
        key_value_head_count: u64,
        head_size: u64,
        scale: f32,
    ) -> Self {
        Self {
            maximum_sequence_length,
            physical_block_count,
            query_head_count,
            key_value_head_count,
            head_size,
            scale,
        }
    }

    #[must_use]
    pub const fn maximum_sequence_length(self) -> u64 {
        self.maximum_sequence_length
    }

    #[must_use]
    pub const fn physical_block_count(self) -> u64 {
        self.physical_block_count
    }

    #[must_use]
    pub const fn query_head_count(self) -> u64 {
        self.query_head_count
    }

    #[must_use]
    pub const fn key_value_head_count(self) -> u64 {
        self.key_value_head_count
    }

    #[must_use]
    pub const fn head_size(self) -> u64 {
        self.head_size
    }

    #[must_use]
    pub const fn scale(self) -> f32 {
        self.scale
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
    short_materialized_token_limit: Option<u64>,
    reduction_profile: AttentionReductionProfile,
    reduction_version: Option<u32>,
    reduction_chunk_elements: Option<u64>,
    maximum_reduction_elements: Option<u64>,
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

    /// Logical-token boundary for a versioned materialized-score prefix
    /// inside an otherwise online backend.
    #[must_use]
    pub const fn short_materialized_token_limit(self) -> Option<u64> {
        self.short_materialized_token_limit
    }

    /// Reduction order fixed into the backend.
    #[must_use]
    pub const fn reduction_profile(self) -> AttentionReductionProfile {
        self.reduction_profile
    }

    /// Fixed reduction contract version, or `None` for canonical order.
    #[must_use]
    pub const fn reduction_version(self) -> Option<u32> {
        self.reduction_version
    }

    /// Elements per fixed reduction chunk, or `None` for canonical order.
    #[must_use]
    pub const fn reduction_chunk_elements(self) -> Option<u64> {
        self.reduction_chunk_elements
    }

    /// Largest supported logical fixed reduction axis.
    #[must_use]
    pub const fn maximum_reduction_elements(self) -> Option<u64> {
        self.maximum_reduction_elements
    }
}

const REFERENCE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: REFERENCE_IMPLEMENTATION_ID,
    head_size: None,
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const ONLINE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: ONLINE_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: false,
    partial_state_merge: true,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const REVIEWED_HF_HYBRID_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: REVIEWED_HF_HYBRID_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: true,
    short_materialized_token_limit: Some(HUGGING_FACE_SHORT_DECODE_MAX_TOKENS),
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const PAGED_REFERENCE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: PAGED_REFERENCE_IMPLEMENTATION_ID,
    head_size: None,
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const PAGED_ONLINE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: PAGED_ONLINE_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: false,
    partial_state_merge: true,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const REVIEWED_HF_PAGED_HYBRID_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: REVIEWED_HF_PAGED_HYBRID_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: true,
    short_materialized_token_limit: Some(HUGGING_FACE_SHORT_DECODE_MAX_TOKENS),
    reduction_profile: AttentionReductionProfile::CanonicalV1,
    reduction_version: None,
    reduction_chunk_elements: None,
    maximum_reduction_elements: None,
};

const FIXED37_REFERENCE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: FIXED37_REFERENCE_IMPLEMENTATION_ID,
    head_size: None,
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_REDUCTION_ELEMENTS),
};

const FIXED37_PAGED_REFERENCE_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: FIXED37_PAGED_REFERENCE_IMPLEMENTATION_ID,
    head_size: None,
    accumulator_dtype: CudaDType::F32,
    materializes_scores: true,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_REDUCTION_ELEMENTS),
};

const FIXED37_TWO_PASS_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: FIXED37_TWO_PASS_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: false,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_TWO_PASS_TOKENS),
};

const FIXED37_PAGED_TWO_PASS_CAPABILITY: DecodeAttentionCapability = DecodeAttentionCapability {
    implementation_id: FIXED37_PAGED_TWO_PASS_IMPLEMENTATION_ID,
    head_size: Some(ONLINE_HEAD_SIZE),
    accumulator_dtype: CudaDType::F32,
    materializes_scores: false,
    partial_state_merge: false,
    short_materialized_token_limit: None,
    reduction_profile: AttentionReductionProfile::FixedContiguous37BalancedV1,
    reduction_version: Some(FIXED37_REDUCTION_VERSION),
    reduction_chunk_elements: Some(FIXED37_CHUNK_ELEMENTS as u64),
    maximum_reduction_elements: Some(FIXED37_MAX_TWO_PASS_TOKENS),
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
    short_materialized_token_limit: Option<u64>,
    reduction_profile: AttentionReductionProfile,
    dynamic_shared_memory_bytes: u64,
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
    #[must_use]
    pub const fn short_materialized_token_limit(self) -> Option<u64> {
        self.short_materialized_token_limit
    }
    /// Immutable reduction profile selected before execution.
    #[must_use]
    pub const fn reduction_profile(self) -> AttentionReductionProfile {
        self.reduction_profile
    }

    /// Maximum dynamic shared-memory bytes launched by this prepared backend.
    #[must_use]
    pub const fn dynamic_shared_memory_bytes(self) -> u64 {
        self.dynamic_shared_memory_bytes
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
    /// Reference BF16 scores or canonical packed F32 states.
    ///
    /// The fixed37 two-pass backend ignores this compatibility field. New
    /// callers that do not already own a workspace can use
    /// [`PreparedDecodeAttention::execute_without_workspace`].
    pub workspace: CudaBufferSpanMut<'a>,
}

/// Device views used by fixed37 two-pass decode without an HBM workspace.
#[derive(Debug)]
pub struct DecodeAttentionNoWorkspaceParams<'a> {
    /// BF16 query `[QH,D]` at the current position.
    pub query: CudaBufferSpan<'a>,
    /// BF16 head-major key cache `[KVH,M,D]`.
    pub key_cache: CudaBufferSpan<'a>,
    /// BF16 head-major value cache `[KVH,M,D]`.
    pub value_cache: CudaBufferSpan<'a>,
    /// BF16 output `[QH,D]`.
    pub output: CudaBufferSpanMut<'a>,
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
        Self::select_with_reduction_profile(
            context,
            request,
            preference,
            AttentionReductionProfile::CanonicalV1,
            availability,
        )
    }

    /// Cold-selects one backend without crossing reduction profiles.
    ///
    /// Fixed `Reference` selects materialized execution. Fixed `Optimized`
    /// prefers no-HBM D64/T8192 two-pass execution and may fall back only to
    /// fixed37 materialized execution. A canonical backend is never used.
    ///
    /// # Errors
    ///
    /// Returns when the selected profile is unavailable or its D/T axes exceed
    /// the fixed37 limit, in addition to ordinary request/architecture errors.
    pub fn select_with_reduction_profile(
        context: &CudaContext,
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_for_compute_capability_and_reduction_profile(
            context,
            context.compute_capability(),
            request,
            preference,
            reduction_profile,
            availability,
        )
    }

    #[cfg(all(test, not(feature = "cuda")))]
    fn select_for_compute_capability(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
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
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_request(request)?;
        require_architecture_support(compute_capability)?;
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
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
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
                    let reason = if is_reviewed_hugging_face_short_decode_shape(
                        request.query_head_count,
                        request.key_value_head_count,
                        request.head_size,
                    ) {
                        DecodeAttentionSelectionReason::ReviewedHuggingFaceShortExactHybrid
                    } else {
                        DecodeAttentionSelectionReason::OptimizedCapabilityMatch
                    };
                    return prepare_selection(
                        context,
                        compute_capability,
                        request,
                        DecodeAttentionBackend::ChunkedOnline,
                        reason,
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

    fn select_fixed37(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: DecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_fixed37_request_axes(
            "select_decode_attention",
            request.maximum_sequence_length,
            request.head_size,
        )?;
        match preference {
            DecodeAttentionPreference::Reference => {
                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_decode_attention",
                        "the fixed37 materialized decode backend is unavailable; canonical fallback is forbidden",
                    ));
                }
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::Fixed37Materialized,
                    DecodeAttentionSelectionReason::ExplicitReference,
                )
            }
            DecodeAttentionPreference::Optimized => {
                let two_pass_reason = if !availability.fixed37_two_pass {
                    DecodeAttentionSelectionReason::OptimizedUnavailableFallback
                } else if request.head_size != ONLINE_HEAD_SIZE {
                    DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback
                } else if request.maximum_sequence_length > FIXED37_MAX_TWO_PASS_TOKENS {
                    DecodeAttentionSelectionReason::UnsupportedSequenceLengthFallback
                } else {
                    return prepare_selection(
                        context,
                        compute_capability,
                        request,
                        DecodeAttentionBackend::Fixed37TwoPass,
                        DecodeAttentionSelectionReason::OptimizedCapabilityMatch,
                    );
                };
                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_decode_attention",
                        format!(
                            "fixed37 two-pass was rejected ({two_pass_reason:?}), fixed37 materialized is unavailable, and canonical fallback is forbidden"
                        ),
                    ));
                }
                prepare_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::Fixed37Materialized,
                    two_pass_reason,
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
    /// Reduction profile fixed into this prepared plan.
    #[must_use]
    pub const fn reduction_profile(&self) -> AttentionReductionProfile {
        self.capability.reduction_profile
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
    #[allow(clippy::too_many_lines)]
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
            DecodeAttentionBackend::Fixed37Materialized => {
                ffi::fixed37_decode_attention_reference_execute(
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
            DecodeAttentionBackend::Fixed37TwoPass => {
                ffi::fixed37_decode_attention_two_pass_execute(
                    params.query.raw(),
                    params.key_cache.raw(),
                    params.value_cache.raw(),
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

    /// Executes the fixed37 two-pass backend without an HBM workspace.
    ///
    /// This method fails closed before launch unless this prepared plan's
    /// backend is exactly [`DecodeAttentionBackend::Fixed37TwoPass`]. No
    /// allocation or backend fallback occurs here.
    ///
    /// # Errors
    ///
    /// Returns for a non-two-pass plan, logical-length, dtype, capacity,
    /// ownership, native launch, or synchronization failure.
    pub fn execute_without_workspace(
        &self,
        logical_token_count: u64,
        params: &mut DecodeAttentionNoWorkspaceParams<'_>,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "fixed37_two_pass_decode_attention";
        if self.backend != DecodeAttentionBackend::Fixed37TwoPass {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "the prepared decode backend is not fixed37 two-pass",
            ));
        }
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        require_nonzero(OPERATION, "logical_token_count", logical_token_count)?;
        if logical_token_count > self.request.maximum_sequence_length {
            return Err(CudaError::out_of_range(
                OPERATION,
                "logical_token_count exceeds the prepared cache capacity",
            ));
        }
        validate_no_workspace_execute_params(self, params, stream)?;

        #[cfg(feature = "cuda")]
        {
            ffi::fixed37_decode_attention_two_pass_execute(
                params.query.raw(),
                params.key_cache.raw(),
                params.value_cache.raw(),
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
        #[cfg(not(feature = "cuda"))]
        {
            let _ = params;
            Err(CudaError::unavailable(OPERATION))
        }
    }
}

/// Device views used by one paged decode execution.
#[derive(Debug)]
pub struct PagedDecodeAttentionParams<'a> {
    /// BF16 query `[QH,D]`.
    pub query: CudaBufferSpan<'a>,
    /// BF16 key pool `[physical_block,KVH,16,D]`.
    pub key_pool: CudaBufferSpan<'a>,
    /// BF16 value pool `[physical_block,KVH,16,D]`.
    pub value_pool: CudaBufferSpan<'a>,
    /// BF16 reference scores or F32 packed states.
    ///
    /// The fixed37 two-pass backend ignores this compatibility field. New
    /// callers can use [`PreparedPagedDecodeAttention::execute_without_workspace`].
    pub workspace: CudaBufferSpanMut<'a>,
    /// BF16 output `[QH,D]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Exact logical-to-physical address translation.
    pub block_table: PagedKvBlockTableV1<'a>,
}

/// Device views used by fixed37 paged two-pass decode without an HBM workspace.
#[derive(Debug)]
pub struct PagedDecodeAttentionNoWorkspaceParams<'a> {
    /// BF16 query `[QH,D]`.
    pub query: CudaBufferSpan<'a>,
    /// BF16 key pool `[physical_block,KVH,16,D]`.
    pub key_pool: CudaBufferSpan<'a>,
    /// BF16 value pool `[physical_block,KVH,16,D]`.
    pub value_pool: CudaBufferSpan<'a>,
    /// BF16 output `[QH,D]`.
    pub output: CudaBufferSpanMut<'a>,
    /// Exact logical-to-physical address translation.
    pub block_table: PagedKvBlockTableV1<'a>,
}

/// Immutable paged decode selection bound to one CUDA context owner.
#[derive(Clone)]
pub struct PreparedPagedDecodeAttention {
    context: Arc<ContextInner>,
    request: PagedDecodeAttentionRequest,
    backend: DecodeAttentionBackend,
    capability: DecodeAttentionCapability,
    trace: DecodeAttentionSelectionTrace,
}

impl fmt::Debug for PreparedPagedDecodeAttention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedPagedDecodeAttention")
            .field("device_ordinal", &self.context.ordinal)
            .field("request", &self.request)
            .field("backend", &self.backend)
            .field("capability", &self.capability)
            .field("trace", &self.trace)
            .finish()
    }
}

impl PreparedPagedDecodeAttention {
    /// Cold-selects the materialized or exact D64 online paged backend.
    ///
    /// # Errors
    ///
    /// Returns for invalid dimensions, arithmetic overflow, architecture
    /// incompatibility, or unavailable required backends.
    pub fn select(
        context: &CudaContext,
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_with_reduction_profile(
            context,
            request,
            preference,
            AttentionReductionProfile::CanonicalV1,
            availability,
        )
    }

    /// Cold-selects one paged backend without crossing reduction profiles.
    ///
    /// Fixed `Optimized` prefers no-HBM D64/T8192 two-pass execution and may
    /// fall back only to the fixed37 materialized paged sibling.
    ///
    /// # Errors
    ///
    /// Returns when the selected profile is unavailable or its logical D/T
    /// axes exceed the fixed37 limit, in addition to ordinary request errors.
    pub fn select_with_reduction_profile(
        context: &CudaContext,
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        Self::select_for_compute_capability_and_reduction_profile(
            context,
            context.compute_capability(),
            request,
            preference,
            reduction_profile,
            availability,
        )
    }

    #[cfg(all(test, not(feature = "cuda")))]
    fn select_for_compute_capability(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
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
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        reduction_profile: AttentionReductionProfile,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_paged_request(request)?;
        require_architecture_support(compute_capability)?;
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
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        match preference {
            DecodeAttentionPreference::Reference => {
                if !availability.reference {
                    return Err(not_supported(
                        "select_paged_decode_attention",
                        "the explicitly requested paged reference is unavailable",
                    ));
                }
                prepare_paged_selection(
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
                } else if !paged_optimized_geometry_supported(request)? {
                    DecodeAttentionSelectionReason::UnsupportedLaunchGeometryFallback
                } else {
                    let reason = if is_reviewed_hugging_face_short_decode_shape(
                        request.query_head_count,
                        request.key_value_head_count,
                        request.head_size,
                    ) {
                        DecodeAttentionSelectionReason::ReviewedHuggingFaceShortExactHybrid
                    } else {
                        DecodeAttentionSelectionReason::OptimizedCapabilityMatch
                    };
                    return prepare_paged_selection(
                        context,
                        compute_capability,
                        request,
                        DecodeAttentionBackend::ChunkedOnline,
                        reason,
                    );
                };
                if !availability.reference {
                    return Err(not_supported(
                        "select_paged_decode_attention",
                        format!(
                            "optimized paged decode was rejected ({fallback_reason:?}) and the reference is unavailable"
                        ),
                    ));
                }
                prepare_paged_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::MaterializedReference,
                    fallback_reason,
                )
            }
        }
    }

    fn select_fixed37(
        context: &CudaContext,
        compute_capability: (u32, u32),
        request: PagedDecodeAttentionRequest,
        preference: DecodeAttentionPreference,
        availability: DecodeAttentionBackendAvailability,
    ) -> CudaResult<Self> {
        validate_fixed37_request_axes(
            "select_paged_decode_attention",
            request.maximum_sequence_length,
            request.head_size,
        )?;
        match preference {
            DecodeAttentionPreference::Reference => {
                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_paged_decode_attention",
                        "the fixed37 materialized paged decode backend is unavailable; canonical fallback is forbidden",
                    ));
                }
                prepare_paged_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::Fixed37Materialized,
                    DecodeAttentionSelectionReason::ExplicitReference,
                )
            }
            DecodeAttentionPreference::Optimized => {
                let two_pass_reason = if !availability.fixed37_two_pass {
                    DecodeAttentionSelectionReason::OptimizedUnavailableFallback
                } else if request.head_size != ONLINE_HEAD_SIZE {
                    DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback
                } else if request.maximum_sequence_length > FIXED37_MAX_TWO_PASS_TOKENS {
                    DecodeAttentionSelectionReason::UnsupportedSequenceLengthFallback
                } else {
                    return prepare_paged_selection(
                        context,
                        compute_capability,
                        request,
                        DecodeAttentionBackend::Fixed37TwoPass,
                        DecodeAttentionSelectionReason::OptimizedCapabilityMatch,
                    );
                };
                if !availability.fixed37_materialized {
                    return Err(not_supported(
                        "select_paged_decode_attention",
                        format!(
                            "fixed37 two-pass was rejected ({two_pass_reason:?}), fixed37 materialized is unavailable, and canonical fallback is forbidden"
                        ),
                    ));
                }
                prepare_paged_selection(
                    context,
                    compute_capability,
                    request,
                    DecodeAttentionBackend::Fixed37Materialized,
                    two_pass_reason,
                )
            }
        }
    }

    #[must_use]
    pub const fn request(&self) -> PagedDecodeAttentionRequest {
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

    /// Reduction profile fixed into this prepared plan.
    #[must_use]
    pub const fn reduction_profile(&self) -> AttentionReductionProfile {
        self.capability.reduction_profile
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

    /// Executes the cold-selected exact backend without allocating.
    ///
    /// # Errors
    ///
    /// Returns before launch for malformed table metadata, incompatible
    /// pool geometry, dtype/capacity/context violations, or native failure.
    #[allow(clippy::too_many_lines)]
    pub fn execute(
        &self,
        params: &mut PagedDecodeAttentionParams<'_>,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "paged_decode_attention";
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        let host = params.block_table.host();
        if host.logical_token_count() > self.request.maximum_sequence_length {
            return Err(CudaError::out_of_range(
                OPERATION,
                "block table logical length exceeds the prepared sequence capacity",
            ));
        }
        if host.physical_block_count() != self.request.physical_block_count {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "block table physical pool size differs from the prepared request",
            ));
        }
        validate_paged_execute_params(self, params, stream)?;

        #[cfg(feature = "cuda")]
        match self.backend {
            DecodeAttentionBackend::MaterializedReference => {
                ffi::paged_decode_attention_reference_execute(
                    params.query.raw(),
                    params.key_pool.raw(),
                    params.value_pool.raw(),
                    params.workspace.raw(),
                    params.output.raw(),
                    params.block_table.device_block_ids.raw(),
                    params.block_table.device_valid_tokens.raw(),
                    host.format_version(),
                    host.logical_token_count(),
                    host.block_count(),
                    host.physical_block_count(),
                    PAGED_KV_BLOCK_SIZE_ABI,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            DecodeAttentionBackend::Fixed37Materialized => {
                ffi::fixed37_paged_decode_attention_reference_execute(
                    params.query.raw(),
                    params.key_pool.raw(),
                    params.value_pool.raw(),
                    params.workspace.raw(),
                    params.output.raw(),
                    params.block_table.device_block_ids.raw(),
                    params.block_table.device_valid_tokens.raw(),
                    host.format_version(),
                    host.logical_token_count(),
                    host.block_count(),
                    host.physical_block_count(),
                    PAGED_KV_BLOCK_SIZE_ABI,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            DecodeAttentionBackend::Fixed37TwoPass => {
                ffi::fixed37_paged_decode_attention_two_pass_execute(
                    params.query.raw(),
                    params.key_pool.raw(),
                    params.value_pool.raw(),
                    params.output.raw(),
                    params.block_table.device_block_ids.raw(),
                    params.block_table.device_valid_tokens.raw(),
                    host.format_version(),
                    host.logical_token_count(),
                    host.block_count(),
                    host.physical_block_count(),
                    PAGED_KV_BLOCK_SIZE_ABI,
                    self.request.query_head_count,
                    self.request.key_value_head_count,
                    self.request.head_size,
                    self.request.scale,
                    &mut stream.native,
                )
            }
            DecodeAttentionBackend::ChunkedOnline => ffi::paged_decode_attention_execute(
                params.query.raw(),
                params.key_pool.raw(),
                params.value_pool.raw(),
                params.workspace.raw(),
                params.output.raw(),
                params.block_table.device_block_ids.raw(),
                params.block_table.device_valid_tokens.raw(),
                host.format_version(),
                host.logical_token_count(),
                host.block_count(),
                host.physical_block_count(),
                PAGED_KV_BLOCK_SIZE_ABI,
                self.request.query_head_count,
                self.request.key_value_head_count,
                self.request.head_size,
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

    /// Executes fixed37 paged two-pass decode without an HBM workspace.
    ///
    /// This method fails closed before launch unless this prepared plan's
    /// backend is exactly [`DecodeAttentionBackend::Fixed37TwoPass`].
    ///
    /// # Errors
    ///
    /// Returns for a non-two-pass plan, malformed table metadata, incompatible
    /// pool geometry, dtype/capacity/context violations, or native failure.
    pub fn execute_without_workspace(
        &self,
        params: &mut PagedDecodeAttentionNoWorkspaceParams<'_>,
        stream: &mut CudaStream,
    ) -> CudaResult<()> {
        const OPERATION: &str = "fixed37_two_pass_paged_decode_attention";
        if self.backend != DecodeAttentionBackend::Fixed37TwoPass {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "the prepared paged decode backend is not fixed37 two-pass",
            ));
        }
        ensure_same_context(&self.context, &stream.context, OPERATION)?;
        let host = params.block_table.host();
        if host.logical_token_count() > self.request.maximum_sequence_length {
            return Err(CudaError::out_of_range(
                OPERATION,
                "block table logical length exceeds the prepared sequence capacity",
            ));
        }
        if host.physical_block_count() != self.request.physical_block_count {
            return Err(CudaError::invalid_argument(
                OPERATION,
                "block table physical pool size differs from the prepared request",
            ));
        }
        validate_paged_no_workspace_execute_params(self, params, stream)?;

        #[cfg(feature = "cuda")]
        {
            ffi::fixed37_paged_decode_attention_two_pass_execute(
                params.query.raw(),
                params.key_pool.raw(),
                params.value_pool.raw(),
                params.output.raw(),
                params.block_table.device_block_ids.raw(),
                params.block_table.device_valid_tokens.raw(),
                host.format_version(),
                host.logical_token_count(),
                host.block_count(),
                host.physical_block_count(),
                PAGED_KV_BLOCK_SIZE_ABI,
                self.request.query_head_count,
                self.request.key_value_head_count,
                self.request.head_size,
                self.request.scale,
                &mut stream.native,
            )
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

#[allow(clippy::too_many_lines)]
fn prepare_paged_selection(
    context: &CudaContext,
    compute_capability: (u32, u32),
    request: PagedDecodeAttentionRequest,
    backend: DecodeAttentionBackend,
    reason: DecodeAttentionSelectionReason,
) -> CudaResult<PreparedPagedDecodeAttention> {
    let (
        capability,
        workspace_dtype,
        workspace_bytes,
        materialized_score_bytes,
        partial_state_bytes,
        partial_state_capacity,
    ) = match backend {
        DecodeAttentionBackend::MaterializedReference => {
            let bytes = checked_bytes(
                "select_paged_decode_attention",
                &[
                    request.query_head_count,
                    request.maximum_sequence_length,
                    BF16_BYTES,
                ],
            )?;
            (
                PAGED_REFERENCE_CAPABILITY,
                CudaDType::BF16,
                bytes,
                bytes,
                0,
                0,
            )
        }
        DecodeAttentionBackend::Fixed37Materialized => {
            let bytes = checked_bytes(
                "select_paged_decode_attention",
                &[
                    request.query_head_count,
                    request.maximum_sequence_length,
                    BF16_BYTES,
                ],
            )?;
            (
                FIXED37_PAGED_REFERENCE_CAPABILITY,
                CudaDType::BF16,
                bytes,
                bytes,
                0,
                0,
            )
        }
        DecodeAttentionBackend::Fixed37TwoPass => (
            FIXED37_PAGED_TWO_PASS_CAPABILITY,
            CudaDType::BF16,
            0,
            0,
            0,
            0,
        ),
        DecodeAttentionBackend::ChunkedOnline => {
            let capacity = paged_block_count(request.maximum_sequence_length)?;
            let layout = DecodePartialStateLayout::new(
                capacity,
                request.query_head_count,
                request.head_size,
            )?;
            let hybrid = is_reviewed_hugging_face_short_decode_shape(
                request.query_head_count,
                request.key_value_head_count,
                request.head_size,
            );
            let materialized_score_bytes = if hybrid {
                hugging_face_short_score_prefix_bytes(
                    "select_paged_decode_attention",
                    request.query_head_count,
                    layout.byte_len(),
                )?
            } else {
                0
            };
            (
                if hybrid {
                    REVIEWED_HF_PAGED_HYBRID_CAPABILITY
                } else {
                    PAGED_ONLINE_CAPABILITY
                },
                CudaDType::F32,
                layout.byte_len(),
                materialized_score_bytes,
                layout.byte_len(),
                capacity,
            )
        }
    };
    Ok(PreparedPagedDecodeAttention {
        context: Arc::clone(&context.inner),
        request,
        backend,
        capability,
        trace: DecodeAttentionSelectionTrace {
            reason,
            implementation_id: capability.implementation_id,
            implementation_version: if capability.short_materialized_token_limit.is_some() {
                REVIEWED_HF_HYBRID_IMPLEMENTATION_VERSION
            } else {
                IMPLEMENTATION_VERSION
            },
            native_dependency: NATIVE_DEPENDENCY,
            compiled_architectures: CUDA_COMPILED_ARCHITECTURES,
            device_ordinal: context.device_ordinal(),
            compute_capability,
            workspace_dtype,
            workspace_bytes,
            materialized_score_bytes,
            partial_state_bytes,
            partial_state_capacity,
            tokens_per_partition: match backend {
                DecodeAttentionBackend::MaterializedReference
                | DecodeAttentionBackend::ChunkedOnline => PAGED_KV_BLOCK_SIZE,
                DecodeAttentionBackend::Fixed37Materialized
                | DecodeAttentionBackend::Fixed37TwoPass => 0,
            },
            short_materialized_token_limit: capability.short_materialized_token_limit,
            reduction_profile: capability.reduction_profile,
            dynamic_shared_memory_bytes: match backend {
                DecodeAttentionBackend::Fixed37Materialized => fixed37_reduction_shared_bytes(
                    "select_paged_decode_attention",
                    request.maximum_sequence_length.max(request.head_size),
                )?,
                DecodeAttentionBackend::Fixed37TwoPass => fixed37_two_pass_shared_bytes(
                    "select_paged_decode_attention",
                    request.maximum_sequence_length,
                    request.head_size,
                )?,
                _ => 0,
            },
        },
    })
}

fn validate_paged_request(request: PagedDecodeAttentionRequest) -> CudaResult<()> {
    const OPERATION: &str = "select_paged_decode_attention";
    for (name, value) in [
        ("maximum_sequence_length", request.maximum_sequence_length),
        ("physical_block_count", request.physical_block_count),
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
    paged_pool_bytes(
        OPERATION,
        request.physical_block_count,
        request.key_value_head_count,
        request.head_size,
    )?;
    paged_block_count(request.maximum_sequence_length)?;
    Ok(())
}

fn paged_optimized_geometry_supported(request: PagedDecodeAttentionRequest) -> CudaResult<bool> {
    Ok(
        paged_block_count(request.maximum_sequence_length)? <= MAXIMUM_GRID_X
            && request.query_head_count <= MAXIMUM_GRID_Y,
    )
}

fn validate_paged_execute_params(
    prepared: &PreparedPagedDecodeAttention,
    params: &PagedDecodeAttentionParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "paged_decode_attention";
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let workspace = if prepared.backend == DecodeAttentionBackend::Fixed37TwoPass {
        None
    } else {
        require_dtype(
            OPERATION,
            "workspace",
            params.workspace.dtype(),
            prepared.workspace_dtype(),
        )?;
        Some(&params.workspace)
    };
    let query_bytes = checked_bytes(
        OPERATION,
        &[
            prepared.request.query_head_count,
            prepared.request.head_size,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        OPERATION,
        prepared.request.physical_block_count,
        prepared.request.key_value_head_count,
        prepared.request.head_size,
    )?;
    for (name, actual, required) in [
        ("query", params.query.byte_len(), query_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
        ("output", params.output.byte_len(), query_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    if let Some(workspace) = workspace {
        require_capacity(
            OPERATION,
            "workspace",
            workspace.byte_len(),
            prepared.workspace_bytes(),
        )?;
        validate_resources(
            OPERATION,
            stream,
            &[
                params.query.buffer(),
                params.key_pool.buffer(),
                params.value_pool.buffer(),
                workspace.buffer(),
                params.output.buffer(),
                params.block_table.device_block_ids.buffer(),
                params.block_table.device_valid_tokens.buffer(),
            ],
        )
    } else {
        validate_resources(
            OPERATION,
            stream,
            &[
                params.query.buffer(),
                params.key_pool.buffer(),
                params.value_pool.buffer(),
                params.output.buffer(),
                params.block_table.device_block_ids.buffer(),
                params.block_table.device_valid_tokens.buffer(),
            ],
        )
    }
}

fn validate_paged_no_workspace_execute_params(
    prepared: &PreparedPagedDecodeAttention,
    params: &PagedDecodeAttentionNoWorkspaceParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_two_pass_paged_decode_attention";
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_pool", params.key_pool.dtype()),
        ("value_pool", params.value_pool.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
    let query_bytes = checked_bytes(
        OPERATION,
        &[
            prepared.request.query_head_count,
            prepared.request.head_size,
            BF16_BYTES,
        ],
    )?;
    let pool_bytes = paged_pool_bytes(
        OPERATION,
        prepared.request.physical_block_count,
        prepared.request.key_value_head_count,
        prepared.request.head_size,
    )?;
    for (name, actual, required) in [
        ("query", params.query.byte_len(), query_bytes),
        ("key_pool", params.key_pool.byte_len(), pool_bytes),
        ("value_pool", params.value_pool.byte_len(), pool_bytes),
        ("output", params.output.byte_len(), query_bytes),
    ] {
        require_capacity(OPERATION, name, actual, required)?;
    }
    validate_resources(
        OPERATION,
        stream,
        &[
            params.query.buffer(),
            params.key_pool.buffer(),
            params.value_pool.buffer(),
            params.output.buffer(),
            params.block_table.device_block_ids.buffer(),
            params.block_table.device_valid_tokens.buffer(),
        ],
    )
}

fn validate_paged_block_table_host(
    format_version: u32,
    block_ids: &[u32],
    valid_tokens: &[u16],
    logical_token_count: u64,
    physical_block_count: u64,
) -> CudaResult<()> {
    const OPERATION: &str = "validate_paged_block_table";
    validate_paged_block_table_shape(
        format_version,
        block_ids,
        valid_tokens,
        logical_token_count,
        physical_block_count,
    )?;
    for (logical_index, (&physical_id, &valid)) in block_ids.iter().zip(valid_tokens).enumerate() {
        validate_paged_block_entry(
            block_ids.len(),
            logical_index,
            physical_id,
            valid,
            logical_token_count,
            physical_block_count,
        )?;
        if block_ids[..logical_index].contains(&physical_id) {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("physical block id {physical_id} appears more than once"),
            ));
        }
    }
    Ok(())
}

fn validate_paged_block_table_host_with_scratch(
    format_version: u32,
    block_ids: &[u32],
    valid_tokens: &[u16],
    logical_token_count: u64,
    physical_block_count: u64,
    duplicate_scratch: &mut [u8],
) -> CudaResult<()> {
    const OPERATION: &str = "validate_paged_block_table";
    validate_paged_block_table_shape(
        format_version,
        block_ids,
        valid_tokens,
        logical_token_count,
        physical_block_count,
    )?;
    let physical_blocks = usize::try_from(physical_block_count).map_err(|_| {
        CudaError::out_of_range(OPERATION, "physical_block_count does not fit host usize")
    })?;
    if duplicate_scratch.len() < physical_blocks {
        return Err(CudaError::out_of_range(
            OPERATION,
            "duplicate scratch is smaller than physical_block_count",
        ));
    }
    duplicate_scratch[..physical_blocks].fill(0);
    for (logical_index, (&physical_id, &valid)) in block_ids.iter().zip(valid_tokens).enumerate() {
        validate_paged_block_entry(
            block_ids.len(),
            logical_index,
            physical_id,
            valid,
            logical_token_count,
            physical_block_count,
        )?;
        let physical_index = usize::try_from(physical_id)
            .expect("a U32 physical block ID fits any supported host usize");
        if duplicate_scratch[physical_index] != 0 {
            return Err(CudaError::invalid_argument(
                OPERATION,
                format!("physical block id {physical_id} appears more than once"),
            ));
        }
        duplicate_scratch[physical_index] = 1;
    }
    Ok(())
}

fn validate_paged_block_table_shape(
    format_version: u32,
    block_ids: &[u32],
    valid_tokens: &[u16],
    logical_token_count: u64,
    physical_block_count: u64,
) -> CudaResult<()> {
    const OPERATION: &str = "validate_paged_block_table";
    if format_version != PAGED_KV_BLOCK_TABLE_VERSION {
        return Err(not_supported(
            OPERATION,
            format!("unsupported paged block-table version {format_version}"),
        ));
    }
    require_nonzero(OPERATION, "logical_token_count", logical_token_count)?;
    require_nonzero(OPERATION, "physical_block_count", physical_block_count)?;
    let expected_blocks = paged_block_count(logical_token_count)?;
    let actual_blocks = u64::try_from(block_ids.len())
        .map_err(|_| CudaError::out_of_range(OPERATION, "host block-id length does not fit u64"))?;
    let actual_valid = u64::try_from(valid_tokens.len()).map_err(|_| {
        CudaError::out_of_range(OPERATION, "host valid-token length does not fit u64")
    })?;
    if actual_blocks != expected_blocks || actual_valid != expected_blocks {
        return Err(CudaError::invalid_argument(
            OPERATION,
            "host block_ids/valid_tokens lengths do not match logical_token_count",
        ));
    }
    if expected_blocks > physical_block_count {
        return Err(CudaError::out_of_range(
            OPERATION,
            "logical block count exceeds physical_block_count",
        ));
    }
    Ok(())
}

fn validate_paged_block_entry(
    block_count: usize,
    logical_index: usize,
    physical_id: u32,
    valid: u16,
    logical_token_count: u64,
    physical_block_count: u64,
) -> CudaResult<()> {
    const OPERATION: &str = "validate_paged_block_table";
    if u64::from(physical_id) >= physical_block_count {
        return Err(CudaError::out_of_range(
            OPERATION,
            format!("physical block id {physical_id} is outside the pool"),
        ));
    }
    let expected_valid = if logical_index + 1 == block_count {
        u16::try_from(((logical_token_count - 1) % PAGED_KV_BLOCK_SIZE) + 1)
            .expect("fixed block size fits u16")
    } else {
        u16::try_from(PAGED_KV_BLOCK_SIZE).expect("fixed block size fits u16")
    };
    if valid != expected_valid {
        return Err(CudaError::invalid_argument(
            OPERATION,
            format!(
                "logical block {logical_index} has {valid} valid tokens; expected {expected_valid}"
            ),
        ));
    }
    Ok(())
}

fn paged_block_count(logical_token_count: u64) -> CudaResult<u64> {
    require_nonzero(
        "paged_block_count",
        "logical_token_count",
        logical_token_count,
    )?;
    Ok(((logical_token_count - 1) / PAGED_KV_BLOCK_SIZE) + 1)
}

fn paged_pool_bytes(
    operation: &'static str,
    physical_block_count: u64,
    key_value_head_count: u64,
    head_size: u64,
) -> CudaResult<u64> {
    checked_bytes(
        operation,
        &[
            physical_block_count,
            key_value_head_count,
            PAGED_KV_BLOCK_SIZE,
            head_size,
            BF16_BYTES,
        ],
    )
}

#[allow(clippy::too_many_lines)]
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
        DecodeAttentionBackend::Fixed37Materialized => {
            let bytes = checked_bytes(
                "select_decode_attention",
                &[
                    request.query_head_count,
                    request.maximum_sequence_length,
                    BF16_BYTES,
                ],
            )?;
            (
                FIXED37_REFERENCE_CAPABILITY,
                CudaDType::BF16,
                bytes,
                bytes,
                0,
                0,
                0,
            )
        }
        DecodeAttentionBackend::Fixed37TwoPass => {
            (FIXED37_TWO_PASS_CAPABILITY, CudaDType::BF16, 0, 0, 0, 0, 0)
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
            let hybrid = is_reviewed_hugging_face_short_decode_shape(
                request.query_head_count,
                request.key_value_head_count,
                request.head_size,
            );
            let materialized_score_bytes = if hybrid {
                hugging_face_short_score_prefix_bytes(
                    "select_decode_attention",
                    request.query_head_count,
                    layout.byte_len(),
                )?
            } else {
                0
            };
            (
                if hybrid {
                    REVIEWED_HF_HYBRID_CAPABILITY
                } else {
                    ONLINE_CAPABILITY
                },
                CudaDType::F32,
                layout.byte_len(),
                materialized_score_bytes,
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
            implementation_version: if capability.short_materialized_token_limit.is_some() {
                REVIEWED_HF_HYBRID_IMPLEMENTATION_VERSION
            } else {
                IMPLEMENTATION_VERSION
            },
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
            short_materialized_token_limit: capability.short_materialized_token_limit,
            reduction_profile: capability.reduction_profile,
            dynamic_shared_memory_bytes: match backend {
                DecodeAttentionBackend::Fixed37Materialized => fixed37_reduction_shared_bytes(
                    "select_decode_attention",
                    request.maximum_sequence_length.max(request.head_size),
                )?,
                DecodeAttentionBackend::Fixed37TwoPass => fixed37_two_pass_shared_bytes(
                    "select_decode_attention",
                    request.maximum_sequence_length,
                    request.head_size,
                )?,
                _ => 0,
            },
        },
    })
}

fn fixed37_two_pass_shared_bytes(
    operation: &'static str,
    token_count: u64,
    head_size: u64,
) -> CudaResult<u64> {
    let token_chunks = fixed37_chunk_count(operation, token_count)?;
    let depth_chunks = fixed37_chunk_count(operation, head_size)?;
    let partial_capacity = token_chunks.max(2).max(depth_chunks);
    let score_bytes = token_count.checked_mul(F32_BYTES).ok_or_else(|| {
        CudaError::out_of_range(operation, "two-pass score scratch arithmetic overflow")
    })?;
    let partial_bytes = partial_capacity.checked_mul(2 * F32_BYTES).ok_or_else(|| {
        CudaError::out_of_range(operation, "two-pass partial scratch arithmetic overflow")
    })?;
    score_bytes.checked_add(partial_bytes).ok_or_else(|| {
        CudaError::out_of_range(operation, "two-pass shared-memory arithmetic overflow")
    })
}

fn fixed37_reduction_shared_bytes(operation: &'static str, element_count: u64) -> CudaResult<u64> {
    fixed37_chunk_count(operation, element_count)?
        .checked_mul(2 * F32_BYTES)
        .ok_or_else(|| {
            CudaError::out_of_range(operation, "fixed37 shared-memory arithmetic overflow")
        })
}

fn fixed37_chunk_count(operation: &'static str, elements: u64) -> CudaResult<u64> {
    elements
        .checked_add(u64::from(FIXED37_CHUNK_ELEMENTS) - 1)
        .map(|rounded| rounded / u64::from(FIXED37_CHUNK_ELEMENTS))
        .ok_or_else(|| {
            CudaError::out_of_range(operation, "fixed37 chunk-count arithmetic overflow")
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

fn validate_fixed37_request_axes(
    operation: &'static str,
    logical_token_count: u64,
    head_size: u64,
) -> CudaResult<()> {
    for (name, value) in [
        ("logical token axis", logical_token_count),
        ("head-size axis", head_size),
    ] {
        if value > FIXED37_MAX_REDUCTION_ELEMENTS {
            return Err(not_supported(
                operation,
                format!(
                    "fixed-contiguous-37-balanced-v1 {name} {value} exceeds the supported maximum {FIXED37_MAX_REDUCTION_ELEMENTS}"
                ),
            ));
        }
    }
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
    let workspace = if prepared.backend == DecodeAttentionBackend::Fixed37TwoPass {
        None
    } else {
        require_dtype(
            OPERATION,
            "workspace",
            params.workspace.dtype(),
            prepared.workspace_dtype(),
        )?;
        Some(&params.workspace)
    };
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
    if let Some(workspace) = workspace {
        require_capacity(
            OPERATION,
            "workspace",
            workspace.byte_len(),
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
                workspace.buffer(),
            ],
        )
    } else {
        validate_resources(
            OPERATION,
            stream,
            &[
                params.query.buffer(),
                params.key_cache.buffer(),
                params.value_cache.buffer(),
                params.output.buffer(),
            ],
        )
    }
}

fn validate_no_workspace_execute_params(
    prepared: &PreparedDecodeAttention,
    params: &DecodeAttentionNoWorkspaceParams<'_>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const OPERATION: &str = "fixed37_two_pass_decode_attention";
    for (name, dtype) in [
        ("query", params.query.dtype()),
        ("key_cache", params.key_cache.dtype()),
        ("value_cache", params.value_cache.dtype()),
        ("output", params.output.dtype()),
    ] {
        require_dtype(OPERATION, name, dtype, CudaDType::BF16)?;
    }
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
    validate_resources(
        OPERATION,
        stream,
        &[
            params.query.buffer(),
            params.key_cache.buffer(),
            params.value_cache.buffer(),
            params.output.buffer(),
        ],
    )
}

const fn is_reviewed_hugging_face_short_decode_shape(
    query_head_count: u64,
    key_value_head_count: u64,
    head_size: u64,
) -> bool {
    query_head_count == REVIEWED_HF_QUERY_HEAD_COUNT
        && key_value_head_count == REVIEWED_HF_KEY_VALUE_HEAD_COUNT
        && head_size == ONLINE_HEAD_SIZE
}

fn hugging_face_short_score_prefix_bytes(
    operation: &'static str,
    query_head_count: u64,
    workspace_bytes: u64,
) -> CudaResult<u64> {
    let required = checked_bytes(
        operation,
        &[
            query_head_count,
            HUGGING_FACE_SHORT_DECODE_MAX_TOKENS,
            BF16_BYTES,
        ],
    )?;
    require_capacity(
        operation,
        "hybrid short-score workspace prefix",
        workspace_bytes,
        required,
    )?;
    Ok(required)
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
        assert_eq!(
            online.selection_trace().reason(),
            DecodeAttentionSelectionReason::ReviewedHuggingFaceShortExactHybrid
        );
        assert_eq!(
            online.selection_trace().implementation_id(),
            REVIEWED_HF_HYBRID_IMPLEMENTATION_ID
        );
        assert_eq!(
            online.selection_trace().implementation_version(),
            REVIEWED_HF_HYBRID_IMPLEMENTATION_VERSION
        );
        assert_eq!(
            online.selection_trace().materialized_score_bytes(),
            9 * 32 * 2
        );
        assert_eq!(
            online.selection_trace().short_materialized_token_limit(),
            Some(32)
        );
        assert!(online.capability().materializes_scores());
        assert!(online.capability().supports_partial_state_merge());
        assert_eq!(
            online.capability().short_materialized_token_limit(),
            Some(32)
        );
        assert!(online.workspace_bytes() >= online.selection_trace().materialized_score_bytes());

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
    fn qwen_and_neighbor_shapes_remain_plain_online_and_fixed37_is_disjoint() {
        let context = test_context();
        let availability =
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37(true, true);
        for request in [
            // Qwen2.5-0.5B geometry.
            DecodeAttentionRequest::new(4096, 14, 2, 64, 0.125),
            DecodeAttentionRequest::new(4096, 8, 2, 64, 0.125),
            DecodeAttentionRequest::new(4096, 9, 1, 64, 0.125),
        ] {
            let prepared = PreparedDecodeAttention::select(
                &context,
                request,
                DecodeAttentionPreference::Optimized,
                availability,
            )
            .unwrap();
            assert_eq!(prepared.backend(), DecodeAttentionBackend::ChunkedOnline);
            assert_eq!(
                prepared.capability().implementation_id(),
                ONLINE_IMPLEMENTATION_ID
            );
            assert_eq!(
                prepared.selection_trace().implementation_version(),
                IMPLEMENTATION_VERSION
            );
            assert_eq!(
                prepared.selection_trace().reason(),
                DecodeAttentionSelectionReason::OptimizedCapabilityMatch
            );
            assert_eq!(prepared.selection_trace().materialized_score_bytes(), 0);
            assert_eq!(
                prepared.selection_trace().short_materialized_token_limit(),
                None
            );
            assert!(!prepared.capability().materializes_scores());
        }

        let fixed37 = PreparedDecodeAttention::select_with_reduction_profile(
            &context,
            DecodeAttentionRequest::new(4096, 9, 3, 64, 0.125),
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
        .unwrap();
        assert_eq!(fixed37.backend(), DecodeAttentionBackend::Fixed37TwoPass);
        assert_eq!(fixed37.capability().short_materialized_token_limit(), None);
        assert_eq!(fixed37.selection_trace().materialized_score_bytes(), 0);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_two_pass_selection_is_no_hbm_and_falls_back_only_inside_profile() {
        let context = test_context();
        let availability =
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37(true, true);
        let request = DecodeAttentionRequest::new(8192, 9, 3, 64, 0.125);
        let prepared = PreparedDecodeAttention::select_with_reduction_profile(
            &context,
            request,
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
        .unwrap();
        assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
        assert_eq!(prepared.workspace_bytes(), 0);
        assert_eq!(prepared.selection_trace().materialized_score_bytes(), 0);
        assert_eq!(prepared.selection_trace().partial_state_bytes(), 0);
        assert_eq!(
            prepared.selection_trace().dynamic_shared_memory_bytes(),
            34_544
        );
        assert_eq!(prepared.workspace_dtype(), CudaDType::BF16);
        assert!(!prepared.capability().materializes_scores());
        assert!(!prepared.capability().supports_partial_state_merge());
        assert_eq!(prepared.capability().head_size(), Some(64));
        assert_eq!(
            prepared.capability().maximum_reduction_elements(),
            Some(8192)
        );
        assert_eq!(
            prepared.capability().implementation_id(),
            FIXED37_TWO_PASS_IMPLEMENTATION_ID
        );
        assert_eq!(
            prepared.selection_trace().reason(),
            DecodeAttentionSelectionReason::OptimizedCapabilityMatch
        );

        let explicit = PreparedDecodeAttention::select_with_reduction_profile(
            &context,
            request,
            DecodeAttentionPreference::Reference,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
        .unwrap();
        assert_eq!(
            explicit.backend(),
            DecodeAttentionBackend::Fixed37Materialized
        );

        for (request, reason) in [
            (
                DecodeAttentionRequest::new(8192, 9, 3, 65, 0.125),
                DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback,
            ),
            (
                DecodeAttentionRequest::new(8193, 9, 3, 64, 0.125),
                DecodeAttentionSelectionReason::UnsupportedSequenceLengthFallback,
            ),
        ] {
            let fallback = PreparedDecodeAttention::select_with_reduction_profile(
                &context,
                request,
                DecodeAttentionPreference::Optimized,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap();
            assert_eq!(
                fallback.backend(),
                DecodeAttentionBackend::Fixed37Materialized
            );
            assert_eq!(fallback.selection_trace().reason(), reason);
            assert!(fallback.workspace_bytes() > 0);
        }

        let error = PreparedDecodeAttention::select_with_reduction_profile(
            &context,
            DecodeAttentionRequest::new(8193, 9, 3, 64, 0.125),
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37(false, true),
        )
        .unwrap_err();
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert!(error.message().contains("canonical fallback is forbidden"));

        for (tokens, expected) in [(1, 20), (36, 160), (37, 164), (38, 168), (8192, 34_544)] {
            assert_eq!(
                fixed37_two_pass_shared_bytes("test", tokens, 64).unwrap(),
                expected
            );
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fixed37_paged_two_pass_pins_page_translation_only_capability() {
        let context = test_context();
        let availability =
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37(true, true);
        let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
            &context,
            PagedDecodeAttentionRequest::new(8192, 512, 9, 3, 64, 0.125),
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
        .unwrap();
        assert_eq!(prepared.backend(), DecodeAttentionBackend::Fixed37TwoPass);
        assert_eq!(prepared.workspace_bytes(), 0);
        assert_eq!(prepared.tokens_per_partition(), 0);
        assert_eq!(
            prepared.selection_trace().dynamic_shared_memory_bytes(),
            34_544
        );
        assert_eq!(
            prepared.capability().implementation_id(),
            FIXED37_PAGED_TWO_PASS_IMPLEMENTATION_ID
        );
        assert!(!prepared.capability().materializes_scores());

        let fallback = PreparedPagedDecodeAttention::select_with_reduction_profile(
            &context,
            PagedDecodeAttentionRequest::new(8193, 513, 9, 3, 64, 0.125),
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            availability,
        )
        .unwrap();
        assert_eq!(
            fallback.backend(),
            DecodeAttentionBackend::Fixed37Materialized
        );
        assert_eq!(
            fallback.selection_trace().reason(),
            DecodeAttentionSelectionReason::UnsupportedSequenceLengthFallback
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_lines)]
    fn fixed37_contiguous_selection_pins_profile_axes_and_no_canonical_fallback() {
        let context = test_context();
        let availability =
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37_materialized(true);
        let request = DecodeAttentionRequest::new(75, 9, 3, 74, 0.125);

        for (preference, reason) in [
            (
                DecodeAttentionPreference::Reference,
                DecodeAttentionSelectionReason::ExplicitReference,
            ),
            (
                DecodeAttentionPreference::Optimized,
                DecodeAttentionSelectionReason::OptimizedUnavailableFallback,
            ),
        ] {
            let prepared = PreparedDecodeAttention::select_with_reduction_profile(
                &context,
                request,
                preference,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap();
            assert_eq!(
                prepared.backend(),
                DecodeAttentionBackend::Fixed37Materialized
            );
            assert_eq!(
                prepared.reduction_profile(),
                AttentionReductionProfile::FixedContiguous37BalancedV1
            );
            assert_eq!(
                prepared.selection_trace().reduction_profile(),
                prepared.reduction_profile()
            );
            assert_eq!(prepared.selection_trace().reason(), reason);
            assert_eq!(prepared.workspace_dtype(), CudaDType::BF16);
            assert_eq!(prepared.workspace_bytes(), 9 * 75 * BF16_BYTES);
            assert_eq!(prepared.tokens_per_partition(), 0);
            let capability = prepared.capability();
            assert_eq!(
                capability.implementation_id(),
                FIXED37_REFERENCE_IMPLEMENTATION_ID
            );
            assert_eq!(
                capability.reduction_version(),
                Some(FIXED37_REDUCTION_VERSION)
            );
            assert_eq!(capability.reduction_chunk_elements(), Some(37));
            assert_eq!(capability.maximum_reduction_elements(), Some(151_552));
        }

        let canonical = PreparedDecodeAttention::select(
            &context,
            request,
            DecodeAttentionPreference::Reference,
            availability,
        )
        .unwrap();
        assert_eq!(
            canonical.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(
            canonical.reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );

        let error = PreparedDecodeAttention::select_with_reduction_profile(
            &context,
            request,
            DecodeAttentionPreference::Reference,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap_err();
        assert_eq!(error.kind(), CudaErrorKind::NotSupported);
        assert_eq!(error.stage(), CudaErrorStage::Prepare);
        assert!(error.message().contains("canonical fallback is forbidden"));

        for rejected in [
            DecodeAttentionRequest::new(FIXED37_MAX_REDUCTION_ELEMENTS + 1, 1, 1, 1, 1.0),
            DecodeAttentionRequest::new(1, 1, 1, FIXED37_MAX_REDUCTION_ELEMENTS + 1, 1.0),
        ] {
            let error = PreparedDecodeAttention::select_with_reduction_profile(
                &context,
                rejected,
                DecodeAttentionPreference::Reference,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap_err();
            assert_eq!(error.kind(), CudaErrorKind::NotSupported);
            assert_eq!(error.stage(), CudaErrorStage::Prepare);
        }

        for accepted in [
            DecodeAttentionRequest::new(FIXED37_MAX_REDUCTION_ELEMENTS, 1, 1, 1, 1.0),
            DecodeAttentionRequest::new(1, 1, 1, FIXED37_MAX_REDUCTION_ELEMENTS, 1.0),
        ] {
            PreparedDecodeAttention::select_with_reduction_profile(
                &context,
                accepted,
                DecodeAttentionPreference::Reference,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap();
        }
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

    #[test]
    fn paged_host_table_accepts_boundaries_and_shuffled_physical_ids() {
        for logical_length in [1_u64, 15, 16, 17, 31, 32, 33, 128, 129] {
            let count = usize::try_from(paged_block_count(logical_length).unwrap()).unwrap();
            let ids: Vec<u32> = (0..count)
                .rev()
                .map(|id| u32::try_from(id + 3).unwrap())
                .collect();
            let mut valid = vec![u16::try_from(PAGED_KV_BLOCK_SIZE).unwrap(); count];
            *valid.last_mut().unwrap() =
                u16::try_from(((logical_length - 1) % PAGED_KV_BLOCK_SIZE) + 1).unwrap();
            let table = PagedKvBlockTableHostV1::new(
                &ids,
                &valid,
                logical_length,
                u64::try_from(count + 3).unwrap(),
            )
            .unwrap();
            assert_eq!(table.block_count(), u64::try_from(count).unwrap());
            assert_eq!(table.logical_token_count(), logical_length);

            let physical_count = count + 3;
            let mut pool = vec![u64::MAX; physical_count * 16];
            for logical in 0..usize::try_from(logical_length).unwrap() {
                let block = logical / 16;
                let offset = logical % 16;
                pool[usize::try_from(ids[block]).unwrap() * 16 + offset] =
                    u64::try_from(logical).unwrap();
            }
            let gathered: Vec<u64> = (0..usize::try_from(logical_length).unwrap())
                .map(|logical| {
                    let block = logical / 16;
                    pool[usize::try_from(ids[block]).unwrap() * 16 + logical % 16]
                })
                .collect();
            assert_eq!(
                gathered,
                (0..logical_length).collect::<Vec<_>>(),
                "logical order must not depend on physical allocation order"
            );
        }
    }

    #[test]
    fn paged_host_table_rejects_version_shape_validity_and_invalid_ids() {
        assert_eq!(
            PagedKvBlockTableHostV1::from_versioned_parts(2, &[0], &[1], 1, 1)
                .unwrap_err()
                .kind(),
            CudaErrorKind::NotSupported
        );
        for (ids, valid, logical, physical, expected) in [
            (
                &[][..],
                &[][..],
                1_u64,
                1_u64,
                CudaErrorKind::InvalidArgument,
            ),
            (&[0][..], &[16][..], 1, 1, CudaErrorKind::InvalidArgument),
            (
                &[0, 1][..],
                &[16, 2][..],
                17,
                2,
                CudaErrorKind::InvalidArgument,
            ),
            (&[0, 2][..], &[16, 1][..], 17, 2, CudaErrorKind::OutOfRange),
            (
                &[0, 0][..],
                &[16, 1][..],
                17,
                2,
                CudaErrorKind::InvalidArgument,
            ),
        ] {
            assert_eq!(
                PagedKvBlockTableHostV1::new(ids, valid, logical, physical)
                    .unwrap_err()
                    .kind(),
                expected
            );
        }
    }

    #[test]
    fn paged_host_table_duplicate_scratch_is_linear_reusable_and_checked() {
        let ids = [7, 1, 5];
        let valid = [16, 16, 1];
        let mut scratch = [0xFF; 8];
        let table =
            PagedKvBlockTableHostV1::new_with_duplicate_scratch(&ids, &valid, 33, 8, &mut scratch)
                .expect("shuffled unique table");
        assert_eq!(table.block_ids(), ids);
        assert_eq!(table.valid_tokens(), valid);

        assert_eq!(
            PagedKvBlockTableHostV1::new_with_duplicate_scratch(
                &[7, 1, 7],
                &valid,
                33,
                8,
                &mut scratch,
            )
            .expect_err("duplicate table")
            .kind(),
            CudaErrorKind::InvalidArgument
        );
        PagedKvBlockTableHostV1::new_with_duplicate_scratch(&ids, &valid, 33, 8, &mut scratch)
            .expect("scratch is reset before reuse");
        assert_eq!(
            PagedKvBlockTableHostV1::new_with_duplicate_scratch(
                &ids,
                &valid,
                33,
                8,
                &mut scratch[..7],
            )
            .expect_err("undersized scratch")
            .kind(),
            CudaErrorKind::OutOfRange
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn paged_selection_uses_one_partial_state_per_block_and_facade_types() {
        let context = test_context();
        let request = PagedDecodeAttentionRequest::new(129, 32, 9, 3, 64, 0.125);
        let online = PreparedPagedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            request,
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(online.backend(), DecodeAttentionBackend::ChunkedOnline);
        assert_eq!(online.partial_state_capacity(), 9);
        assert_eq!(online.workspace_bytes(), 9 * 9 * 66 * 4);
        assert_eq!(online.workspace_dtype(), CudaDType::F32);
        assert_eq!(online.tokens_per_partition(), 16);
        assert_eq!(
            online.capability().implementation_id(),
            REVIEWED_HF_PAGED_HYBRID_IMPLEMENTATION_ID
        );
        assert_eq!(
            online.selection_trace().reason(),
            DecodeAttentionSelectionReason::ReviewedHuggingFaceShortExactHybrid
        );
        assert_eq!(online.selection_trace().implementation_version(), "2");
        assert_eq!(
            online.selection_trace().materialized_score_bytes(),
            9 * 32 * 2
        );
        assert_eq!(
            online.selection_trace().short_materialized_token_limit(),
            Some(32)
        );

        let fallback = PreparedPagedDecodeAttention::select_for_compute_capability(
            &context,
            context.compute_capability(),
            PagedDecodeAttentionRequest::new(129, 32, 9, 3, 32, 0.125),
            DecodeAttentionPreference::Optimized,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap();
        assert_eq!(
            fallback.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(fallback.workspace_bytes(), 129 * 9 * 2);
        assert_eq!(
            fallback.selection_trace().reason(),
            DecodeAttentionSelectionReason::UnsupportedHeadSizeFallback
        );
        assert_eq!(fallback.tokens_per_partition(), PAGED_KV_BLOCK_SIZE);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::too_many_lines)]
    fn fixed37_paged_selection_pins_profile_axes_and_page_translation_metadata() {
        let context = test_context();
        let availability =
            DecodeAttentionBackendAvailability::new(true, true).with_fixed37_materialized(true);
        let request = PagedDecodeAttentionRequest::new(75, 8, 9, 3, 74, 0.125);

        let canonical = PreparedPagedDecodeAttention::select(
            &context,
            request,
            DecodeAttentionPreference::Reference,
            availability,
        )
        .unwrap();
        assert_eq!(
            canonical.backend(),
            DecodeAttentionBackend::MaterializedReference
        );
        assert_eq!(canonical.tokens_per_partition(), PAGED_KV_BLOCK_SIZE);
        assert_eq!(
            canonical.reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );

        for (preference, reason) in [
            (
                DecodeAttentionPreference::Reference,
                DecodeAttentionSelectionReason::ExplicitReference,
            ),
            (
                DecodeAttentionPreference::Optimized,
                DecodeAttentionSelectionReason::OptimizedUnavailableFallback,
            ),
        ] {
            let prepared = PreparedPagedDecodeAttention::select_with_reduction_profile(
                &context,
                request,
                preference,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap();
            assert_eq!(
                prepared.backend(),
                DecodeAttentionBackend::Fixed37Materialized
            );
            assert_eq!(
                prepared.reduction_profile(),
                AttentionReductionProfile::FixedContiguous37BalancedV1
            );
            assert_eq!(
                prepared.selection_trace().reduction_profile(),
                prepared.reduction_profile()
            );
            assert_eq!(prepared.selection_trace().reason(), reason);
            assert_eq!(prepared.workspace_dtype(), CudaDType::BF16);
            assert_eq!(prepared.workspace_bytes(), 9 * 75 * BF16_BYTES);
            assert_eq!(prepared.tokens_per_partition(), 0);
            let capability = prepared.capability();
            assert_eq!(
                capability.implementation_id(),
                FIXED37_PAGED_REFERENCE_IMPLEMENTATION_ID
            );
            assert_eq!(
                capability.reduction_version(),
                Some(FIXED37_REDUCTION_VERSION)
            );
            assert_eq!(capability.reduction_chunk_elements(), Some(37));
            assert_eq!(capability.maximum_reduction_elements(), Some(151_552));
        }

        let unavailable = PreparedPagedDecodeAttention::select_with_reduction_profile(
            &context,
            request,
            DecodeAttentionPreference::Optimized,
            AttentionReductionProfile::FixedContiguous37BalancedV1,
            DecodeAttentionBackendAvailability::new(true, true),
        )
        .unwrap_err();
        assert_eq!(unavailable.kind(), CudaErrorKind::NotSupported);
        assert!(
            unavailable
                .message()
                .contains("canonical fallback is forbidden")
        );

        for rejected in [
            PagedDecodeAttentionRequest::new(FIXED37_MAX_REDUCTION_ELEMENTS + 1, 1, 1, 1, 1, 1.0),
            PagedDecodeAttentionRequest::new(1, 1, 1, 1, FIXED37_MAX_REDUCTION_ELEMENTS + 1, 1.0),
        ] {
            assert_eq!(
                PreparedPagedDecodeAttention::select_with_reduction_profile(
                    &context,
                    rejected,
                    DecodeAttentionPreference::Reference,
                    AttentionReductionProfile::FixedContiguous37BalancedV1,
                    availability,
                )
                .unwrap_err()
                .kind(),
                CudaErrorKind::NotSupported
            );
        }

        for accepted in [
            PagedDecodeAttentionRequest::new(FIXED37_MAX_REDUCTION_ELEMENTS, 1, 1, 1, 1, 1.0),
            PagedDecodeAttentionRequest::new(1, 1, 1, 1, FIXED37_MAX_REDUCTION_ELEMENTS, 1.0),
        ] {
            PreparedPagedDecodeAttention::select_with_reduction_profile(
                &context,
                accepted,
                DecodeAttentionPreference::Reference,
                AttentionReductionProfile::FixedContiguous37BalancedV1,
                availability,
            )
            .unwrap();
        }
    }
}
