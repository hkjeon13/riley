//! Owning contiguous-KV single-request Llama decode path.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::error;
use std::fmt;
use std::mem;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaError,
    CudaGemmConfig, CudaPinnedHostBuffer, CudaStream, DecodeAttentionBackend,
    DecodeAttentionBackendAvailability, DecodeAttentionCapability, DecodeAttentionParams,
    DecodeAttentionPreference, DecodeAttentionRequest, DecodeAttentionSelectionTrace,
    EmbeddingError, EmbeddingParams, GatedMultiplyParams, KvCacheAppendParams, PAGED_KV_BLOCK_SIZE,
    PAGED_KV_BLOCK_TABLE_VERSION, PagedDecodeAttentionParams, PagedDecodeAttentionRequest,
    PagedKvBlockTableHostV1, PagedKvBlockTableV1, PagedKvCacheAppendParams,
    PreparedDecodeAttention, PreparedPagedDecodeAttention, ResidualAddParams, RmsNormParams,
    RopeParams, SiluParams, embedding, gated_multiply, kv_cache_append, paged_kv_cache_append,
    residual_add, rope, silu,
};
use rustinfer_model::LoadedModel;

use crate::paged_kv::{
    BLOCK_TABLE_V1_VERSION, BlockId, KV_BLOCK_SIZE, KvBlockPool, KvBlockPoolStats, KvLayout,
    PagedKvError, SequenceReservation, SequenceState,
};

use super::forward::{
    LlamaForwardError, PreparedLlamaAllocationReport, PreparedLlamaForward,
    PreparedLlamaForwardConfig, PreparedLlamaGemm, execute_gemm, execute_profile_rms_norm,
    execute_projection_bias, poison_for_cuda_error, poison_for_forward_error, span, span_mut,
    weight_span,
};
use super::{ExecutionSite, LlamaOp, LlamaReductionProfile};

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const U32_BYTES: u64 = 4;
const CONTIGUOUS_CACHE_ALLOCATION_COUNT: u64 = 2;
const PAGED_CACHE_ALLOCATION_COUNT: u64 = 4;
const PAGED_CACHE_PINNED_ALLOCATION_COUNT: u64 = 1;
const ROPE_ALLOCATION_COUNT: u64 = 2;
const ATTENTION_ALLOCATION_COUNT: u64 = 1;
const _: () = assert!(KV_BLOCK_SIZE as u64 == PAGED_KV_BLOCK_SIZE);
const _: () = assert!(BLOCK_TABLE_V1_VERSION as u32 == PAGED_KV_BLOCK_TABLE_VERSION);

/// Result type for PR09 single-request preparation and execution.
pub type LlamaDecodeResult<T> = Result<T, LlamaDecodeError>;

/// Request-local resource named in allocation and cleanup diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaDecodeResource {
    KeyCache,
    ValueCache,
    CacheLayerOffsets,
    BlockTableHostEncoding,
    BlockTableDuplicateScratch,
    BlockTableDeviceIds,
    BlockTableDeviceValidTokens,
    BlockTablePinnedStaging,
    RopeCos,
    RopeSin,
    AttentionWorkspace,
    GemmWorkspace,
    HiddenGemm,
    KeyValueGemm,
    IntermediateGemm,
    DownGemm,
    LmHeadGemm,
}

impl LlamaDecodeResource {
    const fn name(self) -> &'static str {
        match self {
            Self::KeyCache => "key_cache",
            Self::ValueCache => "value_cache",
            Self::CacheLayerOffsets => "cache_layer_offsets",
            Self::BlockTableHostEncoding => "block_table_host_encoding",
            Self::BlockTableDuplicateScratch => "block_table_duplicate_scratch",
            Self::BlockTableDeviceIds => "block_table_device_ids",
            Self::BlockTableDeviceValidTokens => "block_table_device_valid_tokens",
            Self::BlockTablePinnedStaging => "block_table_pinned_staging",
            Self::RopeCos => "decode_rope_cos",
            Self::RopeSin => "decode_rope_sin",
            Self::AttentionWorkspace => "decode_attention_workspace",
            Self::GemmWorkspace => "decode_gemm_workspace",
            Self::HiddenGemm => "decode_hidden_gemm",
            Self::KeyValueGemm => "decode_key_value_gemm",
            Self::IntermediateGemm => "decode_intermediate_gemm",
            Self::DownGemm => "decode_down_gemm",
            Self::LmHeadGemm => "decode_lm_head_gemm",
        }
    }
}

impl fmt::Display for LlamaDecodeResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Stable single-request lifecycle phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaDecodePhase {
    /// No prompt cache is published.
    Empty,
    /// The fixed prompt was cached and its last logits are current.
    Prefilled,
    /// At least one one-token decode call has committed.
    Decoding,
}

impl LlamaDecodePhase {
    const fn name(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Prefilled => "prefilled",
            Self::Decoding => "decoding",
        }
    }
}

/// Structured PR09 preparation, state, execution, or cleanup failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaDecodeError {
    Forward(LlamaForwardError),
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    InvalidPromptLength {
        expected: usize,
        actual: usize,
    },
    InvalidState {
        operation: &'static str,
        actual: LlamaDecodePhase,
    },
    CapacityExceeded {
        logical_length: usize,
        maximum_length: usize,
    },
    TokenOutOfRange {
        token_id: u32,
        vocabulary_size: usize,
    },
    OutputNotReady,
    Poisoned,
    InvalidDownloadLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    HostAllocation {
        resource: LlamaDecodeResource,
        requested_bytes: u64,
    },
    PagedKv {
        operation: &'static str,
        source: PagedKvError,
    },
    ArithmeticOverflow {
        resource: LlamaDecodeResource,
    },
    Cuda {
        site: ExecutionSite,
        source: CudaError,
    },
    Embedding {
        site: ExecutionSite,
        source: EmbeddingError,
    },
    Cleanup {
        resource: LlamaDecodeResource,
        source: CudaError,
    },
}

impl LlamaDecodeError {
    fn cuda(site: ExecutionSite, source: CudaError) -> Self {
        Self::Cuda { site, source }
    }

    pub(super) fn into_forward_cache_error(self, site: ExecutionSite) -> LlamaForwardError {
        match self {
            Self::Cuda { source, .. } => LlamaForwardError::cuda(site, source),
            _ => LlamaForwardError::InvalidConfiguration {
                field: "decode_kv_cache",
                reason: "validated cache metadata no longer matches the prefill plan",
            },
        }
    }
}

impl fmt::Display for LlamaDecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forward(source) => source.fmt(formatter),
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama decode configuration {field}: {reason}"
                )
            }
            Self::InvalidPromptLength { expected, actual } => write!(
                formatter,
                "decode owner expects {expected} prompt token IDs, received {actual}"
            ),
            Self::InvalidState { operation, actual } => write!(
                formatter,
                "cannot {operation} while the decode request is {}",
                actual.name()
            ),
            Self::CapacityExceeded {
                logical_length,
                maximum_length,
            } => write!(
                formatter,
                "decode length {logical_length} reached fixed capacity {maximum_length}"
            ),
            Self::TokenOutOfRange {
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "decode token ID {token_id} is outside vocabulary 0..{vocabulary_size}"
            ),
            Self::OutputNotReady => formatter
                .write_str("decode logits are unavailable before successful prefill or decode"),
            Self::Poisoned => {
                formatter.write_str("the Llama decode owner was poisoned by native CUDA execution")
            }
            Self::InvalidDownloadLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "decode logits destination has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::HostAllocation {
                resource,
                requested_bytes,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} host bytes for {resource}"
            ),
            Self::PagedKv { operation, source } => {
                write!(formatter, "paged-KV {operation}: {source}")
            }
            Self::ArithmeticOverflow { resource } => {
                write!(formatter, "decode byte arithmetic overflow for {resource}")
            }
            Self::Cuda { site, source } => write!(formatter, "{site}: {source}"),
            Self::Embedding { site, source } => write!(formatter, "{site}: {source}"),
            Self::Cleanup { resource, source } => {
                write!(formatter, "could not close {resource}: {source}")
            }
        }
    }
}

impl error::Error for LlamaDecodeError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Forward(source) => Some(source),
            Self::Cuda { source, .. } | Self::Cleanup { source, .. } => Some(source),
            Self::Embedding { source, .. } => Some(source),
            Self::PagedKv { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<LlamaForwardError> for LlamaDecodeError {
    fn from(source: LlamaForwardError) -> Self {
        Self::Forward(source)
    }
}

/// Checked head-major layout for each separately allocated K or V cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaKvCacheLayout {
    layer_count: usize,
    key_value_head_count: usize,
    maximum_sequence_length: usize,
    head_dimension: usize,
    head_stride_bytes: u64,
    layer_stride_bytes: u64,
    bytes_per_kind: u64,
    total_bytes: u64,
}

impl LlamaKvCacheLayout {
    /// Computes `[layer, kv_head, max_sequence, head_dimension]` BF16 strides.
    ///
    /// # Errors
    ///
    /// Returns when a dimension is zero, cannot be represented by the native
    /// fixed-width contract, or makes byte-stride arithmetic overflow.
    pub fn checked(
        layer_count: usize,
        key_value_head_count: usize,
        maximum_sequence_length: usize,
        head_dimension: usize,
    ) -> LlamaDecodeResult<Self> {
        if layer_count == 0 {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "layer_count",
                reason: "must be non-zero",
            });
        }
        if key_value_head_count == 0 {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "key_value_head_count",
                reason: "must be non-zero",
            });
        }
        if maximum_sequence_length == 0 {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "maximum_sequence_length",
                reason: "must be non-zero",
            });
        }
        if head_dimension == 0 {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "head_dimension",
                reason: "must be non-zero",
            });
        }
        let heads = decode_u64(key_value_head_count, LlamaDecodeResource::KeyCache)?;
        let maximum = decode_u64(maximum_sequence_length, LlamaDecodeResource::KeyCache)?;
        let dimension = decode_u64(head_dimension, LlamaDecodeResource::KeyCache)?;
        let layers = decode_u64(layer_count, LlamaDecodeResource::KeyCache)?;
        let head_stride_bytes = maximum
            .checked_mul(dimension)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            })?;
        let layer_stride_bytes =
            heads
                .checked_mul(head_stride_bytes)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
        let bytes_per_kind =
            layers
                .checked_mul(layer_stride_bytes)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
        let total_bytes =
            bytes_per_kind
                .checked_mul(2)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::ValueCache,
                })?;
        Ok(Self {
            layer_count,
            key_value_head_count,
            maximum_sequence_length,
            head_dimension,
            head_stride_bytes,
            layer_stride_bytes,
            bytes_per_kind,
            total_bytes,
        })
    }

    #[must_use]
    pub const fn layer_count(self) -> usize {
        self.layer_count
    }
    #[must_use]
    pub const fn key_value_head_count(self) -> usize {
        self.key_value_head_count
    }
    #[must_use]
    pub const fn maximum_sequence_length(self) -> usize {
        self.maximum_sequence_length
    }
    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.head_dimension
    }
    #[must_use]
    pub const fn head_stride_bytes(self) -> u64 {
        self.head_stride_bytes
    }
    #[must_use]
    pub const fn layer_stride_bytes(self) -> u64 {
        self.layer_stride_bytes
    }
    #[must_use]
    pub const fn bytes_per_kind(self) -> u64 {
        self.bytes_per_kind
    }
    #[must_use]
    pub const fn total_bytes(self) -> u64 {
        self.total_bytes
    }

    /// Byte offset of one layer in either the K or V allocation.
    #[must_use]
    pub fn layer_byte_offset(self, layer_index: usize) -> Option<u64> {
        if layer_index >= self.layer_count {
            return None;
        }
        u64::try_from(layer_index)
            .ok()
            .and_then(|layer| layer.checked_mul(self.layer_stride_bytes))
    }
}

/// Physical cache layout fixed during cold decode preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaKvCacheStorageLayout {
    /// PR09 request-local head-major contiguous K/V allocations.
    Contiguous(LlamaKvCacheLayout),
    /// PR10 layer-major fixed-block K/V pool.
    Paged(KvLayout),
}

impl LlamaKvCacheStorageLayout {
    #[must_use]
    pub const fn is_paged(self) -> bool {
        matches!(self, Self::Paged(_))
    }

    #[must_use]
    pub const fn total_kv_bytes(self) -> u64 {
        match self {
            Self::Contiguous(layout) => layout.total_bytes(),
            Self::Paged(layout) => layout.total_bytes(),
        }
    }
}

/// Cold-selected KV address-translation policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaKvCachePolicy {
    /// Preserve the PR09 contiguous cache as an exact reference/rollback path.
    Contiguous,
    /// Use a version-1 block table and a fixed physical pool.
    ///
    /// `physical_block_count=None` provisions exactly enough 16-token blocks
    /// for the configured maximum sequence. A smaller explicit pool is useful
    /// for deterministic OOM/rollback validation.
    Paged { physical_block_count: Option<usize> },
}

impl LlamaKvCachePolicy {
    #[must_use]
    pub const fn paged() -> Self {
        Self::Paged {
            physical_block_count: None,
        }
    }

    #[must_use]
    pub const fn paged_with_blocks(physical_block_count: usize) -> Self {
        Self::Paged {
            physical_block_count: Some(physical_block_count),
        }
    }
}

/// Cold-path settings for one fixed-prompt decode owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaDecodeConfig {
    forward: PreparedLlamaForwardConfig,
    decode_attention_preference: DecodeAttentionPreference,
    kv_cache_policy: LlamaKvCachePolicy,
}

impl PreparedLlamaDecodeConfig {
    #[must_use]
    pub const fn new(forward: PreparedLlamaForwardConfig) -> Self {
        Self {
            forward,
            decode_attention_preference: DecodeAttentionPreference::Optimized,
            kv_cache_policy: LlamaKvCachePolicy::paged(),
        }
    }

    #[must_use]
    pub const fn with_optimized_decode_attention(mut self) -> Self {
        self.decode_attention_preference = DecodeAttentionPreference::Optimized;
        self
    }

    #[must_use]
    pub const fn with_reference_decode_attention(mut self) -> Self {
        self.decode_attention_preference = DecodeAttentionPreference::Reference;
        self
    }

    /// Selects one reduction contract for every supported Llama primitive.
    #[must_use]
    pub const fn with_reduction_profile(mut self, profile: LlamaReductionProfile) -> Self {
        self.forward = self.forward.with_reduction_profile(profile);
        self
    }

    /// Restores the established canonical reduction contract.
    #[must_use]
    pub const fn with_canonical_reductions(self) -> Self {
        self.with_reduction_profile(LlamaReductionProfile::CanonicalV1)
    }

    /// Selects contiguous-37 balanced reductions without canonical fallback.
    #[must_use]
    pub const fn with_fixed37_reductions(self) -> Self {
        self.with_reduction_profile(LlamaReductionProfile::FixedContiguous37BalancedV1)
    }

    /// Selects the PR10 exact paged cache with one block per capacity slice.
    #[must_use]
    pub const fn with_paged_kv_cache(mut self) -> Self {
        self.kv_cache_policy = LlamaKvCachePolicy::paged();
        self
    }

    /// Selects a paged pool with an explicit physical-block budget.
    #[must_use]
    pub const fn with_paged_kv_cache_blocks(mut self, physical_block_count: usize) -> Self {
        self.kv_cache_policy = LlamaKvCachePolicy::paged_with_blocks(physical_block_count);
        self
    }

    /// Selects the PR09 contiguous reference cache.
    #[must_use]
    pub const fn with_contiguous_kv_cache(mut self) -> Self {
        self.kv_cache_policy = LlamaKvCachePolicy::Contiguous;
        self
    }

    #[must_use]
    pub const fn forward(self) -> PreparedLlamaForwardConfig {
        self.forward
    }

    #[must_use]
    pub const fn decode_attention_preference(self) -> DecodeAttentionPreference {
        self.decode_attention_preference
    }

    #[must_use]
    pub const fn reduction_profile(self) -> LlamaReductionProfile {
        self.forward.reduction_profile()
    }

    #[must_use]
    pub const fn kv_cache_policy(self) -> LlamaKvCachePolicy {
        self.kv_cache_policy
    }
}

impl Default for PreparedLlamaDecodeConfig {
    fn default() -> Self {
        Self::new(PreparedLlamaForwardConfig::default())
    }
}

/// Owned device, pinned-host, and paged-metadata payload after preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaDecodeAllocationReport {
    forward: PreparedLlamaAllocationReport,
    kv_cache_bytes: u64,
    block_table_device_bytes: u64,
    block_table_host_bytes: u64,
    cache_unused_capacity_bytes: u64,
    rope_table_bytes: u64,
    attention_workspace_bytes: u64,
    decode_gemm_workspace_bytes: u64,
    additional_device_bytes: u64,
    total_device_bytes: u64,
    device_allocation_count: u64,
    pinned_host_bytes: u64,
    pinned_host_allocation_count: u64,
}

impl PreparedLlamaDecodeAllocationReport {
    #[must_use]
    pub const fn forward(self) -> PreparedLlamaAllocationReport {
        self.forward
    }
    #[must_use]
    pub const fn kv_cache_bytes(self) -> u64 {
        self.kv_cache_bytes
    }
    #[must_use]
    pub const fn block_table_device_bytes(self) -> u64 {
        self.block_table_device_bytes
    }
    #[must_use]
    pub const fn block_table_host_bytes(self) -> u64 {
        self.block_table_host_bytes
    }
    /// Paged-pool VRAM beyond the configured maximum logical length.
    ///
    /// This includes whole-block rounding and any explicit overprovisioning;
    /// it is static, unlike the owner's current tail-block fragmentation.
    #[must_use]
    pub const fn cache_unused_capacity_bytes(self) -> u64 {
        self.cache_unused_capacity_bytes
    }
    #[must_use]
    pub const fn rope_table_bytes(self) -> u64 {
        self.rope_table_bytes
    }
    #[must_use]
    pub const fn attention_workspace_bytes(self) -> u64 {
        self.attention_workspace_bytes
    }
    #[must_use]
    pub const fn decode_gemm_workspace_bytes(self) -> u64 {
        self.decode_gemm_workspace_bytes
    }
    #[must_use]
    pub const fn additional_device_bytes(self) -> u64 {
        self.additional_device_bytes
    }
    #[must_use]
    pub const fn total_device_bytes(self) -> u64 {
        self.total_device_bytes
    }
    #[must_use]
    pub const fn device_allocation_count(self) -> u64 {
        self.device_allocation_count
    }
    #[must_use]
    pub const fn pinned_host_bytes(self) -> u64 {
        self.pinned_host_bytes
    }
    #[must_use]
    pub const fn pinned_host_allocation_count(self) -> u64 {
        self.pinned_host_allocation_count
    }
}

pub(super) struct ContiguousKvCache {
    key: CudaDeviceBuffer,
    value: CudaDeviceBuffer,
    layout: LlamaKvCacheLayout,
    layer_offsets: Box<[u64]>,
}

impl ContiguousKvCache {
    fn prepare(context: &CudaContext, layout: LlamaKvCacheLayout) -> LlamaDecodeResult<Self> {
        let mut offsets = Vec::new();
        let offset_bytes =
            decode_u64(layout.layer_count(), LlamaDecodeResource::CacheLayerOffsets)?
                .checked_mul(u64::try_from(mem::size_of::<u64>()).unwrap_or(u64::MAX))
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::CacheLayerOffsets,
                })?;
        offsets
            .try_reserve_exact(layout.layer_count())
            .map_err(|_| LlamaDecodeError::HostAllocation {
                resource: LlamaDecodeResource::CacheLayerOffsets,
                requested_bytes: offset_bytes,
            })?;
        for layer in 0..layout.layer_count() {
            offsets.push(layout.layer_byte_offset(layer).ok_or(
                LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::CacheLayerOffsets,
                },
            )?);
        }
        let site = ExecutionSite::layer(0, LlamaOp::KvCacheWrite);
        let key = context
            .allocate_device_buffer(layout.bytes_per_kind())
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value = context
            .allocate_device_buffer(layout.bytes_per_kind())
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        Ok(Self {
            key,
            value,
            layout,
            layer_offsets: offsets.into_boxed_slice(),
        })
    }

    pub(super) fn append_layer(
        &mut self,
        layer_index: usize,
        key_source: &CudaDeviceBuffer,
        value_source: &CudaDeviceBuffer,
        source_token_count: u64,
        destination_token_start: u64,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let site = ExecutionSite::layer(layer_index, LlamaOp::KvCacheWrite);
        let offset =
            *self
                .layer_offsets
                .get(layer_index)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::CacheLayerOffsets,
                })?;
        let key_value_heads = decode_u64(
            self.layout.key_value_head_count(),
            LlamaDecodeResource::KeyCache,
        )?;
        let head_size = decode_u64(self.layout.head_dimension(), LlamaDecodeResource::KeyCache)?;
        let source_bytes = source_token_count
            .checked_mul(key_value_heads)
            .and_then(|elements| elements.checked_mul(head_size))
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            })?;
        let key_cache = CudaBufferSpanMut::new(
            &mut self.key,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value_cache = CudaBufferSpanMut::new(
            &mut self.value,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let mut params = KvCacheAppendParams {
            key_source: CudaBufferSpan::new(key_source, CudaDType::BF16, 0, source_bytes)
                .map_err(|source| LlamaDecodeError::cuda(site, source))?,
            value_source: CudaBufferSpan::new(value_source, CudaDType::BF16, 0, source_bytes)
                .map_err(|source| LlamaDecodeError::cuda(site, source))?,
            key_cache,
            value_cache,
            source_token_count,
            destination_token_start,
            maximum_token_count: decode_u64(
                self.layout.maximum_sequence_length(),
                LlamaDecodeResource::KeyCache,
            )?,
            key_value_head_count: key_value_heads,
            head_size,
        };
        kv_cache_append(&mut params, stream).map_err(|source| LlamaDecodeError::cuda(site, source))
    }

    fn layer_spans(
        &self,
        layer_index: usize,
    ) -> LlamaDecodeResult<(CudaBufferSpan<'_>, CudaBufferSpan<'_>)> {
        let offset =
            *self
                .layer_offsets
                .get(layer_index)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::CacheLayerOffsets,
                })?;
        let site = ExecutionSite::layer(layer_index, LlamaOp::DecodeAttention);
        let key = CudaBufferSpan::new(
            &self.key,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value = CudaBufferSpan::new(
            &self.value,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        Ok((key, value))
    }
}

struct PagedKvCache {
    key: CudaDeviceBuffer,
    value: CudaDeviceBuffer,
    device_block_ids: CudaDeviceBuffer,
    device_valid_tokens: CudaDeviceBuffer,
    table_staging: CudaPinnedHostBuffer,
    encoded_block_ids: Box<[u8]>,
    encoded_valid_tokens: Box<[u8]>,
    duplicate_scratch: Box<[u8]>,
    layout: KvLayout,
    pool: KvBlockPool,
    sequence: SequenceState,
}

impl PagedKvCache {
    fn prepare(
        context: &CudaContext,
        layout: KvLayout,
        maximum_sequence_length: usize,
    ) -> LlamaDecodeResult<Self> {
        let mut pool = KvBlockPool::new(layout).map_err(|source| LlamaDecodeError::PagedKv {
            operation: "prepare pool",
            source,
        })?;
        let sequence = pool
            .create_sequence(maximum_sequence_length)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "prepare sequence table",
                source,
            })?;
        let table_capacity = layout.physical_block_count();
        let block_id_bytes = checked_host_bytes(
            table_capacity,
            mem::size_of::<u32>(),
            LlamaDecodeResource::BlockTableHostEncoding,
        )?;
        let valid_token_bytes = checked_host_bytes(
            table_capacity,
            mem::size_of::<u16>(),
            LlamaDecodeResource::BlockTableHostEncoding,
        )?;
        let encoded_block_ids =
            decode_boxed_zeroed(block_id_bytes, LlamaDecodeResource::BlockTableHostEncoding)?;
        let encoded_valid_tokens = decode_boxed_zeroed(
            valid_token_bytes,
            LlamaDecodeResource::BlockTableHostEncoding,
        )?;
        let duplicate_scratch = decode_boxed_zeroed(
            table_capacity,
            LlamaDecodeResource::BlockTableDuplicateScratch,
        )?;
        let site = ExecutionSite::layer(0, LlamaOp::KvCacheWrite);
        let key = context
            .allocate_device_buffer(layout.bytes_per_kind())
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value = context
            .allocate_device_buffer(layout.bytes_per_kind())
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let device_block_ids = context
            .allocate_device_buffer(decode_u64(
                block_id_bytes,
                LlamaDecodeResource::BlockTableDeviceIds,
            )?)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let device_valid_tokens = context
            .allocate_device_buffer(decode_u64(
                valid_token_bytes,
                LlamaDecodeResource::BlockTableDeviceValidTokens,
            )?)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let table_staging = context
            .allocate_pinned_host_buffer(decode_u64(
                block_id_bytes.max(valid_token_bytes),
                LlamaDecodeResource::BlockTablePinnedStaging,
            )?)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        Ok(Self {
            key,
            value,
            device_block_ids,
            device_valid_tokens,
            table_staging,
            encoded_block_ids,
            encoded_valid_tokens,
            duplicate_scratch,
            layout,
            pool,
            sequence,
        })
    }

    fn reserve_to(&mut self, target: usize) -> LlamaDecodeResult<SequenceReservation> {
        self.sequence
            .reserve_to(&mut self.pool, target)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "reserve",
                source,
            })
    }

    fn upload_reserved_table(
        &mut self,
        reservation: &SequenceReservation,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let table = self
            .sequence
            .reserved_block_table(reservation)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "read reserved table",
                source,
            })?;
        let committed_length = usize::try_from(self.sequence.logical_length()).map_err(|_| {
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::BlockTableHostEncoding,
            }
        })?;
        let committed_blocks = committed_length.div_ceil(KV_BLOCK_SIZE);
        let target_blocks = table.block_count();
        let first_valid_block =
            if reservation.target_logical_length() > self.sequence.logical_length() {
                committed_length / KV_BLOCK_SIZE
            } else {
                target_blocks
            };

        for index in committed_blocks..target_blocks {
            let start = index.checked_mul(mem::size_of::<u32>()).ok_or(
                LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::BlockTableHostEncoding,
                },
            )?;
            self.encoded_block_ids[start..start + mem::size_of::<u32>()]
                .copy_from_slice(&table.physical_block_ids()[index].to_ne_bytes());
        }
        for index in first_valid_block..target_blocks {
            let start = index.checked_mul(mem::size_of::<u16>()).ok_or(
                LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::BlockTableHostEncoding,
                },
            )?;
            self.encoded_valid_tokens[start..start + mem::size_of::<u16>()]
                .copy_from_slice(&table.valid_tokens()[index].to_ne_bytes());
        }

        let block_id_start = committed_blocks.checked_mul(mem::size_of::<u32>()).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::BlockTableDeviceIds,
            },
        )?;
        let block_id_end = target_blocks.checked_mul(mem::size_of::<u32>()).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::BlockTableDeviceIds,
            },
        )?;
        if block_id_start != block_id_end {
            self.device_block_ids
                .upload_from_slice(
                    decode_u64(block_id_start, LlamaDecodeResource::BlockTableDeviceIds)?,
                    &self.encoded_block_ids[block_id_start..block_id_end],
                    &mut self.table_staging,
                    stream,
                )
                .map_err(|source| {
                    LlamaDecodeError::cuda(ExecutionSite::layer(0, LlamaOp::KvCacheWrite), source)
                })?;
        }
        let valid_start = first_valid_block.checked_mul(mem::size_of::<u16>()).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::BlockTableDeviceValidTokens,
            },
        )?;
        let valid_end = target_blocks.checked_mul(mem::size_of::<u16>()).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::BlockTableDeviceValidTokens,
            },
        )?;
        if valid_start != valid_end {
            self.device_valid_tokens
                .upload_from_slice(
                    decode_u64(
                        valid_start,
                        LlamaDecodeResource::BlockTableDeviceValidTokens,
                    )?,
                    &self.encoded_valid_tokens[valid_start..valid_end],
                    &mut self.table_staging,
                    stream,
                )
                .map_err(|source| {
                    LlamaDecodeError::cuda(ExecutionSite::layer(0, LlamaOp::KvCacheWrite), source)
                })?;
        }
        Ok(())
    }

    fn begin_execution<'a>(
        &'a mut self,
        reservation: &SequenceReservation,
    ) -> LlamaDecodeResult<PagedKvExecution<'a>> {
        let Self {
            key,
            value,
            device_block_ids,
            device_valid_tokens,
            duplicate_scratch,
            layout,
            sequence,
            ..
        } = self;
        let table = sequence
            .reserved_block_table(reservation)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "bind reserved table",
                source,
            })?;
        let host = PagedKvBlockTableHostV1::new_with_duplicate_scratch(
            table.physical_block_ids(),
            table.valid_tokens(),
            u64::from(table.logical_length()),
            decode_u64(layout.physical_block_count(), LlamaDecodeResource::KeyCache)?,
            duplicate_scratch,
        )
        .map_err(|source| {
            LlamaDecodeError::cuda(ExecutionSite::layer(0, LlamaOp::KvCacheWrite), source)
        })?;
        Ok(PagedKvExecution {
            key,
            value,
            device_block_ids,
            device_valid_tokens,
            layout: *layout,
            host,
        })
    }

    fn commit(&mut self, reservation: SequenceReservation) -> LlamaDecodeResult<()> {
        self.sequence
            .commit(&mut self.pool, reservation)
            .map(|_| ())
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "commit",
                source,
            })
    }

    fn poison(&mut self, reservation: SequenceReservation) -> LlamaDecodeResult<()> {
        self.sequence
            .poison(&mut self.pool, reservation)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "poison reservation",
                source,
            })
    }

    fn reset(&mut self) -> LlamaDecodeResult<()> {
        self.sequence
            .reset(&mut self.pool)
            .map_err(|source| LlamaDecodeError::PagedKv {
                operation: "reset",
                source,
            })
    }

    fn stats(&self) -> KvBlockPoolStats {
        self.pool.stats()
    }

    fn block_id(&self, logical_block_index: usize) -> Option<BlockId> {
        self.sequence.block_id(logical_block_index)
    }

    fn block_table_device_bytes(&self) -> u64 {
        self.device_block_ids
            .byte_len()
            .saturating_add(self.device_valid_tokens.byte_len())
    }

    fn block_table_host_bytes(&self) -> u64 {
        self.sequence
            .host_state_capacity_bytes()
            .saturating_add(u64::try_from(self.encoded_block_ids.len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(self.encoded_valid_tokens.len()).unwrap_or(u64::MAX))
            .saturating_add(u64::try_from(self.duplicate_scratch.len()).unwrap_or(u64::MAX))
            .saturating_add(self.pool.stats().host_pool_metadata_bytes())
    }
}

pub(super) struct PagedKvExecution<'a> {
    key: &'a mut CudaDeviceBuffer,
    value: &'a mut CudaDeviceBuffer,
    device_block_ids: &'a CudaDeviceBuffer,
    device_valid_tokens: &'a CudaDeviceBuffer,
    layout: KvLayout,
    host: PagedKvBlockTableHostV1<'a>,
}

impl PagedKvExecution<'_> {
    fn native_table(&self, site: ExecutionSite) -> LlamaDecodeResult<PagedKvBlockTableV1<'_>> {
        let block_ids = CudaBufferSpan::new(
            self.device_block_ids,
            CudaDType::U32,
            0,
            self.device_block_ids.byte_len(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let valid_tokens = CudaBufferSpan::new(
            self.device_valid_tokens,
            CudaDType::U16,
            0,
            self.device_valid_tokens.byte_len(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        PagedKvBlockTableV1::new(self.host, block_ids, valid_tokens)
            .map_err(|source| LlamaDecodeError::cuda(site, source))
    }

    fn append_layer(
        &mut self,
        layer_index: usize,
        key_source: &CudaDeviceBuffer,
        value_source: &CudaDeviceBuffer,
        source_token_count: u64,
        destination_token_start: u64,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let site = ExecutionSite::layer(layer_index, LlamaOp::KvCacheWrite);
        let offset = self.layout.layer_byte_offset(layer_index).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            },
        )?;
        let key_value_heads = decode_u64(
            self.layout.key_value_head_count(),
            LlamaDecodeResource::KeyCache,
        )?;
        let head_size = decode_u64(self.layout.head_dimension(), LlamaDecodeResource::KeyCache)?;
        let source_bytes = source_token_count
            .checked_mul(key_value_heads)
            .and_then(|elements| elements.checked_mul(head_size))
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            })?;
        let block_ids = CudaBufferSpan::new(
            self.device_block_ids,
            CudaDType::U32,
            0,
            self.device_block_ids.byte_len(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let valid_tokens = CudaBufferSpan::new(
            self.device_valid_tokens,
            CudaDType::U16,
            0,
            self.device_valid_tokens.byte_len(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let table = PagedKvBlockTableV1::new(self.host, block_ids, valid_tokens)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let key_pool = CudaBufferSpanMut::new(
            self.key,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value_pool = CudaBufferSpanMut::new(
            self.value,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let mut params = PagedKvCacheAppendParams {
            key_source: CudaBufferSpan::new(key_source, CudaDType::BF16, 0, source_bytes)
                .map_err(|source| LlamaDecodeError::cuda(site, source))?,
            value_source: CudaBufferSpan::new(value_source, CudaDType::BF16, 0, source_bytes)
                .map_err(|source| LlamaDecodeError::cuda(site, source))?,
            key_pool,
            value_pool,
            block_table: table,
            source_token_count,
            destination_token_start,
            key_value_head_count: key_value_heads,
            head_size,
        };
        paged_kv_cache_append(&mut params, stream)
            .map_err(|source| LlamaDecodeError::cuda(site, source))
    }

    fn layer_spans(
        &self,
        layer_index: usize,
    ) -> LlamaDecodeResult<(CudaBufferSpan<'_>, CudaBufferSpan<'_>)> {
        let offset = self.layout.layer_byte_offset(layer_index).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            },
        )?;
        let site = ExecutionSite::layer(layer_index, LlamaOp::DecodeAttention);
        let key = CudaBufferSpan::new(
            self.key,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        let value = CudaBufferSpan::new(
            self.value,
            CudaDType::BF16,
            offset,
            self.layout.layer_stride_bytes(),
        )
        .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        Ok((key, value))
    }
}

#[allow(clippy::large_enum_variant)]
enum KvCacheStorage {
    Contiguous(ContiguousKvCache),
    Paged(PagedKvCache),
}

enum KvCacheReservation {
    Contiguous,
    Paged(SequenceReservation),
}

impl KvCacheStorage {
    fn is_poisoned(&self) -> bool {
        match self {
            Self::Contiguous(_) => false,
            Self::Paged(cache) => cache.sequence.is_poisoned(),
        }
    }

    fn reserve_to(&mut self, target: usize) -> LlamaDecodeResult<KvCacheReservation> {
        match self {
            Self::Contiguous(_) => Ok(KvCacheReservation::Contiguous),
            Self::Paged(cache) => cache.reserve_to(target).map(KvCacheReservation::Paged),
        }
    }

    fn upload_reserved_table(
        &mut self,
        reservation: &KvCacheReservation,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        match (self, reservation) {
            (Self::Contiguous(_), KvCacheReservation::Contiguous) => Ok(()),
            (Self::Paged(cache), KvCacheReservation::Paged(reservation)) => {
                cache.upload_reserved_table(reservation, stream)
            }
            _ => Err(LlamaDecodeError::InvalidConfiguration {
                field: "kv_cache_reservation",
                reason: "reservation policy differs from prepared cache",
            }),
        }
    }

    fn begin_execution<'a>(
        &'a mut self,
        reservation: &KvCacheReservation,
    ) -> LlamaDecodeResult<PrefillKvCacheSink<'a>> {
        match (self, reservation) {
            (Self::Contiguous(cache), KvCacheReservation::Contiguous) => {
                Ok(PrefillKvCacheSink::Contiguous(cache))
            }
            (Self::Paged(cache), KvCacheReservation::Paged(reservation)) => cache
                .begin_execution(reservation)
                .map(PrefillKvCacheSink::Paged),
            _ => Err(LlamaDecodeError::InvalidConfiguration {
                field: "kv_cache_reservation",
                reason: "reservation policy differs from prepared cache",
            }),
        }
    }

    fn commit(&mut self, reservation: KvCacheReservation) -> LlamaDecodeResult<()> {
        match (self, reservation) {
            (Self::Contiguous(_), KvCacheReservation::Contiguous) => Ok(()),
            (Self::Paged(cache), KvCacheReservation::Paged(reservation)) => {
                cache.commit(reservation)
            }
            _ => Err(LlamaDecodeError::InvalidConfiguration {
                field: "kv_cache_reservation",
                reason: "reservation policy differs from prepared cache",
            }),
        }
    }

    fn poison(&mut self, reservation: KvCacheReservation) -> LlamaDecodeResult<()> {
        match (self, reservation) {
            (Self::Contiguous(_), KvCacheReservation::Contiguous) => Ok(()),
            (Self::Paged(cache), KvCacheReservation::Paged(reservation)) => {
                cache.poison(reservation)
            }
            _ => Err(LlamaDecodeError::InvalidConfiguration {
                field: "kv_cache_reservation",
                reason: "reservation policy differs from prepared cache",
            }),
        }
    }

    fn reset(&mut self) -> LlamaDecodeResult<()> {
        match self {
            Self::Contiguous(_) => Ok(()),
            Self::Paged(cache) => cache.reset(),
        }
    }
}

/// One cold-selected destination for prefill K/V publication.
///
/// Keeping the sink as a closed enum avoids heap allocation and confines the
/// cache-layout branch to the existing per-layer publication point. PR10 adds
/// the paged variant without changing the public cache-free forward path.
pub(super) enum PrefillKvCacheSink<'a> {
    Contiguous(&'a mut ContiguousKvCache),
    Paged(PagedKvExecution<'a>),
}

impl PrefillKvCacheSink<'_> {
    pub(super) fn append_layer(
        &mut self,
        layer_index: usize,
        key_source: &CudaDeviceBuffer,
        value_source: &CudaDeviceBuffer,
        source_token_count: u64,
        destination_token_start: u64,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        match self {
            Self::Contiguous(cache) => cache.append_layer(
                layer_index,
                key_source,
                value_source,
                source_token_count,
                destination_token_start,
                stream,
            ),
            Self::Paged(cache) => cache.append_layer(
                layer_index,
                key_source,
                value_source,
                source_token_count,
                destination_token_start,
                stream,
            ),
        }
    }
}

struct DecodeGemmPlans {
    hidden: PreparedLlamaGemm,
    key_value: PreparedLlamaGemm,
    intermediate: PreparedLlamaGemm,
    down: PreparedLlamaGemm,
    lm_head: PreparedLlamaGemm,
}

/// Prepared attention plan matching the selected cache address space.
#[derive(Clone, Debug)]
pub enum PreparedLlamaDecodeAttention {
    Contiguous(PreparedDecodeAttention),
    Paged(PreparedPagedDecodeAttention),
}

impl PreparedLlamaDecodeAttention {
    #[must_use]
    pub const fn backend(&self) -> DecodeAttentionBackend {
        match self {
            Self::Contiguous(attention) => attention.backend(),
            Self::Paged(attention) => attention.backend(),
        }
    }

    #[must_use]
    pub const fn capability(&self) -> DecodeAttentionCapability {
        match self {
            Self::Contiguous(attention) => attention.capability(),
            Self::Paged(attention) => attention.capability(),
        }
    }

    #[must_use]
    pub const fn selection_trace(&self) -> DecodeAttentionSelectionTrace {
        match self {
            Self::Contiguous(attention) => attention.selection_trace(),
            Self::Paged(attention) => attention.selection_trace(),
        }
    }

    #[must_use]
    pub const fn workspace_bytes(&self) -> u64 {
        match self {
            Self::Contiguous(attention) => attention.workspace_bytes(),
            Self::Paged(attention) => attention.workspace_bytes(),
        }
    }

    #[must_use]
    pub const fn workspace_dtype(&self) -> CudaDType {
        match self {
            Self::Contiguous(attention) => attention.workspace_dtype(),
            Self::Paged(attention) => attention.workspace_dtype(),
        }
    }

    #[must_use]
    pub const fn partial_state_capacity(&self) -> u64 {
        match self {
            Self::Contiguous(attention) => attention.partial_state_capacity(),
            Self::Paged(attention) => attention.partial_state_capacity(),
        }
    }

    #[must_use]
    pub const fn tokens_per_partition(&self) -> u64 {
        match self {
            Self::Contiguous(attention) => attention.tokens_per_partition(),
            Self::Paged(attention) => attention.tokens_per_partition(),
        }
    }
}

impl DecodeGemmPlans {
    fn any_poisoned(&self) -> bool {
        self.hidden.is_poisoned()
            || self.key_value.is_poisoned()
            || self.intermediate.is_poisoned()
            || self.down.is_poisoned()
            || self.lm_head.is_poisoned()
    }

    fn maximum_workspace_bytes(&self) -> u64 {
        [
            self.hidden.workspace_bytes(),
            self.key_value.workspace_bytes(),
            self.intermediate.workspace_bytes(),
            self.down.workspace_bytes(),
            self.lm_head.workspace_bytes(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

struct DecodeBuffers {
    cache: KvCacheStorage,
    rope_cos: CudaDeviceBuffer,
    rope_sin: CudaDeviceBuffer,
    attention_workspace: CudaDeviceBuffer,
    gemm_workspace: Option<CudaDeviceBuffer>,
    rope_table_bytes_per_kind: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LatestOutput {
    Prefill,
    Decode,
}

/// Owning fixed-prompt, fixed-capacity single-request decode executor.
pub struct PreparedLlamaDecode {
    forward: PreparedLlamaForward,
    gemms: DecodeGemmPlans,
    attention: PreparedLlamaDecodeAttention,
    buffers: DecodeBuffers,
    cache_layout: LlamaKvCacheStorageLayout,
    allocation_report: PreparedLlamaDecodeAllocationReport,
    prompt_length: usize,
    maximum_sequence_length: usize,
    logical_length: usize,
    phase: LlamaDecodePhase,
    latest_output: Option<LatestOutput>,
}

impl fmt::Debug for PreparedLlamaDecode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaDecode")
            .field("prompt_length", &self.prompt_length)
            .field("logical_length", &self.logical_length)
            .field("maximum_length", &self.maximum_sequence_length)
            .field("cache_layout", &self.cache_layout)
            .field("phase", &self.phase)
            .field("reduction_profile", &self.reduction_profile())
            .field("attention_backend", &self.attention.backend())
            .field("allocation_report", &self.allocation_report)
            .field("poisoned", &self.is_poisoned())
            .finish_non_exhaustive()
    }
}

impl PreparedLlamaDecode {
    /// Prepares one immutable prompt shape and one fixed maximum cache length.
    ///
    /// The prompt length remains fixed because PR08 prefill GEMMs and attention
    /// are prepared for one exact `S`. Reset permits another prompt with that
    /// same length while reusing every allocation.
    ///
    /// # Errors
    ///
    /// Returns for invalid prompt/capacity bounds, unsupported model geometry,
    /// allocation or upload failure, CUDA-plan selection failure, or checked
    /// byte-arithmetic overflow.
    #[allow(clippy::too_many_lines, clippy::cast_precision_loss)]
    pub fn prepare(
        model: &LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        prompt_length: usize,
        maximum_sequence_length: usize,
        config: PreparedLlamaDecodeConfig,
    ) -> LlamaDecodeResult<Self> {
        if prompt_length == 0 {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "prompt_length",
                reason: "must be non-zero",
            });
        }
        if maximum_sequence_length < prompt_length {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "maximum_sequence_length",
                reason: "must be at least the fixed prompt length",
            });
        }
        if maximum_sequence_length > model.spec().max_sequence_length() {
            return Err(LlamaDecodeError::InvalidConfiguration {
                field: "maximum_sequence_length",
                reason: "exceeds the model sequence limit",
            });
        }

        let mut forward =
            PreparedLlamaForward::prepare(model, context, stream, prompt_length, config.forward())?;
        let dimensions = forward.plan.dimensions();
        let head_size = decode_u64(
            dimensions.head_dimension(),
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let maximum = decode_u64(
            maximum_sequence_length,
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let query_heads = decode_u64(
            dimensions.query_heads(),
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let key_value_heads = decode_u64(
            dimensions.key_value_heads(),
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let scale = 1.0 / (head_size as f32).sqrt();
        let (cache, cache_layout, attention) = match config.kv_cache_policy() {
            LlamaKvCachePolicy::Contiguous => {
                let layout = LlamaKvCacheLayout::checked(
                    forward.plan.layers().len(),
                    dimensions.key_value_heads(),
                    maximum_sequence_length,
                    dimensions.head_dimension(),
                )?;
                let request = DecodeAttentionRequest::new(
                    maximum,
                    query_heads,
                    key_value_heads,
                    head_size,
                    scale,
                );
                let attention = PreparedDecodeAttention::select_with_reduction_profile(
                    context,
                    request,
                    config.decode_attention_preference(),
                    config.forward().reduction_profile().attention_profile(),
                    DecodeAttentionBackendAvailability::linked(),
                )
                .map_err(|source| {
                    LlamaDecodeError::cuda(
                        ExecutionSite::layer(0, LlamaOp::DecodeAttention),
                        source,
                    )
                })?;
                (
                    KvCacheStorage::Contiguous(ContiguousKvCache::prepare(context, layout)?),
                    LlamaKvCacheStorageLayout::Contiguous(layout),
                    PreparedLlamaDecodeAttention::Contiguous(attention),
                )
            }
            LlamaKvCachePolicy::Paged {
                physical_block_count,
            } => {
                let required_blocks = maximum_sequence_length.div_ceil(KV_BLOCK_SIZE);
                let physical_blocks = physical_block_count.unwrap_or(required_blocks);
                let layout = KvLayout::checked(
                    forward.plan.layers().len(),
                    physical_blocks,
                    dimensions.key_value_heads(),
                    dimensions.head_dimension(),
                )
                .map_err(|source| LlamaDecodeError::PagedKv {
                    operation: "validate pool layout",
                    source,
                })?;
                let request = PagedDecodeAttentionRequest::new(
                    maximum,
                    decode_u64(physical_blocks, LlamaDecodeResource::AttentionWorkspace)?,
                    query_heads,
                    key_value_heads,
                    head_size,
                    scale,
                );
                let attention = PreparedPagedDecodeAttention::select_with_reduction_profile(
                    context,
                    request,
                    config.decode_attention_preference(),
                    config.forward().reduction_profile().attention_profile(),
                    DecodeAttentionBackendAvailability::linked(),
                )
                .map_err(|source| {
                    LlamaDecodeError::cuda(
                        ExecutionSite::layer(0, LlamaOp::DecodeAttention),
                        source,
                    )
                })?;
                (
                    KvCacheStorage::Paged(PagedKvCache::prepare(
                        context,
                        layout,
                        maximum_sequence_length,
                    )?),
                    LlamaKvCacheStorageLayout::Paged(layout),
                    PreparedLlamaDecodeAttention::Paged(attention),
                )
            }
        };
        let gemms = prepare_decode_gemms(
            context,
            &forward,
            config.forward().gemm_workspace_cap_bytes(),
        )?;
        let decode_gemm_workspace_bytes = gemms.maximum_workspace_bytes();
        let rope_table_bytes_per_kind =
            rope_table_bytes(maximum_sequence_length, dimensions.head_dimension())?;
        let rope_site = ExecutionSite::layer(0, LlamaOp::QueryRope);
        let mut rope_cos = context
            .allocate_device_buffer(rope_table_bytes_per_kind)
            .map_err(|source| LlamaDecodeError::cuda(rope_site, source))?;
        let mut rope_sin = context
            .allocate_device_buffer(rope_table_bytes_per_kind)
            .map_err(|source| LlamaDecodeError::cuda(rope_site, source))?;
        let attention_workspace = context
            .allocate_device_buffer(attention.workspace_bytes())
            .map_err(|source| {
                LlamaDecodeError::cuda(ExecutionSite::layer(0, LlamaOp::DecodeAttention), source)
            })?;
        let gemm_workspace = if decode_gemm_workspace_bytes == 0 {
            None
        } else {
            Some(
                context
                    .allocate_device_buffer(decode_gemm_workspace_bytes)
                    .map_err(|source| {
                        LlamaDecodeError::cuda(
                            ExecutionSite::layer(0, LlamaOp::QueryProjection),
                            source,
                        )
                    })?,
            )
        };

        let (cos, sin) = build_decode_rope_tables(
            maximum_sequence_length,
            dimensions.head_dimension(),
            forward.plan.rope_theta(),
        )?;
        rope_cos
            .upload_from_slice(0, &cos, &mut forward.io_staging, stream)
            .map_err(|source| LlamaDecodeError::cuda(rope_site, source))?;
        rope_sin
            .upload_from_slice(0, &sin, &mut forward.io_staging, stream)
            .map_err(|source| LlamaDecodeError::cuda(rope_site, source))?;

        let allocation_report = build_decode_allocation_report(
            forward.allocation_report(),
            decode_cache_allocation(&cache, maximum_sequence_length)?,
            rope_table_bytes_per_kind,
            attention.workspace_bytes(),
            decode_gemm_workspace_bytes,
        )?;
        Ok(Self {
            forward,
            gemms,
            attention,
            buffers: DecodeBuffers {
                cache,
                rope_cos,
                rope_sin,
                attention_workspace,
                gemm_workspace,
                rope_table_bytes_per_kind,
            },
            cache_layout,
            allocation_report,
            prompt_length,
            maximum_sequence_length,
            logical_length: 0,
            phase: LlamaDecodePhase::Empty,
            latest_output: None,
        })
    }

    #[must_use]
    pub const fn prompt_length(&self) -> usize {
        self.prompt_length
    }

    #[must_use]
    pub const fn logical_length(&self) -> usize {
        self.logical_length
    }

    #[must_use]
    pub const fn maximum_length(&self) -> usize {
        self.maximum_sequence_length
    }

    #[must_use]
    pub const fn phase(&self) -> LlamaDecodePhase {
        self.phase
    }

    #[must_use]
    pub const fn cache_layout(&self) -> LlamaKvCacheStorageLayout {
        self.cache_layout
    }

    #[must_use]
    pub const fn allocation_report(&self) -> PreparedLlamaDecodeAllocationReport {
        self.allocation_report
    }

    #[must_use]
    pub const fn prepared_attention(&self) -> &PreparedLlamaDecodeAttention {
        &self.attention
    }

    /// Reduction contract selected for every supported decode primitive.
    #[must_use]
    pub const fn reduction_profile(&self) -> LlamaReductionProfile {
        self.forward.reduction_profile()
    }

    #[must_use]
    pub fn paged_pool_stats(&self) -> Option<KvBlockPoolStats> {
        match &self.buffers.cache {
            KvCacheStorage::Contiguous(_) => None,
            KvCacheStorage::Paged(cache) => Some(cache.stats()),
        }
    }

    #[must_use]
    pub fn paged_block_id(&self, logical_block_index: usize) -> Option<BlockId> {
        match &self.buffers.cache {
            KvCacheStorage::Contiguous(_) => None,
            KvCacheStorage::Paged(cache) => cache.block_id(logical_block_index),
        }
    }

    /// Current bytes in the committed tail block that hold no logical token.
    #[must_use]
    pub fn paged_internal_fragmentation_bytes(&self) -> Option<u64> {
        match &self.buffers.cache {
            KvCacheStorage::Contiguous(_) => None,
            KvCacheStorage::Paged(cache) => {
                let unused_tokens =
                    u64::try_from(cache.sequence.internal_fragmentation_tokens()).ok()?;
                unused_tokens
                    .checked_mul(cache.layout.bytes_per_physical_block() / PAGED_KV_BLOCK_SIZE)
            }
        }
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.forward.poisoned || self.gemms.any_poisoned() || self.buffers.cache.is_poisoned()
    }

    /// Uploads and executes the owner's exact fixed-length prompt into cache.
    ///
    /// # Errors
    ///
    /// Returns when the owner is poisoned or not empty, the prompt length is
    /// different from the prepared length, or upload/native execution fails.
    pub fn prefill(&mut self, prompt: &[u32], stream: &mut CudaStream) -> LlamaDecodeResult<()> {
        if self.is_poisoned() {
            return Err(LlamaDecodeError::Poisoned);
        }
        validate_prefill_request(self.phase, self.prompt_length, prompt.len())?;
        self.forward
            .validate_token_ids(prompt)
            .map_err(LlamaDecodeError::Forward)?;
        let reservation = self.buffers.cache.reserve_to(self.prompt_length)?;
        if let Err(error) = self
            .buffers
            .cache
            .upload_reserved_table(&reservation, stream)
        {
            return Err(self.abort_cache_reservation(reservation, error));
        }
        if let Err(source) = self.forward.upload_tokens(prompt, stream) {
            return Err(
                self.abort_cache_reservation(reservation, LlamaDecodeError::Forward(source))
            );
        }
        self.latest_output = None;
        let execution = self
            .buffers
            .cache
            .begin_execution(&reservation)
            .and_then(|cache| {
                self.forward
                    .execute_prefill_into_cache(cache, stream)
                    .map_err(LlamaDecodeError::Forward)
            });
        if let Err(error) = execution {
            return Err(self.abort_cache_reservation(reservation, error));
        }
        if let Err(error) = self.buffers.cache.commit(reservation) {
            self.forward.poisoned = true;
            return Err(error);
        }
        self.logical_length = self.prompt_length;
        self.phase = LlamaDecodePhase::Prefilled;
        self.latest_output = Some(LatestOutput::Prefill);
        Ok(())
    }

    /// Logically clears a completed request without reallocating or zeroing VRAM.
    ///
    /// Stale cache and partial-state bytes are never visible because the next
    /// prefill overwrites every published prompt position and decode attention
    /// reads only the committed logical length.
    ///
    /// # Errors
    ///
    /// Returns if native execution previously poisoned the owner.
    pub fn reset(&mut self) -> LlamaDecodeResult<()> {
        if self.is_poisoned() {
            return Err(LlamaDecodeError::Poisoned);
        }
        self.buffers.cache.reset()?;
        self.logical_length = 0;
        self.phase = LlamaDecodePhase::Empty;
        self.latest_output = None;
        self.forward.tokens_ready = false;
        self.forward.output_ready = false;
        Ok(())
    }

    // HOT_DECODE_BEGIN
    /// Appends one token and produces logits for the following token.
    ///
    /// Capacity, phase, and vocabulary validation complete before the first
    /// device mutation. The logical length commits only after every layer and
    /// the final LM head succeed.
    ///
    /// # Errors
    ///
    /// Returns for a poisoned or unprefilled owner, exhausted fixed capacity,
    /// an out-of-range token ID, or any upload/native execution failure.
    pub fn decode(&mut self, token_id: u32, stream: &mut CudaStream) -> LlamaDecodeResult<()> {
        if self.is_poisoned() {
            return Err(LlamaDecodeError::Poisoned);
        }
        validate_decode_request(self.phase, self.logical_length, self.maximum_length())?;
        let vocabulary_size = self.forward.plan.dimensions().vocabulary_size();
        if usize::try_from(token_id).map_or(true, |token| token >= vocabulary_size) {
            return Err(LlamaDecodeError::TokenOutOfRange {
                token_id,
                vocabulary_size,
            });
        }

        let target_length =
            self.logical_length
                .checked_add(1)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
        let reservation = self.buffers.cache.reserve_to(target_length)?;
        if let Err(error) = self
            .buffers
            .cache
            .upload_reserved_table(&reservation, stream)
        {
            return Err(self.abort_cache_reservation(reservation, error));
        }

        if let Err(error) = self.upload_decode_token(token_id, stream) {
            return Err(self.abort_cache_reservation(reservation, error));
        }
        self.latest_output = None;
        self.forward.output_ready = false;
        let position = self.logical_length;
        if let Err(error) = self.execute_decode_inner(position, &reservation, stream) {
            let error = self.abort_cache_reservation(reservation, error);
            self.forward.poisoned |= self.gemms.any_poisoned();
            return Err(error);
        }
        if let Err(error) = self.buffers.cache.commit(reservation) {
            self.forward.poisoned = true;
            return Err(error);
        }
        self.logical_length = target_length;
        self.phase = LlamaDecodePhase::Decoding;
        self.latest_output = Some(LatestOutput::Decode);
        self.forward.output_ready = true;
        Ok(())
    }

    fn upload_decode_token(
        &mut self,
        token_id: u32,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let bytes = token_id.to_ne_bytes();
        self.forward
            .buffers
            .token_ids
            .upload_from_slice(0, &bytes, &mut self.forward.io_staging, stream)
            .map_err(|source| {
                LlamaDecodeError::cuda(ExecutionSite::global(LlamaOp::Embedding), source)
            })
    }

    #[allow(
        clippy::too_many_lines,
        clippy::cast_precision_loss,
        clippy::similar_names
    )]
    fn execute_decode_inner(
        &mut self,
        position: usize,
        reservation: &KvCacheReservation,
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let forward = &mut self.forward;
        let rms_norm_profile = forward.rms_norm_profile();
        let plan = &forward.plan;
        let weights = &forward.weights;
        let buffers = &mut forward.buffers;
        let gemms = &mut self.gemms;
        let attention = &self.attention;
        let decode_buffers = &mut self.buffers;
        let mut cache = decode_buffers.cache.begin_execution(reservation)?;
        let dimensions = plan.dimensions();
        let hidden = decode_u64(dimensions.hidden_size(), LlamaDecodeResource::GemmWorkspace)?;
        let intermediate = decode_u64(
            dimensions.intermediate_size(),
            LlamaDecodeResource::GemmWorkspace,
        )?;
        let vocabulary = decode_u64(
            dimensions.vocabulary_size(),
            LlamaDecodeResource::GemmWorkspace,
        )?;
        let query_heads = decode_u64(
            dimensions.query_heads(),
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let key_value_heads =
            decode_u64(dimensions.key_value_heads(), LlamaDecodeResource::KeyCache)?;
        let head_size = decode_u64(dimensions.head_dimension(), LlamaDecodeResource::KeyCache)?;
        let key_value_width =
            key_value_heads
                .checked_mul(head_size)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
        let position = decode_u64(position, LlamaDecodeResource::KeyCache)?;
        let logical_token_count =
            position
                .checked_add(1)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
        let hidden_bytes =
            hidden
                .checked_mul(BF16_BYTES)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::GemmWorkspace,
                })?;
        let key_value_bytes = key_value_width.checked_mul(BF16_BYTES).ok_or(
            LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            },
        )?;
        let intermediate_bytes =
            intermediate
                .checked_mul(BF16_BYTES)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::GemmWorkspace,
                })?;
        let logits_bytes =
            vocabulary
                .checked_mul(BF16_BYTES)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::GemmWorkspace,
                })?;

        let embedding_site = ExecutionSite::global(LlamaOp::Embedding);
        let embedding_weight = weight_span(weights, plan.embedding_weight(), embedding_site)?;
        {
            let mut params = EmbeddingParams {
                table: embedding_weight,
                token_ids: span(
                    &buffers.token_ids,
                    CudaDType::U32,
                    U32_BYTES,
                    embedding_site,
                )?,
                output: span_mut(
                    &mut buffers.hidden_current,
                    CudaDType::BF16,
                    hidden_bytes,
                    embedding_site,
                )?,
                error_scratch: span_mut(
                    &mut buffers.embedding_error_scratch,
                    CudaDType::U8,
                    plan.workspace_spec().embedding_error_scratch_bytes(),
                    embedding_site,
                )?,
                token_count: 1,
                vocabulary_size: vocabulary,
                hidden_size: hidden,
            };
            embedding(&mut params, stream).map_err(|source| LlamaDecodeError::Embedding {
                site: embedding_site,
                source,
            })?;
        }

        for layer in plan.layers() {
            let layer_index = layer.index();
            let input_norm_site = ExecutionSite::layer(layer_index, LlamaOp::InputNorm);
            let input_norm_weight =
                weight_span(weights, layer.input_norm_weight(), input_norm_site)?;
            {
                let mut params = RmsNormParams {
                    input: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        hidden_bytes,
                        input_norm_site,
                    )?,
                    weight: input_norm_weight,
                    output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        hidden_bytes,
                        input_norm_site,
                    )?,
                    row_count: 1,
                    hidden_size: hidden,
                    epsilon: layer.input_norm_epsilon(),
                };
                execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(input_norm_site, source))?;
            }

            let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
            let query_weight = weight_span(weights, layer.query_weight(), query_site)?;
            execute_gemm(
                &mut gemms.hidden,
                &buffers.hidden_norm,
                query_weight,
                &mut buffers.hidden_projection,
                &mut decode_buffers.gemm_workspace,
                stream,
                query_site,
            )?;
            execute_projection_bias(
                weights,
                layer.query_bias(),
                &mut buffers.hidden_projection,
                1,
                hidden,
                stream,
                query_site,
            )?;
            let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
            let key_weight = weight_span(weights, layer.key_weight(), key_site)?;
            execute_gemm(
                &mut gemms.key_value,
                &buffers.hidden_norm,
                key_weight,
                &mut buffers.key_raw,
                &mut decode_buffers.gemm_workspace,
                stream,
                key_site,
            )?;
            execute_projection_bias(
                weights,
                layer.key_bias(),
                &mut buffers.key_raw,
                1,
                key_value_width,
                stream,
                key_site,
            )?;
            let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
            let value_weight = weight_span(weights, layer.value_weight(), value_site)?;
            execute_gemm(
                &mut gemms.key_value,
                &buffers.hidden_norm,
                value_weight,
                &mut buffers.value_raw,
                &mut decode_buffers.gemm_workspace,
                stream,
                value_site,
            )?;
            execute_projection_bias(
                weights,
                layer.value_bias(),
                &mut buffers.value_raw,
                1,
                key_value_width,
                stream,
                value_site,
            )?;

            let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
            {
                let mut params = RopeParams {
                    input: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        hidden_bytes,
                        query_rope_site,
                    )?,
                    cos: span(
                        &decode_buffers.rope_cos,
                        CudaDType::F32,
                        decode_buffers.rope_table_bytes_per_kind,
                        query_rope_site,
                    )?,
                    sin: span(
                        &decode_buffers.rope_sin,
                        CudaDType::F32,
                        decode_buffers.rope_table_bytes_per_kind,
                        query_rope_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        hidden_bytes,
                        query_rope_site,
                    )?,
                    token_count: 1,
                    head_count: query_heads,
                    head_size,
                    rotary_dimension: head_size,
                    table_position_count: decode_u64(
                        self.maximum_sequence_length,
                        LlamaDecodeResource::RopeCos,
                    )?,
                    position_offset: position,
                };
                rope(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(query_rope_site, source))?;
            }
            let key_rope_site = ExecutionSite::layer(layer_index, LlamaOp::KeyRope);
            {
                let mut params = RopeParams {
                    input: span(
                        &buffers.key_raw,
                        CudaDType::BF16,
                        key_value_bytes,
                        key_rope_site,
                    )?,
                    cos: span(
                        &decode_buffers.rope_cos,
                        CudaDType::F32,
                        decode_buffers.rope_table_bytes_per_kind,
                        key_rope_site,
                    )?,
                    sin: span(
                        &decode_buffers.rope_sin,
                        CudaDType::F32,
                        decode_buffers.rope_table_bytes_per_kind,
                        key_rope_site,
                    )?,
                    output: span_mut(
                        &mut buffers.key_rotary,
                        CudaDType::BF16,
                        key_value_bytes,
                        key_rope_site,
                    )?,
                    token_count: 1,
                    head_count: key_value_heads,
                    head_size,
                    rotary_dimension: head_size,
                    table_position_count: decode_u64(
                        self.maximum_sequence_length,
                        LlamaDecodeResource::RopeSin,
                    )?,
                    position_offset: position,
                };
                rope(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(key_rope_site, source))?;
            }

            cache.append_layer(
                layer_index,
                &buffers.key_rotary,
                &buffers.value_raw,
                1,
                position,
                stream,
            )?;

            let attention_site = ExecutionSite::layer(layer_index, LlamaOp::DecodeAttention);
            {
                match (attention, &mut cache) {
                    (
                        PreparedLlamaDecodeAttention::Contiguous(attention),
                        PrefillKvCacheSink::Contiguous(cache),
                    ) => {
                        let (key_cache, value_cache) = cache.layer_spans(layer_index)?;
                        let mut params = DecodeAttentionParams {
                            query: span(
                                &buffers.hidden_rotary,
                                CudaDType::BF16,
                                hidden_bytes,
                                attention_site,
                            )?,
                            key_cache,
                            value_cache,
                            output: span_mut(
                                &mut buffers.hidden_context,
                                CudaDType::BF16,
                                hidden_bytes,
                                attention_site,
                            )?,
                            workspace: CudaBufferSpanMut::new(
                                &mut decode_buffers.attention_workspace,
                                attention.workspace_dtype(),
                                0,
                                attention.workspace_bytes(),
                            )
                            .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?,
                        };
                        attention
                            .execute(logical_token_count, &mut params, stream)
                            .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?;
                    }
                    (
                        PreparedLlamaDecodeAttention::Paged(attention),
                        PrefillKvCacheSink::Paged(cache),
                    ) => {
                        let (key_pool, value_pool) = cache.layer_spans(layer_index)?;
                        let block_table = cache.native_table(attention_site)?;
                        let mut params = PagedDecodeAttentionParams {
                            query: span(
                                &buffers.hidden_rotary,
                                CudaDType::BF16,
                                hidden_bytes,
                                attention_site,
                            )?,
                            key_pool,
                            value_pool,
                            output: span_mut(
                                &mut buffers.hidden_context,
                                CudaDType::BF16,
                                hidden_bytes,
                                attention_site,
                            )?,
                            workspace: CudaBufferSpanMut::new(
                                &mut decode_buffers.attention_workspace,
                                attention.workspace_dtype(),
                                0,
                                attention.workspace_bytes(),
                            )
                            .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?,
                            block_table,
                        };
                        attention
                            .execute(&mut params, stream)
                            .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?;
                    }
                    _ => {
                        return Err(LlamaDecodeError::InvalidConfiguration {
                            field: "decode_attention_cache_layout",
                            reason: "prepared attention and cache layouts differ",
                        });
                    }
                }
            }

            let output_site = ExecutionSite::layer(layer_index, LlamaOp::OutputProjection);
            let output_weight = weight_span(weights, layer.output_weight(), output_site)?;
            execute_gemm(
                &mut gemms.hidden,
                &buffers.hidden_context,
                output_weight,
                &mut buffers.hidden_projection,
                &mut decode_buffers.gemm_workspace,
                stream,
                output_site,
            )?;
            execute_projection_bias(
                weights,
                layer.output_bias(),
                &mut buffers.hidden_projection,
                1,
                hidden,
                stream,
                output_site,
            )?;
            let attention_residual_site =
                ExecutionSite::layer(layer_index, LlamaOp::AttentionResidual);
            {
                let mut params = ResidualAddParams {
                    left: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        hidden_bytes,
                        attention_residual_site,
                    )?,
                    right: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        hidden_bytes,
                        attention_residual_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        hidden_bytes,
                        attention_residual_site,
                    )?,
                    element_count: hidden,
                };
                residual_add(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(attention_residual_site, source))?;
            }

            let post_norm_site = ExecutionSite::layer(layer_index, LlamaOp::PostAttentionNorm);
            let post_norm_weight =
                weight_span(weights, layer.post_attention_norm_weight(), post_norm_site)?;
            {
                let mut params = RmsNormParams {
                    input: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        hidden_bytes,
                        post_norm_site,
                    )?,
                    weight: post_norm_weight,
                    output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        hidden_bytes,
                        post_norm_site,
                    )?,
                    row_count: 1,
                    hidden_size: hidden,
                    epsilon: layer.post_attention_norm_epsilon(),
                };
                execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(post_norm_site, source))?;
            }

            let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);
            let gate_weight = weight_span(weights, layer.gate_weight(), gate_site)?;
            execute_gemm(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                gate_weight,
                &mut buffers.gate_raw,
                &mut decode_buffers.gemm_workspace,
                stream,
                gate_site,
            )?;
            let up_site = ExecutionSite::layer(layer_index, LlamaOp::UpProjection);
            let up_weight = weight_span(weights, layer.up_weight(), up_site)?;
            execute_gemm(
                &mut gemms.intermediate,
                &buffers.hidden_norm,
                up_weight,
                &mut buffers.up_raw,
                &mut decode_buffers.gemm_workspace,
                stream,
                up_site,
            )?;
            let silu_site = ExecutionSite::layer(layer_index, LlamaOp::Silu);
            {
                let mut params = SiluParams {
                    input: span(
                        &buffers.gate_raw,
                        CudaDType::BF16,
                        intermediate_bytes,
                        silu_site,
                    )?,
                    output: span_mut(
                        &mut buffers.gate_activated,
                        CudaDType::BF16,
                        intermediate_bytes,
                        silu_site,
                    )?,
                    element_count: intermediate,
                };
                silu(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(silu_site, source))?;
            }
            let gated_site = ExecutionSite::layer(layer_index, LlamaOp::GatedMultiply);
            {
                let mut params = GatedMultiplyParams {
                    activated_gate: span(
                        &buffers.gate_activated,
                        CudaDType::BF16,
                        intermediate_bytes,
                        gated_site,
                    )?,
                    up: span(
                        &buffers.up_raw,
                        CudaDType::BF16,
                        intermediate_bytes,
                        gated_site,
                    )?,
                    output: span_mut(
                        &mut buffers.gated_product,
                        CudaDType::BF16,
                        intermediate_bytes,
                        gated_site,
                    )?,
                    element_count: intermediate,
                };
                gated_multiply(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(gated_site, source))?;
            }

            let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
            let down_weight = weight_span(weights, layer.down_weight(), down_site)?;
            execute_gemm(
                &mut gemms.down,
                &buffers.gated_product,
                down_weight,
                &mut buffers.hidden_current,
                &mut decode_buffers.gemm_workspace,
                stream,
                down_site,
            )?;
            let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);
            {
                let mut params = ResidualAddParams {
                    left: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        hidden_bytes,
                        mlp_residual_site,
                    )?,
                    right: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        hidden_bytes,
                        mlp_residual_site,
                    )?,
                    output: span_mut(
                        &mut buffers.hidden_projection,
                        CudaDType::BF16,
                        hidden_bytes,
                        mlp_residual_site,
                    )?,
                    element_count: hidden,
                };
                residual_add(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(mlp_residual_site, source))?;
            }
            mem::swap(&mut buffers.hidden_current, &mut buffers.hidden_projection);
        }

        let final_norm_site = ExecutionSite::global(LlamaOp::FinalNorm);
        let final_norm_weight = weight_span(weights, plan.final_norm_weight(), final_norm_site)?;
        {
            let mut params = RmsNormParams {
                input: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    hidden_bytes,
                    final_norm_site,
                )?,
                weight: final_norm_weight,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    hidden_bytes,
                    final_norm_site,
                )?,
                row_count: 1,
                hidden_size: hidden,
                epsilon: plan.final_norm_epsilon(),
            };
            execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                .map_err(|source| LlamaDecodeError::cuda(final_norm_site, source))?;
        }
        let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
        let lm_head_weight = weight_span(weights, plan.lm_head_weight(), lm_head_site)?;
        execute_gemm(
            &mut gemms.lm_head,
            &buffers.hidden_norm,
            lm_head_weight,
            &mut buffers.logits,
            &mut decode_buffers.gemm_workspace,
            stream,
            lm_head_site,
        )?;
        debug_assert_eq!(gemms.lm_head.config().output_bytes(), logits_bytes);
        Ok(())
    }

    fn poison_from_decode_error(&mut self, error: &LlamaDecodeError) {
        match error {
            LlamaDecodeError::Forward(source) => {
                poison_for_forward_error(&mut self.forward.poisoned, source);
            }
            LlamaDecodeError::Cuda { source, .. } => {
                poison_for_cuda_error(&mut self.forward.poisoned, source);
            }
            LlamaDecodeError::Embedding { source, .. } => {
                poison_for_cuda_error(&mut self.forward.poisoned, source.cuda_error());
            }
            LlamaDecodeError::ArithmeticOverflow { .. }
            | LlamaDecodeError::InvalidConfiguration { .. }
            | LlamaDecodeError::PagedKv { .. } => {
                self.forward.poisoned = true;
            }
            _ => {}
        }
    }

    fn abort_cache_reservation(
        &mut self,
        reservation: KvCacheReservation,
        primary: LlamaDecodeError,
    ) -> LlamaDecodeError {
        self.poison_from_decode_error(&primary);
        match self.buffers.cache.poison(reservation) {
            Ok(()) => primary,
            Err(cleanup) => {
                // A failed fail-closed transition is the actionable invariant
                // violation, so it takes priority over the original error.
                self.forward.poisoned = true;
                cleanup
            }
        }
    }
    // HOT_DECODE_END

    /// Downloads the vocabulary row produced by the latest successful stage.
    ///
    /// Prefill dispatches to its last sequence row; one-token decode writes and
    /// downloads row zero of the reused logits allocation.
    ///
    /// # Errors
    ///
    /// Returns when the owner is poisoned, no successful output is published,
    /// the destination length differs from one vocabulary row, or the CUDA
    /// download fails.
    pub fn download_last_logits(
        &mut self,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        if self.is_poisoned() {
            return Err(LlamaDecodeError::Poisoned);
        }
        let latest = self.latest_output.ok_or(LlamaDecodeError::OutputNotReady)?;
        let row_bytes = decode_u64(
            self.forward.plan.dimensions().vocabulary_size(),
            LlamaDecodeResource::GemmWorkspace,
        )?
        .checked_mul(BF16_BYTES)
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::GemmWorkspace,
        })?;
        let expected =
            usize::try_from(row_bytes).map_err(|_| LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::GemmWorkspace,
            })?;
        if destination.len() != expected {
            return Err(LlamaDecodeError::InvalidDownloadLength {
                expected_bytes: expected,
                actual_bytes: destination.len(),
            });
        }
        match latest {
            LatestOutput::Prefill => self
                .forward
                .download_last_logits(destination, stream)
                .map_err(Into::into),
            LatestOutput::Decode => self
                .forward
                .download_logits_range(0, destination, stream)
                .map_err(Into::into),
        }
    }

    /// Explicitly closes decode-local resources and then the embedded forward.
    ///
    /// Every close is attempted after the first failure. This is the normal
    /// error-observing path; field `Drop` remains a best-effort fallback.
    ///
    /// # Errors
    ///
    /// Returns the first decode-local or embedded-forward cleanup failure after
    /// attempting to close every owned resource.
    #[allow(clippy::too_many_lines)]
    pub fn close(self) -> LlamaDecodeResult<()> {
        let Self {
            forward,
            gemms,
            attention: _,
            buffers,
            cache_layout: _,
            allocation_report: _,
            prompt_length: _,
            maximum_sequence_length: _,
            logical_length: _,
            phase: _,
            latest_output: _,
        } = self;
        let DecodeGemmPlans {
            hidden,
            key_value,
            intermediate,
            down,
            lm_head,
        } = gemms;
        let DecodeBuffers {
            cache,
            rope_cos,
            rope_sin,
            attention_workspace,
            gemm_workspace,
            rope_table_bytes_per_kind: _,
        } = buffers;
        let mut first = None;
        record_decode_close(&mut first, LlamaDecodeResource::HiddenGemm, hidden.close());
        record_decode_close(
            &mut first,
            LlamaDecodeResource::KeyValueGemm,
            key_value.close(),
        );
        record_decode_close(
            &mut first,
            LlamaDecodeResource::IntermediateGemm,
            intermediate.close(),
        );
        record_decode_close(&mut first, LlamaDecodeResource::DownGemm, down.close());
        record_decode_close(&mut first, LlamaDecodeResource::LmHeadGemm, lm_head.close());
        match cache {
            KvCacheStorage::Contiguous(cache) => {
                let ContiguousKvCache {
                    key,
                    value,
                    layout: _,
                    layer_offsets: _,
                } = cache;
                record_decode_close(&mut first, LlamaDecodeResource::KeyCache, key.close());
                record_decode_close(&mut first, LlamaDecodeResource::ValueCache, value.close());
            }
            KvCacheStorage::Paged(cache) => {
                let PagedKvCache {
                    key,
                    value,
                    device_block_ids,
                    device_valid_tokens,
                    table_staging,
                    encoded_block_ids: _,
                    encoded_valid_tokens: _,
                    duplicate_scratch: _,
                    layout: _,
                    mut pool,
                    mut sequence,
                } = cache;
                if let Err(source) = sequence.close(&mut pool) {
                    if first.is_none() {
                        first = Some(LlamaDecodeError::PagedKv {
                            operation: "close sequence",
                            source,
                        });
                    }
                }
                record_decode_close(&mut first, LlamaDecodeResource::KeyCache, key.close());
                record_decode_close(&mut first, LlamaDecodeResource::ValueCache, value.close());
                record_decode_close(
                    &mut first,
                    LlamaDecodeResource::BlockTableDeviceIds,
                    device_block_ids.close(),
                );
                record_decode_close(
                    &mut first,
                    LlamaDecodeResource::BlockTableDeviceValidTokens,
                    device_valid_tokens.close(),
                );
                record_decode_close(
                    &mut first,
                    LlamaDecodeResource::BlockTablePinnedStaging,
                    table_staging.close(),
                );
            }
        }
        record_decode_close(&mut first, LlamaDecodeResource::RopeCos, rope_cos.close());
        record_decode_close(&mut first, LlamaDecodeResource::RopeSin, rope_sin.close());
        record_decode_close(
            &mut first,
            LlamaDecodeResource::AttentionWorkspace,
            attention_workspace.close(),
        );
        if let Some(workspace) = gemm_workspace {
            record_decode_close(
                &mut first,
                LlamaDecodeResource::GemmWorkspace,
                workspace.close(),
            );
        }
        if let Err(source) = forward.close() {
            if first.is_none() {
                first = Some(LlamaDecodeError::Forward(source));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn record_decode_close(
    first: &mut Option<LlamaDecodeError>,
    resource: LlamaDecodeResource,
    result: Result<(), CudaError>,
) {
    if let Err(source) = result {
        if first.is_none() {
            *first = Some(LlamaDecodeError::Cleanup { resource, source });
        }
    }
}

fn prepare_decode_gemms(
    context: &CudaContext,
    forward: &PreparedLlamaForward,
    workspace_cap: u64,
) -> LlamaDecodeResult<DecodeGemmPlans> {
    let dimensions = forward.plan.dimensions();
    let hidden = decode_u64(dimensions.hidden_size(), LlamaDecodeResource::GemmWorkspace)?;
    let key_value = decode_u64(
        dimensions.key_value_width(),
        LlamaDecodeResource::GemmWorkspace,
    )?;
    let intermediate = decode_u64(
        dimensions.intermediate_size(),
        LlamaDecodeResource::GemmWorkspace,
    )?;
    let vocabulary = decode_u64(
        dimensions.vocabulary_size(),
        LlamaDecodeResource::GemmWorkspace,
    )?;
    let prepare = |m, n, k, site| -> LlamaDecodeResult<PreparedLlamaGemm> {
        let config = CudaGemmConfig::new(m, n, k, workspace_cap)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        match forward.reduction_profile() {
            LlamaReductionProfile::CanonicalV1 => context
                .prepare_gemm(config)
                .map(PreparedLlamaGemm::Canonical)
                .map_err(|source| LlamaDecodeError::cuda(site, source)),
            LlamaReductionProfile::FixedContiguous37BalancedV1 => context
                .prepare_fixed37_gemm(config)
                .map(PreparedLlamaGemm::Fixed37)
                .map_err(|source| LlamaDecodeError::cuda(site, source)),
        }
    };
    Ok(DecodeGemmPlans {
        hidden: prepare(
            1,
            hidden,
            hidden,
            ExecutionSite::layer(0, LlamaOp::QueryProjection),
        )?,
        key_value: prepare(
            1,
            key_value,
            hidden,
            ExecutionSite::layer(0, LlamaOp::KeyProjection),
        )?,
        intermediate: prepare(
            1,
            intermediate,
            hidden,
            ExecutionSite::layer(0, LlamaOp::GateProjection),
        )?,
        down: prepare(
            1,
            hidden,
            intermediate,
            ExecutionSite::layer(0, LlamaOp::DownProjection),
        )?,
        lm_head: prepare(
            1,
            vocabulary,
            hidden,
            ExecutionSite::global(LlamaOp::LmHead),
        )?,
    })
}

fn rope_table_bytes(sequence_length: usize, head_dimension: usize) -> LlamaDecodeResult<u64> {
    let sequence = decode_u64(sequence_length, LlamaDecodeResource::RopeCos)?;
    let half = decode_u64(head_dimension / 2, LlamaDecodeResource::RopeCos)?;
    sequence
        .checked_mul(half)
        .and_then(|elements| elements.checked_mul(F32_BYTES))
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::RopeCos,
        })
}

fn allocate_decode_host_bytes(
    byte_len: u64,
    resource: LlamaDecodeResource,
) -> LlamaDecodeResult<Box<[u8]>> {
    let length =
        usize::try_from(byte_len).map_err(|_| LlamaDecodeError::ArithmeticOverflow { resource })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| LlamaDecodeError::HostAllocation {
            resource,
            requested_bytes: byte_len,
        })?;
    bytes.resize(length, 0);
    Ok(bytes.into_boxed_slice())
}

type DecodeRopeTableBytes = (Box<[u8]>, Box<[u8]>);

#[allow(clippy::cast_precision_loss)]
fn build_decode_rope_tables(
    sequence_length: usize,
    head_dimension: usize,
    theta: f32,
) -> LlamaDecodeResult<DecodeRopeTableBytes> {
    let bytes = rope_table_bytes(sequence_length, head_dimension)?;
    let mut cos = allocate_decode_host_bytes(bytes, LlamaDecodeResource::RopeCos)?;
    let mut sin = allocate_decode_host_bytes(bytes, LlamaDecodeResource::RopeSin)?;
    let half = head_dimension / 2;
    for position in 0..sequence_length {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            let element = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(4))
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::RopeCos,
                })?;
            cos[element..element + 4].copy_from_slice(&cosine.to_ne_bytes());
            sin[element..element + 4].copy_from_slice(&sine.to_ne_bytes());
        }
    }
    Ok((cos, sin))
}

#[derive(Clone, Copy)]
struct DecodeCacheAllocation {
    kv_cache_bytes: u64,
    block_table_device_bytes: u64,
    block_table_host_bytes: u64,
    unused_capacity_bytes: u64,
    device_allocation_count: u64,
    pinned_host_bytes: u64,
    pinned_host_allocation_count: u64,
}

fn decode_cache_allocation(
    cache: &KvCacheStorage,
    maximum_sequence_length: usize,
) -> LlamaDecodeResult<DecodeCacheAllocation> {
    match cache {
        KvCacheStorage::Contiguous(cache) => Ok(DecodeCacheAllocation {
            kv_cache_bytes: cache.layout.total_bytes(),
            block_table_device_bytes: 0,
            block_table_host_bytes: 0,
            unused_capacity_bytes: 0,
            device_allocation_count: CONTIGUOUS_CACHE_ALLOCATION_COUNT,
            pinned_host_bytes: 0,
            pinned_host_allocation_count: 0,
        }),
        KvCacheStorage::Paged(cache) => {
            let physical_tokens = cache
                .layout
                .physical_block_count()
                .checked_mul(KV_BLOCK_SIZE)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
            let unused_tokens =
                physical_tokens.saturating_sub(maximum_sequence_length.min(physical_tokens));
            let bytes_per_token = cache.layout.bytes_per_physical_block() / PAGED_KV_BLOCK_SIZE;
            let unused_capacity_bytes = decode_u64(unused_tokens, LlamaDecodeResource::KeyCache)?
                .checked_mul(bytes_per_token)
                .ok_or(LlamaDecodeError::ArithmeticOverflow {
                    resource: LlamaDecodeResource::KeyCache,
                })?;
            Ok(DecodeCacheAllocation {
                kv_cache_bytes: cache.layout.total_bytes(),
                block_table_device_bytes: cache.block_table_device_bytes(),
                block_table_host_bytes: cache.block_table_host_bytes(),
                unused_capacity_bytes,
                device_allocation_count: PAGED_CACHE_ALLOCATION_COUNT,
                pinned_host_bytes: cache.table_staging.byte_len(),
                pinned_host_allocation_count: PAGED_CACHE_PINNED_ALLOCATION_COUNT,
            })
        }
    }
}

fn build_decode_allocation_report(
    forward: PreparedLlamaAllocationReport,
    cache: DecodeCacheAllocation,
    rope_table_bytes_per_kind: u64,
    attention_workspace_bytes: u64,
    decode_gemm_workspace_bytes: u64,
) -> LlamaDecodeResult<PreparedLlamaDecodeAllocationReport> {
    let rope_table_bytes =
        rope_table_bytes_per_kind
            .checked_mul(2)
            .ok_or(LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::RopeSin,
            })?;
    let additional_device_bytes = cache
        .kv_cache_bytes
        .checked_add(cache.block_table_device_bytes)
        .and_then(|bytes| bytes.checked_add(rope_table_bytes))
        .and_then(|bytes| bytes.checked_add(attention_workspace_bytes))
        .and_then(|bytes| bytes.checked_add(decode_gemm_workspace_bytes))
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::GemmWorkspace,
        })?;
    let total_device_bytes = forward
        .total_device_bytes()
        .checked_add(additional_device_bytes)
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::GemmWorkspace,
        })?;
    let additional_allocations = cache
        .device_allocation_count
        .checked_add(ROPE_ALLOCATION_COUNT)
        .and_then(|count| count.checked_add(ATTENTION_ALLOCATION_COUNT))
        .and_then(|count| count.checked_add(u64::from(decode_gemm_workspace_bytes != 0)))
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::GemmWorkspace,
        })?;
    let device_allocation_count = forward
        .device_allocation_count()
        .checked_add(additional_allocations)
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::GemmWorkspace,
        })?;
    let pinned_host_bytes = forward
        .pinned_host_bytes()
        .checked_add(cache.pinned_host_bytes)
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::BlockTablePinnedStaging,
        })?;
    let pinned_host_allocation_count = forward
        .pinned_host_allocation_count()
        .checked_add(cache.pinned_host_allocation_count)
        .ok_or(LlamaDecodeError::ArithmeticOverflow {
            resource: LlamaDecodeResource::BlockTablePinnedStaging,
        })?;
    Ok(PreparedLlamaDecodeAllocationReport {
        forward,
        kv_cache_bytes: cache.kv_cache_bytes,
        block_table_device_bytes: cache.block_table_device_bytes,
        block_table_host_bytes: cache.block_table_host_bytes,
        cache_unused_capacity_bytes: cache.unused_capacity_bytes,
        rope_table_bytes,
        attention_workspace_bytes,
        decode_gemm_workspace_bytes,
        additional_device_bytes,
        total_device_bytes,
        device_allocation_count,
        pinned_host_bytes,
        pinned_host_allocation_count,
    })
}

fn checked_host_bytes(
    count: usize,
    element_size: usize,
    resource: LlamaDecodeResource,
) -> LlamaDecodeResult<usize> {
    count
        .checked_mul(element_size)
        .ok_or(LlamaDecodeError::ArithmeticOverflow { resource })
}

fn decode_boxed_zeroed(
    byte_len: usize,
    resource: LlamaDecodeResource,
) -> LlamaDecodeResult<Box<[u8]>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(|_| LlamaDecodeError::HostAllocation {
            resource,
            requested_bytes: u64::try_from(byte_len).unwrap_or(u64::MAX),
        })?;
    bytes.resize(byte_len, 0);
    Ok(bytes.into_boxed_slice())
}

fn decode_u64(value: usize, resource: LlamaDecodeResource) -> LlamaDecodeResult<u64> {
    u64::try_from(value).map_err(|_| LlamaDecodeError::ArithmeticOverflow { resource })
}

fn validate_prefill_request(
    phase: LlamaDecodePhase,
    expected_length: usize,
    actual_length: usize,
) -> LlamaDecodeResult<()> {
    if phase != LlamaDecodePhase::Empty {
        return Err(LlamaDecodeError::InvalidState {
            operation: "prefill",
            actual: phase,
        });
    }
    if actual_length != expected_length {
        return Err(LlamaDecodeError::InvalidPromptLength {
            expected: expected_length,
            actual: actual_length,
        });
    }
    Ok(())
}

fn validate_decode_request(
    phase: LlamaDecodePhase,
    logical_length: usize,
    maximum_length: usize,
) -> LlamaDecodeResult<()> {
    if !matches!(
        phase,
        LlamaDecodePhase::Prefilled | LlamaDecodePhase::Decoding
    ) {
        return Err(LlamaDecodeError::InvalidState {
            operation: "decode",
            actual: phase,
        });
    }
    if logical_length >= maximum_length {
        return Err(LlamaDecodeError::CapacityExceeded {
            logical_length,
            maximum_length,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        LlamaDecodeError, LlamaDecodePhase, LlamaKvCacheLayout, validate_decode_request,
        validate_prefill_request,
    };

    #[test]
    fn cache_layout_is_checked_layer_then_head_major() {
        let layout = LlamaKvCacheLayout::checked(30, 3, 8_192, 64).expect("valid layout");
        assert_eq!(layout.head_stride_bytes(), 8_192 * 64 * 2);
        assert_eq!(layout.layer_stride_bytes(), 3 * 8_192 * 64 * 2);
        assert_eq!(layout.layer_byte_offset(0), Some(0));
        assert_eq!(
            layout.layer_byte_offset(1),
            Some(layout.layer_stride_bytes())
        );
        assert_eq!(
            layout.layer_byte_offset(29),
            Some(29 * layout.layer_stride_bytes())
        );
        assert_eq!(layout.layer_byte_offset(30), None);
        assert_eq!(layout.bytes_per_kind(), 30 * layout.layer_stride_bytes());
        assert_eq!(layout.total_bytes(), 2 * layout.bytes_per_kind());
    }

    #[test]
    fn cache_layout_rejects_zero_and_overflow() {
        for dimensions in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)] {
            assert!(
                LlamaKvCacheLayout::checked(dimensions.0, dimensions.1, dimensions.2, dimensions.3)
                    .is_err()
            );
        }
        assert!(LlamaKvCacheLayout::checked(usize::MAX, 2, usize::MAX, 64).is_err());
    }

    #[test]
    fn lifecycle_validation_is_pre_mutation_and_capacity_exact() {
        validate_prefill_request(LlamaDecodePhase::Empty, 7, 7).expect("first prefill");
        assert!(matches!(
            validate_prefill_request(LlamaDecodePhase::Prefilled, 7, 7),
            Err(LlamaDecodeError::InvalidState { .. })
        ));
        assert!(matches!(
            validate_prefill_request(LlamaDecodePhase::Empty, 7, 6),
            Err(LlamaDecodeError::InvalidPromptLength { .. })
        ));

        assert!(matches!(
            validate_decode_request(LlamaDecodePhase::Empty, 0, 8),
            Err(LlamaDecodeError::InvalidState { .. })
        ));
        validate_decode_request(LlamaDecodePhase::Prefilled, 7, 8).expect("last free slot");
        assert!(matches!(
            validate_decode_request(LlamaDecodePhase::Decoding, 8, 8),
            Err(LlamaDecodeError::CapacityExceeded {
                logical_length: 8,
                maximum_length: 8
            })
        ));
    }
}
