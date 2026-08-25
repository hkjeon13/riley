//! Owning contiguous-KV single-request Llama decode path.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::error;
use std::fmt;
use std::mem;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaError,
    CudaGemmConfig, CudaPreparedGemm, CudaStream, DecodeAttentionBackend,
    DecodeAttentionBackendAvailability, DecodeAttentionParams, DecodeAttentionPreference,
    DecodeAttentionRequest, EmbeddingError, EmbeddingParams, GatedMultiplyParams,
    KvCacheAppendParams, PreparedDecodeAttention, ResidualAddParams, RmsNormParams, RopeParams,
    SiluParams, embedding, gated_multiply, kv_cache_append, residual_add, rms_norm, rope, silu,
};
use rustinfer_model::LoadedModel;

use super::forward::{
    LlamaForwardError, PreparedLlamaAllocationReport, PreparedLlamaForward,
    PreparedLlamaForwardConfig, execute_gemm, poison_for_cuda_error, poison_for_forward_error,
    span, span_mut, weight_span,
};
use super::{ExecutionSite, LlamaOp};

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const U32_BYTES: u64 = 4;
const CACHE_ALLOCATION_COUNT: u64 = 2;
const ROPE_ALLOCATION_COUNT: u64 = 2;
const ATTENTION_ALLOCATION_COUNT: u64 = 1;

/// Result type for PR09 single-request preparation and execution.
pub type LlamaDecodeResult<T> = Result<T, LlamaDecodeError>;

/// Request-local resource named in allocation and cleanup diagnostics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaDecodeResource {
    KeyCache,
    ValueCache,
    CacheLayerOffsets,
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
                field: "contiguous_kv_cache",
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

/// Cold-path settings for one fixed-prompt decode owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaDecodeConfig {
    forward: PreparedLlamaForwardConfig,
    decode_attention_preference: DecodeAttentionPreference,
}

impl PreparedLlamaDecodeConfig {
    #[must_use]
    pub const fn new(forward: PreparedLlamaForwardConfig) -> Self {
        Self {
            forward,
            decode_attention_preference: DecodeAttentionPreference::Optimized,
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

    #[must_use]
    pub const fn forward(self) -> PreparedLlamaForwardConfig {
        self.forward
    }

    #[must_use]
    pub const fn decode_attention_preference(self) -> DecodeAttentionPreference {
        self.decode_attention_preference
    }
}

impl Default for PreparedLlamaDecodeConfig {
    fn default() -> Self {
        Self::new(PreparedLlamaForwardConfig::default())
    }
}

/// Exact owned CUDA allocation totals after decode preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaDecodeAllocationReport {
    forward: PreparedLlamaAllocationReport,
    kv_cache_bytes: u64,
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

struct DecodeGemmPlans {
    hidden: CudaPreparedGemm,
    key_value: CudaPreparedGemm,
    intermediate: CudaPreparedGemm,
    down: CudaPreparedGemm,
    lm_head: CudaPreparedGemm,
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
            self.hidden.algorithm_metadata().workspace_bytes(),
            self.key_value.algorithm_metadata().workspace_bytes(),
            self.intermediate.algorithm_metadata().workspace_bytes(),
            self.down.algorithm_metadata().workspace_bytes(),
            self.lm_head.algorithm_metadata().workspace_bytes(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }
}

struct DecodeBuffers {
    cache: ContiguousKvCache,
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
    attention: PreparedDecodeAttention,
    buffers: DecodeBuffers,
    layout: LlamaKvCacheLayout,
    allocation_report: PreparedLlamaDecodeAllocationReport,
    prompt_length: usize,
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
            .field("maximum_length", &self.layout.maximum_sequence_length())
            .field("phase", &self.phase)
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
        let layout = LlamaKvCacheLayout::checked(
            forward.plan.layers().len(),
            dimensions.key_value_heads(),
            maximum_sequence_length,
            dimensions.head_dimension(),
        )?;

        let head_size = decode_u64(
            dimensions.head_dimension(),
            LlamaDecodeResource::AttentionWorkspace,
        )?;
        let request = DecodeAttentionRequest::new(
            decode_u64(
                maximum_sequence_length,
                LlamaDecodeResource::AttentionWorkspace,
            )?,
            decode_u64(
                dimensions.query_heads(),
                LlamaDecodeResource::AttentionWorkspace,
            )?,
            decode_u64(
                dimensions.key_value_heads(),
                LlamaDecodeResource::AttentionWorkspace,
            )?,
            head_size,
            1.0 / (head_size as f32).sqrt(),
        );
        let attention = PreparedDecodeAttention::select(
            context,
            request,
            config.decode_attention_preference(),
            DecodeAttentionBackendAvailability::linked(),
        )
        .map_err(|source| {
            LlamaDecodeError::cuda(ExecutionSite::layer(0, LlamaOp::DecodeAttention), source)
        })?;
        let gemms = prepare_decode_gemms(
            context,
            &forward,
            config.forward().gemm_workspace_cap_bytes(),
        )?;
        let decode_gemm_workspace_bytes = gemms.maximum_workspace_bytes();
        let cache = ContiguousKvCache::prepare(context, layout)?;
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
            layout,
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
            layout,
            allocation_report,
            prompt_length,
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
        self.layout.maximum_sequence_length()
    }

    #[must_use]
    pub const fn phase(&self) -> LlamaDecodePhase {
        self.phase
    }

    #[must_use]
    pub const fn cache_layout(&self) -> LlamaKvCacheLayout {
        self.layout
    }

    #[must_use]
    pub const fn allocation_report(&self) -> PreparedLlamaDecodeAllocationReport {
        self.allocation_report
    }

    #[must_use]
    pub const fn prepared_attention(&self) -> &PreparedDecodeAttention {
        &self.attention
    }

    #[must_use]
    pub fn is_poisoned(&self) -> bool {
        self.forward.poisoned || self.gemms.any_poisoned()
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
        self.forward.upload_tokens(prompt, stream)?;
        self.latest_output = None;
        self.forward
            .execute_prefill_into_cache(&mut self.buffers.cache, stream)?;
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

        if let Err(error) = self.upload_decode_token(token_id, stream) {
            self.poison_from_decode_error(&error);
            return Err(error);
        }
        self.latest_output = None;
        self.forward.output_ready = false;
        let position = self.logical_length;
        if let Err(error) = self.execute_decode_inner(position, stream) {
            self.poison_from_decode_error(&error);
            self.forward.poisoned |= self.gemms.any_poisoned();
            return Err(error);
        }
        self.logical_length += 1;
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
        stream: &mut CudaStream,
    ) -> LlamaDecodeResult<()> {
        let forward = &mut self.forward;
        let plan = &forward.plan;
        let weights = &forward.weights;
        let buffers = &mut forward.buffers;
        let gemms = &mut self.gemms;
        let attention = &self.attention;
        let decode_buffers = &mut self.buffers;
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
        let key_value_bytes = key_value_heads
            .checked_mul(head_size)
            .and_then(|elements| elements.checked_mul(BF16_BYTES))
            .ok_or(LlamaDecodeError::ArithmeticOverflow {
                resource: LlamaDecodeResource::KeyCache,
            })?;
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
                rms_norm(&mut params, stream)
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
                        self.layout.maximum_sequence_length(),
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
                        self.layout.maximum_sequence_length(),
                        LlamaDecodeResource::RopeSin,
                    )?,
                    position_offset: position,
                };
                rope(&mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(key_rope_site, source))?;
            }

            decode_buffers.cache.append_layer(
                layer_index,
                &buffers.key_rotary,
                &buffers.value_raw,
                1,
                position,
                stream,
            )?;

            let attention_site = ExecutionSite::layer(layer_index, LlamaOp::DecodeAttention);
            {
                let (key_cache, value_cache) = decode_buffers.cache.layer_spans(layer_index)?;
                let workspace_dtype = match attention.backend() {
                    DecodeAttentionBackend::MaterializedReference => CudaDType::BF16,
                    DecodeAttentionBackend::ChunkedOnline => CudaDType::F32,
                };
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
                        workspace_dtype,
                        0,
                        attention.workspace_bytes(),
                    )
                    .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?,
                };
                attention
                    .execute(logical_token_count, &mut params, stream)
                    .map_err(|source| LlamaDecodeError::cuda(attention_site, source))?;
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
                rms_norm(&mut params, stream)
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
            rms_norm(&mut params, stream)
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
            | LlamaDecodeError::InvalidConfiguration { .. } => {
                self.forward.poisoned = true;
            }
            _ => {}
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
            layout: _,
            allocation_report: _,
            prompt_length: _,
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
        let ContiguousKvCache {
            key,
            value,
            layout: _,
            layer_offsets: _,
        } = cache;
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
        record_decode_close(&mut first, LlamaDecodeResource::KeyCache, key.close());
        record_decode_close(&mut first, LlamaDecodeResource::ValueCache, value.close());
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
    let prepare = |m, n, k, site| -> LlamaDecodeResult<CudaPreparedGemm> {
        let config = CudaGemmConfig::new(m, n, k, workspace_cap)
            .map_err(|source| LlamaDecodeError::cuda(site, source))?;
        context
            .prepare_gemm(config)
            .map_err(|source| LlamaDecodeError::cuda(site, source))
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

fn build_decode_allocation_report(
    forward: PreparedLlamaAllocationReport,
    layout: LlamaKvCacheLayout,
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
    let additional_device_bytes = layout
        .total_bytes()
        .checked_add(rope_table_bytes)
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
    let additional_allocations = CACHE_ALLOCATION_COUNT
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
    Ok(PreparedLlamaDecodeAllocationReport {
        forward,
        kv_cache_bytes: layout.total_bytes(),
        rope_table_bytes,
        attention_workspace_bytes,
        decode_gemm_workspace_bytes,
        additional_device_bytes,
        total_device_bytes,
        device_allocation_count,
        pinned_host_bytes: forward.pinned_host_bytes(),
        pinned_host_allocation_count: forward.pinned_host_allocation_count(),
    })
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
