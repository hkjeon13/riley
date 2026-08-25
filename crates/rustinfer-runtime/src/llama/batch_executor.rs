//! Owning fixed-width CUDA executor for mixed Llama prefill/decode batches.
//!
//! One cold-prepared owner executes every iteration as a dense tensor graph
//! with `M = max_input_tokens` rows. Only the first `T` rows are active; token
//! zero pads `[T, M)`. Indexed `RoPE`, paged KV scatter, and ragged causal
//! attention preserve each row's absolute sequence position while every GEMM
//! remains the same prepared fixed-M operation.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::error;
use std::fmt;
use std::mem;

use rustinfer_cuda::{
    CudaBufferSpan, CudaBufferSpanMut, CudaContext, CudaDType, CudaDeviceBuffer, CudaError,
    CudaStream, EmbeddingParams, GatedMultiplyParams, IndexedRopeParams, PackedBatchHostV1,
    PackedBatchV1, RaggedPagedAttentionParams, RaggedPagedKvCacheWriteParams, ResidualAddParams,
    RmsNormParams, RowGatherParams, SiluParams, embedding, gated_multiply, indexed_rope,
    ragged_paged_attention, ragged_paged_kv_cache_write, residual_add, rms_norm, row_gather, silu,
};
use rustinfer_model::LoadedModel;

use super::batch::{
    LLAMA_BATCH_METADATA_V1_VERSION, LlamaBatchError, LlamaBatchMetadataConfig, LlamaBatchRow,
    LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
};
use super::forward::{
    LlamaForwardError, PreparedLlamaAllocationReport, PreparedLlamaForward,
    PreparedLlamaForwardConfig, execute_gemm, execute_projection_bias, poison_for_cuda_error,
    poison_for_forward_error, span, span_mut, weight_span,
};
use super::{ExecutionSite, LlamaOp};
use crate::paged_kv::{KV_BLOCK_SIZE, KvLayout, PagedKvError};

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const BF16_BYTES_USIZE: usize = 2;
const F32_BYTES_USIZE: usize = 4;
const U32_BYTES: usize = 4;
const U16_BYTES: usize = 2;
const SUPPORTED_HEAD_DIMENSION: usize = 64;
const BASE_ADDITIONAL_DEVICE_ALLOCATIONS: u64 = 9;

/// Result type for continuous-batch preparation, execution, transfer, and close.
pub type LlamaBatchExecutorResult<T> = Result<T, LlamaBatchExecutorError>;

/// Extra owning resource held by [`PreparedLlamaBatchExecutor`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaBatchExecutorResource {
    KeyCache,
    ValueCache,
    RopeCos,
    RopeSin,
    SequenceBlockOffsets,
    PhysicalBlockIds,
    ValidTokens,
    RowSequenceSlots,
    RowPositions,
    OutputTokenIndices,
    GatheredLogits,
    HostWorkspace,
}

impl LlamaBatchExecutorResource {
    const fn name(self) -> &'static str {
        match self {
            Self::KeyCache => "shared_key_cache",
            Self::ValueCache => "shared_value_cache",
            Self::RopeCos => "absolute_rope_cos",
            Self::RopeSin => "absolute_rope_sin",
            Self::SequenceBlockOffsets => "batch_sequence_block_offsets",
            Self::PhysicalBlockIds => "batch_physical_block_ids",
            Self::ValidTokens => "batch_valid_tokens",
            Self::RowSequenceSlots => "batch_row_sequence_slots",
            Self::RowPositions => "batch_row_positions",
            Self::OutputTokenIndices => "batch_output_token_indices",
            Self::GatheredLogits => "batch_gathered_logits",
            Self::HostWorkspace => "batch_host_workspace",
        }
    }
}

impl fmt::Display for LlamaBatchExecutorResource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Structured failure from the fixed-M continuous-batch executor.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaBatchExecutorError {
    Metadata(LlamaBatchError),
    Forward(LlamaForwardError),
    PagedKv(PagedKvError),
    InvalidConfiguration {
        field: &'static str,
        reason: &'static str,
    },
    UnsupportedHeadDimension {
        expected: usize,
        actual: usize,
    },
    InvalidBatch {
        field: &'static str,
        reason: &'static str,
    },
    TokenOutOfRange {
        position: usize,
        token_id: u32,
        vocabulary_size: usize,
    },
    PositionOutOfRange {
        row: usize,
        position: u32,
        maximum: usize,
    },
    Cuda {
        site: ExecutionSite,
        source: CudaError,
    },
    HostAllocation {
        resource: LlamaBatchExecutorResource,
        requested_bytes: u64,
    },
    ArithmeticOverflow {
        resource: LlamaBatchExecutorResource,
    },
    OutputNotReady,
    Poisoned,
    InvalidDownloadLength {
        expected_bytes: usize,
        actual_bytes: usize,
    },
    Cleanup {
        resource: LlamaBatchExecutorResource,
        source: CudaError,
    },
}

impl fmt::Display for LlamaBatchExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(source) => source.fmt(formatter),
            Self::Forward(source) => source.fmt(formatter),
            Self::PagedKv(source) => source.fmt(formatter),
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama batch executor configuration {field}: {reason}"
                )
            }
            Self::UnsupportedHeadDimension { expected, actual } => write!(
                formatter,
                "continuous-batch attention requires head dimension {expected}, got {actual}"
            ),
            Self::InvalidBatch { field, reason } => {
                write!(
                    formatter,
                    "invalid executable Llama batch {field}: {reason}"
                )
            }
            Self::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "batch token ID {token_id} at flattened position {position} is outside vocabulary 0..{vocabulary_size}"
            ),
            Self::PositionOutOfRange {
                row,
                position,
                maximum,
            } => write!(
                formatter,
                "batch row {row} uses absolute position {position} outside 0..{maximum}"
            ),
            Self::Cuda { site, source } => write!(formatter, "{site}: {source}"),
            Self::HostAllocation {
                resource,
                requested_bytes,
            } => write!(
                formatter,
                "could not reserve {requested_bytes} host bytes for {resource}"
            ),
            Self::ArithmeticOverflow { resource } => {
                write!(formatter, "byte arithmetic overflow for {resource}")
            }
            Self::OutputNotReady => formatter
                .write_str("gathered batch logits are unavailable before successful execution"),
            Self::Poisoned => formatter.write_str(
                "the Llama batch executor was poisoned by a native CUDA execution failure",
            ),
            Self::InvalidDownloadLength {
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "batch-logit destination has {actual_bytes} bytes, expected {expected_bytes}"
            ),
            Self::Cleanup { resource, source } => {
                write!(formatter, "could not close {resource}: {source}")
            }
        }
    }
}

impl error::Error for LlamaBatchExecutorError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Metadata(source) => Some(source),
            Self::Forward(source) => Some(source),
            Self::PagedKv(source) => Some(source),
            Self::Cuda { source, .. } | Self::Cleanup { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl From<LlamaBatchError> for LlamaBatchExecutorError {
    fn from(source: LlamaBatchError) -> Self {
        Self::Metadata(source)
    }
}

impl From<LlamaForwardError> for LlamaBatchExecutorError {
    fn from(source: LlamaForwardError) -> Self {
        Self::Forward(source)
    }
}

impl From<PagedKvError> for LlamaBatchExecutorError {
    fn from(source: PagedKvError) -> Self {
        Self::PagedKv(source)
    }
}

/// Cold bounds for one reusable fixed-M continuous-batch owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaBatchExecutorConfig {
    metadata: LlamaBatchMetadataConfig,
    forward: PreparedLlamaForwardConfig,
}

impl PreparedLlamaBatchExecutorConfig {
    #[must_use]
    pub const fn new(
        metadata: LlamaBatchMetadataConfig,
        forward: PreparedLlamaForwardConfig,
    ) -> Self {
        Self { metadata, forward }
    }

    #[must_use]
    pub const fn metadata(self) -> LlamaBatchMetadataConfig {
        self.metadata
    }

    #[must_use]
    pub const fn forward(self) -> PreparedLlamaForwardConfig {
        self.forward
    }
}

/// Exact owned allocation totals after cold batch preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaBatchAllocationReport {
    forward: PreparedLlamaAllocationReport,
    kv_cache_bytes: u64,
    rope_table_bytes: u64,
    packed_metadata_device_bytes: u64,
    gathered_logits_capacity_bytes: u64,
    additional_device_bytes: u64,
    total_device_bytes: u64,
    additional_device_allocation_count: u64,
    total_device_allocation_count: u64,
    host_workspace_bytes: u64,
}

impl PreparedLlamaBatchAllocationReport {
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
    pub const fn packed_metadata_device_bytes(self) -> u64 {
        self.packed_metadata_device_bytes
    }

    #[must_use]
    pub const fn gathered_logits_capacity_bytes(self) -> u64 {
        self.gathered_logits_capacity_bytes
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
    pub const fn additional_device_allocation_count(self) -> u64 {
        self.additional_device_allocation_count
    }

    #[must_use]
    pub const fn total_device_allocation_count(self) -> u64 {
        self.total_device_allocation_count
    }

    #[must_use]
    pub const fn host_workspace_bytes(self) -> u64 {
        self.host_workspace_bytes
    }

    #[must_use]
    pub const fn pinned_host_bytes(self) -> u64 {
        self.forward.pinned_host_bytes()
    }

    #[must_use]
    pub const fn pinned_host_allocation_count(self) -> u64 {
        self.forward.pinned_host_allocation_count()
    }
}

struct BatchDeviceMetadata {
    sequence_block_offsets: CudaDeviceBuffer,
    physical_block_ids: CudaDeviceBuffer,
    valid_tokens: CudaDeviceBuffer,
    row_sequence_slots: CudaDeviceBuffer,
    row_positions: CudaDeviceBuffer,
    output_token_indices: Option<CudaDeviceBuffer>,
}

struct BatchHostWorkspace {
    padded_tokens: Box<[u32]>,
    sequence_block_offsets: Box<[u8]>,
    physical_block_ids: Box<[u8]>,
    valid_tokens: Box<[u8]>,
    row_sequence_slots: Box<[u8]>,
    row_positions: Box<[u8]>,
    output_token_indices: Box<[u8]>,
}

/// Fixed-width, shared-KV Llama continuous-batch executor.
///
/// The scheduler retains ownership of logical reservations. A successful call
/// only establishes that every synchronous native operation completed; the
/// caller may commit the matching scheduler iteration after `execute` returns.
/// A failed native operation poisons this owner and the caller must abort the
/// iteration instead of publishing any partial KV writes.
pub struct PreparedLlamaBatchExecutor {
    config: PreparedLlamaBatchExecutorConfig,
    metadata: PreparedLlamaBatchMetadata,
    forward: PreparedLlamaForward,
    layout: KvLayout,
    key_cache: CudaDeviceBuffer,
    value_cache: CudaDeviceBuffer,
    absolute_rope_cos: CudaDeviceBuffer,
    absolute_rope_sin: CudaDeviceBuffer,
    device_metadata: BatchDeviceMetadata,
    gathered_logits: Option<CudaDeviceBuffer>,
    host: BatchHostWorkspace,
    allocation_report: PreparedLlamaBatchAllocationReport,
    output_count: usize,
    output_ready: bool,
    poisoned: bool,
}

impl fmt::Debug for PreparedLlamaBatchExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaBatchExecutor")
            .field("config", &self.config)
            .field("layout", &self.layout)
            .field("allocation_report", &self.allocation_report)
            .field("output_count", &self.output_count)
            .field("output_ready", &self.output_ready)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl PreparedLlamaBatchExecutor {
    /// Uploads weights and allocates every host/device byte used by repeated
    /// mixed-batch execution.
    ///
    /// The current ragged kernel is deliberately D64-only. Preparation rejects
    /// other head widths before uploading weights or allocating CUDA storage.
    /// `max_input_tokens` is the fixed dense GEMM row count M and must not
    /// exceed the model's maximum sequence length while this implementation
    /// reuses [`PreparedLlamaForward`]'s fixed-S plans.
    ///
    /// # Errors
    ///
    /// Returns a model/configuration, host allocation, CUDA preparation, weight
    /// upload, or checked KV-layout error. No partially prepared owner is
    /// returned.
    #[allow(clippy::too_many_lines)]
    pub fn prepare(
        model: &LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        config: PreparedLlamaBatchExecutorConfig,
    ) -> LlamaBatchExecutorResult<Self> {
        let config = PreparedLlamaBatchExecutorConfig::new(
            config.metadata,
            config.forward.with_optimized_attention(),
        );
        let spec = model.spec();
        let attention = spec
            .blocks()
            .first()
            .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "model.blocks",
                reason: "the validated model must contain at least one decoder layer",
            })?
            .attention();
        if attention.head_dimension() != SUPPORTED_HEAD_DIMENSION {
            return Err(LlamaBatchExecutorError::UnsupportedHeadDimension {
                expected: SUPPORTED_HEAD_DIMENSION,
                actual: attention.head_dimension(),
            });
        }
        let bounds = config.metadata;
        if bounds.max_input_tokens() > spec.max_sequence_length() {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "max_input_tokens",
                reason: "fixed-M forward reuse currently requires M <= model max sequence length",
            });
        }

        let metadata = PreparedLlamaBatchMetadata::prepare(bounds)?;
        let mut forward = PreparedLlamaForward::prepare(
            model,
            context,
            stream,
            bounds.max_input_tokens(),
            config.forward,
        )?;
        let dimensions = forward.plan.dimensions();
        if dimensions.head_dimension() != SUPPORTED_HEAD_DIMENSION {
            return Err(LlamaBatchExecutorError::UnsupportedHeadDimension {
                expected: SUPPORTED_HEAD_DIMENSION,
                actual: dimensions.head_dimension(),
            });
        }
        let layout = KvLayout::checked(
            forward.plan.layers().len(),
            bounds.physical_block_count(),
            dimensions.key_value_heads(),
            dimensions.head_dimension(),
        )?;

        let key_cache = allocate_device(
            context,
            layout.bytes_per_kind(),
            ExecutionSite::layer(0, LlamaOp::KvCacheWrite),
        )?;
        let value_cache = allocate_device(
            context,
            layout.bytes_per_kind(),
            ExecutionSite::layer(0, LlamaOp::KvCacheWrite),
        )?;
        let rope_bytes_per_kind = checked_product_u64(
            &[
                usize_u64(
                    spec.max_sequence_length(),
                    LlamaBatchExecutorResource::RopeCos,
                )?,
                usize_u64(
                    dimensions.head_dimension() / 2,
                    LlamaBatchExecutorResource::RopeCos,
                )?,
                F32_BYTES,
            ],
            LlamaBatchExecutorResource::RopeCos,
        )?;
        let mut absolute_rope_cos = allocate_device(
            context,
            rope_bytes_per_kind,
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?;
        let mut absolute_rope_sin = allocate_device(
            context,
            rope_bytes_per_kind,
            ExecutionSite::layer(0, LlamaOp::QueryRope),
        )?;
        let (rope_cos, rope_sin) = build_absolute_rope_tables(
            spec.max_sequence_length(),
            dimensions.head_dimension(),
            forward.plan.rope_theta(),
        )?;
        absolute_rope_cos
            .upload_from_slice(0, &rope_cos, &mut forward.io_staging, stream)
            .map_err(|source| batch_cuda(ExecutionSite::layer(0, LlamaOp::QueryRope), source))?;
        absolute_rope_sin
            .upload_from_slice(0, &rope_sin, &mut forward.io_staging, stream)
            .map_err(|source| batch_cuda(ExecutionSite::layer(0, LlamaOp::QueryRope), source))?;

        let device_metadata = allocate_device_metadata(context, bounds)?;
        let gathered_logits_capacity_bytes = checked_product_u64(
            &[
                usize_u64(
                    bounds.max_output_slots(),
                    LlamaBatchExecutorResource::GatheredLogits,
                )?,
                usize_u64(
                    dimensions.vocabulary_size(),
                    LlamaBatchExecutorResource::GatheredLogits,
                )?,
                BF16_BYTES,
            ],
            LlamaBatchExecutorResource::GatheredLogits,
        )?;
        let gathered_logits = if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate_device(
                context,
                gathered_logits_capacity_bytes,
                ExecutionSite::global(LlamaOp::OutputGather),
            )?)
        };
        let host = allocate_host_workspace(bounds)?;
        let allocation_report = build_batch_allocation_report(
            forward.allocation_report(),
            bounds,
            layout,
            rope_bytes_per_kind,
            gathered_logits_capacity_bytes,
            &host,
        )?;

        Ok(Self {
            config,
            metadata,
            forward,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_metadata,
            gathered_logits,
            host,
            allocation_report,
            output_count: 0,
            output_ready: false,
            poisoned: false,
        })
    }

    #[must_use]
    pub const fn config(&self) -> PreparedLlamaBatchExecutorConfig {
        self.config
    }

    /// Vocabulary width of every gathered output row.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.forward.plan.dimensions().vocabulary_size()
    }

    /// Number of absolute positions represented by the cold-prepared `RoPE` tables.
    ///
    /// # Errors
    ///
    /// Returns when the owned table byte shape is inconsistent or cannot be
    /// represented as a host `usize`.
    pub fn maximum_position_count(&self) -> LlamaBatchExecutorResult<usize> {
        let positions = model_max_position(
            &self.absolute_rope_cos,
            self.forward.plan.dimensions().head_dimension(),
        )?;
        usize::try_from(positions).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::RopeCos,
        })
    }

    #[must_use]
    pub const fn kv_layout(&self) -> KvLayout {
        self.layout
    }

    #[must_use]
    pub const fn allocation_report(&self) -> PreparedLlamaBatchAllocationReport {
        self.allocation_report
    }

    #[must_use]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    #[must_use]
    pub const fn output_ready(&self) -> bool {
        self.output_ready
    }

    #[must_use]
    pub const fn is_poisoned(&self) -> bool {
        self.poisoned || self.forward.poisoned
    }

    /// Validates, packs, uploads, and executes one mixed iteration.
    ///
    /// All host/model bounds are checked before the first device mutation.
    /// Packing, encoding, upload, execution, and output routing reuse cold
    /// storage and perform no host or device allocation.
    ///
    /// # Errors
    ///
    /// Returns for malformed or over-capacity metadata, invalid token/position
    /// IDs, a poisoned owner, or any CUDA operation failure. Native execution
    /// failures poison the owner because KV mutation may be partial.
    pub fn execute(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        self.output_ready = false;
        self.output_count = 0;
        self.forward.output_ready = false;
        let Self {
            config,
            metadata,
            forward,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_metadata,
            gathered_logits,
            host,
            allocation_report: _,
            output_count,
            output_ready,
            poisoned,
        } = self;
        let packed = metadata.pack(rows)?;
        validate_for_execution(
            packed,
            forward.plan.dimensions().vocabulary_size(),
            model_max_position(
                absolute_rope_cos,
                forward.plan.dimensions().head_dimension(),
            )?,
            *config,
        )?;

        let result = execute_packed(
            packed,
            *config,
            forward,
            *layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_metadata,
            gathered_logits,
            host,
            stream,
        );
        match result {
            Ok(()) => {
                *output_count = packed.output_count();
                *output_ready = true;
                forward.output_ready = true;
                Ok(())
            }
            Err(error) => {
                poison_for_batch_error(poisoned, forward, &error);
                Err(error)
            }
        }
    }

    /// Exact BF16 byte length of the most recently gathered `[O,V]` output.
    ///
    /// # Errors
    ///
    /// Returns when the output shape cannot be represented as a host byte
    /// length.
    pub fn output_byte_len(&self) -> LlamaBatchExecutorResult<usize> {
        self.output_byte_len_for(self.output_count)
    }

    /// Exact BF16 byte length needed for a prospective `[output_count,V]` download.
    ///
    /// This pre-dispatch query lets orchestration allocate its destination
    /// before any reserved KV range can be mutated.
    ///
    /// # Errors
    ///
    /// Returns when `output_count` exceeds the cold-prepared bound or the byte
    /// length cannot be represented as a host `usize`.
    pub fn output_byte_len_for(&self, output_count: usize) -> LlamaBatchExecutorResult<usize> {
        if output_count > self.config.metadata.max_output_slots() {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "output_count",
                reason: "prospective output count exceeds the cold-prepared bound",
            });
        }
        output_count
            .checked_mul(self.vocabulary_size())
            .and_then(|elements| elements.checked_mul(BF16_BYTES_USIZE))
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::GatheredLogits,
            })
    }

    /// Downloads only gathered sampled rows `[O,V]`, in dense output-slot order.
    ///
    /// # Errors
    ///
    /// Returns when execution has not produced output, the owner is poisoned,
    /// the destination length differs from [`Self::output_byte_len`], or the
    /// synchronous CUDA transfer fails.
    pub fn download_logits(
        &mut self,
        destination: &mut [u8],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaBatchExecutorError::OutputNotReady);
        }
        let expected = self.output_byte_len()?;
        if destination.len() != expected {
            return Err(LlamaBatchExecutorError::InvalidDownloadLength {
                expected_bytes: expected,
                actual_bytes: destination.len(),
            });
        }
        if destination.is_empty() {
            return Ok(());
        }
        let gathered =
            self.gathered_logits
                .as_mut()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "gathered_logits",
                    reason: "non-empty output has no cold-prepared device buffer",
                })?;
        match gathered.download_to_slice(0, destination, &mut self.forward.io_staging, stream) {
            Ok(()) => Ok(()),
            Err(source) => {
                poison_for_cuda_error(&mut self.poisoned, &source);
                Err(batch_cuda(
                    ExecutionSite::global(LlamaOp::OutputGather),
                    source,
                ))
            }
        }
    }

    /// Explicitly closes all extra batch allocations, then the reused forward.
    /// Every resource is attempted even after the first cleanup failure.
    ///
    /// # Errors
    ///
    /// Returns the first CUDA cleanup failure after attempting every owned
    /// device resource and the underlying prepared forward.
    #[allow(clippy::too_many_lines)]
    pub fn close(self) -> LlamaBatchExecutorResult<()> {
        let Self {
            config: _,
            metadata: _,
            forward,
            layout: _,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_metadata,
            gathered_logits,
            host: _,
            allocation_report: _,
            output_count: _,
            output_ready: _,
            poisoned: _,
        } = self;
        let BatchDeviceMetadata {
            sequence_block_offsets,
            physical_block_ids,
            valid_tokens,
            row_sequence_slots,
            row_positions,
            output_token_indices,
        } = device_metadata;
        let mut first = None;
        record_close(
            &mut first,
            LlamaBatchExecutorResource::KeyCache,
            key_cache.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::ValueCache,
            value_cache.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RopeCos,
            absolute_rope_cos.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RopeSin,
            absolute_rope_sin.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::SequenceBlockOffsets,
            sequence_block_offsets.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::PhysicalBlockIds,
            physical_block_ids.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::ValidTokens,
            valid_tokens.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RowSequenceSlots,
            row_sequence_slots.close(),
        );
        record_close(
            &mut first,
            LlamaBatchExecutorResource::RowPositions,
            row_positions.close(),
        );
        if let Some(buffer) = output_token_indices {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::OutputTokenIndices,
                buffer.close(),
            );
        }
        if let Some(buffer) = gathered_logits {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::GatheredLogits,
                buffer.close(),
            );
        }
        let forward_result = forward.close().map_err(LlamaBatchExecutorError::Forward);
        match (first, forward_result) {
            (Some(error), _) => Err(error),
            (None, result) => result,
        }
    }
}

// HOT_BATCH_EXECUTE_BEGIN
#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value
)]
fn execute_packed(
    packed: LlamaPackedBatchMetadata<'_>,
    config: PreparedLlamaBatchExecutorConfig,
    forward: &mut PreparedLlamaForward,
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    device: &mut BatchDeviceMetadata,
    gathered_logits: &mut Option<CudaDeviceBuffer>,
    host: &mut BatchHostWorkspace,
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    let bounds = config.metadata;
    let active = packed.total_input_tokens();
    let metadata_site = ExecutionSite::global(LlamaOp::BatchMetadataUpload);
    let host_batch = PackedBatchHostV1::new(
        packed.block_row_offsets(),
        packed.physical_block_ids(),
        packed.valid_tokens(),
        packed.row_sequence_slots(),
        packed.position_ids(),
        usize_u64(
            bounds.physical_block_count(),
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?,
    )
    .map_err(|source| batch_cuda(metadata_site, source))?;
    host.padded_tokens.fill(0);
    host.padded_tokens[..active].copy_from_slice(packed.input_token_ids());
    forward.upload_tokens(&host.padded_tokens, stream)?;
    encode_u32(packed.block_row_offsets(), &mut host.sequence_block_offsets);
    encode_u32(packed.physical_block_ids(), &mut host.physical_block_ids);
    encode_u16(packed.valid_tokens(), &mut host.valid_tokens);
    encode_u32(packed.row_sequence_slots(), &mut host.row_sequence_slots);
    encode_u32(packed.position_ids(), &mut host.row_positions);
    encode_u32(
        packed.output_token_indices(),
        &mut host.output_token_indices,
    );

    upload_prefix(
        &mut device.sequence_block_offsets,
        &host.sequence_block_offsets,
        packed.block_row_offsets().len() * U32_BYTES,
        &mut forward.io_staging,
        stream,
        metadata_site,
    )?;
    upload_prefix(
        &mut device.physical_block_ids,
        &host.physical_block_ids,
        packed.physical_block_ids().len() * U32_BYTES,
        &mut forward.io_staging,
        stream,
        metadata_site,
    )?;
    upload_prefix(
        &mut device.valid_tokens,
        &host.valid_tokens,
        packed.valid_tokens().len() * U16_BYTES,
        &mut forward.io_staging,
        stream,
        metadata_site,
    )?;
    upload_prefix(
        &mut device.row_sequence_slots,
        &host.row_sequence_slots,
        active * U32_BYTES,
        &mut forward.io_staging,
        stream,
        metadata_site,
    )?;
    upload_prefix(
        &mut device.row_positions,
        &host.row_positions,
        active * U32_BYTES,
        &mut forward.io_staging,
        stream,
        metadata_site,
    )?;
    if packed.output_count() != 0 {
        let output_indices = device.output_token_indices.as_mut().ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_token_indices",
                reason: "non-empty output has no cold-prepared device index buffer",
            },
        )?;
        upload_prefix(
            output_indices,
            &host.output_token_indices,
            packed.output_count() * U32_BYTES,
            &mut forward.io_staging,
            stream,
            metadata_site,
        )?;
    }

    let batch = PackedBatchV1::new(
        host_batch,
        device_span(
            &device.sequence_block_offsets,
            CudaDType::U32,
            packed.block_row_offsets().len() * U32_BYTES,
            metadata_site,
        )?,
        device_span(
            &device.physical_block_ids,
            CudaDType::U32,
            packed.physical_block_ids().len() * U32_BYTES,
            metadata_site,
        )?,
        device_span(
            &device.valid_tokens,
            CudaDType::U16,
            packed.valid_tokens().len() * U16_BYTES,
            metadata_site,
        )?,
        device_span(
            &device.row_sequence_slots,
            CudaDType::U32,
            active * U32_BYTES,
            metadata_site,
        )?,
        device_span(
            &device.row_positions,
            CudaDType::U32,
            active * U32_BYTES,
            metadata_site,
        )?,
    )
    .map_err(|source| batch_cuda(metadata_site, source))?;

    execute_fixed_graph(
        forward,
        layout,
        key_cache,
        value_cache,
        rope_cos,
        rope_sin,
        batch,
        packed.position_ids(),
        stream,
    )?;

    if packed.output_count() != 0 {
        let output_indices = device.output_token_indices.as_ref().ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_token_indices",
                reason: "non-empty output has no cold-prepared device index buffer",
            },
        )?;
        let output =
            gathered_logits
                .as_mut()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "gathered_logits",
                    reason: "non-empty output has no cold-prepared device buffer",
                })?;
        let site = ExecutionSite::global(LlamaOp::OutputGather);
        let mut params = RowGatherParams {
            input: span(
                &forward.buffers.logits,
                CudaDType::BF16,
                forward.plan.workspace_spec().logits_bytes(),
                site,
            )?,
            row_indices: device_span(
                output_indices,
                CudaDType::U32,
                packed.output_count() * U32_BYTES,
                site,
            )?,
            row_indices_host: packed.output_token_indices(),
            output: CudaBufferSpanMut::new(
                output,
                CudaDType::BF16,
                0,
                output_logits_bytes(
                    packed.output_count(),
                    forward.plan.dimensions().vocabulary_size(),
                )?,
            )
            .map_err(|source| batch_cuda(site, source))?,
            input_row_count: usize_u64(
                bounds.max_input_tokens(),
                LlamaBatchExecutorResource::GatheredLogits,
            )?,
            column_count: usize_u64(
                forward.plan.dimensions().vocabulary_size(),
                LlamaBatchExecutorResource::GatheredLogits,
            )?,
        };
        row_gather(&mut params, stream).map_err(|source| batch_cuda(site, source))?;
    }
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value,
    clippy::similar_names
)]
fn execute_fixed_graph(
    forward: &mut PreparedLlamaForward,
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    batch: PackedBatchV1<'_>,
    positions_host: &[u32],
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    let plan = &forward.plan;
    let weights = &forward.weights;
    let gemms = &mut forward.gemms;
    let buffers = &mut forward.buffers;
    let dense_rows = usize_u64(
        plan.sequence_length(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let dimensions = plan.dimensions();
    let hidden = usize_u64(
        dimensions.hidden_size(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let key_value_width = usize_u64(
        dimensions.key_value_width(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let query_heads = usize_u64(
        dimensions.query_heads(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let key_value_heads = usize_u64(
        dimensions.key_value_heads(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let head_size = usize_u64(
        dimensions.head_dimension(),
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let max_positions = model_max_position(rope_cos, dimensions.head_dimension())?;
    let hidden_elements = plan.workspace_spec().hidden_buffer_bytes() / BF16_BYTES;
    let intermediate_elements = plan.workspace_spec().intermediate_buffer_bytes() / BF16_BYTES;

    let embedding_site = ExecutionSite::global(LlamaOp::Embedding);
    let embedding_weight = weight_span(weights, plan.embedding_weight(), embedding_site)?;
    {
        let mut params = EmbeddingParams {
            table: embedding_weight,
            token_ids: span(
                &buffers.token_ids,
                CudaDType::U32,
                plan.workspace_spec().token_ids_bytes(),
                embedding_site,
            )?,
            output: span_mut(
                &mut buffers.hidden_current,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                embedding_site,
            )?,
            error_scratch: span_mut(
                &mut buffers.embedding_error_scratch,
                CudaDType::U8,
                plan.workspace_spec().embedding_error_scratch_bytes(),
                embedding_site,
            )?,
            token_count: dense_rows,
            vocabulary_size: usize_u64(
                dimensions.vocabulary_size(),
                LlamaBatchExecutorResource::GatheredLogits,
            )?,
            hidden_size: hidden,
        };
        embedding(&mut params, stream).map_err(|source| {
            LlamaBatchExecutorError::Forward(LlamaForwardError::Embedding {
                site: embedding_site,
                source,
            })
        })?;
    }

    for layer in plan.layers() {
        let layer_index = layer.index();
        let input_norm_site = ExecutionSite::layer(layer_index, LlamaOp::InputNorm);
        let input_norm_weight = weight_span(weights, layer.input_norm_weight(), input_norm_site)?;
        {
            let mut params = RmsNormParams {
                input: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    input_norm_site,
                )?,
                weight: input_norm_weight,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    input_norm_site,
                )?,
                row_count: dense_rows,
                hidden_size: hidden,
                epsilon: layer.input_norm_epsilon(),
            };
            rms_norm(&mut params, stream).map_err(|source| batch_cuda(input_norm_site, source))?;
        }

        let query_site = ExecutionSite::layer(layer_index, LlamaOp::QueryProjection);
        execute_gemm(
            &mut gemms.hidden,
            &buffers.hidden_norm,
            weight_span(weights, layer.query_weight(), query_site)?,
            &mut buffers.hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            query_site,
        )?;
        execute_projection_bias(
            weights,
            layer.query_bias(),
            &mut buffers.hidden_projection,
            dense_rows,
            hidden,
            stream,
            query_site,
        )?;
        let key_site = ExecutionSite::layer(layer_index, LlamaOp::KeyProjection);
        execute_gemm(
            &mut gemms.key_value,
            &buffers.hidden_norm,
            weight_span(weights, layer.key_weight(), key_site)?,
            &mut buffers.key_raw,
            &mut buffers.gemm_workspace,
            stream,
            key_site,
        )?;
        execute_projection_bias(
            weights,
            layer.key_bias(),
            &mut buffers.key_raw,
            dense_rows,
            key_value_width,
            stream,
            key_site,
        )?;
        let value_site = ExecutionSite::layer(layer_index, LlamaOp::ValueProjection);
        execute_gemm(
            &mut gemms.key_value,
            &buffers.hidden_norm,
            weight_span(weights, layer.value_weight(), value_site)?,
            &mut buffers.value_raw,
            &mut buffers.gemm_workspace,
            stream,
            value_site,
        )?;
        execute_projection_bias(
            weights,
            layer.value_bias(),
            &mut buffers.value_raw,
            dense_rows,
            key_value_width,
            stream,
            value_site,
        )?;

        let query_rope_site = ExecutionSite::layer(layer_index, LlamaOp::QueryRope);
        {
            let mut params = IndexedRopeParams {
                input: span(
                    &buffers.hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    query_rope_site,
                )?,
                cos: CudaBufferSpan::new(rope_cos, CudaDType::F32, 0, rope_cos.byte_len())
                    .map_err(|source| batch_cuda(query_rope_site, source))?,
                sin: CudaBufferSpan::new(rope_sin, CudaDType::F32, 0, rope_sin.byte_len())
                    .map_err(|source| batch_cuda(query_rope_site, source))?,
                positions: batch.device_row_positions(),
                positions_host,
                output: span_mut(
                    &mut buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    query_rope_site,
                )?,
                head_count: query_heads,
                head_size,
                rotary_dimension: head_size,
                table_position_count: max_positions,
            };
            indexed_rope(&mut params, stream)
                .map_err(|source| batch_cuda(query_rope_site, source))?;
        }
        let key_rope_site = ExecutionSite::layer(layer_index, LlamaOp::KeyRope);
        {
            let mut params = IndexedRopeParams {
                input: span(
                    &buffers.key_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    key_rope_site,
                )?,
                cos: CudaBufferSpan::new(rope_cos, CudaDType::F32, 0, rope_cos.byte_len())
                    .map_err(|source| batch_cuda(key_rope_site, source))?,
                sin: CudaBufferSpan::new(rope_sin, CudaDType::F32, 0, rope_sin.byte_len())
                    .map_err(|source| batch_cuda(key_rope_site, source))?,
                positions: batch.device_row_positions(),
                positions_host,
                output: span_mut(
                    &mut buffers.key_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    key_rope_site,
                )?,
                head_count: key_value_heads,
                head_size,
                rotary_dimension: head_size,
                table_position_count: max_positions,
            };
            indexed_rope(&mut params, stream)
                .map_err(|source| batch_cuda(key_rope_site, source))?;
        }

        let cache_site = ExecutionSite::layer(layer_index, LlamaOp::KvCacheWrite);
        let layer_offset = layout.layer_byte_offset(layer_index).ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "KV layer offset",
                reason: "decoder layer lies outside the prepared KV layout",
            },
        )?;
        {
            let mut params = RaggedPagedKvCacheWriteParams {
                key_source: span(
                    &buffers.key_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    cache_site,
                )?,
                value_source: span(
                    &buffers.value_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().key_value_buffer_bytes(),
                    cache_site,
                )?,
                key_pool: CudaBufferSpanMut::new(
                    key_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(cache_site, source))?,
                value_pool: CudaBufferSpanMut::new(
                    value_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(cache_site, source))?,
                batch,
                key_value_head_count: key_value_heads,
                head_size,
            };
            ragged_paged_kv_cache_write(&mut params, stream)
                .map_err(|source| batch_cuda(cache_site, source))?;
        }

        let attention_site = ExecutionSite::layer(layer_index, LlamaOp::RaggedPagedAttention);
        {
            let mut params = RaggedPagedAttentionParams {
                query: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_site,
                )?,
                key_pool: CudaBufferSpan::new(
                    key_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                value_pool: CudaBufferSpan::new(
                    value_cache,
                    CudaDType::BF16,
                    layer_offset,
                    layout.layer_stride_bytes(),
                )
                .map_err(|source| batch_cuda(attention_site, source))?,
                output: span_mut(
                    &mut buffers.hidden_context,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_site,
                )?,
                batch,
                query_head_count: query_heads,
                key_value_head_count: key_value_heads,
                head_size,
                output_row_count: dense_rows,
                scale: 1.0 / (head_size as f32).sqrt(),
            };
            ragged_paged_attention(&mut params, stream)
                .map_err(|source| batch_cuda(attention_site, source))?;
        }

        let output_site = ExecutionSite::layer(layer_index, LlamaOp::OutputProjection);
        execute_gemm(
            &mut gemms.hidden,
            &buffers.hidden_context,
            weight_span(weights, layer.output_weight(), output_site)?,
            &mut buffers.hidden_projection,
            &mut buffers.gemm_workspace,
            stream,
            output_site,
        )?;
        execute_projection_bias(
            weights,
            layer.output_bias(),
            &mut buffers.hidden_projection,
            dense_rows,
            hidden,
            stream,
            output_site,
        )?;
        let attention_residual_site = ExecutionSite::layer(layer_index, LlamaOp::AttentionResidual);
        {
            let mut params = ResidualAddParams {
                left: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_residual_site,
                )?,
                right: span(
                    &buffers.hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_residual_site,
                )?,
                output: span_mut(
                    &mut buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    attention_residual_site,
                )?,
                element_count: hidden_elements,
            };
            residual_add(&mut params, stream)
                .map_err(|source| batch_cuda(attention_residual_site, source))?;
        }

        let post_norm_site = ExecutionSite::layer(layer_index, LlamaOp::PostAttentionNorm);
        {
            let mut params = RmsNormParams {
                input: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    post_norm_site,
                )?,
                weight: weight_span(weights, layer.post_attention_norm_weight(), post_norm_site)?,
                output: span_mut(
                    &mut buffers.hidden_norm,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    post_norm_site,
                )?,
                row_count: dense_rows,
                hidden_size: hidden,
                epsilon: layer.post_attention_norm_epsilon(),
            };
            rms_norm(&mut params, stream).map_err(|source| batch_cuda(post_norm_site, source))?;
        }
        let gate_site = ExecutionSite::layer(layer_index, LlamaOp::GateProjection);
        execute_gemm(
            &mut gemms.intermediate,
            &buffers.hidden_norm,
            weight_span(weights, layer.gate_weight(), gate_site)?,
            &mut buffers.gate_raw,
            &mut buffers.gemm_workspace,
            stream,
            gate_site,
        )?;
        let up_site = ExecutionSite::layer(layer_index, LlamaOp::UpProjection);
        execute_gemm(
            &mut gemms.intermediate,
            &buffers.hidden_norm,
            weight_span(weights, layer.up_weight(), up_site)?,
            &mut buffers.up_raw,
            &mut buffers.gemm_workspace,
            stream,
            up_site,
        )?;
        let silu_site = ExecutionSite::layer(layer_index, LlamaOp::Silu);
        {
            let mut params = SiluParams {
                input: span(
                    &buffers.gate_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    silu_site,
                )?,
                output: span_mut(
                    &mut buffers.gate_activated,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    silu_site,
                )?,
                element_count: intermediate_elements,
            };
            silu(&mut params, stream).map_err(|source| batch_cuda(silu_site, source))?;
        }
        let gated_site = ExecutionSite::layer(layer_index, LlamaOp::GatedMultiply);
        {
            let mut params = GatedMultiplyParams {
                activated_gate: span(
                    &buffers.gate_activated,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                up: span(
                    &buffers.up_raw,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                output: span_mut(
                    &mut buffers.gated_product,
                    CudaDType::BF16,
                    plan.workspace_spec().intermediate_buffer_bytes(),
                    gated_site,
                )?,
                element_count: intermediate_elements,
            };
            gated_multiply(&mut params, stream).map_err(|source| batch_cuda(gated_site, source))?;
        }
        let down_site = ExecutionSite::layer(layer_index, LlamaOp::DownProjection);
        execute_gemm(
            &mut gemms.down,
            &buffers.gated_product,
            weight_span(weights, layer.down_weight(), down_site)?,
            &mut buffers.hidden_current,
            &mut buffers.gemm_workspace,
            stream,
            down_site,
        )?;
        let mlp_residual_site = ExecutionSite::layer(layer_index, LlamaOp::MlpResidual);
        {
            let mut params = ResidualAddParams {
                left: span(
                    &buffers.hidden_rotary,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                right: span(
                    &buffers.hidden_current,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                output: span_mut(
                    &mut buffers.hidden_projection,
                    CudaDType::BF16,
                    plan.workspace_spec().hidden_buffer_bytes(),
                    mlp_residual_site,
                )?,
                element_count: hidden_elements,
            };
            residual_add(&mut params, stream)
                .map_err(|source| batch_cuda(mlp_residual_site, source))?;
        }
        mem::swap(&mut buffers.hidden_current, &mut buffers.hidden_projection);
    }

    let final_norm_site = ExecutionSite::global(LlamaOp::FinalNorm);
    {
        let mut params = RmsNormParams {
            input: span(
                &buffers.hidden_current,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                final_norm_site,
            )?,
            weight: weight_span(weights, plan.final_norm_weight(), final_norm_site)?,
            output: span_mut(
                &mut buffers.hidden_norm,
                CudaDType::BF16,
                plan.workspace_spec().hidden_buffer_bytes(),
                final_norm_site,
            )?,
            row_count: dense_rows,
            hidden_size: hidden,
            epsilon: plan.final_norm_epsilon(),
        };
        rms_norm(&mut params, stream).map_err(|source| batch_cuda(final_norm_site, source))?;
    }
    let lm_head_site = ExecutionSite::global(LlamaOp::LmHead);
    execute_gemm(
        &mut gemms.lm_head,
        &buffers.hidden_norm,
        weight_span(weights, plan.lm_head_weight(), lm_head_site)?,
        &mut buffers.logits,
        &mut buffers.gemm_workspace,
        stream,
        lm_head_site,
    )?;
    Ok(())
}
// HOT_BATCH_EXECUTE_END

#[allow(clippy::large_types_passed_by_value)]
fn validate_for_execution(
    packed: LlamaPackedBatchMetadata<'_>,
    vocabulary_size: usize,
    maximum_position_count: u64,
    config: PreparedLlamaBatchExecutorConfig,
) -> LlamaBatchExecutorResult<()> {
    if packed.schema_version() != LLAMA_BATCH_METADATA_V1_VERSION {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "schema_version",
            reason: "packed metadata version differs from the executor contract",
        });
    }
    if packed.total_input_tokens() > config.metadata.max_input_tokens()
        || packed.row_count() > config.metadata.max_rows()
        || packed.physical_block_ids().len() > config.metadata.max_block_entries()
        || packed.output_count() > config.metadata.max_output_slots()
    {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "capacity",
            reason: "packed metadata exceeds the executor's cold bounds",
        });
    }
    for (position, &token_id) in packed.input_token_ids().iter().enumerate() {
        if usize::try_from(token_id)
            .ok()
            .is_none_or(|token| token >= vocabulary_size)
        {
            return Err(LlamaBatchExecutorError::TokenOutOfRange {
                position,
                token_id,
                vocabulary_size,
            });
        }
    }
    for (&row, &position) in packed
        .row_sequence_slots()
        .iter()
        .zip(packed.position_ids())
    {
        if u64::from(position) >= maximum_position_count {
            return Err(LlamaBatchExecutorError::PositionOutOfRange {
                row: usize::try_from(row).unwrap_or(usize::MAX),
                position,
                maximum: usize::try_from(maximum_position_count).unwrap_or(usize::MAX),
            });
        }
    }
    Ok(())
}

fn allocate_device_metadata(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<BatchDeviceMetadata> {
    let offsets =
        bounds
            .max_rows()
            .checked_add(1)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            })?;
    let allocate = |elements: usize,
                    element_bytes: usize,
                    resource: LlamaBatchExecutorResource|
     -> LlamaBatchExecutorResult<CudaDeviceBuffer> {
        let bytes = checked_host_byte_len(elements, element_bytes, resource)?;
        allocate_device(
            context,
            usize_u64(bytes, resource)?,
            ExecutionSite::global(LlamaOp::BatchMetadataUpload),
        )
    };
    Ok(BatchDeviceMetadata {
        sequence_block_offsets: allocate(
            offsets,
            U32_BYTES,
            LlamaBatchExecutorResource::SequenceBlockOffsets,
        )?,
        physical_block_ids: allocate(
            bounds.max_block_entries(),
            U32_BYTES,
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?,
        valid_tokens: allocate(
            bounds.max_block_entries(),
            U16_BYTES,
            LlamaBatchExecutorResource::ValidTokens,
        )?,
        row_sequence_slots: allocate(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowSequenceSlots,
        )?,
        row_positions: allocate(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowPositions,
        )?,
        output_token_indices: if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate(
                bounds.max_output_slots(),
                U32_BYTES,
                LlamaBatchExecutorResource::OutputTokenIndices,
            )?)
        },
    })
}

fn allocate_host_workspace(
    bounds: LlamaBatchMetadataConfig,
) -> LlamaBatchExecutorResult<BatchHostWorkspace> {
    let offsets =
        bounds
            .max_rows()
            .checked_add(1)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            })?;
    Ok(BatchHostWorkspace {
        padded_tokens: allocate_zeroed_u32(bounds.max_input_tokens())?,
        sequence_block_offsets: allocate_zeroed_bytes(offsets, U32_BYTES)?,
        physical_block_ids: allocate_zeroed_bytes(bounds.max_block_entries(), U32_BYTES)?,
        valid_tokens: allocate_zeroed_bytes(bounds.max_block_entries(), U16_BYTES)?,
        row_sequence_slots: allocate_zeroed_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        row_positions: allocate_zeroed_bytes(bounds.max_input_tokens(), U32_BYTES)?,
        output_token_indices: allocate_zeroed_bytes(bounds.max_output_slots(), U32_BYTES)?,
    })
}

fn allocate_zeroed_u32(elements: usize) -> LlamaBatchExecutorResult<Box<[u32]>> {
    let requested_bytes = checked_host_byte_len(
        elements,
        U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| LlamaBatchExecutorError::HostAllocation {
            resource: LlamaBatchExecutorResource::HostWorkspace,
            requested_bytes: requested_bytes as u64,
        })?;
    values.resize(elements, 0);
    Ok(values.into_boxed_slice())
}

fn allocate_zeroed_bytes(
    elements: usize,
    element_bytes: usize,
) -> LlamaBatchExecutorResult<Box<[u8]>> {
    let requested = checked_host_byte_len(
        elements,
        element_bytes,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(requested)
        .map_err(|_| LlamaBatchExecutorError::HostAllocation {
            resource: LlamaBatchExecutorResource::HostWorkspace,
            requested_bytes: requested as u64,
        })?;
    bytes.resize(requested, 0);
    Ok(bytes.into_boxed_slice())
}

type RopeTableBytes = (Box<[u8]>, Box<[u8]>);

#[allow(clippy::cast_precision_loss)]
fn build_absolute_rope_tables(
    position_count: usize,
    head_dimension: usize,
    theta: f32,
) -> LlamaBatchExecutorResult<RopeTableBytes> {
    let half = head_dimension / 2;
    let elements =
        position_count
            .checked_mul(half)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeCos,
            })?;
    let mut cos = allocate_zeroed_bytes(elements, F32_BYTES_USIZE)?;
    let mut sin = allocate_zeroed_bytes(elements, F32_BYTES_USIZE)?;
    for position in 0..position_count {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let (sine, cosine) = angle.sin_cos();
            let byte_offset = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(F32_BYTES_USIZE))
                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::RopeCos,
                })?;
            cos[byte_offset..byte_offset + F32_BYTES_USIZE].copy_from_slice(&cosine.to_ne_bytes());
            sin[byte_offset..byte_offset + F32_BYTES_USIZE].copy_from_slice(&sine.to_ne_bytes());
        }
    }
    Ok((cos, sin))
}

fn build_batch_allocation_report(
    forward: PreparedLlamaAllocationReport,
    bounds: LlamaBatchMetadataConfig,
    layout: KvLayout,
    rope_bytes_per_kind: u64,
    gathered_logits_capacity_bytes: u64,
    host: &BatchHostWorkspace,
) -> LlamaBatchExecutorResult<PreparedLlamaBatchAllocationReport> {
    let packed_metadata_device_bytes = [
        host.sequence_block_offsets.len(),
        host.physical_block_ids.len(),
        host.valid_tokens.len(),
        host.row_sequence_slots.len(),
        host.row_positions.len(),
        host.output_token_indices.len(),
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(bytes as u64)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })
    })?;
    let rope_table_bytes =
        rope_bytes_per_kind
            .checked_mul(2)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeSin,
            })?;
    let additional_device_bytes = layout
        .total_bytes()
        .checked_add(rope_table_bytes)
        .and_then(|bytes| bytes.checked_add(packed_metadata_device_bytes))
        .and_then(|bytes| bytes.checked_add(gathered_logits_capacity_bytes))
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let total_device_bytes = forward
        .total_device_bytes()
        .checked_add(additional_device_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let output_allocations = u64::from(bounds.max_output_slots() != 0) * 2;
    let additional_device_allocation_count = BASE_ADDITIONAL_DEVICE_ALLOCATIONS
        .checked_add(output_allocations)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let total_device_allocation_count = forward
        .device_allocation_count()
        .checked_add(additional_device_allocation_count)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let host_workspace_bytes = [
        host.padded_tokens.len().checked_mul(U32_BYTES),
        Some(host.sequence_block_offsets.len()),
        Some(host.physical_block_ids.len()),
        Some(host.valid_tokens.len()),
        Some(host.row_sequence_slots.len()),
        Some(host.row_positions.len()),
        Some(host.output_token_indices.len()),
    ]
    .into_iter()
    .try_fold(0_usize, |total, bytes| {
        total
            .checked_add(bytes.ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })?)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })
    })? as u64;
    Ok(PreparedLlamaBatchAllocationReport {
        forward,
        kv_cache_bytes: layout.total_bytes(),
        rope_table_bytes,
        packed_metadata_device_bytes,
        gathered_logits_capacity_bytes,
        additional_device_bytes,
        total_device_bytes,
        additional_device_allocation_count,
        total_device_allocation_count,
        host_workspace_bytes,
    })
}

fn model_max_position(
    rope_cos: &CudaDeviceBuffer,
    head_dimension: usize,
) -> LlamaBatchExecutorResult<u64> {
    let row_bytes = usize_u64(head_dimension / 2, LlamaBatchExecutorResource::RopeCos)?
        .checked_mul(F32_BYTES)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::RopeCos,
        })?;
    Ok(rope_cos.byte_len() / row_bytes)
}

fn output_logits_bytes(outputs: usize, vocabulary: usize) -> LlamaBatchExecutorResult<u64> {
    checked_product_u64(
        &[
            usize_u64(outputs, LlamaBatchExecutorResource::GatheredLogits)?,
            usize_u64(vocabulary, LlamaBatchExecutorResource::GatheredLogits)?,
            BF16_BYTES,
        ],
        LlamaBatchExecutorResource::GatheredLogits,
    )
}

fn allocate_device(
    context: &CudaContext,
    byte_len: u64,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaDeviceBuffer> {
    context
        .allocate_device_buffer(byte_len)
        .map_err(|source| batch_cuda(site, source))
}

fn upload_prefix(
    destination: &mut CudaDeviceBuffer,
    source: &[u8],
    byte_len: usize,
    staging: &mut rustinfer_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    destination
        .upload_from_slice(0, &source[..byte_len], staging, stream)
        .map_err(|source| batch_cuda(site, source))
}

fn device_span(
    buffer: &CudaDeviceBuffer,
    dtype: CudaDType,
    byte_len: usize,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<CudaBufferSpan<'_>> {
    CudaBufferSpan::new(
        buffer,
        dtype,
        0,
        usize_u64(byte_len, LlamaBatchExecutorResource::HostWorkspace)?,
    )
    .map_err(|source| batch_cuda(site, source))
}

fn encode_u32(source: &[u32], destination: &mut [u8]) {
    for (value, bytes) in source.iter().zip(destination.chunks_exact_mut(U32_BYTES)) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
}

fn encode_u16(source: &[u16], destination: &mut [u8]) {
    for (value, bytes) in source.iter().zip(destination.chunks_exact_mut(U16_BYTES)) {
        bytes.copy_from_slice(&value.to_ne_bytes());
    }
}

fn batch_cuda(site: ExecutionSite, source: CudaError) -> LlamaBatchExecutorError {
    LlamaBatchExecutorError::Cuda { site, source }
}

fn poison_for_batch_error(
    poisoned: &mut bool,
    forward: &mut PreparedLlamaForward,
    error: &LlamaBatchExecutorError,
) {
    match error {
        LlamaBatchExecutorError::Cuda { source, .. } => {
            poison_for_cuda_error(poisoned, source);
            poison_for_cuda_error(&mut forward.poisoned, source);
        }
        LlamaBatchExecutorError::Forward(source) => {
            poison_for_forward_error(&mut forward.poisoned, source);
            *poisoned |= forward.poisoned || forward.gemms.any_poisoned();
        }
        LlamaBatchExecutorError::InvalidConfiguration { .. }
        | LlamaBatchExecutorError::ArithmeticOverflow { .. } => {
            *poisoned = true;
            forward.poisoned = true;
        }
        _ => {}
    }
}

fn record_close(
    first: &mut Option<LlamaBatchExecutorError>,
    resource: LlamaBatchExecutorResource,
    result: Result<(), CudaError>,
) {
    if let Err(source) = result {
        if first.is_none() {
            *first = Some(LlamaBatchExecutorError::Cleanup { resource, source });
        }
    }
}

fn checked_host_byte_len(
    elements: usize,
    element_bytes: usize,
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<usize> {
    elements
        .checked_mul(element_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })
}

fn usize_u64(value: usize, resource: LlamaBatchExecutorResource) -> LlamaBatchExecutorResult<u64> {
    u64::try_from(value).map_err(|_| LlamaBatchExecutorError::ArithmeticOverflow { resource })
}

fn checked_product_u64(
    values: &[u64],
    resource: LlamaBatchExecutorResource,
) -> LlamaBatchExecutorResult<u64> {
    values.iter().try_fold(1_u64, |product, &value| {
        product
            .checked_mul(value)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow { resource })
    })
}

const _: () = assert!(KV_BLOCK_SIZE == 16);
const _: () = assert!(SUPPORTED_HEAD_DIMENSION == 64);
