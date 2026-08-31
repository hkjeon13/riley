//! Owning shape-bucketed CUDA executor for mixed Llama prefill/decode batches.
//!
//! One cold-prepared owner shares uploaded weights, maximum-size graph buffers,
//! and paged KV storage across exact-`M` execution-plan/GEMM variants. The
//! rollback policy keeps `M = max_input_tokens`; the active-row policy selects
//! the smallest prepared power-of-two bucket that contains the `T` flattened
//! input rows. Indexed `RoPE`, paged KV scatter, and ragged causal attention
//! preserve each active row's absolute sequence position.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use std::fmt;
use std::mem;

use riley_cuda::{
    AttentionReductionProfile, Bf16ArgmaxParams, CudaBufferSpan, CudaBufferSpanMut, CudaContext,
    CudaDType, CudaDeviceBuffer, CudaError, CudaExecutionStream, CudaStream, EmbeddingParams,
    FIXED37_RAGGED_MAX_LOGICAL_TOKENS, GatedMultiplyParams, IndexedRopeParams, PackedBatchHostV1,
    PackedBatchV1, RaggedPagedAttentionParams, RaggedPagedKvCacheWriteParams, ResidualAddParams,
    ResidualRmsNormParams, RmsNormParams, RopeTableParams, RowGatherParams, SiluParams,
    deterministic_bf16_argmax, embedding, fixed37_ragged_paged_attention, gated_multiply,
    grouped_ragged_paged_attention, indexed_rope, ragged_paged_attention,
    ragged_paged_kv_cache_write, residual_add, rope_table, row_gather, silu,
};
use riley_model::LoadedModel;

use super::batch::{
    LLAMA_BATCH_METADATA_V1_VERSION, LlamaBatchMetadataConfig, LlamaBatchRow,
    LlamaPackedBatchMetadata, PreparedLlamaBatchMetadata,
};
use super::executor::buffers::{
    BatchDeviceInput, BatchHostInput, U16_BYTES, U32_BYTES, allocate_packed_device_input,
    allocate_packed_host_input, allocate_synchronous_device_input, allocate_synchronous_host_input,
    close_device_input, close_host_input,
};
use super::executor::device_views::{packed_device_views, per_operation_device_views};
pub use super::executor::error::{
    LlamaBatchExecutorError, LlamaBatchExecutorResource, LlamaBatchExecutorResult,
};
use super::executor::gemm_plan::{PreparedLlamaBatchShape, prepare_shape_variants};
use super::executor::metadata::{
    PackedIterationLayout, encode_u16, encode_u32, pack_iteration_input,
};
pub use super::executor::metrics::{LlamaBatchShapeBucketHit, LlamaBatchShapeObservation};
use super::executor::output::{GREEDY_RESULT_BYTES, decode_greedy_tokens};
use super::executor::shape::{
    LlamaBatchShapeBuckets, LlamaBatchShapeHistory, batch_shape_policy_id,
    select_smallest_prepared_dense_rows,
};
pub use super::executor::shape::{LlamaBatchShapePolicy, MAX_LLAMA_BATCH_SHAPE_BUCKETS};
use super::forward::{
    ForwardBuffers, GemmPlans, LlamaForwardError, LlamaRmsNormProfile, LlamaRopeTableProfile,
    PreparedLlamaAllocationReport, PreparedLlamaForward, PreparedLlamaForwardConfig, execute_gemm,
    execute_profile_residual_rms_norm, execute_profile_rms_norm, execute_projection_bias,
    poison_for_cuda_error, poison_for_forward_error, span, span_mut, weight_span,
};
use super::{ExecutionSite, LlamaExecutionPlan, LlamaOp, LlamaReductionProfile};
use crate::cuda_weights::CudaUploadedWeights;
use crate::paged_kv::{KV_BLOCK_SIZE, KvLayout};

const BF16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;
const BF16_BYTES_USIZE: usize = 2;
const F32_BYTES_USIZE: usize = 4;
const SUPPORTED_HEAD_DIMENSION: usize = 64;
const PER_OPERATION_BASE_DEVICE_ALLOCATIONS: u64 = 9;
const ITERATION_BATCH_BASE_DEVICE_ALLOCATIONS: u64 = 5;
const RAGGED_PAGED_ATTENTION_LEGACY_D64_V1: &str =
    "riley.cuda.ragged-paged-attention.legacy-d64-v1";
const RAGGED_PAGED_ATTENTION_GROUPED_HEADS_D64_V1: &str =
    "riley.cuda.ragged-paged-attention.grouped-heads-d64-v1";
const RAGGED_PAGED_ATTENTION_FIXED37_TWO_PASS_D64_S8192_V1: &str =
    "riley.cuda.ragged-paged-attention.fixed37-two-pass-d64-s8192-v1";

/// Exact implementation selected for the attention residual/post-norm pair.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResidualNormImplementation {
    /// Standalone residual add followed by standalone `RMSNorm`.
    #[default]
    Separate,
    /// One exact fused residual-add plus `RMSNorm` primitive.
    Fused,
}

/// Completion boundary selected for one fixed-graph batch iteration.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionCompletionImplementation {
    /// Preserve the established primitive-local completion boundary.
    #[default]
    PerOperation,
    /// Submit the fixed graph and optional output gather under one completion guard.
    IterationBatch,
}

/// Host-to-device transport for tokens and packed batch metadata.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum BatchMetadataTransport {
    /// Preserve the established token-plus-six-metadata synchronous uploads.
    #[default]
    Synchronous,
    /// Pack tokens and all six metadata arrays into one aligned pinned slab and
    /// enqueue one stream-ordered H2D copy inside iteration completion.
    PackedAsync,
}

/// Launch implementation selected for canonical ragged paged attention.
///
/// Both variants preserve the canonical per-head online-softmax reduction
/// contract. The grouped variant places several query-head warps in one CTA
/// to reduce the M=1 decode launch count; the legacy variant preserves the
/// established one-warp-per-head launch geometry as the rollback default.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RaggedAttentionImplementation {
    /// Preserve the established one warp per `(row, query-head)` launch.
    #[default]
    Legacy,
    /// Reuse staged K/V tiles across the canonical query-head warps of a GQA
    /// key/value head, with a bounded grouped-head fallback for other shapes.
    GroupedHeads,
}

const fn execution_completion_implementation_id(
    implementation: ExecutionCompletionImplementation,
) -> &'static str {
    match implementation {
        ExecutionCompletionImplementation::PerOperation => "per-operation",
        ExecutionCompletionImplementation::IterationBatch => "iteration-batch",
    }
}

const fn batch_metadata_transport_id(transport: BatchMetadataTransport) -> &'static str {
    match transport {
        BatchMetadataTransport::Synchronous => "synchronous",
        BatchMetadataTransport::PackedAsync => "packed-async",
    }
}

const fn residual_norm_implementation_id(
    implementation: ResidualNormImplementation,
) -> &'static str {
    match implementation {
        ResidualNormImplementation::Separate => "separate",
        ResidualNormImplementation::Fused => "fused",
    }
}

const fn ragged_attention_implementation_id(
    profile: AttentionReductionProfile,
    implementation: RaggedAttentionImplementation,
) -> &'static str {
    match (profile, implementation) {
        (AttentionReductionProfile::CanonicalV1, RaggedAttentionImplementation::Legacy) => {
            RAGGED_PAGED_ATTENTION_LEGACY_D64_V1
        }
        (AttentionReductionProfile::CanonicalV1, RaggedAttentionImplementation::GroupedHeads) => {
            RAGGED_PAGED_ATTENTION_GROUPED_HEADS_D64_V1
        }
        (AttentionReductionProfile::FixedContiguous37BalancedV1, _) => {
            RAGGED_PAGED_ATTENTION_FIXED37_TWO_PASS_D64_S8192_V1
        }
    }
}

const fn runtime_selection_policy_id(profile: LlamaReductionProfile) -> &'static str {
    match profile {
        LlamaReductionProfile::CanonicalV1 => "exact-fallback-allowed",
        LlamaReductionProfile::FixedContiguous37BalancedV1 => "fail-closed",
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BatchOutputMode {
    Logits,
    GreedyTokens,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum BatchDispatchDisposition {
    #[default]
    PreDispatch,
    CommandSubmissionStarted,
}

impl BatchDispatchDisposition {
    const fn mutation_may_have_occurred(self) -> bool {
        matches!(self, Self::CommandSubmissionStarted)
    }
}

/// Cold bounds and shape policy for one reusable continuous-batch owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaBatchExecutorConfig {
    metadata: LlamaBatchMetadataConfig,
    forward: PreparedLlamaForwardConfig,
    ragged_attention_reduction_profile: AttentionReductionProfile,
    ragged_attention_implementation: RaggedAttentionImplementation,
    residual_norm: ResidualNormImplementation,
    execution_completion: ExecutionCompletionImplementation,
    metadata_transport: BatchMetadataTransport,
    shape_policy: LlamaBatchShapePolicy,
    shape_buckets: LlamaBatchShapeBuckets,
}

impl PreparedLlamaBatchExecutorConfig {
    #[must_use]
    pub const fn new(
        metadata: LlamaBatchMetadataConfig,
        forward: PreparedLlamaForwardConfig,
    ) -> Self {
        Self {
            metadata,
            forward,
            ragged_attention_reduction_profile: forward.reduction_profile().attention_profile(),
            ragged_attention_implementation: RaggedAttentionImplementation::Legacy,
            residual_norm: ResidualNormImplementation::Separate,
            execution_completion: ExecutionCompletionImplementation::PerOperation,
            metadata_transport: BatchMetadataTransport::Synchronous,
            shape_policy: LlamaBatchShapePolicy::FixedMaximum,
            shape_buckets: LlamaBatchShapeBuckets::automatic(metadata.max_input_tokens()),
        }
    }

    #[must_use]
    pub const fn metadata(self) -> LlamaBatchMetadataConfig {
        self.metadata
    }

    #[must_use]
    pub const fn forward(self) -> PreparedLlamaForwardConfig {
        self.forward
    }

    /// Selects one complete reduction implementation without cross-profile fallback.
    #[must_use]
    pub const fn with_reduction_profile(mut self, profile: LlamaReductionProfile) -> Self {
        self.forward = self.forward.with_reduction_profile(profile);
        self.ragged_attention_reduction_profile = profile.attention_profile();
        self
    }

    /// Selects every established canonical reduction implementation.
    #[must_use]
    pub const fn with_canonical_reductions(self) -> Self {
        self.with_reduction_profile(LlamaReductionProfile::CanonicalV1)
    }

    /// Selects the complete fixed-contiguous-37 balanced reduction profile.
    #[must_use]
    pub const fn with_fixed37_reductions(self) -> Self {
        self.with_reduction_profile(LlamaReductionProfile::FixedContiguous37BalancedV1)
    }

    /// Returns the forward/decode reduction profile used as the whole-profile source.
    ///
    /// The compatibility-only ragged attention builders can deliberately make
    /// that one primitive differ. Call [`Self::reduction_profile_is_coherent`]
    /// before labeling the executor as a complete whole-profile run.
    #[must_use]
    pub const fn reduction_profile(self) -> LlamaReductionProfile {
        self.forward.reduction_profile()
    }

    /// Whether ragged attention still matches the whole-profile source.
    #[must_use]
    pub fn reduction_profile_is_coherent(self) -> bool {
        self.ragged_attention_reduction_profile
            == self.forward.reduction_profile().attention_profile()
    }

    /// Selects the reduction profile used by ragged paged attention.
    #[must_use]
    pub const fn with_ragged_attention_reduction_profile(
        mut self,
        profile: AttentionReductionProfile,
    ) -> Self {
        self.ragged_attention_reduction_profile = profile;
        self
    }

    /// Selects the existing canonical ragged online-softmax implementation.
    #[must_use]
    pub const fn with_canonical_ragged_attention(mut self) -> Self {
        self.ragged_attention_reduction_profile = AttentionReductionProfile::CanonicalV1;
        self
    }

    /// Selects fixed37 no-HBM two-pass ragged attention.
    ///
    /// Execution rejects logical prefixes above 8192 during host preflight,
    /// before device metadata upload or paged-KV mutation.
    #[must_use]
    pub const fn with_fixed37_ragged_attention(mut self) -> Self {
        self.ragged_attention_reduction_profile =
            AttentionReductionProfile::FixedContiguous37BalancedV1;
        self
    }

    #[must_use]
    pub const fn ragged_attention_reduction_profile(self) -> AttentionReductionProfile {
        self.ragged_attention_reduction_profile
    }

    /// Selects the canonical GQA shared-K/V ragged attention launch.
    ///
    /// The selection applies only to [`AttentionReductionProfile::CanonicalV1`].
    /// Fixed37 retains its separately specified two-pass implementation.
    #[must_use]
    pub const fn with_grouped_ragged_attention_heads(mut self) -> Self {
        self.ragged_attention_implementation = RaggedAttentionImplementation::GroupedHeads;
        self
    }

    /// Restores the established one-warp-per-head ragged attention launch.
    #[must_use]
    pub const fn with_legacy_ragged_attention_heads(mut self) -> Self {
        self.ragged_attention_implementation = RaggedAttentionImplementation::Legacy;
        self
    }

    /// Returns the canonical ragged attention launch implementation.
    #[must_use]
    pub const fn ragged_attention_implementation(self) -> RaggedAttentionImplementation {
        self.ragged_attention_implementation
    }

    /// Selects the exact fused attention residual/post-norm implementation.
    #[must_use]
    pub const fn with_fused_residual_norm(mut self) -> Self {
        self.residual_norm = ResidualNormImplementation::Fused;
        self
    }

    /// Selects the exact standalone rollback implementation.
    #[must_use]
    pub const fn with_separate_residual_norm(mut self) -> Self {
        self.residual_norm = ResidualNormImplementation::Separate;
        self
    }

    #[must_use]
    pub const fn residual_norm_implementation(self) -> ResidualNormImplementation {
        self.residual_norm
    }

    /// Selects the established primitive-local completion boundary.
    #[must_use]
    pub const fn with_per_operation_completion(mut self) -> Self {
        self.execution_completion = ExecutionCompletionImplementation::PerOperation;
        self
    }

    /// Selects one completion boundary for the fixed graph and output gather.
    #[must_use]
    pub const fn with_iteration_batch_completion(mut self) -> Self {
        self.execution_completion = ExecutionCompletionImplementation::IterationBatch;
        self
    }

    #[must_use]
    pub const fn execution_completion_implementation(self) -> ExecutionCompletionImplementation {
        self.execution_completion
    }

    /// Selects the opt-in one-copy pinned metadata transport.
    ///
    /// Packed async requires iteration-batch completion and is rejected during
    /// cold preparation when paired with per-operation completion.
    #[must_use]
    pub const fn with_packed_async_metadata(mut self) -> Self {
        self.metadata_transport = BatchMetadataTransport::PackedAsync;
        self
    }

    /// Restores the established synchronous token-plus-metadata uploads.
    #[must_use]
    pub const fn with_synchronous_metadata(mut self) -> Self {
        self.metadata_transport = BatchMetadataTransport::Synchronous;
        self
    }

    #[must_use]
    pub const fn metadata_transport(self) -> BatchMetadataTransport {
        self.metadata_transport
    }

    fn validate_metadata_transport(self) -> LlamaBatchExecutorResult<()> {
        if self.metadata_transport == BatchMetadataTransport::PackedAsync
            && self.execution_completion != ExecutionCompletionImplementation::IterationBatch
        {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                reason: "packed async metadata requires iteration-batch completion",
            });
        }
        Ok(())
    }

    /// Enables exact active-row power-of-two execution shapes.
    #[must_use]
    pub const fn with_active_row_buckets(mut self) -> Self {
        self.shape_policy = LlamaBatchShapePolicy::ActiveRowBuckets;
        self.shape_buckets = LlamaBatchShapeBuckets::automatic(self.metadata.max_input_tokens());
        self
    }

    /// Enables exact active-row execution with a caller-supplied cold bucket list.
    ///
    /// The list must start at one, be strictly increasing, contain no more
    /// than [`MAX_LLAMA_BATCH_SHAPE_BUCKETS`] entries, and end at exactly the
    /// configured `max_input_tokens` value.
    ///
    /// # Errors
    ///
    /// Returns before changing the configuration when any bucket invariant is
    /// violated.
    pub fn with_custom_active_row_buckets(
        mut self,
        buckets: &[usize],
    ) -> LlamaBatchExecutorResult<Self> {
        self.shape_buckets =
            LlamaBatchShapeBuckets::custom(buckets, self.metadata.max_input_tokens())?;
        self.shape_policy = LlamaBatchShapePolicy::ActiveRowBuckets;
        Ok(self)
    }

    /// Restores the established fixed-maximum rollback graph.
    #[must_use]
    pub const fn with_fixed_maximum_shape(mut self) -> Self {
        self.shape_policy = LlamaBatchShapePolicy::FixedMaximum;
        self
    }

    #[must_use]
    pub const fn shape_policy(self) -> LlamaBatchShapePolicy {
        self.shape_policy
    }

    /// Returns the cold-configured active-row bucket list.
    ///
    /// Fixed-maximum mode ignores this list and executes only the maximum
    /// shape. Re-enabling active-row mode rebuilds the automatic list unless a
    /// custom list is supplied explicitly.
    #[must_use]
    pub const fn configured_shape_buckets(&self) -> &[usize] {
        self.shape_buckets.as_slice()
    }

    /// Selects the exact dense row count for one prospective active batch.
    ///
    /// # Errors
    ///
    /// Returns when `active_rows` is empty or exceeds the metadata capacity.
    pub fn select_dense_rows(self, active_rows: usize) -> LlamaBatchExecutorResult<usize> {
        if self.shape_policy == LlamaBatchShapePolicy::FixedMaximum {
            self.shape_policy
                .select_dense_rows(active_rows, self.metadata.max_input_tokens())
        } else {
            self.shape_buckets.select(active_rows)
        }
    }
}

fn shape_history_for_config(
    config: PreparedLlamaBatchExecutorConfig,
) -> LlamaBatchExecutorResult<LlamaBatchShapeHistory> {
    LlamaBatchShapeHistory::new(
        config.shape_policy,
        config.shape_buckets.as_slice(),
        config.metadata.max_input_tokens(),
    )
}

/// Exact owned allocation totals after cold batch preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedLlamaBatchAllocationReport {
    forward: PreparedLlamaAllocationReport,
    kv_cache_bytes: u64,
    rope_table_bytes: u64,
    packed_metadata_device_bytes: u64,
    batch_input_device_bytes: u64,
    gathered_logits_capacity_bytes: u64,
    greedy_result_capacity_bytes: u64,
    additional_device_bytes: u64,
    total_device_bytes: u64,
    additional_device_allocation_count: u64,
    total_device_allocation_count: u64,
    host_workspace_bytes: u64,
    total_pinned_host_bytes: u64,
    total_pinned_host_allocation_count: u64,
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

    /// Device bytes owned by the selected batch-input transport.
    ///
    /// Per-operation completion owns the six established metadata buffers;
    /// iteration completion owns one aligned token-plus-metadata slab.
    #[must_use]
    pub const fn batch_input_device_bytes(self) -> u64 {
        self.batch_input_device_bytes
    }

    #[must_use]
    pub const fn gathered_logits_capacity_bytes(self) -> u64 {
        self.gathered_logits_capacity_bytes
    }

    #[must_use]
    pub const fn greedy_result_capacity_bytes(self) -> u64 {
        self.greedy_result_capacity_bytes
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
        self.total_pinned_host_bytes
    }

    #[must_use]
    pub const fn pinned_host_allocation_count(self) -> u64 {
        self.total_pinned_host_allocation_count
    }
}

struct BatchHostWorkspace {
    input: BatchHostInput,
    greedy_results: Box<[u8]>,
}

/// Shape-bucketed, shared-KV Llama continuous-batch executor.
///
/// The scheduler retains ownership of logical reservations. A successful call
/// only establishes that every synchronous native operation completed; the
/// caller may commit the matching scheduler iteration after `execute` returns.
/// A failed native operation poisons this owner and the caller must abort the
/// iteration instead of publishing any partial KV writes.
pub struct PreparedLlamaBatchExecutor {
    config: PreparedLlamaBatchExecutorConfig,
    shape_history: LlamaBatchShapeHistory,
    metadata: PreparedLlamaBatchMetadata,
    forward: PreparedLlamaForward,
    shape_variants: Box<[PreparedLlamaBatchShape]>,
    layout: KvLayout,
    key_cache: CudaDeviceBuffer,
    value_cache: CudaDeviceBuffer,
    absolute_rope_cos: CudaDeviceBuffer,
    absolute_rope_sin: CudaDeviceBuffer,
    device_input: BatchDeviceInput,
    gathered_logits: Option<CudaDeviceBuffer>,
    greedy_results: Option<CudaDeviceBuffer>,
    host: BatchHostWorkspace,
    allocation_report: PreparedLlamaBatchAllocationReport,
    output_count: usize,
    output_mode: BatchOutputMode,
    output_ready: bool,
    poisoned: bool,
}

impl fmt::Debug for PreparedLlamaBatchExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaBatchExecutor")
            .field("config", &self.config)
            .field("shape_history", &self.shape_history)
            .field("shape_variant_count", &self.shape_variants.len())
            .field("layout", &self.layout)
            .field("allocation_report", &self.allocation_report)
            .field("output_count", &self.output_count)
            .field("output_mode", &self.output_mode)
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
    /// `max_input_tokens` remains the maximum dense GEMM row count and must not
    /// exceed the model's maximum sequence length. Active-row mode prepares
    /// smaller exact-M plans against the same [`PreparedLlamaForward`] owner.
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
        let config = normalize_prepared_config(config);
        config.validate_metadata_transport()?;
        let mut shape_history = shape_history_for_config(config)?;
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
                reason: "maximum dense rows must not exceed the model sequence length",
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
        let shape_variants = match prepare_shape_variants(
            model,
            context,
            &forward,
            config.shape_policy,
            config.shape_buckets.as_slice(),
        ) {
            Ok(variants) => variants,
            Err(error) => {
                let _ = forward.close();
                return Err(error);
            }
        };
        shape_history.retain_prepared_variants(forward.plan.sequence_length(), |dense_rows| {
            shape_variants
                .iter()
                .any(|shape| shape.dense_rows == dense_rows)
        });
        let required_gemm_workspace_bytes = shape_variants.iter().fold(
            forward.gemms.maximum_workspace_bytes(),
            |required, shape| required.max(shape.gemms.maximum_workspace_bytes()),
        );
        if let Err(error) =
            forward.ensure_batch_shape_gemm_workspace(context, required_gemm_workspace_bytes)
        {
            for shape in shape_variants {
                let _ = shape.close();
            }
            let _ = forward.close();
            return Err(LlamaBatchExecutorError::Forward(error));
        }
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
        let rope_site = ExecutionSite::layer(0, LlamaOp::QueryRope);
        if forward.rope_table_profile() == LlamaRopeTableProfile::HuggingFaceCuda {
            let rope_angles = build_absolute_rope_angles(
                spec.max_sequence_length(),
                dimensions.head_dimension(),
                forward.plan.rope_theta(),
            )?;
            absolute_rope_cos
                .upload_from_slice(0, &rope_angles, &mut forward.io_staging, stream)
                .map_err(|source| batch_cuda(rope_site, source))?;
            let mut rope_table_params = RopeTableParams {
                angles_cos: span_mut(
                    &mut absolute_rope_cos,
                    CudaDType::F32,
                    rope_bytes_per_kind,
                    rope_site,
                )?,
                sin: span_mut(
                    &mut absolute_rope_sin,
                    CudaDType::F32,
                    rope_bytes_per_kind,
                    rope_site,
                )?,
                element_count: rope_bytes_per_kind / F32_BYTES,
            };
            rope_table(&mut rope_table_params, stream)
                .map_err(|source| batch_cuda(rope_site, source))?;
        } else {
            let (rope_cos, rope_sin) = build_absolute_cpu_rope_tables(
                spec.max_sequence_length(),
                dimensions.head_dimension(),
                forward.plan.rope_theta(),
            )?;
            absolute_rope_cos
                .upload_from_slice(0, &rope_cos, &mut forward.io_staging, stream)
                .map_err(|source| batch_cuda(rope_site, source))?;
            absolute_rope_sin
                .upload_from_slice(0, &rope_sin, &mut forward.io_staging, stream)
                .map_err(|source| batch_cuda(rope_site, source))?;
        }

        let device_input = allocate_device_input(context, bounds, config.metadata_transport)?;
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
        let greedy_result_capacity_bytes = checked_product_u64(
            &[
                usize_u64(
                    bounds.max_output_slots(),
                    LlamaBatchExecutorResource::GreedyResults,
                )?,
                GREEDY_RESULT_BYTES as u64,
            ],
            LlamaBatchExecutorResource::GreedyResults,
        )?;
        let greedy_results = if bounds.max_output_slots() == 0 {
            None
        } else {
            Some(allocate_device(
                context,
                greedy_result_capacity_bytes,
                ExecutionSite::global(LlamaOp::OutputGather),
            )?)
        };
        let host = allocate_host_workspace(context, bounds, config.metadata_transport)?;
        let allocation_report = build_batch_allocation_report(
            forward.allocation_report(),
            bounds,
            config.metadata_transport,
            layout,
            rope_bytes_per_kind,
            gathered_logits_capacity_bytes,
            greedy_result_capacity_bytes,
            &host,
        )?;

        Ok(Self {
            config,
            shape_history,
            metadata,
            forward,
            shape_variants,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            allocation_report,
            output_count: 0,
            output_mode: BatchOutputMode::Logits,
            output_ready: false,
            poisoned: false,
        })
    }

    #[must_use]
    pub const fn config(&self) -> PreparedLlamaBatchExecutorConfig {
        self.config
    }

    /// Stable C02 identifier for the completion boundary frozen during cold
    /// preparation. This reads the normalized prepared configuration, not a
    /// caller's requested CLI setting.
    #[must_use]
    pub const fn execution_completion_mode_id(&self) -> &'static str {
        execution_completion_implementation_id(self.config.execution_completion_implementation())
    }

    /// Stable C02 identifier for the cold-prepared metadata transport.
    #[must_use]
    pub const fn metadata_transport_id(&self) -> &'static str {
        batch_metadata_transport_id(self.config.metadata_transport())
    }

    /// Stable C02 identifier for the prepared dense-row shape policy.
    #[must_use]
    pub const fn batch_shape_policy_id(&self) -> &'static str {
        batch_shape_policy_id(self.config.shape_policy())
    }

    /// Exact maximum dense-row budget prepared for this executor.
    ///
    /// The server verifies this equals the scheduler iteration budget before
    /// publishing C02 facts.
    #[must_use]
    pub const fn batch_token_budget(&self) -> usize {
        self.config.metadata().max_input_tokens()
    }

    /// Stable ID of the prefill backend selected during cold preparation.
    ///
    /// The returned value is the prepared forward owner's actual selection
    /// trace, after normalization and capability fallback, not an attention
    /// preference requested by the caller.
    #[must_use]
    pub fn prefill_attention_implementation_id(&self) -> &'static str {
        self.forward.attention_selection().implementation_id()
    }

    /// Stable ID of the ragged paged-attention implementation bound to this
    /// prepared continuous-batch executor.
    #[must_use]
    pub const fn decode_attention_implementation_id(&self) -> &'static str {
        ragged_attention_implementation_id(
            self.config.ragged_attention_reduction_profile(),
            self.config.ragged_attention_implementation(),
        )
    }

    /// Stable aggregate ID for the role-specific prepared GEMM reduction
    /// policy vector.
    ///
    /// The forward owner resolves the role vector during preparation. Its
    /// whole-profile ID is the compact C02 aggregate: `canonical-v1` can
    /// contain the reviewed heterogeneous role vector, while fixed37 resolves
    /// its own fail-closed aggregate. Individual requested CLI values are not
    /// exposed here.
    #[must_use]
    pub const fn gemm_reduction_policy_aggregate_id(&self) -> &'static str {
        self.forward.reduction_profile().id()
    }

    /// Stable C02 value for the residual-plus-RMSNorm implementation.
    #[must_use]
    pub const fn residual_rmsnorm_implementation_id(&self) -> &'static str {
        residual_norm_implementation_id(self.config.residual_norm_implementation())
    }

    /// Runtime selection contract bound to the prepared reduction profile.
    #[must_use]
    pub const fn runtime_selection_policy_id(&self) -> &'static str {
        runtime_selection_policy_id(self.forward.reduction_profile())
    }

    /// Number of exact dense-row plans owned by this executor, including the
    /// maximum rollback plan held by the shared forward owner.
    #[must_use]
    pub fn prepared_shape_count(&self) -> usize {
        self.shape_variants.len() + 1
    }

    /// Selects the prepared dense row count for a prospective active batch.
    ///
    /// # Errors
    ///
    /// Returns when the active row count is empty or exceeds the cold bound.
    pub fn select_dense_rows(&self, active_rows: usize) -> LlamaBatchExecutorResult<usize> {
        select_prepared_dense_rows(
            self.config,
            self.forward.plan.sequence_length(),
            &self.shape_variants,
            active_rows,
        )
    }

    /// Returns shape facts from the most recent successful iteration.
    ///
    /// Failed or rejected iterations do not replace the last successful
    /// observation.
    #[must_use]
    pub const fn last_shape_observation(&self) -> Option<LlamaBatchShapeObservation> {
        self.shape_history.last_success()
    }

    /// Returns cumulative hit counters in ascending cold-prepared bucket order.
    ///
    /// Fixed-maximum mode exposes exactly one entry. The returned slice borrows
    /// inline executor storage and performs no allocation.
    #[must_use]
    pub const fn shape_bucket_hits(&self) -> &[LlamaBatchShapeBucketHit] {
        self.shape_history.entries()
    }

    /// Returns the forward/decode profile selected at cold preparation.
    ///
    /// Use [`Self::reduction_profile_is_coherent`] before treating this value
    /// as the complete graph profile in logs or evidence.
    #[must_use]
    pub const fn reduction_profile(&self) -> LlamaReductionProfile {
        self.config.reduction_profile()
    }

    /// Whether every reduction family matches [`Self::reduction_profile`].
    #[must_use]
    pub fn reduction_profile_is_coherent(&self) -> bool {
        self.config.reduction_profile_is_coherent()
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
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
            || self.forward.poisoned
            || self
                .shape_variants
                .iter()
                .any(|shape| shape.gemms.any_poisoned())
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
        self.execute_output(rows, BatchOutputMode::Logits, stream)
    }

    /// Executes one mixed iteration and reduces gathered BF16 logits to exact
    /// deterministic greedy token IDs on the device.
    ///
    /// This path is valid only when the caller has already proven that every
    /// output row uses unconstrained temperature-zero decoding with repetition
    /// penalty one. It preserves the same post-dispatch poison contract as
    /// [`Self::execute`].
    ///
    /// # Errors
    ///
    /// Returns for the same malformed metadata, capacity, poison, and CUDA
    /// failures as [`Self::execute`], plus unavailable greedy result storage.
    pub fn execute_greedy(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        self.execute_output(rows, BatchOutputMode::GreedyTokens, stream)
    }

    fn execute_output(
        &mut self,
        rows: &[LlamaBatchRow<'_>],
        output_mode_requested: BatchOutputMode,
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
            shape_history,
            metadata,
            forward,
            shape_variants,
            layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            allocation_report: _,
            output_count,
            output_mode,
            output_ready,
            poisoned,
        } = self;
        let packed = metadata.pack(rows)?;
        let active_rows = packed.total_input_tokens();
        let selected_dense_rows = select_prepared_dense_rows(
            *config,
            forward.plan.sequence_length(),
            shape_variants,
            active_rows,
        )?;
        let shape_bucket_index = shape_history.bucket_index(selected_dense_rows)?;
        validate_for_execution(
            packed,
            forward.plan.dimensions().vocabulary_size(),
            model_max_position(
                absolute_rope_cos,
                forward.plan.dimensions().head_dimension(),
            )?,
            *config,
        )?;

        let mut dispatch_disposition = BatchDispatchDisposition::PreDispatch;
        let result = execute_packed(
            packed,
            *config,
            selected_dense_rows,
            forward,
            shape_variants,
            *layout,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            output_mode_requested,
            &mut dispatch_disposition,
            stream,
        );
        match result {
            Ok(()) => {
                shape_history.record_success(shape_bucket_index, active_rows, selected_dense_rows);
                *output_count = packed.output_count();
                *output_mode = output_mode_requested;
                *output_ready = true;
                forward.output_ready = true;
                Ok(())
            }
            Err(error) => {
                // Host packing, pinned writes, descriptor preflight, and
                // command-batch begin failures do not trigger the iteration's
                // blanket mutation-unknown poison. Once command submission can
                // have started, semantic KV state may be partial and the owner
                // must never be reused. Established error-specific and nested
                // GEMM poison handling remains active in both cases below.
                if config.execution_completion == ExecutionCompletionImplementation::IterationBatch
                    && dispatch_disposition.mutation_may_have_occurred()
                {
                    *poisoned = true;
                    forward.poisoned = true;
                }
                poison_for_batch_error(poisoned, forward, &error);
                *poisoned |= shape_variants
                    .iter()
                    .any(|shape| shape.gemms.any_poisoned());
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

    /// Exact byte length of `{token_id,status}` records for a prospective
    /// greedy output count.
    ///
    /// # Errors
    ///
    /// Returns when `output_count` exceeds the cold-prepared output bound or
    /// when the record byte length overflows `usize`.
    pub fn greedy_result_byte_len_for(
        &self,
        output_count: usize,
    ) -> LlamaBatchExecutorResult<usize> {
        if output_count > self.config.metadata.max_output_slots() {
            return Err(LlamaBatchExecutorError::InvalidBatch {
                field: "output_count",
                reason: "prospective output count exceeds the cold-prepared bound",
            });
        }
        output_count.checked_mul(GREEDY_RESULT_BYTES).ok_or(
            LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::GreedyResults,
            },
        )
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
        if self.output_mode != BatchOutputMode::Logits {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_mode",
                reason: "the completed iteration produced greedy tokens, not downloadable logits",
            });
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

    /// Downloads and validates one exact greedy token ID per output slot.
    ///
    /// Device traffic is eight bytes per row: a token ID and a status word.
    /// The caller-owned destination is filled only after every record is
    /// validated, so a non-finite or unknown status cannot publish a partial
    /// sample vector.
    ///
    /// # Errors
    ///
    /// Returns when execution did not produce greedy records, the destination
    /// has the wrong length, a device status is invalid, or transfer fails.
    pub fn download_greedy_tokens(
        &mut self,
        destination: &mut [u32],
        stream: &mut CudaStream,
    ) -> LlamaBatchExecutorResult<()> {
        if self.is_poisoned() {
            return Err(LlamaBatchExecutorError::Poisoned);
        }
        if !self.output_ready {
            return Err(LlamaBatchExecutorError::OutputNotReady);
        }
        if self.output_mode != BatchOutputMode::GreedyTokens {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "output_mode",
                reason: "the completed iteration produced logits, not greedy tokens",
            });
        }
        if destination.len() != self.output_count {
            return Err(LlamaBatchExecutorError::InvalidDownloadLength {
                expected_bytes: self.output_count.saturating_mul(U32_BYTES),
                actual_bytes: destination.len().saturating_mul(U32_BYTES),
            });
        }
        if destination.is_empty() {
            return Ok(());
        }
        let vocabulary_size = self.vocabulary_size();
        let result_bytes = self.greedy_result_byte_len_for(self.output_count)?;
        let device =
            self.greedy_results
                .as_mut()
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "greedy_results",
                    reason: "non-empty output has no cold-prepared greedy result buffer",
                })?;
        let host = self.host.greedy_results.get_mut(..result_bytes).ok_or(
            LlamaBatchExecutorError::InvalidConfiguration {
                field: "greedy_results_host",
                reason: "cold-prepared host result storage is too short",
            },
        )?;
        if let Err(source) = device.download_to_slice(0, host, &mut self.forward.io_staging, stream)
        {
            poison_for_cuda_error(&mut self.poisoned, &source);
            return Err(batch_cuda(
                ExecutionSite::global(LlamaOp::OutputGather),
                source,
            ));
        }
        if let Err(error) = decode_greedy_tokens(host, vocabulary_size, destination) {
            if matches!(&error, LlamaBatchExecutorError::InvalidGreedyResult { .. }) {
                self.poisoned = true;
            }
            return Err(error);
        }
        Ok(())
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
            shape_history: _,
            metadata: _,
            forward,
            shape_variants,
            layout: _,
            key_cache,
            value_cache,
            absolute_rope_cos,
            absolute_rope_sin,
            device_input,
            gathered_logits,
            greedy_results,
            host,
            allocation_report: _,
            output_count: _,
            output_mode: _,
            output_ready: _,
            poisoned: _,
        } = self;
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
        if let Some(error) = close_device_input(device_input) {
            if first.is_none() {
                first = Some(error);
            }
        }
        if let Some(buffer) = gathered_logits {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::GatheredLogits,
                buffer.close(),
            );
        }
        if let Some(buffer) = greedy_results {
            record_close(
                &mut first,
                LlamaBatchExecutorResource::GreedyResults,
                buffer.close(),
            );
        }
        if let Some(error) = close_host_input(host.input) {
            if first.is_none() {
                first = Some(error);
            }
        }
        let mut shape_error = None;
        for shape in shape_variants {
            if let Err(error) = shape.close() {
                if shape_error.is_none() {
                    shape_error = Some(error);
                }
            }
        }
        let forward_result = forward.close().map_err(LlamaBatchExecutorError::Forward);
        match (first, shape_error, forward_result) {
            (Some(error), _, _) | (None, Some(error), _) => Err(error),
            (None, None, result) => result,
        }
    }
}

fn select_prepared_dense_rows(
    config: PreparedLlamaBatchExecutorConfig,
    maximum_rows: usize,
    variants: &[PreparedLlamaBatchShape],
    active_rows: usize,
) -> LlamaBatchExecutorResult<usize> {
    config.select_dense_rows(active_rows)?;
    Ok(select_smallest_prepared_dense_rows(
        active_rows,
        maximum_rows,
        variants.iter().map(|shape| shape.dense_rows),
    ))
}

pub(super) const fn normalize_prepared_config(
    config: PreparedLlamaBatchExecutorConfig,
) -> PreparedLlamaBatchExecutorConfig {
    PreparedLlamaBatchExecutorConfig {
        metadata: config.metadata,
        forward: config.forward.with_optimized_attention(),
        ragged_attention_reduction_profile: config.ragged_attention_reduction_profile,
        ragged_attention_implementation: config.ragged_attention_implementation,
        residual_norm: config.residual_norm,
        execution_completion: config.execution_completion,
        metadata_transport: config.metadata_transport,
        shape_policy: config.shape_policy,
        shape_buckets: config.shape_buckets,
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
    dense_rows: usize,
    forward: &mut PreparedLlamaForward,
    shape_variants: &mut [PreparedLlamaBatchShape],
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    device: &mut BatchDeviceInput,
    gathered_logits: &mut Option<CudaDeviceBuffer>,
    greedy_results: &mut Option<CudaDeviceBuffer>,
    host: &mut BatchHostWorkspace,
    output_mode: BatchOutputMode,
    dispatch_disposition: &mut BatchDispatchDisposition,
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    let bounds = config.metadata;
    let active = packed.total_input_tokens();
    if dense_rows != forward.plan.sequence_length()
        && !shape_variants
            .iter()
            .any(|shape| shape.dense_rows == dense_rows)
    {
        return Err(LlamaBatchExecutorError::InvalidConfiguration {
            field: "shape_variants",
            reason: "selected dense-row bucket was not prepared",
        });
    }
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
    let packed_layout = match (&mut host.input, &mut *device, config.metadata_transport) {
        (
            BatchHostInput::PerOperation(host),
            BatchDeviceInput::PerOperation(device),
            BatchMetadataTransport::Synchronous,
        ) => {
            host.padded_tokens[..dense_rows].fill(0);
            host.padded_tokens[..active].copy_from_slice(packed.input_token_ids());
            upload_batch_tokens(forward, &host.padded_tokens[..dense_rows], stream)?;
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
            None
        }
        (
            BatchHostInput::IterationBatch(host),
            BatchDeviceInput::IterationBatch { slab },
            BatchMetadataTransport::PackedAsync,
        ) => {
            let layout = PackedIterationLayout::for_batch(&packed, dense_rows)?;
            layout.validate_capacity(host.bytes.len())?;
            layout.validate_capacity(usize::try_from(slab.byte_len()).map_err(|_| {
                LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::PackedIterationInput,
                }
            })?)?;
            pack_iteration_input(&packed, dense_rows, layout, &mut host.bytes)?;
            host.pinned
                .write(0, &host.bytes[..layout.total_bytes])
                .map_err(|source| batch_cuda(metadata_site, source))?;
            Some(layout)
        }
        _ => {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                reason: "cold-prepared host/device input transport does not match configuration",
            });
        }
    };

    let mut execute_iteration_body = |batch: PackedBatchV1<'_>,
                                      token_ids: Option<CudaBufferSpan<'_>>,
                                      output_indices: Option<CudaBufferSpan<'_>>,
                                      stream: &mut dyn CudaExecutionStream|
     -> LlamaBatchExecutorResult<()> {
        let rms_norm_profile = forward.rms_norm_profile();
        let PreparedLlamaForward {
            plan: maximum_plan,
            weights,
            gemms: maximum_gemms,
            buffers,
            ..
        } = forward;
        let (plan, gemms) = if dense_rows == maximum_plan.sequence_length() {
            (&*maximum_plan, maximum_gemms)
        } else {
            let shape = shape_variants
                .iter_mut()
                .find(|shape| shape.dense_rows == dense_rows)
                .ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "shape_variants",
                    reason: "selected dense-row bucket was not prepared",
                })?;
            (&shape.plan, &mut shape.gemms)
        };
        execute_fixed_graph(
            plan,
            weights,
            gemms,
            buffers,
            config.residual_norm,
            rms_norm_profile,
            config.ragged_attention_reduction_profile,
            config.ragged_attention_implementation,
            layout,
            key_cache,
            value_cache,
            rope_cos,
            rope_sin,
            token_ids,
            batch,
            packed.position_ids(),
            stream,
        )?;

        if packed.output_count() != 0 {
            let output_indices =
                output_indices.ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "output_token_indices",
                    reason: "non-empty output has no cold-prepared device index buffer",
                })?;
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
                    &buffers.logits,
                    CudaDType::BF16,
                    plan.workspace_spec().logits_bytes(),
                    site,
                )?,
                row_indices: output_indices,
                row_indices_host: packed.output_token_indices(),
                output: CudaBufferSpanMut::new(
                    output,
                    CudaDType::BF16,
                    0,
                    output_logits_bytes(
                        packed.output_count(),
                        plan.dimensions().vocabulary_size(),
                    )?,
                )
                .map_err(|source| batch_cuda(site, source))?,
                input_row_count: usize_u64(dense_rows, LlamaBatchExecutorResource::GatheredLogits)?,
                column_count: usize_u64(
                    plan.dimensions().vocabulary_size(),
                    LlamaBatchExecutorResource::GatheredLogits,
                )?,
            };
            row_gather(&mut params, stream).map_err(|source| batch_cuda(site, source))?;
            if output_mode == BatchOutputMode::GreedyTokens {
                let logits = gathered_logits.as_ref().ok_or(
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "gathered_logits",
                        reason: "greedy selection requires gathered logits",
                    },
                )?;
                let results = greedy_results.as_mut().ok_or(
                    LlamaBatchExecutorError::InvalidConfiguration {
                        field: "greedy_results",
                        reason: "non-empty output has no cold-prepared greedy result buffer",
                    },
                )?;
                let mut argmax = Bf16ArgmaxParams {
                    logits: CudaBufferSpan::new(
                        logits,
                        CudaDType::BF16,
                        0,
                        output_logits_bytes(
                            packed.output_count(),
                            plan.dimensions().vocabulary_size(),
                        )?,
                    )
                    .map_err(|source| batch_cuda(site, source))?,
                    results: CudaBufferSpanMut::new(
                        results,
                        CudaDType::U32,
                        0,
                        usize_u64(
                            packed
                                .output_count()
                                .checked_mul(GREEDY_RESULT_BYTES)
                                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                                    resource: LlamaBatchExecutorResource::GreedyResults,
                                })?,
                            LlamaBatchExecutorResource::GreedyResults,
                        )?,
                    )
                    .map_err(|source| batch_cuda(site, source))?,
                    row_count: usize_u64(
                        packed.output_count(),
                        LlamaBatchExecutorResource::GreedyResults,
                    )?,
                    vocabulary_size: usize_u64(
                        plan.dimensions().vocabulary_size(),
                        LlamaBatchExecutorResource::GreedyResults,
                    )?,
                };
                deterministic_bf16_argmax(&mut argmax, stream)
                    .map_err(|source| batch_cuda(site, source))?;
            }
        }
        Ok(())
    };

    match (config.execution_completion, config.metadata_transport) {
        (ExecutionCompletionImplementation::PerOperation, BatchMetadataTransport::Synchronous) => {
            let BatchDeviceInput::PerOperation(device) = &*device else {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "metadata_transport",
                    reason: "synchronous execution has no per-operation device metadata",
                });
            };
            let views = per_operation_device_views(host_batch, device, &packed, metadata_site)?;
            execute_iteration_body(
                views.batch,
                views.token_ids,
                views.output_token_indices,
                stream,
            )
        }
        (
            ExecutionCompletionImplementation::IterationBatch,
            BatchMetadataTransport::Synchronous,
        ) => {
            let BatchDeviceInput::PerOperation(device) = &*device else {
                return Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "metadata_transport",
                    reason: "synchronous execution has no per-operation device metadata",
                });
            };
            let views = per_operation_device_views(host_batch, device, &packed, metadata_site)?;
            let completion_site = ExecutionSite::global(LlamaOp::IterationCompletion);
            let mut command_batch = stream
                .begin_command_batch()
                .map_err(|source| batch_cuda(completion_site, source))?;
            *dispatch_disposition = BatchDispatchDisposition::CommandSubmissionStarted;
            let body_result = {
                let mut commands = command_batch.commands();
                execute_iteration_body(
                    views.batch,
                    views.token_ids,
                    views.output_token_indices,
                    &mut commands,
                )
            };
            let completion_result = command_batch
                .finish()
                .map_err(|source| batch_cuda(completion_site, source));
            match completion_result {
                Err(error) => Err(error),
                Ok(()) => body_result,
            }
        }
        (
            ExecutionCompletionImplementation::IterationBatch,
            BatchMetadataTransport::PackedAsync,
        ) => {
            let layout = packed_layout.ok_or(LlamaBatchExecutorError::InvalidConfiguration {
                field: "packed_iteration_layout",
                reason: "packed async input was not prepared before command-batch begin",
            })?;
            let copy_byte_len = usize_u64(
                layout.total_bytes,
                LlamaBatchExecutorResource::PackedIterationInput,
            )?;
            match (&*device, &host.input) {
                (BatchDeviceInput::IterationBatch { slab }, BatchHostInput::IterationBatch(_)) => {
                    // Resolve every dtype, alignment, range, host-shape, and
                    // output-view check before the first H2D submission. The
                    // same immutable descriptors are rebound after enqueue;
                    // no shape or offset can change inside the command batch.
                    let _preflight_views =
                        packed_device_views(host_batch, slab, &packed, layout, metadata_site)?;
                }
                _ => {
                    return Err(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "metadata_transport",
                        reason: "packed async execution has no packed host/device slab",
                    });
                }
            }
            let completion_site = ExecutionSite::global(LlamaOp::IterationCompletion);
            let mut command_batch = stream
                .begin_command_batch()
                .map_err(|source| batch_cuda(completion_site, source))?;
            *dispatch_disposition = BatchDispatchDisposition::CommandSubmissionStarted;
            let body_result = {
                let mut commands = command_batch.commands();
                match (&mut *device, &host.input) {
                    (
                        BatchDeviceInput::IterationBatch { slab },
                        BatchHostInput::IterationBatch(host),
                    ) => {
                        let copy_result = slab
                            .copy_from_pinned_in_command_batch(
                                0,
                                &host.pinned,
                                0,
                                copy_byte_len,
                                &mut commands,
                            )
                            .map_err(|source| batch_cuda(metadata_site, source));
                        match copy_result {
                            Err(error) => Err(error),
                            Ok(()) => {
                                match packed_device_views(
                                    host_batch,
                                    slab,
                                    &packed,
                                    layout,
                                    metadata_site,
                                ) {
                                    Err(error) => Err(error),
                                    Ok(views) => execute_iteration_body(
                                        views.batch,
                                        views.token_ids,
                                        views.output_token_indices,
                                        &mut commands,
                                    ),
                                }
                            }
                        }
                    }
                    _ => Err(LlamaBatchExecutorError::InvalidConfiguration {
                        field: "metadata_transport",
                        reason: "packed async execution has no packed host/device slab",
                    }),
                }
            };
            let completion_result = command_batch
                .finish()
                .map_err(|source| batch_cuda(completion_site, source));
            match completion_result {
                Err(error) => Err(error),
                Ok(()) => body_result,
            }
        }
        (ExecutionCompletionImplementation::PerOperation, BatchMetadataTransport::PackedAsync) => {
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                reason: "packed async metadata requires iteration-batch completion",
            })
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::large_types_passed_by_value,
    clippy::similar_names
)]
fn execute_fixed_graph<S: CudaExecutionStream + ?Sized>(
    plan: &LlamaExecutionPlan,
    weights: &CudaUploadedWeights,
    gemms: &mut GemmPlans,
    buffers: &mut ForwardBuffers,
    residual_norm_implementation: ResidualNormImplementation,
    rms_norm_profile: LlamaRmsNormProfile,
    attention_reduction_profile: AttentionReductionProfile,
    attention_implementation: RaggedAttentionImplementation,
    layout: KvLayout,
    key_cache: &mut CudaDeviceBuffer,
    value_cache: &mut CudaDeviceBuffer,
    rope_cos: &CudaDeviceBuffer,
    rope_sin: &CudaDeviceBuffer,
    token_ids: Option<CudaBufferSpan<'_>>,
    batch: PackedBatchV1<'_>,
    positions_host: &[u32],
    stream: &mut S,
) -> LlamaBatchExecutorResult<()> {
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
        let token_ids = match token_ids {
            Some(token_ids) => token_ids,
            None => span(
                &buffers.token_ids,
                CudaDType::U32,
                plan.workspace_spec().token_ids_bytes(),
                embedding_site,
            )?,
        };
        let mut params = EmbeddingParams {
            table: embedding_weight,
            token_ids,
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
            execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
                .map_err(|source| batch_cuda(input_norm_site, source))?;
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
            match attention_reduction_profile {
                AttentionReductionProfile::CanonicalV1 => match attention_implementation {
                    RaggedAttentionImplementation::Legacy => {
                        ragged_paged_attention(&mut params, stream)
                    }
                    RaggedAttentionImplementation::GroupedHeads => {
                        grouped_ragged_paged_attention(&mut params, stream)
                    }
                },
                AttentionReductionProfile::FixedContiguous37BalancedV1 => {
                    fixed37_ragged_paged_attention(&mut params, stream)
                }
            }
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
        let post_norm_site = ExecutionSite::layer(layer_index, LlamaOp::PostAttentionNorm);
        match residual_norm_implementation {
            ResidualNormImplementation::Separate => {
                let mut residual = ResidualAddParams {
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
                residual_add(&mut residual, stream)
                    .map_err(|source| batch_cuda(attention_residual_site, source))?;
                let mut norm = RmsNormParams {
                    input: span(
                        &buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    weight: weight_span(
                        weights,
                        layer.post_attention_norm_weight(),
                        post_norm_site,
                    )?,
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
                execute_profile_rms_norm(rms_norm_profile, &mut norm, stream)
                    .map_err(|source| batch_cuda(post_norm_site, source))?;
            }
            ResidualNormImplementation::Fused => {
                let mut fused = ResidualRmsNormParams {
                    left: span(
                        &buffers.hidden_current,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    right: span(
                        &buffers.hidden_projection,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    weight: weight_span(
                        weights,
                        layer.post_attention_norm_weight(),
                        post_norm_site,
                    )?,
                    residual_output: span_mut(
                        &mut buffers.hidden_rotary,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    normalized_output: span_mut(
                        &mut buffers.hidden_norm,
                        CudaDType::BF16,
                        plan.workspace_spec().hidden_buffer_bytes(),
                        post_norm_site,
                    )?,
                    row_count: dense_rows,
                    hidden_size: hidden,
                    epsilon: layer.post_attention_norm_epsilon(),
                };
                execute_profile_residual_rms_norm(rms_norm_profile, &mut fused, stream)
                    .map_err(|source| batch_cuda(post_norm_site, source))?;
            }
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
        execute_profile_rms_norm(rms_norm_profile, &mut params, stream)
            .map_err(|source| batch_cuda(final_norm_site, source))?;
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
        if config.ragged_attention_reduction_profile
            == AttentionReductionProfile::FixedContiguous37BalancedV1
            && u64::from(position) >= FIXED37_RAGGED_MAX_LOGICAL_TOKENS
        {
            return Err(LlamaBatchExecutorError::PositionOutOfRange {
                row: usize::try_from(row).unwrap_or(usize::MAX),
                position,
                maximum: usize::try_from(FIXED37_RAGGED_MAX_LOGICAL_TOKENS).unwrap_or(usize::MAX),
            });
        }
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

fn allocate_device_input(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
) -> LlamaBatchExecutorResult<BatchDeviceInput> {
    match transport {
        BatchMetadataTransport::Synchronous => allocate_synchronous_device_input(context, bounds),
        BatchMetadataTransport::PackedAsync => {
            let capacity = PackedIterationLayout::capacity(bounds)?.total_bytes;
            allocate_packed_device_input(context, capacity)
        }
    }
}

fn allocate_host_workspace(
    context: &CudaContext,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
) -> LlamaBatchExecutorResult<BatchHostWorkspace> {
    let input = match transport {
        BatchMetadataTransport::Synchronous => allocate_synchronous_host_input(bounds)?,
        BatchMetadataTransport::PackedAsync => {
            let capacity = PackedIterationLayout::capacity(bounds)?.total_bytes;
            allocate_packed_host_input(context, capacity)?
        }
    };
    Ok(BatchHostWorkspace {
        input,
        greedy_results: allocate_zeroed_bytes(bounds.max_output_slots(), GREEDY_RESULT_BYTES)?,
    })
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

#[allow(clippy::cast_precision_loss)]
fn build_absolute_rope_angles(
    position_count: usize,
    head_dimension: usize,
    theta: f32,
) -> LlamaBatchExecutorResult<Box<[u8]>> {
    let half = head_dimension / 2;
    let elements =
        position_count
            .checked_mul(half)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeCos,
            })?;
    let mut angles = allocate_zeroed_bytes(elements, F32_BYTES_USIZE)?;
    for position in 0..position_count {
        for pair in 0..half {
            let exponent = (2 * pair) as f32 / head_dimension as f32;
            let inverse_frequency = 1.0 / theta.powf(exponent);
            let angle = position as f32 * inverse_frequency;
            let byte_offset = position
                .checked_mul(half)
                .and_then(|value| value.checked_add(pair))
                .and_then(|value| value.checked_mul(F32_BYTES_USIZE))
                .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                    resource: LlamaBatchExecutorResource::RopeCos,
                })?;
            angles[byte_offset..byte_offset + F32_BYTES_USIZE]
                .copy_from_slice(&angle.to_ne_bytes());
        }
    }
    Ok(angles)
}

type RopeTableBytes = (Box<[u8]>, Box<[u8]>);

#[allow(clippy::cast_precision_loss)]
fn build_absolute_cpu_rope_tables(
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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn build_batch_allocation_report(
    forward: PreparedLlamaAllocationReport,
    bounds: LlamaBatchMetadataConfig,
    transport: BatchMetadataTransport,
    layout: KvLayout,
    rope_bytes_per_kind: u64,
    gathered_logits_capacity_bytes: u64,
    greedy_result_capacity_bytes: u64,
    host: &BatchHostWorkspace,
) -> LlamaBatchExecutorResult<PreparedLlamaBatchAllocationReport> {
    let offset_count =
        bounds
            .max_rows()
            .checked_add(1)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::SequenceBlockOffsets,
            })?;
    let packed_metadata_device_bytes = [
        checked_host_byte_len(
            offset_count,
            U32_BYTES,
            LlamaBatchExecutorResource::SequenceBlockOffsets,
        )?,
        checked_host_byte_len(
            bounds.max_block_entries(),
            U32_BYTES,
            LlamaBatchExecutorResource::PhysicalBlockIds,
        )?,
        checked_host_byte_len(
            bounds.max_block_entries(),
            U16_BYTES,
            LlamaBatchExecutorResource::ValidTokens,
        )?,
        checked_host_byte_len(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowSequenceSlots,
        )?,
        checked_host_byte_len(
            bounds.max_input_tokens(),
            U32_BYTES,
            LlamaBatchExecutorResource::RowPositions,
        )?,
        checked_host_byte_len(
            bounds.max_output_slots(),
            U32_BYTES,
            LlamaBatchExecutorResource::OutputTokenIndices,
        )?,
    ]
    .into_iter()
    .try_fold(0_u64, |total, bytes| {
        total
            .checked_add(usize_u64(bytes, LlamaBatchExecutorResource::HostWorkspace)?)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })
    })?;
    let batch_input_device_bytes = match transport {
        BatchMetadataTransport::Synchronous => packed_metadata_device_bytes,
        BatchMetadataTransport::PackedAsync => usize_u64(
            PackedIterationLayout::capacity(bounds)?.total_bytes,
            LlamaBatchExecutorResource::PackedIterationInput,
        )?,
    };
    let rope_table_bytes =
        rope_bytes_per_kind
            .checked_mul(2)
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::RopeSin,
            })?;
    let additional_device_bytes = layout
        .total_bytes()
        .checked_add(rope_table_bytes)
        .and_then(|bytes| bytes.checked_add(batch_input_device_bytes))
        .and_then(|bytes| bytes.checked_add(gathered_logits_capacity_bytes))
        .and_then(|bytes| bytes.checked_add(greedy_result_capacity_bytes))
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let total_device_bytes = forward
        .total_device_bytes()
        .checked_add(additional_device_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::GatheredLogits,
        })?;
    let (base_allocations, output_allocations) = match transport {
        BatchMetadataTransport::Synchronous => (
            PER_OPERATION_BASE_DEVICE_ALLOCATIONS,
            u64::from(bounds.max_output_slots() != 0) * 3,
        ),
        BatchMetadataTransport::PackedAsync => (
            ITERATION_BATCH_BASE_DEVICE_ALLOCATIONS,
            u64::from(bounds.max_output_slots() != 0) * 2,
        ),
    };
    let additional_device_allocation_count = base_allocations
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
    let input_host_workspace_bytes = match &host.input {
        BatchHostInput::PerOperation(input) => [
            input.padded_tokens.len().checked_mul(U32_BYTES),
            Some(input.sequence_block_offsets.len()),
            Some(input.physical_block_ids.len()),
            Some(input.valid_tokens.len()),
            Some(input.row_sequence_slots.len()),
            Some(input.row_positions.len()),
            Some(input.output_token_indices.len()),
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
        })?,
        BatchHostInput::IterationBatch(input) => input.bytes.len(),
    };
    let host_workspace_bytes = usize_u64(
        input_host_workspace_bytes
            .checked_add(host.greedy_results.len())
            .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
                resource: LlamaBatchExecutorResource::HostWorkspace,
            })?,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    let (additional_pinned_host_bytes, additional_pinned_host_allocation_count) = match &host.input
    {
        BatchHostInput::PerOperation(_) => (0, 0),
        BatchHostInput::IterationBatch(input) => (input.pinned.byte_len(), 1),
    };
    let total_pinned_host_bytes = forward
        .pinned_host_bytes()
        .checked_add(additional_pinned_host_bytes)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::PinnedIterationInput,
        })?;
    let total_pinned_host_allocation_count = forward
        .pinned_host_allocation_count()
        .checked_add(additional_pinned_host_allocation_count)
        .ok_or(LlamaBatchExecutorError::ArithmeticOverflow {
            resource: LlamaBatchExecutorResource::PinnedIterationInput,
        })?;
    Ok(PreparedLlamaBatchAllocationReport {
        forward,
        kv_cache_bytes: layout.total_bytes(),
        rope_table_bytes,
        packed_metadata_device_bytes,
        batch_input_device_bytes,
        gathered_logits_capacity_bytes,
        greedy_result_capacity_bytes,
        additional_device_bytes,
        total_device_bytes,
        additional_device_allocation_count,
        total_device_allocation_count,
        host_workspace_bytes,
        total_pinned_host_bytes,
        total_pinned_host_allocation_count,
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

fn upload_batch_tokens(
    forward: &mut PreparedLlamaForward,
    token_ids: &[u32],
    stream: &mut CudaStream,
) -> LlamaBatchExecutorResult<()> {
    if token_ids.len() == forward.plan.sequence_length() {
        return forward.upload_tokens(token_ids, stream).map_err(Into::into);
    }
    let byte_len = checked_host_byte_len(
        token_ids.len(),
        U32_BYTES,
        LlamaBatchExecutorResource::HostWorkspace,
    )?;
    if byte_len > forward.token_bytes.len() {
        return Err(LlamaBatchExecutorError::InvalidBatch {
            field: "dense_rows",
            reason: "selected token prefix exceeds the shared maximum buffer",
        });
    }
    encode_u32(token_ids, &mut forward.token_bytes[..byte_len]);
    forward.tokens_ready = false;
    forward.output_ready = false;
    let site = ExecutionSite::global(LlamaOp::Embedding);
    match forward.buffers.token_ids.upload_from_slice(
        0,
        &forward.token_bytes[..byte_len],
        &mut forward.io_staging,
        stream,
    ) {
        Ok(()) => {
            forward.tokens_ready = true;
            Ok(())
        }
        Err(source) => {
            poison_for_cuda_error(&mut forward.poisoned, &source);
            Err(batch_cuda(site, source))
        }
    }
}

fn upload_prefix(
    destination: &mut CudaDeviceBuffer,
    source: &[u8],
    byte_len: usize,
    staging: &mut riley_cuda::CudaPinnedHostBuffer,
    stream: &mut CudaStream,
    site: ExecutionSite,
) -> LlamaBatchExecutorResult<()> {
    destination
        .upload_from_slice(0, &source[..byte_len], staging, stream)
        .map_err(|source| batch_cuda(site, source))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llama::batch::{
        LlamaBatchBlockTable, LlamaBatchRowKind, PreparedLlamaBatchMetadata,
    };
    use crate::llama::executor::metadata::ByteRegion;
    use crate::paged_kv::BLOCK_TABLE_V1_VERSION;

    #[test]
    fn metadata_transport_is_synchronous_by_default_and_explicitly_reversible() {
        let metadata = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(
            defaults.metadata_transport(),
            BatchMetadataTransport::Synchronous
        );
        assert_eq!(
            defaults
                .with_packed_async_metadata()
                .with_synchronous_metadata()
                .metadata_transport(),
            BatchMetadataTransport::Synchronous
        );
        assert!(matches!(
            defaults
                .with_packed_async_metadata()
                .validate_metadata_transport(),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "metadata_transport",
                ..
            })
        ));
        let packed = defaults
            .with_iteration_batch_completion()
            .with_packed_async_metadata();
        packed
            .validate_metadata_transport()
            .expect("iteration completion owns the pinned-source lease");
        assert_eq!(
            normalize_prepared_config(packed).metadata_transport(),
            BatchMetadataTransport::PackedAsync
        );
    }

    #[test]
    fn ragged_attention_implementation_is_legacy_by_default_reversible_and_preserved() {
        let metadata = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(
            defaults.ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
        assert_eq!(
            defaults
                .with_grouped_ragged_attention_heads()
                .with_legacy_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        assert_eq!(
            normalize_prepared_config(defaults.with_grouped_ragged_attention_heads())
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::GroupedHeads
        );
    }

    #[test]
    fn c02_runtime_fact_ids_follow_normalized_prepared_policy() {
        let metadata =
            LlamaBatchMetadataConfig::new(2, 64, 8, 2, 8).expect("valid metadata bounds");
        let normalized = normalize_prepared_config(
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_iteration_batch_completion()
                .with_packed_async_metadata()
                .with_active_row_buckets()
                .with_grouped_ragged_attention_heads(),
        );

        assert_eq!(
            execution_completion_implementation_id(
                normalized.execution_completion_implementation()
            ),
            "iteration-batch"
        );
        assert_eq!(
            batch_metadata_transport_id(normalized.metadata_transport()),
            "packed-async"
        );
        assert_eq!(
            batch_shape_policy_id(normalized.shape_policy()),
            "power-of-two"
        );
        assert_eq!(
            residual_norm_implementation_id(normalized.residual_norm_implementation()),
            "separate"
        );
        assert_eq!(
            ragged_attention_implementation_id(
                normalized.ragged_attention_reduction_profile(),
                normalized.ragged_attention_implementation(),
            ),
            RAGGED_PAGED_ATTENTION_GROUPED_HEADS_D64_V1
        );
        assert_eq!(
            runtime_selection_policy_id(normalized.reduction_profile()),
            "exact-fallback-allowed"
        );

        let fixed = normalized.with_fixed37_reductions();
        assert_eq!(
            ragged_attention_implementation_id(
                fixed.ragged_attention_reduction_profile(),
                fixed.ragged_attention_implementation(),
            ),
            RAGGED_PAGED_ATTENTION_FIXED37_TWO_PASS_D64_S8192_V1
        );
        assert_eq!(
            runtime_selection_policy_id(fixed.reduction_profile()),
            "fail-closed"
        );
        assert_eq!(
            fixed.reduction_profile().id(),
            "fixed-contiguous-37-balanced-v1"
        );
        assert_eq!(
            residual_norm_implementation_id(ResidualNormImplementation::Fused),
            "fused"
        );
    }

    #[test]
    fn dispatch_disposition_distinguishes_preflight_from_unknown_mutation() {
        let mut disposition = BatchDispatchDisposition::PreDispatch;
        assert!(!disposition.mutation_may_have_occurred());

        disposition = BatchDispatchDisposition::CommandSubmissionStarted;
        assert!(disposition.mutation_may_have_occurred());
    }

    #[test]
    fn packed_iteration_layout_is_checked_aligned_and_capacity_bounded() {
        let layout = PackedIterationLayout::checked(4, 3, 5, 5, 3, 3, 2)
            .expect("representable packed layout");

        assert_eq!(
            layout.token_ids,
            ByteRegion {
                offset: 0,
                byte_len: 16
            }
        );
        assert_eq!(
            layout.sequence_block_offsets,
            ByteRegion {
                offset: 16,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.physical_block_ids,
            ByteRegion {
                offset: 28,
                byte_len: 20
            }
        );
        assert_eq!(
            layout.valid_tokens,
            ByteRegion {
                offset: 48,
                byte_len: 10
            }
        );
        assert_eq!(
            layout.row_sequence_slots,
            ByteRegion {
                offset: 60,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.row_positions,
            ByteRegion {
                offset: 72,
                byte_len: 12
            }
        );
        assert_eq!(
            layout.output_token_indices,
            ByteRegion {
                offset: 84,
                byte_len: 8
            }
        );
        assert_eq!(layout.total_bytes, 92);
        for region in [
            layout.token_ids,
            layout.sequence_block_offsets,
            layout.physical_block_ids,
            layout.row_sequence_slots,
            layout.row_positions,
            layout.output_token_indices,
        ] {
            assert_eq!(region.offset % U32_BYTES, 0);
        }
        assert_eq!(layout.valid_tokens.offset % U16_BYTES, 0);
        layout
            .validate_capacity(layout.total_bytes)
            .expect("exact capacity is accepted");
        assert!(matches!(
            layout.validate_capacity(layout.total_bytes - 1),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                ..
            })
        ));

        let bounds = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let capacity = PackedIterationLayout::capacity(bounds).expect("checked cold capacity");
        assert!(capacity.total_bytes >= layout.total_bytes);
    }

    #[test]
    fn packed_iteration_host_bytes_match_all_seven_sources_and_zero_padding() {
        let prefill_tokens = [10, 11, 12];
        let prefill_ids = [2];
        let prefill_valid = [3];
        let decode_tokens = [20];
        let decode_ids = [4, 5];
        let decode_valid = [u16::try_from(KV_BLOCK_SIZE).expect("block size"), 1];
        let rows = [
            LlamaBatchRow::new(
                41,
                LlamaBatchRowKind::Prefill,
                &prefill_tokens,
                3,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &prefill_ids, &prefill_valid, 3),
                Some(1),
            ),
            LlamaBatchRow::new(
                42,
                LlamaBatchRowKind::Decode,
                &decode_tokens,
                17,
                LlamaBatchBlockTable::new(BLOCK_TABLE_V1_VERSION, &decode_ids, &decode_valid, 17),
                Some(0),
            ),
        ];
        let bounds = LlamaBatchMetadataConfig::new(2, 8, 4, 2, 8).expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(bounds).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack mixed rows");
        let layout = PackedIterationLayout::for_batch(&packed, 8).expect("dynamic layout");
        let capacity = PackedIterationLayout::capacity(bounds)
            .expect("cold layout")
            .total_bytes;
        let mut bytes = vec![0xA5; capacity];

        pack_iteration_input(&packed, 8, layout, &mut bytes).expect("pack host input");

        assert_eq!(
            &bytes[layout.token_ids.offset..layout.token_ids.offset + 4 * U32_BYTES],
            &[10_u32, 11, 12, 20]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert!(
            bytes[layout.token_ids.offset + 4 * U32_BYTES..layout.token_ids.end().expect("end")]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert_eq!(
            &bytes[layout.sequence_block_offsets.offset
                ..layout.sequence_block_offsets.end().expect("end")],
            &[0_u32, 1, 3]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.physical_block_ids.offset..layout.physical_block_ids.end().expect("end")],
            &[2_u32, 4, 5]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.valid_tokens.offset..layout.valid_tokens.end().expect("end")],
            &[3_u16, 16, 1]
                .into_iter()
                .flat_map(u16::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.row_sequence_slots.offset..layout.row_sequence_slots.end().expect("end")],
            &[0_u32, 0, 0, 1]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.row_positions.offset..layout.row_positions.end().expect("end")],
            &[0_u32, 1, 2, 16]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &bytes[layout.output_token_indices.offset
                ..layout.output_token_indices.end().expect("end")],
            &[3_u32, 2]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert!(
            bytes[layout.valid_tokens.end().expect("end")..layout.row_sequence_slots.offset]
                .iter()
                .all(|&byte| byte == 0)
        );
        assert!(bytes[layout.total_bytes..].iter().all(|&byte| byte == 0xA5));
    }

    #[test]
    fn packed_iteration_input_preflight_preserves_destination_bytes() {
        let tokens = [7_u32];
        let physical_block_ids = [0_u32];
        let valid_tokens = [1_u16];
        let rows = [LlamaBatchRow::new(
            41,
            LlamaBatchRowKind::Prefill,
            &tokens,
            1,
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                1,
            ),
            None,
        )];
        let bounds = LlamaBatchMetadataConfig::new(1, 1, 1, 0, 1).expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(bounds).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack one row");
        let layout = PackedIterationLayout::for_batch(&packed, 1).expect("dynamic layout");
        let mut bytes = [0xA5_u8; 64];

        assert!(matches!(
            pack_iteration_input(&packed, 1, layout, &mut bytes[..layout.total_bytes - 1],),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "packed_iteration_input",
                reason: "dynamic packed input exceeds the cold-prepared slab",
            })
        ));
        assert!(bytes.iter().all(|&byte| byte == 0xA5));

        let too_small = PackedIterationLayout::for_batch(&packed, 0).expect("representable layout");
        assert!(matches!(
            pack_iteration_input(&packed, 0, too_small, &mut bytes[..too_small.total_bytes],),
            Err(LlamaBatchExecutorError::InvalidBatch {
                field: "dense_rows",
                reason: "active input rows exceed the selected packed token region",
            })
        ));
        assert!(bytes.iter().all(|&byte| byte == 0xA5));
    }

    #[test]
    fn fixed_maximum_shape_is_default_and_reversible() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());

        assert_eq!(defaults.shape_policy(), LlamaBatchShapePolicy::FixedMaximum);
        assert_eq!(defaults.select_dense_rows(1).expect("select fixed M"), 512);
        assert_eq!(
            defaults
                .with_active_row_buckets()
                .with_fixed_maximum_shape()
                .shape_policy(),
            LlamaBatchShapePolicy::FixedMaximum
        );
    }

    #[test]
    fn active_row_policy_selects_smallest_prepared_bucket() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();

        for (active, expected) in [
            (1, 1),
            (2, 2),
            (3, 4),
            (8, 8),
            (9, 16),
            (127, 128),
            (128, 128),
            (129, 256),
            (256, 256),
            (257, 512),
            (512, 512),
        ] {
            assert_eq!(
                config.select_dense_rows(active).expect("select bucket"),
                expected,
                "active rows {active}"
            );
        }
    }

    #[test]
    fn active_row_policy_uses_non_power_of_two_maximum_as_final_bucket() {
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(65, 100)
                .expect("select configured maximum"),
            100
        );
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(200, 300)
                .expect("select power-of-two bucket"),
            256
        );
        assert_eq!(
            LlamaBatchShapePolicy::ActiveRowBuckets
                .select_dense_rows(257, 300)
                .expect("select final bucket"),
            300
        );
    }

    #[test]
    fn custom_active_row_buckets_are_stored_and_select_the_smallest_shape() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let automatic =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();
        assert_eq!(
            automatic.configured_shape_buckets(),
            &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        );

        let custom = automatic
            .with_custom_active_row_buckets(&[1, 3, 7, 64, 512])
            .expect("valid custom buckets");
        assert_eq!(
            custom.shape_policy(),
            LlamaBatchShapePolicy::ActiveRowBuckets
        );
        assert_eq!(custom.configured_shape_buckets(), &[1, 3, 7, 64, 512]);
        for (active, expected) in [(1, 1), (2, 3), (3, 3), (4, 7), (65, 512), (512, 512)] {
            assert_eq!(
                custom.select_dense_rows(active).expect("select custom"),
                expected
            );
        }
        assert_eq!(
            custom
                .with_fixed_maximum_shape()
                .select_dense_rows(1)
                .expect("fixed rollback"),
            512
        );
    }

    #[test]
    fn unavailable_anchored_shape_uses_the_next_prepared_bucket_or_exact_maximum() {
        let prepared = [2, 8, 64];
        assert_eq!(
            select_smallest_prepared_dense_rows(1, 256, prepared.into_iter()),
            2
        );
        assert_eq!(
            select_smallest_prepared_dense_rows(3, 256, prepared.into_iter()),
            8
        );
        assert_eq!(
            select_smallest_prepared_dense_rows(9, 256, prepared.into_iter()),
            64
        );
        assert_eq!(
            select_smallest_prepared_dense_rows(65, 256, prepared.into_iter()),
            256
        );
    }

    #[test]
    fn custom_active_row_buckets_fail_closed_for_every_list_invariant() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let defaults =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        let excessive = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 512];
        for invalid in [
            &[][..],
            &[0, 512][..],
            &[2, 512][..],
            &[1, 2, 2, 512][..],
            &[1, 4, 2, 512][..],
            &[1, 2, 256][..],
            &excessive[..],
        ] {
            assert!(matches!(
                defaults.with_custom_active_row_buckets(invalid),
                Err(LlamaBatchExecutorError::InvalidConfiguration {
                    field: "shape_buckets",
                    ..
                })
            ));
        }
    }

    fn record_shape_success(
        history: &mut LlamaBatchShapeHistory,
        config: PreparedLlamaBatchExecutorConfig,
        active_rows: usize,
    ) {
        let dense_rows = config
            .select_dense_rows(active_rows)
            .expect("valid active rows");
        let bucket_index = history
            .bucket_index(dense_rows)
            .expect("selected bucket is tracked");
        history.record_success(bucket_index, active_rows, dense_rows);
    }

    #[test]
    fn fixed_maximum_shape_history_tracks_padding_and_one_bucket() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        let mut history = shape_history_for_config(config).expect("valid fixed history");
        assert_eq!(history.last_success(), None);

        record_shape_success(&mut history, config, 128);
        record_shape_success(&mut history, config, 1);

        let observation = history
            .last_success()
            .expect("successful shape observation");
        assert_eq!(observation.active_rows(), 1);
        assert_eq!(observation.selected_dense_rows(), 512);
        assert_eq!(observation.padding_rows(), 511);
        let hits = history.entries();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].dense_rows(), 512);
        assert_eq!(hits[0].hit_count(), 2);
    }

    #[test]
    fn active_shape_history_tracks_shape_changes_and_maximum_hits() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_active_row_buckets();
        let mut history = shape_history_for_config(config).expect("valid active history");

        for active_rows in [128, 1, 8, 256, 1, 511] {
            record_shape_success(&mut history, config, active_rows);
        }

        let observation = history
            .last_success()
            .expect("successful shape observation");
        assert_eq!(observation.active_rows(), 511);
        assert_eq!(observation.selected_dense_rows(), 512);
        assert_eq!(observation.padding_rows(), 1);
        let hits = history.entries();
        assert_eq!(hits.len(), 10);
        assert_eq!(hits[0].dense_rows(), 1);
        assert_eq!(hits[0].hit_count(), 2);
        assert_eq!(hits[3].dense_rows(), 8);
        assert_eq!(hits[3].hit_count(), 1);
        assert_eq!(hits[7].dense_rows(), 128);
        assert_eq!(hits[7].hit_count(), 1);
        assert_eq!(hits[8].dense_rows(), 256);
        assert_eq!(hits[8].hit_count(), 1);
        assert_eq!(hits[9].dense_rows(), 512);
        assert_eq!(hits[9].hit_count(), 1);
    }

    #[test]
    fn shape_selection_rejects_empty_and_over_capacity_batches() {
        for active in [0, 513] {
            assert!(matches!(
                LlamaBatchShapePolicy::ActiveRowBuckets.select_dense_rows(active, 512),
                Err(LlamaBatchExecutorError::InvalidBatch {
                    field: "active_rows",
                    ..
                })
            ));
        }
    }

    #[test]
    fn prepare_normalization_preserves_active_row_policy() {
        let metadata =
            LlamaBatchMetadataConfig::new(8, 512, 8, 8, 8).expect("valid metadata bounds");
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_reference_attention(),
        )
        .with_custom_active_row_buckets(&[1, 8, 64, 512])
        .expect("valid custom buckets");

        let normalized = normalize_prepared_config(config);
        assert_eq!(
            normalized.shape_policy(),
            LlamaBatchShapePolicy::ActiveRowBuckets
        );
        assert_eq!(normalized.configured_shape_buckets(), &[1, 8, 64, 512]);
    }

    #[test]
    fn whole_reduction_profile_updates_forward_and_ragged_attention_atomically() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid metadata bounds");
        let canonical =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default());
        assert_eq!(
            canonical.reduction_profile(),
            LlamaReductionProfile::CanonicalV1
        );
        assert_eq!(
            canonical.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(canonical.reduction_profile_is_coherent());

        let fixed = canonical.with_fixed37_reductions();
        assert_eq!(
            fixed.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            fixed.forward().reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            fixed.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
        assert!(fixed.reduction_profile_is_coherent());

        let narrow_rollback = fixed.with_canonical_ragged_attention();
        assert_eq!(
            narrow_rollback.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            narrow_rollback.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(!narrow_rollback.reduction_profile_is_coherent());

        let restored = narrow_rollback.with_canonical_reductions();
        assert_eq!(
            restored.reduction_profile(),
            LlamaReductionProfile::CanonicalV1
        );
        assert_eq!(
            restored.ragged_attention_reduction_profile(),
            AttentionReductionProfile::CanonicalV1
        );
        assert!(restored.reduction_profile_is_coherent());

        let normalized = normalize_prepared_config(fixed);
        assert_eq!(
            normalized.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            normalized.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
    }

    #[test]
    fn new_batch_config_inherits_forward_reduction_profile() {
        let metadata = LlamaBatchMetadataConfig::new(1, 1, 1, 1, 1).expect("valid metadata bounds");
        let config = PreparedLlamaBatchExecutorConfig::new(
            metadata,
            PreparedLlamaForwardConfig::default().with_fixed37_reductions(),
        );

        assert_eq!(
            config.reduction_profile(),
            LlamaReductionProfile::FixedContiguous37BalancedV1
        );
        assert_eq!(
            config.ragged_attention_reduction_profile(),
            AttentionReductionProfile::FixedContiguous37BalancedV1
        );
    }

    #[test]
    fn fixed37_profile_rejects_t8193_in_host_preflight() {
        const LOGICAL_TOKENS: usize = 8_193;
        let block_count = LOGICAL_TOKENS.div_ceil(KV_BLOCK_SIZE);
        let physical_block_ids: Vec<u32> =
            (0..u32::try_from(block_count).expect("block count")).collect();
        let mut valid_tokens = vec![u16::try_from(KV_BLOCK_SIZE).expect("block size"); block_count];
        *valid_tokens.last_mut().expect("last block") = 1;
        let token = [1_u32];
        let rows = [LlamaBatchRow::new(
            1,
            LlamaBatchRowKind::Decode,
            &token,
            u32::try_from(LOGICAL_TOKENS).expect("logical token count"),
            LlamaBatchBlockTable::new(
                BLOCK_TABLE_V1_VERSION,
                &physical_block_ids,
                &valid_tokens,
                u32::try_from(LOGICAL_TOKENS).expect("logical token count"),
            ),
            Some(0),
        )];
        let metadata = LlamaBatchMetadataConfig::new(1, 1, block_count, 1, block_count)
            .expect("valid metadata bounds");
        let mut prepared = PreparedLlamaBatchMetadata::prepare(metadata).expect("prepare metadata");
        let packed = prepared.pack(&rows).expect("pack T=8193 decode row");
        let fixed37 =
            PreparedLlamaBatchExecutorConfig::new(metadata, PreparedLlamaForwardConfig::default())
                .with_fixed37_ragged_attention();

        assert!(matches!(
            validate_for_execution(packed, 2, 16_384, fixed37),
            Err(LlamaBatchExecutorError::PositionOutOfRange {
                position: 8_192,
                maximum: 8_192,
                ..
            })
        ));

        let packed = prepared.pack(&rows).expect("repack after preflight error");
        validate_for_execution(packed, 2, 16_384, fixed37.with_canonical_ragged_attention())
            .expect("canonical ragged attention retains its existing model bound");
    }
}
