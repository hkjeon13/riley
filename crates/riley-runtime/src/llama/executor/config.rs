//! Host-only cold configuration for the Llama continuous-batch executor.
//!
//! This component owns scalar selection and validation. It deliberately does not own CUDA resources, model weights,
//! KV storage, metadata transfer buffers, or execution dispatch.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use riley_cuda::AttentionReductionProfile;

use super::super::LlamaReductionProfile;
use super::super::batch::LlamaBatchMetadataConfig;
use super::super::forward::PreparedLlamaForwardConfig;
use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResult};
use super::shape::{LlamaBatchShapeBuckets, LlamaBatchShapePolicy};

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

pub(in crate::llama) const fn execution_completion_implementation_id(
    implementation: ExecutionCompletionImplementation,
) -> &'static str {
    match implementation {
        ExecutionCompletionImplementation::PerOperation => "per-operation",
        ExecutionCompletionImplementation::IterationBatch => "iteration-batch",
    }
}

pub(in crate::llama) const fn batch_metadata_transport_id(
    transport: BatchMetadataTransport,
) -> &'static str {
    match transport {
        BatchMetadataTransport::Synchronous => "synchronous",
        BatchMetadataTransport::PackedAsync => "packed-async",
    }
}

pub(in crate::llama) const fn residual_norm_implementation_id(
    implementation: ResidualNormImplementation,
) -> &'static str {
    match implementation {
        ResidualNormImplementation::Separate => "separate",
        ResidualNormImplementation::Fused => "fused",
    }
}

pub(in crate::llama) const fn ragged_attention_implementation_id(
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

pub(in crate::llama) const fn runtime_selection_policy_id(
    profile: LlamaReductionProfile,
) -> &'static str {
    match profile {
        LlamaReductionProfile::CanonicalV1 => "exact-fallback-allowed",
        LlamaReductionProfile::FixedContiguous37BalancedV1 => "fail-closed",
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

    pub(in crate::llama) fn validate_metadata_transport(self) -> LlamaBatchExecutorResult<()> {
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
    /// than `MAX_LLAMA_BATCH_SHAPE_BUCKETS` entries, and end at exactly the
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

pub(in crate::llama) const fn normalize_prepared_config(
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
