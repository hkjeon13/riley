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
    vllm_smol_p128_graph: bool,
    shared_rows_graph: bool,
    variable_graph: bool,
    variable_graph_rows: usize,
    packed_prefill: bool,
    mixed_execution: bool,
    flashinfer_experimental: bool,
    fa3_experimental: bool,
    ffn_pipeline: bool,
    prefill_ffn_pipeline: bool,
    adaptive_decode: bool,
    decode_window: bool,
    mixed_time_budget_ns: Option<u64>,
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
            vllm_smol_p128_graph: false,
            shared_rows_graph: false,
            variable_graph: false,
            variable_graph_rows: 8,
            packed_prefill: false,
            mixed_execution: false,
            flashinfer_experimental: false,
            fa3_experimental: false,
            ffn_pipeline: false,
            prefill_ffn_pipeline: false,
            adaptive_decode: false,
            decode_window: false,
            mixed_time_budget_ns: None,
            shape_policy: LlamaBatchShapePolicy::FixedMaximum,
            shape_buckets: LlamaBatchShapeBuckets::automatic(metadata.max_input_tokens()),
        }
    }

    /// Selects the explicit VllmSmolP128V1 owned graph. Eager fallback is forbidden.
    #[must_use]
    pub const fn with_vllm_smol_p128_graph(mut self) -> Self {
        self.variable_graph = false;
        self.packed_prefill = false;self.mixed_execution=false;self.flashinfer_experimental=false;self.fa3_experimental=false;self.ffn_pipeline=false;self.prefill_ffn_pipeline=false;self.adaptive_decode=false;self.decode_window=false;self.mixed_time_budget_ns=None;
        self.vllm_smol_p128_graph = true;
        self.shared_rows_graph = false;
        self
    }
    /// Opt-in variable-prefill SmolLM2 graph with a retained single-request session.
    #[must_use]
    pub const fn with_variable_graph(mut self)->Self {self.variable_graph=true;self.variable_graph_rows=8;self.packed_prefill=false;self.mixed_execution=false;self.flashinfer_experimental=false;self.fa3_experimental=false;self.ffn_pipeline=false;self.prefill_ffn_pipeline=false;self.adaptive_decode=false;self.decode_window=false;self.mixed_time_budget_ns=None;self.vllm_smol_p128_graph=false;self.shared_rows_graph=false;self}
    pub const fn with_variable_graph16(self)->Self {let mut s=self.with_variable_graph();s.variable_graph_rows=16;s}
    pub const fn with_variable_graph32(self)->Self {let mut s=self.with_variable_graph();s.variable_graph_rows=32;s}
    pub const fn with_packed_prefill(self)->Self {let mut s=self.with_variable_graph32();s.packed_prefill=true;s}
    pub const fn with_mixed_execution(self)->Self {let mut s=self.with_packed_prefill();s.mixed_execution=true;s}
    /// Unqualified numerical profile for explicit serving diagnostics only.
    pub const fn with_flashinfer_experimental(self) -> Self {
        let mut s = self.with_mixed_execution();
        s.flashinfer_experimental = true;
        s
    }
    /// Unqualified Hopper attention profile, independently selected from FlashInfer.
    pub const fn with_fa3_experimental(self) -> Self {let mut s=self.with_mixed_execution();s.fa3_experimental=true;s}
    pub const fn fa3_experimental(self) -> bool {self.fa3_experimental}
    pub const fn with_ffn_pipeline(self) -> Self {
        let mut s = self.with_mixed_execution();
        s.ffn_pipeline = true;
        s
    }
    pub const fn with_decode_window(self) -> Self {let mut s=self.with_mixed_execution();s.decode_window=true;s}
    /// Selects exact-order prefill FFN transport, optionally retaining paired decode.
    pub const fn with_prefill_ffn_pipeline(self, paired: bool) -> Self {
        let mut s=self.with_mixed_execution();
        s.prefill_ffn_pipeline=true;s.decode_window=paired;s
    }
    /// Selects the adaptive pure-decode projection batch with optional existing features.
    pub const fn with_adaptive_decode(self, prefill_ffn:bool, paired:bool)->Self {
        let mut s=self.with_mixed_execution();s.adaptive_decode=true;s.prefill_ffn_pipeline=prefill_ffn;s.decode_window=paired;s
    }
    pub const fn adaptive_decode(self)->bool {self.adaptive_decode}
    pub const fn prefill_ffn_pipeline(self) -> bool {self.prefill_ffn_pipeline}
    pub const fn with_mixed_time_budget_ns(mut self, target:Option<u64>)->Self {self.mixed_time_budget_ns=target;self}
    pub const fn mixed_time_budget_ns(self)->Option<u64> {self.mixed_time_budget_ns}
    pub const fn decode_window(self) -> bool {self.decode_window}
    pub const fn ffn_pipeline(self) -> bool { self.ffn_pipeline }
    pub const fn flashinfer_experimental(self) -> bool { self.flashinfer_experimental }
    pub const fn mixed_execution(self)->bool {self.mixed_execution}
    pub const fn packed_prefill(self)->bool {self.packed_prefill}
    pub const fn variable_graph_rows(self)->usize {self.variable_graph_rows}
    pub const fn variable_graph(self)->bool {self.variable_graph}
    /// Opt-in arithmetic-changing QKV/gate-up/head graph; bounded P128 geometry.
    #[must_use]
    pub const fn with_shared_rows_graph(mut self) -> Self {
        self.variable_graph = false;
        self.packed_prefill = false;self.mixed_execution=false;self.flashinfer_experimental=false;self.fa3_experimental=false;self.ffn_pipeline=false;self.prefill_ffn_pipeline=false;self.adaptive_decode=false;self.decode_window=false;self.mixed_time_budget_ns=None;
        self.vllm_smol_p128_graph = true;self.shared_rows_graph = true;self
    }
    #[must_use]
    pub const fn shared_rows_graph(self) -> bool { self.shared_rows_graph }
    /// Whether the bounded vLLM numerical graph is required.
    #[must_use]
    pub const fn vllm_smol_p128_graph(self) -> bool {
        self.vllm_smol_p128_graph
    }

    /// Captures all 128 prompt tokens together while retaining M=1 decode plans.
    #[must_use]
    pub const fn vllm_smol_p128_batched_prefill(self) -> bool {
        self.vllm_smol_p128_graph && self.metadata.max_input_tokens() == 128
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
        vllm_smol_p128_graph: config.vllm_smol_p128_graph,
        shared_rows_graph: config.shared_rows_graph,
        variable_graph: config.variable_graph,
        variable_graph_rows: config.variable_graph_rows,
        packed_prefill: config.packed_prefill,
        mixed_execution: config.mixed_execution,
        flashinfer_experimental: config.flashinfer_experimental,
        fa3_experimental: config.fa3_experimental,
        ffn_pipeline: config.ffn_pipeline,
        prefill_ffn_pipeline: config.prefill_ffn_pipeline,
        adaptive_decode: config.adaptive_decode,
        decode_window: config.decode_window,
        mixed_time_budget_ns: config.mixed_time_budget_ns,
        shape_policy: config.shape_policy,
        shape_buckets: config.shape_buckets,
    }
}

#[cfg(test)]
mod graph_numerical_profile_tests {
    use super::*;
    #[test]
    fn adaptive_decode_selection_is_independent_and_resets() {
        let c=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        for prefill in [false,true] {for paired in [false,true] {
            let p=normalize_prepared_config(c.with_fa3_experimental().with_adaptive_decode(prefill,paired));
            assert!(p.adaptive_decode() && p.mixed_execution());assert!(!p.fa3_experimental() && !p.flashinfer_experimental() && !p.ffn_pipeline());
            assert_eq!(p.prefill_ffn_pipeline(),prefill);assert_eq!(p.decode_window(),paired);
            for reset in [p.with_mixed_execution(),p.with_prefill_ffn_pipeline(true),p.with_fa3_experimental(),p.with_decode_window(),p.with_shared_rows_graph()] {assert!(!reset.adaptive_decode());}
        }}
    }
    #[test]
    fn fa3_profile_does_not_alias_or_survive_an_exact_profile_reset() {
        let c=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        let p=normalize_prepared_config(c.with_flashinfer_experimental().with_fa3_experimental());
        assert!(p.fa3_experimental() && p.mixed_execution() && p.packed_prefill());
        assert_eq!(p.variable_graph_rows(),32);
        assert!(!p.flashinfer_experimental() && !p.ffn_pipeline() && !p.decode_window());
        for reset in [p.with_mixed_execution(),p.with_variable_graph(),p.with_shared_rows_graph(),p.with_flashinfer_experimental(),p.with_ffn_pipeline(),p.with_decode_window()] {assert!(!reset.fa3_experimental());}
    }
    #[test]
    fn mixed_time_budget_survives_normalization_and_resets() {
        let c=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        let p=normalize_prepared_config(c.with_prefill_ffn_pipeline(true).with_mixed_time_budget_ns(Some(4_000_000)));
        assert_eq!(p.mixed_time_budget_ns(),Some(4_000_000));assert!(p.prefill_ffn_pipeline() && p.decode_window());
        assert_eq!(p.with_mixed_execution().mixed_time_budget_ns(),None);
        assert_eq!(p.with_mixed_time_budget_ns(None).mixed_time_budget_ns(),None);
    }

    #[test]
    fn prefill_ffn_selection_preserves_pairing_and_resets() {
        let c=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        for paired in [false,true] {
            let p=normalize_prepared_config(c.with_prefill_ffn_pipeline(paired));
            assert!(p.prefill_ffn_pipeline() && p.mixed_execution());
            assert_eq!(p.decode_window(),paired);
            assert!(!p.ffn_pipeline() && !p.flashinfer_experimental());
            for reset in [p.with_mixed_execution(),p.with_ffn_pipeline(),p.with_flashinfer_experimental(),p.with_decode_window()] {assert!(!reset.prefill_ffn_pipeline());}
        }
    }

    #[test]
    fn decode_window_survives_normalization_and_resets_on_graph_change() {
        let base=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,1,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        assert!(!base.decode_window());
        let paired=normalize_prepared_config(base.with_decode_window());
        assert!(paired.decode_window() && paired.mixed_execution());
        for reset in [paired.with_mixed_execution(),paired.with_variable_graph(),paired.with_shared_rows_graph(),paired.with_vllm_smol_p128_graph(),paired.with_flashinfer_experimental(),paired.with_ffn_pipeline()] {assert!(!reset.decode_window());}
    }

    #[test]
    fn ffn_profile_survives_normalization_and_cannot_leak_into_another_profile() {
        let c = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16).unwrap(),
            PreparedLlamaForwardConfig::default(),
        );
        assert!(!c.ffn_pipeline());
        let p = normalize_prepared_config(c.with_ffn_pipeline());
        assert!(p.ffn_pipeline() && p.mixed_execution() && !p.flashinfer_experimental());
        for reset in [p.with_mixed_execution(), p.with_variable_graph(), p.with_shared_rows_graph(),
            p.with_vllm_smol_p128_graph(), p.with_flashinfer_experimental()] {
            assert!(!reset.ffn_pipeline());
        }
    }

    #[test]
    fn experimental_attention_survives_normalization_and_resets_on_profile_change() {
        let c = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16).unwrap(),
            PreparedLlamaForwardConfig::default(),
        );
        assert!(!c.flashinfer_experimental());
        let experimental = normalize_prepared_config(c.with_flashinfer_experimental());
        assert!(experimental.flashinfer_experimental() && experimental.mixed_execution());
        assert_eq!(experimental.variable_graph_rows(), 32);
        for regular in [experimental.with_mixed_execution(), experimental.with_variable_graph(),
            experimental.with_shared_rows_graph(), experimental.with_vllm_smol_p128_graph()] {
            assert!(!regular.flashinfer_experimental());
        }
    }

    #[test]
    fn explicit_profile_survives_normalization_without_changing_default() {
        let c = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16).unwrap(),
            PreparedLlamaForwardConfig::default(),
        );
        assert!(!normalize_prepared_config(c).vllm_smol_p128_graph());
        assert!(!normalize_prepared_config(c).shared_rows_graph());
        assert!(normalize_prepared_config(c.with_shared_rows_graph()).shared_rows_graph());
        assert!(!normalize_prepared_config(c.with_shared_rows_graph().with_vllm_smol_p128_graph()).shared_rows_graph());
        assert!(normalize_prepared_config(c.with_vllm_smol_p128_graph()).vllm_smol_p128_graph());
    }

    #[test]
    fn batched_prefill_requires_explicit_profile_and_exact_p128_capacity() {
        for tokens in [1, 2, 127, 128, 129] {
            let config = PreparedLlamaBatchExecutorConfig::new(
                LlamaBatchMetadataConfig::new(1, tokens, 16, 1, 16).unwrap(),
                PreparedLlamaForwardConfig::default(),
            );
            assert!(!config.vllm_smol_p128_batched_prefill());
            assert_eq!(
                normalize_prepared_config(config.with_vllm_smol_p128_graph())
                    .vllm_smol_p128_batched_prefill(),
                tokens == 128
            );
        }
    }
}
