//! Host-only cold configuration for the Llama continuous-batch executor.
//!
//! This component owns scalar selection and validation. It deliberately does not own CUDA resources, model weights,
//! KV storage, metadata transfer buffers, or execution dispatch.

#![cfg_attr(all(test, not(feature = "cuda")), allow(dead_code))]

use riley_cuda::AttentionReductionProfile;

use super::super::batch::LlamaBatchMetadataConfig;
use super::super::forward::PreparedLlamaForwardConfig;
use super::super::{LlamaProjectionBiasMode, LlamaReductionProfile};
use super::error::{LlamaBatchExecutorError, LlamaBatchExecutorResult};
use super::shape::{LlamaBatchShapeBuckets, LlamaBatchShapePolicy};

const RAGGED_PAGED_ATTENTION_LEGACY_D64_V1: &str =
    "riley.cuda.ragged-paged-attention.legacy-d64-v1";
const RAGGED_PAGED_ATTENTION_GROUPED_HEADS_D64_V1: &str =
    "riley.cuda.ragged-paged-attention.grouped-heads-d64-v1";
const RAGGED_PAGED_ATTENTION_FIXED37_TWO_PASS_D64_S8192_V1: &str =
    "riley.cuda.ragged-paged-attention.fixed37-two-pass-d64-s8192-v1";
const RAGGED_PAGED_ATTENTION_NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_V2: &str = "riley.cuda.ragged-paged-attention.native-bf16-paged-split-gqa.qwen2.5-3b.d128.qh16.kvh2.block16.transition-v2";

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
    /// Explicit eager-only native-BF16 D128 two-stage GQA implementation for
    /// the Qwen2.5-3B geometry (`QH=16`, `KVH=2`, page size 16). It is a
    /// fail-closed opt-in: a different geometry is rejected during cold
    /// preparation rather than falling back to a D64 implementation.
    NativeBf16PagedSplitGqaD128TwoStage,
    /// Eager D128 reference: BF16 scores, scaled scores, probabilities and output.
    /// Owns its workspace during preparation; graph capture is unsupported.
    MaterializedBf16ReferenceD128,
    /// Explicit eager D64 reference with BF16 score/probability rounding.
    MaterializedBf16ReferenceD64,
    /// Owned BF16 QK/probability/AV with bounded row-tiled FP32 softmax.
    OwnedHfEagerBf16V1,
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
    if matches!(
        implementation,
        RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage
    ) {
        return RAGGED_PAGED_ATTENTION_NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_V2;
    }
    if matches!(implementation,RaggedAttentionImplementation::OwnedHfEagerBf16V1) {
        return "riley.cuda.owned-hf-bf16-gqa-tiled-f32.v1";
    }
    if matches!(implementation,RaggedAttentionImplementation::MaterializedBf16ReferenceD128) {
        return "riley.cuda.ragged-paged-attention.materialized-bf16-reference.d128.v1";
    }
    if matches!(implementation,RaggedAttentionImplementation::MaterializedBf16ReferenceD64) {
        return "riley.cuda.ragged-paged-attention.materialized-bf16-reference.d64.v1";
    }
    match (profile, implementation) {
        (AttentionReductionProfile::CanonicalV1, RaggedAttentionImplementation::Legacy) => {
            RAGGED_PAGED_ATTENTION_LEGACY_D64_V1
        }
        (AttentionReductionProfile::CanonicalV1, RaggedAttentionImplementation::GroupedHeads) => {
            RAGGED_PAGED_ATTENTION_GROUPED_HEADS_D64_V1
        }
        (
            AttentionReductionProfile::CanonicalV1,
            RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage,
        ) => RAGGED_PAGED_ATTENTION_NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_V2,
        (AttentionReductionProfile::CanonicalV1,RaggedAttentionImplementation::MaterializedBf16ReferenceD128) => "riley.cuda.ragged-paged-attention.materialized-bf16-reference.d128.v1",
        (AttentionReductionProfile::CanonicalV1,RaggedAttentionImplementation::MaterializedBf16ReferenceD64) => "riley.cuda.ragged-paged-attention.materialized-bf16-reference.d64.v1",
        (AttentionReductionProfile::CanonicalV1,RaggedAttentionImplementation::OwnedHfEagerBf16V1) => "riley.cuda.owned-hf-bf16-gqa-tiled-f32.v1",
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
    shape_policy: LlamaBatchShapePolicy,
    shape_buckets: LlamaBatchShapeBuckets,
    greedy_candidate_count: Option<usize>,
    d128_global_merge: bool,
    d128_tensor_core_producer: bool,
    prompt_attention_mma: bool,
    fused_silu_multiply: bool,
    fused_qkv_rope: bool,
    fused_kv_write: bool,
    fused_mlp_residual_norm: bool,
    output_row_lm_head: bool,
    serving_sequence_token_bound: Option<usize>,
    owned_mlp_graphs: bool,
    decode_step_capture: bool,
    owned_decode_graph: bool,
    owned_decode_batch: u8,
    staged_step_uploads: bool,
    dense_prompt_attention: bool,
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
            shape_policy: LlamaBatchShapePolicy::FixedMaximum,
            shape_buckets: LlamaBatchShapeBuckets::automatic(metadata.max_input_tokens()),
            greedy_candidate_count: None,
            d128_global_merge: false,
            d128_tensor_core_producer: false,
            prompt_attention_mma: false,
            fused_silu_multiply: false,
            fused_qkv_rope: false,
            fused_kv_write: false,
            fused_mlp_residual_norm: false,
            output_row_lm_head: false,
            serving_sequence_token_bound: None,
            owned_mlp_graphs: false,
            decode_step_capture: false,
            owned_decode_graph: false,
            owned_decode_batch: 0,
            staged_step_uploads: false,
            dense_prompt_attention: false,
        }
    }

    /// Enables bounded MLP-only graphs for independent M1 decode rows.
    #[must_use]
    pub const fn with_owned_mlp_graphs(mut self)->Self { self.owned_mlp_graphs=true;self }
    #[must_use]
    pub const fn owned_mlp_graphs(self)->bool { self.owned_mlp_graphs }

    /// Records the first pure-decode step of each dense-row shape as a CUDA
    /// graph and replays it for later pure-decode steps that fill the same
    /// shape. The recorded operations and their arithmetic are the eager ones;
    /// only their launch mechanism changes. Requires iteration-batch
    /// completion and device-side greedy selection.
    #[must_use]
    pub const fn with_decode_step_capture(mut self) -> Self {
        self.decode_step_capture = true;
        self
    }

    /// Replays fixed decode segments around the dynamic-length owned attention.
    /// The exact profile retains all arithmetic and request GEMM row groups.
    #[must_use]
    pub const fn with_owned_decode_graph(mut self) -> Self {
        self.owned_decode_graph = true;
        self
    }

    #[must_use]
    pub const fn owned_decode_graph(self) -> bool { self.owned_decode_graph }

    /// Submits unchanged M=1 cuBLAS calls under one native validation/lease scope.
    #[must_use]
    pub const fn with_owned_decode_batch(mut self) -> Self {
        self.owned_decode_batch = 1;
        self
    }
    #[must_use]
    pub const fn owned_decode_batch(self) -> u8 { self.owned_decode_batch }

    /// Experimental bounded Qwen Gate/Up row coalescing; independent byte gates required.
    #[must_use]
    pub const fn with_owned_mlp_decode_batch(mut self) -> Self {
        self.owned_decode_batch=2;
        self
    }

    /// Q4 SM89 exact R2 Gate/Up plus once-validated request boundaries.
    #[must_use]
    pub const fn with_exact_r2_decode_batch(mut self)->Self{self.owned_decode_batch=3;self}

    /// Opt-in SM89 Q4 R2 Down/attention-output global-load kernels for at most2
    /// simultaneous sequences. Preserves
    /// eager BF16 arithmetic; throughput and first-token latency are qualified
    /// separately. Existing mode3 and ordinary modes retain their algorithms.
    #[must_use]
    pub const fn with_exact_r2_global_decode_batch(mut self)->Self{self.owned_decode_batch=4;self}


    /// Opt-in SM89 Qwen3-4B/Mistral small-row kernels and private per-actual-row
    /// layer Graph keys, bounded to 32 sequences. Keeps eager arithmetic; first
    /// token and streaming latency tradeoffs are measured separately.
    #[must_use]
    pub const fn with_exact_smallrows_decode_batch(mut self)->Self{self.owned_decode_batch=5;self}

    /// Stages an iteration's token and metadata bytes in one pinned buffer
    /// and copies them to their device buffers inside the iteration's command
    /// batch, instead of one synchronized upload per buffer before it. Same
    /// bytes in the same device buffers. Requires synchronous metadata and
    /// iteration-batch completion.
    #[must_use]
    pub const fn with_staged_step_uploads(mut self) -> Self {
        self.staged_step_uploads = true;
        self
    }

    /// Whether token and metadata uploads are staged into the command batch.
    #[must_use]
    pub const fn staged_step_uploads(self) -> bool {
        self.staged_step_uploads
    }

    /// Whether pure-decode steps are recorded and replayed as CUDA graphs.
    #[must_use]
    pub const fn decode_step_capture(self) -> bool {
        self.decode_step_capture
    }

    /// Declares the largest sequence (prompt plus output tokens) the serving
    /// scheduler will ever admit. Per-row workspaces sized by sequence length,
    /// such as the native D128 partial-state workspace, then use this bound
    /// instead of the checkpoint's maximum context.
    #[must_use]
    pub const fn with_serving_sequence_token_bound(mut self, tokens: usize) -> Self {
        self.serving_sequence_token_bound = Some(tokens);
        self
    }

    /// The declared serving sequence bound, when one was given.
    #[must_use]
    pub const fn serving_sequence_token_bound(self) -> Option<usize> {
        self.serving_sequence_token_bound
    }

    /// Merges the native D128 per-page attention states with the single
    /// global-max kernel instead of the ordered transition + replay pair. This
    /// is a separately selected numerical mode: deterministic and independent
    /// of batch composition, but not bit-identical to the ordered merge.
    #[must_use]
    pub const fn with_d128_global_merge(mut self) -> Self {
        self.d128_global_merge = true;
        self
    }

    /// Whether the native D128 route uses the global-max page merge.
    #[must_use]
    pub const fn d128_global_merge(self) -> bool {
        self.d128_global_merge
    }

    /// Produces the native D128 page states of every row with the tensor-core
    /// producer instead of the shared-KV producer. This is a separately
    /// selected numerical mode: the page states differ in rounding only, the
    /// same way in every batch and chunk. Requires the D128 global merge and
    /// replaces the prompt-row producer.
    #[must_use]
    pub const fn with_d128_tensor_core_producer(mut self) -> Self {
        self.d128_tensor_core_producer = true;
        self
    }

    /// Whether the native D128 route uses the tensor-core page producer.
    #[must_use]
    pub const fn d128_tensor_core_producer(self) -> bool {
        self.d128_tensor_core_producer
    }

    /// Produces the attention of prompt rows with the tensor-core prefill
    /// kernel. A prompt token's attention then rounds differently from the
    /// shared-KV producer, the same way in every batch and chunk; generated
    /// tokens are unaffected. Requires the D128 global merge and synchronous
    /// metadata transport.
    #[must_use]
    pub const fn with_prompt_attention_mma(mut self) -> Self {
        self.prompt_attention_mma = true;
        self
    }

    #[must_use]
    pub const fn prompt_attention_mma(self) -> bool {
        self.prompt_attention_mma
    }

    /// Produces the attention of every prompt row with the generic prompt-row
    /// tensor-core kernel (sixteen rows of a sequence and four pages to a
    /// CTA) next to the tensor-core producer for generated rows, for any head
    /// geometry that producer takes. A prompt token's attention then rounds
    /// differently from the per-page producer, the same way in every batch
    /// and chunk; generated tokens are unaffected. Requires the tensor-core
    /// producer, the D128 global merge and synchronous metadata transport.
    #[must_use]
    pub const fn with_dense_prompt_attention(mut self) -> Self {
        self.prompt_attention_mma = true;
        self.dense_prompt_attention = true;
        self
    }

    /// Whether every prompt row takes the generic prompt-row kernel.
    #[must_use]
    pub const fn dense_prompt_attention(self) -> bool {
        self.dense_prompt_attention
    }

    /// Runs the MLP activation and gate multiply as one BF16 kernel. The
    /// result is bit-identical to the separate SiLU and multiply kernels.
    #[must_use]
    pub const fn with_fused_silu_multiply(mut self) -> Self {
        self.fused_silu_multiply = true;
        self
    }

    /// Applies the query and key `RoPE` inside the merged QKV split kernel.
    /// Only the merged QKV projection uses it; results are bit-identical to
    /// the split followed by the two `RoPE` kernels.
    #[must_use]
    pub const fn with_fused_qkv_rope(mut self) -> Self {
        self.fused_qkv_rope = true;
        self
    }

    /// Whether the merged QKV split also applies `RoPE`.
    #[must_use]
    pub const fn fused_qkv_rope(self) -> bool {
        self.fused_qkv_rope
    }

    /// Runs the KV cache write inside the merged QKV split kernel. Requires
    /// the fused split `RoPE`; the stored bytes are unchanged.
    #[must_use]
    pub const fn with_fused_kv_write(mut self) -> Self {
        self.fused_kv_write = true;
        self
    }

    #[must_use]
    pub const fn fused_kv_write(self) -> bool {
        self.fused_kv_write
    }

    /// Runs each layer's MLP residual add together with the following norm
    /// (the next layer's input norm, or the final norm) as one fused
    /// residual + `RMSNorm` kernel. Requires the fused residual norm.
    #[must_use]
    pub const fn with_fused_mlp_residual_norm(mut self) -> Self {
        self.fused_mlp_residual_norm = true;
        self
    }

    /// Whether the MLP residual add is fused with the following norm.
    #[must_use]
    pub const fn fused_mlp_residual_norm(self) -> bool {
        self.fused_mlp_residual_norm
    }

    /// In a step whose output rows are fewer than its dense rows (prefill
    /// mixed with decode), gathers the normalized output rows first and runs
    /// the LM head only on them, with the anchored plan of the smallest
    /// prepared row bucket that holds them. Logits of the output rows are
    /// bit-identical to the full-sequence LM head.
    #[must_use]
    pub const fn with_output_row_lm_head(mut self) -> Self {
        self.output_row_lm_head = true;
        self
    }

    /// Whether mixed steps run the LM head on output rows only.
    #[must_use]
    pub const fn output_row_lm_head(self) -> bool {
        self.output_row_lm_head
    }

    /// Whether SiLU and the gate multiply run as one kernel.
    #[must_use]
    pub const fn fused_silu_multiply(self) -> bool {
        self.fused_silu_multiply
    }

    /// Restricts device-side greedy selection to token IDs below `count`, for
    /// a tokenizer that addresses fewer IDs than the logit row holds.
    #[must_use]
    pub const fn with_greedy_candidate_count(mut self, count: usize) -> Self {
        self.greedy_candidate_count = Some(count);
        self
    }

    /// Device-side greedy candidate bound, when one is configured.
    #[must_use]
    pub const fn greedy_candidate_count(self) -> Option<usize> {
        self.greedy_candidate_count
    }

    /// Selects the explicit VllmSmolP128V1 owned graph. Eager fallback is forbidden.
    #[must_use]
    pub const fn with_vllm_smol_p128_graph(mut self) -> Self {
        self.vllm_smol_p128_graph = true;
        self
    }
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

    /// Selects the explicit eager-only Qwen2.5-3B D128 native-BF16 GQA
    /// backend. This changes no default selection and never permits a
    /// D64-style fallback for a mismatched model geometry.
    #[must_use]
    pub const fn with_native_bf16_paged_split_gqa_d128_two_stage(mut self) -> Self {
        self.ragged_attention_implementation =
            RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage;
        self
    }

    /// Selects the complete owned arithmetic profile. Other configuration
    /// setters cannot silently replace one component: validate rejects mixes.
    #[must_use]
    pub const fn with_owned_hf_eager_bf16(mut self) -> Self {
        self.forward = self.forward.with_owned_hf_eager_bf16();
        self.ragged_attention_implementation = RaggedAttentionImplementation::OwnedHfEagerBf16V1;
        self
    }

    #[must_use]
    pub const fn owned_hf_eager_bf16(self) -> bool {
        matches!(self.ragged_attention_implementation,RaggedAttentionImplementation::OwnedHfEagerBf16V1)
    }

    /// Selects an eager D128 reference with explicit BF16 probability rounding.
    /// Canonical reductions and graph-disabled execution are required.
    #[must_use]
    pub const fn with_materialized_bf16_reference_d128(mut self) -> Self {
        self.ragged_attention_implementation=RaggedAttentionImplementation::MaterializedBf16ReferenceD128;
        self
    }

    #[must_use]
    pub const fn materialized_bf16_reference_d128(self) -> bool {
        matches!(self.ragged_attention_implementation,RaggedAttentionImplementation::MaterializedBf16ReferenceD128)
    }

    /// Selects the explicit eager D64 BF16 materialized reference.
    #[must_use]
    pub const fn with_materialized_bf16_reference_d64(mut self) -> Self {
        self.ragged_attention_implementation=RaggedAttentionImplementation::MaterializedBf16ReferenceD64;
        self
    }
    #[must_use]
    pub const fn materialized_bf16_reference_d64(self) -> bool {
        matches!(self.ragged_attention_implementation,RaggedAttentionImplementation::MaterializedBf16ReferenceD64)
    }
    pub(in crate::llama) const fn uses_paged_attention_workspace(self) -> bool {
        self.uses_d128_paged_attention() || self.materialized_bf16_reference_d64() || self.owned_hf_eager_bf16()
    }

    /// Returns the canonical ragged attention launch implementation.
    #[must_use]
    pub const fn ragged_attention_implementation(self) -> RaggedAttentionImplementation {
        self.ragged_attention_implementation
    }

    /// Whether this cold configuration selected the explicit D128 two-stage
    /// backend. The owner uses this only to admit the exact model geometry and
    /// allocate its dedicated eager workspace.
    #[must_use]
    pub const fn native_bf16_paged_split_gqa_d128_two_stage(self) -> bool {
        matches!(self.ragged_attention_implementation,RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage)
    }

    /// Common cold geometry/resource admission; it does not identify the arithmetic backend.
    pub(in crate::llama) const fn uses_d128_paged_attention(self) -> bool {
        matches!(
            self.ragged_attention_implementation,
            RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage
                | RaggedAttentionImplementation::MaterializedBf16ReferenceD128
        )
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
        if self.staged_step_uploads
            && (self.metadata_transport != BatchMetadataTransport::Synchronous
                || self.execution_completion != ExecutionCompletionImplementation::IterationBatch)
        {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "staged_step_uploads",
                reason: "staged step uploads require synchronous metadata and iteration-batch completion",
            });
        }
        Ok(())
    }

    /// Rejects unsupported profile combinations before any CUDA resource is
    /// allocated. Geometry is deliberately checked by the prepared owner,
    /// where the immutable model dimensions are available.
    pub(in crate::llama) fn validate_attention_implementation(
        self,
    ) -> LlamaBatchExecutorResult<()> {
        if (self.owned_decode_graph || self.owned_decode_batch != 0) && (!self.owned_hf_eager_bf16()
            || self.execution_completion != ExecutionCompletionImplementation::IterationBatch) {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"owned_decode_graph",reason:"requires owned profile and iteration-batch completion"});
        }
        if self.owned_mlp_graphs && (!self.owned_hf_eager_bf16()
            || self.execution_completion!=ExecutionCompletionImplementation::IterationBatch
            || !self.fused_silu_multiply || !cfg!(feature="cuda-cublas-gemm-probe")) {
            return Err(LlamaBatchExecutorError::InvalidConfiguration{field:"owned_mlp_graphs",reason:"owned HF iteration-batch completion and fused SiLU required"});
        }
        if self.owned_hf_eager_bf16() != self.forward.owned_hf_eager_bf16() {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"owned-hf-eager-bf16-v1",reason:"attention and GEMM/norm profile must be selected together"});
        }
        if self.owned_hf_eager_bf16() && (!cfg!(feature="cuda-cublas-gemm-probe")
            || self.metadata_transport != BatchMetadataTransport::Synchronous
            || self.residual_norm != ResidualNormImplementation::Separate
            || self.staged_step_uploads || self.decode_step_capture || self.fused_mlp_residual_norm
            || self.fused_kv_write) {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"owned-hf-eager-bf16-v1",reason:"requires linked owned CUDA path, synchronous metadata and separate norms without capture"});
        }
        if (self.materialized_bf16_reference_d128() || self.materialized_bf16_reference_d64() || self.owned_hf_eager_bf16())
            && (self.decode_step_capture || self.vllm_smol_p128_graph || self.d128_global_merge || self.d128_tensor_core_producer || self.prompt_attention_mma) {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {field:"materialized_bf16_reference_d128",reason:"reference attention requires eager execution without native producer/merge overrides"});
        }
        if self.uses_paged_attention_workspace()
            && self.ragged_attention_reduction_profile != AttentionReductionProfile::CanonicalV1
        {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "ragged_attention_implementation",
                reason: "native D128 two-stage attention requires canonical-v1 reductions",
            });
        }
        if self.uses_paged_attention_workspace() && self.vllm_smol_p128_graph {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "vllm-smol-p128-v1",
                reason: "native D128 two-stage attention is eager-only and cannot use the D64 graph",
            });
        }
        if self.forward.projection_bias_mode().merges_qkv() && self.vllm_smol_p128_graph {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "vllm-smol-p128-v1",
                reason: "merged QKV projection is eager-only and cannot use the graph",
            });
        }
        if self.forward.projection_bias_mode()
            == LlamaProjectionBiasMode::CublasLtBiasEpilogueExperimentalV1
            && self.vllm_smol_p128_graph
        {
            return Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "vllm-smol-p128-v1",
                reason: "cuBLASLt Q/K/V bias epilogue is eager-only and cannot use the graph",
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
        shape_policy: config.shape_policy,
        shape_buckets: config.shape_buckets,
        greedy_candidate_count: config.greedy_candidate_count,
        d128_global_merge: config.d128_global_merge,
        d128_tensor_core_producer: config.d128_tensor_core_producer,
        prompt_attention_mma: config.prompt_attention_mma,
        fused_silu_multiply: config.fused_silu_multiply,
        fused_qkv_rope: config.fused_qkv_rope,
        fused_kv_write: config.fused_kv_write,
        fused_mlp_residual_norm: config.fused_mlp_residual_norm,
        output_row_lm_head: config.output_row_lm_head,
        serving_sequence_token_bound: config.serving_sequence_token_bound,
        owned_mlp_graphs: config.owned_mlp_graphs,
        decode_step_capture: config.decode_step_capture,
        owned_decode_graph: config.owned_decode_graph,
        owned_decode_batch: config.owned_decode_batch,
        staged_step_uploads: config.staged_step_uploads,
        dense_prompt_attention: config.dense_prompt_attention,
    }
}

#[cfg(test)]
mod graph_numerical_profile_tests {
    use super::*;
    #[test]
    fn d64_reference_is_explicit_eager_and_does_not_admit_d128_native_modes() {
        let base=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,128,16,1,16).unwrap(),PreparedLlamaForwardConfig::default());
        assert!(!base.uses_paged_attention_workspace());
        let config=base.with_materialized_bf16_reference_d64();
        assert!(config.validate_attention_implementation().is_ok());
        assert!(!config.uses_d128_paged_attention());
        assert!(config.uses_paged_attention_workspace());
        assert!(!config.materialized_bf16_reference_d128());
        for invalid in [config.with_decode_step_capture(),config.with_vllm_smol_p128_graph(),config.with_d128_global_merge(),config.with_d128_tensor_core_producer(),config.with_fixed37_reductions()] { assert!(invalid.validate_attention_implementation().is_err()); }
    }

    #[test]
    fn materialized_reference_rejects_incompatible_modes_before_cuda_preparation() {
        let config=PreparedLlamaBatchExecutorConfig::new(LlamaBatchMetadataConfig::new(1,128,16,1,16).unwrap(),PreparedLlamaForwardConfig::default()).with_materialized_bf16_reference_d128();
        assert!(config.validate_attention_implementation().is_ok());
        assert!(!config.native_bf16_paged_split_gqa_d128_two_stage());
        assert!(config.uses_d128_paged_attention());
        for invalid in [config.with_decode_step_capture(),config.with_vllm_smol_p128_graph(),config.with_d128_global_merge(),config.with_d128_tensor_core_producer(),config.with_fixed37_reductions()] {
            assert!(invalid.validate_attention_implementation().is_err());
        }
    }

    #[test]
    fn explicit_profile_survives_normalization_without_changing_default() {
        let c = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16).unwrap(),
            PreparedLlamaForwardConfig::default(),
        );
        assert!(!normalize_prepared_config(c).vllm_smol_p128_graph());
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

    #[test]
    fn native_d128_two_stage_is_explicit_reversible_and_fail_closed_for_profiles() {
        let defaults = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 1, 16, 1, 16).unwrap(),
            PreparedLlamaForwardConfig::default(),
        );
        assert_eq!(
            defaults.ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
        let native = defaults.with_native_bf16_paged_split_gqa_d128_two_stage();
        assert_eq!(
            native.ragged_attention_implementation(),
            RaggedAttentionImplementation::NativeBf16PagedSplitGqaD128TwoStage
        );
        assert_eq!(
            ragged_attention_implementation_id(
                native.ragged_attention_reduction_profile(),
                native.ragged_attention_implementation(),
            ),
            RAGGED_PAGED_ATTENTION_NATIVE_BF16_PAGED_SPLIT_GQA_D128_TWO_STAGE_V2
        );
        native
            .validate_attention_implementation()
            .expect("canonical native D128 selection is valid before geometry admission");
        assert!(matches!(
            native
                .with_fixed37_reductions()
                .validate_attention_implementation(),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "ragged_attention_implementation",
                ..
            })
        ));
        assert!(matches!(
            native
                .with_vllm_smol_p128_graph()
                .validate_attention_implementation(),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "vllm-smol-p128-v1",
                ..
            })
        ));
        assert_eq!(
            native
                .with_legacy_ragged_attention_heads()
                .ragged_attention_implementation(),
            RaggedAttentionImplementation::Legacy
        );
    }

    #[test]
    fn fused_qkv_bias_epilogue_rejects_the_existing_graph_selection() {
        let forward =
            PreparedLlamaForwardConfig::default().with_cublaslt_bias_epilogue_projection_bias();
        let config = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1, 128, 16, 1, 16).unwrap(),
            forward,
        );
        config
            .validate_attention_implementation()
            .expect("eager fused bias selection is valid before model admission");
        assert!(matches!(
            config
                .with_vllm_smol_p128_graph()
                .validate_attention_implementation(),
            Err(LlamaBatchExecutorError::InvalidConfiguration {
                field: "vllm-smol-p128-v1",
                ..
            })
        ));
    }
}

#[cfg(test)]
mod owned_hf_profile_contract_tests {
    use super::*;
    #[test]
    fn owned_hf_profile_rejects_partial_selection_and_execution_overrides() {
        let base = PreparedLlamaBatchExecutorConfig::new(
            LlamaBatchMetadataConfig::new(1,8192,576,1,576).unwrap(), PreparedLlamaForwardConfig::default());
        let own = base.with_owned_hf_eager_bf16();
        assert_eq!(own.validate_attention_implementation().is_ok(), cfg!(feature="cuda-cublas-gemm-probe"));
        assert_eq!(ragged_attention_implementation_id(own.ragged_attention_reduction_profile(),own.ragged_attention_implementation()),"riley.cuda.owned-hf-bf16-gqa-tiled-f32.v1");
        assert!(own.uses_paged_attention_workspace());
        assert!(base.with_owned_decode_graph().validate_attention_implementation().is_err());
        assert!(base.with_owned_decode_batch().validate_attention_implementation().is_err());
        assert!(own.with_owned_decode_batch().validate_attention_implementation().is_err());
        assert_eq!(own.with_iteration_batch_completion().with_owned_decode_batch().validate_attention_implementation().is_ok(), cfg!(feature="cuda-cublas-gemm-probe"));

        assert!(own.with_owned_decode_graph().validate_attention_implementation().is_err());
        assert_eq!(own.with_iteration_batch_completion().with_owned_decode_graph().validate_attention_implementation().is_ok(), cfg!(feature="cuda-cublas-gemm-probe"));

        assert_eq!(own.with_iteration_batch_completion().validate_attention_implementation().is_ok(), cfg!(feature="cuda-cublas-gemm-probe"));
        assert_eq!(own.with_fused_silu_multiply().validate_attention_implementation().is_ok(), cfg!(feature="cuda-cublas-gemm-probe"));
        let graphed=own.with_fused_silu_multiply().with_iteration_batch_completion().with_owned_mlp_graphs();
        assert_eq!(graphed.validate_attention_implementation().is_ok(),cfg!(feature="cuda-cublas-gemm-probe"));
        assert!(super::normalize_prepared_config(graphed).owned_mlp_graphs());
        assert!(own.with_owned_mlp_graphs().validate_attention_implementation().is_err());
        assert!(own.with_fused_silu_multiply().with_owned_mlp_graphs().validate_attention_implementation().is_err());
        for bad in [own.with_fused_residual_norm(),
            own.with_decode_step_capture(),own.with_vllm_smol_p128_graph(),own.with_d128_global_merge(),
            own.with_d128_tensor_core_producer(),own.with_fixed37_reductions(),own.with_legacy_ragged_attention_heads(),
            own.with_materialized_bf16_reference_d64()] {
            assert!(bad.validate_attention_implementation().is_err());
        }
    }
}
