#[cfg(any(feature = "cuda", test))]
use rustinfer_model::{Activation, DecoderWeight, ModelSpec, NormKind, RopeLayout, WeightSlot};
#[cfg(any(feature = "cuda", test))]
use rustinfer_tensor::DType;

#[cfg(any(feature = "cuda", test))]
use super::error::{
    ExecutionSite, LlamaBufferRole, LlamaDimension, LlamaOp, LlamaPlanError, LlamaPlanResult,
    LlamaScalar,
};

#[cfg(any(feature = "cuda", test))]
const BF16_BYTES: u64 = 2;
#[cfg(any(feature = "cuda", test))]
const F32_BYTES: u64 = 4;
#[cfg(any(feature = "cuda", test))]
const U32_BYTES: u64 = 4;
#[cfg(any(feature = "cuda", test))]
const EMBEDDING_ERROR_SCRATCH_BYTES: u64 = 32;

/// Number of independent `[S, H]` buffers required by the reference graph.
pub const HIDDEN_WORKSPACE_BUFFER_COUNT: u64 = 5;
/// Number of independent `[S, KVH * D]` buffers required by the graph.
pub const KEY_VALUE_WORKSPACE_BUFFER_COUNT: u64 = 3;
/// Number of independent `[S, I]` buffers required by the unfused gated MLP.
pub const INTERMEDIATE_WORKSPACE_BUFFER_COUNT: u64 = 4;

/// Owner-bound opaque identity of one uploaded physical tensor.
///
/// This type is deliberately crate-private. The final owning CUDA forward must
/// keep the plan and the exact `CudaUploadedWeights` used to construct it in
/// the same object.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct PhysicalWeightId {
    owner: u64,
    index: usize,
}

#[cfg(any(feature = "cuda", test))]
impl PhysicalWeightId {
    pub(crate) const fn new(owner: u64, index: usize) -> Self {
        Self { owner, index }
    }

    pub(crate) const fn owner(self) -> u64 {
        self.owner
    }

    pub(crate) const fn index(self) -> usize {
        self.index
    }
}

/// Borrowed cold-path metadata for one physical uploaded weight.
#[cfg(any(feature = "cuda", test))]
pub(crate) struct PhysicalWeightMetadata<'a> {
    pub(crate) dtype: DType,
    pub(crate) shape: &'a [usize],
    pub(crate) byte_len: u64,
}

#[cfg(any(feature = "cuda", test))]
pub(crate) trait PlanWeightCatalog {
    fn resolve_slot(&self, slot: WeightSlot) -> Option<PhysicalWeightId>;
    fn physical_metadata(&self, id: PhysicalWeightId) -> Option<PhysicalWeightMetadata<'_>>;
}

/// Immutable dimensions shared by every decoder block in one prepared plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaDimensions {
    hidden_size: usize,
    intermediate_size: usize,
    vocabulary_size: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    key_value_width: usize,
    group_size: usize,
}

impl LlamaDimensions {
    #[must_use]
    pub const fn hidden_size(self) -> usize {
        self.hidden_size
    }

    #[must_use]
    pub const fn intermediate_size(self) -> usize {
        self.intermediate_size
    }

    #[must_use]
    pub const fn vocabulary_size(self) -> usize {
        self.vocabulary_size
    }

    #[must_use]
    pub const fn query_heads(self) -> usize {
        self.query_heads
    }

    #[must_use]
    pub const fn key_value_heads(self) -> usize {
        self.key_value_heads
    }

    #[must_use]
    pub const fn head_dimension(self) -> usize {
        self.head_dimension
    }

    #[must_use]
    pub const fn key_value_width(self) -> usize {
        self.key_value_width
    }

    #[must_use]
    pub const fn group_size(self) -> usize {
        self.group_size
    }
}

/// Exact preallocated byte requirements for a fixed-length reference forward.
///
/// Attention scores transition in-place to probabilities in one BF16 buffer.
/// Cosine/sine tables are F32, token IDs are U32, and all activations/logits
/// are BF16. GEMM algorithm workspace is executor-owned and is intentionally
/// not included because it is known only after CUDA algorithm preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaWorkspaceSpec {
    token_ids_bytes: u64,
    hidden_buffer_bytes: u64,
    key_value_buffer_bytes: u64,
    intermediate_buffer_bytes: u64,
    attention_buffer_bytes: u64,
    rope_cos_bytes: u64,
    rope_sin_bytes: u64,
    logits_bytes: u64,
    embedding_error_scratch_bytes: u64,
    non_attention_planned_bytes: u64,
    total_planned_bytes: u64,
}

impl LlamaWorkspaceSpec {
    #[must_use]
    pub const fn token_ids_bytes(self) -> u64 {
        self.token_ids_bytes
    }

    /// Bytes in each of the five independent hidden-width buffers.
    #[must_use]
    pub const fn hidden_buffer_bytes(self) -> u64 {
        self.hidden_buffer_bytes
    }

    /// Bytes in each of the three independent key/value-width buffers.
    #[must_use]
    pub const fn key_value_buffer_bytes(self) -> u64 {
        self.key_value_buffer_bytes
    }

    /// Bytes in each of the four independent intermediate-width buffers.
    #[must_use]
    pub const fn intermediate_buffer_bytes(self) -> u64 {
        self.intermediate_buffer_bytes
    }

    /// BF16 bytes for `[query_heads, sequence, sequence]` scores, transformed
    /// in-place into causal probabilities before the value product.
    #[must_use]
    pub const fn attention_buffer_bytes(self) -> u64 {
        self.attention_buffer_bytes
    }

    /// F32 bytes in the cosine table `[sequence, head_dimension / 2]`.
    #[must_use]
    pub const fn rope_cos_bytes(self) -> u64 {
        self.rope_cos_bytes
    }

    /// F32 bytes in the sine table `[sequence, head_dimension / 2]`.
    #[must_use]
    pub const fn rope_sin_bytes(self) -> u64 {
        self.rope_sin_bytes
    }

    /// BF16 bytes for full-sequence logits `[sequence, vocabulary]`.
    #[must_use]
    pub const fn logits_bytes(self) -> u64 {
        self.logits_bytes
    }

    #[must_use]
    pub const fn embedding_error_scratch_bytes(self) -> u64 {
        self.embedding_error_scratch_bytes
    }

    /// Sum of every fixed graph allocation except the optional materialized
    /// attention workspace. Online prefill backends use this exact base.
    #[must_use]
    pub const fn non_attention_planned_bytes(self) -> u64 {
        self.non_attention_planned_bytes
    }

    /// Sum of all graph, table, input, and output buffers described above.
    #[must_use]
    pub const fn total_planned_bytes(self) -> u64 {
        self.total_planned_bytes
    }
}

/// Immutable weights and scalar metadata for one decoder block.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaLayerPlan {
    index: usize,
    input_norm: PhysicalWeightId,
    query: PhysicalWeightId,
    query_bias: Option<PhysicalWeightId>,
    key: PhysicalWeightId,
    key_bias: Option<PhysicalWeightId>,
    value: PhysicalWeightId,
    value_bias: Option<PhysicalWeightId>,
    output: PhysicalWeightId,
    output_bias: Option<PhysicalWeightId>,
    post_attention_norm: PhysicalWeightId,
    gate: PhysicalWeightId,
    up: PhysicalWeightId,
    down: PhysicalWeightId,
    input_norm_epsilon: f32,
    post_attention_norm_epsilon: f32,
}

#[allow(dead_code)]
impl LlamaLayerPlan {
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    #[must_use]
    pub const fn input_norm_epsilon(&self) -> f32 {
        self.input_norm_epsilon
    }

    #[must_use]
    pub const fn post_attention_norm_epsilon(&self) -> f32 {
        self.post_attention_norm_epsilon
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn input_norm_weight(&self) -> PhysicalWeightId {
        self.input_norm
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn query_weight(&self) -> PhysicalWeightId {
        self.query
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn query_bias(&self) -> Option<PhysicalWeightId> {
        self.query_bias
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn key_weight(&self) -> PhysicalWeightId {
        self.key
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn key_bias(&self) -> Option<PhysicalWeightId> {
        self.key_bias
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn value_weight(&self) -> PhysicalWeightId {
        self.value
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn value_bias(&self) -> Option<PhysicalWeightId> {
        self.value_bias
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn output_weight(&self) -> PhysicalWeightId {
        self.output
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn output_bias(&self) -> Option<PhysicalWeightId> {
        self.output_bias
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn post_attention_norm_weight(&self) -> PhysicalWeightId {
        self.post_attention_norm
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn gate_weight(&self) -> PhysicalWeightId {
        self.gate
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn up_weight(&self) -> PhysicalWeightId {
        self.up
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn down_weight(&self) -> PhysicalWeightId {
        self.down
    }
}

/// Fixed-sequence, immutable Llama reference-forward topology.
#[derive(Debug, PartialEq)]
pub struct LlamaExecutionPlan {
    sequence_length: usize,
    dimensions: LlamaDimensions,
    embedding: PhysicalWeightId,
    layers: Box<[LlamaLayerPlan]>,
    final_norm: PhysicalWeightId,
    lm_head: PhysicalWeightId,
    final_norm_epsilon: f32,
    rope_theta: f32,
    logical_weight_count: usize,
    workspace: LlamaWorkspaceSpec,
}

#[allow(dead_code)]
impl LlamaExecutionPlan {
    /// Builds a cold immutable plan over one exact uploaded weight owner.
    ///
    /// # Errors
    ///
    /// Returns before execution for unsupported dtype/MLP bias/RoPE, inconsistent
    /// dimensions or layer order, missing/mismatched weights, invalid fixed
    /// sequence length, or checked workspace arithmetic overflow.
    #[cfg(any(feature = "cuda", test))]
    pub fn prepare(
        spec: &ModelSpec,
        weights: &crate::cuda_weights::CudaUploadedWeights,
        sequence_length: usize,
    ) -> LlamaPlanResult<Self> {
        Self::prepare_with_catalog(spec, weights, sequence_length)
    }

    #[cfg(any(feature = "cuda", test))]
    #[allow(clippy::too_many_lines)]
    fn prepare_with_catalog<C: PlanWeightCatalog>(
        spec: &ModelSpec,
        weights: &C,
        sequence_length: usize,
    ) -> LlamaPlanResult<Self> {
        if spec.dtype() != DType::BF16 {
            return Err(LlamaPlanError::UnsupportedDType {
                actual: spec.dtype(),
            });
        }
        if sequence_length == 0 || sequence_length > spec.max_sequence_length() {
            return Err(LlamaPlanError::InvalidSequenceLength {
                requested: sequence_length,
                maximum: spec.max_sequence_length(),
            });
        }

        let embedding_spec = spec.embedding();
        let hidden_size = embedding_spec.hidden_size();
        let vocabulary_size = embedding_spec.vocabulary_size();
        require_positive(
            ExecutionSite::global(LlamaOp::Embedding),
            LlamaDimension::HiddenSize,
            hidden_size,
        )?;
        require_positive(
            ExecutionSite::global(LlamaOp::Embedding),
            LlamaDimension::VocabularySize,
            vocabulary_size,
        )?;

        let first = spec
            .blocks()
            .first()
            .ok_or(LlamaPlanError::InvalidDimension {
                site: ExecutionSite::global(LlamaOp::InputNorm),
                dimension: LlamaDimension::QueryHeads,
            })?;
        let first_attention = first.attention();
        let query_heads = first_attention.query_heads();
        let key_value_heads = first_attention.key_value_heads();
        let head_dimension = first_attention.head_dimension();
        let intermediate_size = first.mlp().intermediate_size();
        let dimensions = validate_global_dimensions(
            hidden_size,
            intermediate_size,
            vocabulary_size,
            query_heads,
            key_value_heads,
            head_dimension,
        )?;

        validate_lm_head(spec, dimensions)?;
        let final_norm_epsilon = validate_norm(
            spec.final_norm(),
            hidden_size,
            ExecutionSite::global(LlamaOp::FinalNorm),
        )?;
        let first_rope = first_attention.rope();
        let rope_theta = finite_positive_f32(
            first_rope.theta(),
            ExecutionSite::layer(0, LlamaOp::QueryRope),
            LlamaScalar::RopeTheta,
        )?;

        let embedding = bind_weight(
            weights,
            ExecutionSite::global(LlamaOp::Embedding),
            WeightSlot::TokenEmbedding,
            &[vocabulary_size, hidden_size],
        )?;

        let mut layers = Vec::with_capacity(spec.blocks().len());
        for (ordinal, block) in spec.blocks().iter().enumerate() {
            if block.index() != ordinal {
                return Err(LlamaPlanError::LayerIndexMismatch {
                    ordinal,
                    declared: block.index(),
                });
            }
            layers.push(build_layer_plan(
                block,
                weights,
                dimensions,
                spec.max_sequence_length(),
                first_rope.theta(),
            )?);
        }

        let final_norm = bind_weight(
            weights,
            ExecutionSite::global(LlamaOp::FinalNorm),
            WeightSlot::FinalNormScale,
            &[hidden_size],
        )?;
        let lm_head = bind_weight(
            weights,
            ExecutionSite::global(LlamaOp::LmHead),
            WeightSlot::LmHead,
            &[vocabulary_size, hidden_size],
        )?;
        if spec.lm_head().tied_to_embedding() && embedding != lm_head {
            return Err(LlamaPlanError::TiedWeightIdentityMismatch);
        }

        let logical_weight_count = layers.iter().try_fold(3_usize, |count, layer| {
            let projection_biases = [
                layer.query_bias,
                layer.key_bias,
                layer.value_bias,
                layer.output_bias,
            ]
            .into_iter()
            .flatten()
            .count();
            count
                .checked_add(9)
                .and_then(|count| count.checked_add(projection_biases))
                .ok_or(LlamaPlanError::LogicalWeightCountOverflow)
        })?;
        let workspace = build_workspace_spec(sequence_length, dimensions)?;

        Ok(Self {
            sequence_length,
            dimensions,
            embedding,
            layers: layers.into_boxed_slice(),
            final_norm,
            lm_head,
            final_norm_epsilon,
            rope_theta,
            logical_weight_count,
            workspace,
        })
    }

    #[must_use]
    pub const fn sequence_length(&self) -> usize {
        self.sequence_length
    }

    #[must_use]
    pub const fn dimensions(&self) -> LlamaDimensions {
        self.dimensions
    }

    #[must_use]
    pub const fn layers(&self) -> &[LlamaLayerPlan] {
        &self.layers
    }

    #[must_use]
    pub const fn final_norm_epsilon(&self) -> f32 {
        self.final_norm_epsilon
    }

    #[must_use]
    pub const fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    #[must_use]
    pub const fn logical_weight_count(&self) -> usize {
        self.logical_weight_count
    }

    #[must_use]
    pub const fn workspace_spec(&self) -> LlamaWorkspaceSpec {
        self.workspace
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn embedding_weight(&self) -> PhysicalWeightId {
        self.embedding
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn final_norm_weight(&self) -> PhysicalWeightId {
        self.final_norm
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) const fn lm_head_weight(&self) -> PhysicalWeightId {
        self.lm_head
    }
}

#[cfg(any(feature = "cuda", test))]
fn validate_global_dimensions(
    hidden_size: usize,
    intermediate_size: usize,
    vocabulary_size: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
) -> LlamaPlanResult<LlamaDimensions> {
    let site = ExecutionSite::layer(0, LlamaOp::AttentionScores);
    for (dimension, value) in [
        (LlamaDimension::IntermediateSize, intermediate_size),
        (LlamaDimension::QueryHeads, query_heads),
        (LlamaDimension::KeyValueHeads, key_value_heads),
        (LlamaDimension::HeadDimension, head_dimension),
    ] {
        require_positive(site, dimension, value)?;
    }
    if query_heads % key_value_heads != 0 {
        return Err(LlamaPlanError::InvalidDimension {
            site,
            dimension: LlamaDimension::KeyValueHeads,
        });
    }
    if head_dimension % 2 != 0 {
        return Err(LlamaPlanError::InvalidDimension {
            site,
            dimension: LlamaDimension::HeadDimension,
        });
    }
    let query_width =
        query_heads
            .checked_mul(head_dimension)
            .ok_or(LlamaPlanError::InvalidDimension {
                site,
                dimension: LlamaDimension::QueryWidth,
            })?;
    require_equal(site, LlamaDimension::QueryWidth, hidden_size, query_width)?;
    let key_value_width =
        key_value_heads
            .checked_mul(head_dimension)
            .ok_or(LlamaPlanError::InvalidDimension {
                site,
                dimension: LlamaDimension::KeyValueWidth,
            })?;
    Ok(LlamaDimensions {
        hidden_size,
        intermediate_size,
        vocabulary_size,
        query_heads,
        key_value_heads,
        head_dimension,
        key_value_width,
        group_size: query_heads / key_value_heads,
    })
}

#[cfg(any(feature = "cuda", test))]
fn validate_lm_head(spec: &ModelSpec, dimensions: LlamaDimensions) -> LlamaPlanResult<()> {
    let site = ExecutionSite::global(LlamaOp::LmHead);
    require_equal(
        site,
        LlamaDimension::LmHeadHiddenSize,
        dimensions.hidden_size,
        spec.lm_head().hidden_size(),
    )?;
    require_equal(
        site,
        LlamaDimension::LmHeadVocabularySize,
        dimensions.vocabulary_size,
        spec.lm_head().vocabulary_size(),
    )
}

#[cfg(any(feature = "cuda", test))]
struct AttentionWeightBindings {
    query: PhysicalWeightId,
    query_bias: Option<PhysicalWeightId>,
    key: PhysicalWeightId,
    key_bias: Option<PhysicalWeightId>,
    value: PhysicalWeightId,
    value_bias: Option<PhysicalWeightId>,
    output: PhysicalWeightId,
    output_bias: Option<PhysicalWeightId>,
}

#[cfg(any(feature = "cuda", test))]
fn build_layer_plan<C: PlanWeightCatalog>(
    block: &rustinfer_model::DecoderBlockSpec,
    weights: &C,
    dimensions: LlamaDimensions,
    model_max_sequence_length: usize,
    expected_rope_theta: f64,
) -> LlamaPlanResult<LlamaLayerPlan> {
    let layer = block.index();
    let input_norm_site = ExecutionSite::layer(layer, LlamaOp::InputNorm);
    let input_norm_epsilon =
        validate_norm(block.input_norm(), dimensions.hidden_size, input_norm_site)?;
    let post_norm_site = ExecutionSite::layer(layer, LlamaOp::PostAttentionNorm);
    let post_attention_norm_epsilon = validate_norm(
        block.post_attention_norm(),
        dimensions.hidden_size,
        post_norm_site,
    )?;
    validate_attention(
        block.attention(),
        dimensions,
        model_max_sequence_length,
        expected_rope_theta,
        layer,
    )?;
    validate_mlp(block.mlp(), dimensions, layer)?;

    let decoder = |parameter| WeightSlot::Decoder { layer, parameter };
    let hidden = dimensions.hidden_size;
    let intermediate = dimensions.intermediate_size;
    let attention_weights = bind_attention_weights(
        block.attention(),
        weights,
        layer,
        hidden,
        dimensions.key_value_width,
    )?;
    Ok(LlamaLayerPlan {
        index: layer,
        input_norm: bind_weight(
            weights,
            input_norm_site,
            decoder(DecoderWeight::InputNormScale),
            &[hidden],
        )?,
        query: attention_weights.query,
        query_bias: attention_weights.query_bias,
        key: attention_weights.key,
        key_bias: attention_weights.key_bias,
        value: attention_weights.value,
        value_bias: attention_weights.value_bias,
        output: attention_weights.output,
        output_bias: attention_weights.output_bias,
        post_attention_norm: bind_weight(
            weights,
            post_norm_site,
            decoder(DecoderWeight::PostAttentionNormScale),
            &[hidden],
        )?,
        gate: bind_weight(
            weights,
            ExecutionSite::layer(layer, LlamaOp::GateProjection),
            decoder(DecoderWeight::GateWeight),
            &[intermediate, hidden],
        )?,
        up: bind_weight(
            weights,
            ExecutionSite::layer(layer, LlamaOp::UpProjection),
            decoder(DecoderWeight::UpWeight),
            &[intermediate, hidden],
        )?,
        down: bind_weight(
            weights,
            ExecutionSite::layer(layer, LlamaOp::DownProjection),
            decoder(DecoderWeight::DownWeight),
            &[hidden, intermediate],
        )?,
        input_norm_epsilon,
        post_attention_norm_epsilon,
    })
}

#[cfg(any(feature = "cuda", test))]
fn bind_attention_weights<C: PlanWeightCatalog>(
    attention: &rustinfer_model::AttentionSpec,
    weights: &C,
    layer: usize,
    hidden: usize,
    key_value_width: usize,
) -> LlamaPlanResult<AttentionWeightBindings> {
    let decoder = |parameter| WeightSlot::Decoder { layer, parameter };
    let query_site = ExecutionSite::layer(layer, LlamaOp::QueryProjection);
    let key_site = ExecutionSite::layer(layer, LlamaOp::KeyProjection);
    let value_site = ExecutionSite::layer(layer, LlamaOp::ValueProjection);
    let output_site = ExecutionSite::layer(layer, LlamaOp::OutputProjection);
    Ok(AttentionWeightBindings {
        query: bind_weight(
            weights,
            query_site,
            decoder(DecoderWeight::QueryWeight),
            &[hidden, hidden],
        )?,
        query_bias: bind_optional_weight(
            attention.bias().query(),
            weights,
            query_site,
            decoder(DecoderWeight::QueryBias),
            &[hidden],
        )?,
        key: bind_weight(
            weights,
            key_site,
            decoder(DecoderWeight::KeyWeight),
            &[key_value_width, hidden],
        )?,
        key_bias: bind_optional_weight(
            attention.bias().key(),
            weights,
            key_site,
            decoder(DecoderWeight::KeyBias),
            &[key_value_width],
        )?,
        value: bind_weight(
            weights,
            value_site,
            decoder(DecoderWeight::ValueWeight),
            &[key_value_width, hidden],
        )?,
        value_bias: bind_optional_weight(
            attention.bias().value(),
            weights,
            value_site,
            decoder(DecoderWeight::ValueBias),
            &[key_value_width],
        )?,
        output: bind_weight(
            weights,
            output_site,
            decoder(DecoderWeight::OutputWeight),
            &[hidden, hidden],
        )?,
        output_bias: bind_optional_weight(
            attention.bias().output(),
            weights,
            output_site,
            decoder(DecoderWeight::OutputBias),
            &[hidden],
        )?,
    })
}

#[cfg(any(feature = "cuda", test))]
fn validate_norm(
    norm: &rustinfer_model::NormSpec,
    hidden_size: usize,
    site: ExecutionSite,
) -> LlamaPlanResult<f32> {
    if norm.kind() != NormKind::RmsNorm {
        return Err(LlamaPlanError::UnsupportedOperationContract { site });
    }
    require_equal(
        site,
        LlamaDimension::HiddenSize,
        hidden_size,
        norm.hidden_size(),
    )?;
    finite_positive_f32(norm.epsilon(), site, LlamaScalar::NormEpsilon)
}

#[cfg(any(feature = "cuda", test))]
fn validate_attention(
    attention: &rustinfer_model::AttentionSpec,
    dimensions: LlamaDimensions,
    model_max_sequence_length: usize,
    expected_rope_theta: f64,
    layer: usize,
) -> LlamaPlanResult<()> {
    let site = ExecutionSite::layer(layer, LlamaOp::AttentionScores);
    for (dimension, expected, actual) in [
        (
            LlamaDimension::HiddenSize,
            dimensions.hidden_size,
            attention.hidden_size(),
        ),
        (
            LlamaDimension::QueryHeads,
            dimensions.query_heads,
            attention.query_heads(),
        ),
        (
            LlamaDimension::KeyValueHeads,
            dimensions.key_value_heads,
            attention.key_value_heads(),
        ),
        (
            LlamaDimension::HeadDimension,
            dimensions.head_dimension,
            attention.head_dimension(),
        ),
    ] {
        require_equal(site, dimension, expected, actual)?;
    }
    let rope = attention.rope();
    let rope_site = ExecutionSite::layer(layer, LlamaOp::QueryRope);
    if rope.layout() != RopeLayout::Standard {
        return Err(LlamaPlanError::UnsupportedOperationContract { site: rope_site });
    }
    require_equal(
        rope_site,
        LlamaDimension::RopeDimension,
        dimensions.head_dimension,
        rope.dimension(),
    )?;
    require_equal(
        rope_site,
        LlamaDimension::RopeMaxSequenceLength,
        model_max_sequence_length,
        rope.max_sequence_length(),
    )?;
    finite_positive_f32(rope.theta(), rope_site, LlamaScalar::RopeTheta)?;
    if rope.theta().to_bits() != expected_rope_theta.to_bits() {
        return Err(LlamaPlanError::ScalarMismatch {
            site: rope_site,
            scalar: LlamaScalar::RopeTheta,
            expected: expected_rope_theta,
            actual: rope.theta(),
        });
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn validate_mlp(
    mlp: &rustinfer_model::GatedMlpSpec,
    dimensions: LlamaDimensions,
    layer: usize,
) -> LlamaPlanResult<()> {
    let site = ExecutionSite::layer(layer, LlamaOp::GateProjection);
    require_equal(
        site,
        LlamaDimension::HiddenSize,
        dimensions.hidden_size,
        mlp.hidden_size(),
    )?;
    require_equal(
        site,
        LlamaDimension::IntermediateSize,
        dimensions.intermediate_size,
        mlp.intermediate_size(),
    )?;
    if mlp.activation() != Activation::Silu {
        return Err(LlamaPlanError::UnsupportedOperationContract { site });
    }
    if mlp.has_bias() {
        return Err(LlamaPlanError::UnsupportedBias { site });
    }
    Ok(())
}

#[cfg(any(feature = "cuda", test))]
fn bind_weight<C: PlanWeightCatalog>(
    weights: &C,
    site: ExecutionSite,
    slot: WeightSlot,
    expected_shape: &[usize],
) -> LlamaPlanResult<PhysicalWeightId> {
    let id = weights
        .resolve_slot(slot)
        .ok_or(LlamaPlanError::MissingWeight { site, slot })?;
    let metadata = weights
        .physical_metadata(id)
        .ok_or(LlamaPlanError::MissingWeight { site, slot })?;
    if metadata.dtype != DType::BF16 {
        return Err(LlamaPlanError::WeightDTypeMismatch {
            site,
            slot,
            expected: DType::BF16,
            actual: metadata.dtype,
        });
    }
    if metadata.shape != expected_shape {
        return Err(LlamaPlanError::WeightShapeMismatch {
            site,
            slot,
            expected: expected_shape.to_vec(),
            actual: metadata.shape.to_vec(),
        });
    }
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1_u64, |product, &dimension| {
            u64::try_from(dimension)
                .ok()
                .and_then(|dimension| product.checked_mul(dimension))
        })
        .and_then(|elements| elements.checked_mul(BF16_BYTES))
        .ok_or(LlamaPlanError::WeightSizeOverflow { site, slot })?;
    if metadata.byte_len != expected_bytes {
        return Err(LlamaPlanError::WeightByteLengthMismatch {
            site,
            slot,
            expected: expected_bytes,
            actual: metadata.byte_len,
        });
    }
    Ok(id)
}

#[cfg(any(feature = "cuda", test))]
fn bind_optional_weight<C: PlanWeightCatalog>(
    present: bool,
    weights: &C,
    site: ExecutionSite,
    slot: WeightSlot,
    expected_shape: &[usize],
) -> LlamaPlanResult<Option<PhysicalWeightId>> {
    present
        .then(|| bind_weight(weights, site, slot, expected_shape))
        .transpose()
}

#[cfg(any(feature = "cuda", test))]
fn build_workspace_spec(
    sequence_length: usize,
    dimensions: LlamaDimensions,
) -> LlamaPlanResult<LlamaWorkspaceSpec> {
    let sequence = to_u64(sequence_length, LlamaBufferRole::Total)?;
    let hidden = to_u64(dimensions.hidden_size, LlamaBufferRole::Hidden)?;
    let key_value = to_u64(dimensions.key_value_width, LlamaBufferRole::KeyValue)?;
    let intermediate = to_u64(dimensions.intermediate_size, LlamaBufferRole::Intermediate)?;
    let query_heads = to_u64(dimensions.query_heads, LlamaBufferRole::AttentionScores)?;
    let head_dimension = to_u64(dimensions.head_dimension, LlamaBufferRole::RopeCos)?;
    let vocabulary = to_u64(dimensions.vocabulary_size, LlamaBufferRole::Logits)?;

    let token_ids_bytes = checked_bytes(LlamaBufferRole::TokenIds, &[sequence], U32_BYTES)?;
    let hidden_buffer_bytes =
        checked_bytes(LlamaBufferRole::Hidden, &[sequence, hidden], BF16_BYTES)?;
    let key_value_buffer_bytes = checked_bytes(
        LlamaBufferRole::KeyValue,
        &[sequence, key_value],
        BF16_BYTES,
    )?;
    let intermediate_buffer_bytes = checked_bytes(
        LlamaBufferRole::Intermediate,
        &[sequence, intermediate],
        BF16_BYTES,
    )?;
    let attention_buffer_bytes = checked_bytes(
        LlamaBufferRole::AttentionScores,
        &[query_heads, sequence, sequence],
        BF16_BYTES,
    )?;
    let rope_columns = head_dimension / 2;
    let rope_cos_bytes = checked_bytes(
        LlamaBufferRole::RopeCos,
        &[sequence, rope_columns],
        F32_BYTES,
    )?;
    let rope_sin_bytes = checked_bytes(
        LlamaBufferRole::RopeSin,
        &[sequence, rope_columns],
        F32_BYTES,
    )?;
    let logits_bytes = checked_bytes(LlamaBufferRole::Logits, &[sequence, vocabulary], BF16_BYTES)?;

    let non_attention_counted = [
        (token_ids_bytes, 1),
        (hidden_buffer_bytes, HIDDEN_WORKSPACE_BUFFER_COUNT),
        (key_value_buffer_bytes, KEY_VALUE_WORKSPACE_BUFFER_COUNT),
        (
            intermediate_buffer_bytes,
            INTERMEDIATE_WORKSPACE_BUFFER_COUNT,
        ),
        (rope_cos_bytes, 1),
        (rope_sin_bytes, 1),
        (logits_bytes, 1),
        (EMBEDDING_ERROR_SCRATCH_BYTES, 1),
    ];
    let non_attention_planned_bytes =
        non_attention_counted
            .iter()
            .try_fold(0_u64, |total, &(bytes, count)| {
                bytes
                    .checked_mul(count)
                    .and_then(|counted_bytes| total.checked_add(counted_bytes))
                    .ok_or(LlamaPlanError::WorkspaceOverflow {
                        role: LlamaBufferRole::Total,
                    })
            })?;
    let total_planned_bytes = non_attention_planned_bytes
        .checked_add(attention_buffer_bytes)
        .ok_or(LlamaPlanError::WorkspaceOverflow {
            role: LlamaBufferRole::Total,
        })?;

    Ok(LlamaWorkspaceSpec {
        token_ids_bytes,
        hidden_buffer_bytes,
        key_value_buffer_bytes,
        intermediate_buffer_bytes,
        attention_buffer_bytes,
        rope_cos_bytes,
        rope_sin_bytes,
        logits_bytes,
        embedding_error_scratch_bytes: EMBEDDING_ERROR_SCRATCH_BYTES,
        non_attention_planned_bytes,
        total_planned_bytes,
    })
}

#[cfg(any(feature = "cuda", test))]
fn checked_bytes(role: LlamaBufferRole, dimensions: &[u64], scalar: u64) -> LlamaPlanResult<u64> {
    dimensions
        .iter()
        .try_fold(1_u64, |product, &dimension| {
            product
                .checked_mul(dimension)
                .ok_or(LlamaPlanError::WorkspaceOverflow { role })
        })?
        .checked_mul(scalar)
        .ok_or(LlamaPlanError::WorkspaceOverflow { role })
}

#[cfg(any(feature = "cuda", test))]
fn to_u64(value: usize, role: LlamaBufferRole) -> LlamaPlanResult<u64> {
    u64::try_from(value).map_err(|_| LlamaPlanError::WorkspaceOverflow { role })
}

#[cfg(any(feature = "cuda", test))]
fn require_positive(
    site: ExecutionSite,
    dimension: LlamaDimension,
    value: usize,
) -> LlamaPlanResult<()> {
    if value == 0 {
        Err(LlamaPlanError::InvalidDimension { site, dimension })
    } else {
        Ok(())
    }
}

#[cfg(any(feature = "cuda", test))]
fn require_equal(
    site: ExecutionSite,
    dimension: LlamaDimension,
    expected: usize,
    actual: usize,
) -> LlamaPlanResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(LlamaPlanError::DimensionMismatch {
            site,
            dimension,
            expected,
            actual,
        })
    }
}

#[cfg(any(feature = "cuda", test))]
fn finite_positive_f32(
    value: f64,
    site: ExecutionSite,
    scalar: LlamaScalar,
) -> LlamaPlanResult<f32> {
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = value as f32;
    if narrowed.is_finite() && narrowed > 0.0 {
        Ok(narrowed)
    } else {
        Err(LlamaPlanError::InvalidScalar { site, scalar })
    }
}

#[cfg(any(feature = "cuda", test))]
impl PlanWeightCatalog for crate::cuda_weights::CudaUploadedWeights {
    fn resolve_slot(&self, slot: WeightSlot) -> Option<PhysicalWeightId> {
        crate::cuda_weights::CudaUploadedWeights::resolve_slot(self, slot)
    }

    fn physical_metadata(&self, id: PhysicalWeightId) -> Option<PhysicalWeightMetadata<'_>> {
        crate::cuda_weights::CudaUploadedWeights::physical_metadata(self, id)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rustinfer_model::{DecoderWeight, LlamaConfig, ModelConfig, ModelSpec, WeightSlot};
    use rustinfer_tensor::DType;

    use super::{
        LlamaExecutionPlan, LlamaWorkspaceSpec, PhysicalWeightId, PhysicalWeightMetadata,
        PlanWeightCatalog, build_workspace_spec,
    };
    use crate::llama::{ExecutionSite, LlamaOp, LlamaPlanError};

    const OWNER: u64 = 7;
    const SMOL_CONFIG: &str = r#"{
      "architectures":["LlamaForCausalLM"],
      "attention_bias":false,
      "attention_dropout":0.0,
      "bos_token_id":1,
      "eos_token_id":0,
      "head_dim":64,
      "hidden_act":"silu",
      "hidden_size":576,
      "initializer_range":0.041666666666666664,
      "intermediate_size":1536,
      "is_llama_config":true,
      "max_position_embeddings":8192,
      "mlp_bias":false,
      "model_type":"llama",
      "num_attention_heads":9,
      "num_hidden_layers":30,
      "num_key_value_heads":3,
      "pretraining_tp":1,
      "rms_norm_eps":1e-5,
      "rope_interleaved":false,
      "rope_scaling":null,
      "rope_theta":100000,
      "tie_word_embeddings":true,
      "torch_dtype":"bfloat16",
      "transformers_version":"4.40.0",
      "use_cache":true,
      "vocab_size":49152
    }"#;

    const TINY_QWEN2_CONFIG: &str = r#"{
      "architectures":["Qwen2ForCausalLM"],
      "bos_token_id":6,
      "eos_token_id":7,
      "hidden_act":"silu",
      "hidden_size":4,
      "intermediate_size":8,
      "max_position_embeddings":16,
      "model_type":"qwen2",
      "num_attention_heads":2,
      "num_hidden_layers":1,
      "num_key_value_heads":1,
      "rms_norm_eps":0.000001,
      "rope_scaling":null,
      "rope_theta":1000000,
      "tie_word_embeddings":true,
      "torch_dtype":"bfloat16",
      "use_sliding_window":false,
      "vocab_size":8
    }"#;

    struct FakeTensor {
        dtype: DType,
        shape: Vec<usize>,
        byte_len: u64,
    }

    #[derive(Default)]
    struct FakeCatalog {
        slots: BTreeMap<WeightSlot, PhysicalWeightId>,
        physical: Vec<FakeTensor>,
    }

    impl FakeCatalog {
        fn insert(&mut self, slot: WeightSlot, shape: &[usize]) -> PhysicalWeightId {
            let elements = shape.iter().copied().product::<usize>();
            let byte_len = u64::try_from(elements).unwrap() * 2;
            let id = PhysicalWeightId::new(OWNER, self.physical.len());
            self.physical.push(FakeTensor {
                dtype: DType::BF16,
                shape: shape.to_vec(),
                byte_len,
            });
            self.slots.insert(slot, id);
            id
        }

        fn alias(&mut self, slot: WeightSlot, target: WeightSlot) {
            let id = self.slots[&target];
            self.slots.insert(slot, id);
        }
    }

    impl PlanWeightCatalog for FakeCatalog {
        fn resolve_slot(&self, slot: WeightSlot) -> Option<PhysicalWeightId> {
            self.slots.get(&slot).copied()
        }

        fn physical_metadata(&self, id: PhysicalWeightId) -> Option<PhysicalWeightMetadata<'_>> {
            if id.owner() != OWNER {
                return None;
            }
            self.physical
                .get(id.index())
                .map(|tensor| PhysicalWeightMetadata {
                    dtype: tensor.dtype,
                    shape: &tensor.shape,
                    byte_len: tensor.byte_len,
                })
        }
    }

    fn spec(config: &str) -> ModelSpec {
        LlamaConfig::from_json_slice(config.as_bytes())
            .expect("synthetic config must parse")
            .to_model_spec()
    }

    fn catalog_for(spec: &ModelSpec) -> FakeCatalog {
        let mut catalog = FakeCatalog::default();
        let hidden = spec.embedding().hidden_size();
        let vocabulary = spec.embedding().vocabulary_size();
        catalog.insert(WeightSlot::TokenEmbedding, &[vocabulary, hidden]);
        for block in spec.blocks() {
            let layer = block.index();
            let decoder = |parameter| WeightSlot::Decoder { layer, parameter };
            let attention = block.attention();
            let kv = attention.key_value_heads() * attention.head_dimension();
            let intermediate = block.mlp().intermediate_size();
            for (parameter, shape) in [
                (DecoderWeight::InputNormScale, vec![hidden]),
                (DecoderWeight::QueryWeight, vec![hidden, hidden]),
                (DecoderWeight::KeyWeight, vec![kv, hidden]),
                (DecoderWeight::ValueWeight, vec![kv, hidden]),
                (DecoderWeight::OutputWeight, vec![hidden, hidden]),
                (DecoderWeight::PostAttentionNormScale, vec![hidden]),
                (DecoderWeight::GateWeight, vec![intermediate, hidden]),
                (DecoderWeight::UpWeight, vec![intermediate, hidden]),
                (DecoderWeight::DownWeight, vec![hidden, intermediate]),
            ] {
                catalog.insert(decoder(parameter), &shape);
            }
            for (present, parameter, width) in [
                (attention.bias().query(), DecoderWeight::QueryBias, hidden),
                (attention.bias().key(), DecoderWeight::KeyBias, kv),
                (attention.bias().value(), DecoderWeight::ValueBias, kv),
                (attention.bias().output(), DecoderWeight::OutputBias, hidden),
            ] {
                if present {
                    catalog.insert(decoder(parameter), &[width]);
                }
            }
        }
        catalog.insert(WeightSlot::FinalNormScale, &[hidden]);
        if spec.lm_head().tied_to_embedding() {
            catalog.alias(WeightSlot::LmHead, WeightSlot::TokenEmbedding);
        } else {
            catalog.insert(WeightSlot::LmHead, &[vocabulary, hidden]);
        }
        catalog
    }

    #[test]
    fn smollm_plan_freezes_all_bindings_and_exact_workspace_bytes() {
        let spec = spec(SMOL_CONFIG);
        let catalog = catalog_for(&spec);
        let plan = LlamaExecutionPlan::prepare_with_catalog(&spec, &catalog, 128).unwrap();

        assert_eq!(plan.sequence_length(), 128);
        assert_eq!(plan.layers().len(), 30);
        assert_eq!(plan.logical_weight_count(), 273);
        assert_eq!(catalog.physical.len(), 272);
        assert_eq!(plan.embedding, plan.lm_head);
        assert_eq!(plan.layers()[14].index(), 14);
        assert_eq!(plan.dimensions().hidden_size(), 576);
        assert_eq!(plan.dimensions().intermediate_size(), 1536);
        assert_eq!(plan.dimensions().key_value_width(), 192);
        assert_eq!(plan.dimensions().group_size(), 3);

        let workspace = plan.workspace_spec();
        assert_eq!(workspace.token_ids_bytes(), 512);
        assert_eq!(workspace.hidden_buffer_bytes(), 147_456);
        assert_eq!(workspace.key_value_buffer_bytes(), 49_152);
        assert_eq!(workspace.intermediate_buffer_bytes(), 393_216);
        assert_eq!(workspace.attention_buffer_bytes(), 294_912);
        assert_eq!(workspace.rope_cos_bytes(), 16_384);
        assert_eq!(workspace.rope_sin_bytes(), 16_384);
        assert_eq!(workspace.logits_bytes(), 12_582_912);
        assert_eq!(workspace.embedding_error_scratch_bytes(), 32);
        assert_eq!(workspace.non_attention_planned_bytes(), 15_073_824);
        assert_eq!(workspace.total_planned_bytes(), 15_368_736);
    }

    #[test]
    fn tied_head_must_use_the_embedding_physical_identity() {
        let spec = spec(SMOL_CONFIG);
        let mut catalog = catalog_for(&spec);
        catalog.insert(WeightSlot::LmHead, &[49_152, 576]);
        assert_eq!(
            LlamaExecutionPlan::prepare_with_catalog(&spec, &catalog, 7).unwrap_err(),
            LlamaPlanError::TiedWeightIdentityMismatch
        );
    }

    #[test]
    fn fixed_sequence_bounds_fail_before_weight_lookup() {
        let spec = spec(SMOL_CONFIG);
        let empty = FakeCatalog::default();
        for (requested, expected) in [
            (
                0,
                LlamaPlanError::InvalidSequenceLength {
                    requested: 0,
                    maximum: 8192,
                },
            ),
            (
                8193,
                LlamaPlanError::InvalidSequenceLength {
                    requested: 8193,
                    maximum: 8192,
                },
            ),
        ] {
            assert_eq!(
                LlamaExecutionPlan::prepare_with_catalog(&spec, &empty, requested).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn projection_biases_are_bound_semantically_and_mlp_bias_fails_closed() {
        let empty = FakeCatalog::default();
        let fp16 = spec(&SMOL_CONFIG.replace("bfloat16", "float16"));
        assert_eq!(
            LlamaExecutionPlan::prepare_with_catalog(&fp16, &empty, 7).unwrap_err(),
            LlamaPlanError::UnsupportedDType { actual: DType::F16 }
        );

        let attention_bias =
            spec(&SMOL_CONFIG.replace("\"attention_bias\":false", "\"attention_bias\":true"));
        let attention_catalog = catalog_for(&attention_bias);
        let attention_plan =
            LlamaExecutionPlan::prepare_with_catalog(&attention_bias, &attention_catalog, 7)
                .unwrap();
        assert_eq!(attention_plan.logical_weight_count(), 393);
        let first = &attention_plan.layers()[0];
        assert!(first.query_bias().is_some());
        assert!(first.key_bias().is_some());
        assert!(first.value_bias().is_some());
        assert!(first.output_bias().is_some());

        let mlp_bias = spec(&SMOL_CONFIG.replace("\"mlp_bias\":false", "\"mlp_bias\":true"));
        let mlp_catalog = catalog_for(&mlp_bias);
        assert!(matches!(
            LlamaExecutionPlan::prepare_with_catalog(&mlp_bias, &mlp_catalog, 7),
            Err(LlamaPlanError::UnsupportedBias { site })
                if site == ExecutionSite::layer(0, LlamaOp::GateProjection)
        ));
    }

    #[test]
    fn qwen2_reuses_the_plan_with_qkv_bias_and_a_tied_head() {
        let spec = ModelConfig::from_json_slice(TINY_QWEN2_CONFIG.as_bytes())
            .unwrap()
            .to_model_spec();
        let catalog = catalog_for(&spec);
        let plan = LlamaExecutionPlan::prepare_with_catalog(&spec, &catalog, 4).unwrap();
        let layer = &plan.layers()[0];

        assert_eq!(plan.logical_weight_count(), 15);
        assert_eq!(catalog.physical.len(), 14);
        assert_eq!(plan.embedding, plan.lm_head);
        assert!(layer.query_bias().is_some());
        assert!(layer.key_bias().is_some());
        assert!(layer.value_bias().is_some());
        assert!(layer.output_bias().is_none());
    }

    #[test]
    fn workspace_arithmetic_is_checked() {
        let dimensions = super::LlamaDimensions {
            hidden_size: 1,
            intermediate_size: 1,
            vocabulary_size: 1,
            query_heads: usize::MAX,
            key_value_heads: 1,
            head_dimension: 2,
            key_value_width: 2,
            group_size: 1,
        };
        assert!(matches!(
            build_workspace_spec(usize::MAX, dimensions),
            Err(LlamaPlanError::WorkspaceOverflow { .. })
        ));
    }

    #[test]
    fn error_context_names_the_exact_layer_and_operation() {
        let error = LlamaPlanError::MissingWeight {
            site: ExecutionSite::layer(17, LlamaOp::QueryProjection),
            slot: WeightSlot::Decoder {
                layer: 17,
                parameter: DecoderWeight::QueryWeight,
            },
        };
        let rendered = error.to_string();
        assert!(rendered.contains("layer=17 op=query_projection"));
    }

    #[test]
    fn workspace_spec_remains_plain_copyable_metadata() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<LlamaWorkspaceSpec>();
    }
}
