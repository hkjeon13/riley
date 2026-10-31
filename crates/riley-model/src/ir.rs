use riley_tensor::DType;
use serde::Serialize;
use serde::Serializer;
use serde::ser::SerializeStruct;

use crate::{ModelError, ModelResult};

/// Supported canonical decoder architecture.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelArchitecture {
    /// Llama-layout dense decoder-only model.
    ///
    /// This canonical topology also covers dense Qwen2 models. Source-family
    /// differences are represented by semantic fields in the IR rather than
    /// by a second execution topology.
    Llama,
}

/// Token embedding metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EmbeddingSpec {
    vocabulary_size: usize,
    hidden_size: usize,
}

impl EmbeddingSpec {
    pub(crate) const fn new(vocabulary_size: usize, hidden_size: usize) -> Self {
        Self {
            vocabulary_size,
            hidden_size,
        }
    }

    /// Returns the number of vocabulary rows.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Returns the embedding width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

/// Canonical normalization operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    /// Root-mean-square normalization.
    RmsNorm,
}

/// Normalization metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormSpec {
    kind: NormKind,
    hidden_size: usize,
    epsilon: f64,
}

impl NormSpec {
    pub(crate) const fn rms(hidden_size: usize, epsilon: f64) -> Self {
        Self {
            kind: NormKind::RmsNorm,
            hidden_size,
            epsilon,
        }
    }

    /// Returns the normalization kind.
    #[must_use]
    pub const fn kind(&self) -> NormKind {
        self.kind
    }

    /// Returns the normalized width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the epsilon applied before reciprocal square root.
    #[must_use]
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }
}

/// Supported rotary coordinate layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeLayout {
    /// Standard non-interleaved Llama rotary layout.
    Standard,
}

/// Rotary position embedding metadata.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RopeSpec {
    dimension: usize,
    theta: f64,
    max_sequence_length: usize,
    layout: RopeLayout,
    /// Frequency rescaling for contexts longer than the pretraining length
    /// (Llama 3.1/3.2); absent for the other models.
    #[serde(skip_serializing_if = "Option::is_none")]
    scaling: Option<RopeScaling>,
}

/// The `llama3` rotary frequency rescaling of Hugging Face Transformers:
/// frequencies whose wavelength exceeds
/// `original_max_sequence_length / low_frequency_factor` are divided by
/// `factor`, those with a wavelength below
/// `original_max_sequence_length / high_frequency_factor` are kept, and the
/// ones in between are interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct RopeScaling {
    factor: f64,
    low_frequency_factor: f64,
    high_frequency_factor: f64,
    original_max_sequence_length: usize,
}

impl RopeScaling {
    pub(crate) const fn llama3(
        factor: f64,
        low_frequency_factor: f64,
        high_frequency_factor: f64,
        original_max_sequence_length: usize,
    ) -> Self {
        Self {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_max_sequence_length,
        }
    }

    /// Divisor applied to the low frequencies.
    #[must_use]
    pub const fn factor(&self) -> f64 {
        self.factor
    }

    /// Wavelength bound divisor above which a frequency counts as low.
    #[must_use]
    pub const fn low_frequency_factor(&self) -> f64 {
        self.low_frequency_factor
    }

    /// Wavelength bound divisor below which a frequency counts as high.
    #[must_use]
    pub const fn high_frequency_factor(&self) -> f64 {
        self.high_frequency_factor
    }

    /// Context length the frequencies were trained for.
    #[must_use]
    pub const fn original_max_sequence_length(&self) -> usize {
        self.original_max_sequence_length
    }
}

impl RopeSpec {
    pub(crate) const fn standard(dimension: usize, theta: f64, max_sequence_length: usize) -> Self {
        Self {
            dimension,
            theta,
            max_sequence_length,
            layout: RopeLayout::Standard,
            scaling: None,
        }
    }

    pub(crate) const fn with_scaling(mut self, scaling: Option<RopeScaling>) -> Self {
        self.scaling = scaling;
        self
    }

    /// Returns the frequency rescaling, when the model has one.
    #[must_use]
    pub const fn scaling(&self) -> Option<&RopeScaling> {
        self.scaling.as_ref()
    }

    /// Returns the rotated head dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the `RoPE` base frequency.
    #[must_use]
    pub const fn theta(&self) -> f64 {
        self.theta
    }

    /// Returns the configured context length.
    #[must_use]
    pub const fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// Returns the coordinate layout.
    #[must_use]
    pub const fn layout(&self) -> RopeLayout {
        self.layout
    }
}

/// Serialized bias presence for each attention projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionBiasSpec {
    mask: u8,
}

impl AttentionBiasSpec {
    const QUERY: u8 = 1 << 0;
    const KEY: u8 = 1 << 1;
    const VALUE: u8 = 1 << 2;
    const OUTPUT: u8 = 1 << 3;
    const ALL: u8 = Self::QUERY | Self::KEY | Self::VALUE | Self::OUTPUT;

    pub(crate) const fn uniform(enabled: bool) -> Self {
        Self {
            mask: if enabled { Self::ALL } else { 0 },
        }
    }

    pub(crate) const fn qkv() -> Self {
        Self {
            mask: Self::QUERY | Self::KEY | Self::VALUE,
        }
    }

    /// Returns whether the query projection has a serialized bias.
    #[must_use]
    pub const fn query(&self) -> bool {
        self.mask & Self::QUERY != 0
    }

    /// Returns whether the key projection has a serialized bias.
    #[must_use]
    pub const fn key(&self) -> bool {
        self.mask & Self::KEY != 0
    }

    /// Returns whether the value projection has a serialized bias.
    #[must_use]
    pub const fn value(&self) -> bool {
        self.mask & Self::VALUE != 0
    }

    /// Returns whether the output projection has a serialized bias.
    #[must_use]
    pub const fn output(&self) -> bool {
        self.mask & Self::OUTPUT != 0
    }

    /// Returns whether any attention projection has a serialized bias.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.mask != 0
    }
}

impl Serialize for AttentionBiasSpec {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("AttentionBiasSpec", 4)?;
        state.serialize_field("query", &self.query())?;
        state.serialize_field("key", &self.key())?;
        state.serialize_field("value", &self.value())?;
        state.serialize_field("output", &self.output())?;
        state.end()
    }
}

/// Self-attention dimensions and projection-specific bias contract.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AttentionSpec {
    hidden_size: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    bias: AttentionBiasSpec,
    rope: RopeSpec,
    /// RMS norm over each query head and each key head before the rotary
    /// embedding (Qwen3); absent for the other families.
    #[serde(skip_serializing_if = "Option::is_none")]
    query_key_norm: Option<NormSpec>,
}

impl AttentionSpec {
    pub(crate) const fn new(
        hidden_size: usize,
        query_heads: usize,
        key_value_heads: usize,
        head_dimension: usize,
        bias: AttentionBiasSpec,
        rope: RopeSpec,
    ) -> Self {
        Self {
            hidden_size,
            query_heads,
            key_value_heads,
            head_dimension,
            bias,
            rope,
            query_key_norm: None,
        }
    }

    pub(crate) fn with_query_key_norm(mut self, norm: Option<NormSpec>) -> Self {
        self.query_key_norm = norm;
        self
    }

    /// Returns the per-head query/key normalization, when the model has one.
    #[must_use]
    pub const fn query_key_norm(&self) -> Option<&NormSpec> {
        self.query_key_norm.as_ref()
    }

    /// Returns the residual width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of query heads.
    #[must_use]
    pub const fn query_heads(&self) -> usize {
        self.query_heads
    }

    /// Returns the number of key/value heads.
    #[must_use]
    pub const fn key_value_heads(&self) -> usize {
        self.key_value_heads
    }

    /// Returns each head width.
    #[must_use]
    pub const fn head_dimension(&self) -> usize {
        self.head_dimension
    }

    /// Returns the projection-specific serialized bias contract.
    #[must_use]
    pub const fn bias(&self) -> &AttentionBiasSpec {
        &self.bias
    }

    /// Returns whether any attention projection has a serialized bias.
    #[must_use]
    pub const fn has_bias(&self) -> bool {
        self.bias.any()
    }

    /// Returns the rotary embedding contract.
    #[must_use]
    pub const fn rope(&self) -> &RopeSpec {
        &self.rope
    }
}

/// Supported gated MLP activation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Activation {
    /// SiLU/Swish activation.
    Silu,
}

/// Canonical gated MLP dimensions.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GatedMlpSpec {
    hidden_size: usize,
    intermediate_size: usize,
    activation: Activation,
    bias: bool,
}

impl GatedMlpSpec {
    pub(crate) const fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: Activation,
        bias: bool,
    ) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            activation,
            bias,
        }
    }

    /// Returns the residual width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the expanded width.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Returns the gate activation.
    #[must_use]
    pub const fn activation(&self) -> Activation {
        self.activation
    }

    /// Returns whether gate/up/down projections have serialized biases.
    #[must_use]
    pub const fn has_bias(&self) -> bool {
        self.bias
    }
}

/// One canonical decoder block.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecoderBlockSpec {
    index: usize,
    input_norm: NormSpec,
    attention: AttentionSpec,
    post_attention_norm: NormSpec,
    mlp: GatedMlpSpec,
}

impl DecoderBlockSpec {
    pub(crate) const fn new(
        index: usize,
        input_norm: NormSpec,
        attention: AttentionSpec,
        post_attention_norm: NormSpec,
        mlp: GatedMlpSpec,
    ) -> Self {
        Self {
            index,
            input_norm,
            attention,
            post_attention_norm,
            mlp,
        }
    }

    /// Returns the zero-based layer index.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Returns the pre-attention RMS normalization.
    #[must_use]
    pub const fn input_norm(&self) -> &NormSpec {
        &self.input_norm
    }

    /// Returns self-attention metadata.
    #[must_use]
    pub const fn attention(&self) -> &AttentionSpec {
        &self.attention
    }

    /// Returns the post-attention RMS normalization.
    #[must_use]
    pub const fn post_attention_norm(&self) -> &NormSpec {
        &self.post_attention_norm
    }

    /// Returns gated MLP metadata.
    #[must_use]
    pub const fn mlp(&self) -> &GatedMlpSpec {
        &self.mlp
    }
}

/// Language-model output projection metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LmHeadSpec {
    vocabulary_size: usize,
    hidden_size: usize,
    tied_to_embedding: bool,
}

impl LmHeadSpec {
    pub(crate) const fn new(
        vocabulary_size: usize,
        hidden_size: usize,
        tied_to_embedding: bool,
    ) -> Self {
        Self {
            vocabulary_size,
            hidden_size,
            tied_to_embedding,
        }
    }

    /// Returns the number of output rows.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Returns the input width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns whether this projection aliases token embeddings.
    #[must_use]
    pub const fn tied_to_embedding(&self) -> bool {
        self.tied_to_embedding
    }
}

/// Model-level special token IDs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct SpecialTokenSpec {
    bos: Option<u32>,
    eos: Box<[u32]>,
}

impl SpecialTokenSpec {
    pub(crate) fn new(bos: Option<u32>, eos: Vec<u32>) -> Self {
        Self {
            bos,
            eos: eos.into_boxed_slice(),
        }
    }

    /// Returns the beginning-of-sequence ID, if configured.
    #[must_use]
    pub const fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// Returns all end-of-sequence IDs in declared order.
    #[must_use]
    pub const fn eos(&self) -> &[u32] {
        &self.eos
    }
}

/// Validated execution-facing model description.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelSpec {
    snapshot_version: &'static str,
    architecture: ModelArchitecture,
    source_architecture: String,
    dtype: DTypeSnapshot,
    max_sequence_length: usize,
    embedding: EmbeddingSpec,
    blocks: Box<[DecoderBlockSpec]>,
    final_norm: NormSpec,
    lm_head: LmHeadSpec,
    special_tokens: SpecialTokenSpec,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum DTypeSnapshot {
    F16,
    Bf16,
}

impl ModelSpec {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        source_architecture: String,
        dtype: DType,
        max_sequence_length: usize,
        embedding: EmbeddingSpec,
        blocks: Vec<DecoderBlockSpec>,
        final_norm: NormSpec,
        lm_head: LmHeadSpec,
        special_tokens: SpecialTokenSpec,
    ) -> Self {
        let dtype = match dtype {
            DType::F16 => DTypeSnapshot::F16,
            DType::BF16 => DTypeSnapshot::Bf16,
            _ => unreachable!("validated model dtype must be f16 or bf16"),
        };
        Self {
            snapshot_version: "riley-model-spec-v2",
            architecture: ModelArchitecture::Llama,
            source_architecture,
            dtype,
            max_sequence_length,
            embedding,
            blocks: blocks.into_boxed_slice(),
            final_norm,
            lm_head,
            special_tokens,
        }
    }

    /// Returns the spec one rank of a `size`-way tensor-parallel group
    /// executes: every block keeps `query_heads / size` query heads,
    /// `key_value_heads / size` KV heads and `intermediate_size / size` MLP
    /// columns; the residual width, embedding, norms and LM head are whole.
    /// `size` 1 returns the spec unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::UnsupportedConfig`] when `size` is zero or does
    /// not divide a block's query heads, KV heads or intermediate size.
    pub fn tensor_parallel_rank_spec(&self, size: usize) -> ModelResult<Self> {
        let unsupported = |field: &str, value: usize| ModelError::UnsupportedConfig {
            field: format!("tensor_parallel_size for {field}"),
            value: format!("{size} does not divide {value}"),
        };
        if size == 0 {
            return Err(ModelError::UnsupportedConfig {
                field: "tensor_parallel_size".to_owned(),
                value: "0".to_owned(),
            });
        }
        let mut spec = self.clone();
        for block in spec.blocks.iter_mut() {
            let attention = &mut block.attention;
            if attention.query_heads % size != 0 {
                return Err(unsupported("num_attention_heads", attention.query_heads));
            }
            if attention.key_value_heads % size != 0 {
                return Err(unsupported("num_key_value_heads", attention.key_value_heads));
            }
            if block.mlp.intermediate_size % size != 0 {
                return Err(unsupported("intermediate_size", block.mlp.intermediate_size));
            }
            attention.query_heads /= size;
            attention.key_value_heads /= size;
            block.mlp.intermediate_size /= size;
        }
        Ok(spec)
    }

    /// Returns the canonical architecture.
    #[must_use]
    pub const fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Returns source metadata used only for diagnostics.
    #[must_use]
    pub fn source_architecture(&self) -> &str {
        &self.source_architecture
    }

    /// Returns the checkpoint scalar dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self.dtype {
            DTypeSnapshot::F16 => DType::F16,
            DTypeSnapshot::Bf16 => DType::BF16,
        }
    }

    /// Returns the maximum configured sequence length.
    #[must_use]
    pub const fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// Returns token embedding metadata.
    #[must_use]
    pub const fn embedding(&self) -> &EmbeddingSpec {
        &self.embedding
    }

    /// Returns all decoder blocks in layer order.
    #[must_use]
    pub const fn blocks(&self) -> &[DecoderBlockSpec] {
        &self.blocks
    }

    /// Returns the final normalization metadata.
    #[must_use]
    pub const fn final_norm(&self) -> &NormSpec {
        &self.final_norm
    }

    /// Returns the language-model head metadata.
    #[must_use]
    pub const fn lm_head(&self) -> &LmHeadSpec {
        &self.lm_head
    }

    /// Returns model special token IDs.
    #[must_use]
    pub const fn special_tokens(&self) -> &SpecialTokenSpec {
        &self.special_tokens
    }

    /// Serializes a deterministic, versioned snapshot without debug formatting.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization unexpectedly fails.
    pub fn snapshot_json(&self) -> ModelResult<String> {
        let block = self
            .blocks
            .first()
            .ok_or_else(|| ModelError::InvalidArtifact {
                artifact: "canonical-model-spec".to_owned(),
                reason: "validated model has no decoder block".to_owned(),
            })?;
        let snapshot = ModelSpecSnapshot {
            snapshot_version: self.snapshot_version,
            architecture: self.architecture,
            source_architecture: &self.source_architecture,
            dtype: self.dtype.name(),
            max_sequence_length: self.max_sequence_length,
            embedding: &self.embedding,
            decoder: DecoderStackSnapshot {
                layer_count: self.blocks.len(),
                input_norm: &block.input_norm,
                attention: &block.attention,
                post_attention_norm: &block.post_attention_norm,
                mlp: &block.mlp,
            },
            final_norm: &self.final_norm,
            lm_head: &self.lm_head,
            special_tokens: &self.special_tokens,
        };
        serde_json::to_string_pretty(&snapshot).map_err(|error| ModelError::InvalidArtifact {
            artifact: "canonical-model-spec".to_owned(),
            reason: error.to_string(),
        })
    }
}

impl DTypeSnapshot {
    const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

#[derive(Serialize)]
struct ModelSpecSnapshot<'a> {
    snapshot_version: &'static str,
    architecture: ModelArchitecture,
    source_architecture: &'a str,
    dtype: &'static str,
    max_sequence_length: usize,
    embedding: &'a EmbeddingSpec,
    decoder: DecoderStackSnapshot<'a>,
    final_norm: &'a NormSpec,
    lm_head: &'a LmHeadSpec,
    special_tokens: &'a SpecialTokenSpec,
}

#[derive(Serialize)]
struct DecoderStackSnapshot<'a> {
    layer_count: usize,
    input_norm: &'a NormSpec,
    attention: &'a AttentionSpec,
    post_attention_norm: &'a NormSpec,
    mlp: &'a GatedMlpSpec,
}
