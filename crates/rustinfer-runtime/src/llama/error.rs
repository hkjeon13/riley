use std::error;
use std::fmt;

use rustinfer_model::WeightSlot;
use rustinfer_tensor::DType;

/// One stable operation in the Llama reference-forward graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaOp {
    /// Token-embedding gather.
    Embedding,
    /// Pre-attention RMS normalization.
    InputNorm,
    /// Query projection.
    QueryProjection,
    /// Key projection.
    KeyProjection,
    /// Value projection.
    ValueProjection,
    /// Query rotary embedding.
    QueryRope,
    /// Key rotary embedding.
    KeyRope,
    /// Materialized attention scores.
    AttentionScores,
    /// Causal scale and mask.
    AttentionScaleMask,
    /// Stable row-wise attention softmax.
    AttentionSoftmax,
    /// Attention probability/value product.
    AttentionValue,
    /// Cold-selected full-sequence prefill attention backend.
    PrefillAttention,
    /// Copy rotated K and raw V into a request-local KV cache.
    KvCacheWrite,
    /// Single-query attention over a request-local KV cache.
    DecodeAttention,
    /// Attention output projection.
    OutputProjection,
    /// Post-attention residual addition.
    AttentionResidual,
    /// Pre-MLP RMS normalization.
    PostAttentionNorm,
    /// Gated-MLP gate projection.
    GateProjection,
    /// Gated-MLP up projection.
    UpProjection,
    /// `SiLU` gate activation.
    Silu,
    /// Activated-gate/up elementwise product.
    GatedMultiply,
    /// Gated-MLP down projection.
    DownProjection,
    /// Post-MLP residual addition.
    MlpResidual,
    /// Final RMS normalization.
    FinalNorm,
    /// Language-model output projection.
    LmHead,
}

impl LlamaOp {
    /// Stable allocation-free diagnostic name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::InputNorm => "input_norm",
            Self::QueryProjection => "query_projection",
            Self::KeyProjection => "key_projection",
            Self::ValueProjection => "value_projection",
            Self::QueryRope => "query_rope",
            Self::KeyRope => "key_rope",
            Self::AttentionScores => "attention_scores",
            Self::AttentionScaleMask => "attention_scale_mask",
            Self::AttentionSoftmax => "attention_softmax",
            Self::AttentionValue => "attention_value",
            Self::PrefillAttention => "prefill_attention",
            Self::KvCacheWrite => "kv_cache_write",
            Self::DecodeAttention => "decode_attention",
            Self::OutputProjection => "output_projection",
            Self::AttentionResidual => "attention_residual",
            Self::PostAttentionNorm => "post_attention_norm",
            Self::GateProjection => "gate_projection",
            Self::UpProjection => "up_projection",
            Self::Silu => "silu",
            Self::GatedMultiply => "gated_multiply",
            Self::DownProjection => "down_projection",
            Self::MlpResidual => "mlp_residual",
            Self::FinalNorm => "final_norm",
            Self::LmHead => "lm_head",
        }
    }
}

impl fmt::Display for LlamaOp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// Model-graph location attached to every plan or execution failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionSite {
    layer: Option<usize>,
    op: LlamaOp,
}

impl ExecutionSite {
    /// Constructs a model-global site.
    #[must_use]
    pub const fn global(op: LlamaOp) -> Self {
        Self { layer: None, op }
    }

    /// Constructs a decoder-layer site.
    #[must_use]
    pub const fn layer(layer: usize, op: LlamaOp) -> Self {
        Self {
            layer: Some(layer),
            op,
        }
    }

    /// Zero-based decoder layer, or none for a model-global operation.
    #[must_use]
    pub const fn layer_index(self) -> Option<usize> {
        self.layer
    }

    /// Operation at this site.
    #[must_use]
    pub const fn op(self) -> LlamaOp {
        self.op
    }
}

impl fmt::Display for ExecutionSite {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.layer {
            Some(layer) => write!(formatter, "layer={layer} op={}", self.op),
            None => write!(formatter, "global op={}", self.op),
        }
    }
}

/// Dimension whose execution contract was violated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaDimension {
    HiddenSize,
    IntermediateSize,
    VocabularySize,
    QueryHeads,
    KeyValueHeads,
    HeadDimension,
    QueryWidth,
    KeyValueWidth,
    RopeDimension,
    RopeMaxSequenceLength,
    LmHeadHiddenSize,
    LmHeadVocabularySize,
}

impl LlamaDimension {
    const fn name(self) -> &'static str {
        match self {
            Self::HiddenSize => "hidden_size",
            Self::IntermediateSize => "intermediate_size",
            Self::VocabularySize => "vocabulary_size",
            Self::QueryHeads => "query_heads",
            Self::KeyValueHeads => "key_value_heads",
            Self::HeadDimension => "head_dimension",
            Self::QueryWidth => "query_width",
            Self::KeyValueWidth => "key_value_width",
            Self::RopeDimension => "rope_dimension",
            Self::RopeMaxSequenceLength => "rope_max_sequence_length",
            Self::LmHeadHiddenSize => "lm_head_hidden_size",
            Self::LmHeadVocabularySize => "lm_head_vocabulary_size",
        }
    }
}

/// Scalar converted into the fixed CUDA execution representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaScalar {
    NormEpsilon,
    RopeTheta,
}

impl LlamaScalar {
    const fn name(self) -> &'static str {
        match self {
            Self::NormEpsilon => "norm_epsilon",
            Self::RopeTheta => "rope_theta",
        }
    }
}

/// Preallocated buffer whose exact byte-size calculation overflowed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum LlamaBufferRole {
    TokenIds,
    Hidden,
    KeyValue,
    Intermediate,
    AttentionScores,
    RopeCos,
    RopeSin,
    Logits,
    Total,
}

impl LlamaBufferRole {
    const fn name(self) -> &'static str {
        match self {
            Self::TokenIds => "token_ids",
            Self::Hidden => "hidden",
            Self::KeyValue => "key_value",
            Self::Intermediate => "intermediate",
            Self::AttentionScores => "attention_scores",
            Self::RopeCos => "rope_cos",
            Self::RopeSin => "rope_sin",
            Self::Logits => "logits",
            Self::Total => "total",
        }
    }
}

/// Stable cold-path failure while constructing an immutable Llama plan.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LlamaPlanError {
    /// Sequence length is zero or exceeds the canonical model limit.
    InvalidSequenceLength { requested: usize, maximum: usize },
    /// The current CUDA forward supports BF16 checkpoints only.
    UnsupportedDType { actual: DType },
    /// A serialized projection bias has no PR07 execution implementation.
    UnsupportedBias { site: ExecutionSite },
    /// Decoder blocks were not in contiguous zero-based order.
    LayerIndexMismatch { ordinal: usize, declared: usize },
    /// A canonical dimension was inconsistent with the model-wide contract.
    DimensionMismatch {
        site: ExecutionSite,
        dimension: LlamaDimension,
        expected: usize,
        actual: usize,
    },
    /// A required positive/divisibility dimension invariant failed.
    InvalidDimension {
        site: ExecutionSite,
        dimension: LlamaDimension,
    },
    /// A normalization, activation, or `RoPE` kind is not executable by PR07.
    UnsupportedOperationContract { site: ExecutionSite },
    /// A validated f64 scalar cannot be represented as finite positive f32.
    InvalidScalar {
        site: ExecutionSite,
        scalar: LlamaScalar,
    },
    /// Layer-local scalar differs from the immutable model-wide value.
    ScalarMismatch {
        site: ExecutionSite,
        scalar: LlamaScalar,
        expected: f64,
        actual: f64,
    },
    /// Checked byte or element arithmetic overflowed u64.
    WorkspaceOverflow { role: LlamaBufferRole },
    /// A required canonical slot was not present in the uploaded mapping.
    MissingWeight {
        site: ExecutionSite,
        slot: WeightSlot,
    },
    /// Uploaded physical storage dtype differs from the plan contract.
    WeightDTypeMismatch {
        site: ExecutionSite,
        slot: WeightSlot,
        expected: DType,
        actual: DType,
    },
    /// Uploaded physical shape differs from the plan contract.
    WeightShapeMismatch {
        site: ExecutionSite,
        slot: WeightSlot,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Uploaded physical bytes differ from the checked shape/dtype length.
    WeightByteLengthMismatch {
        site: ExecutionSite,
        slot: WeightSlot,
        expected: u64,
        actual: u64,
    },
    /// A required weight's checked BF16 byte length overflowed u64.
    WeightSizeOverflow {
        site: ExecutionSite,
        slot: WeightSlot,
    },
    /// A tied language-model head did not resolve to the embedding allocation.
    TiedWeightIdentityMismatch,
    /// The logical binding count overflowed usize.
    LogicalWeightCountOverflow,
}

impl fmt::Display for LlamaPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSequenceLength { requested, maximum } => write!(
                formatter,
                "invalid fixed Llama sequence length {requested}; expected 1..={maximum}"
            ),
            Self::UnsupportedDType { actual } => write!(
                formatter,
                "PR07 Llama forward requires a BF16 checkpoint, got {actual}"
            ),
            Self::UnsupportedBias { site } => {
                write!(
                    formatter,
                    "unsupported serialized projection bias at {site}"
                )
            }
            Self::LayerIndexMismatch { ordinal, declared } => write!(
                formatter,
                "decoder layer order mismatch: ordinal {ordinal} declares index {declared}"
            ),
            Self::DimensionMismatch {
                site,
                dimension,
                expected,
                actual,
            } => write!(
                formatter,
                "{site}: {} must be {expected}, got {actual}",
                dimension.name()
            ),
            Self::InvalidDimension { site, dimension } => {
                write!(formatter, "{site}: invalid {}", dimension.name())
            }
            Self::UnsupportedOperationContract { site } => {
                write!(formatter, "unsupported operation contract at {site}")
            }
            Self::InvalidScalar { site, scalar } => write!(
                formatter,
                "{site}: {} is not finite positive f32",
                scalar.name()
            ),
            Self::ScalarMismatch {
                site,
                scalar,
                expected,
                actual,
            } => write!(
                formatter,
                "{site}: {} must be {expected}, got {actual}",
                scalar.name()
            ),
            Self::WorkspaceOverflow { role } => write!(
                formatter,
                "Llama {} workspace byte calculation overflowed u64",
                role.name()
            ),
            Self::MissingWeight { site, slot } => {
                write!(formatter, "{site}: uploaded weights omit {slot:?}")
            }
            Self::WeightDTypeMismatch {
                site,
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "{site}: {slot:?} dtype must be {expected}, got {actual}"
            ),
            Self::WeightShapeMismatch {
                site,
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "{site}: {slot:?} shape must be {expected:?}, got {actual:?}"
            ),
            Self::WeightByteLengthMismatch {
                site,
                slot,
                expected,
                actual,
            } => write!(
                formatter,
                "{site}: {slot:?} must occupy {expected} bytes, got {actual}"
            ),
            Self::WeightSizeOverflow { site, slot } => {
                write!(
                    formatter,
                    "{site}: {slot:?} BF16 byte length overflowed u64"
                )
            }
            Self::TiedWeightIdentityMismatch => formatter.write_str(
                "tied language-model head does not share the token-embedding physical identity",
            ),
            Self::LogicalWeightCountOverflow => {
                formatter.write_str("logical Llama weight binding count overflowed usize")
            }
        }
    }
}

impl error::Error for LlamaPlanError {}

/// Result of immutable Llama plan construction.
pub type LlamaPlanResult<T> = Result<T, LlamaPlanError>;
