use std::collections::{BTreeMap, BTreeSet};

use rustinfer_tensor::DType;
use serde::Deserialize;
use serde_json::Value;

use crate::ir::{
    Activation, AttentionBiasSpec, AttentionSpec, DecoderBlockSpec, EmbeddingSpec, GatedMlpSpec,
    LmHeadSpec, ModelSpec, NormSpec, RopeSpec, SpecialTokenSpec,
};
use crate::{ArtifactKind, LoadLimits, ModelError, ModelResult, strict_json};

/// Supported source-model families recognized at the cold configuration boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    /// Hugging Face Llama dense decoder configuration.
    Llama,
    /// Hugging Face Qwen2/Qwen2.5 dense decoder configuration.
    Qwen2,
}

/// Validated source configuration dispatched before execution planning.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelConfig {
    /// Llama-compatible source configuration.
    Llama(LlamaConfig),
    /// Dense Qwen2/Qwen2.5 source configuration.
    Qwen2(Qwen2Config),
}

impl ModelConfig {
    /// Parses, dispatches, and validates a bounded `config.json` using production limits.
    ///
    /// # Errors
    ///
    /// Returns a structured error for malformed JSON, an unsupported model
    /// family, inconsistent dimensions, or unsupported execution semantics.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::production())
    }

    /// Parses, dispatches, and validates a `config.json` with explicit limits.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LimitExceeded`] before parsing when the input is
    /// too large, or a structured format/configuration error afterward.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        validate_config_length(input, limits)?;
        let identity: RawModelIdentity = strict_json::from_slice(input, ArtifactKind::Config)?;
        match identity.model_type.as_str() {
            "llama" => LlamaConfig::from_json_slice_with_limits(input, limits).map(Self::Llama),
            "qwen2" => Qwen2Config::from_json_slice_with_limits(input, limits).map(Self::Qwen2),
            other => unsupported("model_type", other),
        }
    }

    /// Returns the source-model family selected at the cold parse boundary.
    #[must_use]
    pub const fn family(&self) -> ModelFamily {
        match self {
            Self::Llama(_) => ModelFamily::Llama,
            Self::Qwen2(_) => ModelFamily::Qwen2,
        }
    }

    /// Converts the validated source configuration into canonical execution IR.
    #[must_use]
    pub fn to_model_spec(&self) -> ModelSpec {
        match self {
            Self::Llama(config) => config.to_model_spec(),
            Self::Qwen2(config) => config.to_model_spec(),
        }
    }

    /// Returns warnings for explicitly recognized inference-inert fields.
    #[must_use]
    pub fn warnings(&self) -> &[ConfigWarning] {
        match self {
            Self::Llama(config) => config.warnings(),
            Self::Qwen2(config) => config.warnings(),
        }
    }

    /// Returns the checkpoint dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Llama(config) => config.dtype(),
            Self::Qwen2(config) => config.dtype(),
        }
    }
}

#[derive(Deserialize)]
struct RawModelIdentity {
    model_type: String,
    #[serde(flatten)]
    _ignored: BTreeMap<String, Value>,
}

/// A known training/export field that does not alter inference execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigWarning {
    field: &'static str,
    reason: &'static str,
}

impl ConfigWarning {
    const fn new(field: &'static str, reason: &'static str) -> Self {
        Self { field, reason }
    }

    /// Returns the ignored field name.
    #[must_use]
    pub const fn field(&self) -> &'static str {
        self.field
    }

    /// Returns why the field is inert for inference.
    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

/// Validated Llama-compatible Hugging Face configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LlamaConfig {
    dtype: DType,
    hidden_size: usize,
    intermediate_size: usize,
    layer_count: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    vocabulary_size: usize,
    max_sequence_length: usize,
    norm_epsilon: f64,
    rope_theta: f64,
    attention_bias: bool,
    mlp_bias: bool,
    tied_embeddings: bool,
    bos_token_id: Option<u32>,
    eos_token_ids: Vec<u32>,
    source_architecture: String,
    warnings: Vec<ConfigWarning>,
}

impl LlamaConfig {
    /// Parses and validates a bounded `config.json` using production limits.
    ///
    /// # Errors
    ///
    /// Returns a structured error for malformed JSON, duplicate/unknown fields,
    /// inconsistent dimensions, or unsupported execution semantics.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::production())
    }

    /// Parses and validates a `config.json` with explicit limits.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LimitExceeded`] before parsing when the input is
    /// too large, or a structured format/configuration error afterward.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        validate_config_length(input, limits)?;
        let raw: RawLlamaConfig = strict_json::from_slice(input, ArtifactKind::Config)?;
        Self::from_raw(raw, limits)
    }

    /// Converts the already validated configuration into canonical execution IR.
    #[must_use]
    pub fn to_model_spec(&self) -> ModelSpec {
        let norm = NormSpec::rms(self.hidden_size, self.norm_epsilon);
        let rope = RopeSpec::standard(
            self.head_dimension,
            self.rope_theta,
            self.max_sequence_length,
        );
        let attention = AttentionSpec::new(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
            AttentionBiasSpec::uniform(self.attention_bias),
            rope,
        );
        let mlp = GatedMlpSpec::new(
            self.hidden_size,
            self.intermediate_size,
            Activation::Silu,
            self.mlp_bias,
        );
        let blocks = (0..self.layer_count)
            .map(|index| {
                DecoderBlockSpec::new(
                    index,
                    norm.clone(),
                    attention.clone(),
                    norm.clone(),
                    mlp.clone(),
                )
            })
            .collect();

        ModelSpec::new(
            self.source_architecture.clone(),
            self.dtype,
            self.max_sequence_length,
            EmbeddingSpec::new(self.vocabulary_size, self.hidden_size),
            blocks,
            norm,
            LmHeadSpec::new(self.vocabulary_size, self.hidden_size, self.tied_embeddings),
            SpecialTokenSpec::new(self.bos_token_id, self.eos_token_ids.clone()),
        )
    }

    /// Returns warnings for explicitly recognized inference-inert fields.
    #[must_use]
    pub fn warnings(&self) -> &[ConfigWarning] {
        &self.warnings
    }

    /// Returns the checkpoint dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    fn from_raw(raw: RawLlamaConfig, limits: LoadLimits) -> ModelResult<Self> {
        if let Some((field, value)) = raw.unknown.iter().next() {
            return Err(ModelError::UnsupportedConfig {
                field: field.clone(),
                value: stable_value(value),
            });
        }
        let (source_architecture, dtype) = validate_identity(&raw)?;
        let dimensions = ValidatedDimensions::from_raw(&raw, limits)?;
        let rope_theta = validate_execution_values(&raw)?;
        let warnings = collect_warnings(&raw);
        let (bos_token_id, eos_token_ids) = validated_special_tokens(
            raw.bos_token_id,
            raw.eos_token_id,
            dimensions.vocabulary_size,
            limits.added_tokens(),
        )?;

        Ok(Self {
            dtype,
            hidden_size: dimensions.hidden_size,
            intermediate_size: dimensions.intermediate_size,
            layer_count: dimensions.layer_count,
            query_heads: dimensions.query_heads,
            key_value_heads: dimensions.key_value_heads,
            head_dimension: dimensions.head_dimension,
            vocabulary_size: dimensions.vocabulary_size,
            max_sequence_length: dimensions.max_sequence_length,
            norm_epsilon: raw.rms_norm_eps,
            rope_theta,
            attention_bias: raw.attention_bias.unwrap_or(false),
            mlp_bias: raw.mlp_bias.unwrap_or(false),
            tied_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            bos_token_id,
            eos_token_ids,
            source_architecture,
            warnings,
        })
    }
}

/// Validated dense Qwen2/Qwen2.5 Hugging Face configuration.
///
/// This adapter intentionally covers only the pinned dense text architecture.
/// Qwen-VL, `MoE`, Qwen3, sliding-window attention, and scaled or multi-axis
/// rotary embeddings are rejected at this cold boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen2Config {
    dtype: DType,
    hidden_size: usize,
    intermediate_size: usize,
    layer_count: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    vocabulary_size: usize,
    max_sequence_length: usize,
    norm_epsilon: f64,
    rope_theta: f64,
    tied_embeddings: bool,
    bos_token_id: Option<u32>,
    eos_token_ids: Vec<u32>,
    source_architecture: String,
    warnings: Vec<ConfigWarning>,
}

impl Qwen2Config {
    /// Parses and validates a bounded dense Qwen2/Qwen2.5 `config.json`.
    ///
    /// # Errors
    ///
    /// Returns a structured error for malformed JSON, duplicate/unknown fields,
    /// inconsistent dimensions, or unsupported Qwen execution semantics.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::production())
    }

    /// Parses and validates a dense Qwen2/Qwen2.5 config with explicit limits.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LimitExceeded`] before parsing when the input is
    /// too large, or a structured format/configuration error afterward.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        validate_config_length(input, limits)?;
        let raw: RawQwen2Config = strict_json::from_slice(input, ArtifactKind::Config)?;
        Self::from_raw(raw, limits)
    }

    /// Converts the validated Qwen source config into the shared canonical IR.
    #[must_use]
    pub fn to_model_spec(&self) -> ModelSpec {
        let norm = NormSpec::rms(self.hidden_size, self.norm_epsilon);
        let rope = RopeSpec::standard(
            self.head_dimension,
            self.rope_theta,
            self.max_sequence_length,
        );
        let attention = AttentionSpec::new(
            self.hidden_size,
            self.query_heads,
            self.key_value_heads,
            self.head_dimension,
            AttentionBiasSpec::qkv(),
            rope,
        );
        let mlp = GatedMlpSpec::new(
            self.hidden_size,
            self.intermediate_size,
            Activation::Silu,
            false,
        );
        let blocks = (0..self.layer_count)
            .map(|index| {
                DecoderBlockSpec::new(
                    index,
                    norm.clone(),
                    attention.clone(),
                    norm.clone(),
                    mlp.clone(),
                )
            })
            .collect();

        ModelSpec::new(
            self.source_architecture.clone(),
            self.dtype,
            self.max_sequence_length,
            EmbeddingSpec::new(self.vocabulary_size, self.hidden_size),
            blocks,
            norm,
            LmHeadSpec::new(self.vocabulary_size, self.hidden_size, self.tied_embeddings),
            SpecialTokenSpec::new(self.bos_token_id, self.eos_token_ids.clone()),
        )
    }

    /// Returns warnings for explicitly recognized inference-inert fields.
    #[must_use]
    pub fn warnings(&self) -> &[ConfigWarning] {
        &self.warnings
    }

    /// Returns the checkpoint dtype.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    fn from_raw(raw: RawQwen2Config, limits: LoadLimits) -> ModelResult<Self> {
        if let Some((field, value)) = raw.unknown.iter().next() {
            return Err(ModelError::UnsupportedConfig {
                field: field.clone(),
                value: stable_value(value),
            });
        }
        let (source_architecture, dtype) = validate_qwen2_identity(&raw)?;
        let dimensions = ValidatedDimensions::from_values(
            raw.hidden_size,
            raw.intermediate_size,
            raw.num_hidden_layers,
            raw.num_attention_heads,
            raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            raw.head_dim,
            raw.vocab_size,
            raw.max_position_embeddings,
            limits,
        )?;
        let rope_theta = validate_qwen2_execution_values(&raw)?;
        let warnings = collect_qwen2_warnings(&raw);
        let (bos_token_id, eos_token_ids) = validated_special_tokens(
            raw.bos_token_id,
            raw.eos_token_id,
            dimensions.vocabulary_size,
            limits.added_tokens(),
        )?;

        Ok(Self {
            dtype,
            hidden_size: dimensions.hidden_size,
            intermediate_size: dimensions.intermediate_size,
            layer_count: dimensions.layer_count,
            query_heads: dimensions.query_heads,
            key_value_heads: dimensions.key_value_heads,
            head_dimension: dimensions.head_dimension,
            vocabulary_size: dimensions.vocabulary_size,
            max_sequence_length: dimensions.max_sequence_length,
            norm_epsilon: raw.rms_norm_eps,
            rope_theta,
            tied_embeddings: raw.tie_word_embeddings.unwrap_or(false),
            bos_token_id,
            eos_token_ids,
            source_architecture,
            warnings,
        })
    }
}

#[derive(Deserialize)]
struct RawLlamaConfig {
    architectures: Option<Vec<String>>,
    attention_bias: Option<bool>,
    attention_dropout: Option<f64>,
    bos_token_id: Option<u64>,
    eos_token_id: Option<OneOrManyIds>,
    head_dim: Option<u64>,
    hidden_act: String,
    hidden_size: u64,
    initializer_range: Option<f64>,
    intermediate_size: u64,
    is_llama_config: Option<bool>,
    max_position_embeddings: u64,
    mlp_bias: Option<bool>,
    model_type: String,
    num_attention_heads: u64,
    num_hidden_layers: u64,
    num_key_value_heads: Option<u64>,
    partial_rotary_factor: Option<f64>,
    pretraining_tp: Option<u64>,
    rms_norm_eps: f64,
    rope_interleaved: Option<bool>,
    rope_scaling: Option<Value>,
    rope_theta: Option<f64>,
    sliding_window: Option<u64>,
    tie_word_embeddings: Option<bool>,
    torch_dtype: String,
    transformers_version: Option<String>,
    use_cache: Option<bool>,
    vocab_size: u64,
    #[serde(rename = "_name_or_path")]
    name_or_path: Option<String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct RawQwen2Config {
    architectures: Option<Vec<String>>,
    attention_dropout: Option<f64>,
    bos_token_id: Option<u64>,
    eos_token_id: Option<OneOrManyIds>,
    head_dim: Option<u64>,
    hidden_act: String,
    hidden_size: u64,
    initializer_range: Option<f64>,
    intermediate_size: u64,
    max_position_embeddings: u64,
    max_window_layers: Option<u64>,
    model_type: String,
    num_attention_heads: u64,
    num_hidden_layers: u64,
    num_key_value_heads: Option<u64>,
    partial_rotary_factor: Option<f64>,
    rms_norm_eps: f64,
    rope_interleaved: Option<bool>,
    rope_scaling: Option<Value>,
    rope_theta: Option<f64>,
    sliding_window: Option<u64>,
    tie_word_embeddings: Option<bool>,
    torch_dtype: String,
    transformers_version: Option<String>,
    use_cache: Option<bool>,
    use_sliding_window: Option<bool>,
    vocab_size: u64,
    #[serde(rename = "_name_or_path")]
    name_or_path: Option<String>,
    #[serde(flatten)]
    unknown: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrManyIds {
    One(u64),
    Many(Vec<u64>),
}

struct ValidatedDimensions {
    hidden_size: usize,
    intermediate_size: usize,
    layer_count: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dimension: usize,
    vocabulary_size: usize,
    max_sequence_length: usize,
}

impl ValidatedDimensions {
    fn from_raw(raw: &RawLlamaConfig, limits: LoadLimits) -> ModelResult<Self> {
        Self::from_values(
            raw.hidden_size,
            raw.intermediate_size,
            raw.num_hidden_layers,
            raw.num_attention_heads,
            raw.num_key_value_heads.unwrap_or(raw.num_attention_heads),
            raw.head_dim,
            raw.vocab_size,
            raw.max_position_embeddings,
            limits,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_values(
        raw_hidden_size: u64,
        raw_intermediate_size: u64,
        raw_layer_count: u64,
        raw_query_heads: u64,
        raw_key_value_heads: u64,
        raw_head_dimension: Option<u64>,
        raw_vocabulary_size: u64,
        raw_max_sequence_length: u64,
        limits: LoadLimits,
    ) -> ModelResult<Self> {
        let hidden_size = bounded_positive_usize(
            "hidden_size",
            "model dimension",
            raw_hidden_size,
            limits.model_dimension(),
        )?;
        let query_heads = bounded_positive_usize(
            "num_attention_heads",
            "attention heads",
            raw_query_heads,
            limits.attention_heads(),
        )?;
        let key_value_heads = bounded_positive_usize(
            "num_key_value_heads",
            "attention heads",
            raw_key_value_heads,
            limits.attention_heads(),
        )?;
        if query_heads % key_value_heads != 0 {
            return invalid(
                "num_key_value_heads",
                "must divide num_attention_heads for MHA/MQA/GQA",
            );
        }
        let head_dimension = if let Some(value) = raw_head_dimension {
            bounded_positive_usize(
                "head_dim",
                "attention head dimension",
                value,
                limits.head_dimension(),
            )?
        } else {
            if hidden_size % query_heads != 0 {
                return invalid(
                    "hidden_size",
                    "must be divisible by num_attention_heads when head_dim is absent",
                );
            }
            hidden_size / query_heads
        };
        if head_dimension > limits.head_dimension() {
            return Err(ModelError::LimitExceeded {
                resource: "attention head dimension",
                limit: u64::try_from(limits.head_dimension()).unwrap_or(u64::MAX),
                actual: u64::try_from(head_dimension).ok(),
            });
        }
        let projected =
            query_heads
                .checked_mul(head_dimension)
                .ok_or_else(|| ModelError::NumericOverflow {
                    field: "num_attention_heads * head_dim".to_owned(),
                })?;
        if projected != hidden_size {
            return invalid(
                "head_dim",
                "num_attention_heads * head_dim must equal hidden_size",
            );
        }
        if head_dimension % 2 != 0 {
            return invalid("head_dim", "standard RoPE requires an even head dimension");
        }

        Ok(Self {
            hidden_size,
            intermediate_size: bounded_positive_usize(
                "intermediate_size",
                "model dimension",
                raw_intermediate_size,
                limits.model_dimension(),
            )?,
            layer_count: bounded_positive_usize(
                "num_hidden_layers",
                "decoder layers",
                raw_layer_count,
                limits.model_layers(),
            )?,
            query_heads,
            key_value_heads,
            head_dimension,
            vocabulary_size: bounded_positive_usize(
                "vocab_size",
                "tokenizer vocabulary",
                raw_vocabulary_size,
                limits.vocabulary_entries(),
            )?,
            max_sequence_length: bounded_positive_usize(
                "max_position_embeddings",
                "model sequence length",
                raw_max_sequence_length,
                limits.sequence_length(),
            )?,
        })
    }
}

fn validate_identity(raw: &RawLlamaConfig) -> ModelResult<(String, DType)> {
    require_exact("model_type", &raw.model_type, "llama")?;
    if raw.is_llama_config == Some(false) {
        return unsupported("is_llama_config", "false");
    }
    require_exact("hidden_act", &raw.hidden_act, "silu")?;
    let source_architecture = match raw.architectures.as_deref() {
        None | Some([]) => "LlamaForCausalLM".to_owned(),
        Some([architecture]) if architecture == "LlamaForCausalLM" => architecture.clone(),
        Some(architectures) => {
            return unsupported("architectures", &format!("{architectures:?}"));
        }
    };
    let dtype = match raw.torch_dtype.as_str() {
        "float16" | "fp16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        other => return unsupported("torch_dtype", other),
    };
    Ok((source_architecture, dtype))
}

fn validate_qwen2_identity(raw: &RawQwen2Config) -> ModelResult<(String, DType)> {
    require_exact("model_type", &raw.model_type, "qwen2")?;
    require_exact("hidden_act", &raw.hidden_act, "silu")?;
    let source_architecture = match raw.architectures.as_deref() {
        None | Some([]) => "Qwen2ForCausalLM".to_owned(),
        Some([architecture]) if architecture == "Qwen2ForCausalLM" => architecture.clone(),
        Some(architectures) => {
            return unsupported("architectures", &format!("{architectures:?}"));
        }
    };
    let dtype = match raw.torch_dtype.as_str() {
        "float16" | "fp16" => DType::F16,
        "bfloat16" | "bf16" => DType::BF16,
        other => return unsupported("torch_dtype", other),
    };
    Ok((source_architecture, dtype))
}

fn validate_execution_values(raw: &RawLlamaConfig) -> ModelResult<f64> {
    if raw.rope_scaling.is_some() {
        return unsupported("rope_scaling", "non-null");
    }
    if raw.rope_interleaved.unwrap_or(false) {
        return unsupported("rope_interleaved", "true");
    }
    if raw.sliding_window.is_some() {
        return unsupported("sliding_window", "non-null");
    }
    if let Some(fraction) = raw.partial_rotary_factor {
        require_finite_positive("partial_rotary_factor", fraction)?;
        if fraction.to_bits() != 1.0_f64.to_bits() {
            return unsupported("partial_rotary_factor", &fraction.to_string());
        }
    }
    require_finite_positive("rms_norm_eps", raw.rms_norm_eps)?;
    let rope_theta = raw.rope_theta.unwrap_or(10_000.0);
    require_finite_positive("rope_theta", rope_theta)?;
    if let Some(dropout) = raw.attention_dropout {
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return invalid("attention_dropout", "must be finite and in [0, 1)");
        }
    }
    if raw.pretraining_tp == Some(0) {
        return invalid("pretraining_tp", "must be positive when present");
    }
    Ok(rope_theta)
}

fn validate_qwen2_execution_values(raw: &RawQwen2Config) -> ModelResult<f64> {
    if raw.rope_scaling.is_some() {
        return unsupported("rope_scaling", "non-null");
    }
    if raw.rope_interleaved.unwrap_or(false) {
        return unsupported("rope_interleaved", "true");
    }
    if raw.use_sliding_window.unwrap_or(false) {
        return unsupported("use_sliding_window", "true");
    }
    if raw.sliding_window == Some(0) {
        return invalid("sliding_window", "must be positive when present");
    }
    if raw.max_window_layers == Some(0) {
        return invalid("max_window_layers", "must be positive when present");
    }
    if let Some(fraction) = raw.partial_rotary_factor {
        require_finite_positive("partial_rotary_factor", fraction)?;
        if fraction.to_bits() != 1.0_f64.to_bits() {
            return unsupported("partial_rotary_factor", &fraction.to_string());
        }
    }
    require_finite_positive("rms_norm_eps", raw.rms_norm_eps)?;
    let rope_theta = raw.rope_theta.unwrap_or(1_000_000.0);
    require_finite_positive("rope_theta", rope_theta)?;
    if let Some(dropout) = raw.attention_dropout {
        if !dropout.is_finite() || !(0.0..1.0).contains(&dropout) {
            return invalid("attention_dropout", "must be finite and in [0, 1)");
        }
    }
    Ok(rope_theta)
}

fn validated_special_tokens(
    bos: Option<u64>,
    eos: Option<OneOrManyIds>,
    vocabulary_size: usize,
    maximum_eos_tokens: usize,
) -> ModelResult<(Option<u32>, Vec<u32>)> {
    let bos = bos
        .map(|id| checked_token_id("bos_token_id", id, vocabulary_size))
        .transpose()?;
    let eos = match eos {
        None => Vec::new(),
        Some(OneOrManyIds::One(id)) => {
            vec![checked_token_id("eos_token_id", id, vocabulary_size)?]
        }
        Some(OneOrManyIds::Many(ids)) => {
            if ids.is_empty() {
                return invalid("eos_token_id", "list must not be empty");
            }
            if ids.len() > maximum_eos_tokens {
                return Err(ModelError::LimitExceeded {
                    resource: "config EOS token IDs",
                    limit: u64::try_from(maximum_eos_tokens).unwrap_or(u64::MAX),
                    actual: u64::try_from(ids.len()).ok(),
                });
            }
            let mut checked = Vec::with_capacity(ids.len());
            let mut seen = BTreeSet::new();
            for id in ids {
                let id = checked_token_id("eos_token_id", id, vocabulary_size)?;
                if !seen.insert(id) {
                    return invalid("eos_token_id", "list contains duplicate IDs");
                }
                checked.push(id);
            }
            checked
        }
    };
    Ok((bos, eos))
}

fn collect_warnings(raw: &RawLlamaConfig) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();
    push_warning(
        &mut warnings,
        raw.attention_dropout.is_some(),
        "attention_dropout",
        "dropout is disabled during inference",
    );
    push_warning(
        &mut warnings,
        raw.initializer_range.is_some(),
        "initializer_range",
        "checkpoint loading does not initialize weights",
    );
    push_warning(
        &mut warnings,
        raw.pretraining_tp.is_some(),
        "pretraining_tp",
        "training tensor parallelism does not alter serialized weight semantics",
    );
    push_warning(
        &mut warnings,
        raw.transformers_version.is_some(),
        "transformers_version",
        "the runtime does not dispatch on a Python library version",
    );
    push_warning(
        &mut warnings,
        raw.use_cache.is_some(),
        "use_cache",
        "KV-cache policy is selected by the runtime",
    );
    push_warning(
        &mut warnings,
        raw.name_or_path.is_some(),
        "_name_or_path",
        "model name is diagnostic metadata only",
    );
    warnings
}

fn collect_qwen2_warnings(raw: &RawQwen2Config) -> Vec<ConfigWarning> {
    let mut warnings = Vec::new();
    push_warning(
        &mut warnings,
        raw.attention_dropout.is_some(),
        "attention_dropout",
        "dropout is disabled during inference",
    );
    push_warning(
        &mut warnings,
        raw.initializer_range.is_some(),
        "initializer_range",
        "checkpoint loading does not initialize weights",
    );
    push_warning(
        &mut warnings,
        raw.max_window_layers.is_some(),
        "max_window_layers",
        "sliding-window attention is disabled",
    );
    push_warning(
        &mut warnings,
        raw.sliding_window.is_some(),
        "sliding_window",
        "sliding-window attention is disabled",
    );
    push_warning(
        &mut warnings,
        raw.transformers_version.is_some(),
        "transformers_version",
        "the runtime does not dispatch on a Python library version",
    );
    push_warning(
        &mut warnings,
        raw.use_cache.is_some(),
        "use_cache",
        "KV-cache policy is selected by the runtime",
    );
    push_warning(
        &mut warnings,
        raw.use_sliding_window.is_some(),
        "use_sliding_window",
        "sliding-window attention is disabled",
    );
    push_warning(
        &mut warnings,
        raw.name_or_path.is_some(),
        "_name_or_path",
        "model name is diagnostic metadata only",
    );
    warnings
}

fn validate_config_length(input: &[u8], limits: LoadLimits) -> ModelResult<()> {
    let input_len = u64::try_from(input.len()).map_err(|_| ModelError::NumericOverflow {
        field: "config byte length".to_owned(),
    })?;
    if input_len > limits.config_bytes() {
        return Err(ModelError::LimitExceeded {
            resource: "config.json",
            limit: limits.config_bytes(),
            actual: Some(input_len),
        });
    }
    Ok(())
}

fn stable_value(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<unrenderable-json>".to_owned())
}

fn require_exact(field: &str, actual: &str, expected: &str) -> ModelResult<()> {
    if actual == expected {
        Ok(())
    } else {
        unsupported(field, actual)
    }
}

fn require_finite_positive(field: &str, value: f64) -> ModelResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        invalid(field, "must be finite and positive")
    }
}

fn bounded_positive_usize(
    field: &str,
    resource: &'static str,
    value: u64,
    limit: usize,
) -> ModelResult<usize> {
    if value == 0 {
        return invalid(field, "must be positive");
    }
    let limit = u64::try_from(limit).unwrap_or(u64::MAX);
    if value > limit {
        return Err(ModelError::LimitExceeded {
            resource,
            limit,
            actual: Some(value),
        });
    }
    usize::try_from(value).map_err(|_| ModelError::NumericOverflow {
        field: field.to_owned(),
    })
}

fn checked_token_id(field: &str, value: u64, vocabulary_size: usize) -> ModelResult<u32> {
    let id = u32::try_from(value).map_err(|_| ModelError::NumericOverflow {
        field: field.to_owned(),
    })?;
    let id_index = usize::try_from(id).map_err(|_| ModelError::NumericOverflow {
        field: field.to_owned(),
    })?;
    if id_index >= vocabulary_size {
        return invalid(field, "must be smaller than vocab_size");
    }
    Ok(id)
}

fn invalid<T>(field: &str, reason: &str) -> ModelResult<T> {
    Err(ModelError::InvalidConfig {
        field: field.to_owned(),
        reason: reason.to_owned(),
    })
}

fn unsupported<T>(field: &str, value: &str) -> ModelResult<T> {
    Err(ModelError::UnsupportedConfig {
        field: field.to_owned(),
        value: value.to_owned(),
    })
}

fn push_warning(
    warnings: &mut Vec<ConfigWarning>,
    present: bool,
    field: &'static str,
    reason: &'static str,
) {
    if present {
        warnings.push(ConfigWarning::new(field, reason));
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaConfig, ModelConfig, ModelFamily, Qwen2Config};
    use crate::{Activation, ModelArchitecture, ModelError, NormKind, RopeLayout};

    const SMOL_CONFIG: &str = r#"{
      "architectures": ["LlamaForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 0,
      "eos_token_id": 0,
      "hidden_act": "silu",
      "hidden_size": 576,
      "initializer_range": 0.041666666666666664,
      "intermediate_size": 1536,
      "is_llama_config": true,
      "max_position_embeddings": 8192,
      "model_type": "llama",
      "num_attention_heads": 9,
      "num_hidden_layers": 30,
      "num_key_value_heads": 3,
      "pretraining_tp": 1,
      "rms_norm_eps": 1e-5,
      "rope_interleaved": false,
      "rope_scaling": null,
      "rope_theta": 100000,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.40.1",
      "use_cache": true,
      "vocab_size": 49152
    }"#;

    const QWEN2_5_CONFIG: &str = r#"{
      "architectures": ["Qwen2ForCausalLM"],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 896,
      "initializer_range": 0.02,
      "intermediate_size": 4864,
      "max_position_embeddings": 32768,
      "max_window_layers": 24,
      "model_type": "qwen2",
      "num_attention_heads": 14,
      "num_hidden_layers": 24,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-6,
      "rope_scaling": null,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.1",
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 151936
    }"#;

    const SMOL_IR_SNAPSHOT: &str = r#"{
  "snapshot_version": "rustinfer-model-spec-v2",
  "architecture": "llama",
  "source_architecture": "LlamaForCausalLM",
  "dtype": "bf16",
  "max_sequence_length": 8192,
  "embedding": {
    "vocabulary_size": 49152,
    "hidden_size": 576
  },
  "decoder": {
    "layer_count": 30,
    "input_norm": {
      "kind": "rms_norm",
      "hidden_size": 576,
      "epsilon": 0.00001
    },
    "attention": {
      "hidden_size": 576,
      "query_heads": 9,
      "key_value_heads": 3,
      "head_dimension": 64,
      "bias": {
        "query": false,
        "key": false,
        "value": false,
        "output": false
      },
      "rope": {
        "dimension": 64,
        "theta": 100000.0,
        "max_sequence_length": 8192,
        "layout": "standard"
      }
    },
    "post_attention_norm": {
      "kind": "rms_norm",
      "hidden_size": 576,
      "epsilon": 0.00001
    },
    "mlp": {
      "hidden_size": 576,
      "intermediate_size": 1536,
      "activation": "silu",
      "bias": false
    }
  },
  "final_norm": {
    "kind": "rms_norm",
    "hidden_size": 576,
    "epsilon": 0.00001
  },
  "lm_head": {
    "vocabulary_size": 49152,
    "hidden_size": 576,
    "tied_to_embedding": true
  },
  "special_tokens": {
    "bos": 0,
    "eos": [
      0
    ]
  }
}"#;

    const QWEN2_5_IR_SNAPSHOT: &str = r#"{
  "snapshot_version": "rustinfer-model-spec-v2",
  "architecture": "llama",
  "source_architecture": "Qwen2ForCausalLM",
  "dtype": "bf16",
  "max_sequence_length": 32768,
  "embedding": {
    "vocabulary_size": 151936,
    "hidden_size": 896
  },
  "decoder": {
    "layer_count": 24,
    "input_norm": {
      "kind": "rms_norm",
      "hidden_size": 896,
      "epsilon": 1e-6
    },
    "attention": {
      "hidden_size": 896,
      "query_heads": 14,
      "key_value_heads": 2,
      "head_dimension": 64,
      "bias": {
        "query": true,
        "key": true,
        "value": true,
        "output": false
      },
      "rope": {
        "dimension": 64,
        "theta": 1000000.0,
        "max_sequence_length": 32768,
        "layout": "standard"
      }
    },
    "post_attention_norm": {
      "kind": "rms_norm",
      "hidden_size": 896,
      "epsilon": 1e-6
    },
    "mlp": {
      "hidden_size": 896,
      "intermediate_size": 4864,
      "activation": "silu",
      "bias": false
    }
  },
  "final_norm": {
    "kind": "rms_norm",
    "hidden_size": 896,
    "epsilon": 1e-6
  },
  "lm_head": {
    "vocabulary_size": 151936,
    "hidden_size": 896,
    "tied_to_embedding": true
  },
  "special_tokens": {
    "bos": 151643,
    "eos": [
      151645
    ]
  }
}"#;

    #[test]
    fn reference_config_becomes_canonical_ir() {
        let config = LlamaConfig::from_json_slice(SMOL_CONFIG.as_bytes()).unwrap();
        let spec = config.to_model_spec();
        assert_eq!(spec.blocks().len(), 30);
        assert_eq!(spec.embedding().vocabulary_size(), 49_152);
        assert_eq!(spec.blocks()[0].attention().query_heads(), 9);
        assert_eq!(spec.blocks()[0].attention().key_value_heads(), 3);
        assert_eq!(spec.blocks()[0].attention().head_dimension(), 64);
        assert_eq!(spec.special_tokens().bos(), Some(0));
        assert_eq!(spec.special_tokens().eos(), [0]);
        assert!(spec.lm_head().tied_to_embedding());
        assert_eq!(config.warnings().len(), 5);
        assert_eq!(spec.snapshot_json().unwrap(), SMOL_IR_SNAPSHOT);
    }

    #[test]
    fn pinned_qwen2_5_config_becomes_shared_canonical_ir() {
        let dispatched = ModelConfig::from_json_slice(QWEN2_5_CONFIG.as_bytes()).unwrap();
        assert_eq!(dispatched.family(), ModelFamily::Qwen2);
        let spec = dispatched.to_model_spec();
        let block = &spec.blocks()[0];
        let bias = block.attention().bias();

        assert_eq!(spec.architecture(), ModelArchitecture::Llama);
        assert_eq!(spec.source_architecture(), "Qwen2ForCausalLM");
        assert_eq!(spec.blocks().len(), 24);
        assert_eq!(block.input_norm().kind(), NormKind::RmsNorm);
        assert_eq!(block.attention().query_heads(), 14);
        assert_eq!(block.attention().key_value_heads(), 2);
        assert_eq!(block.attention().head_dimension(), 64);
        assert!(bias.query());
        assert!(bias.key());
        assert!(bias.value());
        assert!(!bias.output());
        assert_eq!(block.attention().rope().layout(), RopeLayout::Standard);
        assert_eq!(block.mlp().activation(), Activation::Silu);
        assert!(!block.mlp().has_bias());
        assert_eq!(spec.snapshot_json().unwrap(), QWEN2_5_IR_SNAPSHOT);
    }

    #[test]
    fn llama_dispatch_and_projection_bias_regression() {
        let dispatched = ModelConfig::from_json_slice(SMOL_CONFIG.as_bytes()).unwrap();
        assert_eq!(dispatched.family(), ModelFamily::Llama);
        let spec = dispatched.to_model_spec();
        let bias = spec.blocks()[0].attention().bias();
        assert!(!bias.query());
        assert!(!bias.key());
        assert!(!bias.value());
        assert!(!bias.output());

        let enabled = SMOL_CONFIG.replace("\"attention_bias\": false", "\"attention_bias\": true");
        let enabled = LlamaConfig::from_json_slice(enabled.as_bytes())
            .unwrap()
            .to_model_spec();
        let bias = enabled.blocks()[0].attention().bias();
        assert!(bias.query());
        assert!(bias.key());
        assert!(bias.value());
        assert!(bias.output());
    }

    #[test]
    fn qwen2_rejects_sliding_window_rope_extensions_and_other_families() {
        let sliding = QWEN2_5_CONFIG.replace(
            "\"use_sliding_window\": false",
            "\"use_sliding_window\": true",
        );
        let error = Qwen2Config::from_json_slice(sliding.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            ModelError::UnsupportedConfig { field, value }
                if field == "use_sliding_window" && value == "true"
        ));

        for rope_scaling in [
            r#"{"type":"linear","factor":2.0}"#,
            r#"{"type":"mrope","mrope_section":[16,24,24]}"#,
        ] {
            let changed = QWEN2_5_CONFIG.replace(
                "\"rope_scaling\": null",
                &format!("\"rope_scaling\": {rope_scaling}"),
            );
            let error = Qwen2Config::from_json_slice(changed.as_bytes()).unwrap_err();
            assert!(matches!(
                error,
                ModelError::UnsupportedConfig { field, .. } if field == "rope_scaling"
            ));
        }

        for unsupported_family in ["qwen2_moe", "qwen2_vl", "qwen3"] {
            let changed = QWEN2_5_CONFIG.replace(
                "\"model_type\": \"qwen2\"",
                &format!("\"model_type\": \"{unsupported_family}\""),
            );
            let error = ModelConfig::from_json_slice(changed.as_bytes()).unwrap_err();
            assert!(matches!(
                error,
                ModelError::UnsupportedConfig { field, value }
                    if field == "model_type" && value == unsupported_family
            ));
        }

        let architecture = QWEN2_5_CONFIG.replace("Qwen2ForCausalLM", "Qwen2MoeForCausalLM");
        let error = ModelConfig::from_json_slice(architecture.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            ModelError::UnsupportedConfig { field, .. } if field == "architectures"
        ));
    }

    #[test]
    fn rejects_unknown_semantic_field() {
        let changed = SMOL_CONFIG.replace(
            "\"use_cache\": true,",
            "\"use_cache\": true, \"mystery_mode\": true,",
        );
        let error = LlamaConfig::from_json_slice(changed.as_bytes()).unwrap_err();
        assert!(
            matches!(error, ModelError::UnsupportedConfig { field, .. } if field == "mystery_mode")
        );
    }

    #[test]
    fn rejects_unsupported_rope_scaling() {
        let changed = SMOL_CONFIG.replace(
            "\"rope_scaling\": null",
            "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0}",
        );
        let error = LlamaConfig::from_json_slice(changed.as_bytes()).unwrap_err();
        assert!(
            matches!(error, ModelError::UnsupportedConfig { field, .. } if field == "rope_scaling")
        );
    }

    #[test]
    fn rejects_inconsistent_gqa() {
        let changed =
            SMOL_CONFIG.replace("\"num_key_value_heads\": 3", "\"num_key_value_heads\": 4");
        let error = LlamaConfig::from_json_slice(changed.as_bytes()).unwrap_err();
        assert!(
            matches!(error, ModelError::InvalidConfig { field, .. } if field == "num_key_value_heads")
        );
    }

    #[test]
    fn rejects_duplicate_config_key() {
        let changed = SMOL_CONFIG.replace(
            "\"hidden_size\": 576,",
            "\"hidden_size\": 576, \"hidden_size\": 768,",
        );
        let error = LlamaConfig::from_json_slice(changed.as_bytes()).unwrap_err();
        assert!(matches!(error, ModelError::InvalidJson { .. }));
    }

    #[test]
    fn rejects_dimensions_before_canonical_ir_allocation() {
        let layers = SMOL_CONFIG.replace(
            "\"num_hidden_layers\": 30",
            "\"num_hidden_layers\": 18446744073709551615",
        );
        let error = LlamaConfig::from_json_slice(layers.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            ModelError::LimitExceeded {
                resource: "decoder layers",
                limit: 1024,
                actual: Some(u64::MAX),
            }
        ));

        let vocabulary = SMOL_CONFIG.replace("\"vocab_size\": 49152", "\"vocab_size\": 1000001");
        let error = LlamaConfig::from_json_slice(vocabulary.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            ModelError::LimitExceeded {
                resource: "tokenizer vocabulary",
                limit: 1_000_000,
                actual: Some(1_000_001),
            }
        ));
    }
}
