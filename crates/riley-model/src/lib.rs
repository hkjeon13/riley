//! Python-free Hugging Face artifact parsing and canonical model IR.
//!
//! This crate owns the boundary between serialized model artifacts and the
//! execution-facing description consumed by the runtime. Parsing is
//! fail-closed: unsupported model semantics are rejected rather than silently
//! approximated.

mod artifact;
mod checkpoint;
mod config;
mod error;
mod ir;
mod limits;
mod model;
mod pattern;
mod provenance;
mod qwen;
mod safetensors;
mod shard_index;
mod strict_json;
mod tokenizer;
mod weights;

pub use config::{ConfigWarning, LlamaConfig, ModelConfig, ModelFamily, Qwen2Config};
pub use error::{ArtifactKind, ModelError, ModelResult};
pub use ir::{
    Activation, AttentionBiasSpec, AttentionSpec, DecoderBlockSpec, EmbeddingSpec, GatedMlpSpec,
    LmHeadSpec, ModelArchitecture, ModelSpec, NormKind, NormSpec, RopeLayout, RopeSpec,
    SpecialTokenSpec,
};
pub use limits::LoadLimits;
pub use model::{CONFIG_FILENAME, LoadedModel, TOKENIZER_CONFIG_FILENAME, TOKENIZER_FILENAME};
pub use pattern::{PatternId, SEMANTIC_PATTERN_SCHEMA_VERSION, SemanticPattern};
pub use provenance::{CheckpointProvenance, PROVENANCE_FILENAME, ProvenanceFile};
pub use qwen::{ChatMessage, ChatRole, ChatTemplateOptions, Qwen2Tokenizer, Qwen2TokenizerConfig};
pub use tokenizer::{DecodeOptions, EncodeOptions, SmolLm2Tokenizer, Tokenizer};
pub use weights::{
    BoundWeight, DecoderWeight, LoadedWeights, TensorSource, WeightBinding, WeightSlot,
};

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "Python-free model artifact and canonical IR boundary";

/// Returns the lower-layer role this crate is allowed to depend on.
#[must_use]
pub const fn tensor_dependency_role() -> &'static str {
    riley_tensor::CRATE_ROLE
}
