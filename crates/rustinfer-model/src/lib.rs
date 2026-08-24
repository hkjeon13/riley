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
mod provenance;
mod safetensors;
mod shard_index;
mod strict_json;
mod weights;

pub use config::{ConfigWarning, LlamaConfig};
pub use error::{ArtifactKind, ModelError, ModelResult};
pub use ir::{
    Activation, AttentionSpec, DecoderBlockSpec, EmbeddingSpec, GatedMlpSpec, LmHeadSpec,
    ModelArchitecture, ModelSpec, NormKind, NormSpec, RopeLayout, RopeSpec, SpecialTokenSpec,
};
pub use limits::LoadLimits;
pub use provenance::{CheckpointProvenance, PROVENANCE_FILENAME, ProvenanceFile};
pub use weights::{
    BoundWeight, DecoderWeight, LoadedWeights, TensorSource, WeightBinding, WeightSlot,
};

/// The crate's responsibility in the production dependency graph.
pub const CRATE_ROLE: &str = "Python-free model artifact and canonical IR boundary";

/// Returns the lower-layer role this crate is allowed to depend on.
#[must_use]
pub const fn tensor_dependency_role() -> &'static str {
    rustinfer_tensor::CRATE_ROLE
}
