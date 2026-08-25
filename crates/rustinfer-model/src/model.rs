use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::artifact::VerifiedArtifactSession;
use crate::{
    CheckpointProvenance, LlamaConfig, LoadLimits, LoadedWeights, ModelError, ModelResult,
    ModelSpec, SmolLm2Tokenizer,
};

/// Required Hugging Face configuration filename.
pub const CONFIG_FILENAME: &str = "config.json";

/// Required Hugging Face tokenizer filename.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// A checksum-verified, Python-free model artifact ready for runtime planning.
///
/// The checkpoint directory must remain trusted and immutable for the duration
/// of this call. The current portable standard-library reader rejects observed
/// symlinks and verifies every byte, but cannot make path resolution and open
/// atomic on every supported operating system.
pub struct LoadedModel {
    config: LlamaConfig,
    spec: ModelSpec,
    tokenizer: SmolLm2Tokenizer,
    weights: LoadedWeights,
    verified_files: BTreeSet<PathBuf>,
}

impl LoadedModel {
    /// Loads and cross-validates one complete local checkpoint directory.
    ///
    /// The mandatory provenance manifest must list exactly `config.json`,
    /// `tokenizer.json`, and either the single safetensors file or the complete
    /// shard-index layout. Each payload checksum is verified before that
    /// payload's parser runs; the manifest cannot checksum itself. No
    /// subprocess, Python interpreter, network request, or model-specific
    /// executable code is used.
    ///
    /// # Errors
    ///
    /// Returns a structured error for invalid provenance, config, tokenizer,
    /// weights, cross-artifact metadata, or the final consumed file set.
    pub fn load(root: &Path, limits: LoadLimits) -> ModelResult<Self> {
        let mut session = VerifiedArtifactSession::open(root, limits)?;

        let config_artifact = session.read_once(
            Path::new(CONFIG_FILENAME),
            limits.config_bytes(),
            "config.json",
        )?;
        let config = LlamaConfig::from_json_slice_with_limits(config_artifact.bytes(), limits)?;
        let spec = config.to_model_spec();
        validate_manifest_dtype(session.provenance(), &spec)?;

        let tokenizer_artifact = session.read_once(
            Path::new(TOKENIZER_FILENAME),
            limits.tokenizer_bytes(),
            "tokenizer.json",
        )?;
        let tokenizer =
            SmolLm2Tokenizer::from_json_slice_with_limits(tokenizer_artifact.bytes(), limits)?;
        validate_tokenizer_coherence(&spec, &tokenizer)?;

        let weights = LoadedWeights::load_verified_subset(&mut session, &spec, limits)?;
        let verified_files = session.finish_exact()?;

        Ok(Self {
            config,
            spec,
            tokenizer,
            weights,
            verified_files,
        })
    }

    /// Returns the validated source configuration and its inert-field warnings.
    #[must_use]
    pub const fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Returns the canonical execution description.
    #[must_use]
    pub const fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Returns the Rust-native tokenizer backend.
    #[must_use]
    pub const fn tokenizer(&self) -> &SmolLm2Tokenizer {
        &self.tokenizer
    }

    /// Returns validated canonical weight bindings and borrowed tensor views.
    #[must_use]
    pub const fn weights(&self) -> &LoadedWeights {
        &self.weights
    }

    /// Returns the source and checksum provenance shared by every component.
    #[must_use]
    pub const fn provenance(&self) -> &CheckpointProvenance {
        self.weights.provenance()
    }

    /// Returns the exact manifest-relative files consumed by this load.
    #[must_use]
    pub const fn verified_files(&self) -> &BTreeSet<PathBuf> {
        &self.verified_files
    }
}

fn validate_manifest_dtype(provenance: &CheckpointProvenance, spec: &ModelSpec) -> ModelResult<()> {
    if provenance.dtype() != spec.dtype() {
        return Err(ModelError::InvalidArtifact {
            artifact: "rustinfer-checkpoint.json".to_owned(),
            reason: format!(
                "manifest dtype {} differs from model dtype {}",
                provenance.dtype(),
                spec.dtype()
            ),
        });
    }
    Ok(())
}

fn validate_tokenizer_coherence(spec: &ModelSpec, tokenizer: &SmolLm2Tokenizer) -> ModelResult<()> {
    let expected_vocabulary = spec.embedding().vocabulary_size();
    let actual_vocabulary = tokenizer.vocabulary_size();
    if actual_vocabulary != expected_vocabulary {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_FILENAME.to_owned(),
            reason: format!(
                "vocabulary size mismatch: config requires {expected_vocabulary}, tokenizer has {actual_vocabulary}"
            ),
        });
    }

    if let Some(bos) = spec.special_tokens().bos() {
        validate_special_token(tokenizer, "bos_token_id", bos)?;
    }
    for &eos in spec.special_tokens().eos() {
        validate_special_token(tokenizer, "eos_token_id", eos)?;
    }
    Ok(())
}

fn validate_special_token(
    tokenizer: &SmolLm2Tokenizer,
    field: &'static str,
    id: u32,
) -> ModelResult<()> {
    if !tokenizer.contains_id(id) {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_FILENAME.to_owned(),
            reason: format!("config {field} {id} is absent from the tokenizer vocabulary"),
        });
    }
    if !tokenizer.is_special_id(id) {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_FILENAME.to_owned(),
            reason: format!("config {field} {id} is not declared as a special added token"),
        });
    }
    Ok(())
}
