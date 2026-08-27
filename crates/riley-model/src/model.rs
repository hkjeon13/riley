use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::artifact::VerifiedArtifactSession;
use crate::{
    CheckpointProvenance, LoadLimits, LoadedWeights, ModelConfig, ModelError, ModelFamily,
    ModelResult, ModelSpec, Qwen2Tokenizer, Qwen2TokenizerConfig, SmolLm2Tokenizer, Tokenizer,
};

/// Required Hugging Face configuration filename.
pub const CONFIG_FILENAME: &str = "config.json";

/// Required Hugging Face tokenizer filename.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Required Qwen chat/tokenizer metadata filename.
pub const TOKENIZER_CONFIG_FILENAME: &str = "tokenizer_config.json";

/// A checksum-verified, Python-free model artifact ready for runtime planning.
///
/// The checkpoint directory must remain trusted and immutable for the duration
/// of this call. The current portable standard-library reader rejects observed
/// symlinks and verifies every byte, but cannot make path resolution and open
/// atomic on every supported operating system.
pub struct LoadedModel {
    config: ModelConfig,
    spec: ModelSpec,
    tokenizer: Box<dyn Tokenizer>,
    qwen2_tokenizer_config: Option<Qwen2TokenizerConfig>,
    weights: LoadedWeights,
    verified_files: BTreeSet<PathBuf>,
}

impl LoadedModel {
    /// Loads and cross-validates one complete local checkpoint directory.
    ///
    /// The mandatory provenance manifest must list `config.json`,
    /// `tokenizer.json`, Qwen's `tokenizer_config.json` when applicable, and
    /// either the single safetensors file or the complete shard-index layout.
    /// Each payload checksum is verified before that payload's parser runs; the
    /// manifest cannot checksum itself. No subprocess, Python interpreter,
    /// network request, or model-specific executable code is used.
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
        let config = ModelConfig::from_json_slice_with_limits(config_artifact.bytes(), limits)?;
        let spec = config.to_model_spec();
        validate_manifest_dtype(session.provenance(), &spec)?;

        let tokenizer_artifact = session.read_once(
            Path::new(TOKENIZER_FILENAME),
            limits.tokenizer_bytes(),
            "tokenizer.json",
        )?;
        let (tokenizer, qwen2_tokenizer_config): (
            Box<dyn Tokenizer>,
            Option<Qwen2TokenizerConfig>,
        ) = match config.family() {
            ModelFamily::Llama => {
                let tokenizer = SmolLm2Tokenizer::from_json_slice_with_limits(
                    tokenizer_artifact.bytes(),
                    limits,
                )?;
                validate_dense_tokenizer_coherence(&spec, &tokenizer)?;
                (Box::new(tokenizer), None)
            }
            ModelFamily::Qwen2 => {
                let tokenizer = Qwen2Tokenizer::from_json_slice_with_limits(
                    tokenizer_artifact.bytes(),
                    limits,
                )?;
                validate_padded_tokenizer_coherence(&spec, &tokenizer)?;
                let tokenizer_config_artifact = session.read_once(
                    Path::new(TOKENIZER_CONFIG_FILENAME),
                    limits.config_bytes(),
                    "tokenizer_config.json",
                )?;
                let tokenizer_config = Qwen2TokenizerConfig::from_json_slice_with_limits(
                    tokenizer_config_artifact.bytes(),
                    limits,
                )?;
                tokenizer_config.validate_tokenizer(&tokenizer)?;
                validate_qwen_context_bound(&spec, &tokenizer_config)?;
                (Box::new(tokenizer), Some(tokenizer_config))
            }
        };

        let weights = LoadedWeights::load_verified_subset(&mut session, &spec, limits)?;
        let verified_files = session.finish_exact()?;

        Ok(Self {
            config,
            spec,
            tokenizer,
            qwen2_tokenizer_config,
            weights,
            verified_files,
        })
    }

    /// Returns the validated source configuration and its inert-field warnings.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the canonical execution description.
    #[must_use]
    pub const fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// Returns the Rust-native tokenizer backend.
    #[must_use]
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Returns validated Qwen chat metadata, or `None` for a non-Qwen model.
    #[must_use]
    pub const fn qwen2_tokenizer_config(&self) -> Option<&Qwen2TokenizerConfig> {
        self.qwen2_tokenizer_config.as_ref()
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
            artifact: "riley-checkpoint.json".to_owned(),
            reason: format!(
                "manifest dtype {} differs from model dtype {}",
                provenance.dtype(),
                spec.dtype()
            ),
        });
    }
    Ok(())
}

fn validate_dense_tokenizer_coherence(
    spec: &ModelSpec,
    tokenizer: &SmolLm2Tokenizer,
) -> ModelResult<()> {
    let expected_vocabulary = spec.embedding().vocabulary_size();
    let actual_vocabulary = tokenizer.addressable_token_count();
    if actual_vocabulary != expected_vocabulary {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_FILENAME.to_owned(),
            reason: format!(
                "vocabulary size mismatch: config requires {expected_vocabulary}, tokenizer has {actual_vocabulary}"
            ),
        });
    }

    if let Some(bos) = spec.special_tokens().bos() {
        validate_smol_special_token(tokenizer, "bos_token_id", bos)?;
    }
    for &eos in spec.special_tokens().eos() {
        validate_smol_special_token(tokenizer, "eos_token_id", eos)?;
    }
    Ok(())
}

fn validate_padded_tokenizer_coherence(
    spec: &ModelSpec,
    tokenizer: &Qwen2Tokenizer,
) -> ModelResult<()> {
    let model_vocabulary = spec.embedding().vocabulary_size();
    let addressable = tokenizer.addressable_token_count();
    if addressable == 0 || addressable > model_vocabulary {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_FILENAME.to_owned(),
            reason: format!(
                "tokenizer addressable token count {addressable} must be in 1..={model_vocabulary}"
            ),
        });
    }
    if let Some(bos) = spec.special_tokens().bos() {
        validate_qwen_special_token(tokenizer, "bos_token_id", bos)?;
    }
    for &eos in spec.special_tokens().eos() {
        validate_qwen_special_token(tokenizer, "eos_token_id", eos)?;
    }
    Ok(())
}

fn validate_smol_special_token(
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

fn validate_qwen_special_token(
    tokenizer: &Qwen2Tokenizer,
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

fn validate_qwen_context_bound(
    spec: &ModelSpec,
    tokenizer_config: &Qwen2TokenizerConfig,
) -> ModelResult<()> {
    let tokenizer_max = usize::try_from(tokenizer_config.model_max_length()).map_err(|_| {
        ModelError::NumericOverflow {
            field: "tokenizer_config.model_max_length".to_owned(),
        }
    })?;
    if tokenizer_max < spec.max_sequence_length() {
        return Err(ModelError::InvalidArtifact {
            artifact: TOKENIZER_CONFIG_FILENAME.to_owned(),
            reason: format!(
                "tokenizer model_max_length {tokenizer_max} is smaller than model context {}",
                spec.max_sequence_length()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::LoadedModel;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn loaded_model_remains_send_and_sync() {
        assert_send_sync::<LoadedModel>();
    }
}
