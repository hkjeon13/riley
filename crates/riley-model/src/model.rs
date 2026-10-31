use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::artifact::VerifiedArtifactSession;
use crate::{
    CHAT_TEMPLATE_FILENAME, ChatTemplateSource, CheckpointProvenance, LoadLimits, LoadedWeights,
    MistralTokenizer, ModelConfig, ModelError, ModelResult, ModelSpec, Qwen2Tokenizer,
    Qwen2TokenizerConfig, SmolLm2Tokenizer, Tokenizer,
};

/// Required Hugging Face configuration filename.
pub const CONFIG_FILENAME: &str = "config.json";

/// Required Hugging Face tokenizer filename.
pub const TOKENIZER_FILENAME: &str = "tokenizer.json";

/// Optional Hugging Face generation defaults; only its end-of-sequence IDs are
/// read, and only for a Llama-family checkpoint whose manifest lists the file.
pub const GENERATION_CONFIG_FILENAME: &str = "generation_config.json";

/// Tokenizer metadata: required for Qwen, optional for other families, which
/// read only its chat template and special-token strings.
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
    tokenizer: Arc<dyn Tokenizer>,
    qwen2_tokenizer_config: Option<Qwen2TokenizerConfig>,
    chat_template: Option<ChatTemplateSource>,
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
        let mut config = ModelConfig::from_json_slice_with_limits(config_artifact.bytes(), limits)?;
        // A checkpoint whose manifest lists its generation configuration also
        // stops on that file's end-of-sequence IDs, as Hugging Face generation
        // does. A manifest without the file keeps the IDs of `config.json`.
        let generation_config = Path::new(GENERATION_CONFIG_FILENAME);
        if session.provenance().lists(generation_config) {
            let artifact = session.read_once(
                generation_config,
                limits.config_bytes(),
                "generation_config.json",
            )?;
            match &mut config {
                ModelConfig::Llama(llama_config) | ModelConfig::Mistral(llama_config) => {
                    llama_config.extend_eos_from_generation_config(artifact.bytes(), limits)?;
                }
                ModelConfig::Qwen2(qwen_config) => {
                    qwen_config.extend_eos_from_generation_config(artifact.bytes(), limits)?;
                }
            }
        }
        let spec = config.to_model_spec();
        validate_manifest_dtype(session.provenance(), &spec)?;

        let tokenizer_artifact = session.read_once(
            Path::new(TOKENIZER_FILENAME),
            limits.tokenizer_bytes(),
            "tokenizer.json",
        )?;
        let mut chat_template = None;
        let (tokenizer, qwen2_tokenizer_config): (
            Arc<dyn Tokenizer>,
            Option<Qwen2TokenizerConfig>,
        ) = match &config {
            ModelConfig::Llama(llama_config) => {
                // The SmolLM2 artifact shape, else the Llama 3 one; the error
                // reported for neither is the SmolLM2 one.
                let mut tokenizer = SmolLm2Tokenizer::from_json_slice_with_limits(
                    tokenizer_artifact.bytes(),
                    limits,
                )
                .or_else(|error| {
                    SmolLm2Tokenizer::from_llama3_json_slice_with_limits(
                        tokenizer_artifact.bytes(),
                        limits,
                    )
                    .map_err(|_| error)
                })?;
                let tokenizer_config = Path::new(TOKENIZER_CONFIG_FILENAME);
                if tokenizer.bos_token_id().is_none()
                    && session.provenance().lists(tokenizer_config)
                {
                    let artifact = session.read_once(
                        tokenizer_config,
                        limits.config_bytes(),
                        "tokenizer_config.json",
                    )?;
                    if let Some(content) = metadata_bos_token(artifact.bytes()) {
                        tokenizer.set_bos_token_content(&content)?;
                    }
                    chat_template =
                        Some(ChatTemplateSource::from_tokenizer_config(artifact.bytes())?);
                }
                validate_dense_tokenizer_coherence(&spec, &tokenizer, llama_config.pad_token_id())?;
                (Arc::new(tokenizer), None)
            }
            ModelConfig::Mistral(config) => {
                let tokenizer = MistralTokenizer::from_json_slice_with_limits(
                    tokenizer_artifact.bytes(),
                    limits,
                )?;
                if tokenizer.addressable_token_count() != spec.embedding().vocabulary_size()
                    || spec.special_tokens().bos() != Some(tokenizer.bos_token_id())
                    || spec
                        .special_tokens()
                        .eos()
                        .iter()
                        .any(|id| !tokenizer.is_special_id(*id))
                    || config
                        .pad_token_id()
                        .is_some_and(|id| !tokenizer.is_special_id(id))
                {
                    return Err(ModelError::InvalidArtifact {
                        artifact: TOKENIZER_FILENAME.to_owned(),
                        reason:
                            "Mistral tokenizer vocabulary or special tokens disagree with config"
                                .to_owned(),
                    });
                }
                (Arc::new(tokenizer), None)
            }
            ModelConfig::Qwen2(_) => {
                let mut tokenizer = Qwen2Tokenizer::from_json_slice_with_limits(
                    tokenizer_artifact.bytes(),
                    limits,
                )?;
                validate_padded_tokenizer_coherence(&spec, &tokenizer)?;
                let tokenizer_config_artifact = session.read_once(
                    Path::new(TOKENIZER_CONFIG_FILENAME),
                    limits.config_bytes(),
                    "tokenizer_config.json",
                )?;
                // The pinned Qwen2.5 chat profile when the metadata matches it,
                // else the completions-only reading of the same file.
                let tokenizer_config = Qwen2TokenizerConfig::from_json_slice_with_limits(
                    tokenizer_config_artifact.bytes(),
                    limits,
                )
                .or_else(|_| {
                    Qwen2TokenizerConfig::from_json_slice_completions_only_with_limits(
                        tokenizer_config_artifact.bytes(),
                        limits,
                    )
                })?;
                tokenizer_config.validate_tokenizer(&tokenizer)?;
                validate_qwen_context_bound(&spec, &tokenizer_config)?;
                if let Some(content) = metadata_bos_token(tokenizer_config_artifact.bytes()) {
                    tokenizer.set_bos_token_content(&content)?;
                }
                chat_template = Some(ChatTemplateSource::from_tokenizer_config(
                    tokenizer_config_artifact.bytes(),
                )?);
                (Arc::new(tokenizer), Some(tokenizer_config))
            }
        };

        let chat_template = read_chat_template(&mut session, limits, chat_template)?;

        let weights = LoadedWeights::load_verified_subset(&mut session, &spec, limits)?;
        let verified_files = session.finish_exact()?;

        Ok(Self {
            config,
            spec,
            tokenizer,
            qwen2_tokenizer_config,
            chat_template,
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

    /// Returns a shared handle to the same immutable tokenizer, so prompts can
    /// be encoded off the thread that owns the model.
    #[must_use]
    pub fn shared_tokenizer(&self) -> Arc<dyn Tokenizer> {
        Arc::clone(&self.tokenizer)
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

    /// Returns the checkpoint's chat template, if its manifest lists one.
    #[must_use]
    pub const fn chat_template(&self) -> Option<&ChatTemplateSource> {
        self.chat_template.as_ref()
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

/// Completes the chat template: other families read `tokenizer_config.json`
/// only for it, and a listed `chat_template.jinja` replaces the template text.
/// The begin-of-sequence token that Hugging Face `LlamaTokenizerFast` adds
/// from `tokenizer_config.json` when the artifact itself has no template:
/// `add_bos_token: true` and the `bos_token` content (DeepSeek-R1-Distill).
/// Other tokenizer classes keep the artifact's own post-processing.
fn metadata_bos_token(input: &[u8]) -> Option<String> {
    let config: serde_json::Value = serde_json::from_slice(input).ok()?;
    let class = config.get("tokenizer_class")?.as_str()?;
    if !matches!(class, "LlamaTokenizerFast" | "LlamaTokenizer")
        || config.get("add_bos_token")?.as_bool() != Some(true)
    {
        return None;
    }
    match config.get("bos_token")? {
        serde_json::Value::String(content) => Some(content.clone()),
        serde_json::Value::Object(token) => token.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

fn read_chat_template(
    session: &mut VerifiedArtifactSession,
    limits: LoadLimits,
    mut chat_template: Option<ChatTemplateSource>,
) -> ModelResult<Option<ChatTemplateSource>> {
    let tokenizer_config = Path::new(TOKENIZER_CONFIG_FILENAME);
    if chat_template.is_none() && session.provenance().lists(tokenizer_config) {
        let artifact = session.read_once(
            tokenizer_config,
            limits.config_bytes(),
            "tokenizer_config.json",
        )?;
        chat_template = Some(ChatTemplateSource::from_tokenizer_config(artifact.bytes())?);
    }
    let template_file = Path::new(CHAT_TEMPLATE_FILENAME);
    if session.provenance().lists(template_file) {
        let artifact =
            session.read_once(template_file, limits.config_bytes(), "chat_template.jinja")?;
        let template = std::str::from_utf8(artifact.bytes())
            .map_err(|_| ModelError::InvalidArtifact {
                artifact: CHAT_TEMPLATE_FILENAME.to_owned(),
                reason: "template is not UTF-8".to_owned(),
            })?
            .to_owned();
        chat_template
            .get_or_insert_with(ChatTemplateSource::default)
            .template = template;
    }
    Ok(chat_template.filter(|source| !source.template.is_empty()))
}

fn validate_manifest_dtype(provenance: &CheckpointProvenance, spec: &ModelSpec) -> ModelResult<()> {
    if provenance.dtype() != spec.dtype() {
        return Err(ModelError::InvalidArtifact {
            artifact: provenance.manifest_filename().to_owned(),
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
    pad_token_id: Option<u32>,
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
    if let Some(pad) = pad_token_id {
        validate_smol_special_token(tokenizer, "pad_token_id", pad)?;
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
    // Outside the pinned profile the bound is advisory, as in HF tokenizers
    // and vLLM (DeepSeek-R1-Distill-Qwen ships 16384 for a 131072 model).
    if !tokenizer_config.has_pinned_chat_profile() {
        return Ok(());
    }
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

    #[test]
    fn metadata_bos_token_follows_llama_fast_add_bos_token_only() {
        use super::metadata_bos_token;
        let deepseek = br#"{"add_bos_token": true, "tokenizer_class": "LlamaTokenizerFast",
            "bos_token": {"__type": "AddedToken", "content": "<|begin|>", "normalized": true}}"#;
        assert_eq!(metadata_bos_token(deepseek).as_deref(), Some("<|begin|>"));
        for other in [
            br#"{"add_bos_token": false, "tokenizer_class": "LlamaTokenizerFast", "bos_token": "<s>"}"#.as_slice(),
            br#"{"add_bos_token": true, "tokenizer_class": "PreTrainedTokenizer", "bos_token": "<s>"}"#,
            br#"{"add_bos_token": true, "tokenizer_class": "GPT2Tokenizer", "bos_token": "<s>"}"#,
            br#"{"tokenizer_class": "LlamaTokenizerFast", "bos_token": "<s>"}"#,
        ] {
            assert_eq!(metadata_bos_token(other), None);
        }
    }

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn loaded_model_remains_send_and_sync() {
        assert_send_sync::<LoadedModel>();
    }
}
