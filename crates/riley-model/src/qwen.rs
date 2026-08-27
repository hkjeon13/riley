use std::collections::BTreeMap;

use serde::Deserialize;

use crate::artifact::sha256_hex;
use crate::{
    ArtifactKind, DecodeOptions, EncodeOptions, LoadLimits, ModelError, ModelResult,
    SmolLm2Tokenizer, Tokenizer, strict_json,
};

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const MAX_CHAT_TEMPLATE_BYTES: usize = 64 * 1024;
const MAX_CHAT_MESSAGES: usize = 4096;
const MAX_CHAT_CONTENT_BYTES: usize = 4 * 1024 * 1024;
const MAX_RENDERED_CHAT_BYTES: usize = 8 * 1024 * 1024;
const PINNED_CHAT_TEMPLATE_BYTES: usize = 2_507;
const PINNED_CHAT_TEMPLATE_SHA256: &str =
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f";

const PINNED_ADDED_TOKENS: &[(u32, &str, bool)] = &[
    (151_643, "<|endoftext|>", true),
    (151_644, IM_START, true),
    (151_645, IM_END, true),
    (151_646, "<|object_ref_start|>", true),
    (151_647, "<|object_ref_end|>", true),
    (151_648, "<|box_start|>", true),
    (151_649, "<|box_end|>", true),
    (151_650, "<|quad_start|>", true),
    (151_651, "<|quad_end|>", true),
    (151_652, "<|vision_start|>", true),
    (151_653, "<|vision_end|>", true),
    (151_654, "<|vision_pad|>", true),
    (151_655, "<|image_pad|>", true),
    (151_656, "<|video_pad|>", true),
    (151_657, "<tool_call>", false),
    (151_658, "</tool_call>", false),
    (151_659, "<|fim_prefix|>", false),
    (151_660, "<|fim_middle|>", false),
    (151_661, "<|fim_suffix|>", false),
    (151_662, "<|fim_pad|>", false),
    (151_663, "<|repo_name|>", false),
    (151_664, "<|file_sep|>", false),
];

/// Strict tokenizer for the pinned `Qwen2.5-0.5B-Instruct` artifact profile.
///
/// The implementation reuses the same `ByteLevel` BPE engine as `SmolLM2` while
/// validating and executing Qwen's NFC + Split pipeline explicitly.
pub struct Qwen2Tokenizer {
    inner: SmolLm2Tokenizer,
}

impl Qwen2Tokenizer {
    /// Parses a Qwen tokenizer artifact with production limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, oversized, or unsupported tokenizer
    /// metadata and for an inconsistent addressable token domain.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::default())
    }

    /// Parses a Qwen tokenizer artifact with explicit untrusted-input limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed, oversized, or unsupported tokenizer
    /// metadata and for an inconsistent addressable token domain.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        Ok(Self {
            inner: SmolLm2Tokenizer::from_qwen2_json_slice_with_limits(input, limits)?,
        })
    }

    /// Returns the exclusive upper bound of addressable token IDs.
    #[must_use]
    pub fn addressable_token_count(&self) -> usize {
        self.inner.addressable_token_count()
    }

    /// Returns whether an ID is addressable by the tokenizer.
    #[must_use]
    pub fn contains_id(&self, id: u32) -> bool {
        self.inner.contains_id(id)
    }

    /// Returns the serialized token content for an addressable ID.
    #[must_use]
    pub fn token_for_id(&self, id: u32) -> Option<&str> {
        self.inner.token_for_id(id)
    }

    /// Returns whether an ID is declared special by tokenizer metadata.
    #[must_use]
    pub fn is_special_id(&self, id: u32) -> bool {
        self.inner.is_special_id(id)
    }
}

impl Tokenizer for Qwen2Tokenizer {
    fn encode(&self, input: &str, options: EncodeOptions) -> ModelResult<Vec<u32>> {
        self.inner.encode(input, options)
    }

    fn addressable_token_count(&self) -> usize {
        self.inner.addressable_token_count()
    }

    fn maximum_decoded_token_bytes(&self) -> usize {
        self.inner.maximum_decoded_token_bytes()
    }

    fn decoded_bytes_len(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<usize> {
        self.inner.decoded_bytes_len(ids, options)
    }

    fn decode_token_bytes_into(
        &self,
        id: u32,
        options: DecodeOptions,
        destination: &mut [u8],
    ) -> ModelResult<usize> {
        self.inner.decode_token_bytes_into(id, options, destination)
    }

    fn decode_bytes(
        &self,
        ids: &[u32],
        options: DecodeOptions,
        destination: &mut [u8],
    ) -> ModelResult<usize> {
        self.inner.decode_bytes(ids, options, destination)
    }

    fn decode(&self, ids: &[u32], options: DecodeOptions) -> ModelResult<String> {
        self.inner.decode(ids, options)
    }
}

/// Role accepted by the bounded Qwen no-tools chat renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    /// System instruction.
    System,
    /// User message.
    User,
    /// Assistant message without tool calls.
    Assistant,
    /// Tool response, rejected by the PR12 no-tools profile.
    Tool,
}

impl ChatRole {
    const fn serialized(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// Borrowed chat message rendered by [`Qwen2TokenizerConfig::render_chat`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChatMessage<'a> {
    /// Message role.
    pub role: ChatRole,
    /// Unescaped message content, matching the pinned template semantics.
    pub content: &'a str,
    /// Whether assistant tool calls are present.
    ///
    /// Tool calls are rejected by the PR12 no-tools profile.
    pub has_tool_calls: bool,
}

/// Options for the bounded Qwen no-tools chat renderer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ChatTemplateOptions {
    /// Appends the assistant generation prefix.
    pub add_generation_prompt: bool,
    /// Indicates that a tools collection was supplied.
    ///
    /// Any non-empty tools collection is rejected by the PR12 profile.
    pub tools_requested: bool,
}

/// Validated `tokenizer_config.json` metadata for the pinned Qwen profile.
pub struct Qwen2TokenizerConfig {
    chat_template: String,
    added_tokens: BTreeMap<u32, ConfigAddedToken>,
    eos_token: String,
    pad_token: String,
    model_max_length: u64,
}

impl Qwen2TokenizerConfig {
    /// Parses bounded `tokenizer_config.json` metadata.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed metadata, a tokenizer variant other than
    /// the pinned Qwen2.5 profile, or a changed chat-template semantic profile.
    pub fn from_json_slice(input: &[u8]) -> ModelResult<Self> {
        Self::from_json_slice_with_limits(input, LoadLimits::default())
    }

    /// Parses bounded `tokenizer_config.json` metadata with explicit limits.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed metadata, a tokenizer variant other than
    /// the pinned Qwen2.5 profile, or a changed chat-template semantic profile.
    pub fn from_json_slice_with_limits(input: &[u8], limits: LoadLimits) -> ModelResult<Self> {
        enforce_limit("tokenizer_config JSON", limits.config_bytes(), input.len())?;
        let raw: RawQwen2TokenizerConfig = strict_json::from_slice(input, ArtifactKind::Tokenizer)?;
        validate_config_scalars(&raw)?;
        validate_chat_template(&raw.chat_template)?;
        let added_tokens = validate_config_added_tokens(&raw.added_tokens_decoder)?;
        validate_additional_special_tokens(&raw.additional_special_tokens, &added_tokens)?;

        Ok(Self {
            chat_template: raw.chat_template,
            added_tokens,
            eos_token: raw.eos_token,
            pad_token: raw.pad_token,
            model_max_length: raw.model_max_length,
        })
    }

    /// Returns the original validated Jinja template metadata.
    #[must_use]
    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    /// Returns the configured EOS token content.
    #[must_use]
    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    /// Returns the configured padding token content.
    #[must_use]
    pub fn pad_token(&self) -> &str {
        &self.pad_token
    }

    /// Returns the tokenizer metadata context bound.
    #[must_use]
    pub const fn model_max_length(&self) -> u64 {
        self.model_max_length
    }

    /// Cross-checks tokenizer-config added-token metadata against tokenizer.json.
    ///
    /// # Errors
    ///
    /// Returns an error when content, special identity, or the exclusive
    /// addressable-ID upper bound differs.
    pub fn validate_tokenizer(&self, tokenizer: &Qwen2Tokenizer) -> ModelResult<()> {
        let expected_count = PINNED_ADDED_TOKENS
            .last()
            .and_then(|(id, _, _)| usize::try_from(*id).ok())
            .and_then(|id| id.checked_add(1))
            .ok_or_else(|| invalid_config("pinned addressable token count overflow"))?;
        if tokenizer.addressable_token_count() != expected_count {
            return Err(invalid_config(format!(
                "Qwen tokenizer addressable token count must be {expected_count}, found {}",
                tokenizer.addressable_token_count()
            )));
        }
        for (&id, expected) in &self.added_tokens {
            if tokenizer.token_for_id(id) != Some(expected.content.as_str())
                || tokenizer.is_special_id(id) != expected.special
            {
                return Err(invalid_config(format!(
                    "tokenizer metadata disagrees for added token ID {id}"
                )));
            }
        }
        Ok(())
    }

    /// Renders the no-tools branch of the pinned Qwen2.5 chat template.
    ///
    /// If the first message is not `system`, the exact Qwen default system
    /// prompt is inserted. Later system messages are retained, matching the
    /// pinned template. Message content is intentionally not escaped.
    ///
    /// # Errors
    ///
    /// Returns an error for tools, tool calls, tool-role messages, excessive
    /// message/content/output size, or arithmetic overflow.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage<'_>],
        options: ChatTemplateOptions,
    ) -> ModelResult<String> {
        if options.tools_requested {
            return Err(invalid_config(
                "Qwen chat tools are unsupported by the no-tools renderer",
            ));
        }
        if messages.len() > MAX_CHAT_MESSAGES {
            return Err(ModelError::LimitExceeded {
                resource: "Qwen chat messages",
                limit: usize_to_u64(MAX_CHAT_MESSAGES),
                actual: Some(usize_to_u64(messages.len())),
            });
        }

        let mut content_bytes = 0_usize;
        for message in messages {
            if message.role == ChatRole::Tool || message.has_tool_calls {
                return Err(invalid_config(
                    "Qwen chat tool roles and tool_calls are unsupported",
                ));
            }
            content_bytes = content_bytes
                .checked_add(message.content.len())
                .ok_or_else(|| numeric_overflow("Qwen chat content bytes"))?;
        }
        if content_bytes > MAX_CHAT_CONTENT_BYTES {
            return Err(ModelError::LimitExceeded {
                resource: "Qwen chat content bytes",
                limit: usize_to_u64(MAX_CHAT_CONTENT_BYTES),
                actual: Some(usize_to_u64(content_bytes)),
            });
        }

        let first_is_system = messages
            .first()
            .is_some_and(|message| message.role == ChatRole::System);
        let system_content = if first_is_system {
            messages[0].content
        } else {
            DEFAULT_SYSTEM_PROMPT
        };
        let remaining = if first_is_system {
            &messages[1..]
        } else {
            messages
        };

        let mut required = rendered_message_len("system", system_content)?;
        for message in remaining {
            required = required
                .checked_add(rendered_message_len(
                    message.role.serialized(),
                    message.content,
                )?)
                .ok_or_else(|| numeric_overflow("rendered Qwen chat bytes"))?;
        }
        if options.add_generation_prompt {
            required = required
                .checked_add(IM_START.len() + "assistant\n".len())
                .ok_or_else(|| numeric_overflow("rendered Qwen chat bytes"))?;
        }
        if required > MAX_RENDERED_CHAT_BYTES {
            return Err(ModelError::LimitExceeded {
                resource: "rendered Qwen chat bytes",
                limit: usize_to_u64(MAX_RENDERED_CHAT_BYTES),
                actual: Some(usize_to_u64(required)),
            });
        }

        let mut output = String::with_capacity(required);
        push_message(&mut output, "system", system_content);
        for message in remaining {
            push_message(&mut output, message.role.serialized(), message.content);
        }
        if options.add_generation_prompt {
            output.push_str(IM_START);
            output.push_str("assistant\n");
        }
        debug_assert_eq!(output.len(), required);
        Ok(output)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawQwen2TokenizerConfig {
    add_bos_token: RawBool,
    add_prefix_space: RawBool,
    added_tokens_decoder: BTreeMap<String, RawConfigAddedToken>,
    additional_special_tokens: Vec<String>,
    bos_token: Option<String>,
    chat_template: String,
    clean_up_tokenization_spaces: RawBool,
    eos_token: String,
    errors: String,
    model_max_length: u64,
    pad_token: String,
    split_special_tokens: RawBool,
    tokenizer_class: String,
    unk_token: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfigAddedToken {
    content: String,
    lstrip: RawBool,
    normalized: RawBool,
    rstrip: RawBool,
    single_word: RawBool,
    special: RawBool,
}

#[derive(Deserialize)]
#[serde(transparent)]
struct RawBool(bool);

struct ConfigAddedToken {
    content: String,
    special: bool,
}

fn validate_config_scalars(raw: &RawQwen2TokenizerConfig) -> ModelResult<()> {
    if raw.add_bos_token.0
        || raw.add_prefix_space.0
        || raw.clean_up_tokenization_spaces.0
        || raw.split_special_tokens.0
        || raw.bos_token.is_some()
        || raw.unk_token.is_some()
        || raw.errors != "replace"
        || raw.tokenizer_class != "Qwen2Tokenizer"
        || raw.eos_token != IM_END
        || raw.pad_token != "<|endoftext|>"
        || raw.model_max_length == 0
    {
        return Err(invalid_config(
            "unsupported Qwen tokenizer_config scalar profile",
        ));
    }
    Ok(())
}

fn validate_config_added_tokens(
    raw: &BTreeMap<String, RawConfigAddedToken>,
) -> ModelResult<BTreeMap<u32, ConfigAddedToken>> {
    if raw.len() != PINNED_ADDED_TOKENS.len() {
        return Err(invalid_config(format!(
            "pinned Qwen added_tokens_decoder must contain {} entries",
            PINNED_ADDED_TOKENS.len()
        )));
    }
    let mut validated = BTreeMap::new();
    for (expected_id, expected_content, expected_special) in PINNED_ADDED_TOKENS {
        let key = expected_id.to_string();
        let token = raw
            .get(&key)
            .ok_or_else(|| invalid_config(format!("missing added token ID {expected_id}")))?;
        if token.content != *expected_content
            || token.special.0 != *expected_special
            || token.lstrip.0
            || token.normalized.0
            || token.rstrip.0
            || token.single_word.0
        {
            return Err(invalid_config(format!(
                "added token ID {expected_id} differs from the pinned profile"
            )));
        }
        validated.insert(
            *expected_id,
            ConfigAddedToken {
                content: token.content.clone(),
                special: token.special.0,
            },
        );
    }
    Ok(validated)
}

fn validate_additional_special_tokens(
    raw: &[String],
    added: &BTreeMap<u32, ConfigAddedToken>,
) -> ModelResult<()> {
    let expected = &PINNED_ADDED_TOKENS[1..14];
    if raw.len() != expected.len()
        || raw
            .iter()
            .zip(expected)
            .any(|(actual, (_, content, special))| actual != content || !special)
    {
        return Err(invalid_config(
            "additional_special_tokens differs from the pinned ordered profile",
        ));
    }
    if raw.iter().any(|content| {
        !added
            .values()
            .any(|token| token.special && token.content == *content)
    }) {
        return Err(invalid_config(
            "additional_special_tokens references a non-special or unknown token",
        ));
    }
    Ok(())
}

fn validate_chat_template(template: &str) -> ModelResult<()> {
    if template.is_empty() || template.len() > MAX_CHAT_TEMPLATE_BYTES {
        return Err(invalid_config(
            "Qwen chat_template has an invalid byte length",
        ));
    }
    if template.len() != PINNED_CHAT_TEMPLATE_BYTES
        || sha256_hex(template.as_bytes()) != PINNED_CHAT_TEMPLATE_SHA256
    {
        return Err(invalid_config(
            "unsupported Qwen chat_template: content differs from the pinned profile",
        ));
    }
    Ok(())
}

fn rendered_message_len(role: &str, content: &str) -> ModelResult<usize> {
    IM_START
        .len()
        .checked_add(role.len())
        .and_then(|length| length.checked_add(1))
        .and_then(|length| length.checked_add(content.len()))
        .and_then(|length| length.checked_add(IM_END.len()))
        .and_then(|length| length.checked_add(1))
        .ok_or_else(|| numeric_overflow("rendered Qwen chat bytes"))
}

fn push_message(output: &mut String, role: &str, content: &str) {
    output.push_str(IM_START);
    output.push_str(role);
    output.push('\n');
    output.push_str(content);
    output.push_str(IM_END);
    output.push('\n');
}

fn enforce_limit(resource: &'static str, limit: u64, actual: usize) -> ModelResult<()> {
    let actual = usize_to_u64(actual);
    if actual > limit {
        return Err(ModelError::LimitExceeded {
            resource,
            limit,
            actual: Some(actual),
        });
    }
    Ok(())
}

fn numeric_overflow(field: &str) -> ModelError {
    ModelError::NumericOverflow {
        field: field.to_owned(),
    }
}

fn invalid_config(reason: impl Into<String>) -> ModelError {
    ModelError::InvalidTokenizer {
        reason: reason.into(),
    }
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::*;
    use crate::tokenizer::QWEN2_SPLIT_REGEX;

    fn byte_encoder(byte: u8, extra: &mut u32) -> char {
        let codepoint = if (b'!'..=b'~').contains(&byte)
            || (0xA1..=0xAC).contains(&byte)
            || (0xAE..=0xFF).contains(&byte)
        {
            u32::from(byte)
        } else {
            let codepoint = 256 + *extra;
            *extra += 1;
            codepoint
        };
        char::from_u32(codepoint).expect("test ByteLevel code point")
    }

    fn tokenizer_fixture() -> String {
        let mut vocab = Map::new();
        let mut extra = 0_u32;
        for byte in u8::MIN..=u8::MAX {
            vocab.insert(
                byte_encoder(byte, &mut extra).to_string(),
                Value::from(byte),
            );
        }
        serde_json::to_string(&json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {
                    "id": 256,
                    "content": "<|end|>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                },
                {
                    "id": 257,
                    "content": "<fim>",
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": false
                }
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN2_SPLIT_REGEX},
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": false,
                        "use_regex": false
                    }
                ]
            },
            "post_processor": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "",
                "fuse_unk": false,
                "byte_fallback": false,
                "vocab": vocab,
                "merges": []
            }
        }))
        .expect("serialize synthetic Qwen tokenizer")
    }

    fn config_fixture() -> String {
        let mut decoder = Map::new();
        for (id, content, special) in PINNED_ADDED_TOKENS {
            decoder.insert(
                id.to_string(),
                json!({
                    "content": content,
                    "lstrip": false,
                    "normalized": false,
                    "rstrip": false,
                    "single_word": false,
                    "special": special
                }),
            );
        }
        let template = include_str!("../tests/fixtures/qwen2.5-0.5b-instruct-chat-template.jinja");
        serde_json::to_string(&json!({
            "add_bos_token": false,
            "add_prefix_space": false,
            "added_tokens_decoder": decoder,
            "additional_special_tokens": PINNED_ADDED_TOKENS[1..14]
                .iter()
                .map(|(_, content, _)| *content)
                .collect::<Vec<_>>(),
            "bos_token": null,
            "chat_template": template,
            "clean_up_tokenization_spaces": false,
            "eos_token": IM_END,
            "errors": "replace",
            "model_max_length": 131_072,
            "pad_token": "<|endoftext|>",
            "split_special_tokens": false,
            "tokenizer_class": "Qwen2Tokenizer",
            "unk_token": null
        }))
        .expect("serialize synthetic Qwen tokenizer config")
    }

    #[test]
    fn qwen_nfc_split_bytelevel_round_trip_covers_korean_digits_and_newlines() {
        let tokenizer = Qwen2Tokenizer::from_json_slice(tokenizer_fixture().as_bytes())
            .expect("synthetic Qwen tokenizer");
        let decomposed = "\u{1100}\u{1161} CAN'T 12\n\n안녕 /ok";
        let composed = "가 CAN'T 12\n\n안녕 /ok";
        let decomposed_ids = tokenizer
            .encode(decomposed, EncodeOptions::default())
            .expect("encode decomposed NFC input");
        let composed_ids = tokenizer
            .encode(composed, EncodeOptions::default())
            .expect("encode composed NFC input");
        assert_eq!(decomposed_ids, composed_ids);
        assert_eq!(
            tokenizer
                .decode(&decomposed_ids, DecodeOptions::default())
                .expect("strict Qwen decode"),
            composed
        );
        assert_eq!(tokenizer.addressable_token_count(), 258);
        assert!(tokenizer.contains_id(257));
        assert!(!tokenizer.contains_id(258));
    }

    #[test]
    fn qwen_added_tokens_extend_the_domain_and_decode_special_identity_strictly() {
        let tokenizer = Qwen2Tokenizer::from_json_slice(tokenizer_fixture().as_bytes()).unwrap();
        let ids = tokenizer
            .encode("<|end|><fim>", EncodeOptions::default())
            .unwrap();
        assert_eq!(ids, [256, 257]);
        assert!(tokenizer.is_special_id(256));
        assert!(!tokenizer.is_special_id(257));
        assert_eq!(
            tokenizer.decode(&ids, DecodeOptions::default()).unwrap(),
            "<|end|><fim>"
        );
        assert_eq!(
            tokenizer
                .decode(
                    &ids,
                    DecodeOptions {
                        skip_special_tokens: true,
                    },
                )
                .unwrap(),
            "<fim>"
        );
    }

    #[test]
    fn qwen_tokenizer_rejects_pipeline_and_addressable_domain_variants() {
        let fixture = tokenizer_fixture();
        let mut variants = vec![
            fixture.replace(r#"{"type":"NFC"}"#, r#"{"type":"NFD"}"#),
            fixture.replace("(?i:'s", "(?i:'z"),
            fixture.replace(r#""trim_offsets":false"#, r#""trim_offsets":true"#),
            fixture.replace(r#""id":257"#, r#""id":258"#),
            fixture.replace(
                r#""continuing_subword_prefix":"""#,
                r#""continuing_subword_prefix":null"#,
            ),
            fixture.replace(r#""end_of_word_suffix":"""#, r#""end_of_word_suffix":"x""#),
        ];
        for unexpected in [Value::Null, Value::Bool(false)] {
            let mut variant: Value = serde_json::from_str(&fixture).unwrap();
            variant["model"]
                .as_object_mut()
                .unwrap()
                .insert("ignore_merges".to_owned(), unexpected);
            variants.push(serde_json::to_string(&variant).unwrap());
        }
        for variant in variants {
            assert!(matches!(
                Qwen2Tokenizer::from_json_slice(variant.as_bytes()),
                Err(ModelError::InvalidTokenizer { .. })
            ));
        }
    }

    #[test]
    fn qwen_no_tools_chat_renderer_matches_default_and_explicit_system_semantics() {
        let config = Qwen2TokenizerConfig::from_json_slice(config_fixture().as_bytes())
            .expect("synthetic pinned config metadata");
        assert_eq!(config.eos_token(), IM_END);
        assert_eq!(config.pad_token(), "<|endoftext|>");
        assert_eq!(config.model_max_length(), 131_072);

        let rendered = config
            .render_chat(
                &[ChatMessage {
                    role: ChatRole::User,
                    content: "안녕?",
                    has_tool_calls: false,
                }],
                ChatTemplateOptions {
                    add_generation_prompt: true,
                    tools_requested: false,
                },
            )
            .unwrap();
        assert_eq!(
            rendered,
            concat!(
                "<|im_start|>system\n",
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                "<|im_end|>\n",
                "<|im_start|>user\n안녕?<|im_end|>\n",
                "<|im_start|>assistant\n"
            )
        );

        let explicit = config
            .render_chat(
                &[
                    ChatMessage {
                        role: ChatRole::System,
                        content: "first",
                        has_tool_calls: false,
                    },
                    ChatMessage {
                        role: ChatRole::System,
                        content: "second",
                        has_tool_calls: false,
                    },
                    ChatMessage {
                        role: ChatRole::Assistant,
                        content: "done",
                        has_tool_calls: false,
                    },
                ],
                ChatTemplateOptions::default(),
            )
            .unwrap();
        assert_eq!(
            explicit,
            concat!(
                "<|im_start|>system\nfirst<|im_end|>\n",
                "<|im_start|>system\nsecond<|im_end|>\n",
                "<|im_start|>assistant\ndone<|im_end|>\n"
            )
        );
    }

    #[test]
    fn qwen_chat_metadata_and_tool_variants_fail_closed() {
        let fixture = config_fixture();
        let config = Qwen2TokenizerConfig::from_json_slice(fixture.as_bytes()).unwrap();
        assert!(matches!(
            config.render_chat(
                &[],
                ChatTemplateOptions {
                    add_generation_prompt: false,
                    tools_requested: true,
                }
            ),
            Err(ModelError::InvalidTokenizer { .. })
        ));
        for message in [
            ChatMessage {
                role: ChatRole::Tool,
                content: "result",
                has_tool_calls: false,
            },
            ChatMessage {
                role: ChatRole::Assistant,
                content: "call",
                has_tool_calls: true,
            },
        ] {
            assert!(
                config
                    .render_chat(&[message], ChatTemplateOptions::default())
                    .is_err()
            );
        }

        let changed_template = fixture.replace("message.tool_calls", "message.calls");
        assert!(matches!(
            Qwen2TokenizerConfig::from_json_slice(changed_template.as_bytes()),
            Err(ModelError::InvalidTokenizer { .. })
        ));
        let mut changed_semantics: Value = serde_json::from_str(&fixture).unwrap();
        let template = changed_semantics["chat_template"].as_str().unwrap();
        let changed = template.replacen("assistant\\n", "assistant changed\\n", 1);
        assert_ne!(changed, template);
        changed_semantics["chat_template"] = json!(changed);
        assert!(matches!(
            Qwen2TokenizerConfig::from_json_slice(
                serde_json::to_string(&changed_semantics)
                    .unwrap()
                    .as_bytes()
            ),
            Err(ModelError::InvalidTokenizer { .. })
        ));
        let changed_special = fixture.replacen(r#""special":false"#, r#""special":true"#, 1);
        assert!(matches!(
            Qwen2TokenizerConfig::from_json_slice(changed_special.as_bytes()),
            Err(ModelError::InvalidTokenizer { .. })
        ));

        let too_many = vec![
            ChatMessage {
                role: ChatRole::User,
                content: "x",
                has_tool_calls: false,
            };
            MAX_CHAT_MESSAGES + 1
        ];
        assert!(matches!(
            config.render_chat(&too_many, ChatTemplateOptions::default()),
            Err(ModelError::LimitExceeded {
                resource: "Qwen chat messages",
                ..
            })
        ));
    }
}
