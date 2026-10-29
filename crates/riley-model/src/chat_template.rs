//! Chat-template source text read from a checkpoint's tokenizer metadata.
//!
//! The template is data for a sandboxed renderer in the server; nothing here
//! evaluates it. Only the fields templates commonly reference are extracted.

use serde_json::Value;

use crate::{ModelError, ModelResult};

/// Optional standalone template file; it takes precedence over the
/// `chat_template` field of `tokenizer_config.json`, as in Transformers.
pub const CHAT_TEMPLATE_FILENAME: &str = "chat_template.jinja";

/// A checkpoint's Jinja chat template and the special-token strings that
/// templates refer to as `bos_token` and `eos_token`.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatTemplateSource {
    /// Jinja source of the default chat template.
    pub template: String,
    /// Beginning-of-sequence token text, when the tokenizer config names one.
    pub bos_token: Option<String>,
    /// End-of-sequence token text, when the tokenizer config names one.
    pub eos_token: Option<String>,
}

impl ChatTemplateSource {
    /// Reads the template and special-token strings of a `tokenizer_config.json`.
    ///
    /// A config without a template yields an empty `template`. A list of named
    /// templates selects the one named `default`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidArtifact`] for malformed JSON or a
    /// template field of an unexpected shape.
    pub fn from_tokenizer_config(input: &[u8]) -> ModelResult<Self> {
        let config: Value = serde_json::from_slice(input).map_err(|_| invalid("malformed JSON"))?;
        let template = match config.get("chat_template") {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(template)) => template.clone(),
            Some(Value::Array(templates)) => templates
                .iter()
                .find(|entry| entry.get("name").and_then(Value::as_str) == Some("default"))
                .and_then(|entry| entry.get("template").and_then(Value::as_str))
                .ok_or_else(|| invalid("chat_template list has no default template"))?
                .to_owned(),
            Some(_) => return Err(invalid("chat_template must be a string or a list")),
        };
        Ok(Self {
            template,
            bos_token: special_token(&config, "bos_token"),
            eos_token: special_token(&config, "eos_token"),
        })
    }
}

/// A special token is either its text or an added-token object with `content`.
fn special_token(config: &Value, field: &str) -> Option<String> {
    match config.get(field)? {
        Value::String(text) => Some(text.clone()),
        Value::Object(token) => token.get("content")?.as_str().map(str::to_owned),
        _ => None,
    }
}

fn invalid(reason: &str) -> ModelError {
    ModelError::InvalidArtifact {
        artifact: "tokenizer_config.json".to_owned(),
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::ChatTemplateSource;

    #[test]
    fn reads_string_list_and_object_forms() {
        let source = ChatTemplateSource::from_tokenizer_config(
            br#"{"chat_template":"{{ bos_token }}x","bos_token":"<s>",
                "eos_token":{"content":"</s>","lstrip":false}}"#,
        )
        .unwrap();
        assert_eq!(source.template, "{{ bos_token }}x");
        assert_eq!(source.bos_token.as_deref(), Some("<s>"));
        assert_eq!(source.eos_token.as_deref(), Some("</s>"));

        let named = ChatTemplateSource::from_tokenizer_config(
            br#"{"chat_template":[{"name":"tool_use","template":"t"},
                {"name":"default","template":"d"}],"bos_token":null}"#,
        )
        .unwrap();
        assert_eq!(named.template, "d");
        assert_eq!(named.bos_token, None);

        let none = ChatTemplateSource::from_tokenizer_config(br#"{"eos_token":"e"}"#).unwrap();
        assert!(none.template.is_empty());
        assert!(ChatTemplateSource::from_tokenizer_config(br#"{"chat_template":1}"#).is_err());
        assert!(ChatTemplateSource::from_tokenizer_config(b"{").is_err());
    }
}
