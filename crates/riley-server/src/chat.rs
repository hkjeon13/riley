//! `POST /v1/chat/completions` on top of the completion path.
//!
//! The checkpoint's own Jinja chat template renders the messages into one
//! prompt (as Transformers' `apply_chat_template` does), the prompt runs
//! through the unchanged completion validation and engine, and completion
//! responses are re-shaped into chat objects.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind};
use riley_model::ChatTemplateSource;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::domain::{GenerationRequest, RequestLimits};
use crate::openai::{
    CompletionRequest, CompletionStreamOptions, PromptInput, StopInput, ValidationError,
    ValidationErrorCode, normalize_completion_request,
};

const CHAT_OBJECT: &str = "chat.completion";
const CHAT_CHUNK_OBJECT: &str = "chat.completion.chunk";
const TEMPLATE_NAME: &str = "chat";
/// Bounds template execution; real chat templates use a few thousand steps.
const TEMPLATE_FUEL: u64 = 10_000_000;

/// A compiled chat template plus the special-token strings it may reference.
pub struct ChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Compiles a checkpoint chat template with Transformers' Jinja settings.
    ///
    /// # Errors
    ///
    /// Returns the template syntax error.
    pub fn compile(source: &ChatTemplateSource) -> Result<Self, String> {
        let mut environment = Environment::new();
        environment.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(|error| format!("chat template syntax setup failed: {error}"))?,
        );
        environment.set_fuel(Some(TEMPLATE_FUEL));
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment.add_function("strftime_now", |format: String| strftime_now(&format));
        environment
            .add_template_owned(TEMPLATE_NAME, source.template.clone())
            .map_err(|error| format!("chat template does not compile: {error}"))?;
        Ok(Self {
            environment,
            bos_token: source.bos_token.clone().unwrap_or_default(),
            eos_token: source.eos_token.clone().unwrap_or_default(),
        })
    }

    /// Renders messages into one prompt.
    ///
    /// # Errors
    ///
    /// Returns the template's own error, e.g. from `raise_exception`.
    pub fn render(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
        template_kwargs: &Map<String, Value>,
    ) -> Result<String, String> {
        let mut context = template_kwargs.clone();
        context.insert("messages".to_owned(), Value::Array(messages.to_vec()));
        context.insert(
            "add_generation_prompt".to_owned(),
            Value::Bool(add_generation_prompt),
        );
        context.insert(
            "bos_token".to_owned(),
            Value::String(self.bos_token.clone()),
        );
        context.insert(
            "eos_token".to_owned(),
            Value::String(self.eos_token.clone()),
        );
        self.environment
            .get_template(TEMPLATE_NAME)
            .and_then(|template| template.render(Serde(&context)))
            .map_err(|error| match error.kind() {
                ErrorKind::InvalidOperation => error
                    .detail()
                    .map_or_else(|| error.to_string(), str::to_owned),
                _ => format!("chat template rendering failed: {error}"),
            })
    }
}

/// Transformers' `strftime_now`, in UTC, for the common directives.
fn strftime_now(format: &str) -> String {
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let (year, month, day) = civil_from_days(seconds / 86_400);
    let time = seconds % 86_400;
    let month_name = usize::try_from(month - 1)
        .ok()
        .and_then(|index| MONTHS.get(index))
        .copied()
        .unwrap_or("January");
    let mut output = String::new();
    let mut directives = format.chars();
    while let Some(character) = directives.next() {
        if character != '%' {
            output.push(character);
            continue;
        }
        match directives.next() {
            Some('d') => output.push_str(&format!("{day:02}")),
            Some('m') => output.push_str(&format!("{month:02}")),
            Some('Y') => output.push_str(&year.to_string()),
            Some('y') => output.push_str(&format!("{:02}", year % 100)),
            Some('B') => output.push_str(month_name),
            Some('b') => output.push_str(&month_name[..3]),
            Some('H') => output.push_str(&format!("{:02}", time / 3600)),
            Some('M') => output.push_str(&format!("{:02}", time / 60 % 60)),
            Some('S') => output.push_str(&format!("{:02}", time % 60)),
            Some('%') | None => output.push('%'),
            Some(other) => {
                output.push('%');
                output.push(other);
            }
        }
    }
    output
}

/// Proleptic Gregorian date of a day count since 1970-01-01 (Hinnant).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// One chat message; content is a string or a list of text parts.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatMessageInput {
    /// `system`, `user`, `assistant`, or another role the template accepts.
    pub role: String,
    /// Message text; `null` is accepted for an empty assistant turn.
    #[serde(default)]
    pub content: Option<Value>,
    /// Retained so a request with tool calls is rejected explicitly.
    #[serde(flatten)]
    pub unsupported_fields: BTreeMap<String, Value>,
}

/// JSON request accepted by `POST /v1/chat/completions`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    /// Requested model identifier.
    pub model: String,
    /// Conversation rendered with the model's chat template.
    pub messages: Vec<ChatMessageInput>,
    /// Maximum generated tokens; defaults to the server limit.
    #[serde(default)]
    pub max_tokens: Option<u64>,
    /// Newer name of `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<u64>,
    /// Sampling temperature in `[0, 2]`.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus probability in `(0, 1]`.
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Optional deterministic sampling seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// One stop string or an ordered array of stop strings.
    #[serde(default)]
    pub stop: Option<StopInput>,
    /// Selects SSE when true.
    #[serde(default)]
    pub stream: Option<bool>,
    /// Optional terminal usage reporting for streaming responses.
    #[serde(default)]
    pub stream_options: Option<CompletionStreamOptions>,
    /// Exposes actual model token IDs when explicitly enabled.
    #[serde(default)]
    pub return_token_ids: Option<bool>,
    /// Only one choice is supported.
    #[serde(default)]
    pub n: Option<u64>,
    /// Only the neutral value `0` is accepted.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Only the neutral value `0` is accepted.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Log probabilities are not implemented; `false` is accepted.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// End-user label; accepted and ignored.
    #[serde(default)]
    pub user: Option<String>,
    /// Appends the assistant turn header (default true).
    #[serde(default)]
    pub add_generation_prompt: Option<bool>,
    /// Extra template variables, e.g. Qwen3's `enable_thinking`.
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    /// Unknown compatibility fields are retained for an explicit rejection.
    #[serde(flatten)]
    pub unsupported_fields: BTreeMap<String, Value>,
}

/// Why a chat request could not become a generation request.
#[derive(Debug)]
pub enum ChatRequestError {
    /// The same validation failures as `/v1/completions`.
    Invalid(ValidationError),
    /// The chat template rejected the conversation; carries its message.
    Template(String),
}

/// Renders a chat request into a generation request.
///
/// # Errors
///
/// Returns [`ChatRequestError`] for unsupported fields (tools, non-text
/// content), a template error, or any completion validation failure.
pub fn normalize_chat_request(
    request: ChatCompletionRequest,
    template: &ChatTemplate,
    limits: RequestLimits,
) -> Result<GenerationRequest, ChatRequestError> {
    let invalid = |parameter, message| {
        ChatRequestError::Invalid(ValidationError::new(
            parameter,
            ValidationErrorCode::UnsupportedParameter,
            message,
        ))
    };
    if ["tools", "tool_choice", "functions", "function_call"]
        .iter()
        .any(|field| request.unsupported_fields.contains_key(*field))
    {
        return Err(invalid("tools", "tool calling is not supported"));
    }
    if request.logprobs.unwrap_or(false) {
        return Err(invalid("logprobs", "logprobs is not supported"));
    }
    if request.messages.is_empty() {
        return Err(ChatRequestError::Invalid(ValidationError::new(
            "messages",
            ValidationErrorCode::InvalidValue,
            "messages must not be empty",
        )));
    }
    let messages = template_messages(request.messages)?;
    let prompt = template
        .render(
            &messages,
            request.add_generation_prompt.unwrap_or(true),
            &request.chat_template_kwargs.unwrap_or_default(),
        )
        .map_err(ChatRequestError::Template)?;

    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(limits.max_output_tokens as u64);
    let completion = CompletionRequest {
        model: request.model,
        prompt: PromptInput::Text(prompt),
        max_tokens: Some(max_tokens),
        temperature: request.temperature,
        top_p: request.top_p,
        seed: request.seed,
        stop: request.stop,
        stream: request.stream,
        return_token_ids: request.return_token_ids,
        stream_options: request.stream_options,
        n: request.n,
        best_of: None,
        echo: None,
        logprobs: None,
        suffix: None,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        unsupported_fields: request.unsupported_fields,
    };
    let mut generation =
        normalize_completion_request(completion, limits).map_err(ChatRequestError::Invalid)?;
    // The template already wrote any begin-of-text token.
    generation.add_special_tokens = false;
    Ok(generation)
}

/// Converts request messages into the `{role, content, name?}` objects that
/// chat templates read, joining text parts.
fn template_messages(input: Vec<ChatMessageInput>) -> Result<Vec<Value>, ChatRequestError> {
    let invalid = |message| {
        ChatRequestError::Invalid(ValidationError::new(
            "messages",
            ValidationErrorCode::UnsupportedParameter,
            message,
        ))
    };
    let mut messages = Vec::with_capacity(input.len());
    for message in input {
        if !message
            .unsupported_fields
            .keys()
            .all(|field| field == "name")
        {
            return Err(invalid(
                "only role, content and name are supported in messages",
            ));
        }
        let content = match message.content {
            None | Some(Value::Null) => String::new(),
            Some(Value::String(text)) => text,
            Some(Value::Array(parts)) => {
                let mut text = String::new();
                for part in parts {
                    match (part.get("type").and_then(Value::as_str), part.get("text")) {
                        (Some("text"), Some(Value::String(part))) => text.push_str(part),
                        _ => {
                            return Err(invalid("only text content parts are supported"));
                        }
                    }
                }
                text
            }
            Some(_) => {
                return Err(invalid("content must be a string or text parts"));
            }
        };
        let mut rendered = json!({"role": message.role, "content": content});
        if let Some(name) = message.unsupported_fields.get("name") {
            rendered["name"] = name.clone();
        }
        messages.push(rendered);
    }
    Ok(messages)
}

/// Re-shapes a `text_completion` response body into a `chat.completion`.
pub fn chat_response(response: &mut Value) {
    response["object"] = Value::from(CHAT_OBJECT);
    if let Some(choices) = response["choices"].as_array_mut() {
        for choice in choices {
            if let Some(choice) = choice.as_object_mut() {
                let text = choice.remove("text").unwrap_or_else(|| Value::from(""));
                choice.insert(
                    "message".to_owned(),
                    json!({"role": "assistant", "content": text}),
                );
            }
        }
    }
}

/// Re-shapes `text_completion` SSE frames into `chat.completion.chunk` ones.
#[derive(Default)]
pub struct ChatChunkShaper {
    role_sent: bool,
}

impl ChatChunkShaper {
    /// Returns the chat form of one SSE frame; `[DONE]` and errors pass as is.
    #[must_use]
    pub fn shape(&mut self, frame: &str) -> String {
        let Some(payload) = frame
            .strip_prefix("data: ")
            .and_then(|rest| rest.strip_suffix("\n\n"))
        else {
            return frame.to_owned();
        };
        let Ok(mut chunk) = serde_json::from_str::<Value>(payload) else {
            return frame.to_owned();
        };
        if chunk.get("object").and_then(Value::as_str) != Some("text_completion") {
            return frame.to_owned();
        }
        chunk["object"] = Value::from(CHAT_CHUNK_OBJECT);
        if let Some(choices) = chunk["choices"].as_array_mut() {
            for choice in choices {
                let Some(choice) = choice.as_object_mut() else {
                    continue;
                };
                let text = choice.remove("text").unwrap_or_else(|| Value::from(""));
                let finishing = choice
                    .get("finish_reason")
                    .is_some_and(|reason| !reason.is_null());
                let mut delta = Map::new();
                if !self.role_sent {
                    delta.insert("role".to_owned(), Value::from("assistant"));
                    self.role_sent = true;
                }
                if !(finishing && text.as_str() == Some("")) {
                    delta.insert("content".to_owned(), text);
                }
                choice.insert("delta".to_owned(), Value::Object(delta));
            }
        }
        format!("data: {chunk}\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(source: &str) -> ChatTemplate {
        ChatTemplate::compile(&ChatTemplateSource {
            template: source.to_owned(),
            bos_token: Some("<s>".to_owned()),
            eos_token: Some("</s>".to_owned()),
        })
        .unwrap()
    }

    fn limits() -> RequestLimits {
        RequestLimits {
            max_model_bytes: 64,
            max_prompt_bytes: 4096,
            max_output_tokens: 128,
            max_stop_sequences: 4,
            max_stop_sequence_bytes: 64,
            max_total_stop_bytes: 256,
        }
    }

    fn request(body: Value) -> ChatCompletionRequest {
        serde_json::from_value(body).unwrap()
    }

    const CHATML: &str = "{{ bos_token }}{% for message in messages %}\
<|im_start|>{{ message.role }}\n{{ message.content.strip() }}<|im_end|>\n{% endfor %}\
{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}\
{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}";

    #[test]
    fn renders_with_python_methods_kwargs_and_no_auto_bos() {
        let generation = normalize_chat_request(
            request(json!({"model": "m", "messages": [
                {"role": "system", "content": " be brief "},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "chat_template_kwargs": {"enable_thinking": false}})),
            &template(CHATML),
            limits(),
        )
        .unwrap();
        assert_eq!(
            generation.prompt,
            "<s><|im_start|>system\nbe brief<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n\
<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(!generation.add_special_tokens);
        assert_eq!(generation.max_new_tokens, 128);
    }

    #[test]
    fn template_errors_and_unsupported_requests_are_reported() {
        let strict = template(
            "{% if messages[0].role != 'user' %}{{ raise_exception('first must be user') }}{% endif %}",
        );
        let error = normalize_chat_request(
            request(json!({"model": "m", "messages": [{"role": "assistant", "content": "x"}]})),
            &strict,
            limits(),
        )
        .unwrap_err();
        assert!(
            matches!(error, ChatRequestError::Template(message) if message == "first must be user")
        );

        for body in [
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": []}),
            json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
            json!({"model": "m", "messages": [{"role": "user", "content": "x", "tool_calls": []}]}),
            json!({"model": "m", "messages": []}),
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 1000}),
        ] {
            assert!(matches!(
                normalize_chat_request(request(body), &template(CHATML), limits()),
                Err(ChatRequestError::Invalid(_))
            ));
        }
    }

    #[test]
    fn responses_and_chunks_take_the_chat_shape() {
        let mut response = json!({"id": "x", "object": "text_completion", "choices": [
            {"text": "Paris", "index": 0, "logprobs": null, "finish_reason": "stop"}]});
        chat_response(&mut response);
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(
            response["choices"][0]["message"],
            json!({"role": "assistant", "content": "Paris"})
        );
        assert!(response["choices"][0].get("text").is_none());

        let mut shaper = ChatChunkShaper::default();
        let frame = |text: &str, finish: Value| {
            format!(
                "data: {}\n\n",
                json!({"object": "text_completion", "choices": [
                    {"text": text, "index": 0, "finish_reason": finish}]})
            )
        };
        let first: Value = serde_json::from_str(
            shaper
                .shape(&frame("Pa", Value::Null))
                .strip_prefix("data: ")
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(first["object"], "chat.completion.chunk");
        assert_eq!(
            first["choices"][0]["delta"],
            json!({"role": "assistant", "content": "Pa"})
        );
        let last: Value = serde_json::from_str(
            shaper
                .shape(&frame("", Value::from("stop")))
                .strip_prefix("data: ")
                .unwrap()
                .trim(),
        )
        .unwrap();
        assert_eq!(last["choices"][0]["delta"], json!({}));
        assert_eq!(shaper.shape("data: [DONE]\n\n"), "data: [DONE]\n\n");
        let error = "data: {\"error\":{\"message\":\"x\"}}\n\n";
        assert_eq!(shaper.shape(error), error);
    }

    #[test]
    fn strftime_now_formats_utc_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_390), (2025, 10, 29));
        let today = strftime_now("%d %b %Y");
        assert_eq!(today.len(), 11, "{today}");
    }
}
