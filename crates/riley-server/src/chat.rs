//! `POST /v1/chat/completions` on top of the completion path.
//!
//! The checkpoint's own Jinja chat template renders the messages into one
//! prompt (as Transformers' `apply_chat_template` does), the prompt runs
//! through the unchanged completion validation and engine, and completion
//! responses are re-shaped into chat objects.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{Environment, Error, ErrorKind};
use riley_model::ChatTemplateSource;
use riley_runtime::grammar::{Grammar, GrammarElement};
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::domain::{GenerationRequest, RequestLimits};
use crate::openai::{
    CompletionRequest, CompletionStreamOptions, PromptInput, StopInput, ValidationError,
    ValidationErrorCode, normalize_completion_request,
};
use crate::structured::{SchemaCompiler, SchemaError, compile_json_object, compile_schema};

const CHAT_OBJECT: &str = "chat.completion";
const CHAT_CHUNK_OBJECT: &str = "chat.completion.chunk";
const TEMPLATE_NAME: &str = "chat";
/// Bounds template execution; real chat templates use a few thousand steps.
const TEMPLATE_FUEL: u64 = 10_000_000;

/// Bounds `top_logprobs`, as the `OpenAI` API does.
const MAX_TOP_LOGPROBS: u8 = 20;
const HERMES_OPEN: &str = "<tool_call>";
const HERMES_CLOSE: &str = "</tool_call>";
const PYTHON_TAG: &str = "<|python_tag|>";

/// How a model family writes tool calls, known from its chat template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolStyle {
    /// `<tool_call>{"name": ..., "arguments": ...}</tool_call>`, as Qwen does.
    Hermes,
    /// A bare `{"name": ..., "parameters": ...}` object, as Llama 3 does.
    LlamaJson,
}

impl ToolStyle {
    fn detect(template: &str) -> Option<Self> {
        if template.contains(HERMES_OPEN) {
            Some(Self::Hermes)
        } else if template.contains("tool") && template.contains("\"parameters\"") {
            Some(Self::LlamaJson)
        } else {
            None
        }
    }
}

/// A compiled chat template plus the special-token strings it may reference.
pub struct ChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
    tool_style: Option<ToolStyle>,
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
            tool_style: ToolStyle::detect(&source.template),
        })
    }

    /// The format to parse this request's output in, when it offers tools.
    #[must_use]
    pub fn tool_parsing(&self, request: &ChatCompletionRequest) -> Option<ToolStyle> {
        offered_tools(request).and(self.tool_style)
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
        tools: Option<&[Value]>,
    ) -> Result<String, String> {
        let mut context = template_kwargs.clone();
        context.insert("messages".to_owned(), Value::Array(messages.to_vec()));
        if let Some(tools) = tools {
            context.insert("tools".to_owned(), Value::Array(tools.to_vec()));
        }
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
    /// Returns per-token log probabilities when true.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Top alternatives per token, 0 to 20; needs `logprobs`.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// `text`, `json_object` or `json_schema` output.
    #[serde(default)]
    pub response_format: Option<Value>,
    /// Functions the model may call.
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    /// `none`, `auto`, `required`, or one named function.
    #[serde(default)]
    pub tool_choice: Option<Value>,
    /// Whether a forced `tool_choice` may produce several calls.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
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
    /// A schema cannot be enforced; carries the parameter and the reason.
    Schema {
        /// The request field holding the schema.
        parameter: &'static str,
        /// What cannot be enforced.
        message: String,
    },
}

/// The tools a request offers, unless it turned them off.
fn offered_tools(request: &ChatCompletionRequest) -> Option<&[Value]> {
    if request.tool_choice.as_ref().and_then(Value::as_str) == Some("none") {
        return None;
    }
    request
        .tools
        .as_deref()
        .filter(|tools| !tools.is_empty())
}

fn schema_error(parameter: &'static str) -> impl Fn(SchemaError) -> ChatRequestError {
    move |error| ChatRequestError::Schema {
        parameter,
        message: error.0,
    }
}

fn invalid_value(parameter: &'static str, message: &'static str) -> ChatRequestError {
    ChatRequestError::Invalid(ValidationError::new(
        parameter,
        ValidationErrorCode::InvalidValue,
        message,
    ))
}

fn response_format_grammar(format: &Value) -> Result<Option<Arc<Grammar>>, ChatRequestError> {
    match format.get("type").and_then(Value::as_str) {
        Some("text") => Ok(None),
        Some("json_object") => compile_json_object()
            .map(Some)
            .map_err(schema_error("response_format")),
        Some("json_schema") => {
            let schema = format
                .get("json_schema")
                .and_then(|schema| schema.get("schema"))
                .ok_or_else(|| {
                    invalid_value("response_format", "json_schema.schema is required")
                })?;
            compile_schema(schema)
                .map(Some)
                .map_err(schema_error("response_format"))
        }
        _ => Err(invalid_value(
            "response_format",
            "type must be text, json_object or json_schema",
        )),
    }
}

fn tool_function(tool: &Value) -> Result<(&str, Value), ChatRequestError> {
    let function = tool
        .get("function")
        .filter(|_| tool.get("type").and_then(Value::as_str) == Some("function"))
        .ok_or_else(|| invalid_value("tools", "each tool must be a function"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| invalid_value("tools", "each function needs a name"))?;
    let parameters = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
    Ok((name, parameters))
}

fn literal(text: &str) -> Vec<GrammarElement> {
    text.bytes()
        .map(|byte| {
            GrammarElement::Bytes(riley_runtime::grammar::ByteClass::byte(byte))
        })
        .collect()
}

/// The grammar of one or more forced calls of `only`, or of any offered tool.
fn forced_tool_grammar(
    tools: &[Value],
    only: Option<&str>,
    style: ToolStyle,
    parallel: bool,
) -> Result<Arc<Grammar>, ChatRequestError> {
    let (open, close, arguments_key) = match style {
        ToolStyle::Hermes => ("<tool_call>\n", "\n</tool_call>", "arguments"),
        ToolStyle::LlamaJson => ("", "", "parameters"),
    };
    let mut compiler = SchemaCompiler::new(Value::Null);
    let call = compiler.rule();
    let mut matched = false;
    for tool in tools {
        let (name, parameters) = tool_function(tool)?;
        if only.is_some_and(|only| only != name) {
            continue;
        }
        matched = true;
        compiler.set_root(parameters.clone());
        let arguments = compiler
            .schema(&parameters)
            .map_err(schema_error("tools"))?;
        let head = json!(name).to_string();
        let mut elements = literal(&format!(
            "{open}{{\"name\": {head}, \"{arguments_key}\": "
        ));
        elements.push(GrammarElement::Rule(arguments));
        elements.extend(literal(&format!("}}{close}")));
        compiler.alternative(call, elements);
    }
    if !matched {
        return Err(invalid_value(
            "tool_choice",
            "tool_choice names a function that is not in tools",
        ));
    }
    let start = compiler.rule();
    compiler.alternative(start, vec![GrammarElement::Rule(call)]);
    if parallel {
        let separator = if style == ToolStyle::Hermes { "\n" } else { "; " };
        let mut elements = vec![GrammarElement::Rule(call)];
        elements.extend(literal(separator));
        elements.push(GrammarElement::Rule(start));
        compiler.alternative(start, elements);
    }
    compiler.finish(start).map_err(schema_error("tools"))
}

fn tool_grammar(
    request: &ChatCompletionRequest,
    style: ToolStyle,
) -> Result<Option<Arc<Grammar>>, ChatRequestError> {
    let Some(tools) = offered_tools(request) else {
        return Ok(None);
    };
    for tool in tools {
        tool_function(tool)?;
    }
    // A forced grammar that may repeat a call lets a greedy model repeat it
    // until `max_tokens`, so repetition takes an explicit request.
    let parallel = request.parallel_tool_calls.unwrap_or(false);
    match &request.tool_choice {
        None => Ok(None),
        Some(Value::String(choice)) => match choice.as_str() {
            "auto" => Ok(None),
            "required" => forced_tool_grammar(tools, None, style, parallel).map(Some),
            _ => Err(invalid_value(
                "tool_choice",
                "tool_choice must be none, auto, required or a function",
            )),
        },
        Some(choice) => {
            let name = choice
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid_value("tool_choice", "tool_choice must name a function")
                })?;
            forced_tool_grammar(tools, Some(name), style, false).map(Some)
        }
    }
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
    if ["functions", "function_call"]
        .iter()
        .any(|field| request.unsupported_fields.contains_key(*field))
    {
        return Err(invalid("functions", "use tools instead of functions"));
    }
    if request.top_logprobs.is_some_and(|top| top > MAX_TOP_LOGPROBS) {
        return Err(invalid_value(
            "top_logprobs",
            "top_logprobs must be from 0 to 20",
        ));
    }
    if request.top_logprobs.is_some() && !request.logprobs.unwrap_or(false) {
        return Err(invalid_value("top_logprobs", "top_logprobs requires logprobs"));
    }
    let tools = offered_tools(&request).map(<[Value]>::to_vec);
    if tools.is_some() && template.tool_style.is_none() {
        return Err(invalid(
            "tools",
            "this model's chat template does not support tools",
        ));
    }
    let tool_grammar = match (template.tool_style, tools.is_some()) {
        (Some(style), true) => tool_grammar(&request, style)?,
        _ => None,
    };
    let format_grammar = request
        .response_format
        .as_ref()
        .map(response_format_grammar)
        .transpose()?
        .flatten();
    if tool_grammar.is_some() && format_grammar.is_some() {
        return Err(invalid(
            "response_format",
            "response_format cannot be combined with a forced tool_choice",
        ));
    }
    let logprobs_top = request
        .logprobs
        .unwrap_or(false)
        .then(|| request.top_logprobs.unwrap_or(0));
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
            tools.as_deref(),
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
        user: None,
        suffix: None,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        unsupported_fields: request.unsupported_fields,
    };
    let mut generation =
        normalize_completion_request(completion, limits).map_err(ChatRequestError::Invalid)?;
    // The template already wrote any begin-of-text token.
    generation.add_special_tokens = false;
    generation.grammar = tool_grammar.or(format_grammar);
    generation.logprobs_top = logprobs_top;
    generation.include_token_ids |= logprobs_top.is_some();
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
            .all(|field| matches!(field.as_str(), "name" | "tool_calls" | "tool_call_id"))
        {
            return Err(invalid(
                "only role, content, name and tool fields are supported in messages",
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
        if let Some(id) = message.unsupported_fields.get("tool_call_id") {
            rendered["tool_call_id"] = id.clone();
        }
        if let Some(calls) = message.unsupported_fields.get("tool_calls") {
            rendered["tool_calls"] = template_tool_calls(calls);
        }
        messages.push(rendered);
    }
    Ok(messages)
}

/// Clients send `arguments` as a JSON string; templates expect the object.
fn template_tool_calls(calls: &Value) -> Value {
    let mut calls = calls.clone();
    for call in calls.as_array_mut().into_iter().flatten() {
        let arguments = call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            .and_then(|text| serde_json::from_str::<Value>(text).ok());
        if let Some(arguments) = arguments {
            call["function"]["arguments"] = arguments;
        }
    }
    calls
}

/// One parsed call as `(name, arguments as JSON text)`.
type ToolCall = (String, String);

fn tool_call_from_json(value: &Value) -> Option<ToolCall> {
    let name = value.get("name")?.as_str()?.to_owned();
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match arguments {
        Value::String(text) => text,
        other => other.to_string(),
    };
    Some((name, arguments))
}

/// Splits output into leading content and tool calls. `None` means the text
/// is not a well-formed call, so the caller keeps it as ordinary content.
fn parse_tool_calls(style: ToolStyle, text: &str) -> Option<(String, Vec<ToolCall>)> {
    match style {
        ToolStyle::Hermes => {
            let first = text.find(HERMES_OPEN)?;
            let mut calls = Vec::new();
            let mut rest = &text[first..];
            while let Some(start) = rest.find(HERMES_OPEN) {
                let body = &rest[start + HERMES_OPEN.len()..];
                let Some((body, after)) = body.split_once(HERMES_CLOSE) else {
                    // A call cut off by `max_tokens` is not a call, but the
                    // complete ones before it are.
                    if calls.is_empty() {
                        calls.push(tool_call_from_json(
                            &serde_json::from_str(body.trim()).ok()?,
                        )?);
                    }
                    break;
                };
                calls.push(tool_call_from_json(
                    &serde_json::from_str(body.trim()).ok()?,
                )?);
                rest = after;
            }
            Some((text[..first].trim_end().to_owned(), calls))
        }
        ToolStyle::LlamaJson => {
            let body = text.trim();
            let body = body.strip_prefix(PYTHON_TAG).unwrap_or(body);
            let mut calls = Vec::new();
            for part in body.split("; ") {
                calls.push(tool_call_from_json(
                    &serde_json::from_str(part.trim()).ok()?,
                )?);
            }
            Some((String::new(), calls))
        }
    }
}

fn tool_call_values(calls: &[ToolCall], id_base: &str, streaming: bool) -> Value {
    calls
        .iter()
        .enumerate()
        .map(|(index, (name, arguments))| {
            let mut call = json!({
                "id": format!("call_{id_base}_{index}"),
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            });
            if streaming {
                call["index"] = json!(index);
            }
            call
        })
        .collect()
}

/// Re-shapes a `text_completion` response body into a `chat.completion`.
pub fn chat_response(response: &mut Value, tool_style: Option<ToolStyle>) {
    response["object"] = Value::from(CHAT_OBJECT);
    let id_base = response["id"].as_str().unwrap_or_default().to_owned();
    if let Some(choices) = response["choices"].as_array_mut() {
        for choice in choices {
            if let Some(choice) = choice.as_object_mut() {
                let text = choice.remove("text").unwrap_or_else(|| Value::from(""));
                let parsed = tool_style
                    .zip(text.as_str())
                    .and_then(|(style, text)| parse_tool_calls(style, text));
                let message = match parsed {
                    Some((content, calls)) => {
                        choice.insert("finish_reason".to_owned(), Value::from("tool_calls"));
                        json!({
                            "role": "assistant",
                            "content": if content.is_empty() { Value::Null } else { Value::from(content) },
                            "tool_calls": tool_call_values(&calls, &id_base, false),
                        })
                    }
                    None => json!({"role": "assistant", "content": text}),
                };
                choice.insert("message".to_owned(), message);
            }
        }
    }
}

/// Holds back a tool call while the text before it streams as content.
struct ToolStreamFilter {
    style: ToolStyle,
    held: String,
    in_call: bool,
    content_seen: bool,
}

impl ToolStreamFilter {
    /// Takes new text and returns what is safe to show as content.
    fn push(&mut self, text: &str) -> String {
        self.held.push_str(text);
        if self.in_call {
            return String::new();
        }
        match self.style {
            ToolStyle::Hermes => {
                if let Some(start) = self.held.find(HERMES_OPEN) {
                    self.in_call = true;
                    let shown: String = self.held.drain(..start).collect();
                    self.content_seen |= !shown.is_empty();
                    return shown;
                }
                let keep = (1..HERMES_OPEN.len())
                    .rev()
                    .find(|&length| {
                        self.held.is_char_boundary(self.held.len().wrapping_sub(length))
                            && self.held.ends_with(&HERMES_OPEN[..length])
                    })
                    .unwrap_or(0);
                let shown: String = self.held.drain(..self.held.len() - keep).collect();
                self.content_seen |= !shown.is_empty();
                shown
            }
            ToolStyle::LlamaJson => {
                let start = self.held.trim_start();
                if self.content_seen {
                    std::mem::take(&mut self.held)
                } else if start.is_empty()
                    || PYTHON_TAG.starts_with(start)
                    || start.starts_with(PYTHON_TAG)
                    || start.starts_with('{')
                {
                    self.in_call = !start.is_empty() && !PYTHON_TAG.starts_with(start);
                    String::new()
                } else {
                    self.content_seen = true;
                    std::mem::take(&mut self.held)
                }
            }
        }
    }

    /// Resolves the held text at the end of the stream.
    fn finish(&mut self) -> Result<Vec<ToolCall>, String> {
        let held = std::mem::take(&mut self.held);
        match parse_tool_calls(self.style, &held) {
            Some((_, calls)) if self.in_call => Ok(calls),
            _ => Err(held),
        }
    }
}

/// Re-shapes `text_completion` SSE frames into `chat.completion.chunk` ones.
#[derive(Default)]
pub struct ChatChunkShaper {
    role_sent: bool,
    tools: Option<ToolStreamFilter>,
}

impl ChatChunkShaper {
    /// Starts a shaper that parses tool calls in `style`, if given.
    #[must_use]
    pub fn new(style: Option<ToolStyle>) -> Self {
        Self {
            role_sent: false,
            tools: style.map(|style| ToolStreamFilter {
                style,
                held: String::new(),
                in_call: false,
                content_seen: false,
            }),
        }
    }

    /// Returns the chat form of one or more SSE frames; `[DONE]` and errors
    /// pass as is.
    #[must_use]
    pub fn shape(&mut self, frames: &str) -> String {
        frames
            .split_inclusive("\n\n")
            .map(|frame| self.shape_frame(frame))
            .collect()
    }

    fn shape_frame(&mut self, frame: &str) -> String {
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
        let id_base = chunk["id"].as_str().unwrap_or_default().to_owned();
        if let Some(choices) = chunk["choices"].as_array_mut() {
            for choice in choices {
                let Some(choice) = choice.as_object_mut() else {
                    continue;
                };
                let mut text = choice.remove("text").unwrap_or_else(|| Value::from(""));
                let finishing = choice
                    .get("finish_reason")
                    .is_some_and(|reason| !reason.is_null());
                let mut delta = Map::new();
                if !self.role_sent {
                    delta.insert("role".to_owned(), Value::from("assistant"));
                    self.role_sent = true;
                }
                if let Some(filter) = self.tools.as_mut() {
                    text = Value::from(filter.push(text.as_str().unwrap_or_default()));
                    if finishing {
                        match filter.finish() {
                            Ok(calls) => {
                                delta.insert(
                                    "tool_calls".to_owned(),
                                    tool_call_values(&calls, &id_base, true),
                                );
                                choice.insert(
                                    "finish_reason".to_owned(),
                                    Value::from("tool_calls"),
                                );
                            }
                            Err(held) => text = Value::from(held),
                        }
                    }
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
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "functions": []}),
            json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}),
            json!({"model": "m", "messages": [{"role": "user", "content": "x", "wat": []}]}),
            json!({"model": "m", "messages": []}),
            json!({"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 1000}),
        ] {
            assert!(matches!(
                normalize_chat_request(request(body), &template(CHATML), limits()),
                Err(ChatRequestError::Invalid(_))
            ));
        }
    }

    const TOOLS: &str = "{% if tools %}{{ tools | tojson }}<tool_call>{% endif %}\
{% for message in messages %}{{ message.role }}:{{ message.content }}\
{% if message.tool_calls %}{{ message.tool_calls[0].function.arguments.city }}{% endif %}\n{% endfor %}";

    fn weather() -> Value {
        json!([{"type": "function", "function": {"name": "weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])
    }

    #[test]
    fn tools_reach_the_template_and_tool_turns_round_trip() {
        let generation = normalize_chat_request(
            request(json!({"model": "m", "tools": weather(), "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{"id": "c", "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\": \"Seoul\"}"}}]},
                {"role": "tool", "content": "sunny", "tool_call_id": "c"}]})),
            &template(TOOLS),
            limits(),
        )
        .unwrap();
        assert!(generation.prompt.contains("\"weather\""), "{}", generation.prompt);
        assert!(generation.prompt.contains("assistant:Seoul"), "{}", generation.prompt);
        assert!(generation.prompt.contains("tool:sunny"));
        assert!(generation.grammar.is_none());
    }

    #[test]
    fn tool_choice_none_hides_tools_and_unsupported_templates_refuse_them() {
        let body = json!({"model": "m", "tools": weather(), "tool_choice": "none",
            "messages": [{"role": "user", "content": "x"}]});
        let parsed = request(body.clone());
        let tools_template = template(TOOLS);
        assert_eq!(tools_template.tool_parsing(&parsed), None);
        let generation = normalize_chat_request(parsed, &tools_template, limits()).unwrap();
        assert!(!generation.prompt.contains("weather"));

        let mut offered = body;
        offered["tool_choice"] = json!("auto");
        assert!(matches!(
            normalize_chat_request(request(offered), &template(CHATML), limits()),
            Err(ChatRequestError::Invalid(_))
        ));
    }

    #[test]
    fn forced_tool_choice_compiles_a_grammar_and_rejects_unknown_functions() {
        use riley_runtime::grammar::GrammarAutomaton;
        let body = json!({"model": "m", "tools": weather(), "tool_choice": "required",
            "messages": [{"role": "user", "content": "x"}]});
        let generation =
            normalize_chat_request(request(body.clone()), &template(TOOLS), limits()).unwrap();
        let mut automaton = GrammarAutomaton::new(generation.grammar.unwrap());
        let mut cursor = automaton.start();
        let text = "<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\":\"Seoul\"}}\n</tool_call>";
        assert!(automaton.advance(&mut cursor, text.as_bytes()));
        assert!(automaton.is_accepting(&mut cursor));

        let mut named = body;
        named["tool_choice"] = json!({"type": "function", "function": {"name": "nope"}});
        assert!(matches!(
            normalize_chat_request(request(named), &template(TOOLS), limits()),
            Err(ChatRequestError::Invalid(_))
        ));
    }

    #[test]
    fn unenforceable_schemas_and_bad_formats_are_refused() {
        let body = |format: Value| {
            request(json!({"model": "m", "response_format": format,
                "messages": [{"role": "user", "content": "x"}]}))
        };
        assert!(matches!(
            normalize_chat_request(
                body(json!({"type": "json_schema", "json_schema": {"name": "n",
                    "schema": {"type": "array", "uniqueItems": true}}})),
                &template(CHATML),
                limits()
            ),
            Err(ChatRequestError::Schema { parameter: "response_format", .. })
        ));
        assert!(matches!(
            normalize_chat_request(body(json!({"type": "xml"})), &template(CHATML), limits()),
            Err(ChatRequestError::Invalid(_))
        ));
        let text = normalize_chat_request(body(json!({"type": "text"})), &template(CHATML), limits());
        assert!(text.unwrap().grammar.is_none());
        let object =
            normalize_chat_request(body(json!({"type": "json_object"})), &template(CHATML), limits());
        assert!(object.unwrap().grammar.is_some());
    }

    #[test]
    fn logprobs_options_are_checked_and_ask_for_token_delivery() {
        let body = |extra: Value| {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "x"}]});
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            request(body)
        };
        let generation = normalize_chat_request(
            body(json!({"logprobs": true, "top_logprobs": 3})),
            &template(CHATML),
            limits(),
        )
        .unwrap();
        assert_eq!(generation.logprobs_top, Some(3));
        assert!(generation.include_token_ids && !generation.expose_token_ids);
        for extra in [json!({"top_logprobs": 3}), json!({"logprobs": true, "top_logprobs": 21})] {
            assert!(matches!(
                normalize_chat_request(body(extra), &template(CHATML), limits()),
                Err(ChatRequestError::Invalid(_))
            ));
        }
    }

    #[test]
    fn hermes_and_llama_tool_calls_parse_and_bad_ones_stay_content() {
        let text = "let me check\n<tool_call>\n{\"name\": \"a\", \"arguments\": {\"x\": 1}}\n</tool_call>\n<tool_call>\n{\"name\": \"b\", \"arguments\": {}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(ToolStyle::Hermes, text).unwrap();
        assert_eq!(content, "let me check");
        assert_eq!(calls[0], ("a".to_owned(), "{\"x\":1}".to_owned()));
        assert_eq!(calls[1].0, "b");
        let cut = format!("{text}\n<tool_call>\n{{\"name\": \"c");
        assert_eq!(parse_tool_calls(ToolStyle::Hermes, &cut).unwrap().1.len(), 2);
        assert!(parse_tool_calls(ToolStyle::Hermes, "plain").is_none());
        assert!(parse_tool_calls(ToolStyle::Hermes, "<tool_call>{oops</tool_call>").is_none());

        let (_, calls) = parse_tool_calls(
            ToolStyle::LlamaJson,
            "<|python_tag|>{\"name\": \"a\", \"parameters\": {\"x\": 1}}",
        )
        .unwrap();
        assert_eq!(calls, [("a".to_owned(), "{\"x\":1}".to_owned())]);
        assert!(parse_tool_calls(ToolStyle::LlamaJson, "The answer is 4.").is_none());
    }

    fn frame(text: &str, finish: Option<&str>) -> String {
        format!(
            "data: {}\n\n",
            json!({"id": "cmpl-1", "object": "text_completion", "choices": [
                {"text": text, "index": 0, "logprobs": null, "finish_reason": finish}]})
        )
    }

    fn payloads(frames: &str) -> Vec<Value> {
        frames
            .split("\n\n")
            .filter_map(|frame| frame.strip_prefix("data: "))
            .map(|payload| serde_json::from_str(payload).unwrap())
            .collect()
    }

    #[test]
    fn streamed_tool_calls_are_held_and_delivered_at_the_finish() {
        let mut shaper = ChatChunkShaper::new(Some(ToolStyle::Hermes));
        let mut content = String::new();
        for piece in ["ok ", "<tool", "_call>\n{\"name\": \"a\",", " \"arguments\": {}}\n</tool_call>"] {
            let chunk = &payloads(&shaper.shape(&frame(piece, None)))[0];
            content.push_str(chunk["choices"][0]["delta"]["content"].as_str().unwrap_or_default());
        }
        assert_eq!(content, "ok ");
        let last = &payloads(&shaper.shape(&frame("", Some("stop"))))[0];
        assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
        let call = &last["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "a");
        assert_eq!(call["index"], 0);
    }

    #[test]
    fn streamed_text_that_only_resembles_a_call_is_released() {
        let mut shaper = ChatChunkShaper::new(Some(ToolStyle::Hermes));
        let mut content = String::new();
        for piece in ["a <tool", " b"] {
            let chunk = &payloads(&shaper.shape(&frame(piece, None)))[0];
            content.push_str(chunk["choices"][0]["delta"]["content"].as_str().unwrap_or_default());
        }
        assert_eq!(content, "a <tool b");

        let mut shaper = ChatChunkShaper::new(Some(ToolStyle::Hermes));
        let _ = shaper.shape(&frame("<tool_call>broken", None));
        let last = &payloads(&shaper.shape(&frame("", Some("length"))))[0];
        assert_eq!(last["choices"][0]["finish_reason"], "length");
        assert_eq!(last["choices"][0]["delta"]["content"], "<tool_call>broken");
    }

    #[test]
    fn a_usage_frame_joined_to_the_finish_frame_keeps_the_chat_shape() {
        let mut shaper = ChatChunkShaper::default();
        let joined = frame("", Some("stop")) + "data: {\"object\":\"text_completion\",\"choices\":[],\"usage\":{}}\n\n";
        let shaped = payloads(&shaper.shape(&joined));
        assert_eq!(shaped.len(), 2);
        assert!(shaped.iter().all(|chunk| chunk["object"] == CHAT_CHUNK_OBJECT));
    }

    #[test]
    fn responses_and_chunks_take_the_chat_shape() {
        let mut response = json!({"id": "x", "object": "text_completion", "choices": [
            {"text": "Paris", "index": 0, "logprobs": null, "finish_reason": "stop"}]});
        chat_response(&mut response, None);
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
