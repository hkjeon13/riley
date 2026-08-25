//! OpenAI-compatible `/v1/completions` data-transfer objects.
//!
//! This module owns compatibility names, JSON serialization, request
//! normalization, and SSE framing.  It performs no network I/O and never
//! passes an HTTP DTO into the scheduler or runtime.

use std::collections::BTreeMap;
use std::error;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::{
    FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestLimits,
    RequestMetadata, SamplingParameters, ServiceErrorClass, TokenUsage,
};

/// OpenAI-compatible default for `max_tokens`.
pub const DEFAULT_MAX_TOKENS: u64 = 16;
/// OpenAI-compatible default sampling temperature.
pub const DEFAULT_TEMPERATURE: f32 = 1.0;
/// OpenAI-compatible default nucleus probability.
pub const DEFAULT_TOP_P: f32 = 1.0;
/// SSE frame terminating a successfully or unsuccessfully closed stream.
pub const SSE_DONE_FRAME: &str = "data: [DONE]\n\n";

const COMPLETION_OBJECT: &str = "text_completion";
const MODEL_OBJECT: &str = "model";
const MODEL_LIST_OBJECT: &str = "list";

/// Prompt shapes understood by the compatibility decoder.
///
/// Only [`Self::Text`] passes normalization. Other variants are represented
/// so clients receive an explicit validation error instead of an opaque JSON
/// decoding failure.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PromptInput {
    /// The supported single UTF-8 prompt.
    Text(String),
    /// One or more text prompts; unsupported by the initial endpoint.
    Texts(Vec<String>),
    /// A single token-ID prompt; unsupported by the initial endpoint.
    TokenIds(Vec<u32>),
    /// One or more token-ID prompts; unsupported by the initial endpoint.
    TokenBatches(Vec<Vec<u32>>),
}

/// `OpenAI` `stop` accepts either one string or an array of strings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum StopInput {
    /// One stop string.
    One(String),
    /// Ordered stop strings.
    Many(Vec<String>),
}

/// JSON request accepted by `POST /v1/completions`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct CompletionRequest {
    /// Requested model identifier.
    pub model: String,
    /// Prompt compatibility shape. Normalization accepts only one string.
    pub prompt: PromptInput,
    /// Maximum number of generated tokens.
    #[serde(default)]
    pub max_tokens: Option<u64>,
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
    /// The initial endpoint supports exactly one completion.
    #[serde(default)]
    pub n: Option<u64>,
    /// The initial endpoint supports exactly one candidate.
    #[serde(default)]
    pub best_of: Option<u64>,
    /// Prompt echoing is unsupported; false is accepted as a neutral value.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Log probabilities are not implemented.
    #[serde(default)]
    pub logprobs: Option<Value>,
    /// Suffix insertion is not implemented.
    #[serde(default)]
    pub suffix: Option<String>,
    /// Only the neutral value `0` is accepted.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Only the neutral value `0` is accepted.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Unknown compatibility fields are retained for an explicit rejection.
    #[serde(flatten)]
    pub unsupported_fields: BTreeMap<String, Value>,
}

/// Converts an OpenAI-compatible DTO into a transport-independent request.
///
/// # Errors
///
/// Returns a stable [`ValidationError`] for unsupported prompt shapes or
/// options, non-finite/out-of-range sampling values, empty or oversized
/// identifiers, invalid token counts, or stop strings outside `limits`.
pub fn normalize_completion_request(
    request: CompletionRequest,
    limits: RequestLimits,
) -> Result<GenerationRequest, ValidationError> {
    validate_model(&request.model, limits.max_model_bytes)?;
    validate_supported_options(&request)?;

    let prompt = match request.prompt {
        PromptInput::Text(prompt) => prompt,
        PromptInput::Texts(_) | PromptInput::TokenIds(_) | PromptInput::TokenBatches(_) => {
            return Err(ValidationError::new(
                "prompt",
                ValidationErrorCode::UnsupportedParameter,
                "only a single string prompt is supported",
            ));
        }
    };
    if prompt.len() > limits.max_prompt_bytes {
        return Err(ValidationError::new(
            "prompt",
            ValidationErrorCode::TooLarge,
            "prompt exceeds the server byte limit",
        ));
    }

    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens == 0 {
        return Err(ValidationError::new(
            "max_tokens",
            ValidationErrorCode::InvalidValue,
            "max_tokens must be at least one",
        ));
    }
    if max_tokens > usize_as_u64_saturating(limits.max_output_tokens) {
        return Err(ValidationError::new(
            "max_tokens",
            ValidationErrorCode::TooLarge,
            "max_tokens exceeds the server limit",
        ));
    }
    let max_new_tokens = usize::try_from(max_tokens).map_err(|_| {
        ValidationError::new(
            "max_tokens",
            ValidationErrorCode::TooLarge,
            "max_tokens exceeds the server limit",
        )
    })?;

    let temperature = request.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ValidationError::new(
            "temperature",
            ValidationErrorCode::InvalidValue,
            "temperature must be finite and between 0 and 2",
        ));
    }
    let top_p = request.top_p.unwrap_or(DEFAULT_TOP_P);
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) || top_p == 0.0 {
        return Err(ValidationError::new(
            "top_p",
            ValidationErrorCode::InvalidValue,
            "top_p must be finite and greater than 0 and at most 1",
        ));
    }

    let stop_sequences = normalize_stops(request.stop, limits)?;
    Ok(GenerationRequest {
        model_id: request.model,
        prompt,
        max_new_tokens,
        sampling: SamplingParameters {
            temperature,
            top_p,
            seed: request.seed,
        },
        stop_sequences,
        stream: request.stream.unwrap_or(false),
    })
}

fn validate_supported_options(request: &CompletionRequest) -> Result<(), ValidationError> {
    if request.logprobs.is_some() {
        return Err(ValidationError::new(
            "logprobs",
            ValidationErrorCode::UnsupportedParameter,
            "logprobs is not supported",
        ));
    }
    if request.suffix.is_some() {
        return Err(ValidationError::new(
            "suffix",
            ValidationErrorCode::UnsupportedParameter,
            "suffix is not supported",
        ));
    }
    validate_neutral_count("n", request.n)?;
    validate_neutral_count("best_of", request.best_of)?;
    if request.echo.unwrap_or(false) {
        return Err(ValidationError::new(
            "echo",
            ValidationErrorCode::UnsupportedParameter,
            "echo must be false",
        ));
    }
    validate_neutral_penalty("presence_penalty", request.presence_penalty)?;
    validate_neutral_penalty("frequency_penalty", request.frequency_penalty)?;
    if !request.unsupported_fields.is_empty() {
        return Err(ValidationError::new(
            "request",
            ValidationErrorCode::UnsupportedParameter,
            "the request contains an unsupported parameter",
        ));
    }

    Ok(())
}

fn usize_as_u64_saturating(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn validate_model(model: &str, maximum_bytes: usize) -> Result<(), ValidationError> {
    if model.trim().is_empty() {
        return Err(ValidationError::new(
            "model",
            ValidationErrorCode::InvalidValue,
            "model must not be empty",
        ));
    }
    if model.len() > maximum_bytes {
        return Err(ValidationError::new(
            "model",
            ValidationErrorCode::TooLarge,
            "model exceeds the server byte limit",
        ));
    }
    Ok(())
}

fn validate_neutral_count(
    parameter: &'static str,
    value: Option<u64>,
) -> Result<(), ValidationError> {
    if value.is_none_or(|value| value == 1) {
        return Ok(());
    }
    Err(ValidationError::new(
        parameter,
        ValidationErrorCode::UnsupportedParameter,
        "only the neutral value 1 is supported",
    ))
}

fn validate_neutral_penalty(
    parameter: &'static str,
    value: Option<f32>,
) -> Result<(), ValidationError> {
    let Some(value) = value else {
        return Ok(());
    };
    if !value.is_finite() {
        return Err(ValidationError::new(
            parameter,
            ValidationErrorCode::InvalidValue,
            "penalty must be finite",
        ));
    }
    if value != 0.0 {
        return Err(ValidationError::new(
            parameter,
            ValidationErrorCode::UnsupportedParameter,
            "only the neutral penalty value 0 is supported",
        ));
    }
    Ok(())
}

fn normalize_stops(
    stop: Option<StopInput>,
    limits: RequestLimits,
) -> Result<Vec<String>, ValidationError> {
    let stops = match stop {
        None => Vec::new(),
        Some(StopInput::One(stop)) => vec![stop],
        Some(StopInput::Many(stops)) => stops,
    };
    if stops.len() > limits.max_stop_sequences {
        return Err(ValidationError::new(
            "stop",
            ValidationErrorCode::TooMany,
            "stop contains too many strings",
        ));
    }

    let mut total_bytes = 0usize;
    for stop in &stops {
        if stop.is_empty() {
            return Err(ValidationError::new(
                "stop",
                ValidationErrorCode::InvalidValue,
                "stop strings must not be empty",
            ));
        }
        if stop.len() > limits.max_stop_sequence_bytes {
            return Err(ValidationError::new(
                "stop",
                ValidationErrorCode::TooLarge,
                "a stop string exceeds the server byte limit",
            ));
        }
        total_bytes = total_bytes.checked_add(stop.len()).ok_or_else(|| {
            ValidationError::new(
                "stop",
                ValidationErrorCode::TooLarge,
                "stop strings exceed the aggregate server byte limit",
            )
        })?;
        if total_bytes > limits.max_total_stop_bytes {
            return Err(ValidationError::new(
                "stop",
                ValidationErrorCode::TooLarge,
                "stop strings exceed the aggregate server byte limit",
            ));
        }
    }
    Ok(stops)
}

/// Stable request-validation error class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ValidationErrorCode {
    /// A supported parameter has an invalid value.
    InvalidValue,
    /// The initial endpoint does not implement the requested behavior.
    UnsupportedParameter,
    /// A byte or numeric bound was exceeded.
    TooLarge,
    /// A collection count was exceeded.
    TooMany,
}

impl ValidationErrorCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidValue => "invalid_value",
            Self::UnsupportedParameter => "unsupported_parameter",
            Self::TooLarge => "request_too_large",
            Self::TooMany => "too_many_values",
        }
    }
}

/// Stable validation failure containing no model, CUDA, path, or pointer detail.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    parameter: &'static str,
    code: ValidationErrorCode,
    message: &'static str,
}

impl ValidationError {
    const fn new(
        parameter: &'static str,
        code: ValidationErrorCode,
        message: &'static str,
    ) -> Self {
        Self {
            parameter,
            code,
            message,
        }
    }

    /// Request parameter associated with the failure.
    #[must_use]
    pub const fn parameter(&self) -> &'static str {
        self.parameter
    }

    /// Machine-readable validation class.
    #[must_use]
    pub const fn code(&self) -> ValidationErrorCode {
        self.code
    }

    /// Stable, non-sensitive explanation suitable for a JSON response.
    #[must_use]
    pub const fn public_message(&self) -> &'static str {
        self.message
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid {}: {}", self.parameter, self.message)
    }
}

impl error::Error for ValidationError {}

/// API-level error with intentionally no field for sensitive internal detail.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ApiError {
    /// Request validation failed before admission.
    InvalidRequest(ValidationError),
    /// Backend validation rejected a request against loaded-model constraints.
    BackendInvalidRequest,
    /// The bounded admission queue is full.
    Overloaded,
    /// The request deadline elapsed.
    Timeout,
    /// The server is draining.
    ShuttingDown,
    /// The request was cancelled.
    Cancelled,
    /// A non-public internal operation failed.
    Internal,
}

impl ApiError {
    /// HTTP status selected for this public error class.
    #[must_use]
    pub const fn status_code(&self) -> u16 {
        match self {
            Self::InvalidRequest(_) | Self::BackendInvalidRequest => 400,
            Self::Overloaded => 429,
            Self::Timeout | Self::Cancelled => 408,
            Self::ShuttingDown => 503,
            Self::Internal => 500,
        }
    }

    /// Builds the sanitized OpenAI-compatible JSON error object.
    #[must_use]
    pub fn response(&self) -> ErrorResponse {
        let (message, kind, parameter, code) = match self {
            Self::InvalidRequest(source) => (
                source.public_message(),
                "invalid_request_error",
                Some(source.parameter()),
                source.code().as_str(),
            ),
            Self::BackendInvalidRequest => (
                "the request is incompatible with the loaded model",
                "invalid_request_error",
                None,
                "invalid_request",
            ),
            Self::Overloaded => (
                "the server is at capacity; retry later",
                "server_error",
                None,
                "overloaded",
            ),
            Self::Timeout => ("the request timed out", "server_error", None, "timeout"),
            Self::ShuttingDown => (
                "the server is shutting down",
                "server_error",
                None,
                "server_shutting_down",
            ),
            Self::Cancelled => (
                "the request was cancelled",
                "server_error",
                None,
                "request_cancelled",
            ),
            Self::Internal => (
                "the server encountered an internal error",
                "server_error",
                None,
                "internal_error",
            ),
        };
        ErrorResponse {
            error: ErrorObject {
                message: message.to_owned(),
                kind: kind.to_owned(),
                param: parameter.map(str::to_owned),
                code: code.to_owned(),
            },
        }
    }
}

impl From<ServiceErrorClass> for ApiError {
    fn from(value: ServiceErrorClass) -> Self {
        match value {
            ServiceErrorClass::InvalidRequest => Self::BackendInvalidRequest,
            ServiceErrorClass::Overloaded => Self::Overloaded,
            ServiceErrorClass::Timeout => Self::Timeout,
            ServiceErrorClass::ShuttingDown => Self::ShuttingDown,
            ServiceErrorClass::Cancelled => Self::Cancelled,
            ServiceErrorClass::Internal => Self::Internal,
        }
    }
}

/// OpenAI-compatible outer JSON error envelope.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ErrorResponse {
    /// Sanitized error payload.
    pub error: ErrorObject,
}

/// OpenAI-compatible JSON error payload.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ErrorObject {
    /// Stable public explanation.
    pub message: String,
    /// OpenAI-compatible error type.
    #[serde(rename = "type")]
    pub kind: String,
    /// Invalid parameter, when applicable.
    pub param: Option<String>,
    /// Stable machine-readable code.
    pub code: String,
}

/// Finish values exposed by the completions endpoint.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionFinishReason {
    /// EOS or a client stop condition matched.
    Stop,
    /// The maximum output length was reached.
    Length,
}

impl TryFrom<FinishReason> for CompletionFinishReason {
    type Error = UnsupportedFinishReason;

    fn try_from(value: FinishReason) -> Result<Self, Self::Error> {
        match value {
            FinishReason::Stop => Ok(Self::Stop),
            FinishReason::Length => Ok(Self::Length),
            FinishReason::Cancelled | FinishReason::Error => Err(UnsupportedFinishReason(value)),
        }
    }
}

/// A terminal domain reason that must be represented as an API error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnsupportedFinishReason(pub FinishReason);

impl fmt::Display for UnsupportedFinishReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("terminal failure cannot be encoded as a successful completion")
    }
}

impl error::Error for UnsupportedFinishReason {}

/// OpenAI-compatible token usage object.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CompletionUsage {
    /// Encoded prompt token count.
    pub prompt_tokens: u64,
    /// Generated token count.
    pub completion_tokens: u64,
    /// Sum of prompt and generated token counts.
    pub total_tokens: u64,
}

impl From<TokenUsage> for CompletionUsage {
    fn from(value: TokenUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens(),
            completion_tokens: value.completion_tokens(),
            total_tokens: value.total_tokens(),
        }
    }
}

/// One choice in a non-streaming completion response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletionChoice {
    /// Complete generated text.
    pub text: String,
    /// Always zero because `n=1` is the only supported value.
    pub index: u32,
    /// Always null because log probabilities are unsupported.
    pub logprobs: Option<Value>,
    /// Successful terminal reason.
    pub finish_reason: CompletionFinishReason,
}

/// Non-streaming `/v1/completions` response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletionResponse {
    /// Stable request identifier.
    pub id: String,
    /// `OpenAI` object discriminator (`text_completion`).
    pub object: String,
    /// Request admission time in Unix seconds.
    pub created: u64,
    /// Selected model identifier.
    pub model: String,
    /// Exactly one completion choice.
    pub choices: Vec<CompletionChoice>,
    /// Final token counts.
    pub usage: CompletionUsage,
}

impl CompletionResponse {
    /// Builds the single-choice response used by the initial endpoint.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedFinishReason`] when cancellation or execution
    /// failure must instead be returned through [`ErrorResponse`].
    pub fn new(
        metadata: &RequestMetadata,
        text: String,
        reason: FinishReason,
        usage: TokenUsage,
    ) -> Result<Self, UnsupportedFinishReason> {
        Ok(Self {
            id: metadata.request_id.clone(),
            object: COMPLETION_OBJECT.to_owned(),
            created: metadata.created_unix_seconds,
            model: metadata.model_id.clone(),
            choices: vec![CompletionChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: CompletionFinishReason::try_from(reason)?,
            }],
            usage: usage.into(),
        })
    }
}

/// One choice in a streaming completion chunk.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletionChunkChoice {
    /// Newly visible text, or an empty string in the finish chunk.
    pub text: String,
    /// Always zero because `n=1` is the only supported value.
    pub index: u32,
    /// Always null because log probabilities are unsupported.
    pub logprobs: Option<Value>,
    /// Null for deltas and present only in the terminal finish chunk.
    pub finish_reason: Option<CompletionFinishReason>,
}

/// JSON data carried by one SSE completion event.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CompletionChunk {
    /// Stable request identifier shared by every event.
    pub id: String,
    /// `OpenAI` object discriminator (`text_completion`).
    pub object: String,
    /// Request admission time in Unix seconds.
    pub created: u64,
    /// Selected model identifier.
    pub model: String,
    /// Exactly one delta or finish choice.
    pub choices: Vec<CompletionChunkChoice>,
}

impl CompletionChunk {
    fn delta(metadata: &RequestMetadata, text: String) -> Self {
        Self::new(metadata, text, None)
    }

    fn finish(metadata: &RequestMetadata, reason: CompletionFinishReason) -> Self {
        Self::new(metadata, String::new(), Some(reason))
    }

    fn new(
        metadata: &RequestMetadata,
        text: String,
        finish_reason: Option<CompletionFinishReason>,
    ) -> Self {
        Self {
            id: metadata.request_id.clone(),
            object: COMPLETION_OBJECT.to_owned(),
            created: metadata.created_unix_seconds,
            model: metadata.model_id.clone(),
            choices: vec![CompletionChunkChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason,
            }],
        }
    }
}

/// OpenAI-compatible model object.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelObject {
    /// Model identifier accepted by completions.
    pub id: String,
    /// `OpenAI` object discriminator (`model`).
    pub object: String,
    /// Loaded model revision time in Unix seconds.
    pub created: u64,
    /// Public owner label.
    pub owned_by: String,
}

impl From<&ModelMetadata> for ModelObject {
    fn from(metadata: &ModelMetadata) -> Self {
        Self {
            id: metadata.model_id.clone(),
            object: MODEL_OBJECT.to_owned(),
            created: metadata.created_unix_seconds,
            owned_by: metadata.owned_by.clone(),
        }
    }
}

/// OpenAI-compatible model-list response.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ModelListResponse {
    /// `OpenAI` object discriminator (`list`).
    pub object: String,
    /// Models available to this server.
    pub data: Vec<ModelObject>,
}

impl ModelListResponse {
    /// Builds the initial single-model response.
    #[must_use]
    pub fn single(metadata: &ModelMetadata) -> Self {
        Self {
            object: MODEL_LIST_OBJECT.to_owned(),
            data: vec![metadata.into()],
        }
    }
}

/// Deterministic serialization of one JSON value as an SSE `data` event.
///
/// JSON string escaping prevents text deltas from injecting SSE record
/// boundaries. Struct declaration order and compact `serde_json` encoding
/// make repeated serialization byte-for-byte stable.
///
/// # Errors
///
/// Propagates JSON serialization failures from `value`.
pub fn serialize_sse_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    let json = serde_json::to_string(value)?;
    Ok(format!("data: {json}\n\n"))
}

/// Returns the exact SSE terminal sentinel.
#[must_use]
pub const fn serialize_sse_done() -> &'static str {
    SSE_DONE_FRAME
}

/// Observable phase of [`SseStreamEncoder`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SseStreamPhase {
    /// Token deltas or one terminal event may still be emitted.
    Open,
    /// A successful finish chunk was emitted; only `[DONE]` remains valid.
    Finished,
    /// A sanitized error was emitted; only `[DONE]` remains valid.
    Failed,
    /// `[DONE]` was emitted and the encoder is closed.
    Done,
}

/// Stateful SSE encoder enforcing delta → finish → `[DONE]` ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SseStreamEncoder {
    metadata: RequestMetadata,
    phase: SseStreamPhase,
}

impl SseStreamEncoder {
    /// Starts an encoder in the open phase.
    #[must_use]
    pub const fn new(metadata: RequestMetadata) -> Self {
        Self {
            metadata,
            phase: SseStreamPhase::Open,
        }
    }

    /// Current ordering phase.
    #[must_use]
    pub const fn phase(&self) -> SseStreamPhase {
        self.phase
    }

    /// Encodes one internal generation event.
    ///
    /// # Errors
    ///
    /// Returns [`SseEncodingError`] for an invalid event order, a failed JSON
    /// serialization, or a non-success finish reason.
    pub fn encode_event(&mut self, event: &GenerationEvent) -> Result<String, SseEncodingError> {
        match event {
            GenerationEvent::TokenDelta { text } => self.encode_delta(text),
            GenerationEvent::Finished { reason, .. } => match reason {
                FinishReason::Stop | FinishReason::Length => self.encode_finish(*reason),
                FinishReason::Cancelled => self.encode_error(&ApiError::Cancelled),
                FinishReason::Error => self.encode_error(&ApiError::Internal),
            },
            GenerationEvent::Failed { class } => self.encode_error(&ApiError::from(*class)),
        }
    }

    /// Encodes a token text delta while the stream is open.
    ///
    /// # Errors
    ///
    /// Returns [`SseEncodingError`] unless the stream is open, or if JSON
    /// serialization fails.
    pub fn encode_delta(&mut self, text: &str) -> Result<String, SseEncodingError> {
        self.require_phase("token delta", SseStreamPhase::Open)?;
        serialize_sse_json(&CompletionChunk::delta(&self.metadata, text.to_owned()))
            .map_err(SseEncodingError::Json)
    }

    /// Encodes the successful finish chunk while the stream is open.
    ///
    /// # Errors
    ///
    /// Returns [`SseEncodingError`] for invalid ordering, cancellation or
    /// execution-error finish reasons, or JSON serialization failure.
    pub fn encode_finish(&mut self, reason: FinishReason) -> Result<String, SseEncodingError> {
        self.require_phase("finish", SseStreamPhase::Open)?;
        let reason = CompletionFinishReason::try_from(reason)
            .map_err(SseEncodingError::UnsupportedFinish)?;
        let frame = serialize_sse_json(&CompletionChunk::finish(&self.metadata, reason))
            .map_err(SseEncodingError::Json)?;
        self.phase = SseStreamPhase::Finished;
        Ok(frame)
    }

    /// Encodes a sanitized in-stream error while the stream is open.
    ///
    /// # Errors
    ///
    /// Returns [`SseEncodingError`] for invalid ordering or JSON
    /// serialization failure.
    pub fn encode_error(&mut self, error: &ApiError) -> Result<String, SseEncodingError> {
        self.require_phase("error", SseStreamPhase::Open)?;
        let frame = serialize_sse_json(&error.response()).map_err(SseEncodingError::Json)?;
        self.phase = SseStreamPhase::Failed;
        Ok(frame)
    }

    /// Closes a finished or failed stream with `[DONE]`.
    ///
    /// # Errors
    ///
    /// Returns [`SseEncodingError`] before a terminal event or after the
    /// sentinel was already emitted.
    pub fn encode_done(&mut self) -> Result<&'static str, SseEncodingError> {
        if !matches!(
            self.phase,
            SseStreamPhase::Finished | SseStreamPhase::Failed
        ) {
            return Err(SseEncodingError::InvalidTransition {
                operation: "done",
                phase: self.phase,
            });
        }
        self.phase = SseStreamPhase::Done;
        Ok(serialize_sse_done())
    }

    fn require_phase(
        &self,
        operation: &'static str,
        required: SseStreamPhase,
    ) -> Result<(), SseEncodingError> {
        if self.phase == required {
            Ok(())
        } else {
            Err(SseEncodingError::InvalidTransition {
                operation,
                phase: self.phase,
            })
        }
    }
}

/// Checked SSE serialization or ordering failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum SseEncodingError {
    /// The event does not follow the stream state machine.
    InvalidTransition {
        /// Attempted operation.
        operation: &'static str,
        /// Phase observed before the operation.
        phase: SseStreamPhase,
    },
    /// JSON serialization failed.
    Json(serde_json::Error),
    /// A cancellation or internal failure was passed as a success reason.
    UnsupportedFinish(UnsupportedFinishReason),
}

impl fmt::Display for SseEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTransition { operation, phase } => {
                write!(
                    formatter,
                    "cannot encode {operation} while stream is {phase:?}"
                )
            }
            Self::Json(source) => write!(formatter, "could not serialize SSE JSON: {source}"),
            Self::UnsupportedFinish(source) => source.fmt(formatter),
        }
    }
}

impl error::Error for SseEncodingError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Json(source) => Some(source),
            Self::UnsupportedFinish(source) => Some(source),
            Self::InvalidTransition { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{Value, json};

    use super::{
        ApiError, CompletionFinishReason, CompletionRequest, CompletionResponse, PromptInput,
        SSE_DONE_FRAME, SseEncodingError, SseStreamEncoder, SseStreamPhase, StopInput,
        ValidationErrorCode, normalize_completion_request,
    };
    use crate::domain::{
        FinishReason, GenerationEvent, RequestLimits, RequestMetadata, TokenUsage,
    };

    fn minimal_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "fixture-model",
            "prompt": "hello"
        }))
        .expect("minimal completion request must decode")
    }

    #[test]
    fn request_deserialization_accepts_stop_string_or_array() {
        let one: CompletionRequest = serde_json::from_value(json!({
            "model": "fixture-model",
            "prompt": "hello",
            "stop": "END"
        }))
        .expect("single stop must decode");
        assert_eq!(one.stop, Some(StopInput::One("END".to_owned())));

        let many: CompletionRequest = serde_json::from_value(json!({
            "model": "fixture-model",
            "prompt": "hello",
            "stop": ["END", "STOP"]
        }))
        .expect("stop array must decode");
        assert_eq!(
            many.stop,
            Some(StopInput::Many(vec!["END".to_owned(), "STOP".to_owned()]))
        );
    }

    #[test]
    fn normalization_applies_defaults_and_preserves_supported_values() {
        let mut request = minimal_request();
        request.max_tokens = Some(7);
        request.temperature = Some(0.25);
        request.top_p = Some(0.75);
        request.seed = Some(42);
        request.stop = Some(StopInput::Many(vec!["END".to_owned(), "STOP".to_owned()]));
        request.stream = Some(true);
        request.n = Some(1);
        request.best_of = Some(1);
        request.echo = Some(false);
        request.presence_penalty = Some(0.0);
        request.frequency_penalty = Some(-0.0);

        let normalized = normalize_completion_request(request, RequestLimits::default())
            .expect("supported request must normalize");
        assert_eq!(normalized.model_id, "fixture-model");
        assert_eq!(normalized.prompt, "hello");
        assert_eq!(normalized.max_new_tokens, 7);
        assert_eq!(
            normalized.sampling.temperature.to_bits(),
            0.25_f32.to_bits()
        );
        assert_eq!(normalized.sampling.top_p.to_bits(), 0.75_f32.to_bits());
        assert_eq!(normalized.sampling.seed, Some(42));
        assert_eq!(normalized.stop_sequences, ["END", "STOP"]);
        assert!(normalized.stream);

        let defaults = normalize_completion_request(minimal_request(), RequestLimits::default())
            .expect("defaults must normalize");
        assert_eq!(defaults.max_new_tokens, 16);
        assert_eq!(defaults.sampling.temperature.to_bits(), 1.0_f32.to_bits());
        assert_eq!(defaults.sampling.top_p.to_bits(), 1.0_f32.to_bits());
        assert!(!defaults.stream);
    }

    #[test]
    fn multi_or_token_prompt_is_an_explicit_validation_error() {
        for prompt in [
            PromptInput::Texts(vec!["a".to_owned(), "b".to_owned()]),
            PromptInput::TokenIds(vec![1, 2]),
            PromptInput::TokenBatches(vec![vec![1], vec![2]]),
        ] {
            let mut request = minimal_request();
            request.prompt = prompt;
            let error = normalize_completion_request(request, RequestLimits::default())
                .expect_err("unsupported prompt shape must fail");
            assert_eq!(error.parameter(), "prompt");
            assert_eq!(error.code(), ValidationErrorCode::UnsupportedParameter);
        }
    }

    #[test]
    fn non_neutral_and_unsupported_options_fail_explicitly() {
        type UnsupportedCase = (&'static str, fn(&mut CompletionRequest));
        let cases: [UnsupportedCase; 7] = [
            ("n", |request| request.n = Some(2)),
            ("best_of", |request| request.best_of = Some(2)),
            ("echo", |request| request.echo = Some(true)),
            ("presence_penalty", |request| {
                request.presence_penalty = Some(0.5);
            }),
            ("frequency_penalty", |request| {
                request.frequency_penalty = Some(-0.5);
            }),
            ("logprobs", |request| request.logprobs = Some(json!(1))),
            ("suffix", |request| request.suffix = Some("tail".to_owned())),
        ];
        for (parameter, mutate) in cases {
            let mut request = minimal_request();
            mutate(&mut request);
            let error = normalize_completion_request(request, RequestLimits::default())
                .expect_err("unsupported option must fail");
            assert_eq!(error.parameter(), parameter);
            assert_eq!(error.code(), ValidationErrorCode::UnsupportedParameter);
        }

        let mut request = minimal_request();
        request.unsupported_fields = BTreeMap::from([("user".to_owned(), json!("client"))]);
        let error = normalize_completion_request(request, RequestLimits::default())
            .expect_err("unknown field must fail explicitly");
        assert_eq!(error.parameter(), "request");
        assert_eq!(error.code(), ValidationErrorCode::UnsupportedParameter);
    }

    #[test]
    fn numeric_and_byte_bounds_are_enforced_before_admission() {
        let limits = RequestLimits {
            max_model_bytes: 5,
            max_prompt_bytes: 5,
            max_output_tokens: 8,
            max_stop_sequences: 2,
            max_stop_sequence_bytes: 3,
            max_total_stop_bytes: 4,
        };

        let mut request = minimal_request();
        request.model = "123456".to_owned();
        assert_eq!(
            normalize_completion_request(request, limits)
                .expect_err("oversized model")
                .parameter(),
            "model"
        );

        let mut request = minimal_request();
        request.model = "model".to_owned();
        request.prompt = PromptInput::Text("123456".to_owned());
        assert_eq!(
            normalize_completion_request(request, limits)
                .expect_err("oversized prompt")
                .parameter(),
            "prompt"
        );

        for max_tokens in [0, 9] {
            let mut request = minimal_request();
            request.model = "model".to_owned();
            request.max_tokens = Some(max_tokens);
            assert_eq!(
                normalize_completion_request(request, limits)
                    .expect_err("invalid max_tokens")
                    .parameter(),
                "max_tokens"
            );
        }

        let mut request = minimal_request();
        request.model = "model".to_owned();
        request.max_tokens = Some(1);
        request.stop = Some(StopInput::Many(vec![
            "a".to_owned(),
            "b".to_owned(),
            "c".to_owned(),
        ]));
        assert_eq!(
            normalize_completion_request(request, limits)
                .expect_err("too many stops")
                .code(),
            ValidationErrorCode::TooMany
        );

        let mut request = minimal_request();
        request.model = "model".to_owned();
        request.max_tokens = Some(1);
        request.stop = Some(StopInput::One("long".to_owned()));
        assert_eq!(
            normalize_completion_request(request, limits)
                .expect_err("oversized stop")
                .parameter(),
            "stop"
        );

        let mut request = minimal_request();
        request.model = "model".to_owned();
        request.max_tokens = Some(1);
        request.stop = Some(StopInput::Many(vec!["abc".to_owned(), "de".to_owned()]));
        assert_eq!(
            normalize_completion_request(request, limits)
                .expect_err("oversized aggregate stops")
                .parameter(),
            "stop"
        );
    }

    #[test]
    fn non_finite_and_out_of_range_floats_are_rejected() {
        for temperature in [f32::NAN, f32::INFINITY, -0.1, 2.1] {
            let mut request = minimal_request();
            request.temperature = Some(temperature);
            let error = normalize_completion_request(request, RequestLimits::default())
                .expect_err("invalid temperature must fail");
            assert_eq!(error.parameter(), "temperature");
            assert_eq!(error.code(), ValidationErrorCode::InvalidValue);
        }
        for top_p in [f32::NAN, f32::NEG_INFINITY, 0.0, 1.1] {
            let mut request = minimal_request();
            request.top_p = Some(top_p);
            let error = normalize_completion_request(request, RequestLimits::default())
                .expect_err("invalid top_p must fail");
            assert_eq!(error.parameter(), "top_p");
            assert_eq!(error.code(), ValidationErrorCode::InvalidValue);
        }
        let mut request = minimal_request();
        request.presence_penalty = Some(f32::INFINITY);
        let error = normalize_completion_request(request, RequestLimits::default())
            .expect_err("non-finite penalty must fail");
        assert_eq!(error.parameter(), "presence_penalty");
        assert_eq!(error.code(), ValidationErrorCode::InvalidValue);
    }

    #[test]
    fn empty_and_aggregate_stop_constraints_are_checked() {
        let mut request = minimal_request();
        request.stop = Some(StopInput::One(String::new()));
        let error = normalize_completion_request(request, RequestLimits::default())
            .expect_err("empty stop must fail");
        assert_eq!(error.parameter(), "stop");
        assert_eq!(error.code(), ValidationErrorCode::InvalidValue);
    }

    #[test]
    fn sanitized_errors_never_contain_internal_diagnostics() {
        let response = ApiError::Internal.response();
        let json = serde_json::to_string(&response).expect("error response must serialize");
        assert_eq!(response.error.code, "internal_error");
        assert!(!json.contains("/home/private/model.safetensors"));
        assert!(!json.contains("0xdeadbeef"));
        assert!(!json.contains("CUDA_ERROR"));

        let mut request = minimal_request();
        request.logprobs = Some(json!(5));
        let validation = normalize_completion_request(request, RequestLimits::default())
            .expect_err("logprobs must fail");
        let response = ApiError::InvalidRequest(validation).response();
        assert_eq!(response.error.param.as_deref(), Some("logprobs"));
        assert_eq!(response.error.code, "unsupported_parameter");
    }

    #[test]
    fn non_streaming_response_has_one_choice_and_checked_usage() {
        let metadata =
            RequestMetadata::new("cmpl-123", "fixture-model", 42).expect("valid response metadata");
        let usage = TokenUsage::new(3, 2).expect("usage must fit");
        let response =
            CompletionResponse::new(&metadata, "hello".to_owned(), FinishReason::Stop, usage)
                .expect("successful reason must encode");
        let value = serde_json::to_value(response).expect("response must serialize");
        assert_eq!(value["object"], "text_completion");
        assert_eq!(value["choices"][0]["text"], "hello");
        assert_eq!(value["choices"][0]["index"], 0);
        assert_eq!(value["choices"][0]["logprobs"], Value::Null);
        assert_eq!(value["choices"][0]["finish_reason"], "stop");
        assert_eq!(value["usage"]["total_tokens"], 5);

        assert!(
            CompletionResponse::new(&metadata, String::new(), FinishReason::Error, usage).is_err()
        );
    }

    #[test]
    fn sse_encoder_is_deterministic_and_enforces_terminal_order() {
        let metadata =
            RequestMetadata::new("cmpl-123", "fixture-model", 42).expect("valid response metadata");
        let mut encoder = SseStreamEncoder::new(metadata);
        assert!(matches!(
            encoder.encode_done(),
            Err(SseEncodingError::InvalidTransition {
                operation: "done",
                phase: SseStreamPhase::Open
            })
        ));

        let delta = encoder
            .encode_event(&GenerationEvent::TokenDelta {
                text: "line 1\nline 2".to_owned(),
            })
            .expect("delta must encode");
        assert_eq!(
            delta,
            "data: {\"id\":\"cmpl-123\",\"object\":\"text_completion\",\"created\":42,\"model\":\"fixture-model\",\"choices\":[{\"text\":\"line 1\\nline 2\",\"index\":0,\"logprobs\":null,\"finish_reason\":null}]}\n\n"
        );
        assert_eq!(encoder.phase(), SseStreamPhase::Open);

        let usage = TokenUsage::new(2, 1).expect("usage must fit");
        let finish = encoder
            .encode_event(&GenerationEvent::Finished {
                reason: FinishReason::Length,
                usage,
            })
            .expect("finish must encode");
        assert_eq!(
            finish,
            "data: {\"id\":\"cmpl-123\",\"object\":\"text_completion\",\"created\":42,\"model\":\"fixture-model\",\"choices\":[{\"text\":\"\",\"index\":0,\"logprobs\":null,\"finish_reason\":\"length\"}]}\n\n"
        );
        assert_eq!(encoder.phase(), SseStreamPhase::Finished);
        assert!(encoder.encode_delta("late").is_err());
        assert_eq!(
            encoder.encode_done().expect("done must follow finish"),
            SSE_DONE_FRAME
        );
        assert_eq!(encoder.phase(), SseStreamPhase::Done);
        assert!(encoder.encode_done().is_err());
    }

    #[test]
    fn in_stream_failure_is_sanitized_then_done() {
        let metadata =
            RequestMetadata::new("cmpl-123", "fixture-model", 42).expect("valid response metadata");
        let mut encoder = SseStreamEncoder::new(metadata);
        let frame = encoder
            .encode_error(&ApiError::Internal)
            .expect("open stream may encode an error");
        assert!(frame.contains("\"code\":\"internal_error\""));
        assert_eq!(encoder.phase(), SseStreamPhase::Failed);
        assert_eq!(
            encoder.encode_done().expect("done must follow error"),
            SSE_DONE_FRAME
        );
    }

    #[test]
    fn finish_reason_uses_openai_values() {
        assert_eq!(
            CompletionFinishReason::try_from(FinishReason::Stop),
            Ok(CompletionFinishReason::Stop)
        );
        assert_eq!(
            CompletionFinishReason::try_from(FinishReason::Length),
            Ok(CompletionFinishReason::Length)
        );
        assert!(CompletionFinishReason::try_from(FinishReason::Cancelled).is_err());
    }
}
