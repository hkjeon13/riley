//! Transport-independent request and generation event types.
//!
//! The HTTP layer normalizes compatibility DTOs into these types before a
//! request reaches the scheduler.  Keeping this module free of `OpenAI` wire
//! names prevents compatibility details from leaking into execution code.

use std::error;
use std::fmt;

/// Default upper bound for a public request identifier.
pub const MAX_REQUEST_ID_BYTES: usize = 128;
/// Default upper bound for a public model identifier.
pub const MAX_MODEL_ID_BYTES: usize = 256;
/// Default upper bound for the model owner label returned by metadata APIs.
pub const MAX_MODEL_OWNER_BYTES: usize = 128;

/// Limits applied while normalizing an HTTP request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestLimits {
    /// Maximum UTF-8 bytes in a model identifier.
    pub max_model_bytes: usize,
    /// Maximum UTF-8 bytes in the single accepted prompt.
    pub max_prompt_bytes: usize,
    /// Maximum requested output-token count.
    pub max_output_tokens: usize,
    /// Maximum number of stop strings.
    pub max_stop_sequences: usize,
    /// Maximum UTF-8 bytes in one stop string.
    pub max_stop_sequence_bytes: usize,
    /// Maximum aggregate UTF-8 bytes across all stop strings.
    pub max_total_stop_bytes: usize,
}

impl Default for RequestLimits {
    fn default() -> Self {
        Self {
            max_model_bytes: MAX_MODEL_ID_BYTES,
            max_prompt_bytes: 1_048_576,
            max_output_tokens: 65_536,
            max_stop_sequences: 16,
            max_stop_sequence_bytes: 1_024,
            max_total_stop_bytes: 4_096,
        }
    }
}

/// Sampling controls understood by the server's execution layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParameters {
    /// Zero selects greedy decoding; positive values scale logits.
    pub temperature: f32,
    /// Nucleus probability in `(0, 1]`.
    pub top_p: f32,
    /// Optional deterministic request seed.
    pub seed: Option<u64>,
}

/// A validated generation request independent of any HTTP compatibility DTO.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// Model identifier selected by the client.
    pub model_id: String,
    /// The single UTF-8 prompt accepted by the initial API.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling configuration.
    pub sampling: SamplingParameters,
    /// Ordered stop strings, excluded from visible output on a match.
    pub stop_sequences: Vec<String>,
    /// Whether the HTTP response should use SSE.
    pub stream: bool,
}

/// Stable metadata shared by streaming and non-streaming responses.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestMetadata {
    /// Opaque server-generated request identifier.
    pub request_id: String,
    /// Model identifier selected for the request.
    pub model_id: String,
    /// Unix timestamp in seconds captured at admission.
    pub created_unix_seconds: u64,
}

impl RequestMetadata {
    /// Builds checked response metadata.
    ///
    /// # Errors
    ///
    /// Returns [`MetadataError`] when either identifier is empty or exceeds
    /// its fixed public byte bound.
    pub fn new(
        request_id: impl Into<String>,
        model_id: impl Into<String>,
        created_unix_seconds: u64,
    ) -> Result<Self, MetadataError> {
        let request_id = request_id.into();
        let model_id = model_id.into();
        validate_identifier(
            "request_id",
            &request_id,
            MAX_REQUEST_ID_BYTES,
            "request ID",
        )?;
        validate_identifier("model_id", &model_id, MAX_MODEL_ID_BYTES, "model ID")?;
        Ok(Self {
            request_id,
            model_id,
            created_unix_seconds,
        })
    }
}

/// Public metadata for the single loaded model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelMetadata {
    /// Stable model identifier accepted by generation requests.
    pub model_id: String,
    /// Unix timestamp in seconds associated with this loaded model revision.
    pub created_unix_seconds: u64,
    /// Public owner label.
    pub owned_by: String,
    /// Maximum prompt-plus-output token count.
    pub context_window_tokens: usize,
    /// Server-side output-token ceiling.
    pub max_output_tokens: usize,
}

impl ModelMetadata {
    /// Validates model metadata before it is exposed by the API.
    ///
    /// # Errors
    ///
    /// Returns [`MetadataError`] for invalid string bounds, zero token limits,
    /// or an output limit larger than the context window.
    pub fn validate(&self) -> Result<(), MetadataError> {
        validate_identifier("model_id", &self.model_id, MAX_MODEL_ID_BYTES, "model ID")?;
        validate_identifier(
            "owned_by",
            &self.owned_by,
            MAX_MODEL_OWNER_BYTES,
            "model owner",
        )?;
        if self.context_window_tokens == 0 {
            return Err(MetadataError::InvalidValue {
                field: "context_window_tokens",
                reason: "must be at least one",
            });
        }
        if self.max_output_tokens == 0 {
            return Err(MetadataError::InvalidValue {
                field: "max_output_tokens",
                reason: "must be at least one",
            });
        }
        if self.max_output_tokens > self.context_window_tokens {
            return Err(MetadataError::InvalidValue {
                field: "max_output_tokens",
                reason: "must not exceed the context window",
            });
        }
        Ok(())
    }
}

fn validate_identifier(
    field: &'static str,
    value: &str,
    maximum_bytes: usize,
    label: &'static str,
) -> Result<(), MetadataError> {
    if value.is_empty() {
        return Err(MetadataError::InvalidValue {
            field,
            reason: "must not be empty",
        });
    }
    if value.len() > maximum_bytes {
        return Err(MetadataError::TooLong {
            field,
            label,
            maximum_bytes,
        });
    }
    Ok(())
}

/// Checked model or request metadata failure.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MetadataError {
    /// A metadata value violates a semantic invariant.
    InvalidValue {
        /// Invalid field.
        field: &'static str,
        /// Stable, non-sensitive explanation.
        reason: &'static str,
    },
    /// A public string exceeds its byte bound.
    TooLong {
        /// Invalid field.
        field: &'static str,
        /// Human-readable field label.
        label: &'static str,
        /// Maximum accepted UTF-8 bytes.
        maximum_bytes: usize,
    },
}

impl fmt::Display for MetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue { field, reason } => {
                write!(formatter, "invalid {field}: {reason}")
            }
            Self::TooLong {
                field,
                label,
                maximum_bytes,
            } => write!(
                formatter,
                "invalid {field}: {label} exceeds {maximum_bytes} UTF-8 bytes"
            ),
        }
    }
}

impl error::Error for MetadataError {}

/// Prompt, completion, and total token counts for a completed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

impl TokenUsage {
    /// Creates a usage summary with checked total-count arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`UsageOverflow`] when the two component counts cannot be
    /// represented by the total-count field.
    pub fn new(prompt_tokens: u64, completion_tokens: u64) -> Result<Self, UsageOverflow> {
        let total_tokens = prompt_tokens
            .checked_add(completion_tokens)
            .ok_or(UsageOverflow)?;
        Ok(Self {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        })
    }

    /// Number of tokens in the encoded prompt.
    #[must_use]
    pub const fn prompt_tokens(self) -> u64 {
        self.prompt_tokens
    }

    /// Number of model tokens generated for the completion.
    #[must_use]
    pub const fn completion_tokens(self) -> u64 {
        self.completion_tokens
    }

    /// Sum of prompt and completion token counts.
    #[must_use]
    pub const fn total_tokens(self) -> u64 {
        self.total_tokens
    }
}

/// Token-usage arithmetic overflow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UsageOverflow;

impl fmt::Display for UsageOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("token usage total overflowed")
    }
}

impl error::Error for UsageOverflow {}

/// Why generation reached a terminal state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An EOS token or caller-provided stop condition matched.
    Stop,
    /// The requested maximum output length was reached.
    Length,
    /// The request was cancelled before normal completion.
    Cancelled,
    /// An execution or decoding failure terminated generation.
    Error,
}

/// Stable error classes that may cross from execution into the API layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ServiceErrorClass {
    /// A normalized request is incompatible with loaded-model constraints.
    InvalidRequest,
    /// Admission capacity was exhausted.
    Overloaded,
    /// The configured request deadline elapsed.
    Timeout,
    /// The server is draining and no longer accepts work.
    ShuttingDown,
    /// A client or server-side cancellation ended the request.
    Cancelled,
    /// An internal model, CUDA, scheduler, or decoding operation failed.
    Internal,
}

/// Ordered messages produced by generation before transport encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GenerationEvent {
    /// Newly visible UTF-8 completion text.
    TokenDelta {
        /// Text made visible by this event.
        text: String,
    },
    /// Successful terminal event. No later token delta is valid.
    Finished {
        /// Public terminal reason.
        reason: FinishReason,
        /// Final token counts.
        usage: TokenUsage,
    },
    /// Sanitized failure classification. Internal detail stays in logs only.
    Failed {
        /// Stable public failure class.
        class: ServiceErrorClass,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        MAX_MODEL_ID_BYTES, MetadataError, ModelMetadata, RequestMetadata, TokenUsage,
        UsageOverflow,
    };

    #[test]
    fn request_metadata_enforces_public_identifier_bounds() {
        let metadata = RequestMetadata::new("cmpl-123", "fixture-model", 42)
            .expect("bounded metadata must validate");
        assert_eq!(metadata.request_id, "cmpl-123");
        assert_eq!(metadata.created_unix_seconds, 42);

        assert!(matches!(
            RequestMetadata::new("", "fixture-model", 42),
            Err(MetadataError::InvalidValue {
                field: "request_id",
                ..
            })
        ));
        assert!(matches!(
            RequestMetadata::new("cmpl-123", "x".repeat(MAX_MODEL_ID_BYTES + 1), 42),
            Err(MetadataError::TooLong {
                field: "model_id",
                maximum_bytes: MAX_MODEL_ID_BYTES,
                ..
            })
        ));
    }

    #[test]
    fn model_metadata_checks_token_limit_relationship() {
        let mut metadata = ModelMetadata {
            model_id: "fixture-model".to_owned(),
            created_unix_seconds: 42,
            owned_by: "riley".to_owned(),
            context_window_tokens: 4_096,
            max_output_tokens: 1_024,
        };
        metadata.validate().expect("valid model metadata");

        metadata.max_output_tokens = 4_097;
        assert!(matches!(
            metadata.validate(),
            Err(MetadataError::InvalidValue {
                field: "max_output_tokens",
                ..
            })
        ));
    }

    #[test]
    fn usage_total_is_checked_once() {
        let usage = TokenUsage::new(12, 7).expect("small usage must fit");
        assert_eq!(usage.prompt_tokens(), 12);
        assert_eq!(usage.completion_tokens(), 7);
        assert_eq!(usage.total_tokens(), 19);
        assert_eq!(TokenUsage::new(u64::MAX, 1), Err(UsageOverflow));
    }
}
