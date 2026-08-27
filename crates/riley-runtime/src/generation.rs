//! Model-independent state for one autoregressive generation request.
//!
//! This module owns request-local RNG, token history, text assembly, and stop
//! decisions. It deliberately does not own a model, tokenizer, CUDA stream,
//! or KV cache. Callers sample a token with [`GenerationState::sampling_rng`],
//! ask whether its decoded bytes are needed, and then accept it. All storage
//! needed by the accept path is reserved by [`GenerationState::new`].

use std::error;
use std::fmt;
use std::str;

use crate::rng::{Philox4x32Rng, RngError, RngSnapshot};
use crate::sampling::{SamplingError, SamplingParams, SamplingResult};

const TOKEN_SAMPLING_DOMAIN: &[u8] = b"token-sampling";

/// Result type for model-independent generation state operations.
pub type GenerationResult<T> = Result<T, GenerationError>;

/// Why a request stopped producing model tokens.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FinishReason {
    /// An enabled end-of-sequence token was sampled.
    Eos,
    /// An enabled caller-provided stop token was sampled.
    StopToken,
    /// An enabled UTF-8 stop string was completed.
    StopString,
    /// `max_new_tokens` was reached without a higher-priority stop.
    Length,
    /// The request was cancelled between model steps.
    Cancelled,
    /// Generation failed after request construction.
    Error,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Eos => "eos",
            Self::StopToken => "stop token",
            Self::StopString => "stop string",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        })
    }
}

/// Checked request, allocation, decoding, or state-transition failure.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum GenerationError {
    /// A request field violates a generation invariant.
    InvalidConfiguration {
        /// Invalid request field.
        field: &'static str,
        /// Stable explanation of the invariant.
        reason: &'static str,
    },
    /// Sampling parameters do not define a valid distribution transform.
    Sampling(SamplingError),
    /// Request-local RNG derivation or exhaustion failed.
    Rng(RngError),
    /// A prompt, EOS, stop, or generated token is outside the vocabulary.
    TokenOutOfRange {
        /// Field containing the token.
        field: &'static str,
        /// Element offset within that field.
        index: usize,
        /// Invalid token ID.
        token_id: u32,
        /// Fixed vocabulary size.
        vocabulary_size: usize,
    },
    /// Capacity arithmetic overflowed before any hot-path state was created.
    ArithmeticOverflow {
        /// Capacity whose calculation overflowed.
        field: &'static str,
    },
    /// A cold-path host allocation could not reserve its complete capacity.
    HostAllocation {
        /// Buffer that could not be reserved.
        resource: &'static str,
        /// Requested byte or element capacity.
        requested_capacity: usize,
    },
    /// A visible token was accepted without its raw decoded bytes.
    DecodedBytesRequired {
        /// Token requiring decoding.
        token_id: u32,
    },
    /// A tokenizer returned more bytes than its declared per-token maximum.
    DecodedTokenTooLong {
        /// Token whose decoded representation violated the declaration.
        token_id: u32,
        /// Actual decoded byte length.
        byte_len: usize,
        /// Maximum declared at state construction.
        maximum_byte_len: usize,
    },
    /// An internal emission boundary was not strict UTF-8.
    InvalidUtf8 {
        /// Valid prefix length in the currently withheld byte buffer.
        valid_up_to: usize,
        /// Length of the definitely invalid byte sequence, or `None` for an
        /// incomplete sequence at end of generation.
        error_len: Option<usize>,
    },
    /// Decoded output exceeded the checked cold-path byte bound.
    OutputCapacityExceeded {
        /// Maximum decoded output bytes reserved for the request.
        maximum_bytes: usize,
        /// Bytes that would be retained or emitted after the operation.
        requested_bytes: usize,
    },
    /// A step or token acceptance was attempted after a terminal transition.
    AlreadyFinished {
        /// Existing terminal reason.
        reason: FinishReason,
    },
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid generation configuration {field}: {reason}"
                )
            }
            Self::Sampling(source) => write!(formatter, "invalid generation sampling: {source}"),
            Self::Rng(source) => write!(formatter, "generation RNG failed: {source}"),
            Self::TokenOutOfRange {
                field,
                index,
                token_id,
                vocabulary_size,
            } => write!(
                formatter,
                "generation {field}[{index}] token {token_id} is outside vocabulary {vocabulary_size}"
            ),
            Self::ArithmeticOverflow { field } => {
                write!(
                    formatter,
                    "generation capacity arithmetic overflow for {field}"
                )
            }
            Self::HostAllocation {
                resource,
                requested_capacity,
            } => write!(
                formatter,
                "could not reserve generation {resource} capacity {requested_capacity}"
            ),
            Self::DecodedBytesRequired { token_id } => {
                write!(
                    formatter,
                    "decoded bytes are required for visible token {token_id}"
                )
            }
            Self::DecodedTokenTooLong {
                token_id,
                byte_len,
                maximum_byte_len,
            } => write!(
                formatter,
                "decoded token {token_id} has {byte_len} bytes, exceeding declared maximum {maximum_byte_len}"
            ),
            Self::InvalidUtf8 {
                valid_up_to,
                error_len,
            } => write!(
                formatter,
                "decoded generation text is invalid UTF-8 at withheld byte {valid_up_to} (error length {error_len:?})"
            ),
            Self::OutputCapacityExceeded {
                maximum_bytes,
                requested_bytes,
            } => write!(
                formatter,
                "decoded generation output would use {requested_bytes} bytes, exceeding reserved maximum {maximum_bytes}"
            ),
            Self::AlreadyFinished { reason } => {
                write!(formatter, "generation already finished because of {reason}")
            }
        }
    }
}

impl error::Error for GenerationError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Sampling(source) => Some(source),
            Self::Rng(source) => Some(source),
            _ => None,
        }
    }
}

impl From<SamplingError> for GenerationError {
    fn from(source: SamplingError) -> Self {
        Self::Sampling(source)
    }
}

impl From<RngError> for GenerationError {
    fn from(source: RngError) -> Self {
        Self::Rng(source)
    }
}

/// Cold-path description of one generation request.
///
/// Public fields make the transport boundary straightforward. Call
/// [`Self::validate`] before execution; [`GenerationState::new`] always does
/// so again and is the authoritative checked transition into mutable state.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// Stable arbitrary bytes used as the request-local RNG stream identity.
    pub request_id: Vec<u8>,
    /// Master seed used to derive the request-local sampling stream.
    pub seed: u64,
    /// Non-empty prompt token sequence.
    pub prompt_token_ids: Vec<u32>,
    /// Normative CPU logits-processing and sampling parameters.
    pub sampling_params: SamplingParams,
    /// Number of non-stop generated tokens required before stops are enabled.
    ///
    /// A stop string must start in bytes decoded from a token accepted after
    /// this gate opens. A prefix that began while stops were disabled is not
    /// reactivated later.
    pub min_new_tokens: usize,
    /// Maximum number of sampled tokens retained for this request.
    pub max_new_tokens: usize,
    /// Model end-of-sequence token IDs, in caller preference order.
    pub eos_token_ids: Vec<u32>,
    /// Additional hidden stop token IDs, in caller preference order.
    pub stop_token_ids: Vec<u32>,
    /// Strict UTF-8 stop strings, excluded from emitted text when matched.
    pub stop_strings: Vec<String>,
}

impl GenerationRequest {
    /// Validates all model-independent fields against a vocabulary.
    ///
    /// # Errors
    ///
    /// Returns a structured error for an empty request identity or prompt,
    /// inconsistent token limits, empty stop strings, invalid sampling
    /// parameters, or any token outside `vocabulary_size`.
    pub fn validate(&self, vocabulary_size: usize) -> GenerationResult<()> {
        self.sampling_params.validate(vocabulary_size)?;
        if self.request_id.is_empty() {
            return Err(GenerationError::InvalidConfiguration {
                field: "request_id",
                reason: "must not be empty",
            });
        }
        if self.prompt_token_ids.is_empty() {
            return Err(GenerationError::InvalidConfiguration {
                field: "prompt_token_ids",
                reason: "must not be empty",
            });
        }
        if self.min_new_tokens > self.max_new_tokens {
            return Err(GenerationError::InvalidConfiguration {
                field: "min_new_tokens",
                reason: "must not exceed max_new_tokens",
            });
        }
        if self.stop_strings.iter().any(String::is_empty) {
            return Err(GenerationError::InvalidConfiguration {
                field: "stop_strings",
                reason: "must not contain an empty string",
            });
        }

        validate_token_ids("prompt_token_ids", &self.prompt_token_ids, vocabulary_size)?;
        validate_token_ids("eos_token_ids", &self.eos_token_ids, vocabulary_size)?;
        validate_token_ids("stop_token_ids", &self.stop_token_ids, vocabulary_size)
    }
}

/// Reusable UTF-8 and stop-string state for one request.
///
/// Decoded bytes that could become a stop-string prefix or cannot yet be
/// represented as strict UTF-8 are withheld in [`Self::pending_bytes`]. After
/// the first definitively invalid byte, the complete remaining raw tail stays
/// withheld so visible text never reorders or silently drops bytes. Only a
/// complete strict UTF-8 prefix reaches [`Self::text`] or
/// [`Self::last_text_delta`].
#[derive(Debug)]
pub struct StopState {
    stop_strings: Vec<Vec<u8>>,
    pending: Vec<u8>,
    text: String,
    last_delta: String,
    maximum_output_bytes: usize,
    matched_stop_string_index: Option<usize>,
    // Bytes before this pending-buffer offset were accepted while string
    // stops were disabled and remain permanently ineligible for matching.
    stop_search_start: usize,
}

impl StopState {
    /// Complete emitted text, excluding a matched stop string.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Text made visible by the most recent accept or cancellation operation.
    #[must_use]
    pub fn last_text_delta(&self) -> &str {
        &self.last_delta
    }

    /// Bytes withheld for UTF-8 completion or stop-prefix disambiguation.
    ///
    /// A state can retain an incomplete or definitively invalid UTF-8 tail
    /// here. The raw bytes remain lossless while [`Self::text`] stays strict
    /// UTF-8. Logically, visible text bytes followed by this slice are the raw
    /// decoded output retained before any matched stop string.
    #[must_use]
    pub fn pending_bytes(&self) -> &[u8] {
        &self.pending
    }

    /// Index of the request stop string that completed first, if any.
    ///
    /// Earliest completion byte wins. An earlier start and then request order
    /// break ties, making the result independent of decoded-token chunking.
    #[must_use]
    pub const fn matched_stop_string_index(&self) -> Option<usize> {
        self.matched_stop_string_index
    }

    /// Maximum total decoded bytes reserved for this request.
    #[must_use]
    pub const fn maximum_output_bytes(&self) -> usize {
        self.maximum_output_bytes
    }

    fn new(stop_strings: &[String], maximum_output_bytes: usize) -> GenerationResult<Self> {
        let mut patterns = reserve_vec(stop_strings.len(), "stop-string table")?;
        for stop_string in stop_strings {
            let bytes = stop_string.as_bytes();
            let mut pattern = reserve_vec(bytes.len(), "stop-string bytes")?;
            pattern.extend_from_slice(bytes);
            patterns.push(pattern);
        }

        Ok(Self {
            stop_strings: patterns,
            pending: reserve_vec(maximum_output_bytes, "pending decoded bytes")?,
            text: reserve_string(maximum_output_bytes, "emitted text")?,
            last_delta: reserve_string(maximum_output_bytes, "text delta")?,
            maximum_output_bytes,
            matched_stop_string_index: None,
            stop_search_start: 0,
        })
    }

    fn clear_delta(&mut self) {
        self.last_delta.clear();
    }

    fn push(
        &mut self,
        decoded_bytes: &[u8],
        stop_enabled: bool,
        flush_at_end: bool,
    ) -> GenerationResult<bool> {
        self.clear_delta();
        let retained_bytes = self
            .text
            .len()
            .checked_add(self.pending.len())
            .and_then(|value| value.checked_add(decoded_bytes.len()))
            .ok_or(GenerationError::ArithmeticOverflow {
                field: "decoded output bytes",
            })?;
        if retained_bytes > self.maximum_output_bytes {
            return Err(GenerationError::OutputCapacityExceeded {
                maximum_bytes: self.maximum_output_bytes,
                requested_bytes: retained_bytes,
            });
        }
        let previous_pending_len = self.pending.len();
        let previous_stop_search_start = self.stop_search_start;
        self.pending.extend_from_slice(decoded_bytes);

        let result = self.process_pending(stop_enabled, flush_at_end);
        match result {
            Ok(matched) => {
                if !stop_enabled {
                    // The minimum-token gate is monotonic. Any raw tail
                    // retained while it is closed must never become eligible
                    // for a later stop match merely because it could not be
                    // represented in the strict text view.
                    self.stop_search_start = self.pending.len();
                }
                Ok(matched)
            }
            Err(error) => {
                // Every fallible validation in `process_pending` runs before
                // its corresponding emission. The pre-existing prefix is
                // therefore intact and truncation makes a failed token
                // acceptance transactional.
                self.pending.truncate(previous_pending_len);
                self.stop_search_start = previous_stop_search_start;
                Err(error)
            }
        }
    }

    fn process_pending(
        &mut self,
        stop_enabled: bool,
        flush_at_end: bool,
    ) -> GenerationResult<bool> {
        if stop_enabled {
            if let Some((start, pattern_index)) = self.first_stop_match() {
                // Bytes at and after the stop are excluded. A strict prefix is
                // emitted; any unrepresentable raw bytes before the stop stay
                // losslessly retained in `pending`.
                self.pending.truncate(start);
                self.emit_valid_prefix()?;
                self.matched_stop_string_index = Some(pattern_index);
                return Ok(true);
            }
        }

        if flush_at_end {
            self.finish_pending()?;
            return Ok(false);
        }

        let overlap = if stop_enabled {
            self.longest_stop_prefix_suffix()
        } else {
            0
        };
        let candidate_end = self.pending.len() - overlap;
        let emit_end = match str::from_utf8(&self.pending[..candidate_end]) {
            Ok(_) => candidate_end,
            Err(source) => source.valid_up_to(),
        };
        self.emit_prefix(emit_end)?;
        Ok(false)
    }

    fn finish_pending(&mut self) -> GenerationResult<()> {
        self.emit_valid_prefix()
    }

    fn emit_valid_prefix(&mut self) -> GenerationResult<()> {
        let emit_end = str::from_utf8(&self.pending)
            .map_or_else(|source| source.valid_up_to(), |_| self.pending.len());
        self.emit_prefix(emit_end)
    }

    fn emit_prefix(&mut self, byte_len: usize) -> GenerationResult<()> {
        if byte_len == 0 {
            return Ok(());
        }
        let decoded = str::from_utf8(&self.pending[..byte_len]).map_err(invalid_utf8)?;
        self.text.push_str(decoded);
        self.last_delta.push_str(decoded);
        self.pending.copy_within(byte_len.., 0);
        self.pending.truncate(self.pending.len() - byte_len);
        self.stop_search_start = self.stop_search_start.saturating_sub(byte_len);
        Ok(())
    }

    fn first_stop_match(&self) -> Option<(usize, usize)> {
        let mut first = None;
        let eligible = self.pending.get(self.stop_search_start..)?;
        for (pattern_index, pattern) in self.stop_strings.iter().enumerate() {
            for (relative_start, candidate) in eligible.windows(pattern.len()).enumerate() {
                if candidate == pattern {
                    let start = self.stop_search_start + relative_start;
                    let end = start + pattern.len();
                    let replace = first.is_none_or(|(best_end, best_start, best_pattern)| {
                        (end, start, pattern_index) < (best_end, best_start, best_pattern)
                    });
                    if replace {
                        first = Some((end, start, pattern_index));
                    }
                    // Later occurrences of the same pattern cannot complete
                    // earlier than its first occurrence.
                    break;
                }
            }
        }
        first.map(|(_, start, pattern_index)| (start, pattern_index))
    }

    fn longest_stop_prefix_suffix(&self) -> usize {
        let mut longest = 0;
        let eligible_len = self.pending.len().saturating_sub(self.stop_search_start);
        for pattern in &self.stop_strings {
            let maximum = eligible_len.min(pattern.len().saturating_sub(1));
            for byte_len in (longest + 1..=maximum).rev() {
                if self.pending[self.pending.len() - byte_len..] == pattern[..byte_len] {
                    longest = byte_len;
                    break;
                }
            }
        }
        longest
    }
}

/// One sampled token together with the visible text produced by accepting it.
///
/// The text delta borrows preallocated request state and remains valid until
/// that state is mutated again. EOS and stop-token IDs remain in generated
/// history even though their own decoded bytes are excluded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneratedToken<'a> {
    token_id: u32,
    token_logprob: Option<f32>,
    text_delta: &'a str,
    finish_reason: Option<FinishReason>,
}

impl<'a> GeneratedToken<'a> {
    /// Sampled token ID retained in request history.
    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    /// Optional processed-distribution log-probability for the token.
    #[must_use]
    pub const fn token_logprob(self) -> Option<f32> {
        self.token_logprob
    }

    /// Newly visible strict UTF-8 text, excluding matched stop bytes.
    #[must_use]
    pub const fn text_delta(self) -> &'a str {
        self.text_delta
    }

    /// Terminal reason selected while accepting this token, if any.
    #[must_use]
    pub const fn finish_reason(self) -> Option<FinishReason> {
        self.finish_reason
    }
}

/// Mutable, model-independent state for one validated generation request.
#[derive(Debug)]
pub struct GenerationState {
    request: GenerationRequest,
    rng: Philox4x32Rng,
    vocabulary_size: usize,
    maximum_decoded_token_bytes: usize,
    generated_token_ids: Vec<u32>,
    history_token_ids: Vec<u32>,
    masked_finish_token_ids: Vec<u32>,
    stop_state: StopState,
    finish_reason: Option<FinishReason>,
    failure: Option<GenerationError>,
}

impl GenerationState {
    /// Validates a request and reserves every buffer used while accepting tokens.
    ///
    /// `maximum_decoded_token_bytes` must be the tokenizer's checked maximum
    /// raw byte count for any one token. The product with `max_new_tokens`
    /// forms the exact upper bound reserved for pending, delta, and full text.
    ///
    /// # Errors
    ///
    /// Returns for an invalid request, zero tokenizer byte bound, capacity
    /// overflow, host allocation failure, or RNG derivation failure.
    pub fn new(
        request: GenerationRequest,
        vocabulary_size: usize,
        maximum_decoded_token_bytes: usize,
    ) -> GenerationResult<Self> {
        request.validate(vocabulary_size)?;
        if maximum_decoded_token_bytes == 0 {
            return Err(GenerationError::InvalidConfiguration {
                field: "maximum_decoded_token_bytes",
                reason: "must be non-zero",
            });
        }

        let history_capacity = request
            .prompt_token_ids
            .len()
            .checked_add(request.max_new_tokens)
            .ok_or(GenerationError::ArithmeticOverflow {
                field: "history token count",
            })?;
        let maximum_output_bytes = request
            .max_new_tokens
            .checked_mul(maximum_decoded_token_bytes)
            .ok_or(GenerationError::ArithmeticOverflow {
                field: "decoded output bytes",
            })?;

        let mut history_token_ids = reserve_vec(history_capacity, "token history")?;
        history_token_ids.extend_from_slice(&request.prompt_token_ids);
        let generated_token_ids = reserve_vec(request.max_new_tokens, "generated token IDs")?;
        let masked_finish_token_ids = collect_masked_finish_ids(&request)?;
        let stop_state = StopState::new(&request.stop_strings, maximum_output_bytes)?;
        let rng = Philox4x32Rng::new(request.seed, &request.request_id, TOKEN_SAMPLING_DOMAIN)?;
        let finish_reason = (request.max_new_tokens == 0).then_some(FinishReason::Length);

        Ok(Self {
            request,
            rng,
            vocabulary_size,
            maximum_decoded_token_bytes,
            generated_token_ids,
            history_token_ids,
            masked_finish_token_ids,
            stop_state,
            finish_reason,
            failure: None,
        })
    }

    /// Immutable validated request description.
    #[must_use]
    pub const fn request(&self) -> &GenerationRequest {
        &self.request
    }

    /// Vocabulary size used to validate prompt and generated token IDs.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Tokenizer byte bound used to reserve the strict UTF-8 output state.
    #[must_use]
    pub const fn maximum_decoded_token_bytes(&self) -> usize {
        self.maximum_decoded_token_bytes
    }

    /// Prompt followed by every accepted sampled token.
    #[must_use]
    pub fn history_token_ids(&self) -> &[u32] {
        &self.history_token_ids
    }

    /// Every sampled token, including hidden EOS and stop-token IDs.
    #[must_use]
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    /// Token IDs that must be masked while the minimum-token gate is active.
    ///
    /// EOS IDs precede additional stop-token IDs and duplicates are removed.
    #[must_use]
    pub fn masked_finish_token_ids(&self) -> &[u32] {
        if self.stop_conditions_enabled() {
            &[]
        } else {
            &self.masked_finish_token_ids
        }
    }

    /// Whether EOS, stop-token, and stop-string termination is enabled.
    #[must_use]
    pub fn stop_conditions_enabled(&self) -> bool {
        self.generated_token_ids.len() >= self.request.min_new_tokens
    }

    /// Complete visible strict UTF-8 output.
    #[must_use]
    pub fn text(&self) -> &str {
        self.stop_state.text()
    }

    /// UTF-8 and stop-prefix state retained between token boundaries.
    #[must_use]
    pub const fn stop_state(&self) -> &StopState {
        &self.stop_state
    }

    /// Terminal reason, if generation can no longer take a model step.
    #[must_use]
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish_reason
    }

    /// Structured failure retained when the terminal reason is `error`.
    #[must_use]
    pub const fn failure(&self) -> Option<&GenerationError> {
        self.failure.as_ref()
    }

    /// Number of request-local Philox words consumed so far.
    #[must_use]
    pub fn rng_draws(&self) -> u128 {
        self.rng.draws()
    }

    /// Stable versioned algorithm identifier for result metadata.
    #[must_use]
    pub const fn rng_algorithm_id(&self) -> &'static str {
        self.rng.algorithm_id()
    }

    /// Portable snapshot of the next request-local RNG word.
    ///
    /// Snapshotting does not consume a word. This is intended for pause/resume
    /// and final evidence rather than the per-token hot path.
    #[must_use]
    pub fn rng_snapshot(&self) -> RngSnapshot {
        self.rng.snapshot()
    }

    /// Checks the cancellation/finish boundary immediately before model work.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError::AlreadyFinished`] after any terminal state.
    pub fn pre_step(&self) -> GenerationResult<()> {
        if let Some(reason) = self.finish_reason {
            Err(GenerationError::AlreadyFinished { reason })
        } else {
            Ok(())
        }
    }

    /// Borrows the sole RNG stream used for user-visible token sampling.
    ///
    /// Call this only after logits processing succeeds. Greedy sampling does
    /// not draw from the returned generator under the sampling contract.
    ///
    /// # Errors
    ///
    /// Returns [`GenerationError::AlreadyFinished`] after cancellation or any
    /// other terminal transition, preventing post-cancel RNG consumption.
    pub fn sampling_rng(&mut self) -> GenerationResult<&mut Philox4x32Rng> {
        self.pre_step()?;
        Ok(&mut self.rng)
    }

    /// Whether an accepted token needs raw tokenizer bytes.
    ///
    /// Enabled EOS and stop tokens return `false`; their IDs are retained but
    /// their own decoded representation is hidden. Stops masked by
    /// `min_new_tokens` behave as ordinary visible tokens and return `true`.
    ///
    /// # Errors
    ///
    /// Returns after a terminal transition or for an out-of-range token.
    pub fn token_needs_decoding(&self, token_id: u32) -> GenerationResult<bool> {
        self.pre_step()?;
        self.validate_generated_token(token_id)?;
        let stops_enabled = self.stop_conditions_enabled();
        Ok(!(stops_enabled
            && (self.request.eos_token_ids.contains(&token_id)
                || self.request.stop_token_ids.contains(&token_id))))
    }

    /// Accepts one sampler result and returns its borrowed streaming event.
    ///
    /// # Errors
    ///
    /// Returns the same checked failures as [`Self::accept_token`].
    pub fn accept_sample<'state>(
        &'state mut self,
        sample: SamplingResult,
        decoded_bytes: Option<&[u8]>,
    ) -> GenerationResult<GeneratedToken<'state>> {
        self.accept_token(sample.token_id(), sample.token_logprob(), decoded_bytes)
    }

    /// Accepts one sampled token and advances stop and text state.
    ///
    /// Pass `None` after [`Self::token_needs_decoding`] returns `false`.
    /// Otherwise pass the tokenizer's raw bytes without lossy UTF-8 conversion.
    /// Finish precedence is EOS, stop token, stop string, then length.
    ///
    /// # Errors
    ///
    /// Returns for a terminal state, out-of-range token, missing or oversized
    /// decoded bytes, a violated preallocated output bound, or an internal
    /// strict UTF-8 emission-boundary invariant.
    pub fn accept_token<'state>(
        &'state mut self,
        token_id: u32,
        token_logprob: Option<f32>,
        decoded_bytes: Option<&[u8]>,
    ) -> GenerationResult<GeneratedToken<'state>> {
        self.pre_step()?;
        if let Err(error) = self.validate_generated_token(token_id) {
            return self.fail(error);
        }

        let stops_enabled = self.stop_conditions_enabled();
        let is_eos = stops_enabled && self.request.eos_token_ids.contains(&token_id);
        let is_stop_token =
            stops_enabled && self.request.stop_token_ids.contains(&token_id) && !is_eos;
        let visible_bytes = if is_eos || is_stop_token {
            &[][..]
        } else {
            let Some(bytes) = decoded_bytes else {
                return self.fail(GenerationError::DecodedBytesRequired { token_id });
            };
            if bytes.len() > self.maximum_decoded_token_bytes {
                return self.fail(GenerationError::DecodedTokenTooLong {
                    token_id,
                    byte_len: bytes.len(),
                    maximum_byte_len: self.maximum_decoded_token_bytes,
                });
            }
            bytes
        };

        if is_eos || is_stop_token {
            self.stop_state.clear_delta();
            if let Err(error) = self.stop_state.finish_pending() {
                return self.fail(error);
            }
            self.finish_reason = Some(if is_eos {
                FinishReason::Eos
            } else {
                FinishReason::StopToken
            });
        } else {
            let reaches_length = self.generated_token_ids.len() + 1 == self.request.max_new_tokens;
            let matched_stop =
                match self
                    .stop_state
                    .push(visible_bytes, stops_enabled, reaches_length)
                {
                    Ok(matched) => matched,
                    Err(error) => return self.fail(error),
                };
            if matched_stop {
                self.finish_reason = Some(FinishReason::StopString);
            } else if reaches_length {
                self.finish_reason = Some(FinishReason::Length);
            }
        }

        // All fallible text and stop validation has succeeded. These pushes
        // stay within capacities reserved by `new`, so acceptance commits the
        // sampled ID and history atomically after validation.
        self.generated_token_ids.push(token_id);
        self.history_token_ids.push(token_id);

        Ok(GeneratedToken {
            token_id,
            token_logprob,
            text_delta: self.stop_state.last_text_delta(),
            finish_reason: self.finish_reason,
        })
    }

    /// Cancels between model steps without drawing another RNG word.
    ///
    /// Any valid text retained solely as a possible stop prefix is flushed and
    /// returned as the final delta. An incomplete or definitively invalid raw
    /// UTF-8 tail stays losslessly available through
    /// [`StopState::pending_bytes`] and is not replaced in the strict UTF-8
    /// text view.
    ///
    /// # Errors
    ///
    /// Returns after an existing terminal transition or an internal strict
    /// UTF-8 emission-boundary invariant violation.
    pub fn cancel(&mut self) -> GenerationResult<&str> {
        self.pre_step()?;
        self.stop_state.clear_delta();
        if let Err(error) = self.stop_state.finish_pending() {
            return self.fail(error);
        }
        self.finish_reason = Some(FinishReason::Cancelled);
        Ok(self.stop_state.last_text_delta())
    }

    /// Marks a model, tokenizer, sampler adapter, or consumer failure.
    ///
    /// This overrides EOS, stop, length, or cancellation because a consumer
    /// can fail while handling the final token event. It preserves accepted
    /// IDs, text, the last delta, and any structured internal failure. The
    /// return is `true` when the terminal reason changed and `false` when it
    /// was already `error`. No RNG word is consumed.
    pub fn mark_failed(&mut self) -> bool {
        if self.finish_reason == Some(FinishReason::Error) {
            return false;
        }
        self.finish_reason = Some(FinishReason::Error);
        true
    }

    fn validate_generated_token(&self, token_id: u32) -> GenerationResult<()> {
        let in_range = usize::try_from(token_id).is_ok_and(|token| token < self.vocabulary_size);
        if in_range {
            Ok(())
        } else {
            Err(GenerationError::TokenOutOfRange {
                field: "generated_token_ids",
                index: self.generated_token_ids.len(),
                token_id,
                vocabulary_size: self.vocabulary_size,
            })
        }
    }

    fn fail<T>(&mut self, error: GenerationError) -> GenerationResult<T> {
        self.finish_reason = Some(FinishReason::Error);
        self.failure = Some(error.clone());
        Err(error)
    }
}

fn validate_token_ids(
    field: &'static str,
    token_ids: &[u32],
    vocabulary_size: usize,
) -> GenerationResult<()> {
    for (index, &token_id) in token_ids.iter().enumerate() {
        let in_range = usize::try_from(token_id).is_ok_and(|token| token < vocabulary_size);
        if !in_range {
            return Err(GenerationError::TokenOutOfRange {
                field,
                index,
                token_id,
                vocabulary_size,
            });
        }
    }
    Ok(())
}

fn collect_masked_finish_ids(request: &GenerationRequest) -> GenerationResult<Vec<u32>> {
    let capacity = request
        .eos_token_ids
        .len()
        .checked_add(request.stop_token_ids.len())
        .ok_or(GenerationError::ArithmeticOverflow {
            field: "masked finish token count",
        })?;
    let mut token_ids = reserve_vec(capacity, "masked finish token IDs")?;
    for &token_id in request.eos_token_ids.iter().chain(&request.stop_token_ids) {
        if !token_ids.contains(&token_id) {
            token_ids.push(token_id);
        }
    }
    Ok(token_ids)
}

fn reserve_vec<T>(capacity: usize, resource: &'static str) -> GenerationResult<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| GenerationError::HostAllocation {
            resource,
            requested_capacity: capacity,
        })?;
    Ok(values)
}

fn reserve_string(capacity: usize, resource: &'static str) -> GenerationResult<String> {
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| GenerationError::HostAllocation {
            resource,
            requested_capacity: capacity,
        })?;
    Ok(value)
}

fn invalid_utf8(source: str::Utf8Error) -> GenerationError {
    GenerationError::InvalidUtf8 {
        valid_up_to: source.valid_up_to(),
        error_len: source.error_len(),
    }
}

#[cfg(test)]
mod tests {
    use super::{FinishReason, GenerationError, GenerationRequest, GenerationState};
    use crate::sampling::SamplingParams;

    const VOCABULARY_SIZE: usize = 256;
    const MAXIMUM_TOKEN_BYTES: usize = 16;

    fn request(id: &[u8]) -> GenerationRequest {
        GenerationRequest {
            request_id: id.to_vec(),
            seed: 42,
            prompt_token_ids: vec![1, 2],
            sampling_params: SamplingParams::default(),
            min_new_tokens: 0,
            max_new_tokens: 8,
            eos_token_ids: vec![3],
            stop_token_ids: vec![4],
            stop_strings: vec!["</stop>".to_owned()],
        }
    }

    fn state(id: &[u8]) -> GenerationState {
        GenerationState::new(request(id), VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES)
            .expect("test request is valid")
    }

    #[test]
    fn validation_rejects_inconsistent_request_before_rng_use() {
        let mut invalid = request(b"invalid");
        invalid.min_new_tokens = 9;
        assert_eq!(
            GenerationState::new(invalid, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap_err(),
            GenerationError::InvalidConfiguration {
                field: "min_new_tokens",
                reason: "must not exceed max_new_tokens",
            }
        );

        let mut invalid = request(b"invalid");
        invalid.stop_strings.push(String::new());
        assert!(matches!(
            invalid.validate(VOCABULARY_SIZE),
            Err(GenerationError::InvalidConfiguration {
                field: "stop_strings",
                ..
            })
        ));

        let mut invalid = request(b"invalid");
        invalid.prompt_token_ids[1] = u32::MAX;
        assert!(matches!(
            invalid.validate(VOCABULARY_SIZE),
            Err(GenerationError::TokenOutOfRange {
                field: "prompt_token_ids",
                index: 1,
                ..
            })
        ));
    }

    #[test]
    fn token_acceptance_errors_are_terminal_without_partial_commit() {
        let mut out_of_range = state(b"accept-out-of-range");
        assert!(matches!(
            out_of_range.accept_token(u32::MAX, None, Some(b"x")),
            Err(GenerationError::TokenOutOfRange { .. })
        ));
        assert_eq!(out_of_range.finish_reason(), Some(FinishReason::Error));
        assert!(out_of_range.generated_token_ids().is_empty());

        let mut missing_bytes = state(b"accept-missing-bytes");
        assert_eq!(
            missing_bytes.accept_token(10, None, None).unwrap_err(),
            GenerationError::DecodedBytesRequired { token_id: 10 }
        );
        assert_eq!(missing_bytes.finish_reason(), Some(FinishReason::Error));
        assert!(missing_bytes.generated_token_ids().is_empty());

        let mut oversized =
            GenerationState::new(request(b"accept-oversized"), VOCABULARY_SIZE, 1).unwrap();
        assert!(matches!(
            oversized.accept_token(10, None, Some(b"xx")),
            Err(GenerationError::DecodedTokenTooLong { .. })
        ));
        assert_eq!(oversized.finish_reason(), Some(FinishReason::Error));
        assert!(oversized.generated_token_ids().is_empty());
    }

    #[test]
    fn request_rng_is_fixed_seed_reproducible_and_batch_order_isolated() {
        let mut a_then_b_a = state(b"request-a");
        let mut a_then_b_b = state(b"request-b");
        let a_words = [
            a_then_b_a.sampling_rng().unwrap().next_u32().unwrap(),
            a_then_b_a.sampling_rng().unwrap().next_u32().unwrap(),
        ];
        let b_words = [
            a_then_b_b.sampling_rng().unwrap().next_u32().unwrap(),
            a_then_b_b.sampling_rng().unwrap().next_u32().unwrap(),
        ];

        let mut b_then_a_b = state(b"request-b");
        let mut b_then_a_a = state(b"request-a");
        let reordered_b = [
            b_then_a_b.sampling_rng().unwrap().next_u32().unwrap(),
            b_then_a_b.sampling_rng().unwrap().next_u32().unwrap(),
        ];
        let reordered_a = [
            b_then_a_a.sampling_rng().unwrap().next_u32().unwrap(),
            b_then_a_a.sampling_rng().unwrap().next_u32().unwrap(),
        ];
        assert_eq!(a_words, reordered_a);
        assert_eq!(b_words, reordered_b);
        assert_ne!(a_words, b_words);
    }

    #[test]
    fn minimum_tokens_masks_and_disables_all_stop_forms() {
        let mut request = request(b"minimum");
        request.min_new_tokens = 2;
        let mut state =
            GenerationState::new(request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        assert_eq!(state.masked_finish_token_ids(), &[3, 4]);
        assert!(state.token_needs_decoding(3).unwrap());

        let token = state.accept_token(3, None, Some(b"E")).unwrap();
        assert_eq!(token.text_delta(), "E");
        assert_eq!(token.finish_reason(), None);
        let token = state.accept_token(4, None, Some(b"S")).unwrap();
        assert_eq!(token.text_delta(), "S");
        assert_eq!(token.finish_reason(), None);

        assert!(state.stop_conditions_enabled());
        assert!(state.masked_finish_token_ids().is_empty());
        assert!(!state.token_needs_decoding(3).unwrap());
        let token = state.accept_token(3, None, None).unwrap();
        assert_eq!(token.finish_reason(), Some(FinishReason::Eos));
        assert_eq!(state.generated_token_ids(), &[3, 4, 3]);
        assert_eq!(state.text(), "ES");
    }

    #[test]
    fn minimum_token_gate_never_reactivates_past_raw_stop_bytes() {
        let mut invalid_request = request(b"minimum-invalid-stop-watermark");
        invalid_request.min_new_tokens = 2;
        invalid_request.max_new_tokens = 4;
        invalid_request.stop_strings = vec!["END".to_owned()];
        let mut state =
            GenerationState::new(invalid_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();

        state.accept_token(10, None, Some(b"\xffEND")).unwrap();
        state.accept_token(11, None, Some(b"x")).unwrap();
        assert!(state.stop_conditions_enabled());
        assert_eq!(state.stop_state().pending_bytes(), b"\xffENDx");

        let token = state.accept_token(12, None, Some(b"y")).unwrap();
        assert_eq!(token.finish_reason(), None);
        assert_eq!(state.stop_state().pending_bytes(), b"\xffENDxy");

        let token = state.accept_token(13, None, Some(b"ENDdiscarded")).unwrap();
        assert_eq!(token.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(state.stop_state().pending_bytes(), b"\xffENDxy");
        assert_eq!(state.generated_token_ids(), &[10, 11, 12, 13]);

        let mut crossing_request = request(b"minimum-valid-stop-boundary");
        crossing_request.min_new_tokens = 2;
        crossing_request.stop_strings = vec!["END".to_owned()];
        let mut crossing =
            GenerationState::new(crossing_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        crossing.accept_token(10, None, Some(b"prefix")).unwrap();
        assert_eq!(
            crossing
                .accept_token(11, None, Some(b"E"))
                .unwrap()
                .text_delta(),
            "E"
        );
        let token = crossing.accept_token(12, None, Some(b"ND")).unwrap();
        assert_eq!(token.text_delta(), "ND");
        assert_eq!(token.finish_reason(), None);
        assert_eq!(crossing.text(), "prefixEND");
    }

    #[test]
    fn finish_precedence_and_hidden_token_history_are_exact() {
        let mut state = state(b"precedence-eos");
        let token = state.accept_token(3, Some(-0.25), None).unwrap();
        assert_eq!(token.token_id(), 3);
        assert_eq!(token.token_logprob(), Some(-0.25));
        assert_eq!(token.text_delta(), "");
        assert_eq!(token.finish_reason(), Some(FinishReason::Eos));
        assert_eq!(state.generated_token_ids(), &[3]);

        let mut stop_request = request(b"precedence-stop");
        stop_request.eos_token_ids.push(4);
        let mut state =
            GenerationState::new(stop_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        assert_eq!(
            state.accept_token(4, None, None).unwrap().finish_reason(),
            Some(FinishReason::Eos),
            "EOS wins when one ID belongs to both token sets"
        );

        let mut string_request = request(b"precedence-string");
        string_request.max_new_tokens = 1;
        string_request.stop_strings = vec!["END".to_owned()];
        let mut state =
            GenerationState::new(string_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        let token = state
            .accept_token(10, None, Some(b"beforeENDafter"))
            .unwrap();
        assert_eq!(token.text_delta(), "before");
        assert_eq!(token.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(state.text(), "before");
    }

    #[test]
    fn stop_strings_are_excluded_across_token_boundaries() {
        let mut state = state(b"overlap");
        let first = state.accept_token(10, None, Some(b"hello</st")).unwrap();
        assert_eq!(first.text_delta(), "hello");
        assert_eq!(first.finish_reason(), None);
        assert_eq!(state.stop_state().pending_bytes(), b"</st");

        let second = state.accept_token(11, None, Some(b"op>discarded")).unwrap();
        assert_eq!(second.text_delta(), "");
        assert_eq!(second.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(state.stop_state().matched_stop_string_index(), Some(0));
        assert_eq!(state.text(), "hello");
    }

    #[test]
    fn overlapping_stop_strings_use_chunk_independent_earliest_completion() {
        fn overlap_state(id: &[u8]) -> GenerationState {
            let mut request = request(id);
            request.stop_strings = vec!["abcde".to_owned(), "c".to_owned()];
            GenerationState::new(request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap()
        }

        let mut one_token = overlap_state(b"overlap-one-token");
        let token = one_token.accept_token(10, None, Some(b"abcde")).unwrap();
        assert_eq!(token.text_delta(), "ab");
        assert_eq!(token.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(one_token.stop_state().matched_stop_string_index(), Some(1));

        let mut split_tokens = overlap_state(b"overlap-split-tokens");
        assert_eq!(
            split_tokens
                .accept_token(10, None, Some(b"a"))
                .unwrap()
                .text_delta(),
            ""
        );
        assert_eq!(
            split_tokens
                .accept_token(11, None, Some(b"b"))
                .unwrap()
                .text_delta(),
            ""
        );
        let token = split_tokens.accept_token(12, None, Some(b"c")).unwrap();
        assert_eq!(token.text_delta(), "ab");
        assert_eq!(token.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(split_tokens.text(), one_token.text());
        assert_eq!(
            split_tokens.stop_state().matched_stop_string_index(),
            one_token.stop_state().matched_stop_string_index()
        );

        let mut tie_request = request(b"overlap-completion-tie");
        tie_request.stop_strings = vec!["bc".to_owned(), "abc".to_owned()];
        let mut tie =
            GenerationState::new(tie_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        let token = tie.accept_token(10, None, Some(b"abc")).unwrap();
        assert_eq!(token.text_delta(), "");
        assert_eq!(tie.stop_state().matched_stop_string_index(), Some(1));
    }

    #[test]
    fn stop_strings_match_raw_bytes_after_an_unrenderable_utf8_tail() {
        let mut request = request(b"stop-after-invalid-utf8");
        request.stop_strings = vec!["END".to_owned()];
        let mut state = GenerationState::new(request, VOCABULARY_SIZE, 32).unwrap();
        let token = state
            .accept_token(10, None, Some(b"visible\xffrawENDdiscarded"))
            .unwrap();
        assert_eq!(token.text_delta(), "visible");
        assert_eq!(token.finish_reason(), Some(FinishReason::StopString));
        assert_eq!(state.text(), "visible");
        assert_eq!(state.stop_state().pending_bytes(), b"\xffraw");
        assert_eq!(state.stop_state().matched_stop_string_index(), Some(0));
        assert_eq!(state.generated_token_ids(), &[10]);
    }

    #[test]
    fn split_and_invalid_utf8_tails_are_withheld_without_replacement() {
        let mut split_state = state(b"utf8-split");
        let first = split_state
            .accept_token(10, None, Some(b"caf\xc3"))
            .unwrap();
        assert_eq!(first.text_delta(), "caf");
        assert_eq!(split_state.stop_state().pending_bytes(), b"\xc3");
        let second = split_state.accept_token(11, None, Some(b"\xa9!")).unwrap();
        assert_eq!(second.text_delta(), "é!");
        assert_eq!(split_state.text(), "café!");

        let mut invalid = state(b"utf8-invalid");
        let first = invalid
            .accept_token(10, None, Some(b"visible\xff"))
            .unwrap();
        assert_eq!(first.text_delta(), "visible");
        assert_eq!(first.finish_reason(), None);
        assert_eq!(invalid.stop_state().pending_bytes(), b"\xff");
        let second = invalid.accept_token(11, None, Some(b"later")).unwrap();
        assert_eq!(second.text_delta(), "");
        assert_eq!(invalid.text(), "visible");
        assert_eq!(invalid.stop_state().pending_bytes(), b"\xfflater");
    }

    #[test]
    fn terminal_paths_flush_nonmatching_stop_prefixes() {
        let mut request = request(b"length-flush");
        request.max_new_tokens = 1;
        request.stop_strings = vec!["<stop>".to_owned()];
        let mut length_state =
            GenerationState::new(request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        let token = length_state
            .accept_token(10, None, Some(b"visible<st"))
            .unwrap();
        assert_eq!(token.text_delta(), "visible<st");
        assert_eq!(token.finish_reason(), Some(FinishReason::Length));

        let mut state = state(b"eos-flush");
        assert_eq!(
            state
                .accept_token(10, None, Some(b"visible</st"))
                .unwrap()
                .text_delta(),
            "visible"
        );
        let eos = state.accept_token(3, None, None).unwrap();
        assert_eq!(eos.text_delta(), "</st");
        assert_eq!(eos.finish_reason(), Some(FinishReason::Eos));
    }

    #[test]
    fn cancellation_is_pre_step_and_consumes_no_rng() {
        let mut state = state(b"cancel");
        state.accept_token(10, None, Some(b"held</st")).unwrap();
        let draws = state.rng_draws();
        assert_eq!(state.cancel().unwrap(), "</st");
        assert_eq!(state.finish_reason(), Some(FinishReason::Cancelled));
        assert_eq!(state.rng_draws(), draws);
        assert_eq!(
            state.sampling_rng().unwrap_err(),
            GenerationError::AlreadyFinished {
                reason: FinishReason::Cancelled,
            }
        );
    }

    #[test]
    fn external_failure_is_terminal_idempotent_and_consumes_no_rng() {
        let mut failed_state = state(b"external-error");
        let draws = failed_state.rng_draws();
        assert!(failed_state.mark_failed());
        assert_eq!(failed_state.finish_reason(), Some(FinishReason::Error));
        assert_eq!(failed_state.failure(), None);
        assert_eq!(failed_state.rng_draws(), draws);
        assert!(!failed_state.mark_failed());
        assert_eq!(
            failed_state.pre_step(),
            Err(GenerationError::AlreadyFinished {
                reason: FinishReason::Error,
            })
        );

        let mut eos = state(b"external-error-after-eos");
        eos.accept_token(3, None, None).unwrap();
        assert!(eos.mark_failed());
        assert_eq!(eos.finish_reason(), Some(FinishReason::Error));
    }

    #[test]
    fn terminal_unrenderable_utf8_is_retained_without_lossy_replacement() {
        let mut length_request = request(b"utf8-length");
        length_request.max_new_tokens = 1;
        let mut length_state =
            GenerationState::new(length_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        let token = length_state
            .accept_token(10, None, Some(b"visible\xe2\x82"))
            .unwrap();
        assert_eq!(token.text_delta(), "visible");
        assert_eq!(token.finish_reason(), Some(FinishReason::Length));
        assert_eq!(length_state.generated_token_ids(), &[10]);
        assert_eq!(length_state.text(), "visible");
        assert_eq!(length_state.stop_state().pending_bytes(), b"\xe2\x82");

        let mut invalid_request = request(b"utf8-invalid-length");
        invalid_request.max_new_tokens = 1;
        let mut invalid =
            GenerationState::new(invalid_request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        let token = invalid
            .accept_token(10, None, Some(b"visible\xfflater"))
            .unwrap();
        assert_eq!(token.text_delta(), "visible");
        assert_eq!(token.finish_reason(), Some(FinishReason::Length));
        assert_eq!(invalid.text(), "visible");
        assert_eq!(invalid.stop_state().pending_bytes(), b"\xfflater");

        let mut eos = state(b"utf8-eos");
        assert_eq!(
            eos.accept_token(10, None, Some(b"prefix\xe2"))
                .unwrap()
                .text_delta(),
            "prefix"
        );
        let token = eos.accept_token(3, None, None).unwrap();
        assert_eq!(token.text_delta(), "");
        assert_eq!(token.finish_reason(), Some(FinishReason::Eos));
        assert_eq!(eos.stop_state().pending_bytes(), b"\xe2");

        let mut cancelled = state(b"utf8-cancel");
        cancelled
            .accept_token(10, None, Some(b"prefix\xe2"))
            .unwrap();
        assert_eq!(cancelled.cancel().unwrap(), "");
        assert_eq!(cancelled.finish_reason(), Some(FinishReason::Cancelled));
        assert_eq!(cancelled.stop_state().pending_bytes(), b"\xe2");
    }

    #[test]
    fn zero_maximum_is_immediately_length_finished() {
        let mut request = request(b"empty-generation");
        request.min_new_tokens = 0;
        request.max_new_tokens = 0;
        let state = GenerationState::new(request, VOCABULARY_SIZE, MAXIMUM_TOKEN_BYTES).unwrap();
        assert_eq!(state.finish_reason(), Some(FinishReason::Length));
        assert_eq!(
            state.pre_step(),
            Err(GenerationError::AlreadyFinished {
                reason: FinishReason::Length,
            })
        );
    }

    #[test]
    fn accepting_tokens_does_not_change_preallocated_buffer_addresses() {
        let mut request = request(b"allocation-free");
        request.max_new_tokens = 4;
        request.stop_strings = vec!["XYZ".to_owned()];
        let mut state = GenerationState::new(request, VOCABULARY_SIZE, 4).unwrap();

        let generated_pointer = state.generated_token_ids.as_ptr();
        let history_pointer = state.history_token_ids.as_ptr();
        let pending_pointer = state.stop_state.pending.as_ptr();
        let text_pointer = state.stop_state.text.as_ptr();
        let delta_pointer = state.stop_state.last_delta.as_ptr();
        let capacities = (
            state.generated_token_ids.capacity(),
            state.history_token_ids.capacity(),
            state.stop_state.pending.capacity(),
            state.stop_state.text.capacity(),
            state.stop_state.last_delta.capacity(),
        );

        for (token_id, bytes) in [(10, b"ab".as_slice()), (11, b"cX"), (12, b"de")] {
            assert_eq!(
                state
                    .accept_token(token_id, None, Some(bytes))
                    .unwrap()
                    .finish_reason(),
                None
            );
        }
        assert_eq!(
            state
                .accept_token(13, None, Some(b"fg"))
                .unwrap()
                .finish_reason(),
            Some(FinishReason::Length)
        );

        assert_eq!(state.generated_token_ids.as_ptr(), generated_pointer);
        assert_eq!(state.history_token_ids.as_ptr(), history_pointer);
        assert_eq!(state.stop_state.pending.as_ptr(), pending_pointer);
        assert_eq!(state.stop_state.text.as_ptr(), text_pointer);
        assert_eq!(state.stop_state.last_delta.as_ptr(), delta_pointer);
        assert_eq!(
            (
                state.generated_token_ids.capacity(),
                state.history_token_ids.capacity(),
                state.stop_state.pending.capacity(),
                state.stop_state.text.capacity(),
                state.stop_state.last_delta.capacity(),
            ),
            capacities
        );
    }
}
