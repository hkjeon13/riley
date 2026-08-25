//! CUDA-backed single-request autoregressive generation.
//!
//! [`PreparedLlamaGeneration`] connects the reusable Llama prefill/decode
//! owner to model-independent [`GenerationState`]. Every host buffer and CUDA
//! timing event is allocated during preparation. A request producing `N`
//! sampled tokens executes prefill once and decode at most `N - 1` times: the
//! final sampled token is never appended merely to produce unused logits.

use std::convert::Infallible;
use std::error;
use std::fmt;
use std::time::{Duration, Instant};

use rustinfer_cuda::{CudaContext, CudaError, CudaEvent, CudaStream};
use rustinfer_model::{DecodeOptions, LoadedModel, ModelError, Tokenizer};

use super::{
    LlamaDecodeError, LlamaDecodePhase, PreparedLlamaDecode, PreparedLlamaDecodeAllocationReport,
    PreparedLlamaDecodeConfig,
};
use crate::generation::{FinishReason, GeneratedToken, GenerationError, GenerationState};
use crate::paged_kv::KvBlockPoolStats;
use crate::rng::RngError;
use crate::sampling::{SamplingError, SamplingWorkspace, TokenConstraints};

const BF16_BYTES: usize = 2;

/// Result returned by preparation, generation, and explicit cleanup.
pub type LlamaGenerationResult<T, CallbackError = Infallible> =
    Result<T, LlamaGenerationError<CallbackError>>;

/// Model stage whose logits were sampled for one output token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GenerationModelStage {
    /// Full fixed-length prompt execution for the first output token.
    Prefill,
    /// One-token decode of the preceding sampled token.
    Decode,
}

/// Timing boundaries collected for one sampled output token.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationTokenTiming {
    model_stage: GenerationModelStage,
    model_gpu_milliseconds: f32,
    model_wall: Duration,
    logits_download_bytes: usize,
    logits_download_wall: Duration,
    sampling_cpu: Duration,
    detokenize_stop_cpu: Duration,
    total_wall: Duration,
}

impl GenerationTokenTiming {
    /// Whether this token sampled prefill or one-token decode logits.
    #[must_use]
    pub const fn model_stage(self) -> GenerationModelStage {
        self.model_stage
    }

    /// CUDA-event elapsed time around the model stage.
    #[must_use]
    pub const fn model_gpu_milliseconds(self) -> f32 {
        self.model_gpu_milliseconds
    }

    /// Host wall time from model event record through event completion.
    #[must_use]
    pub const fn model_wall(self) -> Duration {
        self.model_wall
    }

    /// Full BF16 vocabulary row copied from device to host.
    #[must_use]
    pub const fn logits_download_bytes(self) -> usize {
        self.logits_download_bytes
    }

    /// Host wall time for the complete logits download boundary.
    #[must_use]
    pub const fn logits_download_wall(self) -> Duration {
        self.logits_download_wall
    }

    /// CPU wall time for constraints, logits processing, and sampling.
    #[must_use]
    pub const fn sampling_cpu(self) -> Duration {
        self.sampling_cpu
    }

    /// CPU wall time for raw token decode, UTF-8 assembly, and stop checks.
    #[must_use]
    pub const fn detokenize_stop_cpu(self) -> Duration {
        self.detokenize_stop_cpu
    }

    /// End-to-end token time before invoking the consumer callback.
    #[must_use]
    pub const fn total_wall(self) -> Duration {
        self.total_wall
    }
}

/// Aggregate timing for one completed generation call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LlamaGenerationTimingSummary {
    sampled_tokens: usize,
    prefill_tokens: usize,
    decode_tokens: usize,
    model_gpu_milliseconds: f64,
    model_wall: Duration,
    logits_download_bytes: u64,
    logits_download_wall: Duration,
    sampling_cpu: Duration,
    detokenize_stop_cpu: Duration,
    token_wall: Duration,
    request_wall: Duration,
}

impl LlamaGenerationTimingSummary {
    /// Number of sampled token events included in the summary.
    #[must_use]
    pub const fn sampled_tokens(self) -> usize {
        self.sampled_tokens
    }

    /// Number of token events produced from prefill logits.
    #[must_use]
    pub const fn prefill_tokens(self) -> usize {
        self.prefill_tokens
    }

    /// Number of token events produced from decode logits.
    #[must_use]
    pub const fn decode_tokens(self) -> usize {
        self.decode_tokens
    }

    /// Sum of CUDA-event model-stage elapsed milliseconds.
    #[must_use]
    pub const fn model_gpu_milliseconds(self) -> f64 {
        self.model_gpu_milliseconds
    }

    /// Sum of host model-stage wall durations.
    #[must_use]
    pub const fn model_wall(self) -> Duration {
        self.model_wall
    }

    /// Sum of copied logits bytes across sampled tokens.
    #[must_use]
    pub const fn logits_download_bytes(self) -> u64 {
        self.logits_download_bytes
    }

    /// Sum of host logits-download wall durations.
    #[must_use]
    pub const fn logits_download_wall(self) -> Duration {
        self.logits_download_wall
    }

    /// Sum of CPU sampling durations.
    #[must_use]
    pub const fn sampling_cpu(self) -> Duration {
        self.sampling_cpu
    }

    /// Sum of CPU detokenization and stop-processing durations.
    #[must_use]
    pub const fn detokenize_stop_cpu(self) -> Duration {
        self.detokenize_stop_cpu
    }

    /// Sum of per-token wall durations, excluding consumer callbacks.
    #[must_use]
    pub const fn token_wall(self) -> Duration {
        self.token_wall
    }

    /// Complete generate-call wall duration before KV reset.
    ///
    /// Unlike [`Self::token_wall`], this includes consumer callback time and a
    /// cancellation callback. It excludes post-request decoder reset.
    #[must_use]
    pub const fn request_wall(self) -> Duration {
        self.request_wall
    }

    fn record(&mut self, timing: GenerationTokenTiming) {
        self.sampled_tokens += 1;
        match timing.model_stage {
            GenerationModelStage::Prefill => self.prefill_tokens += 1,
            GenerationModelStage::Decode => self.decode_tokens += 1,
        }
        self.model_gpu_milliseconds += f64::from(timing.model_gpu_milliseconds);
        self.model_wall += timing.model_wall;
        self.logits_download_bytes = self
            .logits_download_bytes
            .saturating_add(u64::try_from(timing.logits_download_bytes).unwrap_or(u64::MAX));
        self.logits_download_wall += timing.logits_download_wall;
        self.sampling_cpu += timing.sampling_cpu;
        self.detokenize_stop_cpu += timing.detokenize_stop_cpu;
        self.token_wall += timing.total_wall;
    }
}

/// Streaming event delivered after one token or a cancellation flush.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LlamaGenerationEvent<'a> {
    /// One accepted sampled token and its measured execution boundaries.
    Token {
        /// Token ID, visible text delta, log-probability, and finish reason.
        token: GeneratedToken<'a>,
        /// Per-token CUDA and CPU timing.
        timing: GenerationTokenTiming,
    },
    /// Cancellation completed and released any safe withheld stop prefix.
    Cancelled {
        /// Final visible text delta released without sampling another token.
        text_delta: &'a str,
    },
}

/// First failure observed while explicitly releasing generation resources.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaGenerationCleanupError {
    /// Decoder reset or explicit decoder close failed.
    Decode(Box<LlamaDecodeError>),
    /// CUDA timing-event close failed.
    Cuda(CudaError),
}

impl fmt::Display for LlamaGenerationCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(source) => write!(
                formatter,
                "Llama generation decoder cleanup failed: {source}"
            ),
            Self::Cuda(source) => {
                write!(formatter, "Llama generation CUDA cleanup failed: {source}")
            }
        }
    }
}

impl error::Error for LlamaGenerationCleanupError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Decode(source) => Some(source.as_ref()),
            Self::Cuda(source) => Some(source),
        }
    }
}

/// Primary preparation or execution failure.
#[derive(Debug)]
#[non_exhaustive]
pub enum LlamaGenerationFailure<CallbackError> {
    /// Cold or request-level invariant was violated before model mutation.
    InvalidConfiguration {
        /// Invalid field or lifecycle component.
        field: &'static str,
        /// Stable explanation of the required invariant.
        reason: &'static str,
    },
    /// Checked host capacity arithmetic overflowed.
    ArithmeticOverflow {
        /// Capacity whose calculation overflowed.
        field: &'static str,
    },
    /// A cold-path host buffer could not reserve its full capacity.
    HostAllocation {
        /// Buffer being allocated.
        resource: &'static str,
        /// Requested element count.
        requested_elements: usize,
    },
    /// Model-independent request state rejected an operation.
    Generation(GenerationError),
    /// CPU logits processing rejected its inputs or distribution.
    Sampling(SamplingError),
    /// Request-local RNG failed after successful logits processing.
    Rng(RngError),
    /// Raw caller-buffer token decoding failed.
    Tokenizer(ModelError),
    /// Llama prefill, decode, logits download, reset, or close failed.
    Decode(Box<LlamaDecodeError>),
    /// CUDA timing-event creation, record, synchronization, or close failed.
    Cuda(CudaError),
    /// The user-provided token consumer returned an error.
    Callback(CallbackError),
    /// This prepared owner lost its decoder after a fatal native failure.
    Terminal,
    /// Generation completed, but decoder reset or explicit close failed.
    Cleanup,
}

impl<CallbackError: fmt::Display> fmt::Display for LlamaGenerationFailure<CallbackError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration { field, reason } => {
                write!(
                    formatter,
                    "invalid Llama generation configuration {field}: {reason}"
                )
            }
            Self::ArithmeticOverflow { field } => {
                write!(formatter, "Llama generation capacity overflow for {field}")
            }
            Self::HostAllocation {
                resource,
                requested_elements,
            } => write!(
                formatter,
                "could not reserve {requested_elements} elements for Llama generation {resource}"
            ),
            Self::Generation(source) => write!(formatter, "generation state failed: {source}"),
            Self::Sampling(source) => write!(formatter, "generation sampling failed: {source}"),
            Self::Rng(source) => write!(formatter, "generation RNG failed: {source}"),
            Self::Tokenizer(source) => write!(formatter, "generation tokenizer failed: {source}"),
            Self::Decode(source) => write!(formatter, "Llama generation model failed: {source}"),
            Self::Cuda(source) => write!(formatter, "Llama generation timing failed: {source}"),
            Self::Callback(source) => write!(formatter, "generation consumer failed: {source}"),
            Self::Terminal => formatter.write_str(
                "the prepared Llama generation owner is terminal after a native failure",
            ),
            Self::Cleanup => formatter.write_str("Llama generation resource cleanup failed"),
        }
    }
}

/// Structured generation failure plus fail-closed cleanup evidence.
#[derive(Debug)]
pub struct LlamaGenerationError<CallbackError = Infallible> {
    failure: LlamaGenerationFailure<CallbackError>,
    first_cleanup_failure: Option<LlamaGenerationCleanupError>,
    additional_cleanup_failures: usize,
}

impl<CallbackError> LlamaGenerationError<CallbackError> {
    /// Primary preparation, execution, callback, or lifecycle failure.
    #[must_use]
    pub const fn failure(&self) -> &LlamaGenerationFailure<CallbackError> {
        &self.failure
    }

    /// First cleanup failure after every owned resource was attempted.
    #[must_use]
    pub const fn first_cleanup_failure(&self) -> Option<&LlamaGenerationCleanupError> {
        self.first_cleanup_failure.as_ref()
    }

    /// Number of cleanup failures after [`Self::first_cleanup_failure`].
    #[must_use]
    pub const fn additional_cleanup_failures(&self) -> usize {
        self.additional_cleanup_failures
    }

    fn new(failure: LlamaGenerationFailure<CallbackError>, cleanup: CleanupFailures) -> Self {
        Self {
            failure,
            first_cleanup_failure: cleanup.first,
            additional_cleanup_failures: cleanup.additional,
        }
    }
}

impl<CallbackError: fmt::Display> fmt::Display for LlamaGenerationError<CallbackError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.failure.fmt(formatter)?;
        if let Some(cleanup) = &self.first_cleanup_failure {
            write!(
                formatter,
                "; cleanup also failed: {cleanup} ({} additional cleanup failures)",
                self.additional_cleanup_failures
            )?;
        }
        Ok(())
    }
}

impl<CallbackError> error::Error for LlamaGenerationError<CallbackError>
where
    CallbackError: error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match &self.failure {
            LlamaGenerationFailure::Generation(source) => Some(source),
            LlamaGenerationFailure::Sampling(source) => Some(source),
            LlamaGenerationFailure::Rng(source) => Some(source),
            LlamaGenerationFailure::Tokenizer(source) => Some(source),
            LlamaGenerationFailure::Decode(source) => Some(source.as_ref()),
            LlamaGenerationFailure::Cuda(source) => Some(source),
            LlamaGenerationFailure::Callback(source) => Some(source),
            LlamaGenerationFailure::Cleanup => self
                .first_cleanup_failure
                .as_ref()
                .map(|source| source as &(dyn error::Error + 'static)),
            LlamaGenerationFailure::InvalidConfiguration { .. }
            | LlamaGenerationFailure::ArithmeticOverflow { .. }
            | LlamaGenerationFailure::HostAllocation { .. }
            | LlamaGenerationFailure::Terminal => None,
        }
    }
}

#[derive(Debug, Default)]
struct CleanupFailures {
    first: Option<LlamaGenerationCleanupError>,
    additional: usize,
}

impl CleanupFailures {
    fn record(&mut self, failure: LlamaGenerationCleanupError) {
        if self.first.is_none() {
            self.first = Some(failure);
        } else {
            self.additional = self.additional.saturating_add(1);
        }
    }

    fn absorb(&mut self, other: Self) {
        if let Some(first) = other.first {
            self.record(first);
        }
        self.additional = self.additional.saturating_add(other.additional);
    }

    const fn is_empty(&self) -> bool {
        self.first.is_none()
    }
}

/// Reusable CUDA Llama generation owner for one prompt shape and output cap.
///
/// The decoder is wrapped in `Option` so a poisoned native owner can be taken,
/// explicitly closed, and made permanently inaccessible. Healthy finish,
/// cancellation, and callback-error paths reset the decoder and retain every
/// allocation for a later request with the same prompt length.
pub struct PreparedLlamaGeneration<'model> {
    decode: Option<PreparedLlamaDecode>,
    tokenizer: &'model dyn Tokenizer,
    sampling: SamplingWorkspace,
    logits_bf16_native: Vec<u8>,
    allowed_tokens: Vec<bool>,
    decoded_token_bytes: Vec<u8>,
    model_start_event: Option<CudaEvent>,
    model_end_event: Option<CudaEvent>,
    prompt_length: usize,
    output_capacity: usize,
    vocabulary_size: usize,
    terminal: bool,
}

impl fmt::Debug for PreparedLlamaGeneration<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedLlamaGeneration")
            .field("prompt_length", &self.prompt_length)
            .field("output_capacity", &self.output_capacity)
            .field("vocabulary_size", &self.vocabulary_size)
            .field("logits_bytes", &self.logits_bf16_native.len())
            .field("terminal", &self.terminal)
            .finish_non_exhaustive()
    }
}

impl<'model> PreparedLlamaGeneration<'model> {
    /// Prepares a fixed-prompt-shape decoder and all generation hot-path state.
    ///
    /// `output_capacity` is a sampled-token capacity, not a decode-call count.
    /// The underlying maximum sequence is therefore `prompt_length +
    /// output_capacity.saturating_sub(1)`.
    ///
    /// # Errors
    ///
    /// Returns for invalid or overflowing capacities, host allocation failure,
    /// CUDA event creation failure, or decoder preparation failure. Partial
    /// CUDA resources are explicitly closed and their first cleanup failure is
    /// attached to the returned error.
    #[allow(clippy::too_many_lines)]
    pub fn prepare(
        model: &'model LoadedModel,
        context: &CudaContext,
        stream: &mut CudaStream,
        prompt_length: usize,
        output_capacity: usize,
        config: PreparedLlamaDecodeConfig,
    ) -> LlamaGenerationResult<Self> {
        if prompt_length == 0 {
            return Err(LlamaGenerationError::new(
                LlamaGenerationFailure::InvalidConfiguration {
                    field: "prompt_length",
                    reason: "must be non-zero",
                },
                CleanupFailures::default(),
            ));
        }

        let maximum_sequence_length = prompt_length
            .checked_add(output_capacity.saturating_sub(1))
            .ok_or_else(|| {
                LlamaGenerationError::new(
                    LlamaGenerationFailure::ArithmeticOverflow {
                        field: "maximum_sequence_length",
                    },
                    CleanupFailures::default(),
                )
            })?;
        let vocabulary_size = model.spec().embedding().vocabulary_size();
        let logits_bytes = vocabulary_size.checked_mul(BF16_BYTES).ok_or_else(|| {
            LlamaGenerationError::new(
                LlamaGenerationFailure::ArithmeticOverflow {
                    field: "logits row bytes",
                },
                CleanupFailures::default(),
            )
        })?;
        let maximum_decoded_token_bytes = model.tokenizer().maximum_decoded_token_bytes();
        if maximum_decoded_token_bytes == 0 {
            return Err(LlamaGenerationError::new(
                LlamaGenerationFailure::InvalidConfiguration {
                    field: "maximum_decoded_token_bytes",
                    reason: "tokenizer must declare a non-zero bound",
                },
                CleanupFailures::default(),
            ));
        }

        let sampling = SamplingWorkspace::new(vocabulary_size).map_err(|source| {
            LlamaGenerationError::new(
                LlamaGenerationFailure::Sampling(source),
                CleanupFailures::default(),
            )
        })?;
        let logits_bf16_native = allocate_filled(logits_bytes, 0_u8, "logits row")?;
        let allowed_tokens = allocate_filled(vocabulary_size, true, "allowed-token mask")?;
        let decoded_token_bytes =
            allocate_filled(maximum_decoded_token_bytes, 0_u8, "decoded-token bytes")?;

        let model_start_event = context.create_event().map_err(|source| {
            LlamaGenerationError::new(
                LlamaGenerationFailure::Cuda(source),
                CleanupFailures::default(),
            )
        })?;
        let model_end_event = match context.create_event() {
            Ok(event) => event,
            Err(source) => {
                let mut cleanup = CleanupFailures::default();
                if let Err(cleanup_source) = model_start_event.close() {
                    cleanup.record(LlamaGenerationCleanupError::Cuda(cleanup_source));
                }
                return Err(LlamaGenerationError::new(
                    LlamaGenerationFailure::Cuda(source),
                    cleanup,
                ));
            }
        };
        let decode = match PreparedLlamaDecode::prepare(
            model,
            context,
            stream,
            prompt_length,
            maximum_sequence_length,
            config,
        ) {
            Ok(decode) => decode,
            Err(source) => {
                let mut cleanup = CleanupFailures::default();
                if let Err(cleanup_source) = model_start_event.close() {
                    cleanup.record(LlamaGenerationCleanupError::Cuda(cleanup_source));
                }
                if let Err(cleanup_source) = model_end_event.close() {
                    cleanup.record(LlamaGenerationCleanupError::Cuda(cleanup_source));
                }
                return Err(LlamaGenerationError::new(
                    LlamaGenerationFailure::Decode(Box::new(source)),
                    cleanup,
                ));
            }
        };

        Ok(Self {
            decode: Some(decode),
            tokenizer: model.tokenizer(),
            sampling,
            logits_bf16_native,
            allowed_tokens,
            decoded_token_bytes,
            model_start_event: Some(model_start_event),
            model_end_event: Some(model_end_event),
            prompt_length,
            output_capacity,
            vocabulary_size,
            terminal: false,
        })
    }

    /// Fixed prompt token count accepted by this prepared owner.
    #[must_use]
    pub const fn prompt_length(&self) -> usize {
        self.prompt_length
    }

    /// Maximum sampled tokens accepted by one request.
    #[must_use]
    pub const fn output_capacity(&self) -> usize {
        self.output_capacity
    }

    /// Fixed model vocabulary used by the logits row and sampler.
    #[must_use]
    pub const fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// BF16 bytes copied for every sampled token.
    #[must_use]
    pub fn logits_row_bytes(&self) -> usize {
        self.logits_bf16_native.len()
    }

    /// Whether a fatal CUDA/model failure permanently consumed the decoder.
    #[must_use]
    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Current decoder phase, or `None` after terminal cleanup.
    #[must_use]
    pub fn decode_phase(&self) -> Option<LlamaDecodePhase> {
        self.decode.as_ref().map(PreparedLlamaDecode::phase)
    }

    /// Prepared decoder allocation report, or `None` after terminal cleanup.
    #[must_use]
    pub fn decode_allocation_report(&self) -> Option<PreparedLlamaDecodeAllocationReport> {
        self.decode
            .as_ref()
            .map(PreparedLlamaDecode::allocation_report)
    }

    /// Current paged-pool lifecycle statistics when paged KV is selected.
    #[must_use]
    pub fn paged_pool_stats(&self) -> Option<KvBlockPoolStats> {
        self.decode.as_ref()?.paged_pool_stats()
    }

    /// Generates one request, streams events, and resets healthy KV state.
    ///
    /// Cancellation is checked before prefill or decode, then again after the
    /// model stage completes but before logits download or RNG consumption. A
    /// callback `Err` is a checked request failure and still resets a healthy
    /// decoder. CUDA timing errors and poisoned model errors consume and close
    /// the decoder, making this owner terminal. Callback panics are outside the
    /// checked cleanup contract; use `Err` for recoverable consumer failures.
    ///
    /// # Errors
    ///
    /// Returns for a request/owner shape mismatch, model-independent state
    /// error, logits processing or RNG failure, tokenizer failure, CUDA/model
    /// failure, callback error, or cleanup failure.
    pub fn generate<CallbackError, Cancel, Callback>(
        &mut self,
        state: &mut GenerationState,
        stream: &mut CudaStream,
        mut is_cancelled: Cancel,
        mut callback: Callback,
    ) -> LlamaGenerationResult<LlamaGenerationTimingSummary, CallbackError>
    where
        Cancel: FnMut() -> bool,
        Callback: for<'event> FnMut(LlamaGenerationEvent<'event>) -> Result<(), CallbackError>,
    {
        let result = self.generate_inner(state, stream, &mut is_cancelled, &mut callback);
        match result {
            Ok(summary) => {
                let cleanup = self.reset_after_run();
                if cleanup.is_empty() {
                    Ok(summary)
                } else {
                    state.mark_failed();
                    Err(LlamaGenerationError::new(
                        LlamaGenerationFailure::Cleanup,
                        cleanup,
                    ))
                }
            }
            Err(failure) => {
                state.mark_failed();
                let fatal = matches!(&failure, LlamaGenerationFailure::Cuda(_))
                    || self
                        .decode
                        .as_ref()
                        .is_some_and(PreparedLlamaDecode::is_poisoned);
                let cleanup = if fatal {
                    self.close_owned_resources()
                } else {
                    self.reset_after_run()
                };
                Err(LlamaGenerationError::new(failure, cleanup))
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn generate_inner<CallbackError, Cancel, Callback>(
        &mut self,
        state: &mut GenerationState,
        stream: &mut CudaStream,
        is_cancelled: &mut Cancel,
        callback: &mut Callback,
    ) -> Result<LlamaGenerationTimingSummary, LlamaGenerationFailure<CallbackError>>
    where
        Cancel: FnMut() -> bool,
        Callback: for<'event> FnMut(LlamaGenerationEvent<'event>) -> Result<(), CallbackError>,
    {
        self.validate_state(state)?;
        let request_started = Instant::now();
        let mut summary = LlamaGenerationTimingSummary::default();
        if state.finish_reason() == Some(FinishReason::Length) {
            summary.request_wall = request_started.elapsed();
            return Ok(summary);
        }

        let mut decode_token = None;
        loop {
            state
                .pre_step()
                .map_err(LlamaGenerationFailure::Generation)?;
            if is_cancelled() {
                let text_delta = state.cancel().map_err(LlamaGenerationFailure::Generation)?;
                callback(LlamaGenerationEvent::Cancelled { text_delta })
                    .map_err(LlamaGenerationFailure::Callback)?;
                summary.request_wall = request_started.elapsed();
                return Ok(summary);
            }

            let token_started = Instant::now();
            let model_stage = if decode_token.is_some() {
                GenerationModelStage::Decode
            } else {
                GenerationModelStage::Prefill
            };
            let model_started = Instant::now();
            self.model_start_event
                .as_mut()
                .ok_or(LlamaGenerationFailure::Terminal)?
                .record(stream)
                .map_err(LlamaGenerationFailure::Cuda)?;
            match decode_token {
                Some(token_id) => self
                    .decode
                    .as_mut()
                    .ok_or(LlamaGenerationFailure::Terminal)?
                    .decode(token_id, stream)
                    .map_err(|source| LlamaGenerationFailure::Decode(Box::new(source)))?,
                None => self
                    .decode
                    .as_mut()
                    .ok_or(LlamaGenerationFailure::Terminal)?
                    .prefill(&state.request().prompt_token_ids, stream)
                    .map_err(|source| LlamaGenerationFailure::Decode(Box::new(source)))?,
            }
            self.model_end_event
                .as_mut()
                .ok_or(LlamaGenerationFailure::Terminal)?
                .record(stream)
                .map_err(LlamaGenerationFailure::Cuda)?;
            self.model_end_event
                .as_mut()
                .ok_or(LlamaGenerationFailure::Terminal)?
                .synchronize()
                .map_err(LlamaGenerationFailure::Cuda)?;
            let model_wall = model_started.elapsed();
            let model_gpu_milliseconds = self
                .model_start_event
                .as_ref()
                .ok_or(LlamaGenerationFailure::Terminal)?
                .elapsed_ms(
                    self.model_end_event
                        .as_ref()
                        .ok_or(LlamaGenerationFailure::Terminal)?,
                )
                .map_err(LlamaGenerationFailure::Cuda)?;

            // A cancellation that arrived while the GPU stage was in flight
            // discards those logits before D2H and, critically, before the
            // request-local sampler can consume an RNG word.
            if is_cancelled() {
                let text_delta = state.cancel().map_err(LlamaGenerationFailure::Generation)?;
                callback(LlamaGenerationEvent::Cancelled { text_delta })
                    .map_err(LlamaGenerationFailure::Callback)?;
                summary.request_wall = request_started.elapsed();
                return Ok(summary);
            }

            let download_started = Instant::now();
            self.decode
                .as_mut()
                .ok_or(LlamaGenerationFailure::Terminal)?
                .download_last_logits(&mut self.logits_bf16_native, stream)
                .map_err(|source| LlamaGenerationFailure::Decode(Box::new(source)))?;
            let logits_download_wall = download_started.elapsed();

            let sampling_started = Instant::now();
            let masked_finish_ids = state.masked_finish_token_ids();
            let constraints = if masked_finish_ids.is_empty() {
                TokenConstraints::AllowAll
            } else {
                self.allowed_tokens.fill(true);
                for &token_id in masked_finish_ids {
                    let index = usize::try_from(token_id).map_err(|_| {
                        LlamaGenerationFailure::InvalidConfiguration {
                            field: "masked_finish_token_ids",
                            reason: "validated token cannot index the vocabulary",
                        }
                    })?;
                    self.allowed_tokens[index] = false;
                }
                TokenConstraints::AllowedMask(&self.allowed_tokens)
            };
            let distribution = self
                .sampling
                .process_bf16_native(
                    &self.logits_bf16_native,
                    constraints,
                    state.history_token_ids(),
                    state.request().sampling_params,
                )
                .map_err(LlamaGenerationFailure::Sampling)?;
            let sample = distribution
                .sample(
                    state
                        .sampling_rng()
                        .map_err(LlamaGenerationFailure::Generation)?,
                )
                .map_err(LlamaGenerationFailure::Rng)?;
            let sampling_cpu = sampling_started.elapsed();

            let detokenize_started = Instant::now();
            let token_needs_decoding = state
                .token_needs_decoding(sample.token_id())
                .map_err(LlamaGenerationFailure::Generation)?;
            let decoded_byte_count = if token_needs_decoding {
                self.tokenizer
                    .decode_token_bytes_into(
                        sample.token_id(),
                        DecodeOptions {
                            skip_special_tokens: true,
                        },
                        &mut self.decoded_token_bytes,
                    )
                    .map_err(LlamaGenerationFailure::Tokenizer)?
            } else {
                0
            };
            let decoded_bytes =
                token_needs_decoding.then_some(&self.decoded_token_bytes[..decoded_byte_count]);
            let token = state
                .accept_sample(sample, decoded_bytes)
                .map_err(LlamaGenerationFailure::Generation)?;
            let finish_reason = token.finish_reason();
            let next_decode_token = token.token_id();
            let detokenize_stop_cpu = detokenize_started.elapsed();
            let timing = GenerationTokenTiming {
                model_stage,
                model_gpu_milliseconds,
                model_wall,
                logits_download_bytes: self.logits_bf16_native.len(),
                logits_download_wall,
                sampling_cpu,
                detokenize_stop_cpu,
                total_wall: token_started.elapsed(),
            };
            summary.record(timing);
            callback(LlamaGenerationEvent::Token { token, timing })
                .map_err(LlamaGenerationFailure::Callback)?;

            if finish_reason.is_some() {
                summary.request_wall = request_started.elapsed();
                return Ok(summary);
            }
            decode_token = Some(next_decode_token);
        }
    }

    fn validate_state<CallbackError>(
        &self,
        state: &GenerationState,
    ) -> Result<(), LlamaGenerationFailure<CallbackError>> {
        if self.terminal || self.decode.is_none() {
            return Err(LlamaGenerationFailure::Terminal);
        }
        state
            .request()
            .validate(self.vocabulary_size)
            .map_err(LlamaGenerationFailure::Generation)?;
        if state.request().prompt_token_ids.len() != self.prompt_length {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "prompt_token_ids",
                reason: "length differs from the prepared prompt shape",
            });
        }
        if state.request().max_new_tokens > self.output_capacity {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "max_new_tokens",
                reason: "exceeds the prepared output capacity",
            });
        }
        if state.vocabulary_size() != self.vocabulary_size {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "generation_state.vocabulary_size",
                reason: "differs from the prepared model vocabulary",
            });
        }
        if state.maximum_decoded_token_bytes() != self.decoded_token_bytes.len() {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "generation_state.maximum_decoded_token_bytes",
                reason: "differs from the prepared tokenizer bound",
            });
        }
        if !state.generated_token_ids().is_empty() || state.rng_draws() != 0 {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "generation_state",
                reason: "must be fresh and have consumed zero RNG words",
            });
        }
        match state.finish_reason() {
            None => {}
            Some(FinishReason::Length) if state.request().max_new_tokens == 0 => {}
            Some(_) => {
                return Err(LlamaGenerationFailure::InvalidConfiguration {
                    field: "generation_state",
                    reason: "must not already be terminal",
                });
            }
        }
        let decode = self
            .decode
            .as_ref()
            .ok_or(LlamaGenerationFailure::Terminal)?;
        if decode.phase() != LlamaDecodePhase::Empty || decode.logical_length() != 0 {
            return Err(LlamaGenerationFailure::InvalidConfiguration {
                field: "decode",
                reason: "must be reset before starting a request",
            });
        }
        Ok(())
    }

    fn reset_after_run(&mut self) -> CleanupFailures {
        let mut failures = CleanupFailures::default();
        let Some(decode) = self.decode.as_mut() else {
            self.terminal = true;
            return failures;
        };
        if let Err(source) = decode.reset() {
            failures.record(LlamaGenerationCleanupError::Decode(Box::new(source)));
            failures.absorb(self.close_owned_resources());
        }
        failures
    }

    fn close_owned_resources(&mut self) -> CleanupFailures {
        self.terminal = true;
        let mut failures = CleanupFailures::default();
        if let Some(decode) = self.decode.take() {
            if let Err(source) = decode.close() {
                failures.record(LlamaGenerationCleanupError::Decode(Box::new(source)));
            }
        }
        if let Some(event) = self.model_start_event.take() {
            if let Err(source) = event.close() {
                failures.record(LlamaGenerationCleanupError::Cuda(source));
            }
        }
        if let Some(event) = self.model_end_event.take() {
            if let Err(source) = event.close() {
                failures.record(LlamaGenerationCleanupError::Cuda(source));
            }
        }
        failures
    }

    /// Explicitly closes the decoder and timing events, attempting all owners.
    ///
    /// # Errors
    ///
    /// Returns the first cleanup failure and counts any additional failures.
    /// Every resource is attempted before the error is returned.
    pub fn close(mut self) -> LlamaGenerationResult<()> {
        let cleanup = self.close_owned_resources();
        if cleanup.is_empty() {
            Ok(())
        } else {
            Err(LlamaGenerationError::new(
                LlamaGenerationFailure::Cleanup,
                cleanup,
            ))
        }
    }
}

fn allocate_filled<T: Clone>(
    element_count: usize,
    value: T,
    resource: &'static str,
) -> LlamaGenerationResult<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(element_count).map_err(|_| {
        LlamaGenerationError::new(
            LlamaGenerationFailure::HostAllocation {
                resource,
                requested_elements: element_count,
            },
            CleanupFailures::default(),
        )
    })?;
    output.resize(element_count, value);
    Ok(output)
}
