//! Engine backend that serves the HF-eager-exact Qwen P2048 cache-on route.
//!
//! The arithmetic is the source-bound probe sequence qualified by the native
//! full128 and free-running gates: one single-pass 2048-token prefill followed
//! by singleton cache-on decodes over a contiguous KV cache. This backend only
//! drives that owner from the ordinary engine worker; it adds no numerical
//! path of its own. Its scope is deliberately narrow: one active request, a
//! prompt of exactly 2048 tokens, greedy sampling, and at most 128 output
//! tokens. Every other request is rejected at admission instead of being
//! served with unqualified arithmetic.

use std::collections::VecDeque;
use std::sync::Arc;

use riley_model::{DecodeOptions, EncodeOptions, LoadedModel};
use riley_runtime::generation::{
    FinishReason as RuntimeFinishReason, GenerationRequest as RuntimeGenerationRequest,
    GenerationState,
};
use riley_runtime::llama::{
    PreparedLlamaDecode, PreparedLlamaDecodeConfig, PreparedLlamaDecodeM1Trace,
    PreparedLlamaForwardConfig,
};
use riley_runtime::sampling::{SamplingParams, SamplingWorkspace, TokenConstraints};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};

use crate::domain::{
    FinishReason, GenerationEvent, GenerationRequest, ModelMetadata, RequestMetadata,
    ServiceErrorClass, TokenUsage,
};

use super::{
    BackendError, BackendEvent, EngineAllocationMetrics, EngineBackend, EngineConfig, EngineError,
    EngineMetricsSnapshot, EngineRequestId, InferenceEngine, LogitObservationSink,
    private_request_error,
};

/// Stable identifier reported for this numerics profile.
pub const HF_EXACT_PROFILE_ID: &str = "hf-eager-qwen-p2048-o128-probe-v1";

/// Prompt length the exact prefill is qualified for.
pub const HF_EXACT_PROMPT_TOKENS: usize = 2048;
/// Largest output length the exact cache-on decode is qualified for.
pub const HF_EXACT_MAX_OUTPUT_TOKENS: usize = 128;
const MAXIMUM_SEQUENCE_TOKENS: usize = HF_EXACT_PROMPT_TOKENS + HF_EXACT_MAX_OUTPUT_TOKENS - 1;
const BF16_BYTES: usize = 2;

// Budgets of the qualified native gate; the owner fails closed if they differ
// from what its plans require.
const UPLOAD_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const IO_STAGING_BYTES: u64 = 4 * 1024 * 1024;
const GEMM_WORKSPACE_CAP_BYTES: u64 = 1024 * 1024;
const REFERENCE_ATTENTION_BUDGET_BYTES: u64 = 1_342_177_280;
// The trace owner is a required operand of the exact decode entry point; the
// detailed layer matches the qualified gate.
const TRACE_DETAILED_LAYER: usize = 3;

fn internal(detail: impl Into<String>) -> BackendError {
    BackendError::new(ServiceErrorClass::Internal, detail)
}

/// Fully prepared ownership bundle transferred into the engine worker.
pub struct HfExactEngineResources {
    metadata: ModelMetadata,
    model: LoadedModel,
    context: CudaContext,
    stream: CudaStream,
    decode: PreparedLlamaDecode,
    trace: PreparedLlamaDecodeM1Trace,
    max_waiting_requests: usize,
    observation: Option<Arc<dyn LogitObservationSink>>,
}

impl std::fmt::Debug for HfExactEngineResources {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HfExactEngineResources")
            .field("metadata", &self.metadata)
            .field("logit_observation_enabled", &self.observation.is_some())
            .finish_non_exhaustive()
    }
}

impl HfExactEngineResources {
    /// Initializes CUDA and prepares the exact decode owner. No model
    /// execution occurs during this cold preparation.
    ///
    /// # Errors
    ///
    /// Returns a private diagnostic for invalid metadata, device/runtime
    /// initialization, or a model the exact owner does not admit.
    pub fn prepare(
        metadata: ModelMetadata,
        model: LoadedModel,
        device_ordinal: u32,
        max_waiting_requests: usize,
    ) -> Result<Self, BackendError> {
        metadata
            .validate()
            .map_err(|source| internal(format!("invalid model metadata: {source}")))?;
        if metadata.max_output_tokens > HF_EXACT_MAX_OUTPUT_TOKENS
            || metadata.context_window_tokens < HF_EXACT_PROMPT_TOKENS + metadata.max_output_tokens
        {
            return Err(internal(format!(
                "{HF_EXACT_PROFILE_ID} requires at most {HF_EXACT_MAX_OUTPUT_TOKENS} output tokens within the served context"
            )));
        }
        let runtime = CudaRuntime::initialize()
            .map_err(|source| internal(format!("CUDA initialization failed: {source}")))?;
        let context = runtime
            .device(device_ordinal)
            .map_err(|source| internal(format!("CUDA device selection failed: {source}")))?
            .create_context()
            .map_err(|source| internal(format!("CUDA context creation failed: {source}")))?;
        let mut stream = context
            .create_stream()
            .map_err(|source| internal(format!("CUDA stream creation failed: {source}")))?;
        let config = PreparedLlamaDecodeConfig::new(PreparedLlamaForwardConfig::new(
            UPLOAD_STAGING_BYTES,
            IO_STAGING_BYTES,
            GEMM_WORKSPACE_CAP_BYTES,
            REFERENCE_ATTENTION_BUDGET_BYTES,
        ))
        .with_hf_eager_qwen_p2048_cache_on_full128_trace_probe();
        let decode = match PreparedLlamaDecode::prepare(
            &model,
            &context,
            &mut stream,
            HF_EXACT_PROMPT_TOKENS,
            MAXIMUM_SEQUENCE_TOKENS,
            config,
        ) {
            Ok(decode) => decode,
            Err(source) => {
                let _ = stream.close();
                let _ = context.close();
                return Err(internal(format!(
                    "exact decode owner preparation failed: {source}"
                )));
            }
        };
        let trace = match decode
            .prepare_hf_eager_qwen_p2048_cache_on_m1_trace_for_layer(TRACE_DETAILED_LAYER)
        {
            Ok(trace) => trace,
            Err(source) => {
                let _ = decode.close();
                let _ = stream.close();
                let _ = context.close();
                return Err(internal(format!(
                    "exact decode trace preparation failed: {source}"
                )));
            }
        };
        Ok(Self {
            metadata,
            model,
            context,
            stream,
            decode,
            trace,
            max_waiting_requests,
            observation: None,
        })
    }

    /// Attaches a sink that receives every published logit row before its
    /// token is sampled.
    #[must_use]
    pub fn with_logit_observation(mut self, sink: Arc<dyn LogitObservationSink>) -> Self {
        self.observation = Some(sink);
        self
    }
}

impl InferenceEngine {
    /// Starts the HF-exact backend and transfers every CUDA owner to the
    /// worker thread.
    ///
    /// # Errors
    ///
    /// Returns for invalid engine configuration or backend scratch
    /// preparation.
    pub fn start_hf_exact(
        resources: HfExactEngineResources,
        config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let metadata = resources.metadata.clone();
        Self::start_with_factory(metadata, config, move || HfExactBackend::new(resources))
    }
}

struct HfExactRequest {
    engine_id: EngineRequestId,
    server_request_id: String,
    prompt_token_ids: Vec<u32>,
    state: GenerationState,
    decoded_token_bytes: Vec<u8>,
    include_token_ids: bool,
    prompt_ids_for_delivery: Option<Vec<u32>>,
}

struct HfExactBackend {
    metadata: ModelMetadata,
    model: LoadedModel,
    context: Option<CudaContext>,
    stream: Option<CudaStream>,
    decode: Option<PreparedLlamaDecode>,
    trace: PreparedLlamaDecodeM1Trace,
    observation: Option<Arc<dyn LogitObservationSink>>,
    sampling: SamplingWorkspace,
    allowed_tokens: Vec<bool>,
    addressable_tokens: usize,
    logits: Vec<u8>,
    max_waiting_requests: usize,
    waiting: VecDeque<HfExactRequest>,
    active: Option<HfExactRequest>,
    // Whether the cache holds state from a previous or cancelled request.
    cache_dirty: bool,
    final_metrics: EngineMetricsSnapshot,
}

impl HfExactBackend {
    fn new(resources: HfExactEngineResources) -> Result<Self, BackendError> {
        let HfExactEngineResources {
            metadata,
            model,
            context,
            stream,
            decode,
            trace,
            max_waiting_requests,
            observation,
        } = resources;
        let vocabulary_size = model.spec().lm_head().vocabulary_size();
        let addressable_tokens = model.tokenizer().addressable_token_count();
        if addressable_tokens == 0 || addressable_tokens > vocabulary_size {
            return Err(internal(format!(
                "tokenizer addressable bound {addressable_tokens} is outside model vocabulary {vocabulary_size}"
            )));
        }
        let sampling = SamplingWorkspace::new(vocabulary_size)
            .map_err(|source| internal(format!("sampling preparation failed: {source}")))?;
        let mut allowed_tokens = Vec::new();
        allowed_tokens
            .try_reserve_exact(vocabulary_size)
            .map_err(|_| internal("allowed-token mask allocation failed"))?;
        allowed_tokens.resize(vocabulary_size, false);
        let row_bytes = vocabulary_size
            .checked_mul(BF16_BYTES)
            .ok_or_else(|| internal("logit row extent overflowed"))?;
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(row_bytes)
            .map_err(|_| internal("logit row allocation failed"))?;
        logits.resize(row_bytes, 0);
        Ok(Self {
            metadata,
            model,
            context: Some(context),
            stream: Some(stream),
            decode: Some(decode),
            trace,
            observation,
            sampling,
            allowed_tokens,
            addressable_tokens,
            logits,
            max_waiting_requests,
            waiting: VecDeque::new(),
            active: None,
            cache_dirty: false,
            final_metrics: EngineMetricsSnapshot::default(),
        })
    }

    fn usage(request: &HfExactRequest) -> Result<TokenUsage, BackendError> {
        let prompt = u64::try_from(request.prompt_token_ids.len())
            .map_err(|_| internal("prompt token count overflowed"))?;
        let generated = u64::try_from(request.state.generated_token_ids().len())
            .map_err(|_| internal("completion token count overflowed"))?;
        TokenUsage::new(prompt, generated).map_err(|_| internal("usage arithmetic overflowed"))
    }

    fn cancellation_events(mut request: HfExactRequest) -> Result<Vec<BackendEvent>, BackendError> {
        let mut events = Vec::with_capacity(2);
        if request.state.finish_reason().is_none() {
            let delta = request
                .state
                .cancel()
                .map_err(|source| internal(format!("cancellation flush failed: {source}")))?
                .to_owned();
            // As on the scheduler route, the raw-token stream omits the
            // unattributed flush.
            if !delta.is_empty() && !request.include_token_ids {
                events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::TokenDelta { text: delta },
                ));
            }
        }
        let usage = Self::usage(&request)?;
        events.push(BackendEvent::new(
            request.engine_id,
            GenerationEvent::Finished {
                reason: FinishReason::Cancelled,
                usage,
            },
        ));
        Ok(events)
    }

    /// Runs the one model step owed to the active request and leaves its
    /// logits in `self.logits`.
    fn execute_step(&mut self) -> Result<(), BackendError> {
        let decode = self
            .decode
            .as_mut()
            .ok_or_else(|| internal("exact decode owner is closed"))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| internal("CUDA stream is closed"))?;
        let request = self
            .active
            .as_ref()
            .ok_or_else(|| internal("no active request"))?;
        match request.state.generated_token_ids().last() {
            None => {
                if self.cache_dirty {
                    decode
                        .reset()
                        .map_err(|source| internal(format!("KV reset failed: {source}")))?;
                }
                self.cache_dirty = true;
                decode
                    .prefill(&request.prompt_token_ids, stream)
                    .map_err(|source| internal(format!("exact prefill failed: {source}")))?;
            }
            Some(&previous) => {
                decode
                    .decode_hf_eager_qwen_p2048_cache_on_full128_traced(
                        previous,
                        &mut self.trace,
                        stream,
                    )
                    .map_err(|source| internal(format!("exact decode failed: {source}")))?;
            }
        }
        let expected_length =
            request.prompt_token_ids.len() + request.state.generated_token_ids().len();
        if decode.logical_length() != expected_length {
            return Err(internal(
                "committed KV length differs from request progress",
            ));
        }
        decode
            .download_last_logits(&mut self.logits, stream)
            .map_err(|source| internal(format!("logit download failed: {source}")))
    }
}

impl EngineBackend for HfExactBackend {
    fn supports_token_ids(&self) -> bool {
        true
    }

    #[allow(clippy::float_cmp)]
    fn admit(
        &mut self,
        request_id: EngineRequestId,
        metadata: &RequestMetadata,
        request: &GenerationRequest,
    ) -> Result<(), BackendError> {
        if self.decode.is_none() {
            return Err(BackendError::new(
                ServiceErrorClass::ShuttingDown,
                "exact decode owner is closed",
            ));
        }
        if request.model_id != self.metadata.model_id {
            return Err(private_request_error("request selected a different model"));
        }
        if request.max_new_tokens == 0 || request.max_new_tokens > self.metadata.max_output_tokens {
            return Err(private_request_error(
                "requested output token count is outside server bounds",
            ));
        }
        // Only greedy sampling without request stop strings is qualified.
        if request.sampling.temperature != 0.0
            || request.sampling.top_p != 1.0
            || !request.stop_sequences.is_empty()
        {
            return Err(private_request_error(
                "request sampling is outside the exact greedy profile",
            ));
        }
        if self.waiting.len() >= self.max_waiting_requests {
            return Err(BackendError::new(
                ServiceErrorClass::Overloaded,
                "exact route waiting queue is full",
            ));
        }
        let prompt_token_ids = self
            .model
            .tokenizer()
            .encode(
                &request.prompt,
                EncodeOptions {
                    add_special_tokens: true,
                },
            )
            .map_err(|_| private_request_error("prompt tokenization failed"))?;
        if prompt_token_ids.len() != HF_EXACT_PROMPT_TOKENS {
            return Err(private_request_error(
                "prompt length is outside the exact P2048 profile",
            ));
        }
        let maximum_bytes = self.model.tokenizer().maximum_decoded_token_bytes();
        let runtime_request = RuntimeGenerationRequest {
            request_id: metadata.request_id.as_bytes().to_vec(),
            seed: request.sampling.seed.unwrap_or(0),
            prompt_token_ids: prompt_token_ids.clone(),
            sampling_params: SamplingParams {
                temperature: request.sampling.temperature,
                top_k: None,
                top_p: Some(f64::from(request.sampling.top_p)),
                repetition_penalty: 1.0,
            },
            min_new_tokens: 0,
            max_new_tokens: request.max_new_tokens,
            eos_token_ids: self.model.spec().special_tokens().eos().to_vec(),
            stop_token_ids: Vec::new(),
            stop_strings: Vec::new(),
        };
        let state = GenerationState::new(
            runtime_request,
            self.sampling.vocabulary_size(),
            maximum_bytes,
        )
        .map_err(|_| private_request_error("generation request validation failed"))?;
        let mut decoded_token_bytes = Vec::new();
        decoded_token_bytes
            .try_reserve_exact(maximum_bytes)
            .map_err(|_| internal("decoded-token buffer allocation failed"))?;
        decoded_token_bytes.resize(maximum_bytes, 0);
        self.waiting.push_back(HfExactRequest {
            engine_id: request_id,
            server_request_id: metadata.request_id.clone(),
            prompt_ids_for_delivery: request.include_token_ids.then(|| prompt_token_ids.clone()),
            prompt_token_ids,
            state,
            decoded_token_bytes,
            include_token_ids: request.include_token_ids,
        });
        Ok(())
    }

    fn cancel(&mut self, request_id: EngineRequestId) -> Result<Vec<BackendEvent>, BackendError> {
        if let Some(index) = self
            .waiting
            .iter()
            .position(|request| request.engine_id == request_id)
        {
            let request = self
                .waiting
                .remove(index)
                .ok_or_else(|| internal("waiting request disappeared"))?;
            return Self::cancellation_events(request);
        }
        if self
            .active
            .as_ref()
            .is_some_and(|request| request.engine_id == request_id)
        {
            // Steps are synchronous, so no device work is outstanding here.
            // The cache is reset before the next request's prefill.
            let request = self
                .active
                .take()
                .ok_or_else(|| internal("active request disappeared"))?;
            return Self::cancellation_events(request);
        }
        Ok(Vec::new())
    }

    fn step(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
        if self.active.is_none() {
            self.active = self.waiting.pop_front();
        }
        if self.active.is_none() {
            return Ok(Vec::new());
        }
        self.execute_step()?;
        let request = self
            .active
            .as_mut()
            .ok_or_else(|| internal("no active request"))?;
        let generated_index = request.state.generated_token_ids().len();
        if let Some(observation) = self.observation.as_ref() {
            observation
                .observe(&request.server_request_id, generated_index, &self.logits)
                .map_err(|error| internal(format!("logit observation failed: {error}")))?;
        }
        self.allowed_tokens.fill(false);
        self.allowed_tokens[..self.addressable_tokens].fill(true);
        for &masked_id in request.state.masked_finish_token_ids() {
            let index = usize::try_from(masked_id)
                .map_err(|_| internal("masked token ID cannot index vocabulary"))?;
            *self
                .allowed_tokens
                .get_mut(index)
                .ok_or_else(|| internal("masked token ID is outside vocabulary"))? = false;
        }
        let distribution = self
            .sampling
            .process_bf16_native(
                &self.logits,
                TokenConstraints::AllowedMask(&self.allowed_tokens),
                request.state.history_token_ids(),
                request.state.request().sampling_params,
            )
            .map_err(|source| internal(format!("sampling failed: {source}")))?;
        let sample = distribution
            .sample(
                request
                    .state
                    .sampling_rng()
                    .map_err(|source| internal(format!("sampling RNG failed: {source}")))?,
            )
            .map_err(|source| internal(format!("sampling RNG failed: {source}")))?;
        let token_id = sample.token_id();
        let needs_decoding = request
            .state
            .token_needs_decoding(token_id)
            .map_err(|source| internal(format!("token acceptance failed: {source}")))?;
        let decoded_count = if needs_decoding {
            self.model
                .tokenizer()
                .decode_token_bytes_into(
                    token_id,
                    DecodeOptions {
                        skip_special_tokens: true,
                    },
                    &mut request.decoded_token_bytes,
                )
                .map_err(|source| internal(format!("token decoding failed: {source}")))?
        } else {
            0
        };
        let decoded = needs_decoding.then_some(&request.decoded_token_bytes[..decoded_count]);
        let generated = request
            .state
            .accept_token(token_id, sample.token_logprob(), decoded)
            .map_err(|source| internal(format!("token acceptance failed: {source}")))?;
        let text = generated.text_delta().to_owned();
        let finish = generated.finish_reason();
        let mut events = Vec::with_capacity(2);
        let event = if request.include_token_ids {
            GenerationEvent::CommittedToken {
                text,
                token_id,
                generated_index,
                prompt_token_ids: request.prompt_ids_for_delivery.take(),
            }
        } else {
            GenerationEvent::TokenDelta { text }
        };
        events.push(BackendEvent::new(request.engine_id, event));
        if let Some(finish) = finish {
            let reason = match finish {
                RuntimeFinishReason::Length => FinishReason::Length,
                RuntimeFinishReason::Eos
                | RuntimeFinishReason::StopToken
                | RuntimeFinishReason::StopString => FinishReason::Stop,
                _ => {
                    return Err(internal("accepted token produced a non-generation finish"));
                }
            };
            let usage = Self::usage(request)?;
            events.push(BackendEvent::new(
                request.engine_id,
                GenerationEvent::Finished { reason, usage },
            ));
            self.active = None;
        }
        Ok(events)
    }

    fn is_idle(&self) -> bool {
        self.active.is_none() && self.waiting.is_empty()
    }

    fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, BackendError> {
        let allocation = self
            .context
            .as_ref()
            .ok_or_else(|| internal("CUDA context is already closed"))?
            .allocation_stats()
            .map_err(|source| internal(format!("allocation metrics failed: {source}")))?;
        Ok(EngineMetricsSnapshot {
            scheduler: None,
            batch_shapes: None,
            allocation: Some(EngineAllocationMetrics {
                device_live_bytes: allocation.device_live_bytes(),
                device_live_allocations: allocation.device_live_allocations(),
                pinned_host_live_bytes: allocation.pinned_host_live_bytes(),
                pinned_host_live_allocations: allocation.pinned_host_live_allocations(),
            }),
        })
    }

    fn final_metrics_snapshot(&self) -> EngineMetricsSnapshot {
        self.final_metrics
    }

    fn shutdown(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
        if self.decode.is_none() && self.stream.is_none() && self.context.is_none() {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        if let Some(request) = self.active.take() {
            events.extend(Self::cancellation_events(request)?);
        }
        while let Some(request) = self.waiting.pop_front() {
            events.extend(Self::cancellation_events(request)?);
        }
        let mut first_error = None;
        if let Some(decode) = self.decode.take() {
            if let Err(source) = decode.close() {
                first_error = Some(format!("exact decode owner close failed: {source}"));
            }
        }
        if let Some(mut stream) = self.stream.take() {
            if let Err(source) = stream.synchronize() {
                first_error.get_or_insert_with(|| format!("stream synchronize failed: {source}"));
            }
            if let Err(source) = stream.close() {
                first_error.get_or_insert_with(|| format!("stream close failed: {source}"));
            }
        }
        let mut final_allocation = None;
        if first_error.is_none() {
            match self.context.as_ref().map(CudaContext::allocation_stats) {
                Some(Ok(allocation)) => {
                    final_allocation = Some(EngineAllocationMetrics {
                        device_live_bytes: allocation.device_live_bytes(),
                        device_live_allocations: allocation.device_live_allocations(),
                        pinned_host_live_bytes: allocation.pinned_host_live_bytes(),
                        pinned_host_live_allocations: allocation.pinned_host_live_allocations(),
                    });
                }
                Some(Err(source)) => {
                    first_error = Some(format!(
                        "final allocation metrics failed before context close: {source}"
                    ));
                }
                None => first_error = Some("CUDA context is already closed".to_owned()),
            }
        }
        if let Some(context) = self.context.take() {
            if let Err(source) = context.close() {
                first_error.get_or_insert_with(|| format!("context close failed: {source}"));
            }
        }
        match first_error {
            Some(detail) => Err(internal(detail)),
            None => {
                self.final_metrics = EngineMetricsSnapshot {
                    scheduler: None,
                    batch_shapes: None,
                    allocation: final_allocation,
                };
                Ok(events)
            }
        }
    }
}
