//! Engine backend that serves the HF-eager-exact Qwen P2048 cache-on route.
//!
//! The arithmetic is the source-bound probe sequence qualified by the native
//! full128 and free-running gates: one single-pass 2048-token prefill followed
//! by singleton cache-on decodes over a contiguous KV cache. This backend adds
//! no numerical path of its own: the ordinary scheduler admits requests,
//! plans every iteration, reserves and commits KV blocks, settles
//! cancellation and reclaims ownership, and the exact owner executes each
//! planned iteration. With one active sequence the scheduler's block pool is
//! the ownership ledger of the owner's single contiguous cache. The scope is
//! deliberately narrow: one active request, a prompt of exactly 2048 tokens,
//! greedy sampling, and at most 128 output tokens. Every other request is
//! rejected at admission instead of being served with unqualified arithmetic.

use std::sync::Arc;
use std::time::Instant;

use riley_model::{DecodeOptions, EncodeOptions, LoadedModel};
use riley_runtime::generation::{
    FinishReason as RuntimeFinishReason, GenerationRequest as RuntimeGenerationRequest,
    GenerationState,
};
use riley_runtime::llama::{
    PreparedLlamaDecode, PreparedLlamaDecodeConfig, PreparedLlamaDecodeM1Trace,
    PreparedLlamaForwardConfig,
};
use riley_runtime::paged_kv::{KV_BLOCK_SIZE, KvLayout};
use riley_runtime::sampling::{SamplingParams, SamplingWorkspace, TokenConstraints};
use riley_runtime::{CudaContext, CudaRuntime, CudaStream};
use riley_scheduler::{
    ExecutionAbort, IterationOutput, IterationPlan, IterationResult, OverloadPolicy,
    RequestCompletion, RequestDescriptor, RequestFinishReason, RequestId as SchedulerRequestId,
    Scheduler, SchedulerConfig, SchedulerError, WorkKind,
};

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
// Qualified Qwen2.5-3B cache geometry; the owner rejects any other model.
const KV_LAYERS: usize = 36;
const KV_HEADS: usize = 2;
const KV_HEAD_DIMENSION: usize = 128;
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
    scheduler: Scheduler,
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
        let sequence_tokens = HF_EXACT_PROMPT_TOKENS + HF_EXACT_MAX_OUTPUT_TOKENS;
        let blocks = sequence_tokens.div_ceil(KV_BLOCK_SIZE);
        let scheduler = KvLayout::checked(KV_LAYERS, blocks, KV_HEADS, KV_HEAD_DIMENSION)
            .map_err(|source| format!("KV ledger layout failed: {source}"))
            .and_then(|layout| {
                Scheduler::new(
                    SchedulerConfig {
                        max_waiting_requests,
                        max_waiting_prompt_tokens: max_waiting_requests
                            .saturating_mul(HF_EXACT_PROMPT_TOKENS),
                        max_active_sequences: 1,
                        max_sequence_tokens: sequence_tokens,
                        iteration_token_budget: HF_EXACT_PROMPT_TOKENS,
                        max_prefill_chunk_tokens: HF_EXACT_PROMPT_TOKENS,
                        aging_threshold_ns: 100_000_000,
                        overload_policy: OverloadPolicy::Wait,
                        admission_timeout_ns: Some(30_000_000_000),
                        max_promised_kv_blocks: blocks,
                        metrics_window_samples: 1_024,
                    },
                    layout,
                )
                .map_err(|source| format!("scheduler preparation failed: {source}"))
            });
        let scheduler = match scheduler {
            Ok(scheduler) => scheduler,
            Err(detail) => {
                let _ = decode.close();
                let _ = stream.close();
                let _ = context.close();
                return Err(internal(detail));
            }
        };
        Ok(Self {
            metadata,
            model,
            context,
            stream,
            decode,
            trace,
            scheduler,
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
    scheduler_id: SchedulerRequestId,
    server_request_id: String,
    prompt_tokens: u64,
    state: GenerationState,
    decoded_token_bytes: Vec<u8>,
    include_token_ids: bool,
    prompt_ids_for_delivery: Option<Vec<u32>>,
    next_delivery_index: usize,
}

struct HfExactBackend {
    metadata: ModelMetadata,
    model: LoadedModel,
    scheduler: Option<Scheduler>,
    context: Option<CudaContext>,
    stream: Option<CudaStream>,
    decode: Option<PreparedLlamaDecode>,
    trace: PreparedLlamaDecodeM1Trace,
    observation: Option<Arc<dyn LogitObservationSink>>,
    sampling: SamplingWorkspace,
    allowed_tokens: Vec<bool>,
    addressable_tokens: usize,
    logits: Vec<u8>,
    requests: Vec<HfExactRequest>,
    // Whether the contiguous cache holds state from an earlier prefill.
    cache_dirty: bool,
    started: Instant,
    final_metrics: EngineMetricsSnapshot,
}

fn admission_error(source: &SchedulerError) -> BackendError {
    let class = match source {
        SchedulerError::WaitingQueueFull { .. }
        | SchedulerError::WaitingTokenLimit { .. }
        | SchedulerError::ActiveSequenceLimit { .. }
        | SchedulerError::KvCapacityExceeded { .. } => ServiceErrorClass::Overloaded,
        SchedulerError::SchedulerClosed => ServiceErrorClass::ShuttingDown,
        SchedulerError::SequenceTokenLimit { .. } => ServiceErrorClass::InvalidRequest,
        _ => ServiceErrorClass::Internal,
    };
    BackendError::new(class, format!("scheduler admission failed: {source}"))
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
            scheduler,
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
            scheduler: Some(scheduler),
            context: Some(context),
            stream: Some(stream),
            decode: Some(decode),
            trace,
            observation,
            sampling,
            allowed_tokens,
            addressable_tokens,
            logits,
            requests: Vec::new(),
            cache_dirty: false,
            started: Instant::now(),
            final_metrics: EngineMetricsSnapshot::default(),
        })
    }

    fn now_ns(&self) -> u64 {
        u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    fn scheduler_mut(&mut self) -> Result<&mut Scheduler, BackendError> {
        self.scheduler
            .as_mut()
            .ok_or_else(|| internal("scheduler is already closed"))
    }

    fn request_index(&self, scheduler_id: SchedulerRequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|request| request.scheduler_id == scheduler_id)
    }

    /// Converts one scheduler terminal into ordered request events.
    fn process_completion(
        &mut self,
        completion: &RequestCompletion,
    ) -> Result<Vec<BackendEvent>, BackendError> {
        let Some(index) = self.request_index(completion.request_id()) else {
            return Ok(Vec::new());
        };
        let mut request = self.requests.swap_remove(index);
        let mut events = Vec::with_capacity(2);
        if completion.reason() == RequestFinishReason::Cancelled
            && request.state.finish_reason().is_none()
        {
            let delta = request
                .state
                .cancel()
                .map_err(|source| internal(format!("cancellation flush failed: {source}")))?
                .to_owned();
            // As on the batch-executor route, the raw-token stream omits the
            // unattributed flush.
            if !delta.is_empty() && !request.include_token_ids {
                events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::TokenDelta { text: delta },
                ));
            }
        }
        let reason = match completion.reason() {
            RequestFinishReason::Length => FinishReason::Length,
            RequestFinishReason::Stop => FinishReason::Stop,
            RequestFinishReason::Cancelled => FinishReason::Cancelled,
            RequestFinishReason::AdmissionTimeout => {
                events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::Failed {
                        class: ServiceErrorClass::Timeout,
                    },
                ));
                return Ok(events);
            }
            RequestFinishReason::ExecutorFailure => {
                events.push(BackendEvent::new(
                    request.engine_id,
                    GenerationEvent::Failed {
                        class: ServiceErrorClass::Internal,
                    },
                ));
                return Ok(events);
            }
        };
        if matches!(reason, FinishReason::Length | FinishReason::Stop)
            && request.include_token_ids
            && (request.next_delivery_index != completion.generated_token_ids().len()
                || request.prompt_ids_for_delivery.is_some())
        {
            return Err(internal("terminal committed token delivery count differs"));
        }
        let usage = TokenUsage::new(
            request.prompt_tokens,
            u64::try_from(completion.generated_token_ids().len())
                .map_err(|_| internal("completion token count overflowed"))?,
        )
        .map_err(|_| internal("usage arithmetic overflowed"))?;
        events.push(BackendEvent::new(
            request.engine_id,
            GenerationEvent::Finished { reason, usage },
        ));
        Ok(events)
    }

    fn process_completions(
        &mut self,
        completions: &[RequestCompletion],
    ) -> Result<Vec<BackendEvent>, BackendError> {
        let mut events = Vec::new();
        for completion in completions {
            events.extend(self.process_completion(completion)?);
        }
        Ok(events)
    }

    /// Executes the single planned work item on the exact owner and leaves
    /// its logits in `self.logits`. A plan of any other shape is refused.
    fn execute_plan(&mut self, plan: &IterationPlan) -> Result<(), BackendError> {
        let decode = self
            .decode
            .as_mut()
            .ok_or_else(|| internal("exact decode owner is closed"))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| internal("CUDA stream is closed"))?;
        let item = match (plan.prefill_items(), plan.decode_items()) {
            ([item], []) | ([], [item]) => item,
            _ => return Err(internal("exact route requires one planned work item")),
        };
        if item.output_slot().is_none() {
            return Err(internal("exact route requires a published output row"));
        }
        match item.kind() {
            WorkKind::Prefill => {
                if item.input_tokens().len() != HF_EXACT_PROMPT_TOKENS
                    || item.target_logical_length() != HF_EXACT_PROMPT_TOKENS
                {
                    return Err(internal("exact route requires a single-pass P2048 prefill"));
                }
                if self.cache_dirty {
                    decode
                        .reset()
                        .map_err(|source| internal(format!("KV reset failed: {source}")))?;
                }
                self.cache_dirty = true;
                decode
                    .prefill(item.input_tokens(), stream)
                    .map_err(|source| internal(format!("exact prefill failed: {source}")))?;
            }
            WorkKind::Decode => {
                let &[token] = item.input_tokens() else {
                    return Err(internal("exact route requires singleton decode"));
                };
                if item.target_logical_length() != decode.logical_length() + 1 {
                    return Err(internal("planned decode does not extend the owned cache"));
                }
                decode
                    .decode_hf_eager_qwen_p2048_cache_on_full128_traced(
                        token,
                        &mut self.trace,
                        stream,
                    )
                    .map_err(|source| internal(format!("exact decode failed: {source}")))?;
            }
        }
        if decode.logical_length() != item.target_logical_length() {
            return Err(internal(
                "owned cache length differs from the planned target",
            ));
        }
        decode
            .download_last_logits(&mut self.logits, stream)
            .map_err(|source| internal(format!("logit download failed: {source}")))
    }

    /// Samples the planned row and returns its token, stop flag and text.
    fn sample(&mut self, index: usize) -> Result<(u32, bool, String), BackendError> {
        let request = &mut self.requests[index];
        let generated_index = request.state.generated_token_ids().len();
        if let Some(observation) = self.observation.as_ref() {
            observation
                .observe(&request.server_request_id, generated_index, &self.logits)
                .map_err(|error| internal(format!("logit observation failed: {error}")))?;
        }
        self.allowed_tokens.fill(false);
        self.allowed_tokens[..self.addressable_tokens].fill(true);
        for &masked_id in request.state.masked_finish_token_ids() {
            let masked = usize::try_from(masked_id)
                .map_err(|_| internal("masked token ID cannot index vocabulary"))?;
            *self
                .allowed_tokens
                .get_mut(masked)
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
        let stop = matches!(
            generated.finish_reason(),
            Some(
                RuntimeFinishReason::Eos
                    | RuntimeFinishReason::StopToken
                    | RuntimeFinishReason::StopString
            )
        );
        Ok((token_id, stop, generated.text_delta().to_owned()))
    }

    fn allocation(&self) -> Result<EngineAllocationMetrics, BackendError> {
        let allocation = self
            .context
            .as_ref()
            .ok_or_else(|| internal("CUDA context is already closed"))?
            .allocation_stats()
            .map_err(|source| internal(format!("allocation metrics failed: {source}")))?;
        Ok(EngineAllocationMetrics {
            device_live_bytes: allocation.device_live_bytes(),
            device_live_allocations: allocation.device_live_allocations(),
            pinned_host_live_bytes: allocation.pinned_host_live_bytes(),
            pinned_host_live_allocations: allocation.pinned_host_live_allocations(),
        })
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
        let prompt_tokens = u64::try_from(prompt_token_ids.len())
            .map_err(|_| internal("prompt token count overflowed"))?;
        let prompt_ids_for_delivery = request.include_token_ids.then(|| prompt_token_ids.clone());
        let now_ns = self.now_ns();
        let submission = self
            .scheduler_mut()?
            .submit(
                RequestDescriptor::new(prompt_token_ids, request.max_new_tokens),
                now_ns,
            )
            .map_err(|source| admission_error(&source))?;
        self.requests.push(HfExactRequest {
            engine_id: request_id,
            scheduler_id: submission.request_id(),
            server_request_id: metadata.request_id.clone(),
            prompt_tokens,
            state,
            decoded_token_bytes,
            include_token_ids: request.include_token_ids,
            prompt_ids_for_delivery,
            next_delivery_index: 0,
        });
        Ok(())
    }

    fn cancel(&mut self, request_id: EngineRequestId) -> Result<Vec<BackendEvent>, BackendError> {
        let Some(index) = self
            .requests
            .iter()
            .position(|request| request.engine_id == request_id)
        else {
            return Ok(Vec::new());
        };
        let scheduler_id = self.requests[index].scheduler_id;
        let now_ns = self.now_ns();
        let outcome = self
            .scheduler_mut()?
            .cancel(scheduler_id, now_ns)
            .map_err(|source| internal(format!("scheduler cancellation failed: {source}")))?;
        // Every step settles its iteration before returning, so no iteration
        // is outstanding when a cancellation is observed.
        if outcome.deferred_until_iteration_settles() {
            return Err(internal(
                "cancellation deferred without an outstanding iteration",
            ));
        }
        match outcome.completion().cloned() {
            Some(completion) => self.process_completion(&completion),
            None => Ok(Vec::new()),
        }
    }

    fn step(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
        let now_ns = self.now_ns();
        let planning = self
            .scheduler_mut()?
            .plan_iteration(now_ns)
            .map_err(|source| internal(format!("iteration planning failed: {source}")))?;
        let (plan, completions) = planning.into_parts();
        let mut events = self.process_completions(&completions)?;
        let Some(plan) = plan else {
            return Ok(events);
        };
        let executed = self.execute_plan(&plan).and_then(|()| {
            let item = plan
                .prefill_items()
                .iter()
                .chain(plan.decode_items())
                .next()
                .ok_or_else(|| internal("plan has no work item"))?;
            let slot = item
                .output_slot()
                .ok_or_else(|| internal("plan has no output slot"))?;
            let index = self
                .request_index(item.request_id())
                .ok_or_else(|| internal("iteration references unknown request"))?;
            let (token_id, stop, text) = self.sample(index)?;
            Ok((slot, index, token_id, stop, text))
        });
        let (slot, index, token_id, stop, text) = match executed {
            Ok(executed) => executed,
            Err(error) => {
                // Owner calls are synchronous, so the device is quiescent; the
                // scheduler reclaims the reservation before the worker fails.
                let now_ns = self.now_ns();
                let _ = self.scheduler_mut()?.abort_iteration(
                    plan.iteration_id(),
                    ExecutionAbort::DeviceQuiescedMutationUnknown,
                    now_ns,
                );
                return Err(error);
            }
        };
        let result = IterationResult::new(
            plan.iteration_id(),
            vec![IterationOutput::new(slot, token_id, stop)],
            0,
            0,
        )
        .map_err(|source| internal(format!("iteration result construction failed: {source}")))?;
        let now_ns = self.now_ns();
        let updates = self
            .scheduler_mut()?
            .complete_iteration(&result, now_ns)
            .map_err(|source| internal(format!("scheduler commit failed: {source}")))?;
        if !updates.settlement_failures().is_empty() {
            return Err(internal("scheduler settlement failed"));
        }
        // External publication starts only after the authoritative commit.
        let [token] = updates.token_events() else {
            return Err(internal("commit did not publish exactly one token"));
        };
        let request = &mut self.requests[index];
        if token.request_id() != request.scheduler_id || token.token_id() != token_id {
            return Err(internal("committed token differs from staged sample"));
        }
        let event = if request.include_token_ids {
            if token.generated_index() != request.next_delivery_index
                || (token.generated_index() == 0) != request.prompt_ids_for_delivery.is_some()
            {
                return Err(internal(
                    "committed token delivery metadata is inconsistent",
                ));
            }
            request.next_delivery_index += 1;
            GenerationEvent::CommittedToken {
                text,
                token_id,
                generated_index: token.generated_index(),
                prompt_token_ids: request.prompt_ids_for_delivery.take(),
            }
        } else {
            GenerationEvent::TokenDelta { text }
        };
        events.push(BackendEvent::new(request.engine_id, event));
        events.extend(self.process_completions(updates.completions())?);
        Ok(events)
    }

    fn is_idle(&self) -> bool {
        self.requests.is_empty()
    }

    fn metrics_snapshot(&self) -> Result<EngineMetricsSnapshot, BackendError> {
        let scheduler = self
            .scheduler
            .as_ref()
            .ok_or_else(|| internal("scheduler is already closed"))?
            .metrics_snapshot()
            .map_err(|source| internal(format!("scheduler metrics failed: {source}")))?;
        Ok(EngineMetricsSnapshot {
            scheduler: Some(scheduler),
            batch_shapes: None,
            allocation: Some(self.allocation()?),
        })
    }

    fn final_metrics_snapshot(&self) -> EngineMetricsSnapshot {
        self.final_metrics
    }

    fn shutdown(&mut self) -> Result<Vec<BackendEvent>, BackendError> {
        if self.scheduler.is_none()
            && self.decode.is_none()
            && self.stream.is_none()
            && self.context.is_none()
        {
            return Ok(Vec::new());
        }
        if let Some(scheduler) = self.scheduler.as_mut() {
            scheduler.begin_shutdown();
        }
        let request_ids: Vec<_> = self
            .requests
            .iter()
            .map(|request| request.engine_id)
            .collect();
        let mut events = Vec::new();
        for request_id in request_ids {
            events.extend(self.cancel(request_id)?);
        }
        let mut first_error = None;
        let mut final_scheduler = None;
        if let Some(scheduler) = self.scheduler.take() {
            match scheduler.close(self.now_ns(), None) {
                Ok(output) => final_scheduler = Some(output.final_metrics()),
                Err(source) => first_error = Some(format!("scheduler close failed: {source}")),
            }
        }
        if let Some(decode) = self.decode.take() {
            if let Err(source) = decode.close() {
                first_error
                    .get_or_insert_with(|| format!("exact decode owner close failed: {source}"));
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
            match self.allocation() {
                Ok(allocation) => final_allocation = Some(allocation),
                Err(error) => {
                    first_error = Some(format!(
                        "final allocation metrics failed before context close: {}",
                        error.detail()
                    ));
                }
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
                    scheduler: final_scheduler,
                    batch_shapes: None,
                    allocation: final_allocation,
                };
                Ok(events)
            }
        }
    }
}
