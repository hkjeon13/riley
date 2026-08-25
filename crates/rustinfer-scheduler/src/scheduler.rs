//! Deterministic admission, KV ownership, and continuous-iteration planning.
//!
//! The scheduler never launches a CUDA kernel. It owns bounded host metadata
//! and paged-KV reservations, emits immutable executor-neutral plans, and only
//! publishes state after versioned runtime feedback has been validated in full.

use std::collections::VecDeque;
use std::time::Instant;

use rustinfer_runtime::paged_kv::{
    KV_BLOCK_SIZE, KvBlockPool, KvBlockPoolStats, KvLayout, SequenceReservation, SequenceState,
};

use crate::config::{OverloadPolicy, SchedulerConfig};
use crate::error::{SchedulerError, SchedulerResult};
use crate::metrics::{
    IterationMetricSample, SchedulerGauges, SchedulerMetrics, SchedulerMetricsSnapshot,
};
use crate::plan::{
    IterationId, IterationOutput, IterationPlan, IterationResult, OutputSlot, OwnedBlockTable,
    RequestId, WorkItem, WorkKind,
};

/// Model-token input and bounded output capacity for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestDescriptor {
    prompt_token_ids: Vec<u32>,
    max_new_tokens: usize,
}

impl RequestDescriptor {
    /// Creates a scheduler request. Authoritative validation occurs in
    /// [`Scheduler::submit`].
    #[must_use]
    pub fn new(prompt_token_ids: Vec<u32>, max_new_tokens: usize) -> Self {
        Self {
            prompt_token_ids,
            max_new_tokens,
        }
    }

    /// Complete prompt token sequence retained until prefill finishes.
    #[must_use]
    pub fn prompt_token_ids(&self) -> &[u32] {
        &self.prompt_token_ids
    }

    /// Maximum number of runtime outputs accepted for the request.
    #[must_use]
    pub const fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }
}

/// Public request state. All transitions are checked in this module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestState {
    Waiting,
    Admitted,
    Prefilling,
    Decoding,
    Finished,
    Cancelled,
    Failed,
}

impl RequestState {
    const fn name(self) -> &'static str {
        match self {
            Self::Waiting => "Waiting",
            Self::Admitted => "Admitted",
            Self::Prefilling => "Prefilling",
            Self::Decoding => "Decoding",
            Self::Finished => "Finished",
            Self::Cancelled => "Cancelled",
            Self::Failed => "Failed",
        }
    }

    const fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

/// Why a request left the bounded scheduler state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestFinishReason {
    Length,
    Stop,
    Cancelled,
    AdmissionTimeout,
    ExecutorFailure,
}

impl RequestFinishReason {
    const fn state(self) -> RequestState {
        match self {
            Self::Length | Self::Stop => RequestState::Finished,
            Self::Cancelled => RequestState::Cancelled,
            Self::AdmissionTimeout | Self::ExecutorFailure => RequestState::Failed,
        }
    }
}

/// Terminal request payload returned directly to the caller.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestCompletion {
    request_id: RequestId,
    reason: RequestFinishReason,
    generated_token_ids: Vec<u32>,
    completed_at_ns: u64,
}

impl RequestCompletion {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn reason(&self) -> RequestFinishReason {
        self.reason
    }

    #[must_use]
    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    #[must_use]
    pub const fn completed_at_ns(&self) -> u64 {
        self.completed_at_ns
    }
}

/// One accepted runtime token routed back to its request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenEvent {
    request_id: RequestId,
    token_id: u32,
    generated_index: usize,
}

impl TokenEvent {
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn token_id(self) -> u32 {
        self.token_id
    }

    #[must_use]
    pub const fn generated_index(self) -> usize {
        self.generated_index
    }
}

/// Immediate result of a successful submission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Submission {
    request_id: RequestId,
    state: RequestState,
}

impl Submission {
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn state(self) -> RequestState {
        self.state
    }
}

/// Immutable diagnostic view that contains no prompt or generated token data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestSnapshot {
    request_id: RequestId,
    state: RequestState,
    prompt_tokens: usize,
    prefetched_tokens: usize,
    generated_tokens: usize,
    max_new_tokens: usize,
    promised_kv_blocks: usize,
    logical_kv_tokens: usize,
    cancellation_deferred: bool,
    retained_prompt_capacity_tokens: usize,
}

impl RequestSnapshot {
    #[must_use]
    pub const fn request_id(self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn state(self) -> RequestState {
        self.state
    }

    #[must_use]
    pub const fn prompt_tokens(self) -> usize {
        self.prompt_tokens
    }

    #[must_use]
    pub const fn prefetched_tokens(self) -> usize {
        self.prefetched_tokens
    }

    #[must_use]
    pub const fn generated_tokens(self) -> usize {
        self.generated_tokens
    }

    #[must_use]
    pub const fn max_new_tokens(self) -> usize {
        self.max_new_tokens
    }

    #[must_use]
    pub const fn promised_kv_blocks(self) -> usize {
        self.promised_kv_blocks
    }

    #[must_use]
    pub const fn logical_kv_tokens(self) -> usize {
        self.logical_kv_tokens
    }

    #[must_use]
    pub const fn cancellation_deferred(self) -> bool {
        self.cancellation_deferred
    }

    /// Capacity of the scheduler-owned prompt allocation, in token elements.
    /// This diagnostic makes the bounded-copy admission contract observable.
    #[must_use]
    pub const fn retained_prompt_capacity_tokens(self) -> usize {
        self.retained_prompt_capacity_tokens
    }
}

/// Planning return value, including admission timeouts discovered by the tick.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanningOutput {
    plan: Option<IterationPlan>,
    completions: Vec<RequestCompletion>,
}

impl PlanningOutput {
    #[must_use]
    pub const fn plan(&self) -> Option<&IterationPlan> {
        self.plan.as_ref()
    }

    #[must_use]
    pub fn completions(&self) -> &[RequestCompletion] {
        &self.completions
    }

    #[must_use]
    pub fn into_parts(self) -> (Option<IterationPlan>, Vec<RequestCompletion>) {
        (self.plan, self.completions)
    }
}

/// Tokens and terminal requests produced when an iteration settles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IterationUpdates {
    token_events: Vec<TokenEvent>,
    completions: Vec<RequestCompletion>,
    settlement_failures: Vec<RequestSettlementFailure>,
}

impl IterationUpdates {
    fn empty() -> Self {
        Self {
            token_events: Vec::new(),
            completions: Vec::new(),
            settlement_failures: Vec::new(),
        }
    }

    #[must_use]
    pub fn token_events(&self) -> &[TokenEvent] {
        &self.token_events
    }

    #[must_use]
    pub fn completions(&self) -> &[RequestCompletion] {
        &self.completions
    }

    /// Contained host ownership failures discovered after the immutable plan
    /// had already been taken for settlement. Every affected request is failed
    /// and reclaimed before this list is returned.
    #[must_use]
    pub fn settlement_failures(&self) -> &[RequestSettlementFailure] {
        &self.settlement_failures
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Vec<TokenEvent>,
        Vec<RequestCompletion>,
        Vec<RequestSettlementFailure>,
    ) {
        (
            self.token_events,
            self.completions,
            self.settlement_failures,
        )
    }
}

/// One request-scoped ownership failure contained while settling an iteration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestSettlementFailure {
    request_id: RequestId,
    error: SchedulerError,
}

/// All terminal notifications and contained request failures from a consuming close.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SchedulerCloseOutput {
    completions: Vec<RequestCompletion>,
    settlement_failures: Vec<RequestSettlementFailure>,
}

impl SchedulerCloseOutput {
    #[must_use]
    pub fn completions(&self) -> &[RequestCompletion] {
        &self.completions
    }

    #[must_use]
    pub fn settlement_failures(&self) -> &[RequestSettlementFailure] {
        &self.settlement_failures
    }

    #[must_use]
    pub fn into_parts(self) -> (Vec<RequestCompletion>, Vec<RequestSettlementFailure>) {
        (self.completions, self.settlement_failures)
    }
}

impl RequestSettlementFailure {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn error(&self) -> &SchedulerError {
        &self.error
    }
}

/// Whether a failed execution touched the device-visible reserved KV ranges.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExecutionAbort {
    /// The plan was never dispatched; every reservation can be rolled back.
    NotDispatched,
    /// Device mutation may have occurred, but the caller has synchronized the
    /// stream and guarantees that no kernel can still access the plan.
    DeviceQuiescedMutationUnknown,
}

/// Result of a cancellation request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CancellationOutcome {
    request_id: RequestId,
    deferred_until_iteration_settles: bool,
    already_terminal: bool,
    completion: Option<RequestCompletion>,
}

impl CancellationOutcome {
    #[must_use]
    pub const fn request_id(&self) -> RequestId {
        self.request_id
    }

    #[must_use]
    pub const fn deferred_until_iteration_settles(&self) -> bool {
        self.deferred_until_iteration_settles
    }

    #[must_use]
    pub const fn already_terminal(&self) -> bool {
        self.already_terminal
    }

    #[must_use]
    pub const fn completion(&self) -> Option<&RequestCompletion> {
        self.completion.as_ref()
    }
}

#[derive(Debug)]
struct RequestRecord {
    request_id: RequestId,
    descriptor: RequestDescriptor,
    state: RequestState,
    submitted_at_ns: u64,
    ready_since_ns: u64,
    prefill_cursor: usize,
    generated_token_ids: Vec<u32>,
    promised_kv_blocks: usize,
    sequence: Option<SequenceState>,
    cancellation_deferred: bool,
}

impl RequestRecord {
    fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            request_id: self.request_id,
            state: self.state,
            prompt_tokens: self.descriptor.prompt_token_ids.len(),
            prefetched_tokens: self.prefill_cursor,
            generated_tokens: self.generated_token_ids.len(),
            max_new_tokens: self.descriptor.max_new_tokens,
            promised_kv_blocks: self.promised_kv_blocks,
            logical_kv_tokens: self
                .sequence
                .as_ref()
                .map_or(0, |sequence| sequence.logical_length() as usize),
            cancellation_deferred: self.cancellation_deferred,
            retained_prompt_capacity_tokens: self.descriptor.prompt_token_ids.capacity(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TerminalTombstone {
    request_id: RequestId,
    state: RequestState,
}

#[derive(Debug)]
struct InflightItem {
    request_id: RequestId,
    kind: WorkKind,
    target_logical_length: usize,
    output_slot: Option<OutputSlot>,
    previous_state: RequestState,
    previous_ready_since_ns: u64,
    reservation: SequenceReservation,
}

#[derive(Debug)]
struct InflightPlan {
    iteration_id: IterationId,
    prefill_tokens: usize,
    decode_tokens: usize,
    prefill_count: usize,
    decode_count: usize,
    scheduler_cpu_ns: u64,
    expected_output_slots: Vec<OutputSlot>,
    items: Vec<InflightItem>,
}

#[derive(Clone, Copy, Debug)]
struct SettledInflightItem {
    request_id: RequestId,
    kind: WorkKind,
    target_logical_length: usize,
    output: Option<IterationOutput>,
}

/// Bounded, deterministic scheduler and sole owner of host paged-KV state.
#[must_use = "settle every in-flight plan and close the scheduler before discarding it"]
pub struct Scheduler {
    config: SchedulerConfig,
    pool: KvBlockPool,
    requests: Vec<RequestRecord>,
    waiting: VecDeque<RequestId>,
    terminal: VecDeque<TerminalTombstone>,
    terminal_capacity: usize,
    completion_outbox: VecDeque<RequestCompletion>,
    completion_outbox_capacity: usize,
    active_sequences: usize,
    waiting_prompt_tokens: usize,
    promised_kv_blocks: usize,
    next_request_id: u64,
    next_iteration_id: u64,
    last_now_ns: Option<u64>,
    aging_override_last_iteration: bool,
    inflight: Option<InflightPlan>,
    metrics: SchedulerMetrics,
    metrics_degraded: bool,
    accepting: bool,
}

/// A failed consuming close that returns ownership for correction or retry.
#[derive(Debug)]
pub struct SchedulerCloseFailure {
    error: SchedulerError,
    scheduler: Scheduler,
    settlement_failures: Vec<RequestSettlementFailure>,
}

impl SchedulerCloseFailure {
    #[must_use]
    pub const fn error(&self) -> &SchedulerError {
        &self.error
    }

    pub const fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Request-scoped ownership failures already contained before close failed.
    ///
    /// These diagnostics remain attached to the failed close so retrying host
    /// cleanup cannot silently erase failures from the settled GPU iteration.
    #[must_use]
    pub fn settlement_failures(&self) -> &[RequestSettlementFailure] {
        &self.settlement_failures
    }

    pub fn into_parts(self) -> (SchedulerError, Scheduler, Vec<RequestSettlementFailure>) {
        (self.error, self.scheduler, self.settlement_failures)
    }
}

impl std::fmt::Display for SchedulerCloseFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "scheduler close failed: {}", self.error)
    }
}

impl std::error::Error for SchedulerCloseFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Scheduler")
            .field("config", &self.config)
            .field("pool_stats", &self.pool.stats())
            .field("request_count", &self.requests.len())
            .field("waiting_count", &self.waiting.len())
            .field("terminal_count", &self.terminal.len())
            .field("pending_completion_count", &self.completion_outbox.len())
            .field("active_sequences", &self.active_sequences)
            .field("promised_kv_blocks", &self.promised_kv_blocks)
            .field("accepting", &self.accepting)
            .field(
                "inflight",
                &self.inflight.as_ref().map(|plan| plan.iteration_id),
            )
            .finish_non_exhaustive()
    }
}

impl Scheduler {
    /// Creates all bounded scheduler containers and the host paged-KV pool.
    ///
    /// # Errors
    ///
    /// Returns before publishing a scheduler when configuration, KV layout,
    /// checked capacity arithmetic, or host reservation fails.
    pub fn new(config: SchedulerConfig, layout: KvLayout) -> SchedulerResult<Self> {
        let config = config.validate()?;
        if config.max_promised_kv_blocks > layout.physical_block_count() {
            return Err(SchedulerError::InvalidConfiguration {
                field: "max_promised_kv_blocks",
                reason: "must not exceed the physical KV pool block count",
            });
        }
        let request_capacity = config
            .max_waiting_requests
            .checked_add(config.max_active_sequences)
            .ok_or(SchedulerError::ArithmeticOverflow {
                field: "live request capacity",
            })?;
        let mut requests = Vec::new();
        try_reserve_exact(&mut requests, request_capacity, "live requests")?;
        let mut waiting = VecDeque::new();
        waiting
            .try_reserve_exact(config.max_waiting_requests)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "waiting request queue",
                requested_elements: config.max_waiting_requests,
            })?;
        let terminal_capacity = request_capacity;
        let mut terminal = VecDeque::new();
        terminal.try_reserve_exact(terminal_capacity).map_err(|_| {
            SchedulerError::HostAllocation {
                resource: "terminal tombstones",
                requested_elements: terminal_capacity,
            }
        })?;
        let completion_outbox_capacity = request_capacity;
        let mut completion_outbox = VecDeque::new();
        completion_outbox
            .try_reserve_exact(completion_outbox_capacity)
            .map_err(|_| SchedulerError::HostAllocation {
                resource: "completion outbox",
                requested_elements: completion_outbox_capacity,
            })?;
        let metrics = SchedulerMetrics::new(config.metrics_window_samples)?;
        let mut scheduler = Self {
            config,
            pool: KvBlockPool::new(layout)?,
            requests,
            waiting,
            terminal,
            terminal_capacity,
            completion_outbox,
            completion_outbox_capacity,
            active_sequences: 0,
            waiting_prompt_tokens: 0,
            promised_kv_blocks: 0,
            next_request_id: 1,
            next_iteration_id: 1,
            last_now_ns: None,
            aging_override_last_iteration: false,
            inflight: None,
            metrics,
            metrics_degraded: false,
            accepting: true,
        };
        scheduler.refresh_metric_gauges();
        Ok(scheduler)
    }

    #[must_use]
    pub const fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    #[must_use]
    pub fn pool_stats(&self) -> KvBlockPoolStats {
        self.pool.stats()
    }

    #[must_use]
    pub const fn active_sequence_count(&self) -> usize {
        self.active_sequences
    }

    #[must_use]
    pub fn waiting_request_count(&self) -> usize {
        self.waiting.len()
    }

    #[must_use]
    pub const fn promised_kv_blocks(&self) -> usize {
        self.promised_kv_blocks
    }

    #[must_use]
    pub fn inflight_iteration_id(&self) -> Option<IterationId> {
        self.inflight.as_ref().map(|plan| plan.iteration_id)
    }

    /// Whether new request admission remains enabled.
    #[must_use]
    pub const fn is_accepting(&self) -> bool {
        self.accepting
    }

    /// Permanently disables new submissions while allowing outstanding work
    /// to complete or be explicitly aborted.
    pub const fn begin_shutdown(&mut self) {
        self.accepting = false;
    }

    /// Number of terminal notifications retained after a failed operation.
    #[must_use]
    pub fn pending_completion_count(&self) -> usize {
        self.completion_outbox.len()
    }

    /// Recovers one terminal notification retained after a failed operation.
    /// Mutating calls remain blocked until this bounded outbox is empty.
    pub fn pop_pending_completion(&mut self) -> Option<RequestCompletion> {
        self.completion_outbox.pop_front()
    }

    #[must_use]
    pub fn request_snapshot(&self, request_id: RequestId) -> Option<RequestSnapshot> {
        self.requests
            .iter()
            .find(|record| record.request_id == request_id)
            .map(RequestRecord::snapshot)
    }

    #[must_use]
    pub fn request_state(&self, request_id: RequestId) -> Option<RequestState> {
        self.requests
            .iter()
            .find(|record| record.request_id == request_id)
            .map(|record| record.state)
            .or_else(|| {
                self.terminal
                    .iter()
                    .find(|entry| entry.request_id == request_id)
                    .map(|entry| entry.state)
            })
    }

    /// Admits immediately when possible or enters the bounded FCFS queue.
    ///
    /// # Errors
    ///
    /// Rejects invalid requests, clock regression, capacity overflow, immediate
    /// overload, or host allocation before caller ownership is accepted.
    pub fn submit(
        &mut self,
        descriptor: RequestDescriptor,
        now_ns: u64,
    ) -> SchedulerResult<Submission> {
        self.ensure_completion_backlog_empty()?;
        if !self.accepting {
            return Err(SchedulerError::SchedulerClosed);
        }
        self.observe_now(now_ns)?;
        let maximum_logical_length = validate_descriptor(&self.config, &descriptor)?;
        let promised_kv_blocks = maximum_logical_length.div_ceil(KV_BLOCK_SIZE);
        if promised_kv_blocks > self.config.max_promised_kv_blocks {
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_rejection(),
                "request rejection",
            );
            return Err(SchedulerError::KvCapacityExceeded {
                requested_blocks: promised_kv_blocks,
                available_blocks: self.config.max_promised_kv_blocks,
            });
        }

        let can_admit = self.waiting.is_empty() && self.can_admit(promised_kv_blocks);
        if !can_admit && self.config.overload_policy == OverloadPolicy::RejectImmediately {
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_rejection(),
                "request rejection",
            );
            return Err(self.immediate_overload_error(promised_kv_blocks));
        }
        if !can_admit {
            self.validate_waiting_capacity(descriptor.prompt_token_ids.len())?;
        }

        // Do not retain arbitrary caller Vec excess capacity. The scheduler
        // owns an exact, length-bounded copy accounted by its token limits.
        let RequestDescriptor {
            prompt_token_ids,
            max_new_tokens,
        } = descriptor;
        let descriptor = RequestDescriptor::new(
            copy_tokens(&prompt_token_ids, "scheduler-owned prompt tokens")?,
            max_new_tokens,
        );

        let request_id = self.peek_request_id()?;
        let mut generated_token_ids = Vec::new();
        try_reserve_exact(
            &mut generated_token_ids,
            descriptor.max_new_tokens,
            "request generated tokens",
        )?;
        let mut record = RequestRecord {
            request_id,
            descriptor,
            state: RequestState::Waiting,
            submitted_at_ns: now_ns,
            ready_since_ns: now_ns,
            prefill_cursor: 0,
            generated_token_ids,
            promised_kv_blocks,
            sequence: None,
            cancellation_deferred: false,
        };

        let state = if can_admit {
            self.admit_record(&mut record, maximum_logical_length, now_ns)?;
            RequestState::Admitted
        } else {
            self.waiting_prompt_tokens = self
                .waiting_prompt_tokens
                .checked_add(record.descriptor.prompt_token_ids.len())
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "waiting prompt tokens",
                })?;
            self.waiting.push_back(request_id);
            RequestState::Waiting
        };
        self.requests.push(record);
        self.advance_request_id()?;
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_submission(),
            "request submission",
        );
        self.refresh_metric_gauges();
        Ok(Submission { request_id, state })
    }

    /// Expires queued requests, admits FCFS work, and builds at most one plan.
    ///
    /// Exactly one plan may be outstanding. An empty scheduler returns a
    /// successful output with `plan == None`.
    ///
    /// # Errors
    ///
    /// Returns for clock regression, another in-flight plan, checked host/KV
    /// allocation failure, or an internal plan invariant violation.
    pub fn plan_iteration(&mut self, now_ns: u64) -> SchedulerResult<PlanningOutput> {
        let cpu_started = Instant::now();
        self.ensure_completion_backlog_empty()?;
        self.observe_now(now_ns)?;
        if let Some(inflight) = &self.inflight {
            return Err(SchedulerError::IterationInFlight {
                iteration_id: inflight.iteration_id,
            });
        }
        self.ensure_completion_capacity(self.waiting.len())?;
        let mut completions = Vec::new();
        try_reserve_exact(
            &mut completions,
            self.waiting.len(),
            "admission timeout completions",
        )?;
        self.expire_waiting(now_ns)?;
        self.admit_waiting(now_ns)?;

        let (candidates, used_aging_override) = self.select_candidates(now_ns)?;
        if candidates.is_empty() {
            self.aging_override_last_iteration = false;
            self.drain_completion_outbox_into(&mut completions);
            self.refresh_metric_gauges();
            return Ok(PlanningOutput {
                plan: None,
                completions,
            });
        }

        let iteration_id = self.peek_iteration_id()?;
        let plan = self.prepare_plan(iteration_id, &candidates, now_ns)?;
        self.advance_iteration_id()?;
        self.aging_override_last_iteration = used_aging_override;
        if let Some(inflight) = &mut self.inflight {
            inflight.scheduler_cpu_ns = elapsed_ns(cpu_started);
        }
        self.drain_completion_outbox_into(&mut completions);
        self.refresh_metric_gauges();
        Ok(PlanningOutput {
            plan: Some(plan),
            completions,
        })
    }

    /// Commits one fully validated runtime result and routes outputs by slot.
    ///
    /// Malformed, stale, or replayed results are rejected before any sequence,
    /// reservation, token history, or metric is mutated.
    ///
    /// The caller must submit a result only after the executor stream that owns
    /// the corresponding plan has completed or synchronized. No device kernel
    /// may still retain the plan, its block tables, or reserved KV ranges when
    /// host ownership is committed by this method.
    ///
    /// # Errors
    ///
    /// Returns for clock/protocol mismatch, incomplete output routing, or an
    /// unexpected paged-KV ownership failure.
    #[allow(clippy::too_many_lines)]
    pub fn complete_iteration(
        &mut self,
        result: &IterationResult,
        now_ns: u64,
    ) -> SchedulerResult<IterationUpdates> {
        let cpu_started = Instant::now();
        self.ensure_completion_backlog_empty()?;
        self.validate_now(now_ns)?;
        self.validate_iteration_result(result)?;

        let (output_capacity, completion_capacity, batch_size) = {
            let inflight = self
                .inflight
                .as_ref()
                .ok_or(SchedulerError::NoIterationInFlight)?;
            (
                inflight.expected_output_slots.len(),
                inflight.items.len(),
                inflight
                    .prefill_count
                    .checked_add(inflight.decode_count)
                    .ok_or(SchedulerError::ArithmeticOverflow {
                        field: "completed iteration batch size",
                    })?,
            )
        };
        let mut updates = IterationUpdates::empty();
        try_reserve_exact(
            &mut updates.token_events,
            output_capacity,
            "iteration token events",
        )?;
        try_reserve_exact(
            &mut updates.completions,
            completion_capacity,
            "iteration completions",
        )?;
        try_reserve_exact(
            &mut updates.settlement_failures,
            completion_capacity,
            "iteration settlement failures",
        )?;
        self.ensure_completion_capacity(completion_capacity)?;
        let settled_items = self.prevalidate_completion_publication(result)?;
        let mut committed_items = Vec::new();
        try_reserve_exact(
            &mut committed_items,
            completion_capacity,
            "committed iteration items",
        )?;
        self.last_now_ns = Some(now_ns);
        let Some(inflight) = self.inflight.take() else {
            return Err(SchedulerError::NoIterationInFlight);
        };

        let prefill_tokens = inflight.prefill_tokens;
        let decode_tokens = inflight.decode_tokens;
        let scheduler_cpu_ns = inflight.scheduler_cpu_ns;
        let mut commit_failed = false;
        for (item, settled) in inflight.items.into_iter().zip(settled_items) {
            let request_id = item.request_id;
            match self.commit_reservation(item) {
                Ok(()) => committed_items.push(settled),
                Err(error) => {
                    commit_failed = true;
                    let contained = self
                        .force_reclaim_live_request(request_id, now_ns)
                        .err()
                        .unwrap_or(error);
                    updates.settlement_failures.push(RequestSettlementFailure {
                        request_id,
                        error: contained,
                    });
                }
            }
        }
        if commit_failed {
            for item in committed_items {
                if let Err(error) = self.finish_live_request(
                    item.request_id,
                    RequestFinishReason::ExecutorFailure,
                    now_ns,
                ) {
                    let error = self.contain_live_request_failure(item.request_id, now_ns, error);
                    updates.settlement_failures.push(RequestSettlementFailure {
                        request_id: item.request_id,
                        error,
                    });
                }
            }
            self.drain_completion_outbox_into(&mut updates.completions);
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_aborted_iteration(),
                "failed iteration commit",
            );
            self.refresh_metric_gauges();
            return Ok(updates);
        }
        for item in committed_items {
            if let Err(error) = self.publish_committed_item(item, now_ns, &mut updates) {
                let error = self.contain_live_request_failure(item.request_id, now_ns, error);
                updates.settlement_failures.push(RequestSettlementFailure {
                    request_id: item.request_id,
                    error,
                });
            }
        }
        self.drain_completion_outbox_into(&mut updates.completions);
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_iteration(IterationMetricSample {
                batch_size,
                prefill_tokens,
                decode_tokens,
                scheduler_cpu_ns: scheduler_cpu_ns.saturating_add(elapsed_ns(cpu_started)),
                gpu_execution_ns: result.gpu_execution_ns(),
                gpu_idle_gap_ns: result.gpu_idle_gap_ns(),
            }),
            "completed iteration",
        );
        self.refresh_metric_gauges();
        Ok(updates)
    }

    /// Settles a plan that could not produce valid runtime feedback.
    ///
    /// `DeviceQuiescedMutationUnknown` is an explicit safety assertion by the
    /// caller: the owning CUDA stream has completed and no device work can
    /// retain a block-table pointer. Those sequences are poisoned before they
    /// are closed and reported failed.
    ///
    /// # Errors
    ///
    /// Returns for a stale iteration, clock regression, or KV cleanup failure.
    pub fn abort_iteration(
        &mut self,
        iteration_id: IterationId,
        abort: ExecutionAbort,
        now_ns: u64,
    ) -> SchedulerResult<IterationUpdates> {
        let cpu_started = Instant::now();
        self.ensure_completion_backlog_empty()?;
        self.validate_now(now_ns)?;
        let expected = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?
            .iteration_id;
        if iteration_id != expected {
            return Err(SchedulerError::UnexpectedIteration {
                expected,
                actual: iteration_id,
            });
        }
        let completion_capacity = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?
            .items
            .len();
        self.validate_inflight_reservations()?;
        let mut updates = IterationUpdates::empty();
        try_reserve_exact(
            &mut updates.completions,
            completion_capacity,
            "aborted iteration completions",
        )?;
        try_reserve_exact(
            &mut updates.settlement_failures,
            completion_capacity,
            "aborted iteration settlement failures",
        )?;
        self.ensure_completion_capacity(completion_capacity)?;
        self.last_now_ns = Some(now_ns);
        let Some(inflight) = self.inflight.take() else {
            return Err(SchedulerError::NoIterationInFlight);
        };

        for item in inflight.items {
            let request_id = item.request_id;
            let previous_state = item.previous_state;
            let previous_ready_since_ns = item.previous_ready_since_ns;
            let settlement = match abort {
                ExecutionAbort::NotDispatched => ReservationSettlement::Rollback,
                ExecutionAbort::DeviceQuiescedMutationUnknown => ReservationSettlement::Poison,
            };
            if let Err(error) = self.settle_reservation(request_id, item.reservation, settlement) {
                updates
                    .settlement_failures
                    .push(RequestSettlementFailure { request_id, error });
                continue;
            }
            let Some(request_index) = self.record_index(request_id) else {
                updates.settlement_failures.push(RequestSettlementFailure {
                    request_id,
                    error: SchedulerError::UnknownRequest { request_id },
                });
                continue;
            };
            let update_result = match abort {
                ExecutionAbort::NotDispatched => {
                    if self.requests[request_index].cancellation_deferred {
                        self.finish_live_request(request_id, RequestFinishReason::Cancelled, now_ns)
                    } else {
                        let record = &mut self.requests[request_index];
                        record.state = previous_state;
                        record.ready_since_ns = previous_ready_since_ns;
                        Ok(())
                    }
                }
                ExecutionAbort::DeviceQuiescedMutationUnknown => self.finish_live_request(
                    request_id,
                    RequestFinishReason::ExecutorFailure,
                    now_ns,
                ),
            };
            if let Err(error) = update_result {
                let error = self.contain_live_request_failure(request_id, now_ns, error);
                updates
                    .settlement_failures
                    .push(RequestSettlementFailure { request_id, error });
            }
        }
        self.drain_completion_outbox_into(&mut updates.completions);
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_aborted_iteration(),
            "aborted iteration",
        );
        let _scheduler_cpu_ns = elapsed_ns(cpu_started);
        self.refresh_metric_gauges();
        Ok(updates)
    }

    /// Cancels queued or idle work immediately and defers in-flight reclamation.
    ///
    /// Repeated cancellation of a recent terminal request is an idempotent
    /// success. Terminal tombstones use a fixed rolling capacity.
    ///
    /// # Errors
    ///
    /// Returns for an unknown/evicted request, clock regression, or KV cleanup
    /// failure.
    pub fn cancel(
        &mut self,
        request_id: RequestId,
        now_ns: u64,
    ) -> SchedulerResult<CancellationOutcome> {
        self.ensure_completion_backlog_empty()?;
        self.observe_now(now_ns)?;
        if self
            .terminal
            .iter()
            .any(|entry| entry.request_id == request_id)
        {
            return Ok(CancellationOutcome {
                request_id,
                deferred_until_iteration_settles: false,
                already_terminal: true,
                completion: None,
            });
        }
        let index = self
            .record_index(request_id)
            .ok_or(SchedulerError::UnknownRequest { request_id })?;
        if self.requests[index].cancellation_deferred {
            return Ok(CancellationOutcome {
                request_id,
                deferred_until_iteration_settles: true,
                already_terminal: false,
                completion: None,
            });
        }
        let is_inflight = self
            .inflight
            .as_ref()
            .is_some_and(|plan| plan.items.iter().any(|item| item.request_id == request_id));
        if is_inflight {
            self.requests[index].cancellation_deferred = true;
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_cancellation(),
                "request cancellation",
            );
            self.refresh_metric_gauges();
            return Ok(CancellationOutcome {
                request_id,
                deferred_until_iteration_settles: true,
                already_terminal: false,
                completion: None,
            });
        }

        self.ensure_completion_capacity(1)?;
        self.finish_live_request(request_id, RequestFinishReason::Cancelled, now_ns)?;
        let completion = self
            .completion_outbox
            .pop_front()
            .ok_or(SchedulerError::InvalidPlan {
                field: "completion outbox",
                reason: "immediate cancellation did not enqueue its terminal payload",
            })?;
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_cancellation(),
            "request cancellation",
        );
        self.refresh_metric_gauges();
        Ok(CancellationOutcome {
            request_id,
            deferred_until_iteration_settles: false,
            already_terminal: false,
            completion: Some(completion),
        })
    }

    /// Closes every non-in-flight request, normally during server shutdown.
    ///
    /// # Errors
    ///
    /// Refuses shutdown while an executor may still hold an immutable plan.
    pub fn shutdown(&mut self, now_ns: u64) -> SchedulerResult<Vec<RequestCompletion>> {
        self.begin_shutdown();
        self.ensure_completion_backlog_empty()?;
        self.observe_now(now_ns)?;
        if let Some(inflight) = &self.inflight {
            return Err(SchedulerError::IterationInFlight {
                iteration_id: inflight.iteration_id,
            });
        }
        let mut completions = Vec::new();
        try_reserve_exact(
            &mut completions,
            self.requests.len(),
            "shutdown completions",
        )?;
        self.ensure_completion_capacity(self.requests.len())?;
        while let Some(record) = self.requests.last() {
            let request_id = record.request_id;
            self.finish_live_request(request_id, RequestFinishReason::Cancelled, now_ns)?;
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_cancellation(),
                "request cancellation",
            );
        }
        self.drain_completion_outbox_into(&mut completions);
        self.refresh_metric_gauges();
        Ok(completions)
    }

    /// Permanently stops admission, explicitly settles any outstanding plan,
    /// closes every sequence, and consumes the scheduler.
    ///
    /// When a plan is in flight, `inflight_abort` is mandatory. The
    /// `DeviceQuiescedMutationUnknown` variant is a caller assertion that the
    /// owning executor stream has synchronized before this call. If close
    /// fails, [`SchedulerCloseFailure`] returns ownership so cleanup can be
    /// corrected and retried instead of dropping live reservation state.
    ///
    /// # Errors
    ///
    /// Returns ownership in [`SchedulerCloseFailure`] for clock/allocation
    /// errors, a missing in-flight disposition, or a retryable cleanup failure.
    #[allow(clippy::result_large_err)]
    pub fn close(
        mut self,
        now_ns: u64,
        inflight_abort: Option<ExecutionAbort>,
    ) -> Result<SchedulerCloseOutput, SchedulerCloseFailure> {
        self.begin_shutdown();
        if let Err(error) = self.validate_now(now_ns) {
            return Err(SchedulerCloseFailure {
                error,
                scheduler: self,
                settlement_failures: Vec::new(),
            });
        }
        let Some(completion_capacity) = self
            .completion_outbox
            .len()
            .checked_add(self.requests.len())
        else {
            return Err(SchedulerCloseFailure {
                error: SchedulerError::ArithmeticOverflow {
                    field: "close completion capacity",
                },
                scheduler: self,
                settlement_failures: Vec::new(),
            });
        };
        let mut completions = Vec::new();
        if let Err(error) =
            try_reserve_exact(&mut completions, completion_capacity, "close completions")
        {
            return Err(SchedulerCloseFailure {
                error,
                scheduler: self,
                settlement_failures: Vec::new(),
            });
        }
        self.drain_completion_outbox_into(&mut completions);
        self.last_now_ns = Some(now_ns);

        let mut settlement_failures = Vec::new();
        if let Some(iteration_id) = self.inflight_iteration_id() {
            let Some(abort) = inflight_abort else {
                return Err(self.close_failure(
                    SchedulerError::CloseDispositionRequired { iteration_id },
                    completions,
                    Vec::new(),
                ));
            };
            match self.abort_iteration(iteration_id, abort, now_ns) {
                Ok(updates) => {
                    let (token_events, mut aborted, failures) = updates.into_parts();
                    debug_assert!(token_events.is_empty());
                    completions.append(&mut aborted);
                    settlement_failures = failures;
                }
                Err(error) => {
                    return Err(self.close_failure(error, completions, Vec::new()));
                }
            }
        }
        match self.shutdown(now_ns) {
            Ok(mut remaining) => completions.append(&mut remaining),
            Err(error) => {
                return Err(self.close_failure(error, completions, settlement_failures));
            }
        }
        let pool = self.pool.stats();
        if pool.allocated_block_count() != 0
            || self.active_sequences != 0
            || self.promised_kv_blocks != 0
            || self.waiting_prompt_tokens != 0
            || !self.requests.is_empty()
            || !self.waiting.is_empty()
            || !self.completion_outbox.is_empty()
            || self.inflight.is_some()
        {
            return Err(self.close_failure(
                SchedulerError::InvalidPlan {
                    field: "close cleanup",
                    reason: "successful shutdown left live ownership or pending notifications",
                },
                completions,
                settlement_failures,
            ));
        }
        Ok(SchedulerCloseOutput {
            completions,
            settlement_failures,
        })
    }

    /// Returns a bounded rolling metrics snapshot and current gauges.
    ///
    /// # Errors
    ///
    /// Returns if the fixed-size percentile scratch cannot be allocated.
    pub fn metrics_snapshot(&self) -> SchedulerResult<SchedulerMetricsSnapshot> {
        let mut snapshot = self.metrics.snapshot()?;
        snapshot.metrics_degraded = self.metrics_degraded;
        snapshot.gauges = self.current_gauges();
        Ok(snapshot)
    }

    fn expire_waiting(&mut self, now_ns: u64) -> SchedulerResult<()> {
        let Some(timeout_ns) = self.config.admission_timeout_ns else {
            return Ok(());
        };
        loop {
            let Some(request_id) = self.waiting.front().copied() else {
                break;
            };
            let index = self
                .record_index(request_id)
                .ok_or(SchedulerError::UnknownRequest { request_id })?;
            let waited_ns = now_ns.saturating_sub(self.requests[index].submitted_at_ns);
            if waited_ns < timeout_ns {
                break;
            }
            self.finish_live_request(request_id, RequestFinishReason::AdmissionTimeout, now_ns)?;
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_admission_timeout(),
                "admission timeout",
            );
        }
        Ok(())
    }

    fn admit_waiting(&mut self, now_ns: u64) -> SchedulerResult<()> {
        loop {
            let Some(request_id) = self.waiting.front().copied() else {
                return Ok(());
            };
            let index = self
                .record_index(request_id)
                .ok_or(SchedulerError::UnknownRequest { request_id })?;
            let promised = self.requests[index].promised_kv_blocks;
            if !self.can_admit(promised) {
                return Ok(());
            }
            let maximum = maximum_logical_length(&self.requests[index].descriptor)?;
            let sequence = self.pool.create_sequence(maximum)?;
            let prompt_tokens = self.requests[index].descriptor.prompt_token_ids.len();
            let waited_ns = now_ns.saturating_sub(self.requests[index].submitted_at_ns);
            let popped = self.waiting.pop_front();
            debug_assert_eq!(popped, Some(request_id));
            self.waiting_prompt_tokens = self
                .waiting_prompt_tokens
                .checked_sub(prompt_tokens)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "waiting prompt token decrement",
                })?;
            self.active_sequences =
                self.active_sequences
                    .checked_add(1)
                    .ok_or(SchedulerError::ArithmeticOverflow {
                        field: "active sequence count",
                    })?;
            self.promised_kv_blocks = self.promised_kv_blocks.checked_add(promised).ok_or(
                SchedulerError::ArithmeticOverflow {
                    field: "promised KV blocks",
                },
            )?;
            let record = &mut self.requests[index];
            transition(record, RequestState::Admitted)?;
            record.sequence = Some(sequence);
            record.ready_since_ns = now_ns;
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_admission(waited_ns),
                "request admission",
            );
        }
    }

    fn select_candidates(&self, now_ns: u64) -> SchedulerResult<(Vec<Candidate>, bool)> {
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        try_reserve_exact(&mut prefill, self.active_sequences, "prefill candidates")?;
        try_reserve_exact(&mut decode, self.active_sequences, "decode candidates")?;
        for record in &self.requests {
            match record.state {
                RequestState::Admitted | RequestState::Prefilling => {
                    let remaining = record
                        .descriptor
                        .prompt_token_ids
                        .len()
                        .saturating_sub(record.prefill_cursor);
                    if remaining != 0 {
                        prefill.push(ReadyRequest {
                            request_id: record.request_id,
                            ready_since_ns: record.ready_since_ns,
                            remaining_tokens: remaining,
                        });
                    }
                }
                RequestState::Decoding => decode.push(ReadyRequest {
                    request_id: record.request_id,
                    ready_since_ns: record.ready_since_ns,
                    remaining_tokens: 1,
                }),
                RequestState::Waiting
                | RequestState::Finished
                | RequestState::Cancelled
                | RequestState::Failed => {}
            }
        }
        prefill.sort_unstable_by_key(|item| (item.ready_since_ns, item.request_id));
        decode.sort_unstable_by_key(|item| (item.ready_since_ns, item.request_id));

        let mut selected = Vec::new();
        try_reserve_exact(&mut selected, self.active_sequences, "iteration candidates")?;
        let mut budget = self.config.iteration_token_budget;
        let mut aged_request = None;
        let can_override = !decode.is_empty() && !self.aging_override_last_iteration;
        if can_override {
            aged_request = prefill.iter().find_map(|item| {
                (now_ns.saturating_sub(item.ready_since_ns) >= self.config.aging_threshold_ns)
                    .then_some(item.request_id)
            });
        }
        if let Some(request_id) = aged_request {
            let item = prefill
                .iter()
                .find(|item| item.request_id == request_id)
                .ok_or(SchedulerError::InvalidPlan {
                    field: "aging candidate",
                    reason: "selected prefill request disappeared",
                })?;
            let reserved_for_decode = usize::from(budget > 1);
            let usable = budget.saturating_sub(reserved_for_decode).max(1);
            let token_count = item
                .remaining_tokens
                .min(self.config.max_prefill_chunk_tokens)
                .min(usable);
            selected.push(Candidate {
                request_id,
                kind: WorkKind::Prefill,
                token_count,
            });
            budget = budget.saturating_sub(token_count);
        }

        for item in &decode {
            if budget == 0 {
                break;
            }
            selected.push(Candidate {
                request_id: item.request_id,
                kind: WorkKind::Decode,
                token_count: 1,
            });
            budget -= 1;
        }
        for item in &prefill {
            if budget == 0 {
                break;
            }
            if Some(item.request_id) == aged_request {
                continue;
            }
            let token_count = item
                .remaining_tokens
                .min(self.config.max_prefill_chunk_tokens)
                .min(budget);
            selected.push(Candidate {
                request_id: item.request_id,
                kind: WorkKind::Prefill,
                token_count,
            });
            budget -= token_count;
        }
        Ok((selected, aged_request.is_some()))
    }

    #[allow(clippy::too_many_lines)]
    fn prepare_plan(
        &mut self,
        iteration_id: IterationId,
        candidates: &[Candidate],
        now_ns: u64,
    ) -> SchedulerResult<IterationPlan> {
        let mut prefill_items = Vec::new();
        let mut decode_items = Vec::new();
        let mut block_tables = Vec::new();
        let mut inflight_items = Vec::new();
        let mut output_slots = Vec::new();
        let mut payloads = Vec::new();
        let mut transition_indices = Vec::new();
        for (buffer, resource) in [
            (&mut prefill_items, "prefill plan items"),
            (&mut decode_items, "decode plan items"),
        ] {
            try_reserve_exact(buffer, candidates.len(), resource)?;
        }
        try_reserve_exact(
            &mut block_tables,
            candidates.len(),
            "iteration block tables",
        )?;
        try_reserve_exact(
            &mut inflight_items,
            candidates.len(),
            "in-flight reservations",
        )?;
        try_reserve_exact(&mut output_slots, candidates.len(), "expected output slots")?;
        try_reserve_exact(&mut payloads, candidates.len(), "candidate payloads")?;
        try_reserve_exact(
            &mut transition_indices,
            candidates.len(),
            "request transition indices",
        )?;

        for candidate in candidates {
            let index =
                self.record_index(candidate.request_id)
                    .ok_or(SchedulerError::UnknownRequest {
                        request_id: candidate.request_id,
                    })?;
            let (input_tokens, target_logical_length, needs_output) = {
                let record = &self.requests[index];
                match candidate.kind {
                    WorkKind::Prefill => {
                        let end = record
                            .prefill_cursor
                            .checked_add(candidate.token_count)
                            .ok_or(SchedulerError::ArithmeticOverflow {
                                field: "prefill cursor",
                            })?;
                        if end > record.descriptor.prompt_token_ids.len() {
                            return Err(SchedulerError::InvalidPlan {
                                field: "prefill range",
                                reason: "candidate exceeds the retained prompt",
                            });
                        }
                        let input = copy_tokens(
                            &record.descriptor.prompt_token_ids[record.prefill_cursor..end],
                            "prefill input tokens",
                        )?;
                        (input, end, end == record.descriptor.prompt_token_ids.len())
                    }
                    WorkKind::Decode => {
                        let token = record.generated_token_ids.last().copied().ok_or(
                            SchedulerError::InvalidPlan {
                                field: "decode token",
                                reason: "decoding request has no prior generated token",
                            },
                        )?;
                        let logical = record
                            .sequence
                            .as_ref()
                            .ok_or(SchedulerError::InvalidPlan {
                                field: "sequence",
                                reason: "decode candidate has no KV sequence",
                            })?
                            .logical_length() as usize;
                        let target =
                            logical
                                .checked_add(1)
                                .ok_or(SchedulerError::ArithmeticOverflow {
                                    field: "decode target logical length",
                                })?;
                        (copy_tokens(&[token], "decode input token")?, target, true)
                    }
                }
            };
            let output_slot = if needs_output {
                let raw = u32::try_from(output_slots.len()).map_err(|_| {
                    SchedulerError::ArithmeticOverflow {
                        field: "iteration output slot",
                    }
                })?;
                let slot = OutputSlot::new(raw);
                output_slots.push(slot);
                Some(slot)
            } else {
                None
            };
            payloads.push(CandidatePayload {
                candidate: *candidate,
                input_tokens,
                target_logical_length,
                output_slot,
            });
        }

        for payload in payloads {
            let candidate = payload.candidate;
            let Some(index) = self.record_index(candidate.request_id) else {
                return self.rollback_prepared_plan(
                    inflight_items,
                    SchedulerError::UnknownRequest {
                        request_id: candidate.request_id,
                    },
                );
            };
            let previous_state = self.requests[index].state;
            let previous_ready_since_ns = self.requests[index].ready_since_ns;
            let Some(sequence) = self.requests[index].sequence.as_mut() else {
                return self.rollback_prepared_plan(
                    inflight_items,
                    SchedulerError::InvalidPlan {
                        field: "sequence",
                        reason: "active candidate has no KV sequence",
                    },
                );
            };
            let reservation = match sequence
                .reserve_to(&mut self.pool, payload.target_logical_length)
                .map_err(SchedulerError::from)
            {
                Ok(reservation) => reservation,
                Err(error) => {
                    return self.rollback_prepared_plan(inflight_items, error);
                }
            };
            inflight_items.push(InflightItem {
                request_id: candidate.request_id,
                kind: candidate.kind,
                target_logical_length: payload.target_logical_length,
                output_slot: payload.output_slot,
                previous_state,
                previous_ready_since_ns,
                reservation,
            });
            let Some(active_item) = inflight_items.last() else {
                return Err(SchedulerError::InvalidPlan {
                    field: "reservation",
                    reason: "prepared reservation was not retained",
                });
            };
            let table_result = match self.requests[index].sequence.as_ref() {
                Some(sequence) => sequence
                    .reserved_block_table(&active_item.reservation)
                    .map_err(SchedulerError::from)
                    .and_then(|table| OwnedBlockTable::copy_from_v1(candidate.request_id, table)),
                None => Err(SchedulerError::InvalidPlan {
                    field: "sequence",
                    reason: "reserved candidate has no KV sequence",
                }),
            };
            let table = match table_result {
                Ok(table) => table,
                Err(error) => return self.rollback_prepared_plan(inflight_items, error),
            };
            let table_index = block_tables.len();
            let work_item = match WorkItem::new(
                candidate.request_id,
                candidate.kind,
                payload.input_tokens,
                payload.target_logical_length,
                table_index,
                payload.output_slot,
            ) {
                Ok(item) => item,
                Err(error) => return self.rollback_prepared_plan(inflight_items, error),
            };
            block_tables.push(table);
            match candidate.kind {
                WorkKind::Prefill => prefill_items.push(work_item),
                WorkKind::Decode => decode_items.push(work_item),
            }
        }

        let plan = match IterationPlan::new(iteration_id, prefill_items, decode_items, block_tables)
        {
            Ok(plan) => plan,
            Err(error) => return self.rollback_prepared_plan(inflight_items, error),
        };
        let Some(prefill_tokens) = plan
            .prefill_items()
            .iter()
            .map(|item| item.input_tokens().len())
            .try_fold(0_usize, usize::checked_add)
        else {
            return self.rollback_prepared_plan(
                inflight_items,
                SchedulerError::ArithmeticOverflow {
                    field: "prefill plan token count",
                },
            );
        };
        let Some(decode_tokens) = plan
            .decode_items()
            .iter()
            .map(|item| item.input_tokens().len())
            .try_fold(0_usize, usize::checked_add)
        else {
            return self.rollback_prepared_plan(
                inflight_items,
                SchedulerError::ArithmeticOverflow {
                    field: "decode plan token count",
                },
            );
        };
        for item_index in 0..inflight_items.len() {
            let request_id = inflight_items[item_index].request_id;
            let kind = inflight_items[item_index].kind;
            let Some(index) = self.record_index(request_id) else {
                return self.rollback_prepared_plan(
                    inflight_items,
                    SchedulerError::UnknownRequest { request_id },
                );
            };
            let target_state = match kind {
                WorkKind::Prefill => RequestState::Prefilling,
                WorkKind::Decode => RequestState::Decoding,
            };
            if !valid_transition(self.requests[index].state, target_state) {
                let error = SchedulerError::InvalidStateTransition {
                    request_id,
                    from: self.requests[index].state.name(),
                    to: target_state.name(),
                };
                return self.rollback_prepared_plan(inflight_items, error);
            }
            transition_indices.push(index);
        }
        for (item, index) in inflight_items.iter().zip(transition_indices) {
            self.requests[index].state = match item.kind {
                WorkKind::Prefill => RequestState::Prefilling,
                WorkKind::Decode => RequestState::Decoding,
            };
            self.requests[index].ready_since_ns = now_ns;
        }
        let prefill_count = plan.prefill_items().len();
        let decode_count = plan.decode_items().len();
        self.inflight = Some(InflightPlan {
            iteration_id,
            prefill_tokens,
            decode_tokens,
            prefill_count,
            decode_count,
            scheduler_cpu_ns: 0,
            expected_output_slots: output_slots,
            items: inflight_items,
        });
        Ok(plan)
    }

    fn rollback_prepared_plan<T>(
        &mut self,
        items: Vec<InflightItem>,
        original: SchedulerError,
    ) -> SchedulerResult<T> {
        let mut cleanup_error = None;
        for item in items.into_iter().rev() {
            let request_id = item.request_id;
            let previous_state = item.previous_state;
            let previous_ready_since_ns = item.previous_ready_since_ns;
            if let Err(error) = self.settle_reservation(
                request_id,
                item.reservation,
                ReservationSettlement::Rollback,
            ) {
                cleanup_error.get_or_insert(error);
                continue;
            }
            if let Some(index) = self.record_index(request_id) {
                self.requests[index].state = previous_state;
                self.requests[index].ready_since_ns = previous_ready_since_ns;
            }
        }
        Err(cleanup_error.unwrap_or(original))
    }

    fn settle_reservation(
        &mut self,
        request_id: RequestId,
        reservation: SequenceReservation,
        settlement: ReservationSettlement,
    ) -> SchedulerResult<()> {
        let Some(index) = self.record_index(request_id) else {
            return Err(SchedulerError::UnknownRequest { request_id });
        };
        let Some(sequence) = self.requests[index].sequence.as_mut() else {
            return Err(SchedulerError::InvalidPlan {
                field: "sequence",
                reason: "reservation owner has no KV sequence",
            });
        };
        let result = match settlement {
            ReservationSettlement::Rollback => sequence.rollback(&mut self.pool, reservation),
            ReservationSettlement::Poison => sequence.poison(&mut self.pool, reservation),
        };
        match result {
            Ok(()) => Ok(()),
            Err(source) => {
                let original = SchedulerError::PagedKv(source);
                self.force_reclaim_live_request(request_id, self.last_now_ns.unwrap_or(0))?;
                Err(original)
            }
        }
    }

    fn force_reclaim_live_request(
        &mut self,
        request_id: RequestId,
        completed_at_ns: u64,
    ) -> SchedulerResult<()> {
        self.ensure_completion_capacity(1)?;
        let index = self
            .record_index(request_id)
            .ok_or(SchedulerError::UnknownRequest { request_id })?;
        let active_sequences =
            self.active_sequences
                .checked_sub(1)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "forced active sequence decrement",
                })?;
        let promised_kv_blocks = self
            .promised_kv_blocks
            .checked_sub(self.requests[index].promised_kv_blocks)
            .ok_or(SchedulerError::ArithmeticOverflow {
                field: "forced promised KV block decrement",
            })?;
        let sequence = self.requests[index]
            .sequence
            .take()
            .ok_or(SchedulerError::InvalidPlan {
                field: "sequence",
                reason: "forced reclaim request has no KV sequence",
            })?;
        let reclaim = sequence.abandon_for_reclaim();
        self.pool.reclaim_sequence(&reclaim)?;
        self.active_sequences = active_sequences;
        self.promised_kv_blocks = promised_kv_blocks;
        let mut record = self.requests.swap_remove(index);
        self.completion_outbox.push_back(RequestCompletion {
            request_id,
            reason: RequestFinishReason::ExecutorFailure,
            generated_token_ids: std::mem::take(&mut record.generated_token_ids),
            completed_at_ns,
        });
        self.remember_terminal(request_id, RequestState::Failed);
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_failed(),
            "forced request failure",
        );
        Ok(())
    }

    fn contain_live_request_failure(
        &mut self,
        request_id: RequestId,
        now_ns: u64,
        original: SchedulerError,
    ) -> SchedulerError {
        if self.record_index(request_id).is_none() {
            return original;
        }
        self.force_reclaim_live_request(request_id, now_ns)
            .err()
            .unwrap_or(original)
    }

    fn validate_iteration_result(&self, result: &IterationResult) -> SchedulerResult<()> {
        let inflight = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?;
        if result.iteration_id() != inflight.iteration_id {
            return Err(SchedulerError::UnexpectedIteration {
                expected: inflight.iteration_id,
                actual: result.iteration_id(),
            });
        }
        if result.outputs().len() != inflight.expected_output_slots.len() {
            return Err(SchedulerError::InvalidIterationResult {
                field: "outputs",
                reason: "runtime must return exactly every planned output slot",
            });
        }
        for expected in &inflight.expected_output_slots {
            if !result
                .outputs()
                .iter()
                .any(|output| output.slot() == *expected)
            {
                return Err(SchedulerError::InvalidIterationResult {
                    field: "outputs",
                    reason: "runtime omitted a planned output slot",
                });
            }
        }
        for output in result.outputs() {
            if !inflight.expected_output_slots.contains(&output.slot()) {
                return Err(SchedulerError::InvalidIterationResult {
                    field: "outputs",
                    reason: "runtime returned an unplanned output slot",
                });
            }
        }
        Ok(())
    }

    fn validate_inflight_reservations(&self) -> SchedulerResult<()> {
        let inflight = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?;
        for item in &inflight.items {
            let index =
                self.record_index(item.request_id)
                    .ok_or(SchedulerError::UnknownRequest {
                        request_id: item.request_id,
                    })?;
            let sequence =
                self.requests[index]
                    .sequence
                    .as_ref()
                    .ok_or(SchedulerError::InvalidPlan {
                        field: "sequence",
                        reason: "in-flight reservation owner has no KV sequence",
                    })?;
            let table = sequence.reserved_block_table(&item.reservation)?;
            if table.logical_length() as usize != item.target_logical_length {
                return Err(SchedulerError::InvalidPlan {
                    field: "target_logical_length",
                    reason: "reserved KV table differs from the in-flight work target",
                });
            }
        }
        Ok(())
    }

    fn prevalidate_completion_publication(
        &self,
        result: &IterationResult,
    ) -> SchedulerResult<Vec<SettledInflightItem>> {
        let inflight = self
            .inflight
            .as_ref()
            .ok_or(SchedulerError::NoIterationInFlight)?;
        let mut settled = Vec::new();
        try_reserve_exact(
            &mut settled,
            inflight.items.len(),
            "settled iteration metadata",
        )?;
        for item in &inflight.items {
            if item.reservation.target_logical_length() as usize != item.target_logical_length {
                return Err(SchedulerError::InvalidPlan {
                    field: "target_logical_length",
                    reason: "reservation target differs from the in-flight work target",
                });
            }
            let index =
                self.record_index(item.request_id)
                    .ok_or(SchedulerError::UnknownRequest {
                        request_id: item.request_id,
                    })?;
            let record = &self.requests[index];
            let sequence = record
                .sequence
                .as_ref()
                .ok_or(SchedulerError::InvalidPlan {
                    field: "sequence",
                    reason: "in-flight request has no KV sequence",
                })?;
            let table = sequence.reserved_block_table(&item.reservation)?;
            if table.logical_length() as usize != item.target_logical_length {
                return Err(SchedulerError::InvalidPlan {
                    field: "target_logical_length",
                    reason: "reserved KV table differs from the in-flight work target",
                });
            }
            let output = match item.output_slot {
                Some(slot) => Some(
                    result
                        .outputs()
                        .iter()
                        .find(|output| output.slot() == slot)
                        .copied()
                        .ok_or(SchedulerError::InvalidIterationResult {
                            field: "outputs",
                            reason: "validated output slot disappeared",
                        })?,
                ),
                None => None,
            };
            let target_state = if record.cancellation_deferred {
                RequestState::Cancelled
            } else if let Some(output) = output {
                let generated_tokens = record.generated_token_ids.len().checked_add(1).ok_or(
                    SchedulerError::ArithmeticOverflow {
                        field: "generated token count",
                    },
                )?;
                if generated_tokens > record.descriptor.max_new_tokens {
                    return Err(SchedulerError::InvalidPlan {
                        field: "generated token count",
                        reason: "runtime output exceeds the request generation bound",
                    });
                }
                if output.stop() || generated_tokens == record.descriptor.max_new_tokens {
                    RequestState::Finished
                } else {
                    RequestState::Decoding
                }
            } else {
                RequestState::Prefilling
            };
            if !valid_transition(record.state, target_state) {
                return Err(SchedulerError::InvalidStateTransition {
                    request_id: item.request_id,
                    from: record.state.name(),
                    to: target_state.name(),
                });
            }
            settled.push(SettledInflightItem {
                request_id: item.request_id,
                kind: item.kind,
                target_logical_length: item.target_logical_length,
                output,
            });
        }
        Ok(settled)
    }

    fn commit_reservation(&mut self, item: InflightItem) -> SchedulerResult<()> {
        let request_id = item.request_id;
        let Some(index) = self.record_index(request_id) else {
            return Err(SchedulerError::UnknownRequest { request_id });
        };
        let Some(sequence) = self.requests[index].sequence.as_mut() else {
            return Err(SchedulerError::InvalidPlan {
                field: "sequence",
                reason: "in-flight request has no KV sequence",
            });
        };
        let commit = sequence.commit(&mut self.pool, item.reservation)?;
        if commit.logical_length() as usize != item.target_logical_length {
            return Err(SchedulerError::InvalidPlan {
                field: "target_logical_length",
                reason: "paged-KV commit published a different logical length",
            });
        }
        Ok(())
    }

    fn publish_committed_item(
        &mut self,
        item: SettledInflightItem,
        now_ns: u64,
        updates: &mut IterationUpdates,
    ) -> SchedulerResult<()> {
        let request_id = item.request_id;
        let index = self
            .record_index(request_id)
            .ok_or(SchedulerError::UnknownRequest { request_id })?;
        if self.requests[index].cancellation_deferred {
            return self.finish_live_request(request_id, RequestFinishReason::Cancelled, now_ns);
        }
        if item.kind == WorkKind::Prefill {
            self.requests[index].prefill_cursor = item.target_logical_length;
        }
        let Some(output) = item.output else {
            let record = &mut self.requests[index];
            record.state = RequestState::Prefilling;
            record.ready_since_ns = now_ns;
            return Ok(());
        };
        let generated_index = self.requests[index].generated_token_ids.len();
        self.requests[index]
            .generated_token_ids
            .push(output.token_id());
        updates.token_events.push(TokenEvent {
            request_id,
            token_id: output.token_id(),
            generated_index,
        });
        let reached_length = self.requests[index].generated_token_ids.len()
            == self.requests[index].descriptor.max_new_tokens;
        if output.stop() || reached_length {
            let reason = if output.stop() {
                RequestFinishReason::Stop
            } else {
                RequestFinishReason::Length
            };
            self.finish_live_request(request_id, reason, now_ns)
        } else {
            let record = &mut self.requests[index];
            record.state = RequestState::Decoding;
            record.ready_since_ns = now_ns;
            Ok(())
        }
    }

    fn finish_live_request(
        &mut self,
        request_id: RequestId,
        reason: RequestFinishReason,
        now_ns: u64,
    ) -> SchedulerResult<()> {
        self.ensure_completion_capacity(1)?;
        let index = self
            .record_index(request_id)
            .ok_or(SchedulerError::UnknownRequest { request_id })?;
        let previous_state = self.requests[index].state;
        let terminal_state = reason.state();
        if !valid_transition(previous_state, terminal_state) {
            return Err(SchedulerError::InvalidStateTransition {
                request_id,
                from: previous_state.name(),
                to: terminal_state.name(),
            });
        }
        if previous_state == RequestState::Waiting {
            let prompt_tokens = self.requests[index].descriptor.prompt_token_ids.len();
            let queue_index = self
                .waiting
                .iter()
                .position(|candidate| *candidate == request_id)
                .ok_or(SchedulerError::InvalidPlan {
                    field: "waiting queue",
                    reason: "waiting request is absent from the FCFS queue",
                })?;
            let waiting_prompt_tokens = self
                .waiting_prompt_tokens
                .checked_sub(prompt_tokens)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "waiting prompt token decrement",
                })?;
            self.waiting.remove(queue_index);
            self.waiting_prompt_tokens = waiting_prompt_tokens;
        } else {
            let active_sequences =
                self.active_sequences
                    .checked_sub(1)
                    .ok_or(SchedulerError::ArithmeticOverflow {
                        field: "active sequence decrement",
                    })?;
            let promised_kv_blocks = self
                .promised_kv_blocks
                .checked_sub(self.requests[index].promised_kv_blocks)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "promised KV block decrement",
                })?;
            let close_result = self.requests[index]
                .sequence
                .as_mut()
                .ok_or(SchedulerError::InvalidPlan {
                    field: "sequence",
                    reason: "active request has no KV sequence",
                })?
                .close(&mut self.pool);
            if let Err(source) = close_result {
                let original = SchedulerError::PagedKv(source);
                self.force_reclaim_live_request(request_id, now_ns)?;
                return Err(original);
            }
            self.active_sequences = active_sequences;
            self.promised_kv_blocks = promised_kv_blocks;
        }
        let mut record = self.requests.swap_remove(index);
        self.completion_outbox.push_back(RequestCompletion {
            request_id,
            reason,
            generated_token_ids: std::mem::take(&mut record.generated_token_ids),
            completed_at_ns: now_ns,
        });
        self.remember_terminal(request_id, terminal_state);
        match reason {
            RequestFinishReason::Length | RequestFinishReason::Stop => {
                observe_metric(
                    &mut self.metrics_degraded,
                    self.metrics.record_finished(),
                    "request completion",
                );
            }
            RequestFinishReason::AdmissionTimeout | RequestFinishReason::ExecutorFailure => {
                observe_metric(
                    &mut self.metrics_degraded,
                    self.metrics.record_failed(),
                    "request failure",
                );
            }
            RequestFinishReason::Cancelled => {}
        }
        Ok(())
    }

    fn remember_terminal(&mut self, request_id: RequestId, state: RequestState) {
        debug_assert!(state.is_terminal());
        if self.terminal.len() == self.terminal_capacity {
            self.terminal.pop_front();
        }
        self.terminal
            .push_back(TerminalTombstone { request_id, state });
    }

    fn admit_record(
        &mut self,
        record: &mut RequestRecord,
        maximum_logical_length: usize,
        now_ns: u64,
    ) -> SchedulerResult<()> {
        let sequence = self.pool.create_sequence(maximum_logical_length)?;
        transition(record, RequestState::Admitted)?;
        record.sequence = Some(sequence);
        record.ready_since_ns = now_ns;
        self.active_sequences =
            self.active_sequences
                .checked_add(1)
                .ok_or(SchedulerError::ArithmeticOverflow {
                    field: "active sequence count",
                })?;
        self.promised_kv_blocks = self
            .promised_kv_blocks
            .checked_add(record.promised_kv_blocks)
            .ok_or(SchedulerError::ArithmeticOverflow {
                field: "promised KV blocks",
            })?;
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.record_admission(0),
            "request admission",
        );
        Ok(())
    }

    fn can_admit(&self, requested_blocks: usize) -> bool {
        self.active_sequences < self.config.max_active_sequences
            && self
                .promised_kv_blocks
                .checked_add(requested_blocks)
                .is_some_and(|total| total <= self.config.max_promised_kv_blocks)
    }

    fn immediate_overload_error(&self, requested_blocks: usize) -> SchedulerError {
        if self.active_sequences >= self.config.max_active_sequences {
            SchedulerError::ActiveSequenceLimit {
                limit: self.config.max_active_sequences,
            }
        } else {
            SchedulerError::KvCapacityExceeded {
                requested_blocks,
                available_blocks: self
                    .config
                    .max_promised_kv_blocks
                    .saturating_sub(self.promised_kv_blocks),
            }
        }
    }

    fn validate_waiting_capacity(&mut self, prompt_tokens: usize) -> SchedulerResult<()> {
        if self.waiting.len() >= self.config.max_waiting_requests {
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_rejection(),
                "request rejection",
            );
            return Err(SchedulerError::WaitingQueueFull {
                limit: self.config.max_waiting_requests,
            });
        }
        let requested = self
            .waiting_prompt_tokens
            .checked_add(prompt_tokens)
            .ok_or(SchedulerError::ArithmeticOverflow {
                field: "waiting prompt token admission",
            })?;
        if requested > self.config.max_waiting_prompt_tokens {
            observe_metric(
                &mut self.metrics_degraded,
                self.metrics.record_rejection(),
                "request rejection",
            );
            return Err(SchedulerError::WaitingTokenLimit {
                limit: self.config.max_waiting_prompt_tokens,
                requested,
            });
        }
        Ok(())
    }

    fn ensure_completion_backlog_empty(&self) -> SchedulerResult<()> {
        if self.completion_outbox.is_empty() {
            Ok(())
        } else {
            Err(SchedulerError::PendingCompletions {
                count: self.completion_outbox.len(),
            })
        }
    }

    fn ensure_completion_capacity(&self, needed: usize) -> SchedulerResult<()> {
        let total = self.completion_outbox.len().checked_add(needed).ok_or(
            SchedulerError::ArithmeticOverflow {
                field: "completion backlog size",
            },
        )?;
        if total <= self.completion_outbox_capacity {
            Ok(())
        } else {
            Err(SchedulerError::CompletionBacklogCapacity {
                limit: self.completion_outbox_capacity,
                pending: self.completion_outbox.len(),
                needed,
            })
        }
    }

    fn drain_completion_outbox_into(&mut self, target: &mut Vec<RequestCompletion>) {
        while let Some(completion) = self.completion_outbox.pop_front() {
            target.push(completion);
        }
    }

    fn close_failure(
        mut self,
        error: SchedulerError,
        mut recovered: Vec<RequestCompletion>,
        settlement_failures: Vec<RequestSettlementFailure>,
    ) -> SchedulerCloseFailure {
        while let Some(completion) = recovered.pop() {
            self.completion_outbox.push_front(completion);
        }
        SchedulerCloseFailure {
            error,
            scheduler: self,
            settlement_failures,
        }
    }

    fn record_index(&self, request_id: RequestId) -> Option<usize> {
        self.requests
            .iter()
            .position(|record| record.request_id == request_id)
    }

    fn peek_request_id(&self) -> SchedulerResult<RequestId> {
        self.next_request_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "request" })?;
        RequestId::new(self.next_request_id)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "request" })
    }

    fn advance_request_id(&mut self) -> SchedulerResult<()> {
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "request" })?;
        Ok(())
    }

    fn peek_iteration_id(&self) -> SchedulerResult<IterationId> {
        self.next_iteration_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "iteration" })?;
        IterationId::new(self.next_iteration_id)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "iteration" })
    }

    fn advance_iteration_id(&mut self) -> SchedulerResult<()> {
        self.next_iteration_id = self
            .next_iteration_id
            .checked_add(1)
            .ok_or(SchedulerError::IdentifierExhausted { kind: "iteration" })?;
        Ok(())
    }

    fn validate_now(&self, now_ns: u64) -> SchedulerResult<()> {
        if let Some(previous_ns) = self.last_now_ns {
            if now_ns < previous_ns {
                return Err(SchedulerError::ClockRegression {
                    previous_ns,
                    current_ns: now_ns,
                });
            }
        }
        Ok(())
    }

    fn observe_now(&mut self, now_ns: u64) -> SchedulerResult<()> {
        self.validate_now(now_ns)?;
        self.last_now_ns = Some(now_ns);
        Ok(())
    }

    fn refresh_metric_gauges(&mut self) {
        let gauges = self.current_gauges();
        observe_metric(
            &mut self.metrics_degraded,
            self.metrics.set_gauges(gauges),
            "scheduler gauges",
        );
    }

    fn current_gauges(&self) -> SchedulerGauges {
        let pool = self.pool.stats();
        SchedulerGauges {
            waiting_requests: self.waiting.len(),
            waiting_prompt_tokens: self.waiting_prompt_tokens,
            active_sequences: self.active_sequences,
            promised_kv_blocks: self.promised_kv_blocks,
            allocated_kv_blocks: pool.allocated_block_count(),
            physical_kv_blocks: pool.physical_block_count(),
            retained_terminal_requests: self.terminal.len(),
            pending_completions: self.completion_outbox.len(),
            completion_capacity: self.completion_outbox_capacity,
            accepting: self.accepting,
            outstanding_iterations: usize::from(self.inflight.is_some()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ReadyRequest {
    request_id: RequestId,
    ready_since_ns: u64,
    remaining_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Candidate {
    request_id: RequestId,
    kind: WorkKind,
    token_count: usize,
}

#[derive(Debug, Eq, PartialEq)]
struct CandidatePayload {
    candidate: Candidate,
    input_tokens: Vec<u32>,
    target_logical_length: usize,
    output_slot: Option<OutputSlot>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReservationSettlement {
    Rollback,
    Poison,
}

fn validate_descriptor(
    config: &SchedulerConfig,
    descriptor: &RequestDescriptor,
) -> SchedulerResult<usize> {
    if descriptor.prompt_token_ids.is_empty() {
        return Err(SchedulerError::InvalidConfiguration {
            field: "prompt_token_ids",
            reason: "must contain at least one token",
        });
    }
    if descriptor.max_new_tokens == 0 {
        return Err(SchedulerError::InvalidConfiguration {
            field: "max_new_tokens",
            reason: "must be greater than zero",
        });
    }
    let maximum = maximum_logical_length(descriptor)?;
    if maximum > config.max_sequence_tokens {
        return Err(SchedulerError::SequenceTokenLimit {
            limit: config.max_sequence_tokens,
            requested: maximum,
        });
    }
    Ok(maximum)
}

fn maximum_logical_length(descriptor: &RequestDescriptor) -> SchedulerResult<usize> {
    descriptor
        .prompt_token_ids
        .len()
        .checked_add(descriptor.max_new_tokens.saturating_sub(1))
        .ok_or(SchedulerError::ArithmeticOverflow {
            field: "request maximum logical length",
        })
}

fn transition(record: &mut RequestRecord, target: RequestState) -> SchedulerResult<()> {
    let source = record.state;
    if !valid_transition(source, target) {
        return Err(SchedulerError::InvalidStateTransition {
            request_id: record.request_id,
            from: source.name(),
            to: target.name(),
        });
    }
    record.state = target;
    Ok(())
}

fn valid_transition(source: RequestState, target: RequestState) -> bool {
    source == target
        || matches!(
            (source, target),
            (
                RequestState::Waiting,
                RequestState::Admitted | RequestState::Cancelled | RequestState::Failed
            ) | (
                RequestState::Admitted,
                RequestState::Prefilling | RequestState::Cancelled | RequestState::Failed
            ) | (
                RequestState::Prefilling,
                RequestState::Decoding
                    | RequestState::Finished
                    | RequestState::Cancelled
                    | RequestState::Failed
            ) | (
                RequestState::Decoding,
                RequestState::Finished | RequestState::Cancelled | RequestState::Failed
            ) | (RequestState::Cancelled, RequestState::Failed)
        )
}

fn copy_tokens(tokens: &[u32], resource: &'static str) -> SchedulerResult<Vec<u32>> {
    let mut owned = Vec::new();
    try_reserve_exact(&mut owned, tokens.len(), resource)?;
    owned.extend_from_slice(tokens);
    Ok(owned)
}

fn try_reserve_exact<T>(
    values: &mut Vec<T>,
    capacity: usize,
    resource: &'static str,
) -> SchedulerResult<()> {
    values
        .try_reserve_exact(capacity)
        .map_err(|_| SchedulerError::HostAllocation {
            resource,
            requested_elements: capacity,
        })
}

fn elapsed_ns(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn observe_metric(degraded: &mut bool, result: SchedulerResult<()>, operation: &'static str) {
    match result {
        Ok(()) => {}
        Err(_error) => {
            *degraded = true;
            let _ = operation;
        }
    }
}

#[cfg(test)]
mod tests {
    use rustinfer_runtime::paged_kv::KvLayout;

    use super::{ExecutionAbort, RequestDescriptor, RequestFinishReason, RequestId, Scheduler};
    use crate::{OverloadPolicy, SchedulerConfig, SchedulerError};

    fn test_scheduler() -> Scheduler {
        let config = SchedulerConfig {
            max_waiting_requests: 4,
            max_waiting_prompt_tokens: 32,
            max_active_sequences: 1,
            max_sequence_tokens: 32,
            iteration_token_budget: 4,
            max_prefill_chunk_tokens: 4,
            aging_threshold_ns: 10,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: Some(5),
            max_promised_kv_blocks: 8,
            metrics_window_samples: 4,
        };
        Scheduler::new(
            config,
            KvLayout::checked(1, 8, 1, 64).expect("test KV layout"),
        )
        .expect("test scheduler")
    }

    #[test]
    fn failed_tick_preserves_timeout_completion_until_exactly_once_recovery() {
        let mut scheduler = test_scheduler();
        scheduler
            .submit(RequestDescriptor::new(vec![1], 2), 0)
            .expect("active request");
        let timed_out = scheduler
            .submit(RequestDescriptor::new(vec![2], 2), 0)
            .expect("queued request")
            .request_id();
        let missing = RequestId::new(999).expect("nonzero test identity");
        scheduler.waiting.push_back(missing);

        assert!(matches!(
            scheduler.plan_iteration(5),
            Err(SchedulerError::UnknownRequest { request_id }) if request_id == missing
        ));
        assert_eq!(scheduler.pending_completion_count(), 1);
        assert!(matches!(
            scheduler.submit(RequestDescriptor::new(vec![3], 1), 5),
            Err(SchedulerError::PendingCompletions { count: 1 })
        ));

        let completion = scheduler
            .pop_pending_completion()
            .expect("recover timed-out completion");
        assert_eq!(completion.request_id(), timed_out);
        assert_eq!(completion.reason(), RequestFinishReason::AdmissionTimeout);
        assert!(scheduler.pop_pending_completion().is_none());
        assert_eq!(scheduler.waiting.pop_front(), Some(missing));

        let plan = scheduler
            .plan_iteration(5)
            .expect("mutation resumes after recovery")
            .into_parts()
            .0
            .expect("active request plan");
        scheduler
            .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 5)
            .expect("rollback active plan");
        scheduler.shutdown(5).expect("clean shutdown");
    }
}
