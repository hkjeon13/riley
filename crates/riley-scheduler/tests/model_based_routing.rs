//! C03-A CPU-only property coverage for scheduler output routing.
//!
//! This test never enables the CUDA feature or launches an executor.  It builds
//! public immutable plans, supplies a synthetic result in a deliberately
//! permuted output order, and compares the public commit events with an
//! independent `OutputSlot -> (request, generation step)` ledger.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind};

use riley_runtime::paged_kv::{KvBlockPoolStats, KvLayout};
use riley_scheduler::{
    ExecutionAbort, IterationId, IterationOutput, IterationPlan, IterationResult, OutputSlot,
    OverloadPolicy, RequestCompletion, RequestDescriptor, RequestFinishReason, RequestId,
    RequestSnapshot, RequestState, Scheduler, SchedulerCloseOutput, SchedulerConfig,
    SchedulerError, SchedulerMetricsSnapshot, TokenEvent,
};

const TRACE_COUNT: u64 = 10_000;
const FAULT_TRACE_COUNT: u64 = 10_000;
const MIXED_STAGE_TRACE_COUNT: u64 = 10_000;
const MAX_ITERATIONS_PER_TRACE: usize = 256;

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn bounded_usize(&mut self, upper_exclusive: usize) -> usize {
        assert!(upper_exclusive > 0, "test RNG bound must be nonzero");
        usize::try_from(self.next() % u64::try_from(upper_exclusive).expect("usize fits u64"))
            .expect("bounded random value fits usize")
    }
}

#[derive(Clone)]
struct LogicalRequest {
    label: u32,
    prompt: Vec<u32>,
    max_new_tokens: usize,
}

#[derive(Clone)]
struct TraceSpec {
    config: SchedulerConfig,
    requests: Vec<LogicalRequest>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExpectedOutput {
    slot: OutputSlot,
    request_id: RequestId,
    label: u32,
    generated_index: usize,
    token_id: u32,
}

fn trace_spec(seed: u64) -> TraceSpec {
    let mut random = Lcg(seed);
    let request_count = 2 + random.bounded_usize(5);
    let iteration_token_budget = 1 + random.bounded_usize(8);
    let max_prefill_chunk_tokens = 1 + random.bounded_usize(iteration_token_budget);
    let max_active_sequences = 1 + random.bounded_usize(4);
    let mut requests = Vec::with_capacity(request_count);
    for index in 0..request_count {
        let label = u32::try_from(index + 1).expect("small logical request label");
        let prompt_len = 1 + random.bounded_usize(6);
        let prompt = (0..prompt_len)
            .map(|offset| {
                label
                    .checked_mul(1_000)
                    .and_then(|base| base.checked_add(u32::try_from(offset).ok()?))
                    .expect("small symbolic prompt token")
            })
            .collect();
        requests.push(LogicalRequest {
            label,
            prompt,
            max_new_tokens: 1 + random.bounded_usize(4),
        });
    }
    TraceSpec {
        config: SchedulerConfig {
            max_waiting_requests: request_count,
            max_waiting_prompt_tokens: 64,
            max_active_sequences,
            max_sequence_tokens: 16,
            iteration_token_budget,
            max_prefill_chunk_tokens,
            aging_threshold_ns: 3,
            overload_policy: OverloadPolicy::Wait,
            admission_timeout_ns: None,
            max_promised_kv_blocks: 64,
            metrics_window_samples: 8,
        },
        requests,
    }
}

fn new_scheduler(config: SchedulerConfig) -> Scheduler {
    let layout = KvLayout::checked(1, 64, 1, 8).expect("valid C03-A symbolic KV layout");
    Scheduler::new(config, layout).expect("valid C03-A scheduler configuration")
}

fn token_for(label: u32, generated_index: usize) -> u32 {
    label
        .checked_mul(16)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("bounded symbolic output token")
}

fn expected_outputs(
    plan: &IterationPlan,
    label_by_request: &HashMap<RequestId, u32>,
    histories: &BTreeMap<u32, Vec<u32>>,
) -> Vec<ExpectedOutput> {
    let mut outputs = Vec::with_capacity(plan.output_slots().len());
    let mut slots = BTreeSet::new();
    let mut requests = BTreeSet::new();
    for item in plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items().iter())
    {
        let Some(slot) = item.output_slot() else {
            continue;
        };
        assert!(slots.insert(slot), "plan contains a duplicate output slot");
        assert!(
            requests.insert(item.request_id()),
            "plan requests more than one sampled output for one request"
        );
        let label = *label_by_request
            .get(&item.request_id())
            .expect("every planned request has a logical label");
        let generated_index = histories
            .get(&label)
            .expect("every logical label has a history")
            .len();
        outputs.push(ExpectedOutput {
            slot,
            request_id: item.request_id(),
            label,
            generated_index,
            token_id: token_for(label, generated_index),
        });
    }
    assert_eq!(slots.len(), plan.output_slots().len());
    assert_eq!(
        slots.into_iter().collect::<Vec<_>>(),
        plan.output_slots(),
        "plan slots must remain the canonical dense routing domain"
    );
    outputs
}

fn shuffle<T>(values: &mut [T], random: &mut Lcg) {
    for index in (1..values.len()).rev() {
        let other = random.bounded_usize(index + 1);
        values.swap(index, other);
    }
}

fn assert_token_events(expected: &[ExpectedOutput], events: &[TokenEvent], seed: u64, step: usize) {
    assert_eq!(
        events.len(),
        expected.len(),
        "seed {seed:#018x}, iteration {step}: output count differs from the plan ledger"
    );
    let mut expected_by_request = BTreeMap::new();
    for output in expected {
        assert!(
            expected_by_request
                .insert(output.request_id, (output.token_id, output.generated_index))
                .is_none(),
            "seed {seed:#018x}, iteration {step}: reference request is duplicated"
        );
    }
    let mut actual_by_request = BTreeMap::new();
    for event in events {
        assert!(
            actual_by_request
                .insert(
                    event.request_id(),
                    (event.token_id(), event.generated_index())
                )
                .is_none(),
            "seed {seed:#018x}, iteration {step}: scheduler emitted two tokens for one request"
        );
    }
    assert_eq!(
        actual_by_request, expected_by_request,
        "seed {seed:#018x}, iteration {step}: permuted sampled outputs routed to the wrong request"
    );
}

fn assert_completions(
    completions: &[RequestCompletion],
    label_by_request: &HashMap<RequestId, u32>,
    histories: &BTreeMap<u32, Vec<u32>>,
    terminal_labels: &mut BTreeSet<u32>,
    seed: u64,
    step: usize,
) {
    for completion in completions {
        let label = *label_by_request
            .get(&completion.request_id())
            .expect("every completion has a logical label");
        assert!(
            terminal_labels.insert(label),
            "seed {seed:#018x}, iteration {step}: terminal event was duplicated for label {label}"
        );
        assert_eq!(
            completion.generated_token_ids(),
            histories
                .get(&label)
                .expect("completed label retains its reference history"),
            "seed {seed:#018x}, iteration {step}: terminal history differs from the routing ledger"
        );
    }
}

struct TraceState {
    scheduler: Scheduler,
    label_by_request: HashMap<RequestId, u32>,
    histories: BTreeMap<u32, Vec<u32>>,
    terminal_labels: BTreeSet<u32>,
    random: Lcg,
    now_ns: u64,
}

impl TraceState {
    fn new(spec: &TraceSpec, seed: u64, reverse_submission_order: bool) -> Self {
        let mut scheduler = new_scheduler(spec.config.clone());
        let mut labels = (0..spec.requests.len()).collect::<Vec<_>>();
        if reverse_submission_order {
            labels.reverse();
        }
        let mut label_by_request = HashMap::new();
        for index in labels {
            let request = &spec.requests[index];
            let submission = scheduler
                .submit(
                    RequestDescriptor::new(request.prompt.clone(), request.max_new_tokens),
                    0,
                )
                .expect("bounded C03-A submission");
            assert!(
                label_by_request
                    .insert(submission.request_id(), request.label)
                    .is_none(),
                "scheduler issued a duplicate request ID"
            );
        }
        let histories = spec
            .requests
            .iter()
            .map(|request| (request.label, Vec::new()))
            .collect();
        Self {
            scheduler,
            label_by_request,
            histories,
            terminal_labels: BTreeSet::new(),
            random: Lcg(seed ^ 0xa5a5_5a5a_d3c3_b4b4),
            now_ns: 0,
        }
    }

    fn settle_iteration(&mut self, seed: u64, step: usize) -> bool {
        let planning = self
            .scheduler
            .plan_iteration(self.now_ns)
            .unwrap_or_else(|error| {
                panic!("seed {seed:#018x}, iteration {step}: plan failed: {error}")
            });
        let (plan, planning_completions) = planning.into_parts();
        assert!(
            planning_completions.is_empty(),
            "C03-A disables admission timeouts; planning cannot terminally complete a request"
        );
        let Some(plan) = plan else {
            return false;
        };
        let expected = expected_outputs(&plan, &self.label_by_request, &self.histories);
        let mut outputs = expected
            .iter()
            .map(|output| IterationOutput::new(output.slot, output.token_id, false))
            .collect::<Vec<_>>();
        shuffle(&mut outputs, &mut self.random);
        let result = IterationResult::new(plan.iteration_id(), outputs, 0, 0)
            .expect("unique synthetic result slots");
        let updates = self
            .scheduler
            .complete_iteration(&result, self.now_ns)
            .unwrap_or_else(|error| {
                panic!("seed {seed:#018x}, iteration {step}: commit failed: {error}")
            });
        assert!(
            updates.settlement_failures().is_empty(),
            "seed {seed:#018x}, iteration {step}: valid synthetic result was not fully settled"
        );
        assert_token_events(&expected, updates.token_events(), seed, step);
        for output in &expected {
            self.histories
                .get_mut(&output.label)
                .expect("every expected label has a history")
                .push(output.token_id);
        }
        assert_completions(
            updates.completions(),
            &self.label_by_request,
            &self.histories,
            &mut self.terminal_labels,
            seed,
            step,
        );
        self.now_ns = self.now_ns.checked_add(1).expect("bounded trace clock");
        true
    }

    fn close_and_collect(self, spec: &TraceSpec, seed: u64) -> BTreeMap<u32, Vec<u32>> {
        assert_eq!(
            self.terminal_labels.len(),
            spec.requests.len(),
            "seed {seed:#018x}: a request did not receive exactly one terminal event"
        );
        for request in &spec.requests {
            assert_eq!(
                self.histories[&request.label].len(),
                request.max_new_tokens,
                "seed {seed:#018x}: request {} history length drifted",
                request.label
            );
        }
        assert_eq!(self.scheduler.active_sequence_count(), 0);
        assert_eq!(self.scheduler.waiting_request_count(), 0);
        assert_eq!(self.scheduler.promised_kv_blocks(), 0);
        assert_eq!(self.scheduler.pool_stats().allocated_block_count(), 0);
        assert_eq!(self.scheduler.pending_completion_count(), 0);
        let closed = self
            .scheduler
            .close(self.now_ns, None)
            .unwrap_or_else(|failure| {
                panic!("seed {seed:#018x}: close failed: {}", failure.error())
            });
        assert!(closed.completions().is_empty());
        assert!(closed.settlement_failures().is_empty());
        let gauges = closed.final_metrics().gauges;
        assert_eq!(gauges.waiting_requests, 0);
        assert_eq!(gauges.waiting_prompt_tokens, 0);
        assert_eq!(gauges.active_sequences, 0);
        assert_eq!(gauges.promised_kv_blocks, 0);
        assert_eq!(gauges.allocated_kv_blocks, 0);
        assert_eq!(gauges.pending_completions, 0);
        assert_eq!(gauges.outstanding_iterations, 0);
        self.histories
    }
}

fn run_trace(
    spec: &TraceSpec,
    seed: u64,
    reverse_submission_order: bool,
) -> BTreeMap<u32, Vec<u32>> {
    let mut state = TraceState::new(spec, seed, reverse_submission_order);
    for step in 0..MAX_ITERATIONS_PER_TRACE {
        if !state.settle_iteration(seed, step) {
            return state.close_and_collect(spec, seed);
        }
    }
    panic!("seed {seed:#018x}: trace exceeded {MAX_ITERATIONS_PER_TRACE} iterations");
}

#[test]
fn ten_thousand_seeded_cpu_traces_route_permuted_outputs_by_slot() {
    for trace_index in 0..TRACE_COUNT {
        let seed = 0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(trace_index.wrapping_add(1));
        let spec = trace_spec(seed);
        let forward = run_trace(&spec, seed, false);
        let reverse = run_trace(&spec, seed, true);
        assert_eq!(
            forward, reverse,
            "seed {seed:#018x}: submission order changed another request's routed token history"
        );
    }
}

#[derive(Clone, Copy)]
enum FaultAction {
    DeferredCancelThenCommit,
    DeferredCancelThenAbortNotDispatched,
    AbortDeviceQuiescedMutationUnknown,
    TimeoutWaiting,
    InvalidFeedback,
}

impl FaultAction {
    fn for_trace_index(trace_index: u64) -> Self {
        match trace_index % 5 {
            0 => Self::DeferredCancelThenCommit,
            1 => Self::DeferredCancelThenAbortNotDispatched,
            2 => Self::AbortDeviceQuiescedMutationUnknown,
            3 => Self::TimeoutWaiting,
            _ => Self::InvalidFeedback,
        }
    }
}

struct FaultFixture {
    scheduler: Scheduler,
    plan: IterationPlan,
    expected: Vec<ExpectedOutput>,
    label_by_request: HashMap<RequestId, u32>,
}

struct SchedulerSurface {
    request_ids: Vec<RequestId>,
    request_snapshots: Vec<Option<RequestSnapshot>>,
    pool_stats: KvBlockPoolStats,
    metrics: SchedulerMetricsSnapshot,
    inflight_iteration: Option<IterationId>,
    pending_completions: usize,
}

#[derive(Default)]
struct TerminalLedger(BTreeMap<RequestId, (RequestFinishReason, Vec<u32>)>);

impl TerminalLedger {
    fn record(&mut self, completions: &[RequestCompletion]) {
        for completion in completions {
            assert!(
                self.0
                    .insert(
                        completion.request_id(),
                        (
                            completion.reason(),
                            completion.generated_token_ids().to_vec()
                        ),
                    )
                    .is_none(),
                "terminal completion must be emitted at most once per request"
            );
        }
    }
}

fn fault_config(max_active_sequences: usize, admission_timeout_ns: Option<u64>) -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 4,
        max_waiting_prompt_tokens: 16,
        max_active_sequences,
        max_sequence_tokens: 8,
        iteration_token_budget: 4,
        max_prefill_chunk_tokens: 1,
        aging_threshold_ns: 3,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns,
        max_promised_kv_blocks: 16,
        metrics_window_samples: 8,
    }
}

fn fault_fixture(seed: u64) -> FaultFixture {
    let mut scheduler = new_scheduler(fault_config(2, None));
    let requests = [(1_u32, 1_001_u32), (2_u32, 2_001_u32)];
    let mut order = [0_usize, 1];
    if seed & 1 == 1 {
        order.reverse();
    }
    let mut label_by_request = HashMap::new();
    for index in order {
        let (label, prompt_token) = requests[index];
        let submission = scheduler
            .submit(RequestDescriptor::new(vec![prompt_token], 1), 0)
            .expect("bounded fault-microtrace submission");
        assert!(
            label_by_request
                .insert(submission.request_id(), label)
                .is_none(),
            "scheduler issued a duplicate request ID"
        );
    }
    let planning = scheduler
        .plan_iteration(0)
        .expect("bounded fault-microtrace plan");
    let (plan, completions) = planning.into_parts();
    assert!(
        completions.is_empty(),
        "fault fixture does not enable admission timeouts"
    );
    let plan = plan.expect("fault fixture must produce work");
    let histories = BTreeMap::from([(1_u32, Vec::new()), (2_u32, Vec::new())]);
    let expected = expected_outputs(&plan, &label_by_request, &histories);
    assert_eq!(
        expected.len(),
        requests.len(),
        "fault fixture must give every active request one output slot"
    );
    FaultFixture {
        scheduler,
        plan,
        expected,
        label_by_request,
    }
}

fn shuffled_outputs(expected: &[ExpectedOutput], seed: u64) -> Vec<IterationOutput> {
    let mut outputs = expected
        .iter()
        .map(|output| IterationOutput::new(output.slot, output.token_id, false))
        .collect::<Vec<_>>();
    let mut random = Lcg(seed ^ 0x6a09_e667_f3bc_c909);
    shuffle(&mut outputs, &mut random);
    outputs
}

fn shuffled_result(
    plan: &IterationPlan,
    expected: &[ExpectedOutput],
    seed: u64,
) -> IterationResult {
    IterationResult::new(plan.iteration_id(), shuffled_outputs(expected, seed), 0, 0)
        .expect("unique fault-microtrace result slots")
}

fn completion_records(
    completions: &[RequestCompletion],
) -> BTreeMap<RequestId, (RequestFinishReason, Vec<u32>)> {
    let mut ledger = TerminalLedger::default();
    ledger.record(completions);
    ledger.0
}

fn selected_output(expected: &[ExpectedOutput], seed: u64) -> ExpectedOutput {
    expected[usize::try_from(seed >> 1).expect("u64 fits usize") % expected.len()]
}

fn assert_clean_close(scheduler: Scheduler, now_ns: u64, seed: u64) {
    let closed = scheduler.close(now_ns, None).unwrap_or_else(|failure| {
        panic!(
            "seed {seed:#018x}: clean fault-microtrace close failed: {}",
            failure.error()
        )
    });
    assert!(closed.completions().is_empty());
    assert!(closed.settlement_failures().is_empty());
    assert_closed_quiescent(&closed);
}

fn assert_closed_quiescent(closed: &SchedulerCloseOutput) {
    let gauges = closed.final_metrics().gauges;
    assert_eq!(gauges.waiting_requests, 0);
    assert_eq!(gauges.waiting_prompt_tokens, 0);
    assert_eq!(gauges.active_sequences, 0);
    assert_eq!(gauges.promised_kv_blocks, 0);
    assert_eq!(gauges.allocated_kv_blocks, 0);
    assert_eq!(gauges.pending_completions, 0);
    assert_eq!(gauges.outstanding_iterations, 0);
}

fn capture_surface(scheduler: &Scheduler, request_ids: Vec<RequestId>) -> SchedulerSurface {
    let request_snapshots = request_ids
        .iter()
        .map(|request_id| scheduler.request_snapshot(*request_id))
        .collect();
    SchedulerSurface {
        request_ids,
        request_snapshots,
        pool_stats: scheduler.pool_stats(),
        metrics: scheduler
            .metrics_snapshot()
            .expect("bounded fault-microtrace metric snapshot"),
        inflight_iteration: scheduler.inflight_iteration_id(),
        pending_completions: scheduler.pending_completion_count(),
    }
}

fn assert_surface_unchanged(scheduler: &Scheduler, expected: &SchedulerSurface) {
    let actual_snapshots = expected
        .request_ids
        .iter()
        .map(|request_id| scheduler.request_snapshot(*request_id))
        .collect::<Vec<_>>();
    assert_eq!(actual_snapshots, expected.request_snapshots);
    assert_eq!(scheduler.pool_stats(), expected.pool_stats);
    assert_eq!(
        scheduler
            .metrics_snapshot()
            .expect("bounded fault-microtrace metric snapshot"),
        expected.metrics
    );
    assert_eq!(
        scheduler.inflight_iteration_id(),
        expected.inflight_iteration
    );
    assert_eq!(
        scheduler.pending_completion_count(),
        expected.pending_completions
    );
}

fn run_deferred_cancel_then_commit(seed: u64) {
    let FaultFixture {
        mut scheduler,
        plan,
        expected,
        ..
    } = fault_fixture(seed);
    let cancelled = selected_output(&expected, seed);
    let cancellation = scheduler
        .cancel(cancelled.request_id, 1)
        .expect("defer cancellation for an in-flight output");
    assert!(cancellation.deferred_until_iteration_settles());
    assert!(cancellation.completion().is_none());
    let result = shuffled_result(&plan, &expected, seed);
    let updates = scheduler
        .complete_iteration(&result, 1)
        .expect("settle cancelled in-flight result");
    assert!(updates.settlement_failures().is_empty());
    let survivors = expected
        .iter()
        .copied()
        .filter(|output| output.request_id != cancelled.request_id)
        .collect::<Vec<_>>();
    assert_token_events(&survivors, updates.token_events(), seed, 0);
    let completions = completion_records(updates.completions());
    assert_eq!(completions.len(), expected.len());
    for output in expected {
        let expected_record = if output.request_id == cancelled.request_id {
            (RequestFinishReason::Cancelled, Vec::new())
        } else {
            (RequestFinishReason::Length, vec![output.token_id])
        };
        assert_eq!(completions.get(&output.request_id), Some(&expected_record));
    }
    assert_clean_close(scheduler, 2, seed);
}

fn run_deferred_cancel_then_abort_not_dispatched(seed: u64) {
    let FaultFixture {
        mut scheduler,
        plan,
        expected,
        label_by_request,
    } = fault_fixture(seed);
    let cancelled = selected_output(&expected, seed);
    let survivor = expected
        .iter()
        .copied()
        .find(|output| output.request_id != cancelled.request_id)
        .expect("two-request fault fixture has a survivor");
    assert!(
        scheduler
            .cancel(cancelled.request_id, 1)
            .expect("defer cancellation before an undispatched abort")
            .deferred_until_iteration_settles()
    );
    let aborted = scheduler
        .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 1)
        .expect("rollback undispatched iteration");
    assert!(aborted.token_events().is_empty());
    assert!(aborted.settlement_failures().is_empty());
    let mut terminal_ledger = TerminalLedger::default();
    terminal_ledger.record(aborted.completions());
    assert_eq!(terminal_ledger.0.len(), 1);
    assert_eq!(
        terminal_ledger.0.get(&cancelled.request_id),
        Some(&(RequestFinishReason::Cancelled, Vec::new()))
    );
    let planning = scheduler
        .plan_iteration(2)
        .expect("replan surviving request after rollback");
    let (retry_plan, retry_completions) = planning.into_parts();
    assert!(retry_completions.is_empty());
    let retry_plan = retry_plan.expect("survivor must be replanned after rollback");
    let histories = BTreeMap::from([(1_u32, Vec::new()), (2_u32, Vec::new())]);
    let retry_expected = expected_outputs(&retry_plan, &label_by_request, &histories);
    assert_eq!(retry_expected.len(), 1);
    assert_eq!(retry_expected[0].request_id, survivor.request_id);
    assert_eq!(retry_expected[0].generated_index, 0);
    assert_eq!(retry_expected[0].token_id, survivor.token_id);
    let retry_result = shuffled_result(&retry_plan, &retry_expected, seed ^ 0x9e37_79b9);
    let retried = scheduler
        .complete_iteration(&retry_result, 2)
        .expect("commit surviving request after rollback");
    assert!(retried.settlement_failures().is_empty());
    assert_token_events(&retry_expected, retried.token_events(), seed, 1);
    terminal_ledger.record(retried.completions());
    assert_eq!(terminal_ledger.0.len(), expected.len());
    assert_eq!(
        terminal_ledger.0.get(&survivor.request_id),
        Some(&(RequestFinishReason::Length, vec![survivor.token_id]))
    );
    assert_clean_close(scheduler, 3, seed);
}

fn run_device_quiesced_abort(seed: u64) {
    let FaultFixture {
        mut scheduler,
        plan,
        expected,
        ..
    } = fault_fixture(seed);
    let updates = scheduler
        .abort_iteration(
            plan.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            1,
        )
        .expect("host-side quiesced abort disposition");
    assert!(updates.token_events().is_empty());
    assert!(updates.settlement_failures().is_empty());
    let completions = completion_records(updates.completions());
    assert_eq!(completions.len(), expected.len());
    for output in expected {
        assert_eq!(
            completions.get(&output.request_id),
            Some(&(RequestFinishReason::ExecutorFailure, Vec::new()))
        );
    }
    assert_clean_close(scheduler, 2, seed);
}

fn run_timeout_waiting(seed: u64) {
    let mut scheduler = new_scheduler(fault_config(1, Some(1)));
    let requests = [(1_u32, 3_001_u32), (2_u32, 4_001_u32)];
    let mut order = [0_usize, 1];
    if seed & 1 == 1 {
        order.reverse();
    }
    let mut label_by_request = HashMap::new();
    let mut submitted = Vec::new();
    for index in order {
        let (label, prompt_token) = requests[index];
        let submission = scheduler
            .submit(RequestDescriptor::new(vec![prompt_token], 1), 0)
            .expect("bounded timeout-microtrace submission");
        label_by_request.insert(submission.request_id(), label);
        submitted.push(submission.request_id());
    }
    let active = submitted[0];
    let timed_out = submitted[1];
    let planning = scheduler
        .plan_iteration(1)
        .expect("timeout-microtrace plan");
    let (plan, timeout_completions) = planning.into_parts();
    assert_eq!(timeout_completions.len(), 1);
    let mut terminal_ledger = TerminalLedger::default();
    terminal_ledger.record(&timeout_completions);
    assert_eq!(
        terminal_ledger.0.get(&timed_out),
        Some(&(RequestFinishReason::AdmissionTimeout, Vec::new()))
    );
    let plan = plan.expect("active request remains runnable after waiting timeout");
    let histories = BTreeMap::from([(1_u32, Vec::new()), (2_u32, Vec::new())]);
    let expected = expected_outputs(&plan, &label_by_request, &histories);
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].request_id, active);
    let result = shuffled_result(&plan, &expected, seed);
    let updates = scheduler
        .complete_iteration(&result, 1)
        .expect("commit active request after waiting timeout");
    assert!(updates.settlement_failures().is_empty());
    assert_token_events(&expected, updates.token_events(), seed, 0);
    terminal_ledger.record(updates.completions());
    assert_eq!(terminal_ledger.0.len(), requests.len());
    assert_eq!(
        terminal_ledger.0.get(&active),
        Some(&(RequestFinishReason::Length, vec![expected[0].token_id]))
    );
    assert_clean_close(scheduler, 2, seed);
}

fn run_invalid_feedback(seed: u64) {
    let FaultFixture {
        mut scheduler,
        plan,
        expected,
        ..
    } = fault_fixture(seed);
    let surface = capture_surface(
        &scheduler,
        expected.iter().map(|output| output.request_id).collect(),
    );
    let stale_id = IterationId::new(
        plan.iteration_id()
            .get()
            .checked_add(1)
            .expect("bounded fault-microtrace iteration ID"),
    )
    .expect("nonzero stale iteration ID");
    let stale = IterationResult::new(stale_id, shuffled_outputs(&expected, seed), 0, 0)
        .expect("unique stale result slots");
    assert!(matches!(
        scheduler.complete_iteration(&stale, 1),
        Err(SchedulerError::UnexpectedIteration { .. })
    ));
    assert_surface_unchanged(&scheduler, &surface);

    let mut missing_outputs = shuffled_outputs(&expected, seed ^ 0x517c_c1b7);
    missing_outputs.pop();
    let missing = IterationResult::new(plan.iteration_id(), missing_outputs, 0, 0)
        .expect("unique missing-output result slots");
    assert!(matches!(
        scheduler.complete_iteration(&missing, 1),
        Err(SchedulerError::InvalidIterationResult { .. })
    ));
    assert_surface_unchanged(&scheduler, &surface);

    let mut unplanned_outputs = shuffled_outputs(&expected, seed ^ 0x94d0_49bb);
    unplanned_outputs[0] = IterationOutput::new(OutputSlot::new(99), expected[0].token_id, false);
    let unplanned = IterationResult::new(plan.iteration_id(), unplanned_outputs, 0, 0)
        .expect("unique unplanned-output result slots");
    assert!(matches!(
        scheduler.complete_iteration(&unplanned, 1),
        Err(SchedulerError::InvalidIterationResult { .. })
    ));
    assert_surface_unchanged(&scheduler, &surface);

    let result = shuffled_result(&plan, &expected, seed);
    let updates = scheduler
        .complete_iteration(&result, 1)
        .expect("valid result remains retryable after rejected feedback");
    assert!(updates.settlement_failures().is_empty());
    assert_token_events(&expected, updates.token_events(), seed, 0);
    let completions = completion_records(updates.completions());
    assert_eq!(completions.len(), expected.len());
    for output in expected {
        assert_eq!(
            completions.get(&output.request_id),
            Some(&(RequestFinishReason::Length, vec![output.token_id]))
        );
    }
    assert_clean_close(scheduler, 2, seed);
}

#[test]
fn ten_thousand_seeded_fault_microtraces_preserve_routing_and_quiescence() {
    for trace_index in 0..FAULT_TRACE_COUNT {
        let seed = 0xd1b5_4a32_d192_ed03_u64.wrapping_mul(trace_index.wrapping_add(1));
        match FaultAction::for_trace_index(trace_index) {
            FaultAction::DeferredCancelThenCommit => run_deferred_cancel_then_commit(seed),
            FaultAction::DeferredCancelThenAbortNotDispatched => {
                run_deferred_cancel_then_abort_not_dispatched(seed);
            }
            FaultAction::AbortDeviceQuiescedMutationUnknown => run_device_quiesced_abort(seed),
            FaultAction::TimeoutWaiting => run_timeout_waiting(seed),
            FaultAction::InvalidFeedback => run_invalid_feedback(seed),
        }
    }
}

const MIXED_STAGE_SEED: u64 = 0x51f1_edc3_70a5_9b61;
const MIXED_STAGE_SEED_FACTOR: u64 = 0x94d0_49bb_1331_11eb;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MixedStageAction {
    Commit,
    DeferredCancelDecoder,
}

impl MixedStageAction {
    const fn name(self) -> &'static str {
        match self {
            Self::Commit => "commit",
            Self::DeferredCancelDecoder => "deferred-cancel-decoder then commit",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MixedStageTraceV1 {
    seed: u64,
    decoder_max_new_tokens: usize,
    final_prefill_len: usize,
    action: MixedStageAction,
}

impl MixedStageTraceV1 {
    fn from_seed(seed: u64) -> Self {
        let mut random = Lcg(seed ^ 0x6c8e_9cf5_7093_2bd5);
        Self {
            seed,
            decoder_max_new_tokens: 2 + random.bounded_usize(3),
            final_prefill_len: 1 + random.bounded_usize(4),
            action: if random.next() & 1 == 0 {
                MixedStageAction::Commit
            } else {
                MixedStageAction::DeferredCancelDecoder
            },
        }
    }

    fn describe(self) -> String {
        format!(
            "v1 seed={:#018x}; decoder_max_new_tokens={}; final_prefill_len={}; \
             action={}; operations=[submit decoder, plan/commit decoder prime slot 0, \
             submit final-prefill, plan decode decoder slot 0 + final-prefill slot 1, \
             synthetic feedback [slot 1, slot 0], {}, close]",
            self.seed,
            self.decoder_max_new_tokens,
            self.final_prefill_len,
            self.action.name(),
            self.action.name(),
        )
    }
}

const RC1_REVERSE_COMMIT_TRACE_V1: MixedStageTraceV1 = MixedStageTraceV1 {
    seed: MIXED_STAGE_SEED,
    decoder_max_new_tokens: 2,
    final_prefill_len: 1,
    action: MixedStageAction::Commit,
};

const RC1_REVERSE_CANCEL_DECODER_TRACE_V1: MixedStageTraceV1 = MixedStageTraceV1 {
    seed: MIXED_STAGE_SEED,
    decoder_max_new_tokens: 2,
    final_prefill_len: 1,
    action: MixedStageAction::DeferredCancelDecoder,
};

const MIXED_STAGE_CLOSE_DECODER_THREE_TRACE_V1: MixedStageTraceV1 = MixedStageTraceV1 {
    seed: MIXED_STAGE_SEED ^ 1,
    decoder_max_new_tokens: 3,
    final_prefill_len: 2,
    action: MixedStageAction::Commit,
};

const MIXED_STAGE_CLOSE_DECODER_FOUR_TRACE_V1: MixedStageTraceV1 = MixedStageTraceV1 {
    seed: MIXED_STAGE_SEED ^ 2,
    decoder_max_new_tokens: 4,
    final_prefill_len: 4,
    action: MixedStageAction::Commit,
};

const MIXED_STAGE_CORPUS_V1: [MixedStageTraceV1; 4] = [
    RC1_REVERSE_COMMIT_TRACE_V1,
    RC1_REVERSE_CANCEL_DECODER_TRACE_V1,
    MIXED_STAGE_CLOSE_DECODER_THREE_TRACE_V1,
    MIXED_STAGE_CLOSE_DECODER_FOUR_TRACE_V1,
];

struct MixedStageFixture {
    scheduler: Scheduler,
    decoder: RequestId,
    final_prefill: RequestId,
    plan: IterationPlan,
    expected: Vec<ExpectedOutput>,
    label_by_request: HashMap<RequestId, u32>,
    histories: BTreeMap<u32, Vec<u32>>,
    terminal_labels: BTreeSet<u32>,
}

struct MixedStageState {
    scheduler: Scheduler,
    decoder: RequestId,
    label_by_request: HashMap<RequestId, u32>,
    histories: BTreeMap<u32, Vec<u32>>,
    terminal_labels: BTreeSet<u32>,
}

fn mixed_stage_config(trace: MixedStageTraceV1) -> SchedulerConfig {
    assert!(
        (2..=4).contains(&trace.decoder_max_new_tokens),
        "{}: decoder capacity escaped the bounded generator",
        trace.describe()
    );
    assert!(
        (1..=4).contains(&trace.final_prefill_len),
        "{}: final-prefill length escaped the bounded generator",
        trace.describe()
    );
    let iteration_token_budget = trace
        .final_prefill_len
        .checked_add(1)
        .expect("bounded mixed-stage iteration budget");
    SchedulerConfig {
        iteration_token_budget,
        max_prefill_chunk_tokens: trace.final_prefill_len,
        aging_threshold_ns: 100,
        ..fault_config(2, None)
    }
}

fn mixed_stage_final_prefill_prompt(length: usize) -> Vec<u32> {
    (0..length)
        .map(|offset| {
            2_001_u32
                .checked_add(u32::try_from(offset).expect("bounded prompt offset"))
                .expect("bounded symbolic final-prefill token")
        })
        .collect()
}

fn prime_mixed_stage_decoder(trace: MixedStageTraceV1) -> MixedStageState {
    let mut scheduler = new_scheduler(mixed_stage_config(trace));
    let decoder = scheduler
        .submit(
            RequestDescriptor::new(vec![1_001], trace.decoder_max_new_tokens),
            0,
        )
        .expect("prime mixed-stage decoder");
    let label_by_request = HashMap::from([(decoder.request_id(), 1_u32)]);
    let mut histories = BTreeMap::from([(1_u32, Vec::new()), (2_u32, Vec::new())]);
    let mut terminal_labels = BTreeSet::new();

    let planning = scheduler
        .plan_iteration(0)
        .expect("prime mixed-stage prefill");
    let (prime_plan, prime_completions) = planning.into_parts();
    assert!(prime_completions.is_empty());
    let prime_plan = prime_plan.expect("decoder prime plan");
    assert_eq!(prime_plan.prefill_items().len(), 1);
    assert!(prime_plan.decode_items().is_empty());
    assert_eq!(
        prime_plan.prefill_items()[0].request_id(),
        decoder.request_id()
    );
    assert_eq!(prime_plan.output_slots(), &[OutputSlot::new(0)]);
    let prime_expected = expected_outputs(&prime_plan, &label_by_request, &histories);
    let prime_result = IterationResult::new(
        prime_plan.iteration_id(),
        vec![IterationOutput::new(
            prime_expected[0].slot,
            prime_expected[0].token_id,
            false,
        )],
        0,
        0,
    )
    .expect("prime result has one unique slot");
    let prime_updates = scheduler
        .complete_iteration(&prime_result, 0)
        .expect("commit mixed-stage decoder prime");
    assert!(prime_updates.settlement_failures().is_empty());
    assert_token_events(&prime_expected, prime_updates.token_events(), trace.seed, 0);
    append_published_tokens(&mut histories, &prime_expected);
    assert_completions(
        prime_updates.completions(),
        &label_by_request,
        &histories,
        &mut terminal_labels,
        trace.seed,
        0,
    );
    assert_eq!(
        scheduler.request_state(decoder.request_id()),
        Some(RequestState::Decoding)
    );
    MixedStageState {
        scheduler,
        decoder: decoder.request_id(),
        label_by_request,
        histories,
        terminal_labels,
    }
}

fn mixed_stage_fixture(trace: MixedStageTraceV1) -> MixedStageFixture {
    let MixedStageState {
        mut scheduler,
        decoder,
        mut label_by_request,
        histories,
        terminal_labels,
    } = prime_mixed_stage_decoder(trace);
    let decoder_token = histories[&1][0];
    let final_prefill_prompt = mixed_stage_final_prefill_prompt(trace.final_prefill_len);
    let final_prefill = scheduler
        .submit(RequestDescriptor::new(final_prefill_prompt.clone(), 1), 1)
        .expect("submit final-prefill peer");
    assert!(
        label_by_request
            .insert(final_prefill.request_id(), 2)
            .is_none()
    );
    let planning = scheduler.plan_iteration(1).expect("plan mixed-stage trace");
    let (plan, completions) = planning.into_parts();
    assert!(completions.is_empty());
    let plan = plan.expect("mixed-stage plan");
    assert_eq!(plan.prefill_items().len(), 1);
    assert_eq!(plan.decode_items().len(), 1);
    assert_eq!(plan.decode_items()[0].request_id(), decoder);
    assert_eq!(plan.decode_items()[0].input_tokens(), &[decoder_token]);
    assert_eq!(
        plan.decode_items()[0].output_slot(),
        Some(OutputSlot::new(0))
    );
    assert_eq!(
        plan.prefill_items()[0].request_id(),
        final_prefill.request_id()
    );
    assert_eq!(
        plan.prefill_items()[0].input_tokens(),
        final_prefill_prompt.as_slice()
    );
    assert_eq!(
        plan.prefill_items()[0].output_slot(),
        Some(OutputSlot::new(1))
    );
    assert_eq!(
        plan.output_slots(),
        &[OutputSlot::new(0), OutputSlot::new(1)]
    );
    assert_eq!(
        plan.total_tokens(),
        trace.final_prefill_len + 1,
        "{}: mixed plan did not consume its bounded token budget",
        trace.describe()
    );
    let expected = expected_outputs(&plan, &label_by_request, &histories);
    assert_eq!(expected.len(), 2);
    MixedStageFixture {
        scheduler,
        decoder,
        final_prefill: final_prefill.request_id(),
        plan,
        expected,
        label_by_request,
        histories,
        terminal_labels,
    }
}

fn mixed_stage_expected_output(
    fixture: &MixedStageFixture,
    request_id: RequestId,
) -> ExpectedOutput {
    fixture
        .expected
        .iter()
        .copied()
        .find(|output| output.request_id == request_id)
        .expect("mixed-stage request has one output")
}

fn explicit_reverse_mixed_stage_result(fixture: &MixedStageFixture) -> IterationResult {
    let decoder = mixed_stage_expected_output(fixture, fixture.decoder);
    let final_prefill = mixed_stage_expected_output(fixture, fixture.final_prefill);
    assert_eq!(decoder.slot, OutputSlot::new(0));
    assert_eq!(final_prefill.slot, OutputSlot::new(1));
    IterationResult::new(
        fixture.plan.iteration_id(),
        vec![
            IterationOutput::new(final_prefill.slot, final_prefill.token_id, false),
            IterationOutput::new(decoder.slot, decoder.token_id, false),
        ],
        0,
        0,
    )
    .expect("mixed-stage reverse result has unique slots")
}

fn append_published_tokens(histories: &mut BTreeMap<u32, Vec<u32>>, outputs: &[ExpectedOutput]) {
    for output in outputs {
        histories
            .get_mut(&output.label)
            .expect("mixed-stage output label history")
            .push(output.token_id);
    }
}

fn mixed_stage_published_outputs(
    fixture: &MixedStageFixture,
    action: MixedStageAction,
) -> Vec<ExpectedOutput> {
    fixture
        .expected
        .iter()
        .copied()
        .filter(|output| {
            action == MixedStageAction::Commit || output.request_id == fixture.final_prefill
        })
        .collect()
}

fn expected_mixed_stage_iteration_completions(
    fixture: &MixedStageFixture,
    trace: MixedStageTraceV1,
) -> BTreeMap<RequestId, (RequestFinishReason, Vec<u32>)> {
    let mut expected = BTreeMap::from([(
        fixture.final_prefill,
        (RequestFinishReason::Length, vec![token_for(2, 0)]),
    )]);
    match trace.action {
        MixedStageAction::Commit if trace.decoder_max_new_tokens == 2 => {
            expected.insert(
                fixture.decoder,
                (
                    RequestFinishReason::Length,
                    vec![token_for(1, 0), token_for(1, 1)],
                ),
            );
        }
        MixedStageAction::DeferredCancelDecoder => {
            expected.insert(
                fixture.decoder,
                (RequestFinishReason::Cancelled, vec![token_for(1, 0)]),
            );
        }
        MixedStageAction::Commit => {}
    }
    expected
}

fn expected_mixed_stage_close_completions(
    fixture: &MixedStageFixture,
    trace: MixedStageTraceV1,
) -> BTreeMap<RequestId, (RequestFinishReason, Vec<u32>)> {
    if trace.action == MixedStageAction::Commit && trace.decoder_max_new_tokens > 2 {
        BTreeMap::from([(
            fixture.decoder,
            (
                RequestFinishReason::Cancelled,
                vec![token_for(1, 0), token_for(1, 1)],
            ),
        )])
    } else {
        BTreeMap::new()
    }
}

fn settle_mixed_stage_trace(fixture: &mut MixedStageFixture, trace: MixedStageTraceV1) {
    if trace.action == MixedStageAction::DeferredCancelDecoder {
        let cancellation = fixture
            .scheduler
            .cancel(fixture.decoder, 1)
            .expect("defer mixed-stage decoder cancellation");
        assert!(cancellation.deferred_until_iteration_settles());
        assert!(cancellation.completion().is_none());
    }
    let result = explicit_reverse_mixed_stage_result(fixture);
    let updates = fixture
        .scheduler
        .complete_iteration(&result, 1)
        .expect("commit mixed-stage reverse result");
    assert!(updates.settlement_failures().is_empty());
    let published = mixed_stage_published_outputs(fixture, trace.action);
    assert_token_events(&published, updates.token_events(), trace.seed, 1);
    append_published_tokens(&mut fixture.histories, &published);
    assert_completions(
        updates.completions(),
        &fixture.label_by_request,
        &fixture.histories,
        &mut fixture.terminal_labels,
        trace.seed,
        1,
    );
    assert_eq!(
        completion_records(updates.completions()),
        expected_mixed_stage_iteration_completions(fixture, trace),
        "{}: mixed-stage iteration terminals drifted",
        trace.describe()
    );
}

fn close_mixed_stage_trace(fixture: MixedStageFixture, trace: MixedStageTraceV1) {
    let expected_close = expected_mixed_stage_close_completions(&fixture, trace);
    let MixedStageFixture {
        scheduler,
        label_by_request,
        histories,
        mut terminal_labels,
        ..
    } = fixture;
    let closed = scheduler.close(2, None).unwrap_or_else(|failure| {
        panic!(
            "{}: mixed-stage close failed: {}",
            trace.describe(),
            failure.error()
        )
    });
    assert!(closed.settlement_failures().is_empty());
    assert_completions(
        closed.completions(),
        &label_by_request,
        &histories,
        &mut terminal_labels,
        trace.seed,
        2,
    );
    assert_eq!(
        completion_records(closed.completions()),
        expected_close,
        "{}: close terminals drifted",
        trace.describe()
    );
    assert_eq!(
        terminal_labels.len(),
        2,
        "{}: every mixed-stage request must terminally settle exactly once",
        trace.describe()
    );
    assert_closed_quiescent(&closed);
}

fn replay_mixed_stage_trace_inner(trace: MixedStageTraceV1) {
    let mut fixture = mixed_stage_fixture(trace);
    settle_mixed_stage_trace(&mut fixture, trace);
    close_mixed_stage_trace(fixture, trace);
}

fn replay_mixed_stage_trace(trace: MixedStageTraceV1) {
    let replay = catch_unwind(AssertUnwindSafe(|| replay_mixed_stage_trace_inner(trace)));
    assert!(
        replay.is_ok(),
        "C03-A bounded mixed-stage trace failed: {}",
        trace.describe()
    );
}

#[test]
fn replay_mixed_stage_corpus_v1() {
    for trace in MIXED_STAGE_CORPUS_V1 {
        replay_mixed_stage_trace(trace);
    }
}

#[test]
fn ten_thousand_seeded_mixed_stage_traces_preserve_routing_and_quiescence() {
    for trace_index in 0..MIXED_STAGE_TRACE_COUNT {
        let seed = MIXED_STAGE_SEED_FACTOR.wrapping_mul(trace_index.wrapping_add(1));
        replay_mixed_stage_trace(MixedStageTraceV1::from_seed(seed));
    }
}
