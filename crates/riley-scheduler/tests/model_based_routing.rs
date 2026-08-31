//! C03-A CPU-only property coverage for scheduler output routing.
//!
//! This test never enables the CUDA feature or launches an executor.  It builds
//! public immutable plans, supplies a synthetic result in a deliberately
//! permuted output order, and compares the public commit events with an
//! independent `OutputSlot -> (request, generation step)` ledger.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, OutputSlot, OverloadPolicy, RequestCompletion,
    RequestDescriptor, RequestId, Scheduler, SchedulerConfig, TokenEvent,
};

const TRACE_COUNT: u64 = 10_000;
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
