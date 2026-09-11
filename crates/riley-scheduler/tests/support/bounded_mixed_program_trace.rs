//! Canonical descriptors and a pure oracle for bounded settled-boundary
//! mixed-operation programs.
//!
//! This support module deliberately does not import `Scheduler` or `IterationPlan`.
//! The public-API adapter validates the plan projection separately, then gives
//! this oracle only opaque request IDs, an iteration ID, and public updates.
//! Feedback, token history, terminal events, and close expectations come from
//! the descriptor grammar alone.

use std::collections::{BTreeMap, BTreeSet};

use riley_scheduler::{
    CancellationOutcome, IterationId, IterationOutput, IterationResult, IterationUpdates,
    OutputSlot, RequestCompletion, RequestFinishReason, RequestId, SchedulerCloseOutput,
    TokenEvent,
};
use serde::{Deserialize, Serialize};

const DESCRIPTOR_FORMAT: &str = "riley.scheduler.bounded-mixed-program";
const DESCRIPTOR_FORMAT_VERSION: u8 = 1;
const DESCRIPTOR_TRACE_KIND: &str = "bounded-mixed-program-v1";
const MAX_LOGICAL_REQUESTS: usize = 4;
const MAX_LIVE_REQUESTS: usize = 3;
const MAX_PLAN_COMMITS: usize = 4;
const MAX_OPERATIONS: usize = 12;

/// One explicit operation in a bounded settled-boundary program.
///
/// Plan and commit are deliberately one grammar operation in this first raw
/// program slice. This keeps cancellation outside an in-flight iteration while
/// still allowing bounded valid orderings of submissions, settled cancels,
/// commits, and close.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum BoundedMixedProgramOperation {
    /// Admits one unique logical request with a fixed one-token prompt.
    Submit {
        /// Logical request label in the bounded program domain.
        label: u8,
        /// Output capacity in the bounded 1..=2 domain.
        max_new_tokens: u8,
    },
    /// Cancels a live request only when no plan is in flight.
    Cancel {
        /// Logical request label to terminally cancel.
        label: u8,
    },
    /// Plans every live request and commits exact permuted sampled feedback.
    PlanCommit {
        /// Exact permutation of the dense slots for this semantic plan.
        feedback_slot_order: Vec<u8>,
    },
    /// Consumes the scheduler after every preceding operation has settled.
    Close,
}

impl BoundedMixedProgramOperation {
    /// Returns a stable, human-readable operation spelling for failure reports.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Submit {
                label,
                max_new_tokens,
            } => format!("submit(label={label}, max_new={max_new_tokens})"),
            Self::Cancel { label } => format!("cancel(label={label})"),
            Self::PlanCommit {
                feedback_slot_order,
            } => format!("plan-commit(order={feedback_slot_order:?})"),
            Self::Close => "close".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BoundedMixedProgramDescriptorV1 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    source_seed: String,
    operations: Vec<BoundedMixedProgramOperation>,
}

/// Fully specified bounded settled-boundary program.
///
/// The source seed is provenance only. The explicit operation vector is the
/// replay input and is not reconstructed from that seed after parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedMixedProgramTrace {
    /// Source provenance retained in reports and fixtures.
    pub seed: u64,
    /// Complete raw program within the bounded V1 grammar.
    pub operations: Vec<BoundedMixedProgramOperation>,
}

impl BoundedMixedProgramTrace {
    /// Produces one valid bounded raw program from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut random = Lcg(seed ^ 0x8c3c_010d_a3e9_5167);
        let mut state = GenerationState::default();
        let mut operations = Vec::new();
        let mut next_label = 1_u8;

        append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
        match seed & 0b11 {
            0 | 1 => {
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    generated_max_new_tokens(&mut random),
                );
                if random.next() & 1 == 1 {
                    append_generated_submit(
                        &mut state,
                        &mut operations,
                        &mut next_label,
                        generated_max_new_tokens(&mut random),
                    );
                }
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
            }
            2 => {
                // Exercise two prefills in the first plan, followed by a
                // three-slot two-decode-plus-prefill mixed plan.
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 1);
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
            }
            3 => {
                // Exercise the grammar's widest first plan: three prefills,
                // then two decoders plus one final prefill.
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 1);
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 1);
                append_generated_plan_commit(&mut state, &mut operations, &mut random);
            }
            _ => unreachable!("two-bit topology selector"),
        }

        let mut cancellation_emitted = false;
        for _ in 0..random.bounded_usize(3) {
            if !cancellation_emitted && state.live_count() != 0 && random.next() & 1 == 1 {
                let labels = state.live_labels();
                let index = random.bounded_usize(labels.len());
                let label = labels[index];
                operations.push(BoundedMixedProgramOperation::Cancel { label });
                state.cancel(label);
                cancellation_emitted = true;
            }
            if random.next() & 1 == 1 {
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    generated_max_new_tokens(&mut random),
                );
            }
            if state.live_count() == 0 {
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    generated_max_new_tokens(&mut random),
                );
            }
            if state.live_count() == 0 {
                break;
            }
            append_generated_plan_commit(&mut state, &mut operations, &mut random);
        }

        if !cancellation_emitted && state.live_count() != 0 && random.next() & 1 == 1 {
            let labels = state.live_labels();
            let index = random.bounded_usize(labels.len());
            let label = labels[index];
            operations.push(BoundedMixedProgramOperation::Cancel { label });
            state.cancel(label);
        }
        operations.push(BoundedMixedProgramOperation::Close);

        let trace = Self { seed, operations };
        trace
            .validate()
            .expect("seeded bounded mixed program remains in the V1 grammar");
        trace
    }

    /// Returns the explicit program spelling without consulting a public plan.
    #[must_use]
    pub fn describe_operations(&self) -> String {
        self.operations
            .iter()
            .map(BoundedMixedProgramOperation::describe)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Returns deterministic, state-preserving simplifications for the bounded
    /// raw-program V1 grammar.
    ///
    /// The candidate order is settled-cancel removal, one logical submission
    /// removal (and its dependent settled cancel), direct identity feedback
    /// permutations, then one left-to-right adjacent inversion swap. Removing
    /// a cancel or submission recomputes later feedback slots from the
    /// descriptor's semantic label state: submission order first, with decode
    /// requests before prefills. This preserves the source feedback order for
    /// surviving labels and appends candidate-only labels in semantic slot
    /// order. Close, labels, plan topology, and output capacities do not
    /// change in this local reducer.
    #[must_use]
    pub fn shrink_candidates(&self) -> Vec<Self> {
        self.validate()
            .expect("bounded mixed shrinker requires a valid source descriptor");
        let source_rank = self.shrink_rank();
        let mut candidates = Vec::new();

        if let Some(candidate) = self.remove_settled_cancel() {
            push_reduction_candidate(&mut candidates, source_rank, &candidate);
        }
        for label in self.submission_labels() {
            let candidate = self.remove_logical_submission(label);
            push_reduction_candidate(&mut candidates, source_rank, &candidate);
        }
        for (operation_index, operation) in self.operations.iter().enumerate() {
            let BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            } = operation
            else {
                continue;
            };
            let canonical = canonical_slot_order(feedback_slot_order.len());
            if *feedback_slot_order != canonical {
                let mut candidate = self.clone();
                let BoundedMixedProgramOperation::PlanCommit {
                    feedback_slot_order,
                } = &mut candidate.operations[operation_index]
                else {
                    unreachable!("bounded mixed plan index stays a plan commit");
                };
                *feedback_slot_order = canonical;
                push_reduction_candidate(&mut candidates, source_rank, &candidate);
            }
        }
        for (operation_index, operation) in self.operations.iter().enumerate() {
            let BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            } = operation
            else {
                continue;
            };
            for slot_index in 0..feedback_slot_order.len().saturating_sub(1) {
                if feedback_slot_order[slot_index] > feedback_slot_order[slot_index + 1] {
                    let mut candidate = self.clone();
                    let BoundedMixedProgramOperation::PlanCommit {
                        feedback_slot_order,
                    } = &mut candidate.operations[operation_index]
                    else {
                        unreachable!("bounded mixed plan index stays a plan commit");
                    };
                    feedback_slot_order.swap(slot_index, slot_index + 1);
                    push_reduction_candidate(&mut candidates, source_rank, &candidate);
                }
            }
        }
        candidates
    }

    /// Returns the lexicographic rank used by the bounded raw-program local
    /// reducer: submitted logical requests, settled-cancel presence, then
    /// feedback-order inversions. It is not a general counterexample metric.
    #[must_use]
    pub fn shrink_rank(&self) -> (usize, usize, usize) {
        self.validate()
            .expect("bounded mixed shrink rank requires a valid descriptor");
        let mut submission_count = 0_usize;
        let mut cancellation_count = 0_usize;
        let mut feedback_inversions = 0_usize;
        for operation in &self.operations {
            match operation {
                BoundedMixedProgramOperation::Submit { .. } => submission_count += 1,
                BoundedMixedProgramOperation::Cancel { .. } => cancellation_count += 1,
                BoundedMixedProgramOperation::PlanCommit {
                    feedback_slot_order,
                } => feedback_inversions += inversion_count(feedback_slot_order),
                BoundedMixedProgramOperation::Close => {}
            }
        }
        (submission_count, cancellation_count, feedback_inversions)
    }

    /// Validates all syntax-independent V1 state-machine bounds and transitions.
    pub fn validate(&self) -> Result<(), String> {
        validate_operations(&self.operations)
    }

    fn descriptor(&self, case_id: &str) -> Result<BoundedMixedProgramDescriptorV1, String> {
        self.validate()?;
        validate_case_id(case_id)?;
        Ok(BoundedMixedProgramDescriptorV1 {
            format: DESCRIPTOR_FORMAT.to_owned(),
            format_version: DESCRIPTOR_FORMAT_VERSION,
            trace_kind: DESCRIPTOR_TRACE_KIND.to_owned(),
            case_id: case_id.to_owned(),
            source_seed: format!("0x{:016x}", self.seed),
            operations: self.operations.clone(),
        })
    }

    fn submission_labels(&self) -> Vec<u8> {
        self.operations
            .iter()
            .filter_map(|operation| match operation {
                BoundedMixedProgramOperation::Submit { label, .. } => Some(*label),
                BoundedMixedProgramOperation::Cancel { .. }
                | BoundedMixedProgramOperation::PlanCommit { .. }
                | BoundedMixedProgramOperation::Close => None,
            })
            .collect()
    }

    fn remove_settled_cancel(&self) -> Option<Self> {
        let label = self
            .operations
            .iter()
            .find_map(|operation| match operation {
                BoundedMixedProgramOperation::Cancel { label } => Some(*label),
                BoundedMixedProgramOperation::Submit { .. }
                | BoundedMixedProgramOperation::PlanCommit { .. }
                | BoundedMixedProgramOperation::Close => None,
            })?;
        Some(remap_after_removal(
            self,
            ReductionRemoval::SettledCancel { label },
        ))
    }

    fn remove_logical_submission(&self, label: u8) -> Self {
        remap_after_removal(self, ReductionRemoval::Submission { label })
    }
}

#[derive(Clone, Copy)]
enum ReductionRemoval {
    SettledCancel { label: u8 },
    Submission { label: u8 },
}

impl ReductionRemoval {
    const fn removes_submission(self, label: u8) -> bool {
        matches!(self, Self::Submission { label: removed } if removed == label)
    }

    const fn removes_cancel(self, label: u8) -> bool {
        matches!(
            self,
            Self::SettledCancel { label: removed } | Self::Submission { label: removed }
                if removed == label
        )
    }
}

#[derive(Clone)]
struct ReductionRequest {
    label: u8,
    max_new_tokens: u8,
    generated_tokens: u8,
    live: bool,
    submission_order: usize,
}

/// Small semantic state used only to rebase feedback after a local removal.
///
/// Unlike descriptor validation, this preserves public-plan slot semantics:
/// live labels are ordered by submission order inside decode and prefill
/// partitions. It intentionally permits an intermediate candidate to exceed a
/// grammar cap; the caller filters that candidate through `validate` before it
/// becomes observable.
#[derive(Default)]
struct ReductionState {
    requests: Vec<ReductionRequest>,
}

impl ReductionState {
    fn submit(&mut self, label: u8, max_new_tokens: u8) {
        let submission_order = self.requests.len();
        self.requests.push(ReductionRequest {
            label,
            max_new_tokens,
            generated_tokens: 0,
            live: true,
            submission_order,
        });
    }

    fn cancel(&mut self, label: u8) {
        let request = self
            .requests
            .iter_mut()
            .find(|request| request.label == label)
            .expect("valid bounded source cancellation has a submitted label");
        assert!(
            request.live,
            "valid bounded source cancellation has a live label"
        );
        request.live = false;
    }

    fn semantic_slot_labels(&self) -> Vec<u8> {
        let mut live = self
            .requests
            .iter()
            .filter(|request| request.live)
            .collect::<Vec<_>>();
        live.sort_unstable_by_key(|request| {
            (request.generated_tokens == 0, request.submission_order)
        });
        live.into_iter().map(|request| request.label).collect()
    }

    fn commit(&mut self) {
        for request in self.requests.iter_mut().filter(|request| request.live) {
            request.generated_tokens = request
                .generated_tokens
                .checked_add(1)
                .expect("bounded reduction generation count");
            if request.generated_tokens == request.max_new_tokens {
                request.live = false;
            }
        }
    }
}

fn remap_after_removal(
    trace: &BoundedMixedProgramTrace,
    removal: ReductionRemoval,
) -> BoundedMixedProgramTrace {
    let mut source_state = ReductionState::default();
    let mut candidate_state = ReductionState::default();
    let mut operations = Vec::with_capacity(trace.operations.len());

    for operation in &trace.operations {
        match operation {
            BoundedMixedProgramOperation::Submit {
                label,
                max_new_tokens,
            } => {
                source_state.submit(*label, *max_new_tokens);
                if !removal.removes_submission(*label) {
                    candidate_state.submit(*label, *max_new_tokens);
                    operations.push(operation.clone());
                }
            }
            BoundedMixedProgramOperation::Cancel { label } => {
                source_state.cancel(*label);
                if !removal.removes_cancel(*label) {
                    candidate_state.cancel(*label);
                    operations.push(operation.clone());
                }
            }
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            } => {
                let source_slots = source_state.semantic_slot_labels();
                let candidate_slots = candidate_state.semantic_slot_labels();
                operations.push(BoundedMixedProgramOperation::PlanCommit {
                    feedback_slot_order: remap_feedback_slot_order(
                        &source_slots,
                        feedback_slot_order,
                        &candidate_slots,
                    ),
                });
                source_state.commit();
                candidate_state.commit();
            }
            BoundedMixedProgramOperation::Close => operations.push(operation.clone()),
        }
    }

    BoundedMixedProgramTrace {
        seed: trace.seed,
        operations,
    }
}

fn remap_feedback_slot_order(
    source_slots: &[u8],
    source_feedback_slot_order: &[u8],
    candidate_slots: &[u8],
) -> Vec<u8> {
    let candidate_slot_by_label = candidate_slots
        .iter()
        .enumerate()
        .map(|(slot, label)| {
            (
                *label,
                u8::try_from(slot).expect("bounded reduction slot fits u8"),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut emitted_labels = BTreeSet::new();
    let mut remapped = Vec::with_capacity(candidate_slots.len());

    for source_slot in source_feedback_slot_order {
        let label = source_slots[usize::from(*source_slot)];
        if let Some(candidate_slot) = candidate_slot_by_label.get(&label) {
            if emitted_labels.insert(label) {
                remapped.push(*candidate_slot);
            }
        }
    }
    for label in candidate_slots {
        if emitted_labels.insert(*label) {
            remapped.push(
                *candidate_slot_by_label
                    .get(label)
                    .expect("candidate semantic slot has its label mapping"),
            );
        }
    }
    remapped
}

fn push_reduction_candidate(
    candidates: &mut Vec<BoundedMixedProgramTrace>,
    source_rank: (usize, usize, usize),
    candidate: &BoundedMixedProgramTrace,
) {
    if candidate.validate().is_err() {
        return;
    }
    let candidate = strict_round_trip_trace(candidate, "shrink-candidate");
    assert!(
        candidate.shrink_rank() < source_rank,
        "bounded mixed shrink candidate must strictly reduce its rank"
    );
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

/// A parsed corpus descriptor with its stable case identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedBoundedMixedProgramTrace {
    /// Durable corpus or failure-report identifier.
    pub case_id: String,
    /// Fully validated raw program.
    pub trace: BoundedMixedProgramTrace,
}

impl BoundedMixedProgramDescriptorV1 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("bounded mixed program descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err(
                "bounded mixed program descriptor format_version is unsupported".to_owned(),
            );
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND {
            return Err("bounded mixed program descriptor trace_kind is unsupported".to_owned());
        }
        validate_case_id(&self.case_id)?;
        parse_source_seed(&self.source_seed)?;
        validate_operations(&self.operations)
    }

    fn into_named_trace(self) -> Result<NamedBoundedMixedProgramTrace, String> {
        self.validate()?;
        Ok(NamedBoundedMixedProgramTrace {
            case_id: self.case_id,
            trace: BoundedMixedProgramTrace {
                seed: parse_source_seed(&self.source_seed)?,
                operations: self.operations,
            },
        })
    }
}

/// Serializes one valid raw program as exact canonical JSON with a newline.
#[must_use]
pub fn serialize_bounded_mixed_program_descriptor(
    case_id: &str,
    trace: &BoundedMixedProgramTrace,
) -> String {
    let descriptor = trace
        .descriptor(case_id)
        .expect("bounded mixed program is valid before serialization");
    descriptor_document(&descriptor)
}

/// Parses only exact canonical bounded mixed program V1 JSON documents.
pub fn parse_bounded_mixed_program_descriptor(
    document: &str,
) -> Result<NamedBoundedMixedProgramTrace, String> {
    let descriptor = serde_json::from_str::<BoundedMixedProgramDescriptorV1>(document)
        .map_err(|error| format!("bounded mixed program descriptor JSON is invalid: {error}"))?;
    let canonical = descriptor_document(&descriptor);
    if document != canonical {
        return Err("bounded mixed program descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_named_trace()
}

/// Greedily minimizes a reproducing bounded raw program over the fixed local
/// candidate order.
///
/// The source and every candidate are strict-codec round-tripped before the
/// caller's predicate sees them. The returned descriptor is only a local
/// minimum for this bounded grammar and predicate; it does not preserve a
/// panic site, payload, failure signature, or root cause.
pub fn minimize_bounded_mixed_program_trace<F>(
    trace: &BoundedMixedProgramTrace,
    mut reproduces: F,
) -> BoundedMixedProgramTrace
where
    F: FnMut(&BoundedMixedProgramTrace) -> bool,
{
    let mut minimized = strict_round_trip_trace(trace, "shrink-source");
    assert!(
        reproduces(&minimized),
        "bounded mixed minimization requires a reproducing source trace"
    );
    loop {
        let mut next = None;
        for candidate in minimized.shrink_candidates() {
            let candidate = strict_round_trip_trace(&candidate, "shrink-candidate");
            if reproduces(&candidate) {
                next = Some(candidate);
                break;
            }
        }
        let Some(candidate) = next else {
            return minimized;
        };
        minimized = candidate;
    }
}

fn strict_round_trip_trace(
    trace: &BoundedMixedProgramTrace,
    case_id: &str,
) -> BoundedMixedProgramTrace {
    let document = serialize_bounded_mixed_program_descriptor(case_id, trace);
    let parsed = parse_bounded_mixed_program_descriptor(&document)
        .expect("bounded mixed shrink descriptor stays strict-canonical");
    assert_eq!(parsed.case_id, case_id);
    assert_eq!(parsed.trace, *trace);
    parsed.trace
}

const CORPUS_DOCUMENTS_V1: [(&str, &str); 7] = [
    (
        "bounded-mixed-program-v1/three-slot-reverse-mixed.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/three-slot-reverse-mixed.json"
        ),
    ),
    (
        "bounded-mixed-program-v1/settled-cancel-replacement.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/settled-cancel-replacement.json"
        ),
    ),
    (
        "bounded-mixed-program-v1/mixed-close-live.json",
        include_str!("../corpus/output-routing/bounded-mixed-program-v1/mixed-close-live.json"),
    ),
    (
        "bounded-mixed-program-v1/two-prefill-three-slot-mixed.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/two-prefill-three-slot-mixed.json"
        ),
    ),
    (
        "bounded-mixed-program-v1/three-prefill-three-slot-mixed.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/three-prefill-three-slot-mixed.json"
        ),
    ),
    (
        "bounded-mixed-program-v1/settled-decoder-cancel.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/settled-decoder-cancel.json"
        ),
    ),
    (
        "bounded-mixed-program-v1/multi-live-decoder-close.json",
        include_str!(
            "../corpus/output-routing/bounded-mixed-program-v1/multi-live-decoder-close.json"
        ),
    ),
];

/// Loads the committed canonical corpus and rejects duplicate case identifiers.
#[must_use]
pub fn bounded_mixed_program_corpus() -> Vec<NamedBoundedMixedProgramTrace> {
    let mut case_ids = BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V1.len());
    for (path, document) in CORPUS_DOCUMENTS_V1 {
        let named = parse_bounded_mixed_program_descriptor(document).unwrap_or_else(|error| {
            panic!("{path}: bounded mixed program corpus is invalid: {error}")
        });
        assert!(
            case_ids.insert(named.case_id.clone()),
            "{path}: bounded mixed program corpus repeats case_id {:?}",
            named.case_id
        );
        corpus.push(named);
    }
    corpus
}

/// Expected work for one semantic output slot in a grammar-derived plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundedMixedProgramPlanItem {
    /// Logical label that owns this output.
    pub label: u8,
    /// Opaque public request binding.
    pub request_id: RequestId,
    /// Exact one-token public work input.
    pub input_token: u32,
    /// Expected dense output slot.
    pub output_slot: OutputSlot,
    /// Generated-token index emitted by this work item.
    pub generated_index: usize,
}

/// Public-plan projection derived only from the raw program state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundedMixedProgramExpectedPlan {
    /// Decode work in canonical ready/request order.
    pub decode_items: Vec<BoundedMixedProgramPlanItem>,
    /// Prefill work in canonical ready/request order.
    pub prefill_items: Vec<BoundedMixedProgramPlanItem>,
}

impl BoundedMixedProgramExpectedPlan {
    /// Returns the dense output-slot domain in grammar canonical order.
    #[must_use]
    pub fn output_slots(&self) -> Vec<OutputSlot> {
        self.decode_items
            .iter()
            .chain(&self.prefill_items)
            .map(|item| item.output_slot)
            .collect()
    }

    /// Returns the fixed one-token total for every selected live request.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.decode_items.len() + self.prefill_items.len()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TokenProjection {
    token_id: u32,
    generated_index: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CompletionProjection {
    reason: RequestFinishReason,
    generated_token_ids: Vec<u32>,
    completed_at_ns: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OraclePhase {
    Idle,
    InFlight,
    Closed,
}

struct OracleRequest {
    request_id: RequestId,
    max_new_tokens: usize,
    submission_order: usize,
    generated_token_ids: Vec<u32>,
    live: bool,
}

#[derive(Clone, Copy)]
struct PendingOutput {
    label: u8,
    request_id: RequestId,
    generated_index: usize,
    output_slot: OutputSlot,
}

/// Pure routing/lifecycle oracle for bounded settled-boundary programs.
///
/// This is not a general scheduler model. The grammar config ensures every
/// live request is admitted and selected on every plan, so this oracle only
/// models its bounded labels, token histories, deterministic plan projection,
/// terminal ledger, and descriptor-selected feedback permutations.
pub struct BoundedMixedProgramOracle {
    seed: u64,
    requests: BTreeMap<u8, OracleRequest>,
    pending: Option<Vec<PendingOutput>>,
    terminal: BTreeMap<RequestId, CompletionProjection>,
    next_submission_order: usize,
    phase: OraclePhase,
}

impl BoundedMixedProgramOracle {
    /// Creates an unbound oracle before any public request IDs are known.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            requests: BTreeMap::new(),
            pending: None,
            terminal: BTreeMap::new(),
            next_submission_order: 0,
            phase: OraclePhase::Idle,
        }
    }

    /// Binds one descriptor submission to its opaque public request ID.
    pub fn bind_submit(&mut self, label: u8, max_new_tokens: u8, request_id: RequestId) {
        self.assert_phase(OraclePhase::Idle);
        assert!(
            (1..=u8::try_from(MAX_LOGICAL_REQUESTS).expect("bounded label")).contains(&label),
            "seed {:#018x}: bounded program label escaped its grammar",
            self.seed
        );
        assert!(
            (1..=2).contains(&max_new_tokens),
            "seed {:#018x}: bounded program output capacity escaped its grammar",
            self.seed
        );
        assert!(
            self.live_count() < MAX_LIVE_REQUESTS,
            "seed {:#018x}: bounded program exceeded its live request cap",
            self.seed
        );
        assert!(
            self.requests
                .insert(
                    label,
                    OracleRequest {
                        request_id,
                        max_new_tokens: usize::from(max_new_tokens),
                        submission_order: self.next_submission_order,
                        generated_token_ids: Vec::with_capacity(usize::from(max_new_tokens)),
                        live: true,
                    },
                )
                .is_none(),
            "seed {:#018x}: bounded program bound the same logical label twice",
            self.seed
        );
        self.next_submission_order = self
            .next_submission_order
            .checked_add(1)
            .expect("bounded program submission ordinal");
    }

    /// Derives and opens the exact expected public plan before feedback exists.
    #[must_use]
    pub fn begin_plan(&mut self) -> BoundedMixedProgramExpectedPlan {
        self.assert_phase(OraclePhase::Idle);
        let mut live = self
            .requests
            .iter()
            .filter(|(_, request)| request.live)
            .map(|(label, request)| (*label, request))
            .collect::<Vec<_>>();
        assert!(
            !live.is_empty(),
            "seed {:#018x}: bounded program attempted to plan without live work",
            self.seed
        );
        live.sort_unstable_by_key(|(_, request)| request.submission_order);
        let (decode, prefill): (Vec<_>, Vec<_>) = live
            .into_iter()
            .partition(|(_, request)| !request.generated_token_ids.is_empty());
        let mut pending = Vec::with_capacity(decode.len() + prefill.len());
        let mut decode_items = Vec::with_capacity(decode.len());
        let mut prefill_items = Vec::with_capacity(prefill.len());
        for (label, request) in decode.into_iter().chain(prefill) {
            let slot =
                OutputSlot::new(u32::try_from(pending.len()).expect("bounded program output slot"));
            let generated_index = request.generated_token_ids.len();
            let item = BoundedMixedProgramPlanItem {
                label,
                request_id: request.request_id,
                input_token: if generated_index == 0 {
                    symbolic_prompt_token(label)
                } else {
                    *request
                        .generated_token_ids
                        .last()
                        .expect("bounded decoder has one prior generated token")
                },
                output_slot: slot,
                generated_index,
            };
            pending.push(PendingOutput {
                label,
                request_id: request.request_id,
                generated_index,
                output_slot: slot,
            });
            if generated_index == 0 {
                prefill_items.push(item);
            } else {
                decode_items.push(item);
            }
        }
        self.pending = Some(pending);
        self.phase = OraclePhase::InFlight;
        BoundedMixedProgramExpectedPlan {
            decode_items,
            prefill_items,
        }
    }

    /// Builds descriptor-selected feedback without reading a public plan.
    #[must_use]
    pub fn feedback(
        &self,
        iteration_id: IterationId,
        feedback_slot_order: &[u8],
    ) -> IterationResult {
        self.assert_phase(OraclePhase::InFlight);
        let pending = self
            .pending
            .as_ref()
            .expect("bounded program oracle retained its in-flight projection");
        validate_slot_order(feedback_slot_order, pending.len(), "feedback_slot_order")
            .unwrap_or_else(|error| {
                panic!(
                    "seed {:#018x}: bounded program feedback permutation is invalid: {error}",
                    self.seed
                )
            });
        let outputs = feedback_slot_order
            .iter()
            .map(|slot| {
                let pending_item = pending[usize::from(*slot)];
                IterationOutput::new(
                    pending_item.output_slot,
                    symbolic_token(pending_item.label, pending_item.generated_index),
                    false,
                )
            })
            .collect();
        IterationResult::new(iteration_id, outputs, 0, 0)
            .expect("bounded program oracle feedback has unique slots")
    }

    /// Compares one valid settled plan commit with the grammar-derived ledger.
    pub fn record_plan_commit(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(OraclePhase::InFlight);
        let pending = self
            .pending
            .take()
            .expect("bounded program oracle retained its in-flight projection");
        let mut expected_tokens = BTreeMap::new();
        let mut expected_completions = BTreeMap::new();
        for item in &pending {
            let request = self.request(item.label);
            assert!(request.live);
            assert_eq!(request.request_id, item.request_id);
            assert_eq!(request.generated_token_ids.len(), item.generated_index);
            let token_id = symbolic_token(item.label, item.generated_index);
            assert!(
                expected_tokens
                    .insert(
                        item.request_id,
                        TokenProjection {
                            token_id,
                            generated_index: item.generated_index,
                        },
                    )
                    .is_none(),
                "seed {:#018x}: bounded program repeated a pending request",
                self.seed
            );
            let mut history = request.generated_token_ids.clone();
            history.push(token_id);
            if history.len() == request.max_new_tokens {
                assert!(
                    expected_completions
                        .insert(
                            item.request_id,
                            CompletionProjection {
                                reason: RequestFinishReason::Length,
                                generated_token_ids: history,
                                completed_at_ns: now_ns,
                            },
                        )
                        .is_none(),
                    "seed {:#018x}: bounded program repeated a terminal request",
                    self.seed
                );
            }
        }
        self.assert_updates(updates, &expected_tokens, expected_completions);
        for item in pending {
            let token_id = symbolic_token(item.label, item.generated_index);
            let request = self.request_mut(item.label);
            request.generated_token_ids.push(token_id);
            if request.generated_token_ids.len() == request.max_new_tokens {
                request.live = false;
            }
        }
        self.phase = OraclePhase::Idle;
    }

    /// Compares one immediate settled-boundary cancellation outcome.
    pub fn record_settled_cancel(&mut self, label: u8, outcome: &CancellationOutcome, now_ns: u64) {
        self.assert_phase(OraclePhase::Idle);
        let request = self.request(label);
        assert!(
            request.live,
            "seed {:#018x}: bounded program cancelled a non-live label {label}",
            self.seed
        );
        assert_eq!(outcome.request_id(), request.request_id);
        assert!(!outcome.deferred_until_iteration_settles());
        assert!(!outcome.already_terminal());
        let expected = BTreeMap::from([(
            request.request_id,
            CompletionProjection {
                reason: RequestFinishReason::Cancelled,
                generated_token_ids: request.generated_token_ids.clone(),
                completed_at_ns: now_ns,
            },
        )]);
        let completion = outcome
            .completion()
            .expect("settled-boundary cancellation must publish one completion");
        self.assert_completions(std::slice::from_ref(completion), expected);
        self.request_mut(label).live = false;
    }

    /// Compares consuming close with every still-live request in the program.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(OraclePhase::Idle);
        assert!(
            closed.settlement_failures().is_empty(),
            "seed {:#018x}: bounded program close contained a settlement failure",
            self.seed
        );
        let expected = self
            .requests
            .values()
            .filter(|request| request.live)
            .map(|request| {
                (
                    request.request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: request.generated_token_ids.clone(),
                        completed_at_ns: now_ns,
                    },
                )
            })
            .collect();
        self.assert_completions(closed.completions(), expected);
        for request in self.requests.values_mut() {
            request.live = false;
        }
        assert_eq!(
            self.terminal.len(),
            self.requests.len(),
            "seed {:#018x}: bounded program close did not terminally settle every submission",
            self.seed
        );
        self.phase = OraclePhase::Closed;
    }

    /// Asserts that the adapter reached the exact terminal close state.
    pub fn assert_closed(&self) {
        self.assert_phase(OraclePhase::Closed);
        assert_eq!(
            self.terminal.len(),
            self.requests.len(),
            "seed {:#018x}: bounded program terminal ledger is incomplete",
            self.seed
        );
    }

    fn assert_updates(
        &mut self,
        updates: &IterationUpdates,
        expected_tokens: &BTreeMap<RequestId, TokenProjection>,
        expected_completions: BTreeMap<RequestId, CompletionProjection>,
    ) {
        assert!(
            updates.settlement_failures().is_empty(),
            "seed {:#018x}: bounded program update contained a settlement failure",
            self.seed
        );
        assert_eq!(
            &token_projections(updates.token_events()),
            expected_tokens,
            "seed {:#018x}: bounded program token routing drifted",
            self.seed
        );
        self.assert_completions(updates.completions(), expected_completions);
    }

    fn assert_completions(
        &mut self,
        completions: &[RequestCompletion],
        expected_completions: BTreeMap<RequestId, CompletionProjection>,
    ) {
        assert_eq!(
            completion_projections(completions),
            expected_completions,
            "seed {:#018x}: bounded program terminal ledger drifted",
            self.seed
        );
        for (request_id, completion) in expected_completions {
            assert!(
                self.terminal.insert(request_id, completion).is_none(),
                "seed {:#018x}: bounded program emitted a duplicate terminal request",
                self.seed
            );
        }
    }

    fn assert_phase(&self, expected: OraclePhase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: bounded program oracle phase drifted",
            self.seed
        );
    }

    fn live_count(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.live)
            .count()
    }

    fn request(&self, label: u8) -> &OracleRequest {
        self.requests.get(&label).unwrap_or_else(|| {
            panic!(
                "seed {:#018x}: bounded program label {label} is unbound",
                self.seed
            )
        })
    }

    fn request_mut(&mut self, label: u8) -> &mut OracleRequest {
        self.requests.get_mut(&label).unwrap_or_else(|| {
            panic!(
                "seed {:#018x}: bounded program label {label} is unbound",
                self.seed
            )
        })
    }
}

/// Returns the one-token prompt bound to a logical program label.
#[must_use]
pub fn symbolic_prompt_token(label: u8) -> u32 {
    1_000_u32
        .checked_add(u32::from(label))
        .expect("bounded program symbolic prompt token")
}

/// Returns the exact symbolic sampled token for one logical generation index.
#[must_use]
pub fn symbolic_token(label: u8, generated_index: usize) -> u32 {
    u32::from(label)
        .checked_mul(16)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("bounded program symbolic output token")
}

#[derive(Clone)]
struct ValidationRequest {
    max_new_tokens: usize,
    generated_tokens: usize,
    live: bool,
}

// Keeping the full finite transition table together makes grammar review
// possible without following state mutation across helper calls.
#[allow(clippy::too_many_lines)]
fn validate_operations(operations: &[BoundedMixedProgramOperation]) -> Result<(), String> {
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(format!(
            "bounded mixed program descriptor operations must contain 1..={MAX_OPERATIONS} entries"
        ));
    }
    let mut requests = BTreeMap::<u8, ValidationRequest>::new();
    let mut close_seen = false;
    let mut plan_commits = 0_usize;
    let mut cancellation_count = 0_usize;
    let mut mixed_plan_seen = false;
    for (index, operation) in operations.iter().enumerate() {
        if close_seen {
            return Err(
                "bounded mixed program descriptor contains an operation after close".to_owned(),
            );
        }
        match operation {
            BoundedMixedProgramOperation::Submit {
                label,
                max_new_tokens,
            } => {
                validate_label(*label)?;
                if !(1..=2).contains(max_new_tokens) {
                    return Err(
                        "bounded mixed program descriptor max_new_tokens must be in 1..=2"
                            .to_owned(),
                    );
                }
                if requests.len() >= MAX_LOGICAL_REQUESTS {
                    return Err(
                        "bounded mixed program descriptor exceeds its unique request cap"
                            .to_owned(),
                    );
                }
                if live_request_count(&requests) >= MAX_LIVE_REQUESTS {
                    return Err(
                        "bounded mixed program descriptor exceeds its live request cap".to_owned(),
                    );
                }
                if requests
                    .insert(
                        *label,
                        ValidationRequest {
                            max_new_tokens: usize::from(*max_new_tokens),
                            generated_tokens: 0,
                            live: true,
                        },
                    )
                    .is_some()
                {
                    return Err(
                        "bounded mixed program descriptor submits one label more than once"
                            .to_owned(),
                    );
                }
            }
            BoundedMixedProgramOperation::Cancel { label } => {
                validate_label(*label)?;
                if cancellation_count >= 1 {
                    return Err(
                        "bounded mixed program descriptor allows at most one settled cancellation"
                            .to_owned(),
                    );
                }
                let Some(request) = requests.get_mut(label) else {
                    return Err(
                        "bounded mixed program descriptor cancels an unsubmitted label".to_owned(),
                    );
                };
                if !request.live {
                    return Err(
                        "bounded mixed program descriptor cancels a non-live label".to_owned()
                    );
                }
                request.live = false;
                cancellation_count += 1;
            }
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            } => {
                let live_count = live_request_count(&requests);
                if live_count == 0 {
                    return Err(
                        "bounded mixed program descriptor plans without live requests".to_owned(),
                    );
                }
                if plan_commits >= MAX_PLAN_COMMITS {
                    return Err(
                        "bounded mixed program descriptor exceeds its plan commit cap".to_owned(),
                    );
                }
                validate_slot_order(feedback_slot_order, live_count, "feedback_slot_order")?;
                let has_decode = requests
                    .values()
                    .any(|request| request.live && request.generated_tokens != 0);
                let has_prefill = requests
                    .values()
                    .any(|request| request.live && request.generated_tokens == 0);
                mixed_plan_seen |= has_decode && has_prefill;
                for request in requests.values_mut().filter(|request| request.live) {
                    request.generated_tokens =
                        request.generated_tokens.checked_add(1).ok_or_else(|| {
                            "bounded mixed program descriptor generation count overflowed"
                                .to_owned()
                        })?;
                    if request.generated_tokens == request.max_new_tokens {
                        request.live = false;
                    }
                }
                plan_commits += 1;
            }
            BoundedMixedProgramOperation::Close => {
                if index + 1 != operations.len() {
                    return Err(
                        "bounded mixed program descriptor close must be the final operation"
                            .to_owned(),
                    );
                }
                close_seen = true;
                for request in requests.values_mut() {
                    request.live = false;
                }
            }
        }
    }
    if !close_seen {
        return Err("bounded mixed program descriptor must end with close".to_owned());
    }
    if plan_commits < 2 {
        return Err(
            "bounded mixed program descriptor requires at least two plan commits".to_owned(),
        );
    }
    if !mixed_plan_seen {
        return Err(
            "bounded mixed program descriptor requires one decode plus prefill mixed plan"
                .to_owned(),
        );
    }
    Ok(())
}

fn live_request_count(requests: &BTreeMap<u8, ValidationRequest>) -> usize {
    requests.values().filter(|request| request.live).count()
}

fn validate_slot_order(order: &[u8], slot_count: usize, field: &str) -> Result<(), String> {
    if order.len() != slot_count {
        return Err(format!(
            "bounded mixed program descriptor {field} must contain exactly {slot_count} slots"
        ));
    }
    let mut seen = BTreeSet::new();
    for slot in order {
        if usize::from(*slot) >= slot_count {
            return Err(format!(
                "bounded mixed program descriptor {field} contains an out-of-range slot"
            ));
        }
        if !seen.insert(*slot) {
            return Err(format!(
                "bounded mixed program descriptor {field} contains a duplicate slot"
            ));
        }
    }
    Ok(())
}

fn canonical_slot_order(slot_count: usize) -> Vec<u8> {
    (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("bounded mixed slot fits u8"))
        .collect()
}

fn inversion_count(order: &[u8]) -> usize {
    order
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            order[index + 1..]
                .iter()
                .filter(|other| slot > *other)
                .count()
        })
        .sum()
}

fn validate_label(label: u8) -> Result<(), String> {
    if !(1..=u8::try_from(MAX_LOGICAL_REQUESTS).expect("bounded label")).contains(&label) {
        return Err("bounded mixed program descriptor label must be in 1..=4".to_owned());
    }
    Ok(())
}

fn validate_case_id(case_id: &str) -> Result<(), String> {
    let bytes = case_id.as_bytes();
    if !(1..=96).contains(&bytes.len())
        || bytes.first() == Some(&b'-')
        || bytes.last() == Some(&b'-')
        || !bytes
            .iter()
            .copied()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    {
        return Err(
            "bounded mixed program descriptor case_id must be a bounded lowercase identifier"
                .to_owned(),
        );
    }
    Ok(())
}

fn parse_source_seed(source_seed: &str) -> Result<u64, String> {
    let Some(hex) = source_seed.strip_prefix("0x") else {
        return Err("bounded mixed program descriptor source_seed must start with 0x".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(
            "bounded mixed program descriptor source_seed must be 16 lowercase hexadecimal digits"
                .to_owned(),
        );
    }
    u64::from_str_radix(hex, 16)
        .map_err(|_| "bounded mixed program descriptor source_seed does not fit u64".to_owned())
}

fn descriptor_document(descriptor: &BoundedMixedProgramDescriptorV1) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("bounded mixed program descriptor serializes");
    document.push('\n');
    document
}

fn token_projections(events: &[TokenEvent]) -> BTreeMap<RequestId, TokenProjection> {
    let mut projections = BTreeMap::new();
    for event in events {
        assert!(
            projections
                .insert(
                    event.request_id(),
                    TokenProjection {
                        token_id: event.token_id(),
                        generated_index: event.generated_index(),
                    },
                )
                .is_none(),
            "bounded program public updates emitted duplicate token events for one request"
        );
    }
    projections
}

fn completion_projections(
    completions: &[RequestCompletion],
) -> BTreeMap<RequestId, CompletionProjection> {
    let mut projections = BTreeMap::new();
    for completion in completions {
        assert!(
            projections
                .insert(
                    completion.request_id(),
                    CompletionProjection {
                        reason: completion.reason(),
                        generated_token_ids: completion.generated_token_ids().to_vec(),
                        completed_at_ns: completion.completed_at_ns(),
                    },
                )
                .is_none(),
            "bounded program public updates emitted duplicate terminal completions"
        );
    }
    projections
}

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
        assert!(
            upper_exclusive != 0,
            "bounded program RNG bound must be nonzero"
        );
        usize::try_from(self.next() % u64::try_from(upper_exclusive).expect("usize fits u64"))
            .expect("bounded program RNG value fits usize")
    }
}

#[derive(Clone)]
struct GeneratedRequest {
    label: u8,
    max_new_tokens: usize,
    generated_tokens: usize,
    live: bool,
}

#[derive(Default)]
struct GenerationState {
    requests: Vec<GeneratedRequest>,
}

impl GenerationState {
    fn live_count(&self) -> usize {
        self.requests.iter().filter(|request| request.live).count()
    }

    fn live_labels(&self) -> Vec<u8> {
        self.requests
            .iter()
            .filter(|request| request.live)
            .map(|request| request.label)
            .collect()
    }

    fn submit(&mut self, label: u8, max_new_tokens: u8) {
        assert!(self.requests.len() < MAX_LOGICAL_REQUESTS);
        assert!(self.live_count() < MAX_LIVE_REQUESTS);
        self.requests.push(GeneratedRequest {
            label,
            max_new_tokens: usize::from(max_new_tokens),
            generated_tokens: 0,
            live: true,
        });
    }

    fn cancel(&mut self, label: u8) {
        let request = self
            .requests
            .iter_mut()
            .find(|request| request.label == label)
            .expect("generated bounded program cancels a submitted label");
        assert!(request.live);
        request.live = false;
    }

    fn commit(&mut self) {
        for request in self.requests.iter_mut().filter(|request| request.live) {
            request.generated_tokens += 1;
            if request.generated_tokens == request.max_new_tokens {
                request.live = false;
            }
        }
    }
}

fn generated_max_new_tokens(random: &mut Lcg) -> u8 {
    1 + u8::try_from(random.bounded_usize(2)).expect("bounded max_new token count")
}

fn append_generated_submit(
    state: &mut GenerationState,
    operations: &mut Vec<BoundedMixedProgramOperation>,
    next_label: &mut u8,
    max_new_tokens: u8,
) {
    if usize::from(*next_label) > MAX_LOGICAL_REQUESTS || state.live_count() >= MAX_LIVE_REQUESTS {
        return;
    }
    let label = *next_label;
    state.submit(label, max_new_tokens);
    operations.push(BoundedMixedProgramOperation::Submit {
        label,
        max_new_tokens,
    });
    *next_label = next_label
        .checked_add(1)
        .expect("bounded program next logical label");
}

fn append_generated_plan_commit(
    state: &mut GenerationState,
    operations: &mut Vec<BoundedMixedProgramOperation>,
    random: &mut Lcg,
) {
    let mut feedback_slot_order = (0..state.live_count())
        .map(|slot| u8::try_from(slot).expect("bounded program feedback slot"))
        .collect::<Vec<_>>();
    for index in (1..feedback_slot_order.len()).rev() {
        let other = random.bounded_usize(index + 1);
        feedback_slot_order.swap(index, other);
    }
    operations.push(BoundedMixedProgramOperation::PlanCommit {
        feedback_slot_order,
    });
    state.commit();
}
