//! Strict descriptors and a pure reference model for variable-length mixed
//! scheduler programs.
//!
//! This test support deliberately does not import `Scheduler` or
//! `IterationPlan`. The public-API adapter validates those plan projections
//! independently and gives the model only opaque request IDs, iteration IDs,
//! cancellation outcomes, and public updates.

use std::collections::{BTreeMap, BTreeSet};

use riley_scheduler::{
    CancellationOutcome, IterationId, IterationOutput, IterationResult, IterationUpdates,
    OutputSlot, RequestCompletion, RequestFinishReason, RequestId, SchedulerCloseOutput,
    TokenEvent,
};
use serde::{Deserialize, Serialize};

const DESCRIPTOR_FORMAT: &str = "riley.scheduler.general-mixed-program";
const DESCRIPTOR_FORMAT_VERSION: u8 = 1;
const DESCRIPTOR_TRACE_KIND: &str = "general-mixed-program-v1";
const MAX_LOGICAL_REQUESTS: usize = 6;
const MAX_OPERATIONS: usize = 24;
const MAX_PLAN_OPERATIONS: usize = 8;
const MAX_CANCELLATIONS: usize = 6;

/// Maximum simultaneously live requests in this strict V1 grammar.
pub const GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS: usize = 4;
/// Maximum symbolic prompt width accepted by one V1 submission.
pub const GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS: usize = 3;
/// Maximum generated-token capacity accepted by one V1 submission.
pub const GENERAL_MIXED_PROGRAM_MAX_NEW_TOKENS: usize = 3;
/// Token budget that selects every live request without partial prefill.
pub const GENERAL_MIXED_PROGRAM_ITERATION_TOKEN_BUDGET: usize =
    GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS * GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS;
/// Per-request chunk bound that completes every V1 prompt in one plan.
pub const GENERAL_MIXED_PROGRAM_MAX_PREFILL_CHUNK_TOKENS: usize =
    GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS;
/// Scheduler sequence bound needed by the largest symbolic V1 request.
pub const GENERAL_MIXED_PROGRAM_MAX_SEQUENCE_TOKENS: usize =
    GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS + GENERAL_MIXED_PROGRAM_MAX_NEW_TOKENS - 1;

/// One explicit operation in a bounded, variable-length mixed program.
///
/// The vector is arbitrary only within this validator's finite V1 transition
/// table. It is not an unbounded scheduler grammar or a model of queueing,
/// aging, partial prefill, invalid feedback, or device execution.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum GeneralMixedProgramOperation {
    /// Submits one unique logical request while no plan is outstanding.
    Submit {
        /// Stable logical request label in the V1 domain.
        label: u8,
        /// Exact symbolic prompt width in 1..=3.
        prompt_len: u8,
        /// Exact sampled-token capacity in 1..=3.
        max_new_tokens: u8,
    },
    /// Opens an immutable plan for every currently live request.
    Plan,
    /// Cancels a live request immediately when idle or defers it when planned.
    Cancel {
        /// Logical label to cancel.
        label: u8,
    },
    /// Commits all outstanding output slots in this exact permutation.
    Complete {
        /// Exact permutation of the pending dense output-slot domain.
        feedback_slot_order: Vec<u8>,
    },
    /// Rolls an undispatched plan back and permits a fresh plan.
    AbortNotDispatched,
    /// Consumes the scheduler after all plans have settled.
    Close,
}

impl GeneralMixedProgramOperation {
    /// Returns a stable, human-readable spelling for diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Submit {
                label,
                prompt_len,
                max_new_tokens,
            } => {
                format!("submit(label={label}, prompt_len={prompt_len}, max_new={max_new_tokens})")
            }
            Self::Plan => "plan".to_owned(),
            Self::Cancel { label } => format!("cancel(label={label})"),
            Self::Complete {
                feedback_slot_order,
            } => format!("complete(order={feedback_slot_order:?})"),
            Self::AbortNotDispatched => "abort(not-dispatched)".to_owned(),
            Self::Close => "close".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralMixedProgramDescriptorV1 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    source_seed: String,
    operations: Vec<GeneralMixedProgramOperation>,
}

/// Fully specified variable-length program within the strict V1 grammar.
///
/// The source seed records provenance only. Its operation vector, not a seed
/// re-generation recipe, is the complete replay input after parsing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMixedProgramTrace {
    /// Source provenance retained in reports and committed corpus fixtures.
    pub seed: u64,
    /// Complete validator-accepted V1 operation vector.
    pub operations: Vec<GeneralMixedProgramOperation>,
}

impl GeneralMixedProgramTrace {
    /// Produces one statefully generated valid V1 program from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut random = Lcg(seed ^ 0x4c7d_8b21_f093_a65e);
        let mut state = GenerationState::default();
        let mut operations = Vec::new();
        let mut next_label = 1_u8;

        // Force a small semantic prefix that leaves one decoder live, then
        // opens one genuine decode-plus-prefill mixed plan. The variable tail
        // below is statefully selected rather than topology-enumerated.
        append_generated_submit(
            &mut state,
            &mut operations,
            &mut next_label,
            generated_prompt_len(&mut random),
            2 + u8::try_from(random.bounded_usize(2)).expect("generated max_new fits u8"),
        );
        append_generated_plan(&mut state, &mut operations);
        append_generated_complete(&mut state, &mut operations, &mut random);
        append_generated_submit(
            &mut state,
            &mut operations,
            &mut next_label,
            generated_prompt_len(&mut random),
            generated_max_new_tokens(&mut random),
        );
        append_generated_plan(&mut state, &mut operations);
        append_generated_complete(&mut state, &mut operations, &mut random);

        // At most four extra complete-or-abort cycles fit here, retaining the
        // validator's eight-plan cap even if every idle choice opens a plan.
        let tail_steps = 1 + random.bounded_usize(8);
        for _ in 0..tail_steps {
            if operations.len() + 1 >= MAX_OPERATIONS {
                break;
            }
            if state.pending.is_some() {
                if state.can_defer_cancel() && random.next() % 5 == 0 {
                    let label = state.random_pending_label(&mut random);
                    operations.push(GeneralMixedProgramOperation::Cancel { label });
                    state.defer_cancel(label);
                } else if random.next() & 1 == 0 {
                    append_generated_complete(&mut state, &mut operations, &mut random);
                } else {
                    operations.push(GeneralMixedProgramOperation::AbortNotDispatched);
                    state.abort_not_dispatched();
                }
                continue;
            }

            let can_submit = state.can_submit() && operations.len() + 2 < MAX_OPERATIONS;
            let live = state.live_count();
            match random.bounded_usize(4) {
                0 if can_submit => append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    generated_prompt_len(&mut random),
                    generated_max_new_tokens(&mut random),
                ),
                1 if live != 0 && random.next() % 3 == 0 => {
                    let label = state.random_live_label(&mut random);
                    operations.push(GeneralMixedProgramOperation::Cancel { label });
                    state.cancel_idle(label);
                }
                _ if live != 0 => append_generated_plan(&mut state, &mut operations),
                _ if can_submit => append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    generated_prompt_len(&mut random),
                    generated_max_new_tokens(&mut random),
                ),
                _ => break,
            }
        }
        if state.pending.is_some() {
            append_generated_complete(&mut state, &mut operations, &mut random);
        }
        operations.push(GeneralMixedProgramOperation::Close);
        let trace = Self { seed, operations };
        trace
            .validate()
            .expect("seeded general mixed program stays inside V1 grammar");
        trace
    }

    /// Returns this explicit raw program's stable operation spelling.
    #[must_use]
    pub fn describe_operations(&self) -> String {
        self.operations
            .iter()
            .map(GeneralMixedProgramOperation::describe)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Returns deterministic semantic-local simplifications.
    ///
    /// Candidate order is cancellation removal, submission removal with its
    /// dependent cancellations, prompt/output-capacity contraction when it
    /// remains valid, complete-order identity, then adjacent inversions. Every
    /// edit replays semantic labels to rebase later feedback permutations; it
    /// intentionally does not delete or reorder Plan/Complete/Abort blocks.
    #[must_use]
    pub fn shrink_candidates(&self) -> Vec<Self> {
        self.validate()
            .expect("general mixed shrinker requires a valid V1 descriptor");
        let source_rank = self.shrink_rank();
        let mut candidates = Vec::new();
        for (index, operation) in self.operations.iter().enumerate() {
            if matches!(operation, GeneralMixedProgramOperation::Cancel { .. }) {
                if let Some(candidate) =
                    rebase_after_edit(self, ReductionEdit::RemoveCancel { index })
                {
                    push_reduction_candidate(&mut candidates, source_rank, candidate);
                }
            }
        }
        for label in self.submission_labels() {
            if let Some(candidate) = rebase_after_edit(self, ReductionEdit::RemoveSubmit { label })
            {
                push_reduction_candidate(&mut candidates, source_rank, candidate);
            }
        }
        for (index, operation) in self.operations.iter().enumerate() {
            let GeneralMixedProgramOperation::Submit { prompt_len, .. } = operation else {
                continue;
            };
            if *prompt_len > 1 {
                if let Some(candidate) =
                    rebase_after_edit(self, ReductionEdit::ReducePrompt { index })
                {
                    push_reduction_candidate(&mut candidates, source_rank, candidate);
                }
            }
        }
        for (index, operation) in self.operations.iter().enumerate() {
            let GeneralMixedProgramOperation::Submit { max_new_tokens, .. } = operation else {
                continue;
            };
            if *max_new_tokens > 1 {
                if let Some(candidate) =
                    rebase_after_edit(self, ReductionEdit::ReduceMaxNew { index })
                {
                    push_reduction_candidate(&mut candidates, source_rank, candidate);
                }
            }
        }
        for (index, operation) in self.operations.iter().enumerate() {
            let GeneralMixedProgramOperation::Complete {
                feedback_slot_order,
            } = operation
            else {
                continue;
            };
            let identity = canonical_slot_order(feedback_slot_order.len());
            if *feedback_slot_order != identity {
                let mut candidate = self.clone();
                let GeneralMixedProgramOperation::Complete {
                    feedback_slot_order,
                } = &mut candidate.operations[index]
                else {
                    unreachable!("complete operation stays a complete");
                };
                *feedback_slot_order = identity;
                push_reduction_candidate(&mut candidates, source_rank, candidate);
            }
        }
        for (index, operation) in self.operations.iter().enumerate() {
            let GeneralMixedProgramOperation::Complete {
                feedback_slot_order,
            } = operation
            else {
                continue;
            };
            for slot_index in 0..feedback_slot_order.len().saturating_sub(1) {
                if feedback_slot_order[slot_index] > feedback_slot_order[slot_index + 1] {
                    let mut candidate = self.clone();
                    let GeneralMixedProgramOperation::Complete {
                        feedback_slot_order,
                    } = &mut candidate.operations[index]
                    else {
                        unreachable!("complete operation stays a complete");
                    };
                    feedback_slot_order.swap(slot_index, slot_index + 1);
                    push_reduction_candidate(&mut candidates, source_rank, candidate);
                }
            }
        }
        candidates
    }

    /// Returns the V1 semantic-local reducer rank.
    #[must_use]
    pub fn shrink_rank(&self) -> (usize, usize, usize, usize, usize, usize) {
        self.validate()
            .expect("general mixed shrink rank requires a valid V1 descriptor");
        let mut submissions = 0_usize;
        let mut prompt_tokens = 0_usize;
        let mut max_new_tokens = 0_usize;
        let mut cancellations = 0_usize;
        let mut inversions = 0_usize;
        for operation in &self.operations {
            match operation {
                GeneralMixedProgramOperation::Submit {
                    prompt_len,
                    max_new_tokens: capacity,
                    ..
                } => {
                    submissions += 1;
                    prompt_tokens += usize::from(*prompt_len);
                    max_new_tokens += usize::from(*capacity);
                }
                GeneralMixedProgramOperation::Cancel { .. } => cancellations += 1,
                GeneralMixedProgramOperation::Complete {
                    feedback_slot_order,
                } => inversions += inversion_count(feedback_slot_order),
                GeneralMixedProgramOperation::Plan
                | GeneralMixedProgramOperation::AbortNotDispatched
                | GeneralMixedProgramOperation::Close => {}
            }
        }
        (
            self.operations.len(),
            submissions,
            prompt_tokens,
            max_new_tokens,
            cancellations,
            inversions,
        )
    }

    /// Validates all V1 syntax-independent state-machine transitions and bounds.
    pub fn validate(&self) -> Result<(), String> {
        validate_operations(&self.operations)
    }

    fn descriptor(&self, case_id: &str) -> Result<GeneralMixedProgramDescriptorV1, String> {
        self.validate()?;
        validate_case_id(case_id)?;
        Ok(GeneralMixedProgramDescriptorV1 {
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
                GeneralMixedProgramOperation::Submit { label, .. } => Some(*label),
                GeneralMixedProgramOperation::Plan
                | GeneralMixedProgramOperation::Cancel { .. }
                | GeneralMixedProgramOperation::Complete { .. }
                | GeneralMixedProgramOperation::AbortNotDispatched
                | GeneralMixedProgramOperation::Close => None,
            })
            .collect()
    }
}

/// A parsed strict V1 descriptor with its stable case identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedGeneralMixedProgramTrace {
    /// Durable corpus or diagnostic case identifier.
    pub case_id: String,
    /// Fully validated variable-length V1 program.
    pub trace: GeneralMixedProgramTrace,
}

impl GeneralMixedProgramDescriptorV1 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("general mixed program descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err(
                "general mixed program descriptor format_version is unsupported".to_owned(),
            );
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND {
            return Err("general mixed program descriptor trace_kind is unsupported".to_owned());
        }
        validate_case_id(&self.case_id)?;
        parse_source_seed(&self.source_seed)?;
        validate_operations(&self.operations)
    }

    fn into_named_trace(self) -> Result<NamedGeneralMixedProgramTrace, String> {
        self.validate()?;
        Ok(NamedGeneralMixedProgramTrace {
            case_id: self.case_id,
            trace: GeneralMixedProgramTrace {
                seed: parse_source_seed(&self.source_seed)?,
                operations: self.operations,
            },
        })
    }
}

/// Serializes one valid V1 program as exact canonical JSON with a newline.
#[must_use]
pub fn serialize_general_mixed_program_descriptor(
    case_id: &str,
    trace: &GeneralMixedProgramTrace,
) -> String {
    let descriptor = trace
        .descriptor(case_id)
        .expect("general mixed program trace is valid before serialization");
    descriptor_document(&descriptor)
}

/// Parses only exact canonical general-mixed-program V1 JSON documents.
pub fn parse_general_mixed_program_descriptor(
    document: &str,
) -> Result<NamedGeneralMixedProgramTrace, String> {
    let descriptor = serde_json::from_str::<GeneralMixedProgramDescriptorV1>(document)
        .map_err(|error| format!("general mixed program descriptor JSON is invalid: {error}"))?;
    if document != descriptor_document(&descriptor) {
        return Err("general mixed program descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_named_trace()
}

/// Greedily minimizes a reproducing trace over the fixed semantic-local order.
///
/// Source and candidates are strict-codec round-tripped before the predicate.
/// The result is only a local minimum for this grammar and panic-only predicate.
pub fn minimize_general_mixed_program_trace<F>(
    trace: &GeneralMixedProgramTrace,
    mut reproduces: F,
) -> GeneralMixedProgramTrace
where
    F: FnMut(&GeneralMixedProgramTrace) -> bool,
{
    let mut minimized = strict_round_trip_trace(trace, "shrink-source");
    assert!(
        reproduces(&minimized),
        "general mixed program minimization requires a reproducing source trace"
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
    trace: &GeneralMixedProgramTrace,
    case_id: &str,
) -> GeneralMixedProgramTrace {
    let document = serialize_general_mixed_program_descriptor(case_id, trace);
    let parsed = parse_general_mixed_program_descriptor(&document)
        .expect("general mixed program shrink descriptor remains strict-canonical");
    assert_eq!(parsed.case_id, case_id);
    assert_eq!(parsed.trace, *trace);
    parsed.trace
}

const CORPUS_DOCUMENTS_V1: [(&str, &str); 4] = [
    (
        "general-mixed-program-v1/variable-prompt-reverse-mixed.json",
        include_str!(
            "../corpus/output-routing/general-mixed-program-v1/variable-prompt-reverse-mixed.json"
        ),
    ),
    (
        "general-mixed-program-v1/deferred-cancel-abort-retry.json",
        include_str!(
            "../corpus/output-routing/general-mixed-program-v1/deferred-cancel-abort-retry.json"
        ),
    ),
    (
        "general-mixed-program-v1/settled-cancel-replacement.json",
        include_str!(
            "../corpus/output-routing/general-mixed-program-v1/settled-cancel-replacement.json"
        ),
    ),
    (
        "general-mixed-program-v1/variable-length-multi-plan-close.json",
        include_str!(
            "../corpus/output-routing/general-mixed-program-v1/variable-length-multi-plan-close.json"
        ),
    ),
];

/// Loads the committed canonical V1 corpus and rejects duplicate case IDs.
#[must_use]
pub fn general_mixed_program_corpus() -> Vec<NamedGeneralMixedProgramTrace> {
    let mut case_ids = BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V1.len());
    for (path, document) in CORPUS_DOCUMENTS_V1 {
        let named = parse_general_mixed_program_descriptor(document).unwrap_or_else(|error| {
            panic!("{path}: general mixed program corpus is invalid: {error}")
        });
        assert!(
            case_ids.insert(named.case_id.clone()),
            "{path}: general mixed program corpus repeats case_id {:?}",
            named.case_id
        );
        corpus.push(named);
    }
    corpus
}

/// One grammar-derived expected public work item.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMixedProgramPlanItem {
    /// Logical label that owns this work item and output slot.
    pub label: u8,
    /// Opaque public request binding supplied by the adapter.
    pub request_id: RequestId,
    /// Exact symbolic prefill or decode input tokens.
    pub input_tokens: Vec<u32>,
    /// Expected dense output slot.
    pub output_slot: OutputSlot,
    /// Generated-token index produced by this work item.
    pub generated_index: usize,
}

/// Public-plan projection derived from only reference-model state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMixedProgramExpectedPlan {
    /// Decode items in canonical ready/request order.
    pub decode_items: Vec<GeneralMixedProgramPlanItem>,
    /// Fully completing prefill items in canonical ready/request order.
    pub prefill_items: Vec<GeneralMixedProgramPlanItem>,
}

impl GeneralMixedProgramExpectedPlan {
    /// Returns the exact dense output-slot domain in canonical plan order.
    #[must_use]
    pub fn output_slots(&self) -> Vec<OutputSlot> {
        self.decode_items
            .iter()
            .chain(&self.prefill_items)
            .map(|item| item.output_slot)
            .collect()
    }

    /// Returns total input-token count represented by this plan.
    #[must_use]
    pub fn total_tokens(&self) -> usize {
        self.decode_items
            .iter()
            .chain(&self.prefill_items)
            .map(|item| item.input_tokens.len())
            .sum()
    }

    /// Returns the count of selected request work items.
    #[must_use]
    pub fn batch_size(&self) -> usize {
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
enum ModelPhase {
    Idle,
    InFlight,
    Closed,
}

struct ModelRequest {
    request_id: RequestId,
    prompt_len: usize,
    max_new_tokens: usize,
    submission_order: usize,
    generated_token_ids: Vec<u32>,
    live: bool,
    cancellation_deferred: bool,
}

#[derive(Clone, Copy)]
struct PendingOutput {
    label: u8,
    request_id: RequestId,
    generated_index: usize,
    output_slot: OutputSlot,
}

/// Pure request/token/terminal reference model for general-mixed-program V1.
///
/// This model is intentionally narrower than a scheduler: the V1 adapter uses
/// a fixed all-live configuration, so admission queues, aging, partial prefill,
/// and physical KV policy are not modeled here.
pub struct GeneralMixedProgramReferenceModel {
    seed: u64,
    requests: BTreeMap<u8, ModelRequest>,
    pending: Option<Vec<PendingOutput>>,
    terminal: BTreeMap<RequestId, CompletionProjection>,
    next_submission_order: usize,
    phase: ModelPhase,
}

impl GeneralMixedProgramReferenceModel {
    /// Creates an unbound reference model before public request IDs are known.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            seed,
            requests: BTreeMap::new(),
            pending: None,
            terminal: BTreeMap::new(),
            next_submission_order: 0,
            phase: ModelPhase::Idle,
        }
    }

    /// Binds one V1 submission to an opaque public request identity.
    pub fn bind_submit(
        &mut self,
        label: u8,
        prompt_len: u8,
        max_new_tokens: u8,
        request_id: RequestId,
    ) {
        self.assert_phase(ModelPhase::Idle);
        validate_label(label).expect("reference model label remains in V1 domain");
        assert!(
            (1..=u8::try_from(GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS)
                .expect("prompt cap fits u8"))
                .contains(&prompt_len)
        );
        assert!(
            (1..=u8::try_from(GENERAL_MIXED_PROGRAM_MAX_NEW_TOKENS).expect("capacity cap fits u8"))
                .contains(&max_new_tokens)
        );
        assert!(self.requests.len() < MAX_LOGICAL_REQUESTS);
        assert!(self.live_count() < GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS);
        assert!(
            self.requests
                .insert(
                    label,
                    ModelRequest {
                        request_id,
                        prompt_len: usize::from(prompt_len),
                        max_new_tokens: usize::from(max_new_tokens),
                        submission_order: self.next_submission_order,
                        generated_token_ids: Vec::with_capacity(usize::from(max_new_tokens)),
                        live: true,
                        cancellation_deferred: false,
                    },
                )
                .is_none(),
            "seed {:#018x}: general mixed model bound one label twice",
            self.seed
        );
        self.next_submission_order = self
            .next_submission_order
            .checked_add(1)
            .expect("general mixed model submission ordinal");
    }

    /// Derives and opens the full expected public plan before feedback exists.
    #[must_use]
    pub fn begin_plan(&mut self) -> GeneralMixedProgramExpectedPlan {
        self.assert_phase(ModelPhase::Idle);
        let mut live = self
            .requests
            .iter()
            .filter(|(_, request)| request.live)
            .map(|(label, request)| (*label, request))
            .collect::<Vec<_>>();
        assert!(
            !live.is_empty(),
            "seed {:#018x}: general mixed model planned without live work",
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
            let output_slot = OutputSlot::new(
                u32::try_from(pending.len()).expect("general mixed output slot fits u32"),
            );
            let generated_index = request.generated_token_ids.len();
            let input_tokens = if generated_index == 0 {
                general_mixed_program_prompt_tokens(
                    label,
                    u8::try_from(request.prompt_len).expect("V1 prompt width fits u8"),
                )
            } else {
                vec![
                    *request
                        .generated_token_ids
                        .last()
                        .expect("general mixed decoder retains prior generated token"),
                ]
            };
            let item = GeneralMixedProgramPlanItem {
                label,
                request_id: request.request_id,
                input_tokens,
                output_slot,
                generated_index,
            };
            pending.push(PendingOutput {
                label,
                request_id: request.request_id,
                generated_index,
                output_slot,
            });
            if generated_index == 0 {
                prefill_items.push(item);
            } else {
                decode_items.push(item);
            }
        }
        self.pending = Some(pending);
        self.phase = ModelPhase::InFlight;
        GeneralMixedProgramExpectedPlan {
            decode_items,
            prefill_items,
        }
    }

    /// Compares one immediate idle cancellation with the reference ledger.
    pub fn record_settled_cancel(&mut self, label: u8, outcome: &CancellationOutcome, now_ns: u64) {
        self.assert_phase(ModelPhase::Idle);
        let request = self.request(label);
        assert!(request.live);
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
            .expect("settled general mixed cancellation publishes one completion");
        self.assert_completions(std::slice::from_ref(completion), expected);
        self.request_mut(label).live = false;
    }

    /// Records one deferred cancellation for an outstanding-plan label.
    pub fn defer_cancel(&mut self, label: u8, outcome: &CancellationOutcome) {
        self.assert_phase(ModelPhase::InFlight);
        let pending = self
            .pending
            .as_ref()
            .expect("general mixed model retained pending plan");
        assert!(pending.iter().any(|item| item.label == label));
        let request = self.request(label);
        assert!(request.live);
        assert!(!request.cancellation_deferred);
        assert_eq!(outcome.request_id(), request.request_id);
        assert!(outcome.deferred_until_iteration_settles());
        assert!(!outcome.already_terminal());
        assert!(outcome.completion().is_none());
        self.request_mut(label).cancellation_deferred = true;
    }

    /// Builds exact descriptor-selected feedback without reading a public plan.
    #[must_use]
    pub fn feedback(
        &self,
        iteration_id: IterationId,
        feedback_slot_order: &[u8],
    ) -> IterationResult {
        self.assert_phase(ModelPhase::InFlight);
        let pending = self
            .pending
            .as_ref()
            .expect("general mixed model retained pending plan");
        validate_slot_order(feedback_slot_order, pending.len(), "feedback_slot_order")
            .unwrap_or_else(|error| {
                panic!(
                    "seed {:#018x}: general mixed feedback permutation is invalid: {error}",
                    self.seed
                )
            });
        let outputs = feedback_slot_order
            .iter()
            .map(|slot| {
                let item = pending[usize::from(*slot)];
                IterationOutput::new(
                    item.output_slot,
                    general_mixed_program_token(item.label, item.generated_index),
                    false,
                )
            })
            .collect();
        IterationResult::new(iteration_id, outputs, 0, 0)
            .expect("general mixed reference feedback has unique slots")
    }

    /// Compares one valid complete with token, terminal, and deferred-cancel state.
    pub fn record_complete(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(ModelPhase::InFlight);
        let pending = self
            .pending
            .take()
            .expect("general mixed model retained pending plan");
        let mut expected_tokens = BTreeMap::new();
        let mut expected_completions = BTreeMap::new();
        for item in &pending {
            let request = self.request(item.label);
            assert!(request.live);
            assert_eq!(request.request_id, item.request_id);
            assert_eq!(request.generated_token_ids.len(), item.generated_index);
            if request.cancellation_deferred {
                insert_expected_completion(
                    &mut expected_completions,
                    item.request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: request.generated_token_ids.clone(),
                        completed_at_ns: now_ns,
                    },
                    self.seed,
                );
                continue;
            }
            let token_id = general_mixed_program_token(item.label, item.generated_index);
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
                "seed {:#018x}: general mixed model repeated a pending request",
                self.seed
            );
            let mut history = request.generated_token_ids.clone();
            history.push(token_id);
            if history.len() == request.max_new_tokens {
                insert_expected_completion(
                    &mut expected_completions,
                    item.request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Length,
                        generated_token_ids: history,
                        completed_at_ns: now_ns,
                    },
                    self.seed,
                );
            }
        }
        self.assert_updates(updates, &expected_tokens, expected_completions);
        for item in pending {
            let request = self.request_mut(item.label);
            if request.cancellation_deferred {
                request.cancellation_deferred = false;
                request.live = false;
                continue;
            }
            request
                .generated_token_ids
                .push(general_mixed_program_token(
                    item.label,
                    item.generated_index,
                ));
            if request.generated_token_ids.len() == request.max_new_tokens {
                request.live = false;
            }
        }
        self.phase = ModelPhase::Idle;
    }

    /// Compares a not-dispatched rollback with the reference terminal ledger.
    pub fn record_not_dispatched_abort(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(ModelPhase::InFlight);
        let pending = self
            .pending
            .take()
            .expect("general mixed model retained pending plan");
        let mut expected_completions = BTreeMap::new();
        for item in &pending {
            let request = self.request(item.label);
            assert!(request.live);
            assert_eq!(request.request_id, item.request_id);
            if request.cancellation_deferred {
                insert_expected_completion(
                    &mut expected_completions,
                    item.request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: request.generated_token_ids.clone(),
                        completed_at_ns: now_ns,
                    },
                    self.seed,
                );
            }
        }
        self.assert_updates(updates, &BTreeMap::new(), expected_completions);
        for item in pending {
            let request = self.request_mut(item.label);
            if request.cancellation_deferred {
                request.cancellation_deferred = false;
                request.live = false;
            }
        }
        self.phase = ModelPhase::Idle;
    }

    /// Compares consuming close with every still-live request's terminal event.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(ModelPhase::Idle);
        assert!(
            closed.settlement_failures().is_empty(),
            "seed {:#018x}: general mixed close contained settlement failures",
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
            request.cancellation_deferred = false;
        }
        assert_eq!(
            self.terminal.len(),
            self.requests.len(),
            "seed {:#018x}: general mixed close did not terminally settle every request",
            self.seed
        );
        self.phase = ModelPhase::Closed;
    }

    /// Asserts that the adapter reached exactly one terminal close state.
    pub fn assert_closed(&self) {
        self.assert_phase(ModelPhase::Closed);
        assert_eq!(
            self.terminal.len(),
            self.requests.len(),
            "seed {:#018x}: general mixed terminal ledger is incomplete",
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
            "seed {:#018x}: general mixed update contained settlement failures",
            self.seed
        );
        assert_eq!(
            &token_projections(updates.token_events()),
            expected_tokens,
            "seed {:#018x}: general mixed token routing drifted",
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
            "seed {:#018x}: general mixed terminal ledger drifted",
            self.seed
        );
        for (request_id, completion) in expected_completions {
            assert!(
                self.terminal.insert(request_id, completion).is_none(),
                "seed {:#018x}: general mixed emitted a duplicate terminal request",
                self.seed
            );
        }
    }

    fn assert_phase(&self, expected: ModelPhase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: general mixed model phase drifted",
            self.seed
        );
    }

    fn live_count(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.live)
            .count()
    }

    fn request(&self, label: u8) -> &ModelRequest {
        self.requests.get(&label).unwrap_or_else(|| {
            panic!(
                "seed {:#018x}: general mixed model label {label} is unbound",
                self.seed
            )
        })
    }

    fn request_mut(&mut self, label: u8) -> &mut ModelRequest {
        self.requests.get_mut(&label).unwrap_or_else(|| {
            panic!(
                "seed {:#018x}: general mixed model label {label} is unbound",
                self.seed
            )
        })
    }
}

/// Returns a deterministic symbolic prompt for one logical label and width.
#[must_use]
pub fn general_mixed_program_prompt_tokens(label: u8, prompt_len: u8) -> Vec<u32> {
    validate_label(label).expect("general mixed prompt label remains in V1 domain");
    assert!(
        (1..=u8::try_from(GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS).expect("prompt cap fits u8"))
            .contains(&prompt_len)
    );
    (0..prompt_len)
        .map(|offset| {
            3_000_u32
                .checked_add(u32::from(label) * 16)
                .and_then(|base| base.checked_add(u32::from(offset)))
                .expect("general mixed symbolic prompt token")
        })
        .collect()
}

/// Returns the deterministic sampled token for one label and generation index.
#[must_use]
pub fn general_mixed_program_token(label: u8, generated_index: usize) -> u32 {
    validate_label(label).expect("general mixed sampled-token label remains in V1 domain");
    u32::from(label)
        .checked_mul(64)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("general mixed symbolic output token")
}

#[derive(Clone)]
struct GrammarRequest {
    max_new_tokens: usize,
    generated_tokens: usize,
    submission_order: usize,
    live: bool,
    cancellation_deferred: bool,
}

#[derive(Clone)]
struct GrammarPending {
    labels: Vec<u8>,
}

#[derive(Default)]
struct GrammarState {
    requests: BTreeMap<u8, GrammarRequest>,
    pending: Option<GrammarPending>,
    close_seen: bool,
    plan_count: usize,
    cancellation_count: usize,
    next_submission_order: usize,
    mixed_plan_seen: bool,
}

impl GrammarState {
    fn apply(
        &mut self,
        operation: &GeneralMixedProgramOperation,
        operation_index: usize,
        operation_count: usize,
        validate_feedback: bool,
    ) -> Result<Option<Vec<u8>>, String> {
        if self.close_seen {
            return Err(
                "general mixed program descriptor contains an operation after close".to_owned(),
            );
        }
        match operation {
            GeneralMixedProgramOperation::Submit {
                label,
                prompt_len,
                max_new_tokens,
            } => {
                if self.pending.is_some() {
                    return Err(
                        "general mixed program descriptor submits while a plan is outstanding"
                            .to_owned(),
                    );
                }
                validate_label(*label)?;
                validate_prompt_len(*prompt_len)?;
                validate_max_new_tokens(*max_new_tokens)?;
                if self.requests.len() >= MAX_LOGICAL_REQUESTS {
                    return Err(
                        "general mixed program descriptor exceeds its unique request cap"
                            .to_owned(),
                    );
                }
                if self.live_count() >= GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS {
                    return Err(
                        "general mixed program descriptor exceeds its live request cap".to_owned(),
                    );
                }
                if self
                    .requests
                    .insert(
                        *label,
                        GrammarRequest {
                            max_new_tokens: usize::from(*max_new_tokens),
                            generated_tokens: 0,
                            submission_order: self.next_submission_order,
                            live: true,
                            cancellation_deferred: false,
                        },
                    )
                    .is_some()
                {
                    return Err(
                        "general mixed program descriptor submits one label more than once"
                            .to_owned(),
                    );
                }
                self.next_submission_order =
                    self.next_submission_order.checked_add(1).ok_or_else(|| {
                        "general mixed program submission ordinal overflowed".to_owned()
                    })?;
                Ok(None)
            }
            GeneralMixedProgramOperation::Plan => {
                if self.pending.is_some() {
                    return Err(
                        "general mixed program descriptor opens a second pending plan".to_owned(),
                    );
                }
                if self.plan_count >= MAX_PLAN_OPERATIONS {
                    return Err("general mixed program descriptor exceeds its plan cap".to_owned());
                }
                let labels = self.semantic_plan_labels();
                if labels.is_empty() {
                    return Err(
                        "general mixed program descriptor plans without live requests".to_owned(),
                    );
                }
                let has_decode = labels.iter().any(|label| {
                    self.requests
                        .get(label)
                        .is_some_and(|request| request.generated_tokens != 0)
                });
                let has_prefill = labels.iter().any(|label| {
                    self.requests
                        .get(label)
                        .is_some_and(|request| request.generated_tokens == 0)
                });
                self.mixed_plan_seen |= has_decode && has_prefill;
                self.pending = Some(GrammarPending { labels });
                self.plan_count += 1;
                Ok(None)
            }
            GeneralMixedProgramOperation::Cancel { label } => {
                validate_label(*label)?;
                if self.cancellation_count >= MAX_CANCELLATIONS {
                    return Err(
                        "general mixed program descriptor exceeds its cancellation cap".to_owned(),
                    );
                }
                let Some(request) = self.requests.get(label) else {
                    return Err(
                        "general mixed program descriptor cancels an unsubmitted label".to_owned(),
                    );
                };
                if !request.live {
                    return Err(
                        "general mixed program descriptor cancels a non-live label".to_owned()
                    );
                }
                if self.pending.is_some() {
                    let pending = self
                        .pending
                        .as_ref()
                        .expect("checked pending state remains present");
                    if !pending.labels.contains(label) {
                        return Err(
                            "general mixed program descriptor cancels an unplanned label"
                                .to_owned(),
                        );
                    }
                    if request.cancellation_deferred {
                        return Err(
                            "general mixed program descriptor defers one label twice".to_owned()
                        );
                    }
                    self.requests
                        .get_mut(label)
                        .expect("checked live request remains present")
                        .cancellation_deferred = true;
                } else {
                    self.requests
                        .get_mut(label)
                        .expect("checked live request remains present")
                        .live = false;
                }
                self.cancellation_count += 1;
                Ok(None)
            }
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order,
            } => {
                let pending = self.pending.take().ok_or_else(|| {
                    "general mixed program descriptor completes without a pending plan".to_owned()
                })?;
                if validate_feedback {
                    validate_slot_order(
                        feedback_slot_order,
                        pending.labels.len(),
                        "feedback_slot_order",
                    )?;
                }
                for label in &pending.labels {
                    let request = self
                        .requests
                        .get_mut(label)
                        .expect("pending general mixed label remains submitted");
                    if request.cancellation_deferred {
                        request.cancellation_deferred = false;
                        request.live = false;
                        continue;
                    }
                    request.generated_tokens =
                        request.generated_tokens.checked_add(1).ok_or_else(|| {
                            "general mixed program descriptor generation count overflowed"
                                .to_owned()
                        })?;
                    if request.generated_tokens == request.max_new_tokens {
                        request.live = false;
                    }
                }
                Ok(Some(pending.labels))
            }
            GeneralMixedProgramOperation::AbortNotDispatched => {
                let pending = self.pending.take().ok_or_else(|| {
                    "general mixed program descriptor aborts without a pending plan".to_owned()
                })?;
                for label in &pending.labels {
                    let request = self
                        .requests
                        .get_mut(label)
                        .expect("pending general mixed label remains submitted");
                    if request.cancellation_deferred {
                        request.cancellation_deferred = false;
                        request.live = false;
                    }
                }
                Ok(None)
            }
            GeneralMixedProgramOperation::Close => {
                if self.pending.is_some() {
                    return Err(
                        "general mixed program descriptor closes with a pending plan".to_owned(),
                    );
                }
                if operation_index + 1 != operation_count {
                    return Err(
                        "general mixed program descriptor close must be the final operation"
                            .to_owned(),
                    );
                }
                for request in self.requests.values_mut() {
                    request.live = false;
                    request.cancellation_deferred = false;
                }
                self.close_seen = true;
                Ok(None)
            }
        }
    }

    fn semantic_plan_labels(&self) -> Vec<u8> {
        let mut live = self
            .requests
            .iter()
            .filter(|(_, request)| request.live)
            .map(|(label, request)| (*label, request))
            .collect::<Vec<_>>();
        live.sort_unstable_by_key(|(_, request)| request.submission_order);
        let (decode, prefill): (Vec<_>, Vec<_>) = live
            .into_iter()
            .partition(|(_, request)| request.generated_tokens != 0);
        decode
            .into_iter()
            .chain(prefill)
            .map(|(label, _)| label)
            .collect()
    }

    fn live_count(&self) -> usize {
        self.requests
            .values()
            .filter(|request| request.live)
            .count()
    }
}

struct TraceSemantics {
    pending_before_complete: BTreeMap<usize, Vec<u8>>,
}

fn validate_operations(operations: &[GeneralMixedProgramOperation]) -> Result<(), String> {
    simulate_operations(operations, true).map(|_| ())
}

fn simulate_operations(
    operations: &[GeneralMixedProgramOperation],
    validate_feedback: bool,
) -> Result<TraceSemantics, String> {
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(format!(
            "general mixed program descriptor operations must contain 1..={MAX_OPERATIONS} entries"
        ));
    }
    let mut state = GrammarState::default();
    let mut pending_before_complete = BTreeMap::new();
    for (index, operation) in operations.iter().enumerate() {
        let complete_labels = state.apply(operation, index, operations.len(), validate_feedback)?;
        if let Some(labels) = complete_labels {
            pending_before_complete.insert(index, labels);
        }
    }
    if !state.close_seen {
        return Err("general mixed program descriptor must end with close".to_owned());
    }
    if state.plan_count < 2 {
        return Err("general mixed program descriptor requires at least two plans".to_owned());
    }
    if !state.mixed_plan_seen {
        return Err(
            "general mixed program descriptor requires one decode plus prefill mixed plan"
                .to_owned(),
        );
    }
    Ok(TraceSemantics {
        pending_before_complete,
    })
}

#[derive(Clone)]
struct IndexedOperation {
    source_index: usize,
    operation: GeneralMixedProgramOperation,
}

#[derive(Clone, Copy)]
enum ReductionEdit {
    RemoveCancel { index: usize },
    RemoveSubmit { label: u8 },
    ReducePrompt { index: usize },
    ReduceMaxNew { index: usize },
}

fn rebase_after_edit(
    source: &GeneralMixedProgramTrace,
    edit: ReductionEdit,
) -> Option<GeneralMixedProgramTrace> {
    let source_semantics = simulate_operations(&source.operations, true).ok()?;
    let mut indexed = Vec::new();
    for (source_index, source_operation) in source.operations.iter().enumerate() {
        if matches!(edit, ReductionEdit::RemoveCancel { index } if index == source_index) {
            continue;
        }
        if let ReductionEdit::RemoveSubmit { label } = edit {
            if matches!(source_operation, GeneralMixedProgramOperation::Submit { label: current, .. } if *current == label)
                || matches!(source_operation, GeneralMixedProgramOperation::Cancel { label: current } if *current == label)
            {
                continue;
            }
        }
        let mut operation = source_operation.clone();
        match edit {
            ReductionEdit::ReducePrompt { index } if index == source_index => {
                let GeneralMixedProgramOperation::Submit { prompt_len, .. } = &mut operation else {
                    return None;
                };
                *prompt_len = 1;
            }
            ReductionEdit::ReduceMaxNew { index } if index == source_index => {
                let GeneralMixedProgramOperation::Submit { max_new_tokens, .. } = &mut operation
                else {
                    return None;
                };
                *max_new_tokens = 1;
            }
            ReductionEdit::RemoveCancel { .. }
            | ReductionEdit::RemoveSubmit { .. }
            | ReductionEdit::ReducePrompt { .. }
            | ReductionEdit::ReduceMaxNew { .. } => {}
        }
        indexed.push(IndexedOperation {
            source_index,
            operation,
        });
    }
    let candidate_operations = indexed
        .iter()
        .map(|indexed_operation| indexed_operation.operation.clone())
        .collect::<Vec<_>>();
    let candidate_semantics = simulate_operations(&candidate_operations, false).ok()?;
    let mut rebased = candidate_operations;
    for (candidate_index, indexed_operation) in indexed.iter().enumerate() {
        let GeneralMixedProgramOperation::Complete {
            feedback_slot_order,
        } = &indexed_operation.operation
        else {
            continue;
        };
        let source_labels = source_semantics
            .pending_before_complete
            .get(&indexed_operation.source_index)?;
        let candidate_labels = candidate_semantics
            .pending_before_complete
            .get(&candidate_index)?;
        let source_order = feedback_slot_order
            .iter()
            .map(|slot| source_labels[usize::from(*slot)])
            .collect::<Vec<_>>();
        let mut projected = source_order
            .into_iter()
            .filter(|label| candidate_labels.contains(label))
            .collect::<Vec<_>>();
        for label in candidate_labels {
            if !projected.contains(label) {
                projected.push(*label);
            }
        }
        let rebased_order = projected
            .iter()
            .map(|label| {
                u8::try_from(
                    candidate_labels
                        .iter()
                        .position(|candidate_label| candidate_label == label)
                        .expect("projected label remains in candidate pending plan"),
                )
                .expect("candidate output slot fits u8")
            })
            .collect::<Vec<_>>();
        let GeneralMixedProgramOperation::Complete {
            feedback_slot_order,
        } = &mut rebased[candidate_index]
        else {
            unreachable!("candidate complete remains a complete");
        };
        *feedback_slot_order = rebased_order;
    }
    let candidate = GeneralMixedProgramTrace {
        seed: source.seed,
        operations: rebased,
    };
    candidate.validate().ok()?;
    Some(candidate)
}

fn push_reduction_candidate(
    candidates: &mut Vec<GeneralMixedProgramTrace>,
    source_rank: (usize, usize, usize, usize, usize, usize),
    candidate: GeneralMixedProgramTrace,
) {
    candidate
        .validate()
        .expect("general mixed reduction candidate remains a valid V1 descriptor");
    assert!(
        candidate.shrink_rank() < source_rank,
        "general mixed reduction candidate must strictly lower rank"
    );
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn validate_slot_order(order: &[u8], slot_count: usize, field: &str) -> Result<(), String> {
    if order.len() != slot_count {
        return Err(format!(
            "general mixed program descriptor {field} must contain exactly {slot_count} slots"
        ));
    }
    let mut seen = BTreeSet::new();
    for slot in order {
        if usize::from(*slot) >= slot_count {
            return Err(format!(
                "general mixed program descriptor {field} contains an out-of-range slot"
            ));
        }
        if !seen.insert(*slot) {
            return Err(format!(
                "general mixed program descriptor {field} contains a duplicate slot"
            ));
        }
    }
    Ok(())
}

fn canonical_slot_order(slot_count: usize) -> Vec<u8> {
    (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("general mixed slot fits u8"))
        .collect()
}

fn inversion_count(order: &[u8]) -> usize {
    order
        .iter()
        .enumerate()
        .map(|(index, slot)| {
            order[index + 1..]
                .iter()
                .filter(|later| slot > *later)
                .count()
        })
        .sum()
}

fn validate_label(label: u8) -> Result<(), String> {
    if !(1..=u8::try_from(MAX_LOGICAL_REQUESTS).expect("label cap fits u8")).contains(&label) {
        return Err("general mixed program descriptor label must be in 1..=6".to_owned());
    }
    Ok(())
}

fn validate_prompt_len(prompt_len: u8) -> Result<(), String> {
    if !(1..=u8::try_from(GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS).expect("prompt cap fits u8"))
        .contains(&prompt_len)
    {
        return Err("general mixed program descriptor prompt_len must be in 1..=3".to_owned());
    }
    Ok(())
}

fn validate_max_new_tokens(max_new_tokens: u8) -> Result<(), String> {
    if !(1..=u8::try_from(GENERAL_MIXED_PROGRAM_MAX_NEW_TOKENS).expect("capacity cap fits u8"))
        .contains(&max_new_tokens)
    {
        return Err("general mixed program descriptor max_new_tokens must be in 1..=3".to_owned());
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
            "general mixed program descriptor case_id must be a bounded lowercase identifier"
                .to_owned(),
        );
    }
    Ok(())
}

fn parse_source_seed(source_seed: &str) -> Result<u64, String> {
    let Some(hex) = source_seed.strip_prefix("0x") else {
        return Err("general mixed program descriptor source_seed must start with 0x".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(
            "general mixed program descriptor source_seed must be 16 lowercase hexadecimal digits"
                .to_owned(),
        );
    }
    u64::from_str_radix(hex, 16)
        .map_err(|_| "general mixed program descriptor source_seed does not fit u64".to_owned())
}

fn descriptor_document(descriptor: &GeneralMixedProgramDescriptorV1) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("general mixed program descriptor serializes");
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
            "general mixed public updates emitted duplicate token events"
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
            "general mixed public updates emitted duplicate terminal completions"
        );
    }
    projections
}

fn insert_expected_completion(
    expected: &mut BTreeMap<RequestId, CompletionProjection>,
    request_id: RequestId,
    completion: CompletionProjection,
    seed: u64,
) {
    assert!(
        expected.insert(request_id, completion).is_none(),
        "seed {seed:#018x}: general mixed model repeated an expected terminal request"
    );
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
            "general mixed RNG bound must be nonzero"
        );
        usize::try_from(self.next() % u64::try_from(upper_exclusive).expect("usize fits u64"))
            .expect("general mixed RNG value fits usize")
    }
}

#[derive(Clone)]
struct GeneratedRequest {
    label: u8,
    max_new_tokens: usize,
    generated_tokens: usize,
    live: bool,
    cancellation_deferred: bool,
}

#[derive(Clone)]
struct GeneratedPending {
    labels: Vec<u8>,
}

#[derive(Default)]
struct GenerationState {
    requests: Vec<GeneratedRequest>,
    pending: Option<GeneratedPending>,
}

impl GenerationState {
    fn live_count(&self) -> usize {
        self.requests.iter().filter(|request| request.live).count()
    }

    fn can_submit(&self) -> bool {
        self.pending.is_none()
            && self.requests.len() < MAX_LOGICAL_REQUESTS
            && self.live_count() < GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS
    }

    fn submit(&mut self, label: u8, max_new_tokens: u8) {
        assert!(self.can_submit());
        self.requests.push(GeneratedRequest {
            label,
            max_new_tokens: usize::from(max_new_tokens),
            generated_tokens: 0,
            live: true,
            cancellation_deferred: false,
        });
    }

    fn begin_plan(&mut self) {
        assert!(self.pending.is_none());
        let labels = self.semantic_plan_labels();
        assert!(!labels.is_empty());
        self.pending = Some(GeneratedPending { labels });
    }

    fn complete(&mut self) {
        let pending = self.pending.take().expect("generated plan exists");
        for label in pending.labels {
            let request = self.request_mut(label);
            if request.cancellation_deferred {
                request.cancellation_deferred = false;
                request.live = false;
                continue;
            }
            request.generated_tokens += 1;
            if request.generated_tokens == request.max_new_tokens {
                request.live = false;
            }
        }
    }

    fn abort_not_dispatched(&mut self) {
        let pending = self.pending.take().expect("generated plan exists");
        for label in pending.labels {
            let request = self.request_mut(label);
            if request.cancellation_deferred {
                request.cancellation_deferred = false;
                request.live = false;
            }
        }
    }

    fn cancel_idle(&mut self, label: u8) {
        assert!(self.pending.is_none());
        self.request_mut(label).live = false;
    }

    fn defer_cancel(&mut self, label: u8) {
        let pending = self.pending.as_ref().expect("generated plan exists");
        assert!(pending.labels.contains(&label));
        let request = self.request_mut(label);
        assert!(request.live && !request.cancellation_deferred);
        request.cancellation_deferred = true;
    }

    fn can_defer_cancel(&self) -> bool {
        self.pending.as_ref().is_some_and(|pending| {
            pending.labels.iter().any(|label| {
                self.requests
                    .iter()
                    .find(|request| request.label == *label)
                    .is_some_and(|request| request.live && !request.cancellation_deferred)
            })
        })
    }

    fn random_live_label(&self, random: &mut Lcg) -> u8 {
        let labels = self
            .requests
            .iter()
            .filter(|request| request.live)
            .map(|request| request.label)
            .collect::<Vec<_>>();
        labels[random.bounded_usize(labels.len())]
    }

    fn random_pending_label(&self, random: &mut Lcg) -> u8 {
        let pending = self.pending.as_ref().expect("generated plan exists");
        let labels = pending
            .labels
            .iter()
            .copied()
            .filter(|label| {
                self.requests
                    .iter()
                    .find(|request| request.label == *label)
                    .is_some_and(|request| request.live && !request.cancellation_deferred)
            })
            .collect::<Vec<_>>();
        labels[random.bounded_usize(labels.len())]
    }

    fn semantic_plan_labels(&self) -> Vec<u8> {
        let (decode, prefill): (Vec<_>, Vec<_>) = self
            .requests
            .iter()
            .filter(|request| request.live)
            .partition(|request| request.generated_tokens != 0);
        decode
            .into_iter()
            .chain(prefill)
            .map(|request| request.label)
            .collect()
    }

    fn request_mut(&mut self, label: u8) -> &mut GeneratedRequest {
        self.requests
            .iter_mut()
            .find(|request| request.label == label)
            .expect("generated program label is submitted")
    }
}

fn generated_prompt_len(random: &mut Lcg) -> u8 {
    1 + u8::try_from(random.bounded_usize(GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS))
        .expect("generated prompt width fits u8")
}

fn generated_max_new_tokens(random: &mut Lcg) -> u8 {
    1 + u8::try_from(random.bounded_usize(GENERAL_MIXED_PROGRAM_MAX_NEW_TOKENS))
        .expect("generated output capacity fits u8")
}

fn append_generated_submit(
    state: &mut GenerationState,
    operations: &mut Vec<GeneralMixedProgramOperation>,
    next_label: &mut u8,
    prompt_len: u8,
    max_new_tokens: u8,
) {
    if !state.can_submit() || usize::from(*next_label) > MAX_LOGICAL_REQUESTS {
        return;
    }
    let label = *next_label;
    state.submit(label, max_new_tokens);
    operations.push(GeneralMixedProgramOperation::Submit {
        label,
        prompt_len,
        max_new_tokens,
    });
    *next_label = next_label
        .checked_add(1)
        .expect("generated general mixed next label");
}

fn append_generated_plan(
    state: &mut GenerationState,
    operations: &mut Vec<GeneralMixedProgramOperation>,
) {
    state.begin_plan();
    operations.push(GeneralMixedProgramOperation::Plan);
}

fn append_generated_complete(
    state: &mut GenerationState,
    operations: &mut Vec<GeneralMixedProgramOperation>,
    random: &mut Lcg,
) {
    let slot_count = state
        .pending
        .as_ref()
        .expect("generated plan exists")
        .labels
        .len();
    let mut feedback_slot_order = canonical_slot_order(slot_count);
    for index in (1..feedback_slot_order.len()).rev() {
        let other = random.bounded_usize(index + 1);
        feedback_slot_order.swap(index, other);
    }
    operations.push(GeneralMixedProgramOperation::Complete {
        feedback_slot_order,
    });
    state.complete();
}
