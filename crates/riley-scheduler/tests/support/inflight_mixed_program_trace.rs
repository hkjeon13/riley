//! Canonical descriptors and a pure oracle for bounded in-flight mixed programs.
//!
//! This module deliberately does not import `Scheduler` or `IterationPlan`.
//! The public-API adapter validates a public plan projection independently and
//! supplies only opaque request IDs, iteration IDs, and public updates here.

use std::collections::{BTreeMap, BTreeSet};

use riley_scheduler::{
    CancellationOutcome, IterationId, IterationOutput, IterationResult, IterationUpdates,
    OutputSlot, RequestCompletion, RequestFinishReason, RequestId, SchedulerCloseOutput,
    TokenEvent,
};
use serde::{Deserialize, Serialize};

const DESCRIPTOR_FORMAT: &str = "riley.scheduler.inflight-mixed-program";
const DESCRIPTOR_FORMAT_VERSION: u8 = 1;
const DESCRIPTOR_TRACE_KIND: &str = "inflight-mixed-program-v1";
const MAX_LOGICAL_REQUESTS: usize = 4;
const MAX_LIVE_REQUESTS: usize = 3;
const MAX_PLAN_OPERATIONS: usize = 4;
const MAX_OPERATIONS: usize = 16;

/// One explicit operation in a bounded in-flight mixed program.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum InflightMixedProgramOperation {
    /// Admits one unique request with a fixed one-token prompt.
    Submit {
        /// Logical label in the bounded program domain.
        label: u8,
        /// Output capacity in the bounded 1..=2 domain.
        max_new_tokens: u8,
    },
    /// Opens one immutable plan for every currently live request.
    Plan,
    /// Defers cancellation of one request selected by the outstanding plan.
    Cancel {
        /// Logical label selected by the outstanding plan.
        label: u8,
    },
    /// Commits exact permuted feedback for every outstanding dense slot.
    Complete {
        /// Exact permutation of the outstanding semantic slot domain.
        feedback_slot_order: Vec<u8>,
    },
    /// Rolls back an undispatched plan and requires a fresh plan next.
    AbortNotDispatched,
    /// Consumes the scheduler after every preceding plan has settled.
    Close,
}

impl InflightMixedProgramOperation {
    /// Returns a stable, human-readable operation spelling for diagnostics.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Submit {
                label,
                max_new_tokens,
            } => format!("submit(label={label}, max_new={max_new_tokens})"),
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
struct InflightMixedProgramDescriptorV1 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    source_seed: String,
    operations: Vec<InflightMixedProgramOperation>,
}

/// Fully specified bounded in-flight raw program.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InflightMixedProgramTrace {
    /// Source provenance retained in reports and fixtures.
    pub seed: u64,
    /// Complete raw program within the bounded V1 grammar.
    pub operations: Vec<InflightMixedProgramOperation>,
}

impl InflightMixedProgramTrace {
    /// Produces one valid bounded in-flight program from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut random = Lcg(seed ^ 0x751d_c802_9af4_6b3e);
        let mut state = GenerationState::default();
        let mut operations = Vec::new();
        let mut next_label = 1_u8;
        let mode = (seed >> 2) % 3;
        let final_max_new_tokens = if mode == 1 { 2 } else { 1 };

        match seed & 0b11 {
            0 | 3 => {
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_plan(&mut operations);
                append_generated_complete(&mut state, &mut operations, &mut random, None);
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    final_max_new_tokens,
                );
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    final_max_new_tokens,
                );
            }
            1 => {
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_plan(&mut operations);
                append_generated_complete(&mut state, &mut operations, &mut random, None);
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    final_max_new_tokens,
                );
            }
            2 => {
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 2);
                append_generated_submit(&mut state, &mut operations, &mut next_label, 1);
                append_generated_plan(&mut operations);
                append_generated_complete(&mut state, &mut operations, &mut random, None);
                append_generated_submit(
                    &mut state,
                    &mut operations,
                    &mut next_label,
                    final_max_new_tokens,
                );
            }
            _ => unreachable!("two-bit topology selector"),
        }

        append_generated_plan(&mut operations);
        let deferred_label = if mode == 1 {
            None
        } else {
            let labels = state.live_labels();
            let label = labels[random.bounded_usize(labels.len())];
            operations.push(InflightMixedProgramOperation::Cancel { label });
            Some(label)
        };
        match mode {
            0 => {
                append_generated_complete(&mut state, &mut operations, &mut random, deferred_label);
            }
            1 | 2 => {
                operations.push(InflightMixedProgramOperation::AbortNotDispatched);
                state.abort_not_dispatched(deferred_label);
                append_generated_plan(&mut operations);
                append_generated_complete(&mut state, &mut operations, &mut random, None);
            }
            _ => unreachable!("three-way settlement selector"),
        }
        operations.push(InflightMixedProgramOperation::Close);

        let trace = Self { seed, operations };
        trace
            .validate()
            .expect("seeded in-flight mixed program remains in the V1 grammar");
        trace
    }

    /// Returns the explicit program spelling without consulting a public plan.
    #[must_use]
    pub fn describe_operations(&self) -> String {
        self.operations
            .iter()
            .map(InflightMixedProgramOperation::describe)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    /// Validates all syntax-independent V1 state-machine bounds and transitions.
    pub fn validate(&self) -> Result<(), String> {
        validate_operations(&self.operations)
    }

    fn descriptor(&self, case_id: &str) -> Result<InflightMixedProgramDescriptorV1, String> {
        self.validate()?;
        validate_case_id(case_id)?;
        Ok(InflightMixedProgramDescriptorV1 {
            format: DESCRIPTOR_FORMAT.to_owned(),
            format_version: DESCRIPTOR_FORMAT_VERSION,
            trace_kind: DESCRIPTOR_TRACE_KIND.to_owned(),
            case_id: case_id.to_owned(),
            source_seed: format!("0x{:016x}", self.seed),
            operations: self.operations.clone(),
        })
    }
}

/// A parsed corpus descriptor with its stable case identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedInflightMixedProgramTrace {
    /// Durable corpus or failure-report identifier.
    pub case_id: String,
    /// Fully validated raw program.
    pub trace: InflightMixedProgramTrace,
}

impl InflightMixedProgramDescriptorV1 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("in-flight mixed program descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err(
                "in-flight mixed program descriptor format_version is unsupported".to_owned(),
            );
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND {
            return Err("in-flight mixed program descriptor trace_kind is unsupported".to_owned());
        }
        validate_case_id(&self.case_id)?;
        parse_source_seed(&self.source_seed)?;
        validate_operations(&self.operations)
    }

    fn into_named_trace(self) -> Result<NamedInflightMixedProgramTrace, String> {
        self.validate()?;
        Ok(NamedInflightMixedProgramTrace {
            case_id: self.case_id,
            trace: InflightMixedProgramTrace {
                seed: parse_source_seed(&self.source_seed)?,
                operations: self.operations,
            },
        })
    }
}

/// Serializes one valid raw program as exact canonical JSON with a newline.
#[must_use]
pub fn serialize_inflight_mixed_program_descriptor(
    case_id: &str,
    trace: &InflightMixedProgramTrace,
) -> String {
    let descriptor = trace
        .descriptor(case_id)
        .expect("in-flight mixed program is valid before serialization");
    descriptor_document(&descriptor)
}

/// Parses only exact canonical in-flight mixed program V1 JSON documents.
pub fn parse_inflight_mixed_program_descriptor(
    document: &str,
) -> Result<NamedInflightMixedProgramTrace, String> {
    let descriptor = serde_json::from_str::<InflightMixedProgramDescriptorV1>(document)
        .map_err(|error| format!("in-flight mixed program descriptor JSON is invalid: {error}"))?;
    if document != descriptor_document(&descriptor) {
        return Err("in-flight mixed program descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_named_trace()
}

const CORPUS_DOCUMENTS_V1: [(&str, &str); 4] = [
    (
        "inflight-mixed-program-v1/deferred-prefill-cancel-reverse-complete.json",
        include_str!(
            "../corpus/output-routing/inflight-mixed-program-v1/deferred-prefill-cancel-reverse-complete.json"
        ),
    ),
    (
        "inflight-mixed-program-v1/deferred-decoder-cancel-abort-retry.json",
        include_str!(
            "../corpus/output-routing/inflight-mixed-program-v1/deferred-decoder-cancel-abort-retry.json"
        ),
    ),
    (
        "inflight-mixed-program-v1/abort-retry-preserves-history-three-slot.json",
        include_str!(
            "../corpus/output-routing/inflight-mixed-program-v1/abort-retry-preserves-history-three-slot.json"
        ),
    ),
    (
        "inflight-mixed-program-v1/abort-retry-close-live-survivor.json",
        include_str!(
            "../corpus/output-routing/inflight-mixed-program-v1/abort-retry-close-live-survivor.json"
        ),
    ),
];

/// Loads the committed canonical corpus and rejects duplicate case identifiers.
#[must_use]
pub fn inflight_mixed_program_corpus() -> Vec<NamedInflightMixedProgramTrace> {
    let mut case_ids = BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V1.len());
    for (path, document) in CORPUS_DOCUMENTS_V1 {
        let named = parse_inflight_mixed_program_descriptor(document).unwrap_or_else(|error| {
            panic!("{path}: in-flight mixed program corpus is invalid: {error}")
        });
        assert!(
            case_ids.insert(named.case_id.clone()),
            "{path}: in-flight mixed program corpus repeats case_id {:?}",
            named.case_id
        );
        corpus.push(named);
    }
    corpus
}

#[derive(Clone)]
struct ValidationRequest {
    max_new_tokens: usize,
    generated_tokens: usize,
    submission_order: usize,
    live: bool,
}

struct PendingValidation {
    labels: Vec<u8>,
    deferred_label: Option<u8>,
}

// Keeping the finite transition table together makes grammar review possible
// without following state mutations through several helper functions.
#[allow(clippy::too_many_lines)]
fn validate_operations(operations: &[InflightMixedProgramOperation]) -> Result<(), String> {
    if operations.is_empty() || operations.len() > MAX_OPERATIONS {
        return Err(format!(
            "in-flight mixed program descriptor operations must contain 1..={MAX_OPERATIONS} entries"
        ));
    }
    let mut requests = BTreeMap::<u8, ValidationRequest>::new();
    let mut pending = None::<PendingValidation>;
    let mut close_seen = false;
    let mut plan_count = 0_usize;
    let mut abort_count = 0_usize;
    let mut cancellation_count = 0_usize;
    let mut next_submission_order = 0_usize;
    let mut mixed_plan_seen = false;

    for (index, operation) in operations.iter().enumerate() {
        if close_seen {
            return Err(
                "in-flight mixed program descriptor contains an operation after close".to_owned(),
            );
        }
        match operation {
            InflightMixedProgramOperation::Submit {
                label,
                max_new_tokens,
            } => {
                if pending.is_some() {
                    return Err(
                        "in-flight mixed program descriptor submits while a plan is outstanding"
                            .to_owned(),
                    );
                }
                validate_label(*label)?;
                if !(1..=2).contains(max_new_tokens) {
                    return Err(
                        "in-flight mixed program descriptor max_new_tokens must be in 1..=2"
                            .to_owned(),
                    );
                }
                if requests.len() >= MAX_LOGICAL_REQUESTS {
                    return Err(
                        "in-flight mixed program descriptor exceeds its unique request cap"
                            .to_owned(),
                    );
                }
                if live_request_count(&requests) >= MAX_LIVE_REQUESTS {
                    return Err(
                        "in-flight mixed program descriptor exceeds its live request cap"
                            .to_owned(),
                    );
                }
                if requests
                    .insert(
                        *label,
                        ValidationRequest {
                            max_new_tokens: usize::from(*max_new_tokens),
                            generated_tokens: 0,
                            submission_order: next_submission_order,
                            live: true,
                        },
                    )
                    .is_some()
                {
                    return Err(
                        "in-flight mixed program descriptor submits one label more than once"
                            .to_owned(),
                    );
                }
                next_submission_order = next_submission_order.checked_add(1).ok_or_else(|| {
                    "in-flight mixed program descriptor submission order overflowed".to_owned()
                })?;
            }
            InflightMixedProgramOperation::Plan => {
                if pending.is_some() {
                    return Err(
                        "in-flight mixed program descriptor plans while a plan is outstanding"
                            .to_owned(),
                    );
                }
                if plan_count >= MAX_PLAN_OPERATIONS {
                    return Err(
                        "in-flight mixed program descriptor exceeds its plan cap".to_owned()
                    );
                }
                let labels = plan_labels(&requests);
                if labels.is_empty() {
                    return Err(
                        "in-flight mixed program descriptor plans without live requests".to_owned(),
                    );
                }
                let has_decode = labels
                    .iter()
                    .any(|label| requests[label].generated_tokens != 0);
                let has_prefill = labels
                    .iter()
                    .any(|label| requests[label].generated_tokens == 0);
                mixed_plan_seen |= has_decode && has_prefill;
                pending = Some(PendingValidation {
                    labels,
                    deferred_label: None,
                });
                plan_count += 1;
            }
            InflightMixedProgramOperation::Cancel { label } => {
                validate_label(*label)?;
                if cancellation_count >= 1 {
                    return Err(
                        "in-flight mixed program descriptor allows at most one deferred cancellation"
                            .to_owned(),
                    );
                }
                let pending = pending.as_mut().ok_or_else(|| {
                    "in-flight mixed program descriptor cancels without an outstanding plan"
                        .to_owned()
                })?;
                if pending.deferred_label.is_some() {
                    return Err(
                        "in-flight mixed program descriptor cancels more than one pending label"
                            .to_owned(),
                    );
                }
                if !pending.labels.contains(label) {
                    return Err(
                        "in-flight mixed program descriptor cancels a label outside its outstanding plan"
                            .to_owned(),
                    );
                }
                if !requests.get(label).is_some_and(|request| request.live) {
                    return Err(
                        "in-flight mixed program descriptor cancels a non-live label".to_owned(),
                    );
                }
                pending.deferred_label = Some(*label);
                cancellation_count += 1;
            }
            InflightMixedProgramOperation::Complete {
                feedback_slot_order,
            } => {
                let pending = pending.take().ok_or_else(|| {
                    "in-flight mixed program descriptor completes without an outstanding plan"
                        .to_owned()
                })?;
                validate_slot_order(
                    feedback_slot_order,
                    pending.labels.len(),
                    "feedback_slot_order",
                )?;
                for label in pending.labels {
                    let request = requests.get_mut(&label).ok_or_else(|| {
                        "in-flight mixed program descriptor lost a pending request".to_owned()
                    })?;
                    if !request.live {
                        return Err(
                            "in-flight mixed program descriptor completed a non-live request"
                                .to_owned(),
                        );
                    }
                    if pending.deferred_label == Some(label) {
                        request.live = false;
                        continue;
                    }
                    request.generated_tokens =
                        request.generated_tokens.checked_add(1).ok_or_else(|| {
                            "in-flight mixed program descriptor generation count overflowed"
                                .to_owned()
                        })?;
                    if request.generated_tokens == request.max_new_tokens {
                        request.live = false;
                    }
                }
            }
            InflightMixedProgramOperation::AbortNotDispatched => {
                if abort_count >= 1 {
                    return Err(
                        "in-flight mixed program descriptor allows at most one not-dispatched abort"
                            .to_owned(),
                    );
                }
                if index + 1 == operations.len()
                    || !matches!(operations[index + 1], InflightMixedProgramOperation::Plan)
                {
                    return Err(
                        "in-flight mixed program descriptor requires a fresh plan immediately after not-dispatched abort"
                            .to_owned(),
                    );
                }
                let pending = pending.take().ok_or_else(|| {
                    "in-flight mixed program descriptor aborts without an outstanding plan"
                        .to_owned()
                })?;
                if let Some(label) = pending.deferred_label {
                    let request = requests.get_mut(&label).ok_or_else(|| {
                        "in-flight mixed program descriptor lost a deferred request".to_owned()
                    })?;
                    request.live = false;
                }
                abort_count += 1;
            }
            InflightMixedProgramOperation::Close => {
                if pending.is_some() {
                    return Err(
                        "in-flight mixed program descriptor closes with a plan outstanding"
                            .to_owned(),
                    );
                }
                if index + 1 != operations.len() {
                    return Err(
                        "in-flight mixed program descriptor close must be the final operation"
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
        return Err("in-flight mixed program descriptor must end with close".to_owned());
    }
    if plan_count < 2 {
        return Err(
            "in-flight mixed program descriptor requires at least two plan operations".to_owned(),
        );
    }
    if !mixed_plan_seen {
        return Err(
            "in-flight mixed program descriptor requires one decode plus prefill mixed plan"
                .to_owned(),
        );
    }
    Ok(())
}

fn plan_labels(requests: &BTreeMap<u8, ValidationRequest>) -> Vec<u8> {
    let mut live = requests
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

fn live_request_count(requests: &BTreeMap<u8, ValidationRequest>) -> usize {
    requests.values().filter(|request| request.live).count()
}

fn validate_slot_order(order: &[u8], slot_count: usize, field: &str) -> Result<(), String> {
    if order.len() != slot_count {
        return Err(format!(
            "in-flight mixed program descriptor {field} must contain exactly {slot_count} slots"
        ));
    }
    let mut seen = BTreeSet::new();
    for slot in order {
        if usize::from(*slot) >= slot_count {
            return Err(format!(
                "in-flight mixed program descriptor {field} contains an out-of-range slot"
            ));
        }
        if !seen.insert(*slot) {
            return Err(format!(
                "in-flight mixed program descriptor {field} contains a duplicate slot"
            ));
        }
    }
    Ok(())
}

fn validate_label(label: u8) -> Result<(), String> {
    if !(1..=u8::try_from(MAX_LOGICAL_REQUESTS).expect("bounded label")).contains(&label) {
        return Err("in-flight mixed program descriptor label must be in 1..=4".to_owned());
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
            "in-flight mixed program descriptor case_id must be a bounded lowercase identifier"
                .to_owned(),
        );
    }
    Ok(())
}

fn parse_source_seed(source_seed: &str) -> Result<u64, String> {
    let Some(hex) = source_seed.strip_prefix("0x") else {
        return Err("in-flight mixed program descriptor source_seed must start with 0x".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(
            "in-flight mixed program descriptor source_seed must be 16 lowercase hexadecimal digits"
                .to_owned(),
        );
    }
    u64::from_str_radix(hex, 16)
        .map_err(|_| "in-flight mixed program descriptor source_seed does not fit u64".to_owned())
}

fn descriptor_document(descriptor: &InflightMixedProgramDescriptorV1) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("in-flight mixed program descriptor serializes");
    document.push('\n');
    document
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
            "in-flight mixed program RNG bound must be nonzero"
        );
        usize::try_from(self.next() % u64::try_from(upper_exclusive).expect("usize fits u64"))
            .expect("in-flight mixed program RNG value fits usize")
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

    fn complete(&mut self, deferred_label: Option<u8>) {
        for request in self.requests.iter_mut().filter(|request| request.live) {
            if deferred_label == Some(request.label) {
                request.live = false;
                continue;
            }
            request.generated_tokens += 1;
            if request.generated_tokens == request.max_new_tokens {
                request.live = false;
            }
        }
    }

    fn abort_not_dispatched(&mut self, deferred_label: Option<u8>) {
        if let Some(label) = deferred_label {
            let request = self
                .requests
                .iter_mut()
                .find(|request| request.label == label)
                .expect("generated in-flight program aborts a submitted label");
            assert!(request.live);
            request.live = false;
        }
    }
}

fn append_generated_submit(
    state: &mut GenerationState,
    operations: &mut Vec<InflightMixedProgramOperation>,
    next_label: &mut u8,
    max_new_tokens: u8,
) {
    let label = *next_label;
    state.submit(label, max_new_tokens);
    operations.push(InflightMixedProgramOperation::Submit {
        label,
        max_new_tokens,
    });
    *next_label = next_label
        .checked_add(1)
        .expect("in-flight mixed program next logical label");
}

fn append_generated_plan(operations: &mut Vec<InflightMixedProgramOperation>) {
    operations.push(InflightMixedProgramOperation::Plan);
}

fn append_generated_complete(
    state: &mut GenerationState,
    operations: &mut Vec<InflightMixedProgramOperation>,
    random: &mut Lcg,
    deferred_label: Option<u8>,
) {
    let mut feedback_slot_order = (0..state.live_count())
        .map(|slot| u8::try_from(slot).expect("in-flight mixed program feedback slot"))
        .collect::<Vec<_>>();
    for index in (1..feedback_slot_order.len()).rev() {
        let other = random.bounded_usize(index + 1);
        feedback_slot_order.swap(index, other);
    }
    operations.push(InflightMixedProgramOperation::Complete {
        feedback_slot_order,
    });
    state.complete(deferred_label);
}

/// Expected work for one semantic output slot in a grammar-derived plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InflightMixedProgramPlanItem {
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

/// Public-plan projection derived only from raw-program state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InflightMixedProgramExpectedPlan {
    /// Decode work in canonical ready/request order.
    pub decode_items: Vec<InflightMixedProgramPlanItem>,
    /// Prefill work in canonical ready/request order.
    pub prefill_items: Vec<InflightMixedProgramPlanItem>,
}

impl InflightMixedProgramExpectedPlan {
    /// Returns the dense output-slot domain in grammar canonical order.
    #[must_use]
    pub fn output_slots(&self) -> Vec<OutputSlot> {
        self.decode_items
            .iter()
            .chain(&self.prefill_items)
            .map(|item| item.output_slot)
            .collect()
    }

    /// Returns the one-token work total for every selected live request.
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
    cancellation_deferred: bool,
}

#[derive(Clone, Copy)]
struct PendingOutput {
    label: u8,
    request_id: RequestId,
    generated_index: usize,
    output_slot: OutputSlot,
}

/// Pure routing and lifecycle oracle for bounded in-flight raw programs.
///
/// This is not a general scheduler model. The grammar config makes every live
/// request admissible and selected on each plan, so the oracle only tracks
/// bounded labels, output histories, deferred cancellation, plan projection,
/// terminal events, and `NotDispatched` rollback.
pub struct InflightMixedProgramOracle {
    seed: u64,
    requests: BTreeMap<u8, OracleRequest>,
    pending: Option<Vec<PendingOutput>>,
    terminal: BTreeMap<RequestId, CompletionProjection>,
    next_submission_order: usize,
    phase: OraclePhase,
}

impl InflightMixedProgramOracle {
    /// Creates an unbound oracle before public request IDs are known.
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
            "seed {:#018x}: in-flight program label escaped its grammar",
            self.seed
        );
        assert!(
            (1..=2).contains(&max_new_tokens),
            "seed {:#018x}: in-flight program output capacity escaped its grammar",
            self.seed
        );
        assert!(
            self.live_count() < MAX_LIVE_REQUESTS,
            "seed {:#018x}: in-flight program exceeded its live request cap",
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
                        cancellation_deferred: false,
                    },
                )
                .is_none(),
            "seed {:#018x}: in-flight program bound the same logical label twice",
            self.seed
        );
        self.next_submission_order = self
            .next_submission_order
            .checked_add(1)
            .expect("in-flight program submission ordinal");
    }

    /// Derives and opens the exact expected public plan before feedback exists.
    #[must_use]
    pub fn begin_plan(&mut self) -> InflightMixedProgramExpectedPlan {
        self.assert_phase(OraclePhase::Idle);
        let mut live = self
            .requests
            .iter()
            .filter(|(_, request)| request.live)
            .map(|(label, request)| (*label, request))
            .collect::<Vec<_>>();
        assert!(
            !live.is_empty(),
            "seed {:#018x}: in-flight program attempted to plan without live work",
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
            let output_slot =
                OutputSlot::new(u32::try_from(pending.len()).expect("in-flight output slot"));
            let generated_index = request.generated_token_ids.len();
            let item = InflightMixedProgramPlanItem {
                label,
                request_id: request.request_id,
                input_token: if generated_index == 0 {
                    inflight_symbolic_prompt_token(label)
                } else {
                    *request
                        .generated_token_ids
                        .last()
                        .expect("in-flight decoder has one prior generated token")
                },
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
        self.phase = OraclePhase::InFlight;
        InflightMixedProgramExpectedPlan {
            decode_items,
            prefill_items,
        }
    }

    /// Records one public deferred cancellation of an outstanding-plan label.
    pub fn defer_cancel(&mut self, label: u8, outcome: &CancellationOutcome) {
        self.assert_phase(OraclePhase::InFlight);
        let pending = self
            .pending
            .as_ref()
            .expect("in-flight program oracle retained its plan projection");
        assert!(
            pending.iter().any(|item| item.label == label),
            "seed {:#018x}: in-flight program cancelled an unplanned label {label}",
            self.seed
        );
        let request = self.request(label);
        assert!(request.live);
        assert!(!request.cancellation_deferred);
        assert_eq!(outcome.request_id(), request.request_id);
        assert!(outcome.deferred_until_iteration_settles());
        assert!(!outcome.already_terminal());
        assert!(outcome.completion().is_none());
        self.request_mut(label).cancellation_deferred = true;
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
            .expect("in-flight program oracle retained its plan projection");
        validate_slot_order(feedback_slot_order, pending.len(), "feedback_slot_order")
            .unwrap_or_else(|error| {
                panic!(
                    "seed {:#018x}: in-flight program feedback permutation is invalid: {error}",
                    self.seed
                )
            });
        let outputs = feedback_slot_order
            .iter()
            .map(|slot| {
                let item = pending[usize::from(*slot)];
                IterationOutput::new(
                    item.output_slot,
                    inflight_symbolic_token(item.label, item.generated_index),
                    false,
                )
            })
            .collect();
        IterationResult::new(iteration_id, outputs, 0, 0)
            .expect("in-flight program oracle feedback has unique slots")
    }

    /// Compares valid completion with the grammar-derived output ledger.
    pub fn record_complete(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(OraclePhase::InFlight);
        let pending = self
            .pending
            .take()
            .expect("in-flight program oracle retained its plan projection");
        let mut expected_tokens = BTreeMap::new();
        let mut expected_completions = BTreeMap::new();
        for item in &pending {
            let request = self.request(item.label);
            assert!(request.live);
            assert_eq!(request.request_id, item.request_id);
            assert_eq!(request.generated_token_ids.len(), item.generated_index);
            if request.cancellation_deferred {
                assert!(
                    expected_completions
                        .insert(
                            item.request_id,
                            CompletionProjection {
                                reason: RequestFinishReason::Cancelled,
                                generated_token_ids: request.generated_token_ids.clone(),
                                completed_at_ns: now_ns,
                            },
                        )
                        .is_none(),
                    "seed {:#018x}: in-flight program repeated a deferred completion",
                    self.seed
                );
                continue;
            }
            let token_id = inflight_symbolic_token(item.label, item.generated_index);
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
                "seed {:#018x}: in-flight program repeated a pending request",
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
                    "seed {:#018x}: in-flight program repeated a terminal request",
                    self.seed
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
                .push(inflight_symbolic_token(item.label, item.generated_index));
            if request.generated_token_ids.len() == request.max_new_tokens {
                request.live = false;
            }
        }
        self.phase = OraclePhase::Idle;
    }

    /// Compares a `NotDispatched` rollback with the grammar-derived ledger.
    pub fn record_not_dispatched_abort(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(OraclePhase::InFlight);
        let pending = self
            .pending
            .take()
            .expect("in-flight program oracle retained its plan projection");
        let mut expected_completions = BTreeMap::new();
        for item in &pending {
            let request = self.request(item.label);
            assert!(request.live);
            assert_eq!(request.request_id, item.request_id);
            if request.cancellation_deferred {
                assert!(
                    expected_completions
                        .insert(
                            item.request_id,
                            CompletionProjection {
                                reason: RequestFinishReason::Cancelled,
                                generated_token_ids: request.generated_token_ids.clone(),
                                completed_at_ns: now_ns,
                            },
                        )
                        .is_none(),
                    "seed {:#018x}: in-flight program repeated an abort cancellation",
                    self.seed
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
        self.phase = OraclePhase::Idle;
    }

    /// Compares consuming close with every remaining live request.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(OraclePhase::Idle);
        assert!(
            closed.settlement_failures().is_empty(),
            "seed {:#018x}: in-flight program close contained a settlement failure",
            self.seed
        );
        let expected = self
            .requests
            .values()
            .filter(|request| request.live)
            .map(|request| {
                assert!(!request.cancellation_deferred);
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
            "seed {:#018x}: in-flight program close did not settle every submission",
            self.seed
        );
        self.phase = OraclePhase::Closed;
    }

    /// Asserts the adapter reached the exact terminal close state.
    pub fn assert_closed(&self) {
        self.assert_phase(OraclePhase::Closed);
        assert_eq!(
            self.terminal.len(),
            self.requests.len(),
            "seed {:#018x}: in-flight program terminal ledger is incomplete",
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
            "seed {:#018x}: in-flight program update contained a settlement failure",
            self.seed
        );
        assert_eq!(
            &token_projections(updates.token_events()),
            expected_tokens,
            "seed {:#018x}: in-flight program token routing drifted",
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
            "seed {:#018x}: in-flight program terminal ledger drifted",
            self.seed
        );
        for (request_id, completion) in expected_completions {
            assert!(
                self.terminal.insert(request_id, completion).is_none(),
                "seed {:#018x}: in-flight program emitted a duplicate terminal request",
                self.seed
            );
        }
    }

    fn assert_phase(&self, expected: OraclePhase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: in-flight program oracle phase drifted",
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
                "seed {:#018x}: in-flight program label {label} is unbound",
                self.seed
            )
        })
    }

    fn request_mut(&mut self, label: u8) -> &mut OracleRequest {
        self.requests.get_mut(&label).unwrap_or_else(|| {
            panic!(
                "seed {:#018x}: in-flight program label {label} is unbound",
                self.seed
            )
        })
    }
}

/// Returns the one-token prompt bound to a logical program label.
#[must_use]
pub fn inflight_symbolic_prompt_token(label: u8) -> u32 {
    2_000_u32
        .checked_add(u32::from(label))
        .expect("in-flight program symbolic prompt token")
}

/// Returns the sampled token bound to a logical label and generation index.
#[must_use]
pub fn inflight_symbolic_token(label: u8, generated_index: usize) -> u32 {
    u32::from(label)
        .checked_mul(32)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("in-flight program symbolic output token")
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
            "in-flight program public updates emitted duplicate token events for one request"
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
            "in-flight program public updates emitted duplicate terminal completions"
        );
    }
    projections
}
