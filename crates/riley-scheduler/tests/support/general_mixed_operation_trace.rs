//! Canonical descriptors and a pure oracle for the bounded C03-A
//! general-mixed-operation V1 grammar.
//!
//! This support module deliberately does not import `Scheduler` or `IterationPlan`.
//! The public-API adapter validates those plan projections separately, then
//! gives this oracle only opaque request IDs and an iteration ID. Slot-to-
//! request feedback, token history, terminal completion, and close expectations
//! come from the descriptor grammar alone.

use std::collections::BTreeMap;

use riley_scheduler::{
    IterationId, IterationOutput, IterationResult, IterationUpdates, OutputSlot, RequestCompletion,
    RequestFinishReason, RequestId, SchedulerCloseOutput, TokenEvent,
};
use serde::{Deserialize, Serialize};

const DESCRIPTOR_FORMAT: &str = "riley.scheduler.general-mixed-operation";
const DESCRIPTOR_FORMAT_VERSION: u8 = 1;
const DESCRIPTOR_TRACE_KIND: &str = "general-mixed-operation-v1";
const MAX_WAVE_WIDTH: usize = 3;

/// Chooses the final transition for the second mixed wave.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneralMixedOperationSettlement {
    /// Submit grammar-generated, descriptor-permuted feedback.
    Commit,
    /// Roll the in-flight plan back before it is dispatched.
    AbortNotDispatched,
}

impl GeneralMixedOperationSettlement {
    /// Human-readable operation spelling used in failure reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Commit => "complete",
            Self::AbortNotDispatched => "abort(not-dispatched)",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
struct RequiredNullable<T>(Option<T>);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralMixedOperationTraceDescriptorV1 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    source_seed: String,
    decoder_count: u8,
    final_prefill_count: u8,
    prime_slot_order: Vec<u8>,
    mixed_slot_order: Vec<u8>,
    cancel_decoder_index: RequiredNullable<u8>,
    settlement: GeneralMixedOperationSettlement,
}

/// Full replay selector set for the parameterized two-wave V1 grammar.
///
/// Every decoder has prompt length one and a generation cap of two. Every
/// final-prefill request has prompt length one and a generation cap of one.
/// The descriptor's two slot orders are exact permutations, not random replay
/// recipes; seed is provenance only.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMixedOperationTrace {
    /// Source provenance for reports and committed fixtures.
    pub seed: u64,
    /// Number of A decoder requests in 1..=3.
    pub decoder_count: usize,
    /// Number of B final-prefill requests in 1..=3.
    pub final_prefill_count: usize,
    /// Exact feedback order for prime slots `0..decoder_count`.
    pub prime_slot_order: Vec<u8>,
    /// Exact feedback order for mixed slots `0..decoder_count+final_prefill_count`.
    pub mixed_slot_order: Vec<u8>,
    /// Optional decoder A index cancelled while the mixed plan is in flight.
    pub cancel_decoder_index: Option<usize>,
    /// Commit or pre-dispatch abort for the mixed plan.
    pub settlement: GeneralMixedOperationSettlement,
}

impl GeneralMixedOperationTrace {
    /// Derives one bounded descriptor from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let mut random = Lcg(seed ^ 0x2ed0_76f7_4f11_b6c9);
        let decoder_count = 1 + random.bounded_usize(MAX_WAVE_WIDTH);
        let final_prefill_count = 1 + random.bounded_usize(MAX_WAVE_WIDTH);
        let mut prime_slot_order = (0..decoder_count)
            .map(|slot| u8::try_from(slot).expect("bounded prime slot fits u8"))
            .collect::<Vec<_>>();
        let mut mixed_slot_order = (0..decoder_count + final_prefill_count)
            .map(|slot| u8::try_from(slot).expect("bounded mixed slot fits u8"))
            .collect::<Vec<_>>();
        shuffle_slots(&mut prime_slot_order, &mut random);
        shuffle_slots(&mut mixed_slot_order, &mut random);
        let cancel_decoder_index = match random.bounded_usize(decoder_count + 1) {
            0 => None,
            selected => Some(selected - 1),
        };
        Self {
            seed,
            decoder_count,
            final_prefill_count,
            prime_slot_order,
            mixed_slot_order,
            cancel_decoder_index,
            settlement: if random.next() & 1 == 0 {
                GeneralMixedOperationSettlement::Commit
            } else {
                GeneralMixedOperationSettlement::AbortNotDispatched
            },
        }
    }

    /// Describes the canonical operation sequence without consulting a plan.
    #[must_use]
    pub fn describe_operations(&self) -> String {
        let cancellation = self.cancel_decoder_index.map_or_else(
            || "none".to_owned(),
            |index| format!("cancel decoder[{index}]"),
        );
        format!(
            "submit decoder[0..{}) -> plan-prime -> complete-prime(order={:?}) -> \
             submit final-prefill[0..{}) -> plan-mixed -> {} -> {}(order={:?}) -> close",
            self.decoder_count,
            self.prime_slot_order,
            self.final_prefill_count,
            cancellation,
            self.settlement.name(),
            self.mixed_slot_order,
        )
    }

    /// Returns deterministic, valid simplifications for this V1 grammar only.
    ///
    /// The order is cancellation removal, lower cancellation target, one
    /// decoder removal at each index, one final-prefill removal at each index,
    /// direct identity permutations, then one adjacent inversion swap at a
    /// time. Seed, settlement, and the two-wave topology never change.
    #[must_use]
    pub fn shrink_candidates(&self) -> Vec<Self> {
        self.validate()
            .expect("general mixed shrinker requires a valid source descriptor");
        let source_rank = self.shrink_rank();
        let mut candidates = Vec::new();
        if let Some(cancelled) = self.cancel_decoder_index {
            push_unique_candidate(
                &mut candidates,
                Self {
                    cancel_decoder_index: None,
                    ..self.clone()
                },
            );
            for simplified in 0..cancelled {
                push_unique_candidate(
                    &mut candidates,
                    Self {
                        cancel_decoder_index: Some(simplified),
                        ..self.clone()
                    },
                );
            }
        }
        if self.decoder_count > 1 {
            for removed in 0..self.decoder_count {
                push_unique_candidate(&mut candidates, self.remove_decoder(removed));
            }
        }
        if self.final_prefill_count > 1 {
            for removed in 0..self.final_prefill_count {
                push_unique_candidate(&mut candidates, self.remove_final_prefill(removed));
            }
        }
        let canonical_prime = canonical_slot_order(self.decoder_count);
        if self.prime_slot_order != canonical_prime {
            push_unique_candidate(
                &mut candidates,
                Self {
                    prime_slot_order: canonical_prime,
                    ..self.clone()
                },
            );
        }
        let mixed_slot_count = self
            .decoder_count
            .checked_add(self.final_prefill_count)
            .expect("bounded V1 mixed slot count");
        let canonical_mixed = canonical_slot_order(mixed_slot_count);
        if self.mixed_slot_order != canonical_mixed {
            push_unique_candidate(
                &mut candidates,
                Self {
                    mixed_slot_order: canonical_mixed,
                    ..self.clone()
                },
            );
        }
        for index in 0..self.prime_slot_order.len().saturating_sub(1) {
            if self.prime_slot_order[index] > self.prime_slot_order[index + 1] {
                let mut prime_slot_order = self.prime_slot_order.clone();
                prime_slot_order.swap(index, index + 1);
                push_unique_candidate(
                    &mut candidates,
                    Self {
                        prime_slot_order,
                        ..self.clone()
                    },
                );
            }
        }
        for index in 0..self.mixed_slot_order.len().saturating_sub(1) {
            if self.mixed_slot_order[index] > self.mixed_slot_order[index + 1] {
                let mut mixed_slot_order = self.mixed_slot_order.clone();
                mixed_slot_order.swap(index, index + 1);
                push_unique_candidate(
                    &mut candidates,
                    Self {
                        mixed_slot_order,
                        ..self.clone()
                    },
                );
            }
        }
        for candidate in &candidates {
            candidate
                .validate()
                .expect("general mixed shrink candidate remains a valid descriptor");
            assert!(
                candidate.shrink_rank() < source_rank,
                "general mixed shrink candidate must strictly reduce its rank"
            );
            assert_eq!(candidate.seed, self.seed);
            assert_eq!(candidate.settlement, self.settlement);
        }
        candidates
    }

    /// Returns the lexicographic rank used by the bounded V1 local reducer.
    ///
    /// It is total request width, cancellation presence, cancellation target,
    /// and total inversion count. It is not a general counterexample metric.
    #[must_use]
    pub fn shrink_rank(&self) -> (usize, usize, usize, usize) {
        self.validate()
            .expect("general mixed shrink rank requires a valid descriptor");
        (
            self.decoder_count + self.final_prefill_count,
            usize::from(self.cancel_decoder_index.is_some()),
            self.cancel_decoder_index.unwrap_or_default(),
            inversion_count(&self.prime_slot_order) + inversion_count(&self.mixed_slot_order),
        )
    }

    fn validate(&self) -> Result<(), String> {
        validate_trace_fields(
            self.decoder_count,
            self.final_prefill_count,
            &self.prime_slot_order,
            &self.mixed_slot_order,
            self.cancel_decoder_index,
        )
    }

    fn remove_decoder(&self, removed: usize) -> Self {
        assert!(
            self.decoder_count > 1 && removed < self.decoder_count,
            "general mixed reducer must remove one existing decoder"
        );
        let cancel_decoder_index = match self.cancel_decoder_index {
            None => None,
            Some(index) if index == removed => None,
            Some(index) if index < removed => Some(index),
            Some(index) => Some(index - 1),
        };
        Self {
            decoder_count: self.decoder_count - 1,
            prime_slot_order: project_removed_slot(&self.prime_slot_order, removed),
            mixed_slot_order: project_removed_slot(&self.mixed_slot_order, removed),
            cancel_decoder_index,
            ..self.clone()
        }
    }

    fn remove_final_prefill(&self, removed: usize) -> Self {
        assert!(
            self.final_prefill_count > 1 && removed < self.final_prefill_count,
            "general mixed reducer must remove one existing final-prefill"
        );
        let removed_slot = self
            .decoder_count
            .checked_add(removed)
            .expect("bounded V1 final-prefill slot");
        Self {
            final_prefill_count: self.final_prefill_count - 1,
            mixed_slot_order: project_removed_slot(&self.mixed_slot_order, removed_slot),
            ..self.clone()
        }
    }

    fn descriptor(&self, case_id: &str) -> Result<GeneralMixedOperationTraceDescriptorV1, String> {
        self.validate()?;
        validate_case_id(case_id)?;
        Ok(GeneralMixedOperationTraceDescriptorV1 {
            format: DESCRIPTOR_FORMAT.to_owned(),
            format_version: DESCRIPTOR_FORMAT_VERSION,
            trace_kind: DESCRIPTOR_TRACE_KIND.to_owned(),
            case_id: case_id.to_owned(),
            source_seed: format!("0x{:016x}", self.seed),
            decoder_count: u8::try_from(self.decoder_count)
                .map_err(|_| "general mixed descriptor decoder_count does not fit u8".to_owned())?,
            final_prefill_count: u8::try_from(self.final_prefill_count).map_err(|_| {
                "general mixed descriptor final_prefill_count does not fit u8".to_owned()
            })?,
            prime_slot_order: self.prime_slot_order.clone(),
            mixed_slot_order: self.mixed_slot_order.clone(),
            cancel_decoder_index: RequiredNullable(
                self.cancel_decoder_index
                    .map(|index| {
                        u8::try_from(index).map_err(|_| {
                            "general mixed descriptor cancel index does not fit u8".to_owned()
                        })
                    })
                    .transpose()?,
            ),
            settlement: self.settlement,
        })
    }
}

/// A parsed corpus descriptor with its durable case identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedGeneralMixedOperationTrace {
    /// Identifier that stays stable across corpus replays.
    pub case_id: String,
    /// Fully validated grammar selectors.
    pub trace: GeneralMixedOperationTrace,
}

impl GeneralMixedOperationTraceDescriptorV1 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("general mixed descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err("general mixed descriptor format_version is unsupported".to_owned());
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND {
            return Err("general mixed descriptor trace_kind is unsupported".to_owned());
        }
        validate_case_id(&self.case_id)?;
        parse_source_seed(&self.source_seed)?;
        validate_trace_fields(
            usize::from(self.decoder_count),
            usize::from(self.final_prefill_count),
            &self.prime_slot_order,
            &self.mixed_slot_order,
            self.cancel_decoder_index.0.map(usize::from),
        )
    }

    fn into_named_trace(self) -> Result<NamedGeneralMixedOperationTrace, String> {
        self.validate()?;
        Ok(NamedGeneralMixedOperationTrace {
            case_id: self.case_id,
            trace: GeneralMixedOperationTrace {
                seed: parse_source_seed(&self.source_seed)?,
                decoder_count: usize::from(self.decoder_count),
                final_prefill_count: usize::from(self.final_prefill_count),
                prime_slot_order: self.prime_slot_order,
                mixed_slot_order: self.mixed_slot_order,
                cancel_decoder_index: self.cancel_decoder_index.0.map(usize::from),
                settlement: self.settlement,
            },
        })
    }
}

/// Serializes a fully specified trace as exact canonical JSON with a newline.
#[must_use]
pub fn serialize_general_mixed_operation_trace_descriptor(
    case_id: &str,
    trace: &GeneralMixedOperationTrace,
) -> String {
    let descriptor = trace
        .descriptor(case_id)
        .expect("general mixed trace is valid before serialization");
    descriptor_document(&descriptor)
}

/// Parses only strict canonical general-mixed-operation V1 JSON documents.
pub fn parse_general_mixed_operation_trace_descriptor(
    document: &str,
) -> Result<NamedGeneralMixedOperationTrace, String> {
    let descriptor = serde_json::from_str::<GeneralMixedOperationTraceDescriptorV1>(document)
        .map_err(|error| format!("general mixed descriptor JSON is invalid: {error}"))?;
    let canonical = descriptor_document(&descriptor);
    if document != canonical {
        return Err("general mixed descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_named_trace()
}

/// Greedily minimizes a reproducing V1 trace over the fixed candidate order.
///
/// Every candidate is strict-codec serialized and parsed again before the
/// caller's predicate sees it. The returned descriptor is a local minimum only
/// for this bounded grammar and this predicate; it does not preserve a panic
/// site, payload, failure signature, or root cause.
pub fn minimize_general_mixed_operation_trace<F>(
    trace: &GeneralMixedOperationTrace,
    mut reproduces: F,
) -> GeneralMixedOperationTrace
where
    F: FnMut(&GeneralMixedOperationTrace) -> bool,
{
    let mut minimized = strict_round_trip_trace(trace, "shrink-source");
    assert!(
        reproduces(&minimized),
        "general mixed minimization requires a reproducing source trace"
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

const CORPUS_DOCUMENTS_V1: [(&str, &str); 3] = [
    (
        "general-mixed-operation-v1/two-by-two-reverse-commit.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v1/two-by-two-reverse-commit.json"
        ),
    ),
    (
        "general-mixed-operation-v1/three-by-one-cancel-commit.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v1/three-by-one-cancel-commit.json"
        ),
    ),
    (
        "general-mixed-operation-v1/three-by-three-permuted-abort.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v1/three-by-three-permuted-abort.json"
        ),
    ),
];

/// Loads the committed canonical corpus and rejects duplicate case IDs.
#[must_use]
pub fn general_mixed_operation_corpus() -> Vec<NamedGeneralMixedOperationTrace> {
    let mut case_ids = std::collections::BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V1.len());
    for (path, document) in CORPUS_DOCUMENTS_V1 {
        let named = parse_general_mixed_operation_trace_descriptor(document)
            .unwrap_or_else(|error| panic!("{path}: general mixed corpus is invalid: {error}"));
        assert!(
            case_ids.insert(named.case_id.clone()),
            "{path}: general mixed corpus repeats case_id {:?}",
            named.case_id
        );
        corpus.push(named);
    }
    corpus
}

/// Returns the fixed symbolic token emitted by decoder A at one generation index.
#[must_use]
pub fn decoder_symbolic_token(index: usize, generated_index: usize) -> u32 {
    token_for(decoder_label(index), generated_index)
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
enum Phase {
    BindingDecoders,
    DecodersBound,
    PrimePlanned,
    PrimeSettled,
    BindingFinalPrefills,
    FinalPrefillsBound,
    MixedPlanned,
    Settled,
    Closed,
}

/// Pure routing/lifecycle oracle for the parameterized two-wave V1 grammar.
///
/// The first wave assigns slot i to decoder A[index]. The mixed wave assigns
/// `0..decoder_count` to decoder A and the following slots to final-prefill B.
/// Descriptor-selected feedback permutations rearrange only those fixed slots;
/// no plan item participates in feedback construction.
pub struct GeneralMixedOperationOracle {
    seed: u64,
    decoder_count: usize,
    final_prefill_count: usize,
    decoders: Vec<Option<RequestId>>,
    final_prefills: Vec<Option<RequestId>>,
    decoder_histories: Vec<Vec<u32>>,
    final_prefill_histories: Vec<Vec<u32>>,
    cancelled_decoder_index: Option<usize>,
    terminal: BTreeMap<RequestId, CompletionProjection>,
    phase: Phase,
}

impl GeneralMixedOperationOracle {
    /// Creates an unbound pure oracle for a descriptor's two fixed waves.
    #[must_use]
    pub fn new(seed: u64, decoder_count: usize, final_prefill_count: usize) -> Self {
        assert!(
            (1..=MAX_WAVE_WIDTH).contains(&decoder_count),
            "seed {seed:#018x}: decoder count escaped the V1 grammar"
        );
        assert!(
            (1..=MAX_WAVE_WIDTH).contains(&final_prefill_count),
            "seed {seed:#018x}: final-prefill count escaped the V1 grammar"
        );
        Self {
            seed,
            decoder_count,
            final_prefill_count,
            decoders: vec![None; decoder_count],
            final_prefills: vec![None; final_prefill_count],
            decoder_histories: vec![Vec::with_capacity(2); decoder_count],
            final_prefill_histories: vec![Vec::with_capacity(1); final_prefill_count],
            cancelled_decoder_index: None,
            terminal: BTreeMap::new(),
            phase: Phase::BindingDecoders,
        }
    }

    /// Binds one opaque public request ID for decoder A[index].
    pub fn bind_decoder(&mut self, index: usize, request_id: RequestId) {
        self.assert_phase(Phase::BindingDecoders);
        assert!(
            index < self.decoder_count,
            "seed {:#018x}: decoder binding escaped the grammar width",
            self.seed
        );
        assert!(
            self.decoders[index].replace(request_id).is_none(),
            "seed {:#018x}: decoder {index} was bound twice",
            self.seed
        );
        if self.decoders.iter().all(Option::is_some) {
            self.phase = Phase::DecodersBound;
        }
    }

    /// Records the adapter's independently checked prime plan projection.
    pub fn observe_prime_plan(&mut self) {
        self.assert_phase(Phase::DecodersBound);
        self.phase = Phase::PrimePlanned;
    }

    /// Builds prime feedback from the descriptor permutation and fixed slots.
    #[must_use]
    pub fn prime_feedback(&self, iteration_id: IterationId, slot_order: &[u8]) -> IterationResult {
        self.assert_phase(Phase::PrimePlanned);
        self.assert_slot_order(slot_order, self.decoder_count, "prime");
        let outputs = slot_order
            .iter()
            .map(|slot| {
                let index = usize::from(*slot);
                IterationOutput::new(
                    OutputSlot::new(u32::from(*slot)),
                    decoder_symbolic_token(index, 0),
                    false,
                )
            })
            .collect();
        Self::result(iteration_id, outputs)
    }

    /// Compares prime commit updates with the grammar-derived ledger.
    pub fn record_prime_commit(&mut self, updates: &IterationUpdates) {
        self.assert_phase(Phase::PrimePlanned);
        let expected_tokens = (0..self.decoder_count)
            .map(|index| {
                (
                    self.decoder(index),
                    TokenProjection {
                        token_id: decoder_symbolic_token(index, 0),
                        generated_index: 0,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        self.assert_updates(updates, &expected_tokens, BTreeMap::new());
        for index in 0..self.decoder_count {
            self.decoder_histories[index].push(decoder_symbolic_token(index, 0));
        }
        self.phase = Phase::PrimeSettled;
    }

    /// Binds one opaque public request ID for final-prefill B[index].
    pub fn bind_final_prefill(&mut self, index: usize, request_id: RequestId) {
        assert!(
            matches!(
                self.phase,
                Phase::PrimeSettled | Phase::BindingFinalPrefills
            ),
            "seed {:#018x}: final-prefill binding occurred outside the prime-to-mixed transition",
            self.seed
        );
        assert!(
            index < self.final_prefill_count,
            "seed {:#018x}: final-prefill binding escaped the grammar width",
            self.seed
        );
        assert!(
            self.final_prefills[index].replace(request_id).is_none(),
            "seed {:#018x}: final-prefill {index} was bound twice",
            self.seed
        );
        self.phase = if self.final_prefills.iter().all(Option::is_some) {
            Phase::FinalPrefillsBound
        } else {
            Phase::BindingFinalPrefills
        };
    }

    /// Records the adapter's independently checked mixed plan projection.
    pub fn observe_mixed_plan(&mut self) {
        self.assert_phase(Phase::FinalPrefillsBound);
        self.phase = Phase::MixedPlanned;
    }

    /// Records the descriptor's optional in-flight cancellation.
    pub fn defer_decoder_cancel(&mut self, index: usize) {
        self.assert_phase(Phase::MixedPlanned);
        assert!(
            index < self.decoder_count,
            "seed {:#018x}: cancelled decoder escaped the grammar width",
            self.seed
        );
        assert!(
            self.cancelled_decoder_index.replace(index).is_none(),
            "seed {:#018x}: more than one decoder was cancelled in V1",
            self.seed
        );
    }

    /// Builds mixed feedback from fixed decoder/final-prefill slots only.
    #[must_use]
    pub fn mixed_feedback(&self, iteration_id: IterationId, slot_order: &[u8]) -> IterationResult {
        self.assert_phase(Phase::MixedPlanned);
        let slot_count = self
            .decoder_count
            .checked_add(self.final_prefill_count)
            .expect("bounded V1 mixed slot count");
        self.assert_slot_order(slot_order, slot_count, "mixed");
        let outputs = slot_order
            .iter()
            .map(|slot| {
                let index = usize::from(*slot);
                let token = if index < self.decoder_count {
                    decoder_symbolic_token(index, 1)
                } else {
                    final_prefill_symbolic_token(index - self.decoder_count, 0)
                };
                IterationOutput::new(OutputSlot::new(u32::from(*slot)), token, false)
            })
            .collect();
        Self::result(iteration_id, outputs)
    }

    /// Compares mixed commit updates with the grammar-derived terminal ledger.
    pub fn record_mixed_commit(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(Phase::MixedPlanned);
        let mut expected_tokens = BTreeMap::new();
        let mut expected_completions = BTreeMap::new();
        for index in 0..self.final_prefill_count {
            let request_id = self.final_prefill(index);
            let token_id = final_prefill_symbolic_token(index, 0);
            assert!(
                expected_tokens
                    .insert(
                        request_id,
                        TokenProjection {
                            token_id,
                            generated_index: 0,
                        },
                    )
                    .is_none(),
                "seed {:#018x}: V1 final-prefill token map repeated a request",
                self.seed
            );
            assert!(
                expected_completions
                    .insert(
                        request_id,
                        CompletionProjection {
                            reason: RequestFinishReason::Length,
                            generated_token_ids: vec![token_id],
                            completed_at_ns: now_ns,
                        },
                    )
                    .is_none(),
                "seed {:#018x}: V1 final-prefill completion map repeated a request",
                self.seed
            );
        }
        for index in 0..self.decoder_count {
            let request_id = self.decoder(index);
            if self.cancelled_decoder_index == Some(index) {
                expected_completions.insert(
                    request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: self.decoder_histories[index].clone(),
                        completed_at_ns: now_ns,
                    },
                );
            } else {
                let token_id = decoder_symbolic_token(index, 1);
                expected_tokens.insert(
                    request_id,
                    TokenProjection {
                        token_id,
                        generated_index: 1,
                    },
                );
                expected_completions.insert(
                    request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Length,
                        generated_token_ids: vec![
                            decoder_symbolic_token(index, 0),
                            decoder_symbolic_token(index, 1),
                        ],
                        completed_at_ns: now_ns,
                    },
                );
            }
        }
        self.assert_updates(updates, &expected_tokens, expected_completions);
        for index in 0..self.final_prefill_count {
            self.final_prefill_histories[index].push(final_prefill_symbolic_token(index, 0));
        }
        for index in 0..self.decoder_count {
            if self.cancelled_decoder_index != Some(index) {
                self.decoder_histories[index].push(decoder_symbolic_token(index, 1));
            }
        }
        self.phase = Phase::Settled;
    }

    /// Compares a `NotDispatched` rollback with the grammar-derived ledger.
    pub fn record_not_dispatched_abort(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(Phase::MixedPlanned);
        let expected_completions =
            self.cancelled_decoder_index
                .map_or_else(BTreeMap::new, |index| {
                    BTreeMap::from([(
                        self.decoder(index),
                        CompletionProjection {
                            reason: RequestFinishReason::Cancelled,
                            generated_token_ids: self.decoder_histories[index].clone(),
                            completed_at_ns: now_ns,
                        },
                    )])
                });
        self.assert_updates(updates, &BTreeMap::new(), expected_completions);
        self.phase = Phase::Settled;
    }

    /// Compares consuming close with every still-live grammar request.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(Phase::Settled);
        assert!(
            closed.settlement_failures().is_empty(),
            "seed {:#018x}: V1 close contained a settlement failure",
            self.seed
        );
        let mut expected_completions = BTreeMap::new();
        for index in 0..self.decoder_count {
            let request_id = self.decoder(index);
            if !self.terminal.contains_key(&request_id) {
                expected_completions.insert(
                    request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: self.decoder_histories[index].clone(),
                        completed_at_ns: now_ns,
                    },
                );
            }
        }
        for index in 0..self.final_prefill_count {
            let request_id = self.final_prefill(index);
            if !self.terminal.contains_key(&request_id) {
                expected_completions.insert(
                    request_id,
                    CompletionProjection {
                        reason: RequestFinishReason::Cancelled,
                        generated_token_ids: self.final_prefill_histories[index].clone(),
                        completed_at_ns: now_ns,
                    },
                );
            }
        }
        self.assert_completions(closed.completions(), expected_completions);
        assert_eq!(
            self.terminal.len(),
            self.decoder_count + self.final_prefill_count,
            "seed {:#018x}: V1 close did not terminally settle every request",
            self.seed
        );
        self.phase = Phase::Closed;
    }

    /// Asserts that the adapter reached exactly one terminal entry per request.
    pub fn assert_closed(&self) {
        self.assert_phase(Phase::Closed);
        assert_eq!(
            self.terminal.len(),
            self.decoder_count + self.final_prefill_count,
            "seed {:#018x}: V1 terminal ledger is incomplete",
            self.seed
        );
    }

    fn result(iteration_id: IterationId, outputs: Vec<IterationOutput>) -> IterationResult {
        IterationResult::new(iteration_id, outputs, 0, 0)
            .expect("general mixed oracle feedback has unique output slots")
    }

    fn assert_updates(
        &mut self,
        updates: &IterationUpdates,
        expected_tokens: &BTreeMap<RequestId, TokenProjection>,
        expected_completions: BTreeMap<RequestId, CompletionProjection>,
    ) {
        assert!(
            updates.settlement_failures().is_empty(),
            "seed {:#018x}: V1 update contained a settlement failure",
            self.seed
        );
        assert_eq!(
            &token_projections(updates.token_events()),
            expected_tokens,
            "seed {:#018x}: V1 grammar token routing drifted",
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
            "seed {:#018x}: V1 grammar terminal ledger drifted",
            self.seed
        );
        for (request_id, completion) in expected_completions {
            assert!(
                self.terminal.insert(request_id, completion).is_none(),
                "seed {:#018x}: V1 grammar emitted a duplicate terminal request",
                self.seed
            );
        }
    }

    fn assert_slot_order(&self, order: &[u8], slot_count: usize, phase: &str) {
        validate_slot_order(order, slot_count, phase).unwrap_or_else(|error| {
            panic!(
                "seed {:#018x}: V1 {phase} slot order is invalid: {error}",
                self.seed
            )
        });
    }

    fn assert_phase(&self, expected: Phase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: V1 oracle phase drifted",
            self.seed
        );
    }

    fn decoder(&self, index: usize) -> RequestId {
        self.decoders[index].expect("general mixed decoder request ID is bound")
    }

    fn final_prefill(&self, index: usize) -> RequestId {
        self.final_prefills[index].expect("general mixed final-prefill request ID is bound")
    }
}

fn validate_trace_fields(
    decoder_count: usize,
    final_prefill_count: usize,
    prime_slot_order: &[u8],
    mixed_slot_order: &[u8],
    cancel_decoder_index: Option<usize>,
) -> Result<(), String> {
    if !(1..=MAX_WAVE_WIDTH).contains(&decoder_count) {
        return Err("general mixed descriptor decoder_count must be in 1..=3".to_owned());
    }
    if !(1..=MAX_WAVE_WIDTH).contains(&final_prefill_count) {
        return Err("general mixed descriptor final_prefill_count must be in 1..=3".to_owned());
    }
    validate_slot_order(prime_slot_order, decoder_count, "prime_slot_order")?;
    let mixed_slot_count = decoder_count
        .checked_add(final_prefill_count)
        .ok_or_else(|| "general mixed descriptor mixed slot count overflowed".to_owned())?;
    validate_slot_order(mixed_slot_order, mixed_slot_count, "mixed_slot_order")?;
    if cancel_decoder_index.is_some_and(|index| index >= decoder_count) {
        return Err(
            "general mixed descriptor cancel_decoder_index must select a decoder".to_owned(),
        );
    }
    Ok(())
}

fn validate_slot_order(order: &[u8], slot_count: usize, field: &str) -> Result<(), String> {
    if order.len() != slot_count {
        return Err(format!(
            "general mixed descriptor {field} must contain exactly {slot_count} slots"
        ));
    }
    let mut seen = std::collections::BTreeSet::new();
    for slot in order {
        if usize::from(*slot) >= slot_count {
            return Err(format!(
                "general mixed descriptor {field} contains an out-of-range slot"
            ));
        }
        if !seen.insert(*slot) {
            return Err(format!(
                "general mixed descriptor {field} contains a duplicate slot"
            ));
        }
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
            "general mixed descriptor case_id must be a bounded lowercase identifier".to_owned(),
        );
    }
    Ok(())
}

fn parse_source_seed(source_seed: &str) -> Result<u64, String> {
    let Some(hex) = source_seed.strip_prefix("0x") else {
        return Err("general mixed descriptor source_seed must start with 0x".to_owned());
    };
    if hex.len() != 16
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(
            "general mixed descriptor source_seed must be 16 lowercase hexadecimal digits"
                .to_owned(),
        );
    }
    u64::from_str_radix(hex, 16)
        .map_err(|_| "general mixed descriptor source_seed does not fit u64".to_owned())
}

fn descriptor_document(descriptor: &GeneralMixedOperationTraceDescriptorV1) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("general mixed descriptor serializes");
    document.push('\n');
    document
}

fn strict_round_trip_trace(
    trace: &GeneralMixedOperationTrace,
    case_id: &str,
) -> GeneralMixedOperationTrace {
    let document = serialize_general_mixed_operation_trace_descriptor(case_id, trace);
    let parsed = parse_general_mixed_operation_trace_descriptor(&document)
        .expect("general mixed reducer descriptor remains strict-canonical");
    assert_eq!(
        parsed.trace, *trace,
        "general mixed reducer round-trip changed a selector"
    );
    parsed.trace
}

fn push_unique_candidate(
    candidates: &mut Vec<GeneralMixedOperationTrace>,
    candidate: GeneralMixedOperationTrace,
) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn canonical_slot_order(slot_count: usize) -> Vec<u8> {
    (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("bounded V1 canonical slot fits u8"))
        .collect()
}

fn project_removed_slot(order: &[u8], removed_slot: usize) -> Vec<u8> {
    order
        .iter()
        .filter_map(|slot| {
            let slot = usize::from(*slot);
            match slot.cmp(&removed_slot) {
                std::cmp::Ordering::Less => {
                    Some(u8::try_from(slot).expect("bounded V1 retained slot fits u8"))
                }
                std::cmp::Ordering::Equal => None,
                std::cmp::Ordering::Greater => {
                    Some(u8::try_from(slot - 1).expect("bounded V1 compacted slot fits u8"))
                }
            }
        })
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

fn decoder_label(index: usize) -> u32 {
    u32::try_from(index + 1).expect("bounded general mixed decoder label")
}

fn final_prefill_label(index: usize) -> u32 {
    32_u32
        .checked_add(u32::try_from(index).expect("bounded general mixed final-prefill label"))
        .expect("bounded general mixed final-prefill label")
}

fn final_prefill_symbolic_token(index: usize, generated_index: usize) -> u32 {
    token_for(final_prefill_label(index), generated_index)
}

fn token_for(label: u32, generated_index: usize) -> u32 {
    label
        .checked_mul(16)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("bounded general mixed symbolic output token")
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
            "general mixed public updates emitted duplicate token events for one request"
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
            "general mixed public updates emitted duplicate terminal completions for one request"
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
            upper_exclusive > 0,
            "general mixed descriptor RNG bound must be nonzero"
        );
        usize::try_from(self.next() % u64::try_from(upper_exclusive).expect("usize fits u64"))
            .expect("bounded general mixed random value fits usize")
    }
}

fn shuffle_slots(slots: &mut [u8], random: &mut Lcg) {
    for end in (1..slots.len()).rev() {
        let index = random.bounded_usize(end + 1);
        slots.swap(end, index);
    }
}

/// A deliberately malformed feedback shape injected before the V2 mixed-plan
/// settlement.
///
/// The V2 grammar permits at most one such rejection. It does not model
/// arbitrary invalid feedback streams or advance the pure oracle's pending
/// plan state.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneralMixedOperationRejectedFeedback {
    /// Uses a different iteration identifier while retaining grammar-derived slots.
    Stale,
    /// Omits one otherwise grammar-derived dense output slot.
    Missing,
    /// Replaces one grammar-derived dense output slot with an unplanned slot.
    Unplanned,
}

impl GeneralMixedOperationRejectedFeedback {
    /// Human-readable operation spelling used in V2 failure reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Stale => "stale",
            Self::Missing => "missing",
            Self::Unplanned => "unplanned",
        }
    }
}

const DESCRIPTOR_TRACE_KIND_V2: &str = "general-mixed-operation-v2";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct GeneralMixedOperationTraceDescriptorV2 {
    format: String,
    format_version: u8,
    trace_kind: String,
    case_id: String,
    source_seed: String,
    decoder_count: u8,
    final_prefill_count: u8,
    prime_slot_order: Vec<u8>,
    mixed_slot_order: Vec<u8>,
    cancel_decoder_index: RequiredNullable<u8>,
    rejected_feedback: RequiredNullable<GeneralMixedOperationRejectedFeedback>,
    settlement: GeneralMixedOperationSettlement,
}

/// Full replay selector set for the bounded two-wave V2 grammar.
///
/// V2 preserves V1's decoder/final-prefill topology and adds one optional,
/// explicitly serialized rejected mixed-feedback attempt before the same
/// pending plan is validly settled. It is not an arbitrary raw operation
/// grammar or a general scheduler model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMixedOperationTraceV2 {
    /// Source provenance for reports and committed fixtures.
    pub seed: u64,
    /// Number of A decoder requests in 1..=3.
    pub decoder_count: usize,
    /// Number of B final-prefill requests in 1..=3.
    pub final_prefill_count: usize,
    /// Exact feedback order for prime slots `0..decoder_count`.
    pub prime_slot_order: Vec<u8>,
    /// Exact feedback order for mixed slots `0..decoder_count+final_prefill_count`.
    pub mixed_slot_order: Vec<u8>,
    /// Optional decoder A index cancelled while the mixed plan is in flight.
    pub cancel_decoder_index: Option<usize>,
    /// Optional rejected feedback attempted before the valid terminal settlement.
    pub rejected_feedback: Option<GeneralMixedOperationRejectedFeedback>,
    /// Commit or pre-dispatch abort for the mixed plan after any rejection.
    pub settlement: GeneralMixedOperationSettlement,
}

impl GeneralMixedOperationTraceV2 {
    /// Derives one bounded V2 descriptor from a deterministic seed.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        let v1 = GeneralMixedOperationTrace::from_seed(seed);
        let mut random = Lcg(seed ^ 0x724a_1fc5_d31e_8b69);
        let rejected_feedback = match random.bounded_usize(4) {
            0 => None,
            1 => Some(GeneralMixedOperationRejectedFeedback::Stale),
            2 => Some(GeneralMixedOperationRejectedFeedback::Missing),
            _ => Some(GeneralMixedOperationRejectedFeedback::Unplanned),
        };
        Self {
            seed,
            decoder_count: v1.decoder_count,
            final_prefill_count: v1.final_prefill_count,
            prime_slot_order: v1.prime_slot_order,
            mixed_slot_order: v1.mixed_slot_order,
            cancel_decoder_index: v1.cancel_decoder_index,
            rejected_feedback,
            settlement: v1.settlement,
        }
    }

    /// Describes the canonical V2 operation sequence without consulting a plan.
    #[must_use]
    pub fn describe_operations(&self) -> String {
        let cancellation = self.cancel_decoder_index.map_or_else(
            || "none".to_owned(),
            |index| format!("cancel decoder[{index}]"),
        );
        let rejection = self.rejected_feedback.map_or_else(
            || "none".to_owned(),
            |kind| format!("reject-feedback({})", kind.name()),
        );
        format!(
            "submit decoder[0..{}) -> plan-prime -> complete-prime(order={:?}) -> \
             submit final-prefill[0..{}) -> plan-mixed -> {} -> {} -> {}(order={:?}) -> close",
            self.decoder_count,
            self.prime_slot_order,
            self.final_prefill_count,
            cancellation,
            rejection,
            self.settlement.name(),
            self.mixed_slot_order,
        )
    }

    /// Returns deterministic, valid V2 simplifications.
    ///
    /// The order is rejected-feedback removal, then V1's cancellation,
    /// selector-rebase, identity-permutation, and adjacent-swap order. The
    /// source seed, settlement, and two-wave topology remain fixed.
    #[must_use]
    pub fn shrink_candidates(&self) -> Vec<Self> {
        self.validate()
            .expect("general mixed V2 shrinker requires a valid source descriptor");
        let source_rank = self.shrink_rank();
        let mut candidates = Vec::new();
        if self.rejected_feedback.is_some() {
            push_unique_v2_candidate(
                &mut candidates,
                Self {
                    rejected_feedback: None,
                    ..self.clone()
                },
            );
        }
        for candidate in self.v1_equivalent().shrink_candidates() {
            push_unique_v2_candidate(
                &mut candidates,
                Self::from_v1_equivalent(candidate, self.rejected_feedback),
            );
        }
        for candidate in &candidates {
            candidate
                .validate()
                .expect("general mixed V2 shrink candidate remains a valid descriptor");
            assert!(
                candidate.shrink_rank() < source_rank,
                "general mixed V2 shrink candidate must strictly reduce its rank"
            );
            assert_eq!(candidate.seed, self.seed);
            assert_eq!(candidate.settlement, self.settlement);
        }
        candidates
    }

    /// Returns the lexicographic rank used by the bounded V2 local reducer.
    ///
    /// This is request width, rejected-feedback presence, V1 cancellation
    /// fields, and total inversion count. It is not a general counterexample
    /// metric.
    #[must_use]
    pub fn shrink_rank(&self) -> (usize, usize, usize, usize, usize) {
        self.validate()
            .expect("general mixed V2 shrink rank requires a valid descriptor");
        let v1_rank = self.v1_equivalent().shrink_rank();
        (
            v1_rank.0,
            usize::from(self.rejected_feedback.is_some()),
            v1_rank.1,
            v1_rank.2,
            v1_rank.3,
        )
    }

    fn validate(&self) -> Result<(), String> {
        validate_trace_fields(
            self.decoder_count,
            self.final_prefill_count,
            &self.prime_slot_order,
            &self.mixed_slot_order,
            self.cancel_decoder_index,
        )
    }

    fn v1_equivalent(&self) -> GeneralMixedOperationTrace {
        GeneralMixedOperationTrace {
            seed: self.seed,
            decoder_count: self.decoder_count,
            final_prefill_count: self.final_prefill_count,
            prime_slot_order: self.prime_slot_order.clone(),
            mixed_slot_order: self.mixed_slot_order.clone(),
            cancel_decoder_index: self.cancel_decoder_index,
            settlement: self.settlement,
        }
    }

    fn from_v1_equivalent(
        trace: GeneralMixedOperationTrace,
        rejected_feedback: Option<GeneralMixedOperationRejectedFeedback>,
    ) -> Self {
        Self {
            seed: trace.seed,
            decoder_count: trace.decoder_count,
            final_prefill_count: trace.final_prefill_count,
            prime_slot_order: trace.prime_slot_order,
            mixed_slot_order: trace.mixed_slot_order,
            cancel_decoder_index: trace.cancel_decoder_index,
            rejected_feedback,
            settlement: trace.settlement,
        }
    }

    fn descriptor(&self, case_id: &str) -> Result<GeneralMixedOperationTraceDescriptorV2, String> {
        self.validate()?;
        validate_case_id(case_id)?;
        Ok(GeneralMixedOperationTraceDescriptorV2 {
            format: DESCRIPTOR_FORMAT.to_owned(),
            format_version: DESCRIPTOR_FORMAT_VERSION,
            trace_kind: DESCRIPTOR_TRACE_KIND_V2.to_owned(),
            case_id: case_id.to_owned(),
            source_seed: format!("0x{:016x}", self.seed),
            decoder_count: u8::try_from(self.decoder_count).map_err(|_| {
                "general mixed V2 descriptor decoder_count does not fit u8".to_owned()
            })?,
            final_prefill_count: u8::try_from(self.final_prefill_count).map_err(|_| {
                "general mixed V2 descriptor final_prefill_count does not fit u8".to_owned()
            })?,
            prime_slot_order: self.prime_slot_order.clone(),
            mixed_slot_order: self.mixed_slot_order.clone(),
            cancel_decoder_index: RequiredNullable(
                self.cancel_decoder_index
                    .map(|index| {
                        u8::try_from(index).map_err(|_| {
                            "general mixed V2 descriptor cancel index does not fit u8".to_owned()
                        })
                    })
                    .transpose()?,
            ),
            rejected_feedback: RequiredNullable(self.rejected_feedback),
            settlement: self.settlement,
        })
    }
}

/// A parsed V2 corpus descriptor with its durable case identifier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NamedGeneralMixedOperationTraceV2 {
    /// Identifier that stays stable across V2 corpus replays.
    pub case_id: String,
    /// Fully validated V2 grammar selectors.
    pub trace: GeneralMixedOperationTraceV2,
}

impl GeneralMixedOperationTraceDescriptorV2 {
    fn validate(&self) -> Result<(), String> {
        if self.format != DESCRIPTOR_FORMAT {
            return Err("general mixed V2 descriptor format is unsupported".to_owned());
        }
        if self.format_version != DESCRIPTOR_FORMAT_VERSION {
            return Err("general mixed V2 descriptor format_version is unsupported".to_owned());
        }
        if self.trace_kind != DESCRIPTOR_TRACE_KIND_V2 {
            return Err("general mixed V2 descriptor trace_kind is unsupported".to_owned());
        }
        validate_case_id(&self.case_id)?;
        parse_source_seed(&self.source_seed)?;
        validate_trace_fields(
            usize::from(self.decoder_count),
            usize::from(self.final_prefill_count),
            &self.prime_slot_order,
            &self.mixed_slot_order,
            self.cancel_decoder_index.0.map(usize::from),
        )
    }

    fn into_named_trace(self) -> Result<NamedGeneralMixedOperationTraceV2, String> {
        self.validate()?;
        Ok(NamedGeneralMixedOperationTraceV2 {
            case_id: self.case_id,
            trace: GeneralMixedOperationTraceV2 {
                seed: parse_source_seed(&self.source_seed)?,
                decoder_count: usize::from(self.decoder_count),
                final_prefill_count: usize::from(self.final_prefill_count),
                prime_slot_order: self.prime_slot_order,
                mixed_slot_order: self.mixed_slot_order,
                cancel_decoder_index: self.cancel_decoder_index.0.map(usize::from),
                rejected_feedback: self.rejected_feedback.0,
                settlement: self.settlement,
            },
        })
    }
}

/// Serializes a fully specified V2 trace as exact canonical JSON with a newline.
#[must_use]
pub fn serialize_general_mixed_operation_v2_trace_descriptor(
    case_id: &str,
    trace: &GeneralMixedOperationTraceV2,
) -> String {
    let descriptor = trace
        .descriptor(case_id)
        .expect("general mixed V2 trace is valid before serialization");
    v2_descriptor_document(&descriptor)
}

/// Parses only strict canonical general-mixed-operation V2 JSON documents.
pub fn parse_general_mixed_operation_v2_trace_descriptor(
    document: &str,
) -> Result<NamedGeneralMixedOperationTraceV2, String> {
    let descriptor = serde_json::from_str::<GeneralMixedOperationTraceDescriptorV2>(document)
        .map_err(|error| format!("general mixed V2 descriptor JSON is invalid: {error}"))?;
    let canonical = v2_descriptor_document(&descriptor);
    if document != canonical {
        return Err("general mixed V2 descriptor JSON is not canonical".to_owned());
    }
    descriptor.into_named_trace()
}

/// Greedily minimizes a reproducing V2 trace over the fixed candidate order.
///
/// Every candidate is strict-codec serialized and parsed again before the
/// caller's predicate sees it. The returned descriptor is a local minimum only
/// for this bounded grammar and predicate.
pub fn minimize_general_mixed_operation_v2_trace<F>(
    trace: &GeneralMixedOperationTraceV2,
    mut reproduces: F,
) -> GeneralMixedOperationTraceV2
where
    F: FnMut(&GeneralMixedOperationTraceV2) -> bool,
{
    let mut minimized = strict_round_trip_v2_trace(trace, "v2-shrink-source");
    assert!(
        reproduces(&minimized),
        "general mixed V2 minimization requires a reproducing source trace"
    );
    loop {
        let mut next = None;
        for candidate in minimized.shrink_candidates() {
            let candidate = strict_round_trip_v2_trace(&candidate, "v2-shrink-candidate");
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

const CORPUS_DOCUMENTS_V2: [(&str, &str); 3] = [
    (
        "general-mixed-operation-v2/two-by-two-stale-retry-commit.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v2/two-by-two-stale-retry-commit.json"
        ),
    ),
    (
        "general-mixed-operation-v2/three-by-one-missing-retry-commit.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v2/three-by-one-missing-retry-commit.json"
        ),
    ),
    (
        "general-mixed-operation-v2/three-by-three-unplanned-retry-abort.json",
        include_str!(
            "../corpus/output-routing/general-mixed-operation-v2/three-by-three-unplanned-retry-abort.json"
        ),
    ),
];

/// Loads the committed canonical V2 corpus and rejects duplicate case IDs.
#[must_use]
pub fn general_mixed_operation_v2_corpus() -> Vec<NamedGeneralMixedOperationTraceV2> {
    let mut case_ids = std::collections::BTreeSet::new();
    let mut corpus = Vec::with_capacity(CORPUS_DOCUMENTS_V2.len());
    for (path, document) in CORPUS_DOCUMENTS_V2 {
        let named = parse_general_mixed_operation_v2_trace_descriptor(document)
            .unwrap_or_else(|error| panic!("{path}: general mixed V2 corpus is invalid: {error}"));
        assert!(
            case_ids.insert(named.case_id.clone()),
            "{path}: general mixed V2 corpus repeats case_id {:?}",
            named.case_id
        );
        corpus.push(named);
    }
    corpus
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GeneralMixedOperationV2OraclePhase {
    BindingDecoders,
    DecodersBound,
    PrimePlanned,
    PrimeSettled,
    BindingFinalPrefills,
    FinalPrefillsBound,
    MixedPlanned,
    Settled,
    Closed,
}

/// Pure routing/lifecycle oracle for the bounded V2 grammar.
///
/// This wraps the unchanged V1 routing ledger and independently constructs the
/// optional malformed feedback from V2 descriptor fields. It never imports a
/// production scheduler or iteration plan.
pub struct GeneralMixedOperationOracleV2 {
    inner: GeneralMixedOperationOracle,
    seed: u64,
    decoder_count: usize,
    final_prefill_count: usize,
    bound_decoders: usize,
    bound_final_prefills: usize,
    phase: GeneralMixedOperationV2OraclePhase,
}

impl GeneralMixedOperationOracleV2 {
    /// Creates an unbound pure V2 oracle for one descriptor's fixed waves.
    #[must_use]
    pub fn new(seed: u64, decoder_count: usize, final_prefill_count: usize) -> Self {
        Self {
            inner: GeneralMixedOperationOracle::new(seed, decoder_count, final_prefill_count),
            seed,
            decoder_count,
            final_prefill_count,
            bound_decoders: 0,
            bound_final_prefills: 0,
            phase: GeneralMixedOperationV2OraclePhase::BindingDecoders,
        }
    }

    /// Binds one opaque public request ID for decoder A[index].
    pub fn bind_decoder(&mut self, index: usize, request_id: RequestId) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::BindingDecoders);
        self.inner.bind_decoder(index, request_id);
        self.bound_decoders = self
            .bound_decoders
            .checked_add(1)
            .expect("bounded V2 decoder binding count");
        self.phase = if self.bound_decoders == self.decoder_count {
            GeneralMixedOperationV2OraclePhase::DecodersBound
        } else {
            GeneralMixedOperationV2OraclePhase::BindingDecoders
        };
    }

    /// Records the independently validated public prime plan.
    pub fn observe_prime_plan(&mut self) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::DecodersBound);
        self.inner.observe_prime_plan();
        self.phase = GeneralMixedOperationV2OraclePhase::PrimePlanned;
    }

    /// Builds prime feedback from fixed V2 grammar slots.
    #[must_use]
    pub fn prime_feedback(&self, iteration_id: IterationId, slot_order: &[u8]) -> IterationResult {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::PrimePlanned);
        self.inner.prime_feedback(iteration_id, slot_order)
    }

    /// Compares prime commit updates with the grammar-derived ledger.
    pub fn record_prime_commit(&mut self, updates: &IterationUpdates) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::PrimePlanned);
        self.inner.record_prime_commit(updates);
        self.phase = GeneralMixedOperationV2OraclePhase::PrimeSettled;
    }

    /// Binds one opaque public request ID for final-prefill B[index].
    pub fn bind_final_prefill(&mut self, index: usize, request_id: RequestId) {
        assert!(
            matches!(
                self.phase,
                GeneralMixedOperationV2OraclePhase::PrimeSettled
                    | GeneralMixedOperationV2OraclePhase::BindingFinalPrefills
            ),
            "seed {:#018x}: V2 final-prefill binding occurred outside the prime-to-mixed transition",
            self.seed
        );
        self.inner.bind_final_prefill(index, request_id);
        self.bound_final_prefills = self
            .bound_final_prefills
            .checked_add(1)
            .expect("bounded V2 final-prefill binding count");
        self.phase = if self.bound_final_prefills == self.final_prefill_count {
            GeneralMixedOperationV2OraclePhase::FinalPrefillsBound
        } else {
            GeneralMixedOperationV2OraclePhase::BindingFinalPrefills
        };
    }

    /// Records the independently validated public mixed plan.
    pub fn observe_mixed_plan(&mut self) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::FinalPrefillsBound);
        self.inner.observe_mixed_plan();
        self.phase = GeneralMixedOperationV2OraclePhase::MixedPlanned;
    }

    /// Records V2's optional deferred decoder cancellation.
    pub fn defer_decoder_cancel(&mut self, index: usize) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
        self.inner.defer_decoder_cancel(index);
    }

    /// Builds valid V2 mixed feedback from grammar slots only.
    #[must_use]
    pub fn mixed_feedback(&self, iteration_id: IterationId, slot_order: &[u8]) -> IterationResult {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
        self.inner.mixed_feedback(iteration_id, slot_order)
    }

    /// Builds one descriptor-selected rejected mixed feedback without changing
    /// the pending oracle state.
    #[must_use]
    pub fn rejected_mixed_feedback(
        &self,
        iteration_id: IterationId,
        slot_order: &[u8],
        rejected_feedback: GeneralMixedOperationRejectedFeedback,
    ) -> IterationResult {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
        let slot_count = self
            .decoder_count
            .checked_add(self.final_prefill_count)
            .expect("bounded V2 mixed slot count");
        validate_slot_order(slot_order, slot_count, "mixed_slot_order").unwrap_or_else(|error| {
            panic!(
                "seed {:#018x}: V2 rejected mixed slot order is invalid: {error}",
                self.seed
            )
        });
        let mut outputs = v2_mixed_outputs(slot_order, self.decoder_count);
        let result_iteration_id = match rejected_feedback {
            GeneralMixedOperationRejectedFeedback::Stale => IterationId::new(
                iteration_id
                    .get()
                    .checked_add(1)
                    .expect("bounded V2 stale iteration ID"),
            )
            .expect("bounded V2 stale iteration ID is nonzero"),
            GeneralMixedOperationRejectedFeedback::Missing
            | GeneralMixedOperationRejectedFeedback::Unplanned => iteration_id,
        };
        match rejected_feedback {
            GeneralMixedOperationRejectedFeedback::Stale => {}
            GeneralMixedOperationRejectedFeedback::Missing => {
                outputs.pop();
            }
            GeneralMixedOperationRejectedFeedback::Unplanned => {
                let replaced_slot = *slot_order
                    .first()
                    .expect("bounded V2 mixed feedback has at least two outputs");
                outputs[0] = IterationOutput::new(
                    OutputSlot::new(
                        u32::try_from(slot_count).expect("bounded V2 unplanned slot fits u32"),
                    ),
                    v2_mixed_token(usize::from(replaced_slot), self.decoder_count),
                    false,
                );
            }
        }
        IterationResult::new(result_iteration_id, outputs, 0, 0)
            .expect("general mixed V2 rejected feedback retains unique transport slots")
    }

    /// Asserts that an attempted rejection left the pure oracle's pending plan
    /// unadvanced; the adapter separately verifies the production surface.
    pub fn assert_rejection_preserves_pending(&self) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
    }

    /// Compares valid mixed commit updates with the grammar-derived ledger.
    pub fn record_mixed_commit(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
        self.inner.record_mixed_commit(updates, now_ns);
        self.phase = GeneralMixedOperationV2OraclePhase::Settled;
    }

    /// Compares a not-dispatched rollback with the grammar-derived ledger.
    pub fn record_not_dispatched_abort(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::MixedPlanned);
        self.inner.record_not_dispatched_abort(updates, now_ns);
        self.phase = GeneralMixedOperationV2OraclePhase::Settled;
    }

    /// Compares consuming close with every still-live grammar request.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::Settled);
        self.inner.record_close(closed, now_ns);
        self.phase = GeneralMixedOperationV2OraclePhase::Closed;
    }

    /// Asserts that the public adapter reached terminal close.
    pub fn assert_closed(&self) {
        self.assert_phase(GeneralMixedOperationV2OraclePhase::Closed);
        self.inner.assert_closed();
    }

    fn assert_phase(&self, expected: GeneralMixedOperationV2OraclePhase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: general mixed V2 oracle phase drifted",
            self.seed
        );
    }
}

fn v2_mixed_outputs(slot_order: &[u8], decoder_count: usize) -> Vec<IterationOutput> {
    slot_order
        .iter()
        .copied()
        .map(|slot| {
            IterationOutput::new(
                OutputSlot::new(u32::from(slot)),
                v2_mixed_token(usize::from(slot), decoder_count),
                false,
            )
        })
        .collect()
}

fn v2_mixed_token(slot: usize, decoder_count: usize) -> u32 {
    if slot < decoder_count {
        decoder_symbolic_token(slot, 1)
    } else {
        final_prefill_symbolic_token(slot - decoder_count, 0)
    }
}

fn v2_descriptor_document(descriptor: &GeneralMixedOperationTraceDescriptorV2) -> String {
    let mut document =
        serde_json::to_string(descriptor).expect("general mixed V2 descriptor serializes");
    document.push('\n');
    document
}

fn strict_round_trip_v2_trace(
    trace: &GeneralMixedOperationTraceV2,
    case_id: &str,
) -> GeneralMixedOperationTraceV2 {
    let document = serialize_general_mixed_operation_v2_trace_descriptor(case_id, trace);
    let parsed = parse_general_mixed_operation_v2_trace_descriptor(&document)
        .expect("general mixed V2 descriptor stays strict-canonical");
    assert_eq!(parsed.case_id, case_id);
    parsed.trace
}

fn push_unique_v2_candidate(
    candidates: &mut Vec<GeneralMixedOperationTraceV2>,
    candidate: GeneralMixedOperationTraceV2,
) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

#[cfg(test)]
mod general_mixed_operation_v2_tests {
    use super::{
        GeneralMixedOperationRejectedFeedback, GeneralMixedOperationSettlement,
        GeneralMixedOperationTrace, GeneralMixedOperationTraceV2,
        general_mixed_operation_v2_corpus, minimize_general_mixed_operation_v2_trace,
        parse_general_mixed_operation_trace_descriptor,
        parse_general_mixed_operation_v2_trace_descriptor,
        serialize_general_mixed_operation_trace_descriptor,
        serialize_general_mixed_operation_v2_trace_descriptor,
    };

    fn fixture() -> GeneralMixedOperationTraceV2 {
        GeneralMixedOperationTraceV2 {
            seed: 0x67d1_4b2a_c982_ef30,
            decoder_count: 3,
            final_prefill_count: 2,
            prime_slot_order: vec![2, 0, 1],
            mixed_slot_order: vec![4, 1, 3, 0, 2],
            cancel_decoder_index: Some(2),
            rejected_feedback: Some(GeneralMixedOperationRejectedFeedback::Missing),
            settlement: GeneralMixedOperationSettlement::AbortNotDispatched,
        }
    }

    #[test]
    fn v2_codec_is_strict_and_isolated_from_v1() {
        let trace = fixture();
        let document =
            serialize_general_mixed_operation_v2_trace_descriptor("v2-codec-round-trip", &trace);
        let parsed = parse_general_mixed_operation_v2_trace_descriptor(&document)
            .expect("canonical V2 descriptor parses");
        assert_eq!(parsed.case_id, "v2-codec-round-trip");
        assert_eq!(parsed.trace, trace);
        assert_eq!(
            serialize_general_mixed_operation_v2_trace_descriptor(&parsed.case_id, &parsed.trace),
            document
        );
        assert!(
            trace
                .describe_operations()
                .contains("reject-feedback(missing)"),
            "V2 operation spelling must retain the rejected-feedback selector"
        );
        assert!(
            parse_general_mixed_operation_trace_descriptor(&document).is_err(),
            "V1 parser must reject the V2 trace kind"
        );

        let v1 = GeneralMixedOperationTrace {
            seed: trace.seed,
            decoder_count: trace.decoder_count,
            final_prefill_count: trace.final_prefill_count,
            prime_slot_order: trace.prime_slot_order.clone(),
            mixed_slot_order: trace.mixed_slot_order.clone(),
            cancel_decoder_index: trace.cancel_decoder_index,
            settlement: trace.settlement,
        };
        let v1_document = serialize_general_mixed_operation_trace_descriptor("v1-retag", &v1);
        assert!(
            parse_general_mixed_operation_v2_trace_descriptor(&v1_document).is_err(),
            "V2 parser must reject the V1 trace kind"
        );
        let v1_retagged_as_v2 = v1_document.replacen(
            "\"trace_kind\":\"general-mixed-operation-v1\"",
            "\"trace_kind\":\"general-mixed-operation-v2\"",
            1,
        );
        assert!(
            parse_general_mixed_operation_v2_trace_descriptor(&v1_retagged_as_v2).is_err(),
            "V2 parser must require nullable rejected_feedback even after retagging V1"
        );
        for invalid in [
            document.replacen("\"rejected_feedback\":\"missing\",", "", 1),
            document.replacen(
                "\"rejected_feedback\":\"missing\"",
                "\"rejected_feedback\":\"unknown\"",
                1,
            ),
            document.replacen(
                "\"trace_kind\":\"general-mixed-operation-v2\"",
                "\"trace_kind\":\"general-mixed-operation-v1\"",
                1,
            ),
            format!(" {document}"),
        ] {
            assert!(
                parse_general_mixed_operation_v2_trace_descriptor(&invalid).is_err(),
                "noncanonical or invalid V2 descriptor unexpectedly parsed: {invalid}"
            );
        }
    }

    #[test]
    fn v2_seeded_generator_and_corpus_stay_strict_canonical() {
        for seed in 0..64_u64 {
            let trace = GeneralMixedOperationTraceV2::from_seed(seed);
            let document =
                serialize_general_mixed_operation_v2_trace_descriptor("v2-seeded", &trace);
            let parsed = parse_general_mixed_operation_v2_trace_descriptor(&document)
                .expect("seeded V2 descriptor parses");
            assert_eq!(parsed.trace, trace);
        }
        let corpus = general_mixed_operation_v2_corpus();
        assert_eq!(corpus.len(), 3);
        assert_eq!(
            corpus
                .iter()
                .filter_map(|named| named.trace.rejected_feedback)
                .collect::<Vec<_>>(),
            vec![
                GeneralMixedOperationRejectedFeedback::Stale,
                GeneralMixedOperationRejectedFeedback::Missing,
                GeneralMixedOperationRejectedFeedback::Unplanned,
            ]
        );
    }

    #[test]
    fn v2_reducer_removes_rejection_before_v1_reductions_and_reaches_a_local_minimum() {
        let source = fixture();
        let candidates = source.shrink_candidates();
        assert_eq!(
            candidates
                .first()
                .map(|candidate| candidate.rejected_feedback),
            Some(None),
            "V2 reducer must remove its optional rejection first"
        );
        let source_rank = source.shrink_rank();
        for (index, candidate) in candidates.iter().enumerate() {
            assert!(
                !candidates[..index].contains(candidate),
                "V2 reducer emitted a duplicate candidate"
            );
            assert!(candidate.shrink_rank() < source_rank);
            let document =
                serialize_general_mixed_operation_v2_trace_descriptor("v2-candidate", candidate);
            let parsed = parse_general_mixed_operation_v2_trace_descriptor(&document)
                .expect("V2 reducer candidate is strict-canonical");
            assert_eq!(parsed.trace, *candidate);
        }
        let predicate = |trace: &GeneralMixedOperationTraceV2| {
            trace.rejected_feedback == Some(GeneralMixedOperationRejectedFeedback::Missing)
                && trace.decoder_count >= 2
                && trace.final_prefill_count >= 2
                && trace.shrink_rank().4 > 0
        };
        assert!(predicate(&source));
        let minimized = minimize_general_mixed_operation_v2_trace(&source, predicate);
        assert!(predicate(&minimized));
        assert_eq!(
            minimize_general_mixed_operation_v2_trace(&minimized, predicate),
            minimized,
            "V2 minimizer must be idempotent at its local minimum"
        );
        assert!(
            minimized
                .shrink_candidates()
                .iter()
                .all(|candidate| !predicate(candidate)),
            "V2 minimizer must return a deterministic local minimum"
        );
    }
}
