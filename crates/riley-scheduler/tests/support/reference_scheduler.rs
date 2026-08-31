//! Narrow, pure routing/lifecycle oracle for the bounded C03-A V2 grammar.
//!
//! This module owns no scheduler and never derives expected slot routing from
//! a production iteration plan. The integration-test adapter binds opaque
//! public request IDs, checks the public plan separately, asks this oracle to
//! build feedback, and compares public updates with its ledger.

use std::collections::BTreeMap;

use riley_scheduler::{
    IterationId, IterationOutput, IterationResult, IterationUpdates, OutputSlot, RequestCompletion,
    RequestFinishReason, RequestId, SchedulerCloseOutput, TokenEvent,
};

const DECODER_LABEL: u32 = 1;
const FINAL_PREFILL_LABEL: u32 = 2;

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
    AwaitingDecoder,
    DecoderBound,
    PrimePlanned,
    PrimeSettled,
    FinalPrefillBound,
    MixedPlanned,
    Settled,
    Closed,
}

/// Pure oracle for the fixed two-request mixed-operation V2 grammar.
///
/// The grammar fixes decoder A to slot zero and final-prefill B to slot one.
/// The adapter must validate that the public plan agrees before handing the
/// public iteration ID here. Feedback, token projections, terminal ledgers,
/// cancellation behavior, and close behavior are then derived only from that
/// grammar and opaque request-ID bindings.
pub struct V2RoutingOracle {
    seed: u64,
    decoder_max_new_tokens: usize,
    decoder: Option<RequestId>,
    final_prefill: Option<RequestId>,
    decoder_history: Vec<u32>,
    final_prefill_history: Vec<u32>,
    cancellation_deferred: bool,
    terminal: BTreeMap<RequestId, CompletionProjection>,
    phase: Phase,
}

impl V2RoutingOracle {
    /// Creates a bounded V2 routing oracle before any request IDs are known.
    #[must_use]
    pub fn new(seed: u64, decoder_max_new_tokens: usize, final_prefill_len: usize) -> Self {
        assert!(
            (2..=4).contains(&decoder_max_new_tokens),
            "seed {seed:#018x}: V2 decoder capacity escaped its bounded grammar"
        );
        assert!(
            (1..=4).contains(&final_prefill_len),
            "seed {seed:#018x}: V2 final-prefill length escaped its bounded grammar"
        );
        Self {
            seed,
            decoder_max_new_tokens,
            decoder: None,
            final_prefill: None,
            decoder_history: Vec::with_capacity(decoder_max_new_tokens),
            final_prefill_history: Vec::with_capacity(1),
            cancellation_deferred: false,
            terminal: BTreeMap::new(),
            phase: Phase::AwaitingDecoder,
        }
    }

    /// Binds the public ID returned for the fixed decoder A request.
    pub fn bind_decoder(&mut self, request_id: RequestId) {
        self.assert_phase(Phase::AwaitingDecoder);
        assert!(
            self.decoder.replace(request_id).is_none(),
            "seed {:#018x}: V2 decoder was bound twice",
            self.seed
        );
        self.phase = Phase::DecoderBound;
    }

    /// Records the public prime plan after the adapter has validated it.
    pub fn observe_prime_plan(&mut self) {
        self.assert_phase(Phase::DecoderBound);
        self.phase = Phase::PrimePlanned;
    }

    /// Builds the only valid prime feedback from the grammar, not the plan.
    #[must_use]
    pub fn prime_feedback(&self, iteration_id: IterationId) -> IterationResult {
        self.assert_phase(Phase::PrimePlanned);
        Self::result(
            iteration_id,
            vec![IterationOutput::new(
                OutputSlot::new(0),
                token_for(DECODER_LABEL, 0),
                false,
            )],
        )
    }

    /// Compares the prime commit with the grammar-derived routing ledger.
    pub fn record_prime_commit(&mut self, updates: &IterationUpdates) {
        self.assert_phase(Phase::PrimePlanned);
        let decoder = self.decoder();
        let expected_tokens = BTreeMap::from([(
            decoder,
            TokenProjection {
                token_id: token_for(DECODER_LABEL, 0),
                generated_index: 0,
            },
        )]);
        self.assert_updates(updates, &expected_tokens, BTreeMap::new());
        self.decoder_history.push(token_for(DECODER_LABEL, 0));
        self.phase = Phase::PrimeSettled;
    }

    /// Binds the public ID returned for the fixed final-prefill B request.
    pub fn bind_final_prefill(&mut self, request_id: RequestId) {
        self.assert_phase(Phase::PrimeSettled);
        assert!(
            self.final_prefill.replace(request_id).is_none(),
            "seed {:#018x}: V2 final-prefill request was bound twice",
            self.seed
        );
        self.phase = Phase::FinalPrefillBound;
    }

    /// Records the public mixed plan after the adapter has validated it.
    pub fn observe_mixed_plan(&mut self) {
        self.assert_phase(Phase::FinalPrefillBound);
        self.phase = Phase::MixedPlanned;
    }

    /// Records the only deferred cancellation allowed by the V2 grammar.
    pub fn defer_decoder_cancel(&mut self) {
        self.assert_phase(Phase::MixedPlanned);
        assert!(
            !self.cancellation_deferred,
            "seed {:#018x}: V2 decoder cancellation was deferred twice",
            self.seed
        );
        self.cancellation_deferred = true;
    }

    /// Builds reversed valid mixed feedback from the grammar, not the plan.
    #[must_use]
    pub fn mixed_reverse_feedback(&self, iteration_id: IterationId) -> IterationResult {
        self.assert_phase(Phase::MixedPlanned);
        Self::result(iteration_id, Self::mixed_outputs())
    }

    /// Builds stale mixed feedback without consulting the production plan map.
    #[must_use]
    pub fn stale_feedback(&self, iteration_id: IterationId) -> IterationResult {
        self.assert_phase(Phase::MixedPlanned);
        let stale_id = IterationId::new(
            iteration_id
                .get()
                .checked_add(1)
                .expect("bounded V2 stale iteration ID"),
        )
        .expect("bounded V2 stale iteration ID is nonzero");
        Self::result(stale_id, Self::mixed_outputs())
    }

    /// Builds mixed feedback with a missing output slot.
    #[must_use]
    pub fn missing_feedback(&self, iteration_id: IterationId) -> IterationResult {
        self.assert_phase(Phase::MixedPlanned);
        let mut outputs = Self::mixed_outputs();
        outputs.pop();
        Self::result(iteration_id, outputs)
    }

    /// Builds mixed feedback with an unplanned output slot.
    #[must_use]
    pub fn unplanned_feedback(&self, iteration_id: IterationId) -> IterationResult {
        self.assert_phase(Phase::MixedPlanned);
        let mut outputs = Self::mixed_outputs();
        outputs[0] = IterationOutput::new(
            OutputSlot::new(99),
            token_for(FINAL_PREFILL_LABEL, 0),
            false,
        );
        Self::result(iteration_id, outputs)
    }

    /// Compares a valid reverse mixed commit with the grammar-derived ledger.
    pub fn record_mixed_commit(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(Phase::MixedPlanned);
        let decoder = self.decoder();
        let final_prefill = self.final_prefill();
        let mut expected_tokens = BTreeMap::from([(
            final_prefill,
            TokenProjection {
                token_id: token_for(FINAL_PREFILL_LABEL, 0),
                generated_index: 0,
            },
        )]);
        self.final_prefill_history
            .push(token_for(FINAL_PREFILL_LABEL, 0));
        if !self.cancellation_deferred {
            expected_tokens.insert(
                decoder,
                TokenProjection {
                    token_id: token_for(DECODER_LABEL, 1),
                    generated_index: 1,
                },
            );
            self.decoder_history.push(token_for(DECODER_LABEL, 1));
        }
        let mut expected_completions = BTreeMap::from([(
            final_prefill,
            CompletionProjection {
                reason: RequestFinishReason::Length,
                generated_token_ids: self.final_prefill_history.clone(),
                completed_at_ns: now_ns,
            },
        )]);
        if self.cancellation_deferred {
            expected_completions.insert(
                decoder,
                CompletionProjection {
                    reason: RequestFinishReason::Cancelled,
                    generated_token_ids: self.decoder_history.clone(),
                    completed_at_ns: now_ns,
                },
            );
        } else if self.decoder_max_new_tokens == 2 {
            expected_completions.insert(
                decoder,
                CompletionProjection {
                    reason: RequestFinishReason::Length,
                    generated_token_ids: self.decoder_history.clone(),
                    completed_at_ns: now_ns,
                },
            );
        }
        self.assert_updates(updates, &expected_tokens, expected_completions);
        self.phase = Phase::Settled;
    }

    /// Compares a not-dispatched rollback with the grammar-derived ledger.
    pub fn record_not_dispatched_abort(&mut self, updates: &IterationUpdates, now_ns: u64) {
        self.assert_phase(Phase::MixedPlanned);
        let expected_completions = if self.cancellation_deferred {
            BTreeMap::from([(
                self.decoder(),
                CompletionProjection {
                    reason: RequestFinishReason::Cancelled,
                    generated_token_ids: self.decoder_history.clone(),
                    completed_at_ns: now_ns,
                },
            )])
        } else {
            BTreeMap::new()
        };
        self.assert_updates(updates, &BTreeMap::new(), expected_completions);
        self.phase = Phase::Settled;
    }

    /// Compares consuming close with every still-live grammar request.
    pub fn record_close(&mut self, closed: &SchedulerCloseOutput, now_ns: u64) {
        self.assert_phase(Phase::Settled);
        assert!(
            closed.settlement_failures().is_empty(),
            "seed {:#018x}: V2 close contained a settlement failure",
            self.seed
        );
        let mut expected_completions = BTreeMap::new();
        let decoder = self.decoder();
        if !self.terminal.contains_key(&decoder) {
            expected_completions.insert(
                decoder,
                CompletionProjection {
                    reason: RequestFinishReason::Cancelled,
                    generated_token_ids: self.decoder_history.clone(),
                    completed_at_ns: now_ns,
                },
            );
        }
        let final_prefill = self.final_prefill();
        if !self.terminal.contains_key(&final_prefill) {
            expected_completions.insert(
                final_prefill,
                CompletionProjection {
                    reason: RequestFinishReason::Cancelled,
                    generated_token_ids: self.final_prefill_history.clone(),
                    completed_at_ns: now_ns,
                },
            );
        }
        self.assert_completions(closed.completions(), expected_completions);
        assert_eq!(
            self.terminal.len(),
            2,
            "seed {:#018x}: V2 close did not terminally settle both logical requests",
            self.seed
        );
        self.phase = Phase::Closed;
    }

    /// Asserts that the public adapter has driven the oracle to terminal close.
    pub fn assert_closed(&self) {
        self.assert_phase(Phase::Closed);
        assert_eq!(
            self.terminal.len(),
            2,
            "seed {:#018x}: V2 oracle terminal ledger is incomplete",
            self.seed
        );
    }

    fn result(iteration_id: IterationId, outputs: Vec<IterationOutput>) -> IterationResult {
        IterationResult::new(iteration_id, outputs, 0, 0)
            .expect("bounded V2 oracle feedback has unique output slots")
    }

    fn mixed_outputs() -> Vec<IterationOutput> {
        vec![
            IterationOutput::new(OutputSlot::new(1), token_for(FINAL_PREFILL_LABEL, 0), false),
            IterationOutput::new(OutputSlot::new(0), token_for(DECODER_LABEL, 1), false),
        ]
    }

    fn assert_updates(
        &mut self,
        updates: &IterationUpdates,
        expected_tokens: &BTreeMap<RequestId, TokenProjection>,
        expected_completions: BTreeMap<RequestId, CompletionProjection>,
    ) {
        assert!(
            updates.settlement_failures().is_empty(),
            "seed {:#018x}: V2 update contained a settlement failure",
            self.seed
        );
        assert_eq!(
            &token_projections(updates.token_events()),
            expected_tokens,
            "seed {:#018x}: V2 grammar token routing drifted",
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
            "seed {:#018x}: V2 grammar terminal ledger drifted",
            self.seed
        );
        for (request_id, completion) in expected_completions {
            assert!(
                self.terminal.insert(request_id, completion).is_none(),
                "seed {:#018x}: V2 grammar emitted a duplicate terminal request",
                self.seed
            );
        }
    }

    fn assert_phase(&self, expected: Phase) {
        assert_eq!(
            self.phase, expected,
            "seed {:#018x}: V2 oracle phase drifted",
            self.seed
        );
    }

    fn decoder(&self) -> RequestId {
        self.decoder
            .expect("bounded V2 oracle decoder request ID is bound")
    }

    fn final_prefill(&self) -> RequestId {
        self.final_prefill
            .expect("bounded V2 oracle final-prefill request ID is bound")
    }
}

fn token_for(label: u32, generated_index: usize) -> u32 {
    label
        .checked_mul(16)
        .and_then(|base| base.checked_add(u32::try_from(generated_index).ok()?))
        .expect("bounded V2 symbolic output token")
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
            "V2 public updates emitted duplicate token events for one request"
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
            "V2 public updates emitted duplicate terminal completions for one request"
        );
    }
    projections
}
