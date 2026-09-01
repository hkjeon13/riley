//! C03-A CPU-only replay for strict variable-length general mixed programs.
//!
//! The support reference model does not inspect `Scheduler` or `IterationPlan`.
//! This adapter validates every public plan projection independently, then
//! gives the model only opaque IDs, public cancellation outcomes, and updates.

#[allow(clippy::too_many_lines)]
#[path = "support/general_mixed_program_trace.rs"]
mod general_mixed_program_trace;
#[path = "support/routing_fuzz_rotation.rs"]
mod routing_fuzz_rotation;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use general_mixed_program_trace::{
    GENERAL_MIXED_PROGRAM_ITERATION_TOKEN_BUDGET, GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS,
    GENERAL_MIXED_PROGRAM_MAX_PREFILL_CHUNK_TOKENS, GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS,
    GENERAL_MIXED_PROGRAM_MAX_SEQUENCE_TOKENS, GeneralMixedProgramExpectedPlan,
    GeneralMixedProgramOperation, GeneralMixedProgramReferenceModel, GeneralMixedProgramTrace,
    NamedGeneralMixedProgramTrace, general_mixed_program_corpus,
    general_mixed_program_prompt_tokens, minimize_general_mixed_program_trace,
    parse_general_mixed_program_descriptor, serialize_general_mixed_program_descriptor,
};
use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, IterationId, IterationPlan, OverloadPolicy, RequestDescriptor, RequestId,
    Scheduler, SchedulerCloseOutput, SchedulerConfig, SchedulerError, WorkKind,
};

const GENERAL_MIXED_PROGRAM_TRACE_COUNT: u64 = 10_000;

fn general_mixed_program_config() -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 1,
        max_waiting_prompt_tokens: GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS
            * GENERAL_MIXED_PROGRAM_MAX_PROMPT_TOKENS,
        max_active_sequences: GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS,
        max_sequence_tokens: GENERAL_MIXED_PROGRAM_MAX_SEQUENCE_TOKENS,
        iteration_token_budget: GENERAL_MIXED_PROGRAM_ITERATION_TOKEN_BUDGET,
        max_prefill_chunk_tokens: GENERAL_MIXED_PROGRAM_MAX_PREFILL_CHUNK_TOKENS,
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        max_promised_kv_blocks: GENERAL_MIXED_PROGRAM_MAX_LIVE_REQUESTS,
        metrics_window_samples: 8,
    }
}

fn new_scheduler() -> Scheduler {
    Scheduler::new(
        general_mixed_program_config(),
        KvLayout::checked(1, 64, 1, 8).expect("valid general mixed symbolic KV layout"),
    )
    .expect("valid general mixed scheduler configuration")
}

fn assert_plan_projection(plan: &IterationPlan, expected: &GeneralMixedProgramExpectedPlan) {
    assert_eq!(plan.decode_items().len(), expected.decode_items.len());
    assert_eq!(plan.prefill_items().len(), expected.prefill_items.len());
    for (actual, expected) in plan.decode_items().iter().zip(&expected.decode_items) {
        assert_eq!(actual.kind(), WorkKind::Decode);
        assert_eq!(actual.request_id(), expected.request_id);
        assert_eq!(actual.input_tokens(), expected.input_tokens.as_slice());
        assert_eq!(actual.output_slot(), Some(expected.output_slot));
    }
    for (actual, expected) in plan.prefill_items().iter().zip(&expected.prefill_items) {
        assert_eq!(actual.kind(), WorkKind::Prefill);
        assert_eq!(actual.request_id(), expected.request_id);
        assert_eq!(actual.input_tokens(), expected.input_tokens.as_slice());
        assert_eq!(actual.output_slot(), Some(expected.output_slot));
    }
    let output_slots = expected.output_slots();
    assert_eq!(plan.output_slots(), output_slots.as_slice());
    assert_eq!(plan.total_tokens(), expected.total_tokens());
    assert_eq!(plan.batch_size(), expected.batch_size());
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

struct GeneralMixedProgramAdapter {
    scheduler: Option<Scheduler>,
    model: GeneralMixedProgramReferenceModel,
    request_ids: BTreeMap<u8, RequestId>,
    pending_plan: Option<IterationPlan>,
    last_aborted_iteration_id: Option<IterationId>,
    now_ns: u64,
    closed: bool,
}

impl GeneralMixedProgramAdapter {
    fn new(seed: u64) -> Self {
        Self {
            scheduler: Some(new_scheduler()),
            model: GeneralMixedProgramReferenceModel::new(seed),
            request_ids: BTreeMap::new(),
            pending_plan: None,
            last_aborted_iteration_id: None,
            now_ns: 0,
            closed: false,
        }
    }

    fn apply(&mut self, operation: &GeneralMixedProgramOperation) {
        match operation {
            GeneralMixedProgramOperation::Submit {
                label,
                prompt_len,
                max_new_tokens,
            } => self.submit(*label, *prompt_len, *max_new_tokens),
            GeneralMixedProgramOperation::Plan => self.plan(),
            GeneralMixedProgramOperation::Cancel { label } => self.cancel(*label),
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order,
            } => self.complete(feedback_slot_order),
            GeneralMixedProgramOperation::AbortNotDispatched => self.abort_not_dispatched(),
            GeneralMixedProgramOperation::Close => self.close(),
        }
        self.now_ns = self
            .now_ns
            .checked_add(1)
            .expect("general mixed operation clock");
    }

    fn finish(self) {
        assert!(self.closed, "general mixed program must consume close");
        assert!(self.scheduler.is_none());
        assert!(self.pending_plan.is_none());
        self.model.assert_closed();
    }

    fn submit(&mut self, label: u8, prompt_len: u8, max_new_tokens: u8) {
        let submission = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live before close")
            .submit(
                RequestDescriptor::new(
                    general_mixed_program_prompt_tokens(label, prompt_len),
                    usize::from(max_new_tokens),
                ),
                self.now_ns,
            )
            .expect("general mixed bounded submission succeeds");
        assert!(
            self.request_ids
                .insert(label, submission.request_id())
                .is_none(),
            "general mixed program submitted one label twice"
        );
        self.model
            .bind_submit(label, prompt_len, max_new_tokens, submission.request_id());
    }

    fn plan(&mut self) {
        let expected = self.model.begin_plan();
        let planning = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live before close")
            .plan_iteration(self.now_ns)
            .expect("general mixed plan succeeds");
        let (plan, completions) = planning.into_parts();
        assert!(
            completions.is_empty(),
            "general mixed config does not permit waiting completions"
        );
        let plan = plan.expect("general mixed live work produces a plan");
        assert_plan_projection(&plan, &expected);
        if let Some(aborted_iteration_id) = self.last_aborted_iteration_id.take() {
            assert_ne!(
                plan.iteration_id(),
                aborted_iteration_id,
                "general mixed retry must allocate a new iteration ID after NotDispatched"
            );
        }
        assert_eq!(
            self.scheduler
                .as_ref()
                .expect("general mixed scheduler remains live")
                .inflight_iteration_id(),
            Some(plan.iteration_id())
        );
        let concurrent = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live")
            .plan_iteration(self.now_ns);
        assert!(
            matches!(
                concurrent,
                Err(SchedulerError::IterationInFlight { iteration_id }) if iteration_id == plan.iteration_id()
            ),
            "general mixed grammar allowed a second plan before settlement"
        );
        assert!(
            self.pending_plan.replace(plan).is_none(),
            "general mixed adapter replaced an outstanding plan"
        );
    }

    fn cancel(&mut self, label: u8) {
        let request_id = *self
            .request_ids
            .get(&label)
            .expect("general mixed cancel label is submitted");
        let outcome = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live before close")
            .cancel(request_id, self.now_ns)
            .expect("general mixed cancellation succeeds");
        if self.pending_plan.is_some() {
            let snapshot = self
                .scheduler
                .as_ref()
                .expect("general mixed scheduler remains live")
                .request_snapshot(request_id)
                .expect("deferred cancellation retains request snapshot");
            assert!(snapshot.cancellation_deferred());
            self.model.defer_cancel(label, &outcome);
        } else {
            self.model
                .record_settled_cancel(label, &outcome, self.now_ns);
        }
    }

    fn complete(&mut self, feedback_slot_order: &[u8]) {
        let plan = self
            .pending_plan
            .take()
            .expect("general mixed complete has one pending plan");
        let result = self
            .model
            .feedback(plan.iteration_id(), feedback_slot_order);
        let updates = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live before close")
            .complete_iteration(&result, self.now_ns)
            .expect("general mixed valid feedback commits");
        assert_eq!(
            self.scheduler
                .as_ref()
                .expect("general mixed scheduler remains live")
                .inflight_iteration_id(),
            None
        );
        self.model.record_complete(&updates, self.now_ns);
    }

    fn abort_not_dispatched(&mut self) {
        let plan = self
            .pending_plan
            .take()
            .expect("general mixed abort has one pending plan");
        let aborted_iteration_id = plan.iteration_id();
        let updates = self
            .scheduler
            .as_mut()
            .expect("general mixed scheduler remains live before close")
            .abort_iteration(
                aborted_iteration_id,
                ExecutionAbort::NotDispatched,
                self.now_ns,
            )
            .expect("general mixed not-dispatched abort succeeds");
        assert_eq!(
            self.scheduler
                .as_ref()
                .expect("general mixed scheduler remains live")
                .inflight_iteration_id(),
            None
        );
        self.model
            .record_not_dispatched_abort(&updates, self.now_ns);
        assert!(
            self.last_aborted_iteration_id
                .replace(aborted_iteration_id)
                .is_none(),
            "general mixed grammar must retry or close before another abort"
        );
    }

    fn close(&mut self) {
        assert!(self.pending_plan.is_none());
        let closed = self
            .scheduler
            .take()
            .expect("general mixed scheduler closes exactly once")
            .close(self.now_ns, None)
            .unwrap_or_else(|failure| {
                panic!(
                    "general mixed close failed at {:#018x}: {}",
                    self.now_ns,
                    failure.error()
                )
            });
        self.model.record_close(&closed, self.now_ns);
        assert_closed_quiescent(&closed);
        self.closed = true;
    }
}

fn replay_general_mixed_program_inner(trace: &GeneralMixedProgramTrace) {
    trace
        .validate()
        .expect("general mixed replay receives a valid V1 descriptor");
    let mut adapter = GeneralMixedProgramAdapter::new(trace.seed);
    for operation in &trace.operations {
        adapter.apply(operation);
    }
    adapter.finish();
}

fn general_mixed_program_fails(trace: &GeneralMixedProgramTrace) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        replay_general_mixed_program_inner(trace);
    }))
    .is_err()
}

fn general_mixed_program_failure_report(
    source: &GeneralMixedProgramTrace,
    minimized: &GeneralMixedProgramTrace,
) -> String {
    let source_descriptor = serialize_general_mixed_program_descriptor("failing-original", source);
    let minimized_descriptor =
        serialize_general_mixed_program_descriptor("failing-minimized", minimized);
    format!(
        "C03-A general-mixed-program-v1 failed\n\
         original_descriptor_json:\n\
         {source_descriptor}\
         minimized_descriptor_json:\n\
         {minimized_descriptor}\
         original_operations=[{}]\n\
         minimized_operations=[{}]\n\
         reducer_scope=v1-semantic-local\n\
         failure_predicate=inner-replayer-panicked-only\n\
         not_established=panic-site,payload,failure-signature,root-cause,general-or-global-minimum,plan-complete-abort-deletion,label-renumber,delta-debugging,invalid-feedback,queue-aging,partial-prefill,timeout,device-quiesced-abort,receipt,gpu,c02-qualification",
        source.describe_operations(),
        minimized.describe_operations(),
    )
}

fn replay_general_mixed_program(trace: &GeneralMixedProgramTrace) {
    if !general_mixed_program_fails(trace) {
        return;
    }
    let minimized = minimize_general_mixed_program_trace(trace, general_mixed_program_fails);
    panic!(
        "{}",
        general_mixed_program_failure_report(trace, &minimized)
    );
}

fn replay_named_general_mixed_program(named: &NamedGeneralMixedProgramTrace) {
    replay_general_mixed_program(&named.trace);
}

fn four_slot_mixed_trace(feedback_slot_order: Vec<u8>) -> GeneralMixedProgramTrace {
    GeneralMixedProgramTrace {
        seed: 0x2d8c_60be_5af1_9734,
        operations: vec![
            GeneralMixedProgramOperation::Submit {
                label: 1,
                prompt_len: 3,
                max_new_tokens: 3,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            GeneralMixedProgramOperation::Submit {
                label: 2,
                prompt_len: 1,
                max_new_tokens: 3,
            },
            GeneralMixedProgramOperation::Submit {
                label: 3,
                prompt_len: 2,
                max_new_tokens: 2,
            },
            GeneralMixedProgramOperation::Submit {
                label: 4,
                prompt_len: 3,
                max_new_tokens: 1,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order,
            },
            GeneralMixedProgramOperation::Close,
        ],
    }
}

fn semantic_reducer_fixture() -> GeneralMixedProgramTrace {
    GeneralMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            GeneralMixedProgramOperation::Submit {
                label: 1,
                prompt_len: 2,
                max_new_tokens: 3,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            GeneralMixedProgramOperation::Submit {
                label: 2,
                prompt_len: 1,
                max_new_tokens: 3,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![1, 0],
            },
            GeneralMixedProgramOperation::Cancel { label: 2 },
            GeneralMixedProgramOperation::Submit {
                label: 3,
                prompt_len: 3,
                max_new_tokens: 2,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![1, 0],
            },
            GeneralMixedProgramOperation::Close,
        ],
    }
}

fn semantic_cancel_removed_trace() -> GeneralMixedProgramTrace {
    let mut trace = semantic_reducer_fixture();
    trace.operations.remove(6);
    let GeneralMixedProgramOperation::Complete {
        feedback_slot_order,
    } = &mut trace.operations[8]
    else {
        unreachable!("third complete remains present after cancel removal");
    };
    *feedback_slot_order = vec![2, 0, 1];
    trace
}

fn semantic_label_two_removed_trace() -> GeneralMixedProgramTrace {
    GeneralMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            GeneralMixedProgramOperation::Submit {
                label: 1,
                prompt_len: 2,
                max_new_tokens: 3,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            GeneralMixedProgramOperation::Submit {
                label: 3,
                prompt_len: 3,
                max_new_tokens: 2,
            },
            GeneralMixedProgramOperation::Plan,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order: vec![1, 0],
            },
            GeneralMixedProgramOperation::Close,
        ],
    }
}

fn slot_permutations(slot_count: usize) -> Vec<Vec<u8>> {
    let mut values = (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("four-slot permutation fits u8"))
        .collect::<Vec<_>>();
    let mut permutations = Vec::new();
    collect_slot_permutations(&mut values, 0, &mut permutations);
    permutations
}

fn collect_slot_permutations(values: &mut [u8], start: usize, permutations: &mut Vec<Vec<u8>>) {
    if start == values.len() {
        permutations.push(values.to_vec());
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        collect_slot_permutations(values, start + 1, permutations);
        values.swap(start, index);
    }
}

fn descriptor_document_with_operations(operations: &str) -> String {
    format!(
        "{{\"format\":\"riley.scheduler.general-mixed-program\",\"format_version\":1,\"trace_kind\":\"general-mixed-program-v1\",\"case_id\":\"codec-strict\",\"source_seed\":\"0x2d8c60be5af19734\",\"operations\":[{operations}]}}\n"
    )
}

fn assert_invalid_document(case: &str, document: &str) {
    assert!(
        parse_general_mixed_program_descriptor(document).is_err(),
        "{case}: invalid general mixed descriptor was accepted: {document:?}"
    );
}

#[test]
fn general_mixed_program_corpus_is_canonical_and_replays() {
    for named in general_mixed_program_corpus() {
        let document = serialize_general_mixed_program_descriptor(&named.case_id, &named.trace);
        let parsed = parse_general_mixed_program_descriptor(&document)
            .expect("general mixed corpus stays strict-canonical");
        assert_eq!(parsed, named);
        replay_named_general_mixed_program(&parsed);
    }
}

#[test]
fn general_mixed_program_codec_round_trips_variable_length_programs() {
    let trace = four_slot_mixed_trace(vec![3, 1, 2, 0]);
    let document = serialize_general_mixed_program_descriptor("codec-round-trip", &trace);
    let parsed = parse_general_mixed_program_descriptor(&document)
        .expect("canonical general mixed descriptor parses");
    assert_eq!(parsed.case_id, "codec-round-trip");
    assert_eq!(parsed.trace, trace);
    assert_eq!(
        serialize_general_mixed_program_descriptor(&parsed.case_id, &parsed.trace),
        document
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn general_mixed_program_codec_rejects_noncanonical_and_invalid_documents() {
    let valid = serialize_general_mixed_program_descriptor(
        "codec-strict",
        &four_slot_mixed_trace(vec![3, 1, 2, 0]),
    );
    let too_many_operations =
        [r#"{"op":"submit","label":1,"prompt_len":1,"max_new_tokens":1}"#; 24].join(",");
    for (case, document) in [
        (
            "unsupported format",
            valid.replacen(
                "\"format\":\"riley.scheduler.general-mixed-program\"",
                "\"format\":\"riley.scheduler.unknown\"",
                1,
            ),
        ),
        (
            "unsupported version",
            valid.replacen("\"format_version\":1", "\"format_version\":2", 1),
        ),
        (
            "unsupported trace kind",
            valid.replacen(
                "\"trace_kind\":\"general-mixed-program-v1\"",
                "\"trace_kind\":\"general-mixed-operation-v1\"",
                1,
            ),
        ),
        (
            "uppercase source seed",
            valid.replacen(
                "\"source_seed\":\"0x2d8c60be5af19734\"",
                "\"source_seed\":\"0X2D8C60BE5AF19734\"",
                1,
            ),
        ),
        (
            "invalid case id",
            valid.replacen("\"case_id\":\"codec-strict\"", "\"case_id\":\"Codec\"", 1),
        ),
        (
            "duplicate outer field",
            valid.replacen(
                "{\"format\":\"riley.scheduler.general-mixed-program\",",
                "{\"format\":\"riley.scheduler.general-mixed-program\",\"format\":\"riley.scheduler.general-mixed-program\",",
                1,
            ),
        ),
        (
            "missing outer field",
            valid.replacen("\"source_seed\":\"0x2d8c60be5af19734\",", "", 1),
        ),
        (
            "unknown outer field",
            valid.replacen("\"operations\":[", "\"unexpected\":true,\"operations\":[", 1),
        ),
        (
            "duplicate nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"prompt_len\":3,\"max_new_tokens\":3}",
                "{\"op\":\"submit\",\"label\":1,\"label\":1,\"prompt_len\":3,\"max_new_tokens\":3}",
                1,
            ),
        ),
        (
            "unknown nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"prompt_len\":3,\"max_new_tokens\":3}",
                "{\"op\":\"submit\",\"label\":1,\"prompt_len\":3,\"max_new_tokens\":3,\"unexpected\":true}",
                1,
            ),
        ),
        (
            "missing nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"prompt_len\":3,\"max_new_tokens\":3}",
                "{\"op\":\"submit\",\"label\":1,\"prompt_len\":3}",
                1,
            ),
        ),
        ("out of range label", valid.replacen("\"label\":4", "\"label\":7", 1)),
        (
            "out of range prompt",
            valid.replacen("\"prompt_len\":3", "\"prompt_len\":4", 1),
        ),
        (
            "out of range capacity",
            valid.replacen("\"max_new_tokens\":3", "\"max_new_tokens\":4", 1),
        ),
        (
            "duplicate feedback slot",
            valid.replacen(
                "\"feedback_slot_order\":[3,1,2,0]",
                "\"feedback_slot_order\":[3,1,1,0]",
                1,
            ),
        ),
        (
            "short feedback order",
            valid.replacen(
                "\"feedback_slot_order\":[3,1,2,0]",
                "\"feedback_slot_order\":[0,1,2]",
                1,
            ),
        ),
        (
            "submit while pending",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"prompt_len":1,"max_new_tokens":2},{"op":"plan"},{"op":"submit","label":2,"prompt_len":1,"max_new_tokens":1},{"op":"close"}"#,
            ),
        ),
        (
            "complete without plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"prompt_len":1,"max_new_tokens":2},{"op":"complete","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        (
            "abort without plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"prompt_len":1,"max_new_tokens":2},{"op":"abort_not_dispatched"},{"op":"close"}"#,
            ),
        ),
        (
            "close while pending",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"prompt_len":1,"max_new_tokens":2},{"op":"plan"},{"op":"close"}"#,
            ),
        ),
        (
            "operation cap",
            descriptor_document_with_operations(&format!("{too_many_operations},{{\"op\":\"close\"}}")),
        ),
        ("noncanonical whitespace", format!(" {valid}")),
    ] {
        assert_invalid_document(case, &document);
    }
}

#[test]
fn general_mixed_program_replays_every_four_slot_feedback_permutation() {
    for feedback_slot_order in slot_permutations(4) {
        replay_general_mixed_program(&four_slot_mixed_trace(feedback_slot_order));
    }
}

#[test]
fn general_mixed_program_reducer_rebases_semantic_cancel_and_submission_removals() {
    let source = semantic_reducer_fixture();
    source
        .validate()
        .expect("semantic reducer fixture remains valid");
    let cancel_removed = semantic_cancel_removed_trace();
    let label_two_removed = semantic_label_two_removed_trace();
    let candidates = source.shrink_candidates();
    assert_eq!(candidates.first(), Some(&cancel_removed));
    assert!(
        candidates.contains(&label_two_removed),
        "semantic submit removal must project later feedback slots"
    );
    for candidate in [cancel_removed, label_two_removed] {
        let document = serialize_general_mixed_program_descriptor("semantic-rebase", &candidate);
        let parsed = parse_general_mixed_program_descriptor(&document)
            .expect("semantic reducer candidate remains strict-canonical");
        assert_eq!(parsed.trace, candidate);
        replay_general_mixed_program_inner(&candidate);
    }
}

#[test]
fn general_mixed_program_reducer_candidates_are_ranked_canonical_and_replayable() {
    let mut sources = general_mixed_program_corpus()
        .into_iter()
        .map(|named| named.trace)
        .collect::<Vec<_>>();
    sources.push(semantic_reducer_fixture());
    sources.extend((1_u64..=1_024).map(|index| {
        GeneralMixedProgramTrace::from_seed(0x8ea6_2c14_d7b9_305f_u64.wrapping_mul(index))
    }));
    for source in sources {
        let source_rank = source.shrink_rank();
        let candidates = source.shrink_candidates();
        for (index, candidate) in candidates.iter().enumerate() {
            assert!(
                !candidates[..index].contains(candidate),
                "general mixed reducer emitted a duplicate candidate"
            );
            assert_eq!(candidate.seed, source.seed);
            assert!(candidate.shrink_rank() < source_rank);
            let document =
                serialize_general_mixed_program_descriptor("candidate-canonical", candidate);
            let parsed = parse_general_mixed_program_descriptor(&document)
                .expect("semantic candidate remains strict-canonical");
            assert_eq!(parsed.trace, *candidate);
            replay_general_mixed_program_inner(candidate);
        }
    }
}

fn semantic_reducer_predicate(trace: &GeneralMixedProgramTrace) -> bool {
    trace
        .operations
        .iter()
        .filter(|operation| matches!(operation, GeneralMixedProgramOperation::Plan))
        .count()
        >= 3
        && trace
            .operations
            .iter()
            .any(|operation| matches!(operation, GeneralMixedProgramOperation::Cancel { label: 2 }))
}

fn contains_feedback_order(trace: &GeneralMixedProgramTrace, expected: &[u8]) -> bool {
    trace.operations.iter().any(|operation| {
        matches!(
            operation,
            GeneralMixedProgramOperation::Complete {
                feedback_slot_order
            } if feedback_slot_order.as_slice() == expected
        )
    })
}

#[test]
fn general_mixed_program_reducer_keeps_its_semantic_candidate_order() {
    let source = semantic_reducer_fixture();
    let candidates = source.shrink_candidates();
    assert_eq!(candidates.first(), Some(&semantic_cancel_removed_trace()));
    let first_nonremoval = candidates
        .iter()
        .position(|candidate| candidate.operations.len() == source.operations.len())
        .expect("semantic reducer must expose contraction or feedback candidates");
    assert!(
        candidates[..first_nonremoval].contains(&semantic_label_two_removed_trace()),
        "submission removal must stay after cancellation removal and before non-removal edits"
    );

    let rebased = semantic_cancel_removed_trace();
    let rebased_candidates = rebased.shrink_candidates();
    let identity_index = rebased_candidates
        .iter()
        .position(|candidate| contains_feedback_order(candidate, &[0, 1, 2]))
        .expect("semantic reducer must offer a three-slot identity order");
    let adjacent_inversion_index = rebased_candidates
        .iter()
        .position(|candidate| {
            contains_feedback_order(candidate, &[0, 2, 1])
                || contains_feedback_order(candidate, &[2, 1, 0])
        })
        .expect("semantic reducer must offer a one-inversion three-slot order");
    assert!(
        identity_index < adjacent_inversion_index,
        "identity normalization must precede adjacent inversion reduction"
    );
}

fn report_descriptor(report: &str, start: &str, end: &str) -> String {
    report
        .split_once(start)
        .expect("general mixed report contains descriptor start")
        .1
        .split_once(end)
        .expect("general mixed report contains descriptor end")
        .0
        .to_owned()
}

#[test]
fn general_mixed_program_reducer_finds_a_semantic_local_minimum_and_binds_reports() {
    let source = semantic_reducer_fixture();
    assert!(semantic_reducer_predicate(&source));
    let minimized = minimize_general_mixed_program_trace(&source, semantic_reducer_predicate);
    assert!(semantic_reducer_predicate(&minimized));
    replay_general_mixed_program_inner(&minimized);
    assert_eq!(
        minimize_general_mixed_program_trace(&minimized, semantic_reducer_predicate),
        minimized,
        "semantic minimizer must be idempotent at its deterministic local minimum"
    );
    assert!(
        minimized
            .shrink_candidates()
            .iter()
            .all(|candidate| !semantic_reducer_predicate(candidate)),
        "semantic minimizer must return a local minimum for its fixed candidate order"
    );
    let report = general_mixed_program_failure_report(&source, &minimized);
    let source_document = report_descriptor(
        &report,
        "original_descriptor_json:\n",
        "minimized_descriptor_json:\n",
    );
    let minimized_document = report_descriptor(
        &report,
        "minimized_descriptor_json:\n",
        "original_operations=[",
    );
    assert_eq!(
        parse_general_mixed_program_descriptor(&source_document)
            .expect("failure report source descriptor is strict-canonical")
            .trace,
        source
    );
    assert_eq!(
        parse_general_mixed_program_descriptor(&minimized_document)
            .expect("failure report minimized descriptor is strict-canonical")
            .trace,
        minimized
    );
    assert!(report.contains("reducer_scope=v1-semantic-local"));
    assert!(report.contains("failure_predicate=inner-replayer-panicked-only"));
}

#[test]
fn ten_thousand_seeded_general_mixed_program_traces_round_trip_and_replay() {
    let rotation = routing_fuzz_rotation::configured_seed_rotation();
    for local_trace_index in 0..GENERAL_MIXED_PROGRAM_TRACE_COUNT {
        let trace_index = rotation
            .trace_index(local_trace_index, GENERAL_MIXED_PROGRAM_TRACE_COUNT)
            .expect("configured routing-fuzz rotation fits general mixed window");
        let seed = 0xa5f3_19c7_e04d_6b82_u64.wrapping_mul(trace_index.wrapping_add(1));
        let trace = GeneralMixedProgramTrace::from_seed(seed);
        let case_id = format!("seed-{seed:016x}");
        let document = serialize_general_mixed_program_descriptor(&case_id, &trace);
        let parsed = parse_general_mixed_program_descriptor(&document)
            .expect("seeded general mixed descriptor parses");
        assert_eq!(parsed.trace, trace);
        replay_named_general_mixed_program(&parsed);
    }
}
