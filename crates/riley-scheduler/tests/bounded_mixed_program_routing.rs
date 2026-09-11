//! C03-A CPU-only replay for bounded settled-boundary mixed-operation programs.
//!
//! The raw descriptor contains explicit submits, settled cancellations, plan
//! commits with semantic output permutations, and close. The support oracle
//! never reads `Scheduler` or `IterationPlan`; this adapter validates that public
//! projection before it asks the oracle to construct synthetic feedback.

#[path = "support/bounded_mixed_program_trace.rs"]
mod bounded_mixed_program_trace;
#[path = "support/routing_fuzz_rotation.rs"]
mod routing_fuzz_rotation;

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use bounded_mixed_program_trace::{
    BoundedMixedProgramExpectedPlan, BoundedMixedProgramOperation, BoundedMixedProgramOracle,
    BoundedMixedProgramTrace, bounded_mixed_program_corpus, minimize_bounded_mixed_program_trace,
    parse_bounded_mixed_program_descriptor, serialize_bounded_mixed_program_descriptor,
    symbolic_prompt_token,
};
use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationPlan, OverloadPolicy, RequestDescriptor, RequestId, Scheduler, SchedulerCloseOutput,
    SchedulerConfig, WorkKind,
};

const BOUNDED_MIXED_PROGRAM_TRACE_COUNT: u64 = 10_000;

fn bounded_mixed_program_config() -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 3,
        max_waiting_prompt_tokens: 3,
        max_active_sequences: 3,
        max_sequence_tokens: 3,
        iteration_token_budget: 3,
        max_prefill_chunk_tokens: 1,
        aging_threshold_ns: 1_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 3,
        metrics_window_samples: 8,
    }
}

fn new_scheduler() -> Scheduler {
    let layout = KvLayout::checked(1, 64, 1, 8).expect("valid C03-A symbolic KV layout");
    Scheduler::new(bounded_mixed_program_config(), layout)
        .expect("valid C03-A bounded mixed program scheduler configuration")
}

fn assert_plan_projection(plan: &IterationPlan, expected: &BoundedMixedProgramExpectedPlan) {
    assert_eq!(plan.decode_items().len(), expected.decode_items.len());
    assert_eq!(plan.prefill_items().len(), expected.prefill_items.len());
    for (actual, expected) in plan.decode_items().iter().zip(&expected.decode_items) {
        assert_eq!(actual.kind(), WorkKind::Decode);
        assert_eq!(actual.request_id(), expected.request_id);
        assert_eq!(actual.input_tokens(), &[expected.input_token]);
        assert_eq!(actual.output_slot(), Some(expected.output_slot));
    }
    for (actual, expected) in plan.prefill_items().iter().zip(&expected.prefill_items) {
        assert_eq!(actual.kind(), WorkKind::Prefill);
        assert_eq!(actual.request_id(), expected.request_id);
        assert_eq!(actual.input_tokens(), &[expected.input_token]);
        assert_eq!(actual.output_slot(), Some(expected.output_slot));
    }
    let output_slots = expected.output_slots();
    assert_eq!(plan.output_slots(), output_slots.as_slice());
    assert_eq!(plan.total_tokens(), expected.total_tokens());
    assert_eq!(plan.batch_size(), expected.total_tokens());
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

fn replay_bounded_mixed_program_inner(trace: &BoundedMixedProgramTrace) {
    trace
        .validate()
        .expect("bounded mixed program replay receives a valid descriptor");
    let mut scheduler = Some(new_scheduler());
    let mut oracle = BoundedMixedProgramOracle::new(trace.seed);
    let mut request_ids = BTreeMap::<u8, RequestId>::new();
    let mut now_ns = 0_u64;
    let mut close_seen = false;

    for operation in &trace.operations {
        match operation {
            BoundedMixedProgramOperation::Submit {
                label,
                max_new_tokens,
            } => {
                let submission = scheduler
                    .as_mut()
                    .expect("bounded mixed program scheduler remains live before close")
                    .submit(
                        RequestDescriptor::new(
                            vec![symbolic_prompt_token(*label)],
                            usize::from(*max_new_tokens),
                        ),
                        now_ns,
                    )
                    .expect("bounded mixed program submission");
                assert!(
                    request_ids
                        .insert(*label, submission.request_id())
                        .is_none(),
                    "bounded mixed program submitted one label twice"
                );
                oracle.bind_submit(*label, *max_new_tokens, submission.request_id());
            }
            BoundedMixedProgramOperation::Cancel { label } => {
                let request_id = *request_ids
                    .get(label)
                    .expect("bounded mixed program cancel label is submitted");
                let outcome = scheduler
                    .as_mut()
                    .expect("bounded mixed program scheduler remains live before close")
                    .cancel(request_id, now_ns)
                    .expect("settled-boundary cancellation succeeds");
                oracle.record_settled_cancel(*label, &outcome, now_ns);
            }
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            } => {
                let expected = oracle.begin_plan();
                let planning = scheduler
                    .as_mut()
                    .expect("bounded mixed program scheduler remains live before close")
                    .plan_iteration(now_ns)
                    .expect("bounded mixed program plan succeeds");
                let (plan, completions) = planning.into_parts();
                assert!(
                    completions.is_empty(),
                    "bounded mixed program does not enable admission timeouts"
                );
                let plan = plan.expect("bounded mixed program has live work to plan");
                assert_plan_projection(&plan, &expected);
                let result = oracle.feedback(plan.iteration_id(), feedback_slot_order);
                let updates = scheduler
                    .as_mut()
                    .expect("bounded mixed program scheduler remains live before close")
                    .complete_iteration(&result, now_ns)
                    .expect("bounded mixed program commit succeeds");
                oracle.record_plan_commit(&updates, now_ns);
            }
            BoundedMixedProgramOperation::Close => {
                let closed = scheduler
                    .take()
                    .expect("bounded mixed program closes its scheduler exactly once")
                    .close(now_ns, None)
                    .unwrap_or_else(|failure| {
                        panic!(
                            "seed {:#018x}: bounded mixed program close failed: {}",
                            trace.seed,
                            failure.error()
                        )
                    });
                oracle.record_close(&closed, now_ns);
                assert_closed_quiescent(&closed);
                close_seen = true;
            }
        }
        now_ns = now_ns
            .checked_add(1)
            .expect("bounded mixed program operation clock");
    }
    assert!(close_seen);
    assert!(scheduler.is_none());
    oracle.assert_closed();
}

fn bounded_mixed_program_fails(trace: &BoundedMixedProgramTrace) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        replay_bounded_mixed_program_inner(trace);
    }))
    .is_err()
}

fn bounded_mixed_program_failure_report(
    source: &BoundedMixedProgramTrace,
    minimized: &BoundedMixedProgramTrace,
) -> String {
    let source_descriptor = serialize_bounded_mixed_program_descriptor("failing-original", source);
    let minimized_descriptor =
        serialize_bounded_mixed_program_descriptor("failing-minimized", minimized);
    format!(
        "C03-A bounded-mixed-program-v1 failed\n\
         original_descriptor_json:\n\
         {source_descriptor}\
         minimized_descriptor_json:\n\
         {minimized_descriptor}\
         original_operations=[{}]\n\
         minimized_operations=[{}]\n\
         reducer_scope=v1-stateful-operation-local\n\
         failure_predicate=inner-replayer-panicked-only\n\
         not_established=panic-site,payload,failure-signature,root-cause,general-or-global-minimum,unbounded-or-general-scheduler,plan-complete-split,inflight-cancel,abort-retry,invalid-feedback,partial-prefill,queue-aging,fault-injection,receipt,gpu,c02-qualification",
        source.describe_operations(),
        minimized.describe_operations(),
    )
}

fn replay_bounded_mixed_program(trace: &BoundedMixedProgramTrace) {
    if !bounded_mixed_program_fails(trace) {
        return;
    }
    let minimized = minimize_bounded_mixed_program_trace(trace, bounded_mixed_program_fails);
    panic!(
        "{}",
        bounded_mixed_program_failure_report(trace, &minimized)
    );
}

fn three_slot_mixed_trace(feedback_slot_order: Vec<u8>) -> BoundedMixedProgramTrace {
    BoundedMixedProgramTrace {
        seed: 0x4c3d_2e1f_0a9b_8c7d,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order,
            },
            BoundedMixedProgramOperation::Close,
        ],
    }
}

fn stateful_reduction_trace() -> BoundedMixedProgramTrace {
    BoundedMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![2, 0, 1],
            },
            BoundedMixedProgramOperation::Cancel { label: 2 },
            BoundedMixedProgramOperation::Submit {
                label: 4,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Close,
        ],
    }
}

fn stateful_cancel_removal_trace() -> BoundedMixedProgramTrace {
    BoundedMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![2, 0, 1],
            },
            BoundedMixedProgramOperation::Submit {
                label: 4,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![1, 0],
            },
            BoundedMixedProgramOperation::Close,
        ],
    }
}

fn stateful_label_two_removal_trace() -> BoundedMixedProgramTrace {
    BoundedMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![1, 0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 4,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Close,
        ],
    }
}

fn stateful_label_three_removal_trace() -> BoundedMixedProgramTrace {
    BoundedMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0, 1],
            },
            BoundedMixedProgramOperation::Cancel { label: 2 },
            BoundedMixedProgramOperation::Submit {
                label: 4,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Close,
        ],
    }
}

fn stateful_reducer_predicate(trace: &BoundedMixedProgramTrace) -> bool {
    let submitted_labels = trace
        .operations
        .iter()
        .filter_map(|operation| match operation {
            BoundedMixedProgramOperation::Submit { label, .. } => Some(*label),
            BoundedMixedProgramOperation::Cancel { .. }
            | BoundedMixedProgramOperation::PlanCommit { .. }
            | BoundedMixedProgramOperation::Close => None,
        })
        .collect::<BTreeSet<_>>();
    submitted_labels.iter().copied().eq([1, 2, 3, 4])
        && trace
            .operations
            .iter()
            .filter(|operation| {
                matches!(operation, BoundedMixedProgramOperation::PlanCommit { .. })
            })
            .count()
            == 3
        && trace.operations.iter().any(|operation| {
            matches!(
                operation,
                BoundedMixedProgramOperation::PlanCommit {
                    feedback_slot_order
                } if feedback_slot_order == &[2, 0, 1]
            )
        })
}

fn report_descriptor(report: &str, start: &str, end: &str) -> String {
    report
        .split_once(start)
        .expect("bounded reducer report contains the descriptor start")
        .1
        .split_once(end)
        .expect("bounded reducer report contains the descriptor end")
        .0
        .to_owned()
}

fn descriptor_document_with_operations(operations: &str) -> String {
    format!(
        "{{\"format\":\"riley.scheduler.bounded-mixed-program\",\"format_version\":1,\"trace_kind\":\"bounded-mixed-program-v1\",\"case_id\":\"codec-strict\",\"source_seed\":\"0x4c3d2e1f0a9b8c7d\",\"operations\":[{operations}]}}\n"
    )
}

#[test]
fn bounded_mixed_program_corpus_is_canonical_and_replays() {
    for named in bounded_mixed_program_corpus() {
        let document = serialize_bounded_mixed_program_descriptor(&named.case_id, &named.trace);
        let parsed = parse_bounded_mixed_program_descriptor(&document)
            .expect("bounded mixed program corpus stays strict-canonical");
        assert_eq!(parsed, named);
        replay_bounded_mixed_program(&parsed.trace);
    }
}

#[test]
fn bounded_mixed_program_codec_round_trips_a_raw_program() {
    let trace = three_slot_mixed_trace(vec![2, 1, 0]);
    let document = serialize_bounded_mixed_program_descriptor("codec-round-trip", &trace);
    let parsed = parse_bounded_mixed_program_descriptor(&document)
        .expect("canonical bounded mixed program descriptor parses");
    assert_eq!(parsed.case_id, "codec-round-trip");
    assert_eq!(parsed.trace, trace);
    assert_eq!(
        serialize_bounded_mixed_program_descriptor(&parsed.case_id, &parsed.trace),
        document
    );
}

// Keeping this invalid-document matrix together makes the strict codec
// boundary auditable without spreading its accepted grammar across helpers.
#[allow(clippy::too_many_lines)]
#[test]
fn bounded_mixed_program_codec_rejects_noncanonical_and_invalid_documents() {
    let valid = serialize_bounded_mixed_program_descriptor(
        "codec-strict",
        &three_slot_mixed_trace(vec![2, 1, 0]),
    );
    let over_operation_cap = [r#"{"op":"submit","label":1,"max_new_tokens":1}"#; 12].join(",");
    let invalid_documents = [
        (
            "unsupported format",
            valid.replacen(
                "\"format\":\"riley.scheduler.bounded-mixed-program\"",
                "\"format\":\"riley.scheduler.unknown\"",
                1,
            ),
        ),
        (
            "unsupported trace kind",
            valid.replacen(
                "\"trace_kind\":\"bounded-mixed-program-v1\"",
                "\"trace_kind\":\"bounded-mixed-program-v2\"",
                1,
            ),
        ),
        (
            "unsupported version",
            valid.replacen("\"format_version\":1", "\"format_version\":2", 1),
        ),
        (
            "uppercase source seed",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\"",
                "\"source_seed\":\"0X4c3d2e1f0a9b8c7d\"",
                1,
            ),
        ),
        (
            "source seed without prefix",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\"",
                "\"source_seed\":\"4c3d2e1f0a9b8c7d\"",
                1,
            ),
        ),
        (
            "short source seed",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\"",
                "\"source_seed\":\"0x4c3d\"",
                1,
            ),
        ),
        (
            "nonhex source seed",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\"",
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7g\"",
                1,
            ),
        ),
        (
            "invalid case identifier",
            valid.replacen("\"case_id\":\"codec-strict\"", "\"case_id\":\"Codec-Strict\"", 1),
        ),
        (
            "duplicate outer field",
            valid.replacen(
                "{\"format\":\"riley.scheduler.bounded-mixed-program\",",
                "{\"format\":\"riley.scheduler.bounded-mixed-program\",\"format\":\"riley.scheduler.bounded-mixed-program\",",
                1,
            ),
        ),
        (
            "reordered outer field",
            valid.replacen(
                "\"format\":\"riley.scheduler.bounded-mixed-program\",\"format_version\":1",
                "\"format_version\":1,\"format\":\"riley.scheduler.bounded-mixed-program\"",
                1,
            ),
        ),
        (
            "missing outer field",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\",",
                "",
                1,
            ),
        ),
        (
            "unknown outer field",
            valid.replacen(
                "\"operations\":[",
                "\"unexpected\":true,\"operations\":[",
                1,
            ),
        ),
        (
            "duplicate nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2}",
                "{\"op\":\"submit\",\"label\":1,\"label\":1,\"max_new_tokens\":2}",
                1,
            ),
        ),
        (
            "unknown nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2}",
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2,\"unexpected\":true}",
                1,
            ),
        ),
        (
            "missing nested field",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2}",
                "{\"op\":\"submit\",\"label\":1}",
                1,
            ),
        ),
        (
            "duplicate logical label",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"plan_commit\"",
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"plan_commit\"",
                1,
            ),
        ),
        (
            "out-of-range logical label",
            valid.replacen("\"label\":1", "\"label\":5", 1),
        ),
        (
            "out-of-range feedback slot",
            valid.replacen(
                "\"feedback_slot_order\":[2,1,0]",
                "\"feedback_slot_order\":[3,1,0]",
                1,
            ),
        ),
        (
            "short feedback slot order",
            valid.replacen(
                "\"feedback_slot_order\":[2,1,0]",
                "\"feedback_slot_order\":[0,1]",
                1,
            ),
        ),
        (
            "duplicate feedback slot",
            valid.replacen(
                "\"feedback_slot_order\":[2,1,0]",
                "\"feedback_slot_order\":[0,0,1]",
                1,
            ),
        ),
        (
            "output capacity bound",
            valid.replacen("\"max_new_tokens\":2", "\"max_new_tokens\":3", 1),
        ),
        (
            "live request cap",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"submit","label":2,"max_new_tokens":2},{"op":"submit","label":3,"max_new_tokens":2},{"op":"submit","label":4,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]}"#,
            ),
        ),
        (
            "operation cap",
            descriptor_document_with_operations(&format!(
                "{over_operation_cap},{{\"op\":\"close\"}}"
            )),
        ),
        (
            "plan commit cap",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"submit","label":2,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0,1]},{"op":"submit","label":3,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0,1]},{"op":"submit","label":4,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0,1]},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        (
            "settled cancellation cap",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"submit","label":2,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0,1]},{"op":"cancel","label":2},{"op":"submit","label":3,"max_new_tokens":2},{"op":"cancel","label":3},{"op":"close"}"#,
            ),
        ),
        (
            "plan without live work",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":1},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        (
            "missing mixed plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        (
            "missing final close",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0]},{"op":"submit","label":2,"max_new_tokens":2},{"op":"plan_commit","feedback_slot_order":[0,1]}"#,
            ),
        ),
        (
            "post-close operation",
            valid.replacen(
                "{\"op\":\"close\"}",
                "{\"op\":\"close\"},{\"op\":\"submit\",\"label\":4,\"max_new_tokens\":1}",
                1,
            ),
        ),
        ("noncanonical whitespace", format!(" {valid}")),
    ];
    for (case, document) in invalid_documents {
        assert!(
            parse_bounded_mixed_program_descriptor(&document).is_err(),
            "{case}: invalid bounded mixed program descriptor was accepted: {document:?}"
        );
    }
}

#[test]
fn bounded_mixed_program_replays_every_three_slot_feedback_permutation() {
    for feedback_slot_order in [
        vec![0, 1, 2],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![2, 1, 0],
    ] {
        replay_bounded_mixed_program(&three_slot_mixed_trace(feedback_slot_order));
    }
}

#[test]
fn bounded_mixed_program_reducer_rebases_stateful_removals() {
    let source = stateful_reduction_trace();
    source
        .validate()
        .expect("stateful bounded reducer fixture stays valid");
    let cancel_removed = stateful_cancel_removal_trace();
    let label_two_removed = stateful_label_two_removal_trace();
    let label_three_removed = stateful_label_three_removal_trace();
    let candidates = source.shrink_candidates();

    assert_eq!(candidates.first(), Some(&cancel_removed));
    assert!(
        candidates.contains(&label_two_removed),
        "label-two removal must rebase later feedback slots"
    );
    assert!(
        candidates.contains(&label_three_removed),
        "label-three removal must project later feedback slots"
    );
    for candidate in [cancel_removed, label_two_removed, label_three_removed] {
        let document = serialize_bounded_mixed_program_descriptor("stateful-rebase", &candidate);
        let parsed = parse_bounded_mixed_program_descriptor(&document)
            .expect("stateful bounded reducer candidate stays strict-canonical");
        assert_eq!(parsed.trace, candidate);
        replay_bounded_mixed_program_inner(&candidate);
    }
}

#[test]
fn bounded_mixed_program_reducer_candidates_are_deduped_ranked_and_canonical() {
    let mut sources = bounded_mixed_program_corpus()
        .into_iter()
        .map(|named| named.trace)
        .collect::<Vec<_>>();
    sources.push(stateful_reduction_trace());
    sources.extend((1_u64..=1_024).map(|index| {
        BoundedMixedProgramTrace::from_seed(0xe703_7ed1_a0b4_285d_u64.wrapping_mul(index))
    }));

    for source in sources {
        let source_rank = source.shrink_rank();
        let candidates = source.shrink_candidates();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            assert!(
                !candidates[..candidate_index].contains(candidate),
                "bounded reducer emitted a duplicate candidate"
            );
            assert_eq!(candidate.seed, source.seed);
            assert!(candidate.shrink_rank() < source_rank);
            let document =
                serialize_bounded_mixed_program_descriptor("candidate-canonical", candidate);
            let parsed = parse_bounded_mixed_program_descriptor(&document)
                .expect("bounded reducer candidate stays strict-canonical");
            assert_eq!(parsed.trace, *candidate);
            replay_bounded_mixed_program_inner(candidate);
        }
    }
}

#[test]
fn bounded_mixed_program_reducer_finds_a_stateful_local_minimum() {
    let source = stateful_reduction_trace();
    assert!(stateful_reducer_predicate(&source));
    let minimized = minimize_bounded_mixed_program_trace(&source, stateful_reducer_predicate);
    let expected = BoundedMixedProgramTrace {
        seed: source.seed,
        operations: vec![
            BoundedMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0],
            },
            BoundedMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            BoundedMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![2, 0, 1],
            },
            BoundedMixedProgramOperation::Submit {
                label: 4,
                max_new_tokens: 1,
            },
            BoundedMixedProgramOperation::PlanCommit {
                feedback_slot_order: vec![0, 1],
            },
            BoundedMixedProgramOperation::Close,
        ],
    };
    assert_eq!(minimized, expected);
    assert_eq!(minimized.seed, source.seed);
    assert!(minimized.shrink_rank() < source.shrink_rank());
    assert!(stateful_reducer_predicate(&minimized));
    let document = serialize_bounded_mixed_program_descriptor("stateful-minimized", &minimized);
    let parsed = parse_bounded_mixed_program_descriptor(&document)
        .expect("stateful bounded local minimum stays strict-canonical");
    assert_eq!(parsed.trace, minimized);
    replay_bounded_mixed_program_inner(&source);
    replay_bounded_mixed_program_inner(&minimized);
    assert_eq!(
        minimize_bounded_mixed_program_trace(&minimized, stateful_reducer_predicate),
        minimized
    );
    assert!(
        minimized
            .shrink_candidates()
            .iter()
            .all(|candidate| !stateful_reducer_predicate(candidate)),
        "bounded reducer result must be a local minimum for its fixed candidate order"
    );
}

#[test]
fn bounded_mixed_program_failure_report_preserves_source_and_local_minimum() {
    let source = stateful_reduction_trace();
    let minimized = minimize_bounded_mixed_program_trace(&source, stateful_reducer_predicate);
    let report = bounded_mixed_program_failure_report(&source, &minimized);
    assert!(report.contains("reducer_scope=v1-stateful-operation-local"));
    assert!(report.contains("failure_predicate=inner-replayer-panicked-only"));
    assert!(report.contains("not_established=panic-site,payload,failure-signature,root-cause"));
    assert!(report.contains("original_operations=["));
    assert!(report.contains("minimized_operations=["));

    let original_document = report_descriptor(
        &report,
        "original_descriptor_json:\n",
        "minimized_descriptor_json:\n",
    );
    let minimized_document = report_descriptor(
        &report,
        "minimized_descriptor_json:\n",
        "original_operations=[",
    );
    let parsed_original = parse_bounded_mixed_program_descriptor(&original_document)
        .expect("failure report source descriptor stays strict-canonical");
    let parsed_minimized = parse_bounded_mixed_program_descriptor(&minimized_document)
        .expect("failure report minimized descriptor stays strict-canonical");
    assert_eq!(parsed_original.case_id, "failing-original");
    assert_eq!(parsed_original.trace, source);
    assert_eq!(parsed_minimized.case_id, "failing-minimized");
    assert_eq!(parsed_minimized.trace, minimized);
}

#[test]
fn ten_thousand_seeded_bounded_mixed_programs_round_trip_and_replay() {
    let mut saw_two_prefill_first_commit = false;
    let mut saw_three_prefill_first_commit = false;
    let rotation = routing_fuzz_rotation::configured_seed_rotation();
    for local_trace_index in 0..BOUNDED_MIXED_PROGRAM_TRACE_COUNT {
        let trace_index = rotation
            .trace_index(local_trace_index, BOUNDED_MIXED_PROGRAM_TRACE_COUNT)
            .expect("configured routing-fuzz rotation fits the bounded mixed trace window");
        let seed = 0xe703_7ed1_a0b4_285d_u64.wrapping_mul(trace_index.wrapping_add(1));
        let trace = BoundedMixedProgramTrace::from_seed(seed);
        let first_plan_index = trace
            .operations
            .iter()
            .position(|operation| {
                matches!(operation, BoundedMixedProgramOperation::PlanCommit { .. })
            })
            .expect("seeded bounded mixed program contains a plan commit");
        let plan_slot_counts = trace
            .operations
            .iter()
            .filter_map(|operation| match operation {
                BoundedMixedProgramOperation::PlanCommit {
                    feedback_slot_order,
                } => Some(feedback_slot_order.len()),
                BoundedMixedProgramOperation::Submit { .. }
                | BoundedMixedProgramOperation::Cancel { .. }
                | BoundedMixedProgramOperation::Close => None,
            })
            .collect::<Vec<_>>();
        assert!(
            plan_slot_counts.len() >= 2,
            "seeded bounded mixed program contains two required plan commits"
        );
        match seed & 0b11 {
            2 => {
                assert_eq!(first_plan_index, 2);
                assert_eq!(plan_slot_counts[..2], [2, 3]);
                saw_two_prefill_first_commit = true;
            }
            3 => {
                assert_eq!(first_plan_index, 3);
                assert_eq!(plan_slot_counts[..2], [3, 3]);
                saw_three_prefill_first_commit = true;
            }
            _ => {
                assert_eq!(first_plan_index, 1);
                assert_eq!(plan_slot_counts[0], 1);
            }
        }
        let document = serialize_bounded_mixed_program_descriptor("seeded-program", &trace);
        let parsed = parse_bounded_mixed_program_descriptor(&document)
            .expect("seeded bounded mixed program stays strict-canonical");
        assert_eq!(parsed.trace, trace);
        replay_bounded_mixed_program(&parsed.trace);
    }
    assert!(
        saw_two_prefill_first_commit,
        "seeded bounded mixed program generator did not exercise two-prefill-first commits"
    );
    assert!(
        saw_three_prefill_first_commit,
        "seeded bounded mixed program generator did not exercise three-prefill-first commits"
    );
}
