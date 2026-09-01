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

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use bounded_mixed_program_trace::{
    BoundedMixedProgramExpectedPlan, BoundedMixedProgramOperation, BoundedMixedProgramOracle,
    BoundedMixedProgramTrace, bounded_mixed_program_corpus, parse_bounded_mixed_program_descriptor,
    serialize_bounded_mixed_program_descriptor, symbolic_prompt_token,
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

fn bounded_mixed_program_failure_report(trace: &BoundedMixedProgramTrace) -> String {
    let descriptor = serialize_bounded_mixed_program_descriptor("failing-original", trace);
    format!(
        "C03-A bounded-mixed-program-v1 failed\n\
         source_descriptor_json:\n\
         {descriptor}\
         source_operations=[{}]\n\
         scope=bounded-valid-settled-boundary-program-only\n\
         not_established=unbounded-or-general-scheduler,plan-complete-split,inflight-cancel,abort-retry,invalid-feedback,partial-prefill,queue-aging,fault-injection,general-reducer,receipt,gpu,c02-qualification",
        trace.describe_operations(),
    )
}

fn replay_bounded_mixed_program(trace: &BoundedMixedProgramTrace) {
    if !bounded_mixed_program_fails(trace) {
        return;
    }
    panic!("{}", bounded_mixed_program_failure_report(trace));
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
