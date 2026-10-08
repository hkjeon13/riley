//! C03-A CPU-only replay for bounded in-flight mixed-operation programs.
//!
//! The raw descriptor separates planning from settlement. The support oracle
//! never reads `Scheduler` or `IterationPlan`; this adapter first validates the
//! public plan projection and then supplies only its iteration ID to feedback.

#[path = "support/inflight_mixed_program_trace.rs"]
mod inflight_mixed_program_trace;
#[path = "support/routing_fuzz_rotation.rs"]
mod routing_fuzz_rotation;

use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

use inflight_mixed_program_trace::{
    InflightMixedProgramExpectedPlan, InflightMixedProgramOperation,
    InflightMixedProgramOperationV2, InflightMixedProgramOracle, InflightMixedProgramTrace,
    InflightMixedProgramTraceV2, inflight_mixed_program_corpus, inflight_mixed_program_v2_corpus,
    inflight_symbolic_prompt_token, minimize_inflight_mixed_program_trace,
    minimize_inflight_mixed_program_v2_trace, parse_inflight_mixed_program_descriptor,
    parse_inflight_mixed_program_v2_descriptor, serialize_inflight_mixed_program_descriptor,
    serialize_inflight_mixed_program_v2_descriptor,
};
use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, IterationPlan, OverloadPolicy, RequestDescriptor, RequestId, RequestState,
    Scheduler, SchedulerCloseOutput, SchedulerConfig, SchedulerError, WorkKind,
};

const INFLIGHT_MIXED_PROGRAM_TRACE_COUNT: u64 = 10_000;
const INFLIGHT_MIXED_PROGRAM_V2_TRACE_COUNT: u64 = 10_000;

fn inflight_mixed_program_config() -> SchedulerConfig {
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
    Scheduler::new(inflight_mixed_program_config(), layout)
        .expect("valid C03-A in-flight mixed program scheduler configuration")
}

fn assert_plan_projection(plan: &IterationPlan, expected: &InflightMixedProgramExpectedPlan) {
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

// Keeping the public-API operation adapter together makes it possible to audit
// every raw grammar operation against its scheduler call and oracle transition.
#[allow(clippy::too_many_lines)]
fn replay_inflight_mixed_program_inner(trace: &InflightMixedProgramTrace) {
    trace
        .validate()
        .expect("in-flight mixed program replay receives a valid descriptor");
    let mut scheduler = Some(new_scheduler());
    let mut oracle = InflightMixedProgramOracle::new(trace.seed);
    let mut request_ids = BTreeMap::<u8, RequestId>::new();
    let mut pending_plan = None::<IterationPlan>;
    let mut retry_after_abort = None;
    let mut now_ns = 0_u64;
    let mut close_seen = false;

    for operation in &trace.operations {
        match operation {
            InflightMixedProgramOperation::Submit {
                label,
                max_new_tokens,
            } => {
                let submission = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live before close")
                    .submit(
                        RequestDescriptor::new(
                            vec![inflight_symbolic_prompt_token(*label)],
                            usize::from(*max_new_tokens),
                        ),
                        now_ns,
                    )
                    .expect("in-flight mixed program submission");
                assert!(
                    request_ids
                        .insert(*label, submission.request_id())
                        .is_none(),
                    "in-flight mixed program submitted one label twice"
                );
                oracle.bind_submit(*label, *max_new_tokens, submission.request_id());
            }
            InflightMixedProgramOperation::Plan => {
                let expected = oracle.begin_plan();
                let planning = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live before close")
                    .plan_iteration(now_ns)
                    .expect("in-flight mixed program plan succeeds");
                let (plan, completions) = planning.into_parts();
                assert!(
                    completions.is_empty(),
                    "in-flight mixed program does not enable admission timeouts"
                );
                let plan = plan.expect("in-flight mixed program has live work to plan");
                if let Some(aborted_iteration_id) = retry_after_abort.take() {
                    assert_ne!(
                        plan.iteration_id(),
                        aborted_iteration_id,
                        "in-flight mixed program retry reused its aborted iteration ID"
                    );
                }
                assert_plan_projection(&plan, &expected);
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight program scheduler remains live")
                        .inflight_iteration_id(),
                    Some(plan.iteration_id())
                );
                let concurrent_plan = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live")
                    .plan_iteration(now_ns);
                assert!(
                    matches!(
                        concurrent_plan,
                        Err(SchedulerError::IterationInFlight { iteration_id })
                            if iteration_id == plan.iteration_id()
                    ),
                    "in-flight mixed program allowed a second plan before settlement"
                );
                assert!(
                    pending_plan.replace(plan).is_none(),
                    "in-flight mixed program replaced its outstanding plan"
                );
            }
            InflightMixedProgramOperation::Cancel { label } => {
                let request_id = *request_ids
                    .get(label)
                    .expect("in-flight mixed program cancel label is submitted");
                let outcome = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live before close")
                    .cancel(request_id, now_ns)
                    .expect("in-flight cancellation succeeds");
                let snapshot = scheduler
                    .as_ref()
                    .expect("in-flight program scheduler remains live")
                    .request_snapshot(request_id)
                    .expect("deferred cancellation retains its live request snapshot");
                assert!(snapshot.cancellation_deferred());
                oracle.defer_cancel(*label, &outcome);
            }
            InflightMixedProgramOperation::Complete {
                feedback_slot_order,
            } => {
                let plan = pending_plan
                    .take()
                    .expect("in-flight mixed program has a plan to complete");
                let result = oracle.feedback(plan.iteration_id(), feedback_slot_order);
                let updates = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live before close")
                    .complete_iteration(&result, now_ns)
                    .expect("in-flight mixed program completion succeeds");
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight program scheduler remains live")
                        .inflight_iteration_id(),
                    None
                );
                oracle.record_complete(&updates, now_ns);
            }
            InflightMixedProgramOperation::AbortNotDispatched => {
                let plan = pending_plan
                    .take()
                    .expect("in-flight mixed program has a plan to abort");
                let aborted_iteration_id = plan.iteration_id();
                let updates = scheduler
                    .as_mut()
                    .expect("in-flight program scheduler remains live before close")
                    .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, now_ns)
                    .expect("in-flight mixed program not-dispatched abort succeeds");
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight program scheduler remains live")
                        .inflight_iteration_id(),
                    None
                );
                oracle.record_not_dispatched_abort(&updates, now_ns);
                assert!(
                    retry_after_abort.replace(aborted_iteration_id).is_none(),
                    "in-flight mixed program aborted more than one plan"
                );
            }
            InflightMixedProgramOperation::Close => {
                assert!(pending_plan.is_none());
                let closed = scheduler
                    .take()
                    .expect("in-flight mixed program closes its scheduler exactly once")
                    .close(now_ns, None)
                    .unwrap_or_else(|failure| {
                        panic!(
                            "seed {:#018x}: in-flight mixed program close failed: {}",
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
            .expect("in-flight mixed program operation clock");
    }
    assert!(close_seen);
    assert!(scheduler.is_none());
    assert!(pending_plan.is_none());
    assert!(retry_after_abort.is_none());
    oracle.assert_closed();
}

fn inflight_mixed_program_fails(trace: &InflightMixedProgramTrace) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        replay_inflight_mixed_program_inner(trace);
    }))
    .is_err()
}

fn inflight_mixed_program_failure_report(
    source: &InflightMixedProgramTrace,
    minimized: &InflightMixedProgramTrace,
) -> String {
    let source_descriptor = serialize_inflight_mixed_program_descriptor("failing-original", source);
    let minimized_descriptor =
        serialize_inflight_mixed_program_descriptor("failing-minimized", minimized);
    format!(
        "C03-A inflight-mixed-program-v1 failed\n\
         original_descriptor_json:\n\
         {source_descriptor}\
         minimized_descriptor_json:\n\
         {minimized_descriptor}\
         original_operations=[{}]\n\
         minimized_operations=[{}]\n\
         reducer_scope=v1-raw-pending-lifecycle-local\n\
         failure_predicate=inner-replayer-panicked-only\n\
         not_established=panic-site,payload,failure-signature,root-cause,general-or-global-minimum,label-or-slot-rebase,arbitrary-operation-deletion,unbounded-or-general-scheduler,device-quiesced-abort,pending-close,invalid-feedback,partial-prefill,queue-aging,fault-injection,receipt,gpu,c02-qualification",
        source.describe_operations(),
        minimized.describe_operations(),
    )
}

fn replay_inflight_mixed_program(trace: &InflightMixedProgramTrace) {
    if !inflight_mixed_program_fails(trace) {
        return;
    }
    let minimized = minimize_inflight_mixed_program_trace(trace, inflight_mixed_program_fails);
    panic!(
        "{}",
        inflight_mixed_program_failure_report(trace, &minimized)
    );
}

fn three_slot_mixed_trace(feedback_slot_order: Vec<u8>) -> InflightMixedProgramTrace {
    InflightMixedProgramTrace {
        seed: 0x4c3d_2e1f_0a9b_8c7d,
        operations: vec![
            InflightMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            InflightMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Complete {
                feedback_slot_order,
            },
            InflightMixedProgramOperation::Close,
        ],
    }
}

fn raw_abort_retry_trace() -> InflightMixedProgramTrace {
    InflightMixedProgramTrace {
        seed: 0x7b6a_5948_3726_1504,
        operations: vec![
            InflightMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            InflightMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::AbortNotDispatched,
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Complete {
                feedback_slot_order: vec![2, 1, 0],
            },
            InflightMixedProgramOperation::Close,
        ],
    }
}

fn raw_deferred_complete_trace() -> InflightMixedProgramTrace {
    InflightMixedProgramTrace {
        seed: 0x5a4b_3c2d_1e0f_9876,
        operations: vec![
            InflightMixedProgramOperation::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Complete {
                feedback_slot_order: vec![0],
            },
            InflightMixedProgramOperation::Submit {
                label: 2,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperation::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperation::Plan,
            InflightMixedProgramOperation::Cancel { label: 3 },
            InflightMixedProgramOperation::Complete {
                feedback_slot_order: vec![2, 0, 1],
            },
            InflightMixedProgramOperation::Close,
        ],
    }
}

fn raw_lifecycle_reducer_predicate(trace: &InflightMixedProgramTrace) -> bool {
    trace
        .operations
        .iter()
        .filter(|operation| matches!(operation, InflightMixedProgramOperation::Plan))
        .count()
        >= 2
        && trace.operations.iter().any(|operation| {
            matches!(
                operation,
                InflightMixedProgramOperation::Complete {
                    feedback_slot_order
                } if feedback_slot_order == &[2, 1, 0]
            )
        })
}

fn report_descriptor(report: &str, start: &str, end: &str) -> String {
    report
        .split_once(start)
        .expect("in-flight reducer report contains the descriptor start")
        .1
        .split_once(end)
        .expect("in-flight reducer report contains the descriptor end")
        .0
        .to_owned()
}

fn descriptor_document_with_operations(operations: &str) -> String {
    format!(
        "{{\"format\":\"riley.scheduler.inflight-mixed-program\",\"format_version\":1,\"trace_kind\":\"inflight-mixed-program-v1\",\"case_id\":\"codec-strict\",\"source_seed\":\"0x4c3d2e1f0a9b8c7d\",\"operations\":[{operations}]}}\n"
    )
}

fn v2_descriptor_document_with_operations(operations: &str) -> String {
    format!(
        "{{\"format\":\"riley.scheduler.inflight-mixed-program\",\"format_version\":1,\"trace_kind\":\"inflight-mixed-program-v2\",\"case_id\":\"codec-strict-v2\",\"source_seed\":\"0x4c3d2e1f0a9b8c7d\",\"operations\":[{operations}]}}\n"
    )
}

#[test]
fn inflight_mixed_program_corpus_is_canonical_and_replays() {
    for named in inflight_mixed_program_corpus() {
        let document = serialize_inflight_mixed_program_descriptor(&named.case_id, &named.trace);
        let parsed = parse_inflight_mixed_program_descriptor(&document)
            .expect("in-flight mixed program corpus stays strict-canonical");
        assert_eq!(parsed, named);
        replay_inflight_mixed_program(&parsed.trace);
    }
}

#[test]
fn inflight_mixed_program_codec_round_trips_a_raw_program() {
    let trace = three_slot_mixed_trace(vec![2, 1, 0]);
    let document = serialize_inflight_mixed_program_descriptor("codec-round-trip", &trace);
    let parsed = parse_inflight_mixed_program_descriptor(&document)
        .expect("canonical in-flight mixed program descriptor parses");
    assert_eq!(parsed.case_id, "codec-round-trip");
    assert_eq!(parsed.trace, trace);
    assert_eq!(
        serialize_inflight_mixed_program_descriptor(&parsed.case_id, &parsed.trace),
        document
    );
}

// Keeping this invalid-document matrix together makes the strict codec
// boundary auditable without spreading its accepted grammar across helpers.
#[allow(clippy::too_many_lines)]
#[test]
fn inflight_mixed_program_codec_rejects_noncanonical_and_invalid_documents() {
    let valid = serialize_inflight_mixed_program_descriptor(
        "codec-strict",
        &three_slot_mixed_trace(vec![2, 1, 0]),
    );
    let invalid_documents = [
        (
            "unsupported format",
            valid.replacen(
                "\"format\":\"riley.scheduler.inflight-mixed-program\"",
                "\"format\":\"riley.scheduler.unknown\"",
                1,
            ),
        ),
        (
            "unsupported trace kind",
            valid.replacen(
                "\"trace_kind\":\"inflight-mixed-program-v1\"",
                "\"trace_kind\":\"inflight-mixed-program-v2\"",
                1,
            ),
        ),
        (
            "unsupported version",
            valid.replacen("\"format_version\":1", "\"format_version\":2", 1),
        ),
        (
            "invalid case identifier",
            valid.replacen("\"case_id\":\"codec-strict\"", "\"case_id\":\"Codec-Strict\"", 1),
        ),
        (
            "invalid source seed",
            valid.replacen(
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7d\"",
                "\"source_seed\":\"0x4c3d2e1f0a9b8c7g\"",
                1,
            ),
        ),
        (
            "duplicate outer field",
            valid.replacen(
                "{\"format\":\"riley.scheduler.inflight-mixed-program\",",
                "{\"format\":\"riley.scheduler.inflight-mixed-program\",\"format\":\"riley.scheduler.inflight-mixed-program\",",
                1,
            ),
        ),
        (
            "reordered outer field",
            valid.replacen(
                "\"format\":\"riley.scheduler.inflight-mixed-program\",\"format_version\":1",
                "\"format_version\":1,\"format\":\"riley.scheduler.inflight-mixed-program\"",
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
            "missing nested field",
            valid.replacen(
                "{\"op\":\"complete\",\"feedback_slot_order\":[0]}",
                "{\"op\":\"complete\"}",
                1,
            ),
        ),
        (
            "unknown nested field",
            valid.replacen(
                "{\"op\":\"close\"}",
                "{\"op\":\"close\",\"unexpected\":true}",
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
            "duplicate logical label",
            valid.replacen(
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"plan\"}",
                "{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"submit\",\"label\":1,\"max_new_tokens\":2},{\"op\":\"plan\"}",
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
            "complete without plan",
            descriptor_document_with_operations(
                r#"{"op":"complete","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        (
            "cancel without plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"cancel","label":1},{"op":"close"}"#,
            ),
        ),
        (
            "second plan while pending",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan"},{"op":"plan"},{"op":"close"}"#,
            ),
        ),
        (
            "close while pending",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan"},{"op":"close"}"#,
            ),
        ),
        (
            "abort without immediate retry plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan"},{"op":"abort_not_dispatched"},{"op":"close"}"#,
            ),
        ),
        (
            "missing mixed plan",
            descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan"},{"op":"complete","feedback_slot_order":[0]},{"op":"plan"},{"op":"complete","feedback_slot_order":[0]},{"op":"close"}"#,
            ),
        ),
        ("noncanonical whitespace", format!(" {valid}")),
    ];
    for (case, document) in invalid_documents {
        assert!(
            parse_inflight_mixed_program_descriptor(&document).is_err(),
            "{case}: invalid in-flight mixed program descriptor was accepted: {document:?}"
        );
    }
}

#[test]
fn inflight_mixed_program_replays_every_three_slot_feedback_permutation() {
    for feedback_slot_order in [
        vec![0, 1, 2],
        vec![0, 2, 1],
        vec![1, 0, 2],
        vec![1, 2, 0],
        vec![2, 0, 1],
        vec![2, 1, 0],
    ] {
        replay_inflight_mixed_program(&three_slot_mixed_trace(feedback_slot_order));
    }
}

#[test]
fn inflight_mixed_program_reducer_contracts_only_valid_raw_lifecycles() {
    let abort_source = raw_abort_retry_trace();
    abort_source
        .validate()
        .expect("raw abort-retry reducer fixture stays valid");
    let mut contracted = abort_source.clone();
    contracted.operations.remove(6);
    contracted.operations.remove(6);
    contracted
        .validate()
        .expect("raw abort-retry contraction stays valid");

    let mut reduced_capacity = abort_source.clone();
    let InflightMixedProgramOperation::Submit { max_new_tokens, .. } =
        &mut reduced_capacity.operations[3]
    else {
        panic!("raw abort-retry fixture keeps label-two submission at index three");
    };
    *max_new_tokens = 1;
    reduced_capacity
        .validate()
        .expect("raw output-capacity reduction stays valid");

    let mut identity = abort_source.clone();
    let InflightMixedProgramOperation::Complete {
        feedback_slot_order,
    } = &mut identity.operations[8]
    else {
        panic!("raw abort-retry fixture keeps final complete at index eight");
    };
    *feedback_slot_order = vec![0, 1, 2];
    let mut adjacent = abort_source.clone();
    let InflightMixedProgramOperation::Complete {
        feedback_slot_order,
    } = &mut adjacent.operations[8]
    else {
        panic!("raw abort-retry fixture keeps final complete at index eight");
    };
    *feedback_slot_order = vec![1, 2, 0];

    let abort_candidates = abort_source.shrink_candidates();
    assert_eq!(abort_candidates.first(), Some(&contracted));
    assert!(abort_candidates.contains(&reduced_capacity));
    assert!(abort_candidates.contains(&identity));
    assert!(abort_candidates.contains(&adjacent));

    let deferred_source = raw_deferred_complete_trace();
    deferred_source
        .validate()
        .expect("raw deferred-complete reducer fixture stays valid");
    let mut cancel_removed = deferred_source.clone();
    cancel_removed.operations.remove(6);
    cancel_removed
        .validate()
        .expect("raw deferred cancellation removal stays valid");
    assert_eq!(
        deferred_source.shrink_candidates().first(),
        Some(&cancel_removed)
    );

    for candidate in [
        contracted,
        reduced_capacity,
        identity,
        adjacent,
        cancel_removed,
    ] {
        let document = serialize_inflight_mixed_program_descriptor("raw-lifecycle", &candidate);
        let parsed = parse_inflight_mixed_program_descriptor(&document)
            .expect("raw lifecycle candidate stays strict-canonical");
        assert_eq!(parsed.trace, candidate);
        replay_inflight_mixed_program_inner(&candidate);
    }
}

#[test]
fn inflight_mixed_program_reducer_rejects_raw_lifecycle_mutations_that_need_rebase() {
    let source = inflight_mixed_program_corpus()
        .into_iter()
        .find(|named| named.case_id == "deferred-decoder-cancel-abort-retry")
        .expect("C03 in-flight corpus keeps the deferred decoder abort/retry boundary")
        .trace;
    let candidates = source.shrink_candidates();

    let mut cancel_removed = source.clone();
    cancel_removed.operations.remove(6);
    assert!(
        cancel_removed.validate().is_err(),
        "removing the deferred cancel changes the retry complete arity without slot rebase"
    );
    assert!(
        !candidates.contains(&cancel_removed),
        "raw reducer must reject deferred-cancel deletion that needs slot rebase"
    );

    let mut abort_retry_removed = source.clone();
    abort_retry_removed.operations.remove(7);
    abort_retry_removed.operations.remove(7);
    assert!(
        abort_retry_removed.validate().is_err(),
        "removing abort/retry keeps a two-slot complete for its original three-slot plan"
    );
    assert!(
        !candidates.contains(&abort_retry_removed),
        "raw reducer must reject abort/retry deletion that needs pending-plan rebase"
    );

    let mut first_wave_capacity_reduced = source.clone();
    let InflightMixedProgramOperation::Submit { max_new_tokens, .. } =
        &mut first_wave_capacity_reduced.operations[1]
    else {
        panic!("deferred decoder corpus keeps first-wave label-two submit at index one");
    };
    *max_new_tokens = 1;
    assert!(
        first_wave_capacity_reduced.validate().is_err(),
        "reducing first-wave capacity changes the retry complete arity without slot rebase"
    );
    assert!(
        !candidates.contains(&first_wave_capacity_reduced),
        "raw reducer must reject capacity reduction that needs retry-slot rebase"
    );
}

#[test]
fn inflight_mixed_program_reducer_candidates_are_deduped_ranked_and_canonical() {
    let mut sources = inflight_mixed_program_corpus()
        .into_iter()
        .map(|named| named.trace)
        .collect::<Vec<_>>();
    sources.push(raw_abort_retry_trace());
    sources.push(raw_deferred_complete_trace());
    sources.extend((1_u64..=1_024).map(|index| {
        InflightMixedProgramTrace::from_seed(0x9c54_0fe2_a731_b86d_u64.wrapping_mul(index))
    }));

    for source in sources {
        let source_rank = source.shrink_rank();
        let candidates = source.shrink_candidates();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            assert!(
                !candidates[..candidate_index].contains(candidate),
                "in-flight reducer emitted a duplicate candidate"
            );
            assert_eq!(candidate.seed, source.seed);
            assert!(matches!(
                candidate.operations.last(),
                Some(InflightMixedProgramOperation::Close)
            ));
            assert!(candidate.shrink_rank() < source_rank);
            let document =
                serialize_inflight_mixed_program_descriptor("candidate-canonical", candidate);
            let parsed = parse_inflight_mixed_program_descriptor(&document)
                .expect("in-flight reducer candidate stays strict-canonical");
            assert_eq!(parsed.trace, *candidate);
            replay_inflight_mixed_program_inner(candidate);
        }
    }
}

#[test]
fn inflight_mixed_program_reducer_finds_a_raw_lifecycle_local_minimum() {
    let source = raw_abort_retry_trace();
    assert!(raw_lifecycle_reducer_predicate(&source));
    let minimized = minimize_inflight_mixed_program_trace(&source, raw_lifecycle_reducer_predicate);
    assert_eq!(minimized.seed, source.seed);
    assert!(minimized.shrink_rank() < source.shrink_rank());
    assert!(raw_lifecycle_reducer_predicate(&minimized));
    assert!(
        !minimized.operations.iter().any(|operation| {
            matches!(operation, InflightMixedProgramOperation::AbortNotDispatched)
        }),
        "synthetic predicate keeps the raw abort/retry contraction"
    );
    let document = serialize_inflight_mixed_program_descriptor("raw-local-minimum", &minimized);
    let parsed = parse_inflight_mixed_program_descriptor(&document)
        .expect("raw in-flight local minimum stays strict-canonical");
    assert_eq!(parsed.trace, minimized);
    replay_inflight_mixed_program_inner(&source);
    replay_inflight_mixed_program_inner(&minimized);
    assert_eq!(
        minimize_inflight_mixed_program_trace(&minimized, raw_lifecycle_reducer_predicate),
        minimized
    );
    assert!(
        minimized
            .shrink_candidates()
            .iter()
            .all(|candidate| !raw_lifecycle_reducer_predicate(candidate)),
        "in-flight reducer result must be a local minimum for its fixed candidate order"
    );
}

#[test]
fn inflight_mixed_program_failure_report_preserves_source_and_local_minimum() {
    let source = raw_abort_retry_trace();
    let minimized = minimize_inflight_mixed_program_trace(&source, raw_lifecycle_reducer_predicate);
    let report = inflight_mixed_program_failure_report(&source, &minimized);
    assert!(report.contains("reducer_scope=v1-raw-pending-lifecycle-local"));
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
    let parsed_original = parse_inflight_mixed_program_descriptor(&original_document)
        .expect("failure report source descriptor stays strict-canonical");
    let parsed_minimized = parse_inflight_mixed_program_descriptor(&minimized_document)
        .expect("failure report minimized descriptor stays strict-canonical");
    assert_eq!(parsed_original.case_id, "failing-original");
    assert_eq!(parsed_original.trace, source);
    assert_eq!(parsed_minimized.case_id, "failing-minimized");
    assert_eq!(parsed_minimized.trace, minimized);
}

#[test]
fn ten_thousand_seeded_inflight_mixed_programs_round_trip_and_replay() {
    let mut saw_deferred_complete = false;
    let mut saw_abort_without_cancel = false;
    let mut saw_deferred_abort = false;
    let mut saw_abort_retry_with_live_close = false;
    let rotation = routing_fuzz_rotation::configured_seed_rotation();
    for local_trace_index in 0..INFLIGHT_MIXED_PROGRAM_TRACE_COUNT {
        let trace_index = rotation
            .trace_index(local_trace_index, INFLIGHT_MIXED_PROGRAM_TRACE_COUNT)
            .expect("configured routing-fuzz rotation fits the in-flight mixed trace window");
        let seed = 0x9c54_0fe2_a731_b86d_u64.wrapping_mul(trace_index.wrapping_add(1));
        let trace = InflightMixedProgramTrace::from_seed(seed);
        let has_cancel = trace
            .operations
            .iter()
            .any(|operation| matches!(operation, InflightMixedProgramOperation::Cancel { .. }));
        let has_abort = trace.operations.iter().any(|operation| {
            matches!(operation, InflightMixedProgramOperation::AbortNotDispatched)
        });
        match (has_cancel, has_abort) {
            (true, false) => saw_deferred_complete = true,
            (false, true) => {
                saw_abort_without_cancel = true;
                let last_submit_max_new_tokens =
                    trace
                        .operations
                        .iter()
                        .rev()
                        .find_map(|operation| match operation {
                            InflightMixedProgramOperation::Submit { max_new_tokens, .. } => {
                                Some(*max_new_tokens)
                            }
                            InflightMixedProgramOperation::Plan
                            | InflightMixedProgramOperation::Cancel { .. }
                            | InflightMixedProgramOperation::Complete { .. }
                            | InflightMixedProgramOperation::AbortNotDispatched
                            | InflightMixedProgramOperation::Close => None,
                        });
                assert_eq!(last_submit_max_new_tokens, Some(2));
                saw_abort_retry_with_live_close = true;
            }
            (true, true) => saw_deferred_abort = true,
            (false, false) => panic!("seeded in-flight program omitted its second-plan settlement"),
        }
        let document = serialize_inflight_mixed_program_descriptor("seeded-program", &trace);
        let parsed = parse_inflight_mixed_program_descriptor(&document)
            .expect("seeded in-flight mixed program stays strict-canonical");
        assert_eq!(parsed.trace, trace);
        replay_inflight_mixed_program(&parsed.trace);
    }
    assert!(saw_deferred_complete);
    assert!(saw_abort_without_cancel);
    assert!(saw_deferred_abort);
    assert!(saw_abort_retry_with_live_close);
}

// V2 is intentionally a separate public-API adapter: the V1 raw descriptor
// remains a closed grammar and cannot accidentally begin accepting a terminal
// device-quiesced disposition.
#[allow(clippy::too_many_lines)]
fn replay_inflight_mixed_program_v2_inner(trace: &InflightMixedProgramTraceV2) {
    trace
        .validate()
        .expect("in-flight mixed program V2 replay receives a valid descriptor");
    let mut scheduler = Some(new_scheduler());
    let mut oracle = InflightMixedProgramOracle::new(trace.seed);
    let mut request_ids = BTreeMap::<u8, RequestId>::new();
    let mut pending_plan = None::<IterationPlan>;
    let mut retry_after_not_dispatched_abort = None;
    let mut now_ns = 0_u64;
    let mut close_seen = false;

    for operation in &trace.operations {
        match operation {
            InflightMixedProgramOperationV2::Submit {
                label,
                max_new_tokens,
            } => {
                let submission = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .submit(
                        RequestDescriptor::new(
                            vec![inflight_symbolic_prompt_token(*label)],
                            usize::from(*max_new_tokens),
                        ),
                        now_ns,
                    )
                    .expect("in-flight mixed program V2 submission");
                assert!(
                    request_ids
                        .insert(*label, submission.request_id())
                        .is_none(),
                    "in-flight mixed program V2 submitted one label twice"
                );
                oracle.bind_submit(*label, *max_new_tokens, submission.request_id());
            }
            InflightMixedProgramOperationV2::Plan => {
                let expected = oracle.begin_plan();
                let planning = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .plan_iteration(now_ns)
                    .expect("in-flight mixed program V2 plan succeeds");
                let (plan, completions) = planning.into_parts();
                assert!(
                    completions.is_empty(),
                    "in-flight mixed program V2 does not enable admission timeouts"
                );
                let plan = plan.expect("in-flight mixed program V2 has live work to plan");
                if let Some(aborted_iteration_id) = retry_after_not_dispatched_abort.take() {
                    assert_ne!(
                        plan.iteration_id(),
                        aborted_iteration_id,
                        "in-flight mixed program V2 retry reused its aborted iteration ID"
                    );
                }
                assert_plan_projection(&plan, &expected);
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight V2 scheduler remains live")
                        .inflight_iteration_id(),
                    Some(plan.iteration_id())
                );
                let concurrent_plan = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live")
                    .plan_iteration(now_ns);
                assert!(
                    matches!(
                        concurrent_plan,
                        Err(SchedulerError::IterationInFlight { iteration_id })
                            if iteration_id == plan.iteration_id()
                    ),
                    "in-flight mixed program V2 allowed a second plan before settlement"
                );
                assert!(
                    pending_plan.replace(plan).is_none(),
                    "in-flight mixed program V2 replaced its outstanding plan"
                );
            }
            InflightMixedProgramOperationV2::Cancel { label } => {
                let request_id = *request_ids
                    .get(label)
                    .expect("in-flight mixed program V2 cancel label is submitted");
                let outcome = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .cancel(request_id, now_ns)
                    .expect("in-flight mixed program V2 cancellation succeeds");
                let snapshot = scheduler
                    .as_ref()
                    .expect("in-flight V2 scheduler remains live")
                    .request_snapshot(request_id)
                    .expect("deferred V2 cancellation retains its live request snapshot");
                assert!(snapshot.cancellation_deferred());
                oracle.defer_cancel(*label, &outcome);
            }
            InflightMixedProgramOperationV2::Complete {
                feedback_slot_order,
            } => {
                let plan = pending_plan
                    .take()
                    .expect("in-flight mixed program V2 has a plan to complete");
                let result = oracle.feedback(plan.iteration_id(), feedback_slot_order);
                let updates = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .complete_iteration(&result, now_ns)
                    .expect("in-flight mixed program V2 completion succeeds");
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight V2 scheduler remains live")
                        .inflight_iteration_id(),
                    None
                );
                oracle.record_complete(&updates, now_ns);
            }
            InflightMixedProgramOperationV2::AbortNotDispatched => {
                let plan = pending_plan
                    .take()
                    .expect("in-flight mixed program V2 has a plan to abort");
                let aborted_iteration_id = plan.iteration_id();
                let updates = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, now_ns)
                    .expect("in-flight mixed program V2 not-dispatched abort succeeds");
                assert_eq!(
                    scheduler
                        .as_ref()
                        .expect("in-flight V2 scheduler remains live")
                        .inflight_iteration_id(),
                    None
                );
                oracle.record_not_dispatched_abort(&updates, now_ns);
                assert!(
                    retry_after_not_dispatched_abort
                        .replace(aborted_iteration_id)
                        .is_none(),
                    "in-flight mixed program V2 aborted more than one retryable plan"
                );
            }
            InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown => {
                let plan = pending_plan
                    .take()
                    .expect("in-flight mixed program V2 has a plan to terminally abort");
                let planned_request_ids = plan
                    .decode_items()
                    .iter()
                    .chain(plan.prefill_items())
                    .map(riley_scheduler::WorkItem::request_id)
                    .collect::<Vec<_>>();
                let updates = scheduler
                    .as_mut()
                    .expect("in-flight V2 scheduler remains live before close")
                    .abort_iteration(
                        plan.iteration_id(),
                        ExecutionAbort::DeviceQuiescedMutationUnknown,
                        now_ns,
                    )
                    .expect("in-flight mixed program V2 quiesced abort succeeds");
                let scheduler_ref = scheduler
                    .as_ref()
                    .expect("in-flight V2 scheduler remains live");
                assert_eq!(scheduler_ref.inflight_iteration_id(), None);
                assert_eq!(scheduler_ref.active_sequence_count(), 0);
                assert!(
                    updates.iteration_metric().is_none(),
                    "device-quiesced terminal abort must not publish an iteration metric"
                );
                assert!(
                    updates.token_events().is_empty(),
                    "device-quiesced terminal abort must not publish token events"
                );
                assert!(
                    updates.settlement_failures().is_empty(),
                    "device-quiesced terminal abort must not report settlement failures"
                );
                for request_id in planned_request_ids {
                    assert_eq!(
                        scheduler_ref.request_state(request_id),
                        Some(RequestState::Failed),
                        "device-quiesced terminal abort must mark every planned request failed"
                    );
                }
                oracle.record_device_quiesced_mutation_unknown_abort(&updates, now_ns);
            }
            InflightMixedProgramOperationV2::Close => {
                assert!(pending_plan.is_none());
                let closed = scheduler
                    .take()
                    .expect("in-flight mixed program V2 closes its scheduler exactly once")
                    .close(now_ns, None)
                    .unwrap_or_else(|failure| {
                        panic!(
                            "seed {:#018x}: in-flight mixed program V2 close failed: {}",
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
            .expect("in-flight mixed program V2 operation clock");
    }
    assert!(close_seen);
    assert!(scheduler.is_none());
    assert!(pending_plan.is_none());
    assert!(retry_after_not_dispatched_abort.is_none());
    oracle.assert_closed();
}

fn inflight_mixed_program_v2_fails(trace: &InflightMixedProgramTraceV2) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        replay_inflight_mixed_program_v2_inner(trace);
    }))
    .is_err()
}

fn inflight_mixed_program_v2_failure_report(
    source: &InflightMixedProgramTraceV2,
    minimized: &InflightMixedProgramTraceV2,
) -> String {
    let source_descriptor =
        serialize_inflight_mixed_program_v2_descriptor("failing-original", source);
    let minimized_descriptor =
        serialize_inflight_mixed_program_v2_descriptor("failing-minimized", minimized);
    format!(
        "C03-A inflight-mixed-program-v2 failed\n\\
         original_descriptor_json:\n\\
         {source_descriptor}\\
         minimized_descriptor_json:\n\\
         {minimized_descriptor}\\
         original_operations=[{}]\n\\
         minimized_operations=[{}]\n\\
         reducer_scope=v2-raw-pending-lifecycle-local\n\\
         failure_predicate=inner-replayer-panicked-only\n\\
         not_established=panic-site,payload,failure-signature,root-cause,general-or-global-minimum,label-or-slot-rebase,arbitrary-operation-deletion,unbounded-or-general-scheduler,actual-cuda-stream-quiescence,device-mutation-parity,pending-close,invalid-feedback,partial-prefill,queue-aging,fault-injection,receipt,gpu,c02-qualification",
        source.describe_operations(),
        minimized.describe_operations(),
    )
}

fn replay_inflight_mixed_program_v2(trace: &InflightMixedProgramTraceV2) {
    if !inflight_mixed_program_v2_fails(trace) {
        return;
    }
    let minimized =
        minimize_inflight_mixed_program_v2_trace(trace, inflight_mixed_program_v2_fails);
    panic!(
        "{}",
        inflight_mixed_program_v2_failure_report(trace, &minimized)
    );
}

fn raw_v2_device_quiesced_abort_trace() -> InflightMixedProgramTraceV2 {
    InflightMixedProgramTraceV2 {
        seed: 0x7e6d_5c4b_3a29_1807,
        operations: vec![
            InflightMixedProgramOperationV2::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperationV2::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperationV2::Plan,
            InflightMixedProgramOperationV2::Complete {
                feedback_slot_order: vec![1, 0],
            },
            InflightMixedProgramOperationV2::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperationV2::Plan,
            InflightMixedProgramOperationV2::Cancel { label: 1 },
            InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown,
            InflightMixedProgramOperationV2::Close,
        ],
    }
}

fn raw_v2_not_dispatched_retry_trace() -> InflightMixedProgramTraceV2 {
    InflightMixedProgramTraceV2 {
        seed: 0x6d5c_4b3a_2918_07f6,
        operations: vec![
            InflightMixedProgramOperationV2::Submit {
                label: 1,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperationV2::Plan,
            InflightMixedProgramOperationV2::Complete {
                feedback_slot_order: vec![0],
            },
            InflightMixedProgramOperationV2::Submit {
                label: 2,
                max_new_tokens: 2,
            },
            InflightMixedProgramOperationV2::Submit {
                label: 3,
                max_new_tokens: 1,
            },
            InflightMixedProgramOperationV2::Plan,
            InflightMixedProgramOperationV2::AbortNotDispatched,
            InflightMixedProgramOperationV2::Plan,
            InflightMixedProgramOperationV2::Complete {
                feedback_slot_order: vec![2, 1, 0],
            },
            InflightMixedProgramOperationV2::Close,
        ],
    }
}

fn raw_v2_device_abort_reducer_predicate(trace: &InflightMixedProgramTraceV2) -> bool {
    trace.operations.iter().any(|operation| {
        matches!(
            operation,
            InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown
        )
    }) && trace
        .operations
        .iter()
        .any(|operation| matches!(operation, InflightMixedProgramOperationV2::Cancel { .. }))
}

#[test]
fn inflight_mixed_program_v2_corpus_is_canonical_and_replays() {
    for named in inflight_mixed_program_v2_corpus() {
        let document = serialize_inflight_mixed_program_v2_descriptor(&named.case_id, &named.trace);
        let parsed = parse_inflight_mixed_program_v2_descriptor(&document)
            .expect("in-flight mixed program V2 corpus stays strict-canonical");
        assert_eq!(parsed, named);
        replay_inflight_mixed_program_v2(&parsed.trace);
    }
}

#[test]
fn inflight_mixed_program_v2_codec_keeps_v1_and_v2_parser_surfaces_disjoint() {
    let v1_document = serialize_inflight_mixed_program_descriptor(
        "cross-parser-v1",
        &three_slot_mixed_trace(vec![2, 1, 0]),
    );
    let v2_trace = raw_v2_device_quiesced_abort_trace();
    let v2_document = serialize_inflight_mixed_program_v2_descriptor("cross-parser-v2", &v2_trace);

    assert!(
        parse_inflight_mixed_program_descriptor(&v1_document).is_ok(),
        "V1 parser must continue to accept the V1 grammar"
    );
    assert!(
        parse_inflight_mixed_program_v2_descriptor(&v2_document).is_ok(),
        "V2 parser must accept the V2 grammar"
    );
    let parsed_v2 = parse_inflight_mixed_program_v2_descriptor(&v2_document)
        .expect("canonical V2 descriptor parses");
    assert_eq!(parsed_v2.case_id, "cross-parser-v2");
    assert_eq!(parsed_v2.trace, v2_trace);
    assert_eq!(
        serialize_inflight_mixed_program_v2_descriptor(&parsed_v2.case_id, &parsed_v2.trace),
        v2_document,
        "V2 codec must round-trip its canonical document byte-for-byte"
    );
    assert!(
        parse_inflight_mixed_program_descriptor(&v2_document).is_err(),
        "V1 parser must reject the V2 quiesced-abort grammar"
    );
    let retagged_v2_document = v2_document.replacen(
        "\"trace_kind\":\"inflight-mixed-program-v2\"",
        "\"trace_kind\":\"inflight-mixed-program-v1\"",
        1,
    );
    assert!(
        parse_inflight_mixed_program_descriptor(&retagged_v2_document).is_err(),
        "V1 parser must reject V2-only abort operations even if a document is retagged"
    );
    assert!(
        parse_inflight_mixed_program_v2_descriptor(&v1_document).is_err(),
        "V2 parser must reject the V1 descriptor kind"
    );
}

#[test]
fn inflight_mixed_program_v2_codec_rejects_noncanonical_and_invalid_documents() {
    let valid = serialize_inflight_mixed_program_v2_descriptor(
        "codec-strict-v2",
        &raw_v2_device_quiesced_abort_trace(),
    );
    let invalid_documents = [
        (
            "unsupported V2 trace kind",
            valid.replacen(
                "\"trace_kind\":\"inflight-mixed-program-v2\"",
                "\"trace_kind\":\"inflight-mixed-program-v3\"",
                1,
            ),
        ),
        (
            "unsupported version",
            valid.replacen("\"format_version\":1", "\"format_version\":2", 1),
        ),
        (
            "duplicate outer field",
            valid.replacen(
                "{\"format\":\"riley.scheduler.inflight-mixed-program\",",
                "{\"format\":\"riley.scheduler.inflight-mixed-program\",\"format\":\"riley.scheduler.inflight-mixed-program\",",
                1,
            ),
        ),
        (
            "reordered outer field",
            valid.replacen(
                "\"format\":\"riley.scheduler.inflight-mixed-program\",\"format_version\":1",
                "\"format_version\":1,\"format\":\"riley.scheduler.inflight-mixed-program\"",
                1,
            ),
        ),
        (
            "unknown outer field",
            valid.replacen("\"operations\":[", "\"unexpected\":true,\"operations\":[", 1),
        ),
        (
            "unknown nested field",
            valid.replacen(
                "{\"op\":\"abort_device_quiesced_mutation_unknown\"}",
                "{\"op\":\"abort_device_quiesced_mutation_unknown\",\"unexpected\":true}",
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
            "terminal device abort must be immediately closed",
            valid.replacen(
                "{\"op\":\"abort_device_quiesced_mutation_unknown\"},{\"op\":\"close\"}",
                "{\"op\":\"abort_device_quiesced_mutation_unknown\"},{\"op\":\"plan\"},{\"op\":\"close\"}",
                1,
            ),
        ),
        (
            "terminal device abort must consume a mixed plan",
            v2_descriptor_document_with_operations(
                r#"{"op":"submit","label":1,"max_new_tokens":2},{"op":"plan"},{"op":"complete","feedback_slot_order":[0]},{"op":"submit","label":2,"max_new_tokens":2},{"op":"plan"},{"op":"complete","feedback_slot_order":[0,1]},{"op":"plan"},{"op":"abort_device_quiesced_mutation_unknown"},{"op":"close"}"#,
            ),
        ),
        ("malformed JSON", "{not-json}\n".to_owned()),
        ("noncanonical whitespace", format!(" {valid}")),
    ];
    for (case, document) in invalid_documents {
        assert!(
            parse_inflight_mixed_program_v2_descriptor(&document).is_err(),
            "{case}: invalid V2 in-flight mixed program descriptor was accepted: {document:?}"
        );
    }
}

#[test]
fn inflight_mixed_program_v2_replays_not_dispatched_retry_separately() {
    let trace = raw_v2_not_dispatched_retry_trace();
    trace
        .validate()
        .expect("V2 not-dispatched retry fixture stays within the V2 grammar");
    replay_inflight_mixed_program_v2(&trace);
}

#[test]
fn inflight_mixed_program_v2_reducer_preserves_terminal_device_abort() {
    let source = raw_v2_device_quiesced_abort_trace();
    source
        .validate()
        .expect("V2 quiesced-abort reducer fixture stays valid");
    let candidates = source.shrink_candidates();

    let mut abort_removed = source.clone();
    abort_removed.operations.remove(7);
    assert!(
        abort_removed.validate().is_err(),
        "removing the terminal device abort leaves the plan pending at close"
    );
    assert!(
        !candidates.contains(&abort_removed),
        "V2 reducer must not remove the terminal device abort"
    );
    assert!(
        candidates.iter().all(|candidate| {
            candidate.operations.iter().any(|operation| {
                matches!(
                    operation,
                    InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown
                )
            })
        }),
        "every V2 reduction must retain the terminal device abort"
    );

    let minimized =
        minimize_inflight_mixed_program_v2_trace(&source, raw_v2_device_abort_reducer_predicate);
    assert!(raw_v2_device_abort_reducer_predicate(&minimized));
    assert!(
        minimized.operations.iter().any(|operation| {
            matches!(
                operation,
                InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown
            )
        }),
        "a V2 local minimum preserving the predicate retains the terminal device abort"
    );
    let document = serialize_inflight_mixed_program_v2_descriptor("v2-device-abort", &minimized);
    let parsed = parse_inflight_mixed_program_v2_descriptor(&document)
        .expect("V2 reducer result stays strict-canonical");
    assert_eq!(parsed.trace, minimized);
    replay_inflight_mixed_program_v2(&source);
    replay_inflight_mixed_program_v2(&minimized);
}

#[test]
fn ten_thousand_seeded_inflight_mixed_program_v2s_round_trip_and_replay() {
    let mut saw_quiesced_abort_without_deferred_cancel = false;
    let mut saw_quiesced_abort_with_deferred_cancel = false;
    let rotation = routing_fuzz_rotation::configured_seed_rotation();
    for local_trace_index in 0..INFLIGHT_MIXED_PROGRAM_V2_TRACE_COUNT {
        let trace_index = rotation
            .trace_index(local_trace_index, INFLIGHT_MIXED_PROGRAM_V2_TRACE_COUNT)
            .expect("configured routing-fuzz rotation fits the V2 in-flight mixed trace window");
        let seed = 0x4dc1_5e5d_9c2a_73b1_u64.wrapping_mul(trace_index.wrapping_add(1));
        let trace = InflightMixedProgramTraceV2::from_seed(seed);
        let has_deferred_cancel = trace
            .operations
            .iter()
            .any(|operation| matches!(operation, InflightMixedProgramOperationV2::Cancel { .. }));
        let quiesced_abort_count = trace
            .operations
            .iter()
            .filter(|operation| {
                matches!(
                    operation,
                    InflightMixedProgramOperationV2::AbortDeviceQuiescedMutationUnknown
                )
            })
            .count();
        assert!(
            quiesced_abort_count <= 1,
            "a seeded V2 trace must not mix terminal device-quiesced aborts"
        );
        if quiesced_abort_count == 1 {
            if has_deferred_cancel {
                saw_quiesced_abort_with_deferred_cancel = true;
            } else {
                saw_quiesced_abort_without_deferred_cancel = true;
            }
        }
        let document = serialize_inflight_mixed_program_v2_descriptor("seeded-program-v2", &trace);
        let parsed = parse_inflight_mixed_program_v2_descriptor(&document)
            .expect("seeded in-flight mixed program V2 stays strict-canonical");
        assert_eq!(parsed.trace, trace);
        replay_inflight_mixed_program_v2(&parsed.trace);
    }
    assert!(saw_quiesced_abort_without_deferred_cancel);
    assert!(saw_quiesced_abort_with_deferred_cancel);
}
