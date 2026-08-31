//! C03-A CPU-only replay for the parameterized general-mixed-operation V1 grammar.
//!
//! The descriptor and its support oracle construct every feedback slot without
//! reading an `IterationPlan`. This adapter validates the public plan separately,
//! then exercises only public scheduler APIs.

#[path = "support/general_mixed_operation_trace.rs"]
mod general_mixed_operation_trace;

use std::panic::{AssertUnwindSafe, catch_unwind};

use general_mixed_operation_trace::{
    GeneralMixedOperationOracle, GeneralMixedOperationSettlement, GeneralMixedOperationTrace,
    NamedGeneralMixedOperationTrace, decoder_symbolic_token, general_mixed_operation_corpus,
    parse_general_mixed_operation_trace_descriptor,
    serialize_general_mixed_operation_trace_descriptor,
};
use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, IterationPlan, OutputSlot, OverloadPolicy, RequestDescriptor, RequestId,
    RequestState, Scheduler, SchedulerCloseOutput, SchedulerConfig,
};

const GENERAL_MIXED_OPERATION_TRACE_COUNT: u64 = 10_000;

fn new_scheduler(config: SchedulerConfig) -> Scheduler {
    let layout = KvLayout::checked(1, 64, 1, 8).expect("valid C03-A symbolic KV layout");
    Scheduler::new(config, layout).expect("valid C03-A general mixed scheduler configuration")
}

fn general_mixed_operation_config(trace: &GeneralMixedOperationTrace) -> SchedulerConfig {
    let width = trace
        .decoder_count
        .checked_add(trace.final_prefill_count)
        .expect("bounded general mixed request width");
    SchedulerConfig {
        max_waiting_requests: width,
        max_waiting_prompt_tokens: width,
        max_active_sequences: width,
        max_sequence_tokens: 3,
        iteration_token_budget: width,
        max_prefill_chunk_tokens: 1,
        aging_threshold_ns: 2,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: width,
        metrics_window_samples: 8,
    }
}

fn decoder_prompt_token(index: usize) -> u32 {
    1_000_u32
        .checked_add(u32::try_from(index).expect("bounded decoder prompt index"))
        .expect("bounded decoder prompt token")
}

fn final_prefill_prompt_token(index: usize) -> u32 {
    2_000_u32
        .checked_add(u32::try_from(index).expect("bounded final-prefill prompt index"))
        .expect("bounded final-prefill prompt token")
}

fn slot(index: usize) -> OutputSlot {
    OutputSlot::new(u32::try_from(index).expect("bounded general mixed slot"))
}

fn expected_slots(count: usize) -> Vec<OutputSlot> {
    (0..count).map(slot).collect()
}

fn take_plan(planning: riley_scheduler::PlanningOutput, stage: &str) -> IterationPlan {
    let (plan, completions) = planning.into_parts();
    assert!(
        completions.is_empty(),
        "{stage}: fixed V1 grammar does not enable timeout completions"
    );
    plan.unwrap_or_else(|| panic!("{stage}: fixed V1 grammar must produce a plan"))
}

fn assert_prime_plan(
    trace: &GeneralMixedOperationTrace,
    plan: &IterationPlan,
    decoder_ids: &[RequestId],
) {
    assert_eq!(decoder_ids.len(), trace.decoder_count);
    assert_eq!(plan.prefill_items().len(), trace.decoder_count);
    assert!(plan.decode_items().is_empty());
    for (index, request_id) in decoder_ids.iter().copied().enumerate() {
        let item = &plan.prefill_items()[index];
        assert_eq!(item.request_id(), request_id);
        assert_eq!(item.input_tokens(), &[decoder_prompt_token(index)]);
        assert_eq!(item.output_slot(), Some(slot(index)));
    }
    assert_eq!(
        plan.output_slots(),
        expected_slots(trace.decoder_count).as_slice()
    );
    assert_eq!(plan.total_tokens(), trace.decoder_count);
}

fn assert_mixed_plan(
    trace: &GeneralMixedOperationTrace,
    plan: &IterationPlan,
    decoder_ids: &[RequestId],
    final_prefill_ids: &[RequestId],
) {
    assert_eq!(decoder_ids.len(), trace.decoder_count);
    assert_eq!(final_prefill_ids.len(), trace.final_prefill_count);
    assert_eq!(plan.decode_items().len(), trace.decoder_count);
    assert_eq!(plan.prefill_items().len(), trace.final_prefill_count);
    for (index, request_id) in decoder_ids.iter().copied().enumerate() {
        let item = &plan.decode_items()[index];
        assert_eq!(item.request_id(), request_id);
        assert_eq!(item.input_tokens(), &[decoder_symbolic_token(index, 0)]);
        assert_eq!(item.output_slot(), Some(slot(index)));
    }
    for (index, request_id) in final_prefill_ids.iter().copied().enumerate() {
        let item = &plan.prefill_items()[index];
        assert_eq!(item.request_id(), request_id);
        assert_eq!(item.input_tokens(), &[final_prefill_prompt_token(index)]);
        assert_eq!(item.output_slot(), Some(slot(trace.decoder_count + index)));
    }
    let total_slots = trace
        .decoder_count
        .checked_add(trace.final_prefill_count)
        .expect("bounded general mixed slot count");
    assert_eq!(plan.output_slots(), expected_slots(total_slots).as_slice());
    assert_eq!(plan.total_tokens(), total_slots);
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

fn replay_general_mixed_operation_inner(trace: &GeneralMixedOperationTrace) {
    let mut scheduler = new_scheduler(general_mixed_operation_config(trace));
    let mut oracle = GeneralMixedOperationOracle::new(
        trace.seed,
        trace.decoder_count,
        trace.final_prefill_count,
    );
    let mut decoder_ids = Vec::with_capacity(trace.decoder_count);
    for index in 0..trace.decoder_count {
        let submission = scheduler
            .submit(
                RequestDescriptor::new(vec![decoder_prompt_token(index)], 2),
                0,
            )
            .expect("submit bounded general mixed decoder");
        oracle.bind_decoder(index, submission.request_id());
        decoder_ids.push(submission.request_id());
    }

    let prime_plan = take_plan(
        scheduler
            .plan_iteration(0)
            .expect("plan general mixed prime"),
        "prime",
    );
    assert_prime_plan(trace, &prime_plan, &decoder_ids);
    oracle.observe_prime_plan();
    let prime_result = oracle.prime_feedback(prime_plan.iteration_id(), &trace.prime_slot_order);
    let prime_updates = scheduler
        .complete_iteration(&prime_result, 0)
        .expect("commit general mixed prime");
    oracle.record_prime_commit(&prime_updates);
    for request_id in &decoder_ids {
        assert_eq!(
            scheduler.request_state(*request_id),
            Some(RequestState::Decoding)
        );
    }

    let mut final_prefill_ids = Vec::with_capacity(trace.final_prefill_count);
    for index in 0..trace.final_prefill_count {
        let submission = scheduler
            .submit(
                RequestDescriptor::new(vec![final_prefill_prompt_token(index)], 1),
                1,
            )
            .expect("submit bounded general mixed final-prefill");
        oracle.bind_final_prefill(index, submission.request_id());
        final_prefill_ids.push(submission.request_id());
    }

    let mixed_plan = take_plan(
        scheduler
            .plan_iteration(1)
            .expect("plan general mixed wave"),
        "mixed",
    );
    assert_mixed_plan(trace, &mixed_plan, &decoder_ids, &final_prefill_ids);
    oracle.observe_mixed_plan();
    if let Some(index) = trace.cancel_decoder_index {
        let cancellation = scheduler
            .cancel(decoder_ids[index], 1)
            .expect("defer bounded general mixed decoder cancellation");
        assert!(cancellation.deferred_until_iteration_settles());
        assert!(cancellation.completion().is_none());
        oracle.defer_decoder_cancel(index);
    }
    match trace.settlement {
        GeneralMixedOperationSettlement::Commit => {
            let result = oracle.mixed_feedback(mixed_plan.iteration_id(), &trace.mixed_slot_order);
            let updates = scheduler
                .complete_iteration(&result, 1)
                .expect("commit bounded general mixed wave");
            oracle.record_mixed_commit(&updates, 1);
        }
        GeneralMixedOperationSettlement::AbortNotDispatched => {
            let updates = scheduler
                .abort_iteration(mixed_plan.iteration_id(), ExecutionAbort::NotDispatched, 1)
                .expect("abort bounded general mixed wave before dispatch");
            oracle.record_not_dispatched_abort(&updates, 1);
        }
    }

    let closed = scheduler.close(2, None).unwrap_or_else(|failure| {
        panic!(
            "general mixed close failed for seed {:#018x}: {}",
            trace.seed,
            failure.error()
        )
    });
    oracle.record_close(&closed, 2);
    assert_closed_quiescent(&closed);
    oracle.assert_closed();
}

fn replay_general_mixed_operation(case_id: &str, trace: &GeneralMixedOperationTrace) {
    let document = serialize_general_mixed_operation_trace_descriptor(case_id, trace);
    let replay = catch_unwind(AssertUnwindSafe(|| {
        replay_general_mixed_operation_inner(trace);
    }));
    assert!(
        replay.is_ok(),
        "C03-A general mixed operation failed: case_id={case_id}; descriptor={} operations=[{}]",
        document.trim_end(),
        trace.describe_operations(),
    );
}

fn replay_named_general_mixed_operation(named: &NamedGeneralMixedOperationTrace) {
    replay_general_mixed_operation(&named.case_id, &named.trace);
}

fn descriptor_fixture() -> GeneralMixedOperationTrace {
    GeneralMixedOperationTrace {
        seed: 0x12ab_34cd_56ef_7890,
        decoder_count: 2,
        final_prefill_count: 2,
        prime_slot_order: vec![0, 1],
        mixed_slot_order: vec![0, 1, 2, 3],
        cancel_decoder_index: None,
        settlement: GeneralMixedOperationSettlement::Commit,
    }
}

fn cancel_case_component(cancel_decoder_index: Option<usize>) -> String {
    cancel_decoder_index.map_or_else(|| "none".to_owned(), |index| format!("decoder-{index}"))
}

fn settlement_case_component(settlement: GeneralMixedOperationSettlement) -> &'static str {
    match settlement {
        GeneralMixedOperationSettlement::Commit => "commit",
        GeneralMixedOperationSettlement::AbortNotDispatched => "abort",
    }
}

#[test]
fn general_mixed_operation_codec_round_trips_the_bounded_grammar() {
    let trace = descriptor_fixture();
    let document = serialize_general_mixed_operation_trace_descriptor("codec-round-trip", &trace);
    let parsed = parse_general_mixed_operation_trace_descriptor(&document)
        .expect("canonical general mixed descriptor parses");
    assert_eq!(parsed.case_id, "codec-round-trip");
    assert_eq!(parsed.trace, trace);
    assert_eq!(
        serialize_general_mixed_operation_trace_descriptor(&parsed.case_id, &parsed.trace),
        document
    );
}

#[test]
fn general_mixed_operation_codec_rejects_noncanonical_or_invalid_documents() {
    let valid =
        serialize_general_mixed_operation_trace_descriptor("codec-strict", &descriptor_fixture());
    let invalid_documents = [
        valid.replacen("\"format_version\":1", "\"format_version\":2", 1),
        valid.replacen(
            "\"trace_kind\":\"general-mixed-operation-v1\"",
            "\"trace_kind\":\"other\"",
            1,
        ),
        valid.replacen("\"case_id\":\"codec-strict\",", "", 1),
        valid.replacen(
            "\"decoder_count\":2,",
            "\"decoder_count\":2,\"decoder_count\":2,",
            1,
        ),
        valid.replacen("\"decoder_count\":2", "\"decoder_count\":0", 1),
        valid.replacen(
            "\"prime_slot_order\":[0,1]",
            "\"prime_slot_order\":[0,0]",
            1,
        ),
        valid.replacen(
            "\"mixed_slot_order\":[0,1,2,3]",
            "\"mixed_slot_order\":[0,1,2,4]",
            1,
        ),
        valid.replacen(
            "\"cancel_decoder_index\":null",
            "\"cancel_decoder_index\":2",
            1,
        ),
        valid.replacen(
            "\"source_seed\":\"0x12ab34cd56ef7890\"",
            "\"source_seed\":\"0X12AB34CD56EF7890\"",
            1,
        ),
        valid.replacen("\"settlement\":\"commit\"", "\"settlement\":\"invalid\"", 1),
        valid.replacen('{', "{\"unexpected\":true,", 1),
        format!(" {valid}"),
    ];
    for document in invalid_documents {
        assert!(
            parse_general_mixed_operation_trace_descriptor(&document).is_err(),
            "invalid general mixed descriptor unexpectedly parsed: {document}"
        );
    }
}

#[test]
fn general_mixed_operation_corpus_is_canonical_round_trips_and_replays() {
    for named in general_mixed_operation_corpus() {
        let document =
            serialize_general_mixed_operation_trace_descriptor(&named.case_id, &named.trace);
        let parsed = parse_general_mixed_operation_trace_descriptor(&document)
            .expect("re-serialized general mixed corpus document parses");
        assert_eq!(parsed, named);
        replay_named_general_mixed_operation(&parsed);
    }
}

#[test]
fn general_mixed_operation_matrix_preserves_routing_and_quiescence() {
    for decoder_count in 1..=3 {
        for final_prefill_count in 1..=3 {
            for prime_reversed in [false, true] {
                for mixed_reversed in [false, true] {
                    let mut cancellations = vec![None, Some(0)];
                    if decoder_count > 1 {
                        cancellations.push(Some(decoder_count - 1));
                    }
                    for cancel_decoder_index in cancellations {
                        for settlement in [
                            GeneralMixedOperationSettlement::Commit,
                            GeneralMixedOperationSettlement::AbortNotDispatched,
                        ] {
                            let mut prime_slot_order = (0..decoder_count)
                                .map(|slot| u8::try_from(slot).expect("bounded matrix prime slot"))
                                .collect::<Vec<_>>();
                            let mut mixed_slot_order = (0..decoder_count + final_prefill_count)
                                .map(|slot| u8::try_from(slot).expect("bounded matrix mixed slot"))
                                .collect::<Vec<_>>();
                            if prime_reversed {
                                prime_slot_order.reverse();
                            }
                            if mixed_reversed {
                                mixed_slot_order.reverse();
                            }
                            let cancel_component = cancel_case_component(cancel_decoder_index);
                            let seed = u64::try_from(decoder_count).expect("small decoder count")
                                | (u64::try_from(final_prefill_count)
                                    .expect("small final-prefill count")
                                    << 8)
                                | (u64::from(u8::from(prime_reversed)) << 16)
                                | (u64::from(u8::from(mixed_reversed)) << 17)
                                | (u64::from(u8::from(cancel_decoder_index.is_some())) << 18)
                                | (u64::from(u8::from(matches!(
                                    settlement,
                                    GeneralMixedOperationSettlement::AbortNotDispatched
                                ))) << 19);
                            let trace = GeneralMixedOperationTrace {
                                seed,
                                decoder_count,
                                final_prefill_count,
                                prime_slot_order,
                                mixed_slot_order,
                                cancel_decoder_index,
                                settlement,
                            };
                            let case_id = format!(
                                "matrix-d{decoder_count}-p{final_prefill_count}-prime-{prime_reversed}-mixed-{mixed_reversed}-cancel-{cancel_component}-{}",
                                settlement_case_component(settlement)
                            );
                            let document = serialize_general_mixed_operation_trace_descriptor(
                                &case_id, &trace,
                            );
                            let parsed = parse_general_mixed_operation_trace_descriptor(&document)
                                .expect("matrix descriptor parses");
                            assert_eq!(parsed.trace, trace);
                            replay_named_general_mixed_operation(&parsed);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn ten_thousand_seeded_general_mixed_operation_traces_round_trip_and_replay() {
    for trace_index in 0..GENERAL_MIXED_OPERATION_TRACE_COUNT {
        let seed = 0xd1b5_4a32_d192_ed03_u64.wrapping_mul(trace_index.wrapping_add(1));
        let trace = GeneralMixedOperationTrace::from_seed(seed);
        let case_id = format!("seed-{seed:016x}");
        let document = serialize_general_mixed_operation_trace_descriptor(&case_id, &trace);
        let parsed = parse_general_mixed_operation_trace_descriptor(&document)
            .expect("seeded general mixed descriptor parses");
        assert_eq!(parsed.trace, trace);
        replay_named_general_mixed_operation(&parsed);
    }
}
