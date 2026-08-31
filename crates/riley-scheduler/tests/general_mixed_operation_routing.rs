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
    minimize_general_mixed_operation_trace, parse_general_mixed_operation_trace_descriptor,
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

fn general_mixed_operation_fails(trace: &GeneralMixedOperationTrace) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        replay_general_mixed_operation_inner(trace);
    }))
    .is_err()
}

fn general_mixed_operation_failure_report(
    case_id: &str,
    trace: &GeneralMixedOperationTrace,
    minimized: &GeneralMixedOperationTrace,
) -> String {
    let original_descriptor = serialize_general_mixed_operation_trace_descriptor(case_id, trace);
    let minimized_descriptor =
        serialize_general_mixed_operation_trace_descriptor("failing-minimized", minimized);
    format!(
        "C03-A general-mixed-operation-v1 failed\n\
         source_case_id={case_id}\n\
         reducer_scope=v1-selector-local\n\
         failure_predicate=inner-replayer-panicked-only\n\
         original_descriptor_json:\n\
         {}\n\
         original_operations=[{}]\n\
         minimized_descriptor_json:\n\
         {}\n\
         minimized_operations=[{}]\n\
         note=local reducer preserves only the replay panic predicate, not panic site, payload, failure signature, or root cause",
        original_descriptor.trim_end(),
        trace.describe_operations(),
        minimized_descriptor.trim_end(),
        minimized.describe_operations(),
    )
}

fn replay_general_mixed_operation(case_id: &str, trace: &GeneralMixedOperationTrace) {
    if !general_mixed_operation_fails(trace) {
        return;
    }
    let minimized = minimize_general_mixed_operation_trace(trace, general_mixed_operation_fails);
    panic!(
        "{}",
        general_mixed_operation_failure_report(case_id, trace, &minimized)
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

fn slot_permutations(slot_count: usize) -> Vec<Vec<u8>> {
    let mut values = (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("bounded permutation slot"))
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

fn canonical_slot_order(slot_count: usize) -> Vec<u8> {
    (0..slot_count)
        .map(|slot| u8::try_from(slot).expect("bounded canonical slot"))
        .collect()
}

fn reducer_fixture() -> GeneralMixedOperationTrace {
    GeneralMixedOperationTrace {
        seed: 0x75c4_81a2_3b9d_e6f0,
        decoder_count: 3,
        final_prefill_count: 2,
        prime_slot_order: vec![2, 0, 1],
        mixed_slot_order: vec![4, 1, 3, 0, 2],
        cancel_decoder_index: Some(2),
        settlement: GeneralMixedOperationSettlement::AbortNotDispatched,
    }
}

#[test]
fn general_mixed_operation_reducer_rebases_removed_request_selectors() {
    let source = reducer_fixture();
    let candidates = source.shrink_candidates();
    let remove_decoder_zero = GeneralMixedOperationTrace {
        decoder_count: 2,
        prime_slot_order: vec![1, 0],
        mixed_slot_order: vec![3, 0, 2, 1],
        cancel_decoder_index: Some(1),
        ..source.clone()
    };
    let remove_decoder_one = GeneralMixedOperationTrace {
        decoder_count: 2,
        prime_slot_order: vec![1, 0],
        mixed_slot_order: vec![3, 2, 0, 1],
        cancel_decoder_index: Some(1),
        ..source.clone()
    };
    let remove_decoder_two = GeneralMixedOperationTrace {
        decoder_count: 2,
        prime_slot_order: vec![0, 1],
        mixed_slot_order: vec![3, 1, 2, 0],
        cancel_decoder_index: None,
        ..source.clone()
    };
    let remove_final_prefill_zero = GeneralMixedOperationTrace {
        final_prefill_count: 1,
        mixed_slot_order: vec![3, 1, 0, 2],
        ..source.clone()
    };
    for expected in [
        remove_decoder_zero,
        remove_decoder_one,
        remove_decoder_two,
        remove_final_prefill_zero,
    ] {
        assert!(
            candidates.contains(&expected),
            "selector-aware reducer omitted the expected rebase candidate: {expected:?}"
        );
        let document =
            serialize_general_mixed_operation_trace_descriptor("rebase-candidate", &expected);
        let parsed = parse_general_mixed_operation_trace_descriptor(&document)
            .expect("rebased candidate is strict-canonical");
        assert_eq!(parsed.trace, expected);
        replay_general_mixed_operation_inner(&parsed.trace);
    }

    let final_prefill_source = GeneralMixedOperationTrace {
        seed: 0x0d1b_4e8a_7c02_f593,
        decoder_count: 2,
        final_prefill_count: 3,
        prime_slot_order: vec![1, 0],
        mixed_slot_order: vec![2, 0, 3, 1, 4],
        cancel_decoder_index: Some(1),
        settlement: GeneralMixedOperationSettlement::Commit,
    };
    let final_prefill_candidates = final_prefill_source.shrink_candidates();
    for expected in [
        GeneralMixedOperationTrace {
            final_prefill_count: 2,
            mixed_slot_order: vec![2, 0, 1, 3],
            ..final_prefill_source.clone()
        },
        GeneralMixedOperationTrace {
            final_prefill_count: 2,
            mixed_slot_order: vec![2, 0, 3, 1],
            ..final_prefill_source.clone()
        },
    ] {
        assert!(
            final_prefill_candidates.contains(&expected),
            "selector-aware reducer omitted a later final-prefill rebase: {expected:?}"
        );
        let document =
            serialize_general_mixed_operation_trace_descriptor("later-prefill-rebase", &expected);
        let parsed = parse_general_mixed_operation_trace_descriptor(&document)
            .expect("later final-prefill candidate is strict-canonical");
        assert_eq!(parsed.trace, expected);
        replay_general_mixed_operation_inner(&parsed.trace);
    }
}

#[test]
fn general_mixed_operation_reducer_orders_identity_before_adjacent_swaps() {
    let source = reducer_fixture();
    let candidates = source.shrink_candidates();
    let cancellation_removed = GeneralMixedOperationTrace {
        cancel_decoder_index: None,
        ..source.clone()
    };
    let lower_cancel_zero = GeneralMixedOperationTrace {
        cancel_decoder_index: Some(0),
        ..source.clone()
    };
    let lower_cancel_one = GeneralMixedOperationTrace {
        cancel_decoder_index: Some(1),
        ..source.clone()
    };
    let direct_prime_identity = GeneralMixedOperationTrace {
        prime_slot_order: canonical_slot_order(source.decoder_count),
        ..source.clone()
    };
    let direct_mixed_identity = GeneralMixedOperationTrace {
        mixed_slot_order: canonical_slot_order(source.decoder_count + source.final_prefill_count),
        ..source.clone()
    };
    let adjacent_prime_swap = GeneralMixedOperationTrace {
        prime_slot_order: vec![0, 2, 1],
        ..source.clone()
    };
    let adjacent_mixed_swap = GeneralMixedOperationTrace {
        mixed_slot_order: vec![1, 4, 3, 0, 2],
        ..source.clone()
    };
    let cancellation_removed_position = candidates
        .iter()
        .position(|candidate| candidate == &cancellation_removed)
        .expect("reducer removes the cancellation before structural edits");
    let lower_cancel_zero_position = candidates
        .iter()
        .position(|candidate| candidate == &lower_cancel_zero)
        .expect("reducer emits the lowest cancellation target");
    let lower_cancel_one_position = candidates
        .iter()
        .position(|candidate| candidate == &lower_cancel_one)
        .expect("reducer emits every lower cancellation target");
    let first_decoder_removal_position = candidates
        .iter()
        .position(|candidate| candidate.decoder_count < source.decoder_count)
        .expect("reducer emits decoder removals");
    let first_final_prefill_removal_position = candidates
        .iter()
        .position(|candidate| candidate.final_prefill_count < source.final_prefill_count)
        .expect("reducer emits final-prefill removals");
    let direct_prime_position = candidates
        .iter()
        .position(|candidate| candidate == &direct_prime_identity)
        .expect("reducer emits a direct prime identity candidate");
    let direct_mixed_position = candidates
        .iter()
        .position(|candidate| candidate == &direct_mixed_identity)
        .expect("reducer emits a direct mixed identity candidate");
    let adjacent_prime_position = candidates
        .iter()
        .position(|candidate| candidate == &adjacent_prime_swap)
        .expect("reducer emits the left-most adjacent prime inversion swap");
    let adjacent_mixed_position = candidates
        .iter()
        .position(|candidate| candidate == &adjacent_mixed_swap)
        .expect("reducer emits the left-most adjacent mixed inversion swap");
    assert!(
        cancellation_removed_position < lower_cancel_zero_position
            && lower_cancel_zero_position < lower_cancel_one_position
            && lower_cancel_one_position < first_decoder_removal_position
            && first_decoder_removal_position < first_final_prefill_removal_position
            && first_final_prefill_removal_position < direct_prime_position
            && direct_prime_position < direct_mixed_position
            && direct_mixed_position < adjacent_prime_position
            && adjacent_prime_position < adjacent_mixed_position,
        "V1 reducer candidate order must remain cancellation, removals, identities, then swaps"
    );
    assert_eq!(
        adjacent_prime_swap.shrink_rank().3 + 1,
        source.shrink_rank().3,
        "one adjacent inversion swap must lower only one inversion"
    );
}

#[test]
fn general_mixed_operation_reducer_candidates_are_deduped_ranked_and_canonical() {
    let mut source_count = 0_u64;
    for decoder_count in 1..=3 {
        let prime_orders = slot_permutations(decoder_count);
        for final_prefill_count in 1..=3 {
            let mixed_orders = slot_permutations(decoder_count + final_prefill_count);
            for prime_slot_order in &prime_orders {
                for mixed_slot_order in &mixed_orders {
                    for cancel_decoder_index in
                        std::iter::once(None).chain((0..decoder_count).map(Some))
                    {
                        for settlement in [
                            GeneralMixedOperationSettlement::Commit,
                            GeneralMixedOperationSettlement::AbortNotDispatched,
                        ] {
                            source_count += 1;
                            let trace = GeneralMixedOperationTrace {
                                seed: source_count,
                                decoder_count,
                                final_prefill_count,
                                prime_slot_order: prime_slot_order.clone(),
                                mixed_slot_order: mixed_slot_order.clone(),
                                cancel_decoder_index,
                                settlement,
                            };
                            let source_rank = trace.shrink_rank();
                            let candidates = trace.shrink_candidates();
                            for (index, candidate) in candidates.iter().enumerate() {
                                assert!(
                                    !candidates[..index].contains(candidate),
                                    "reducer emitted a duplicate candidate"
                                );
                                assert!(
                                    candidate.shrink_rank() < source_rank,
                                    "reducer candidate rank must strictly decrease"
                                );
                                assert_eq!(candidate.seed, trace.seed);
                                assert_eq!(candidate.settlement, trace.settlement);
                                let document = serialize_general_mixed_operation_trace_descriptor(
                                    "selector-candidate",
                                    candidate,
                                );
                                let parsed =
                                    parse_general_mixed_operation_trace_descriptor(&document)
                                        .expect("every reducer candidate is strict-canonical");
                                assert_eq!(parsed.trace, *candidate);
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        source_count, 43_400,
        "selector matrix must cover every bounded V1 descriptor shape"
    );
}

fn synthetic_general_mixed_operation_failure(trace: &GeneralMixedOperationTrace) -> bool {
    trace.decoder_count >= 2
        && trace.final_prefill_count >= 2
        && trace.cancel_decoder_index == Some(0)
        && trace.settlement == GeneralMixedOperationSettlement::AbortNotDispatched
        && trace.shrink_rank().3 > 0
}

#[test]
fn general_mixed_operation_reducer_preserves_and_minimizes_a_synthetic_failure() {
    let source = GeneralMixedOperationTrace {
        seed: 0x9462_f1b8_0e37_a5cd,
        decoder_count: 3,
        final_prefill_count: 3,
        prime_slot_order: vec![2, 0, 1],
        mixed_slot_order: vec![5, 4, 3, 2, 1, 0],
        cancel_decoder_index: Some(0),
        settlement: GeneralMixedOperationSettlement::AbortNotDispatched,
    };
    assert!(synthetic_general_mixed_operation_failure(&source));
    replay_general_mixed_operation_inner(&source);
    let minimized =
        minimize_general_mixed_operation_trace(&source, synthetic_general_mixed_operation_failure);
    assert_eq!(minimized.seed, source.seed);
    assert_eq!(minimized.decoder_count, 2);
    assert_eq!(minimized.final_prefill_count, 2);
    assert_eq!(minimized.cancel_decoder_index, Some(0));
    assert_eq!(minimized.prime_slot_order, vec![0, 1]);
    assert_eq!(minimized.mixed_slot_order, vec![1, 0, 2, 3]);
    assert_eq!(
        minimized.settlement,
        GeneralMixedOperationSettlement::AbortNotDispatched
    );
    assert!(synthetic_general_mixed_operation_failure(&minimized));
    replay_general_mixed_operation_inner(&minimized);
    let document =
        serialize_general_mixed_operation_trace_descriptor("synthetic-minimized", &minimized);
    let parsed = parse_general_mixed_operation_trace_descriptor(&document)
        .expect("minimized synthetic descriptor is strict-canonical");
    assert_eq!(parsed.trace, minimized);
    assert_eq!(
        minimize_general_mixed_operation_trace(
            &minimized,
            synthetic_general_mixed_operation_failure
        ),
        minimized,
        "V1 minimizer must be idempotent at its deterministic local minimum"
    );
    assert!(
        minimized
            .shrink_candidates()
            .iter()
            .all(|candidate| !synthetic_general_mixed_operation_failure(candidate)),
        "V1 minimizer must return a local minimum for its deterministic candidate order"
    );
}

#[test]
fn general_mixed_operation_failure_report_delimits_its_panic_only_scope() {
    let source = reducer_fixture();
    let minimized = GeneralMixedOperationTrace {
        decoder_count: 2,
        final_prefill_count: 1,
        prime_slot_order: vec![0, 1],
        mixed_slot_order: vec![0, 1, 2],
        cancel_decoder_index: None,
        ..source.clone()
    };
    let report = general_mixed_operation_failure_report("report-source", &source, &minimized);
    for expected in [
        "source_case_id=report-source",
        "reducer_scope=v1-selector-local",
        "failure_predicate=inner-replayer-panicked-only",
        "original_descriptor_json:",
        "minimized_descriptor_json:",
        "original_operations=[",
        "minimized_operations=[",
        "not panic site, payload, failure signature, or root cause",
    ] {
        assert!(
            report.contains(expected),
            "failure report omitted its required boundary: {expected}"
        );
    }
    let original = serialize_general_mixed_operation_trace_descriptor("report-source", &source);
    let minimized_document =
        serialize_general_mixed_operation_trace_descriptor("failing-minimized", &minimized);
    assert!(
        report.contains(original.trim_end()),
        "failure report must retain the full original canonical descriptor"
    );
    assert!(
        report.contains(minimized_document.trim_end()),
        "failure report must retain the full minimized canonical descriptor"
    );
    assert_eq!(
        parse_general_mixed_operation_trace_descriptor(&original)
            .expect("original report descriptor parses")
            .trace,
        source
    );
    assert_eq!(
        parse_general_mixed_operation_trace_descriptor(&minimized_document)
            .expect("minimized report descriptor parses")
            .trace,
        minimized
    );
}
