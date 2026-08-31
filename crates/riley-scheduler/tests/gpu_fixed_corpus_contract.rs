//! CPU-only contract coverage for the fixed C03-B GPU fixture source.
//!
//! This test deliberately never initializes CUDA.  It validates that the
//! checked-in corpus has an exact canonical shape and that each descriptor's
//! public scheduler topology remains meaningful before the ignored GPU target
//! replays it against a candidate-bound CUDA executor.

#[path = "support/gpu_fixed_corpus.rs"]
mod gpu_fixed_corpus;

use std::collections::BTreeMap;

use gpu_fixed_corpus::{
    GpuFixedCorpusCase, GpuFixedPhase, GpuFixedSettlement, GpuFixedTerminalReason,
    GpuFixedWorkKind, gpu_fixed_corpus, parse_gpu_fixed_corpus_descriptor,
    serialize_gpu_fixed_corpus_descriptor,
};
use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, IterationOutput, IterationPlan, IterationResult, OutputSlot, OverloadPolicy,
    RequestCompletion, RequestDescriptor, RequestFinishReason, RequestId, RequestState, Scheduler,
    SchedulerCloseOutput, SchedulerConfig, WorkItem, WorkKind,
};

fn scheduler_config(case: &GpuFixedCorpusCase) -> SchedulerConfig {
    let maximum_prompt_tokens = case.decoder_prompt_token_count() * case.primed_labels().len()
        + case.final_prefill_prompt_token_count() * case.final_prefill_labels().len();
    SchedulerConfig {
        max_waiting_requests: case.concurrency(),
        max_waiting_prompt_tokens: maximum_prompt_tokens,
        max_active_sequences: case.concurrency(),
        max_sequence_tokens: case.maximum_sequence_tokens(),
        iteration_token_budget: case.maximum_iteration_input_tokens(),
        max_prefill_chunk_tokens: case.decoder_prompt_token_count(),
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        max_promised_kv_blocks: case.promised_kv_blocks(),
        metrics_window_samples: 8,
    }
}

fn new_scheduler(case: &GpuFixedCorpusCase) -> Scheduler {
    let layout = KvLayout::checked(1, case.promised_kv_blocks(), 1, 8)
        .expect("fixed C03-B CPU contract KV layout");
    Scheduler::new(scheduler_config(case), layout)
        .expect("fixed C03-B CPU contract scheduler configuration")
}

fn prompt_for(label: u8, token_count: usize) -> Vec<u32> {
    let first = 512_u32 + u32::from(label) * 32;
    (0..token_count)
        .map(|index| first + u32::try_from(index).expect("small fixed prompt index"))
        .collect()
}

fn generated_token(label: u8, generation_step: usize) -> u32 {
    4_000 + u32::from(label) * 16 + u32::try_from(generation_step).expect("small fixed step")
}

fn submit_labels(
    scheduler: &mut Scheduler,
    labels: &[u8],
    prompt_token_count: usize,
    max_new_tokens: usize,
    now_ns: u64,
    request_ids: &mut BTreeMap<u8, RequestId>,
) {
    for &label in labels {
        let submission = scheduler
            .submit(
                RequestDescriptor::new(prompt_for(label, prompt_token_count), max_new_tokens),
                now_ns,
            )
            .expect("fixed C03-B submission");
        assert_eq!(submission.state(), RequestState::Admitted);
        assert!(request_ids.insert(label, submission.request_id()).is_none());
    }
}

fn plan_now(scheduler: &mut Scheduler, now_ns: u64) -> IterationPlan {
    let planning = scheduler
        .plan_iteration(now_ns)
        .expect("fixed C03-B CPU plan");
    let (plan, completions) = planning.into_parts();
    assert!(completions.is_empty());
    plan.expect("fixed C03-B CPU fixture has planned live work")
}

fn assert_phase_plan(
    case: &GpuFixedCorpusCase,
    phase: GpuFixedPhase,
    plan: &IterationPlan,
    request_ids: &BTreeMap<u8, RequestId>,
) {
    let routes = case.routes_for_phase(phase);
    let decode_routes = routes
        .iter()
        .copied()
        .filter(|route| route.work_kind == GpuFixedWorkKind::Decode)
        .collect::<Vec<_>>();
    let prefill_routes = routes
        .iter()
        .copied()
        .filter(|route| route.work_kind == GpuFixedWorkKind::Prefill)
        .collect::<Vec<_>>();
    assert_eq!(plan.decode_items().len(), decode_routes.len());
    assert_eq!(plan.prefill_items().len(), prefill_routes.len());
    assert_eq!(plan.batch_size(), routes.len());
    assert_eq!(
        plan.total_tokens(),
        routes
            .iter()
            .map(|route| route.input_token_count)
            .sum::<usize>()
    );
    assert_eq!(
        plan.output_slots(),
        routes
            .iter()
            .map(|route| OutputSlot::new(u32::try_from(route.output_slot).expect("slot")))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    for (item, expected) in plan.decode_items().iter().zip(&decode_routes) {
        assert_work_item_without_table(item, *expected, request_ids, phase);
        assert_plan_table(plan, item, *expected);
    }
    for (item, expected) in plan.prefill_items().iter().zip(&prefill_routes) {
        assert_work_item_without_table(item, *expected, request_ids, phase);
        assert_plan_table(plan, item, *expected);
    }
}

fn assert_work_item_without_table(
    item: &WorkItem,
    expected: gpu_fixed_corpus::GpuFixedRoute,
    request_ids: &BTreeMap<u8, RequestId>,
    phase: GpuFixedPhase,
) {
    let expected_kind = match expected.work_kind {
        GpuFixedWorkKind::Prefill => WorkKind::Prefill,
        GpuFixedWorkKind::Decode => WorkKind::Decode,
    };
    assert_eq!(
        item.kind(),
        expected_kind,
        "{phase:?} label {}",
        expected.label
    );
    assert_eq!(
        item.request_id(),
        *request_ids
            .get(&expected.label)
            .expect("descriptor label was submitted"),
    );
    let expected_input = match expected.work_kind {
        GpuFixedWorkKind::Prefill => prompt_for(expected.label, expected.input_token_count),
        GpuFixedWorkKind::Decode => vec![generated_token(
            expected.label,
            expected
                .generation_step
                .checked_sub(1)
                .expect("decode follows an earlier generated token"),
        )],
    };
    assert_eq!(item.input_tokens(), expected_input.as_slice());
    assert_eq!(item.target_logical_length(), expected.target_logical_length);
    assert_eq!(
        item.output_slot(),
        Some(OutputSlot::new(
            u32::try_from(expected.output_slot).expect("fixed output slot fits u32"),
        )),
    );
}

fn assert_plan_table(
    plan: &IterationPlan,
    item: &WorkItem,
    expected: gpu_fixed_corpus::GpuFixedRoute,
) {
    let table = plan
        .block_tables()
        .get(item.block_table_index())
        .expect("work item has a copied block table");
    assert_eq!(table.request_id(), item.request_id());
    assert_eq!(
        usize::try_from(table.logical_length()).expect("fixed logical length"),
        expected.target_logical_length,
    );
    assert_eq!(
        table.valid_tokens(),
        GpuFixedCorpusCase::valid_tokens_for(expected.target_logical_length).as_slice(),
    );
}

fn synthetic_result(
    case: &GpuFixedCorpusCase,
    phase: GpuFixedPhase,
    plan: &IterationPlan,
) -> IterationResult {
    let mut outputs = case
        .routes_for_phase(phase)
        .into_iter()
        .map(|route| {
            IterationOutput::new(
                OutputSlot::new(u32::try_from(route.output_slot).expect("fixed output slot")),
                generated_token(route.label, route.generation_step),
                false,
            )
        })
        .collect::<Vec<_>>();
    outputs.reverse();
    IterationResult::new(plan.iteration_id(), outputs, 0, 0)
        .expect("fixed C03-B synthetic result has unique slots")
}

fn label_for(request_ids: &BTreeMap<u8, RequestId>, request_id: RequestId) -> u8 {
    request_ids
        .iter()
        .find_map(|(&label, &candidate)| (candidate == request_id).then_some(label))
        .expect("every public request ID has a fixed corpus label")
}

fn record_updates(
    updates: &riley_scheduler::IterationUpdates,
    request_ids: &BTreeMap<u8, RequestId>,
    events: &mut BTreeMap<(u8, usize), u32>,
    completions: &mut BTreeMap<u8, RequestCompletion>,
) {
    assert!(updates.settlement_failures().is_empty());
    for event in updates.token_events() {
        let label = label_for(request_ids, event.request_id());
        assert!(
            events
                .insert((label, event.generated_index()), event.token_id())
                .is_none(),
            "one fixed route may publish at most one token"
        );
    }
    for completion in updates.completions() {
        let label = label_for(request_ids, completion.request_id());
        assert!(
            completions.insert(label, completion.clone()).is_none(),
            "one fixed request may terminally complete only once"
        );
    }
}

fn assert_closed_quiescent(closed: &SchedulerCloseOutput) {
    assert!(closed.completions().is_empty());
    assert!(closed.settlement_failures().is_empty());
    let gauges = closed.final_metrics().gauges;
    assert_eq!(gauges.waiting_requests, 0);
    assert_eq!(gauges.waiting_prompt_tokens, 0);
    assert_eq!(gauges.active_sequences, 0);
    assert_eq!(gauges.promised_kv_blocks, 0);
    assert_eq!(gauges.allocated_kv_blocks, 0);
    assert_eq!(gauges.pending_completions, 0);
    assert_eq!(gauges.outstanding_iterations, 0);
}

fn expected_normal_history(case: &GpuFixedCorpusCase, label: u8) -> Vec<u32> {
    let generated_count = if matches!(
        case.settlement(),
        GpuFixedSettlement::DeferredCancel {
            label: cancelled_label
        } if *cancelled_label == label
    ) {
        1
    } else if case.primed_labels().contains(&label) {
        case.decoder_max_new_tokens()
    } else {
        case.final_prefill_max_new_tokens()
    };
    (0..generated_count)
        .map(|step| generated_token(label, step))
        .collect()
}

fn replay_normal_case(case: &GpuFixedCorpusCase) {
    assert!(!case.is_commit_data_assembly_failure());
    let mut scheduler = new_scheduler(case);
    let mut request_ids = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut completions = BTreeMap::new();

    submit_labels(
        &mut scheduler,
        case.primed_labels(),
        case.decoder_prompt_token_count(),
        case.decoder_max_new_tokens(),
        0,
        &mut request_ids,
    );
    let prime = plan_now(&mut scheduler, 1);
    assert_phase_plan(case, GpuFixedPhase::Prime, &prime, &request_ids);
    let updates = scheduler
        .complete_iteration(&synthetic_result(case, GpuFixedPhase::Prime, &prime), 2)
        .expect("prime synthetic commit");
    record_updates(&updates, &request_ids, &mut events, &mut completions);
    assert!(updates.completions().is_empty());

    submit_labels(
        &mut scheduler,
        case.final_prefill_labels(),
        case.final_prefill_prompt_token_count(),
        case.final_prefill_max_new_tokens(),
        3,
        &mut request_ids,
    );
    let mixed = plan_now(&mut scheduler, 4);
    assert_phase_plan(case, GpuFixedPhase::Mixed, &mixed, &request_ids);
    if let GpuFixedSettlement::DeferredCancel { label } = case.settlement() {
        let request_id = *request_ids
            .get(label)
            .expect("cancelled label was submitted");
        let outcome = scheduler
            .cancel(request_id, 5)
            .expect("fixed post-download cancellation");
        assert_eq!(outcome.request_id(), request_id);
        assert!(outcome.deferred_until_iteration_settles());
        assert!(!outcome.already_terminal());
        assert!(outcome.completion().is_none());
    }
    let updates = scheduler
        .complete_iteration(&synthetic_result(case, GpuFixedPhase::Mixed, &mixed), 6)
        .expect("mixed synthetic commit");
    record_updates(&updates, &request_ids, &mut events, &mut completions);

    if case.requires_boundary_decode() {
        let boundary = plan_now(&mut scheduler, 7);
        assert_phase_plan(case, GpuFixedPhase::BoundaryDecode, &boundary, &request_ids);
        let updates = scheduler
            .complete_iteration(
                &synthetic_result(case, GpuFixedPhase::BoundaryDecode, &boundary),
                8,
            )
            .expect("C=5 boundary synthetic commit");
        record_updates(&updates, &request_ids, &mut events, &mut completions);
    }

    for route in [
        case.routes_for_phase(GpuFixedPhase::Prime),
        case.routes_for_phase(GpuFixedPhase::Mixed),
        case.requires_boundary_decode()
            .then(|| case.routes_for_phase(GpuFixedPhase::BoundaryDecode))
            .unwrap_or_default(),
    ]
    .into_iter()
    .flatten()
    {
        let cancelled = matches!(
            case.settlement(),
            GpuFixedSettlement::DeferredCancel { label } if *label == route.label
        ) && route.generation_step == 1;
        if !cancelled {
            assert_eq!(
                events.get(&(route.label, route.generation_step)),
                Some(&generated_token(route.label, route.generation_step)),
            );
        }
    }
    assert_eq!(completions.len(), case.concurrency());
    for label in case.all_labels() {
        let completion = completions
            .get(&label)
            .expect("every normal fixed request completes exactly once");
        let expected_reason = match case.normal_terminal_reason(label) {
            GpuFixedTerminalReason::Cancelled => RequestFinishReason::Cancelled,
            GpuFixedTerminalReason::Length => RequestFinishReason::Length,
        };
        assert_eq!(completion.reason(), expected_reason);
        assert_eq!(
            completion.generated_token_ids(),
            expected_normal_history(case, label)
        );
    }
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let closed = scheduler.close(9, None).expect("normal fixed close");
    assert_closed_quiescent(&closed);
}

fn replay_commit_data_assembly_abort_case(case: &GpuFixedCorpusCase) {
    assert!(case.is_commit_data_assembly_failure());
    let mut scheduler = new_scheduler(case);
    let mut request_ids = BTreeMap::new();
    let mut events = BTreeMap::new();
    let mut completions = BTreeMap::new();

    submit_labels(
        &mut scheduler,
        case.primed_labels(),
        case.decoder_prompt_token_count(),
        case.decoder_max_new_tokens(),
        0,
        &mut request_ids,
    );
    let prime = plan_now(&mut scheduler, 1);
    assert_phase_plan(case, GpuFixedPhase::Prime, &prime, &request_ids);
    let updates = scheduler
        .complete_iteration(&synthetic_result(case, GpuFixedPhase::Prime, &prime), 2)
        .expect("failure-fixture prime commit");
    record_updates(&updates, &request_ids, &mut events, &mut completions);

    submit_labels(
        &mut scheduler,
        case.final_prefill_labels(),
        case.final_prefill_prompt_token_count(),
        case.final_prefill_max_new_tokens(),
        3,
        &mut request_ids,
    );
    let mixed = plan_now(&mut scheduler, 4);
    assert_phase_plan(case, GpuFixedPhase::Mixed, &mixed, &request_ids);
    // The ignored CUDA fixture reaches this path through a real downloaded
    // result whose public `into_result()` rejects seven samples for eight rows.
    // CPU source validation checks only the returned safe scheduler disposition.
    let updates = scheduler
        .abort_iteration(
            mixed.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            5,
        )
        .expect("fixed commit-data assembly abort");
    assert!(updates.token_events().is_empty());
    record_updates(&updates, &request_ids, &mut events, &mut completions);
    assert_eq!(events.len(), case.primed_labels().len());
    assert_eq!(completions.len(), case.concurrency());
    for label in case.all_labels() {
        let completion = completions
            .get(&label)
            .expect("poisoned mixed plan terminally closes every request");
        assert_eq!(completion.reason(), RequestFinishReason::ExecutorFailure);
        let expected_history = if case.primed_labels().contains(&label) {
            vec![generated_token(label, 0)]
        } else {
            Vec::new()
        };
        assert_eq!(completion.generated_token_ids(), expected_history);
    }
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let closed = scheduler.close(6, None).expect("aborted fixed close");
    assert_closed_quiescent(&closed);
}

#[test]
fn fixed_gpu_corpus_is_canonical_and_closed() {
    let corpus = gpu_fixed_corpus();
    assert_eq!(corpus.len(), 3);
    for case in &corpus {
        let canonical = serialize_gpu_fixed_corpus_descriptor(case);
        assert_eq!(
            parse_gpu_fixed_corpus_descriptor(&canonical),
            Ok(case.clone())
        );
        assert!(parse_gpu_fixed_corpus_descriptor(&format!(" {canonical}")).is_err());
    }
    let c5 = corpus
        .iter()
        .find(|case| case.case_id() == "c5-kv15-to17-mixed-greedy")
        .expect("C=5 source fixture");
    assert_eq!(GpuFixedCorpusCase::kv_block_token_count(), 16);
    assert_eq!(c5.promised_kv_blocks(), 7);
    assert_eq!(c5.maximum_iteration_input_tokens(), 30);
    assert_eq!(c5.maximum_plan_block_entries(), 5);
    assert_eq!(
        c5.routes_for_phase(GpuFixedPhase::BoundaryDecode)
            .iter()
            .map(|route| route.target_logical_length)
            .collect::<Vec<_>>(),
        [17, 17]
    );
    let drifted =
        serialize_gpu_fixed_corpus_descriptor(c5).replace("\"concurrency\":5", "\"concurrency\":6");
    assert!(parse_gpu_fixed_corpus_descriptor(&drifted).is_err());

    let c8_deferred_cancel = corpus
        .iter()
        .find(|case| case.case_id() == "c8-mixed-deferred-cancel")
        .expect("C=8 deferred-cancel source fixture");
    let c5_canonical = serialize_gpu_fixed_corpus_descriptor(c5);
    let c8_canonical = serialize_gpu_fixed_corpus_descriptor(c8_deferred_cancel);
    let oversized_labels = (1_u16..=255)
        .map(|label| label.to_string())
        .chain(std::iter::once("0".to_owned()))
        .collect::<Vec<_>>()
        .join(",");
    let oversized_label_domain = format!(
        "{{\"format\":\"riley.scheduler.gpu-fixed-corpus\",\"format_version\":1,\"trace_kind\":\"gpu-fixed-v1\",\"case_id\":\"c5-kv15-to17-mixed-greedy\",\"concurrency\":256,\"decoder_prompt_token_count\":15,\"decoder_max_new_tokens\":3,\"final_prefill_prompt_token_count\":1,\"final_prefill_max_new_tokens\":1,\"primed_labels\":[{oversized_labels}],\"final_prefill_labels\":[],\"settlement\":{{\"kind\":\"commit\"}}}}\n"
    );
    let hostile_documents = [
        c5_canonical.replacen(
            "\"format\":\"riley.scheduler.gpu-fixed-corpus\",",
            "\"format\":\"riley.scheduler.gpu-fixed-corpus\",\"format\":\"riley.scheduler.gpu-fixed-corpus\",",
            1,
        ),
        c5_canonical.replace(
            "\"settlement\":{\"kind\":\"commit\"}",
            "\"settlement\":{\"kind\":\"commit\",\"kind\":\"commit\"}",
        ),
        c5_canonical.replacen(
            "{\"format\":\"riley.scheduler.gpu-fixed-corpus\",\"format_version\":1",
            "{\"format_version\":1,\"format\":\"riley.scheduler.gpu-fixed-corpus\"",
            1,
        ),
        c8_canonical.replace(
            "\"settlement\":{\"kind\":\"deferred_cancel\",\"label\":2}",
            "\"settlement\":{\"label\":2,\"kind\":\"deferred_cancel\"}",
        ),
        c5_canonical.replacen("}}\n", "},\"unexpected\":0}\n", 1),
        c8_canonical.replace("\"label\":2", "\"label\":3"),
        c5_canonical.replace("\"concurrency\":5", "\"concurrency\":\"5\""),
        c5_canonical.replace(
            "\"decoder_prompt_token_count\":15",
            "\"decoder_prompt_token_count\":18446744073709551616",
        ),
        oversized_label_domain,
    ];
    for hostile in hostile_documents {
        assert!(
            parse_gpu_fixed_corpus_descriptor(&hostile).is_err(),
            "non-canonical or invalid fixed GPU descriptor was accepted: {hostile}"
        );
    }
}

#[test]
fn normal_gpu_fixture_topologies_replay_on_cpu() {
    for case in gpu_fixed_corpus()
        .iter()
        .filter(|case| !case.is_commit_data_assembly_failure())
    {
        replay_normal_case(case);
    }
}

#[test]
fn commit_data_assembly_failure_topology_aborts_on_cpu() {
    let case = gpu_fixed_corpus()
        .into_iter()
        .find(GpuFixedCorpusCase::is_commit_data_assembly_failure)
        .expect("fixed C=8 commit-data assembly failure fixture");
    replay_commit_data_assembly_abort_case(&case);
}
