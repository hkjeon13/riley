//! Model-free qualification of the opt-in complete-P128/DecodeN host policy.
//! Synthetic executor results prove scheduler contracts only, never GPU parity.

use std::collections::HashSet;

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, ExecutionShapePolicy, IterationOutput, IterationPlan, IterationResult,
    OverloadPolicy, RequestDescriptor, RequestFinishReason, RequestId, RequestState, Scheduler,
    SchedulerConfig, SchedulerError, WorkKind,
};

fn config(capacity: usize) -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 8,
        max_waiting_prompt_tokens: 1024,
        max_active_sequences: capacity,
        max_sequence_tokens: 160,
        iteration_token_budget: 128,
        max_prefill_chunk_tokens: 128,
        aging_threshold_ns: 1,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: capacity * 10,
        metrics_window_samples: 16,
    }
}

fn layout(blocks: usize) -> KvLayout {
    KvLayout::checked(30, blocks, 3, 64).unwrap()
}

fn scheduler(capacity: usize) -> Scheduler {
    Scheduler::new_with_execution_shape(
        config(capacity),
        layout(capacity * 10),
        ExecutionShapePolicy::CompletePrefill128DecodeN,
    )
    .unwrap()
}

fn submit(scheduler: &mut Scheduler, seed: u32, outputs: usize, now: u64) -> RequestId {
    scheduler
        .submit(RequestDescriptor::new(vec![seed; 128], outputs), now)
        .unwrap()
        .request_id()
}

fn plan(scheduler: &mut Scheduler, now: u64) -> IterationPlan {
    let output = scheduler.plan_iteration(now).unwrap();
    assert!(output.completions().is_empty());
    let plan = output.into_parts().0.unwrap();
    if plan.prefill_items().is_empty() {
        assert!((1..=4).contains(&plan.decode_items().len()));
        assert_eq!(plan.total_tokens(), plan.decode_items().len());
        assert!(plan.decode_items().iter().all(|item| {
            item.input_tokens().len() == 1 && (129..=159).contains(&item.target_logical_length())
        }));
    } else {
        assert_eq!(plan.prefill_items().len(), 1);
        assert!(plan.decode_items().is_empty());
        assert_eq!(plan.total_tokens(), 128);
        assert_eq!(plan.prefill_items()[0].target_logical_length(), 128);
    }
    assert_eq!(plan.output_slots().len(), plan.batch_size());
    let mut blocks = HashSet::new();
    for table in plan.block_tables() {
        assert_eq!(table.physical_block_ids().len(), table.valid_tokens().len());
        assert!(
            table
                .physical_block_ids()
                .iter()
                .all(|&id| blocks.insert(id))
        );
        assert_eq!(
            table
                .valid_tokens()
                .iter()
                .map(|&v| u32::from(v))
                .sum::<u32>(),
            table.logical_length()
        );
    }
    plan
}

fn result(plan: &IterationPlan, stop: Option<RequestId>) -> IterationResult {
    // Reverse result order to exercise routing by dense slot, not vector order.
    let outputs = plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items())
        .rev()
        .map(|item| {
            IterationOutput::new(
                item.output_slot().unwrap(),
                u32::try_from(item.request_id().get()).unwrap(),
                Some(item.request_id()) == stop,
            )
        })
        .collect();
    IterationResult::new(plan.iteration_id(), outputs, 1, 0).unwrap()
}

fn complete(scheduler: &mut Scheduler, plan: &IterationPlan, now: u64) {
    let update = scheduler
        .complete_iteration(&result(plan, None), now)
        .unwrap();
    assert!(update.settlement_failures().is_empty());
    for event in update.token_events() {
        assert_eq!(u64::from(event.token_id()), event.request_id().get());
    }
}

fn close(scheduler: Scheduler, now: u64) {
    let output = scheduler.close(now, None).unwrap();
    assert!(output.settlement_failures().is_empty());
}

#[test]
fn policy_rejects_incompatible_configuration_before_pool_construction() {
    let mutations: &[fn(&mut SchedulerConfig)] = &[
        |c| c.max_active_sequences = 1,
        |c| c.max_active_sequences = 3,
        |c| c.max_active_sequences = 5,
        |c| c.iteration_token_budget = 127,
        |c| c.iteration_token_budget = 129,
        |c| c.max_prefill_chunk_tokens = 127,
        |c| c.max_sequence_tokens = 159,
        |c| c.max_sequence_tokens = 161,
        |c| c.max_waiting_prompt_tokens = 127,
        |c| c.max_promised_kv_blocks = 19,
    ];
    for mutate in mutations {
        let mut c = config(2);
        mutate(&mut c);
        assert!(
            Scheduler::new_with_execution_shape(
                c,
                layout(40),
                ExecutionShapePolicy::CompletePrefill128DecodeN
            )
            .is_err()
        );
    }
    assert!(
        Scheduler::new_with_execution_shape(
            config(2),
            layout(19),
            ExecutionShapePolicy::CompletePrefill128DecodeN
        )
        .is_err()
    );
    assert!(
        Scheduler::new_with_execution_shape(
            config(2),
            KvLayout::checked(1, 20, 1, 8).unwrap(),
            ExecutionShapePolicy::CompletePrefill128DecodeN
        )
        .is_err()
    );
}

#[test]
fn invalid_requests_never_take_promises_ids_or_waiting_capacity() {
    let mut scheduler = scheduler(2);
    for (tokens, outputs) in [
        (vec![1; 127], 1),
        (vec![1; 129], 1),
        (vec![1; 128], 0),
        (vec![1; 128], 33),
        (vec![49_152; 128], 1),
    ] {
        assert!(
            scheduler
                .submit(RequestDescriptor::new(tokens, outputs), 0)
                .is_err()
        );
        assert_eq!(scheduler.active_sequence_count(), 0);
        assert_eq!(scheduler.waiting_request_count(), 0);
        assert_eq!(scheduler.promised_kv_blocks(), 0);
        assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    }
    assert_eq!(submit(&mut scheduler, 1, 1, 0).get(), 1);
    assert_eq!(scheduler.promised_kv_blocks(), 10);
    close(scheduler, 1);
}

#[test]
fn default_constructor_retains_mixed_127_token_prefill_behavior() {
    let mut scheduler = Scheduler::new(config(2), layout(20)).unwrap();
    assert_eq!(
        scheduler.execution_shape_policy(),
        ExecutionShapePolicy::General
    );
    submit(&mut scheduler, 1, 3, 0);
    let first = scheduler.plan_iteration(1).unwrap().into_parts().0.unwrap();
    complete(&mut scheduler, &first, 2);
    submit(&mut scheduler, 2, 3, 3);
    let mixed = scheduler.plan_iteration(4).unwrap().into_parts().0.unwrap();
    assert_eq!(mixed.decode_items().len(), 1);
    assert_eq!(mixed.prefill_items()[0].input_tokens().len(), 127);
    scheduler
        .abort_iteration(mixed.iteration_id(), ExecutionAbort::NotDispatched, 5)
        .unwrap();
    close(scheduler, 6);
}

#[test]
fn aged_prefill_pressure_alternates_whole_prompts_and_immediate_decode_buckets() {
    let mut scheduler = scheduler(4);
    let ids: Vec<_> = (1..=4)
        .map(|id| submit(&mut scheduler, id, 32, 0))
        .collect();
    let mut now = 1;
    for (kind, count) in [
        (WorkKind::Prefill, 1),
        (WorkKind::Decode, 1),
        (WorkKind::Prefill, 1),
        (WorkKind::Decode, 2),
        (WorkKind::Prefill, 1),
        (WorkKind::Decode, 3),
        (WorkKind::Prefill, 1),
        (WorkKind::Decode, 4),
    ] {
        let p = plan(&mut scheduler, now);
        if kind == WorkKind::Prefill {
            assert_eq!(p.prefill_items().len(), count);
        } else {
            assert_eq!(p.decode_items().len(), count);
            assert_eq!(
                p.decode_items()
                    .iter()
                    .map(|item| item.request_id())
                    .collect::<Vec<_>>(),
                ids[..count]
            );
        }
        complete(&mut scheduler, &p, now + 1);
        now += 2;
    }
    assert_eq!(scheduler.promised_kv_blocks(), 40);
    close(scheduler, now);
}

#[test]
fn waiting_short_request_still_reserves_ten_blocks_after_admission() {
    let mut scheduler = scheduler(2);
    let first = submit(&mut scheduler, 1, 1, 0);
    submit(&mut scheduler, 2, 32, 0);
    let waiting = submit(&mut scheduler, 3, 1, 0);
    assert_eq!(
        scheduler.request_state(waiting),
        Some(RequestState::Waiting)
    );
    assert_eq!(scheduler.promised_kv_blocks(), 20);
    let p = plan(&mut scheduler, 1);
    assert_eq!(p.prefill_items()[0].request_id(), first);
    complete(&mut scheduler, &p, 2);
    let p = plan(&mut scheduler, 3);
    assert_eq!(
        scheduler
            .request_snapshot(waiting)
            .unwrap()
            .promised_kv_blocks(),
        10
    );
    assert_eq!(scheduler.promised_kv_blocks(), 20);
    complete(&mut scheduler, &p, 4);
    close(scheduler, 5);
}

#[test]
fn undispatched_retry_and_rejected_results_do_not_advance_class_progress() {
    let mut scheduler = scheduler(2);
    submit(&mut scheduler, 1, 8, 0);
    submit(&mut scheduler, 2, 8, 0);
    let p = plan(&mut scheduler, 1);
    let first = p.prefill_items()[0].request_id();
    let incomplete = IterationResult::new(p.iteration_id(), vec![], 0, 0).unwrap();
    assert!(scheduler.complete_iteration(&incomplete, 2).is_err());
    scheduler
        .abort_iteration(p.iteration_id(), ExecutionAbort::NotDispatched, 2)
        .unwrap();
    let retry = plan(&mut scheduler, 3);
    assert_eq!(retry.prefill_items()[0].request_id(), first);
    complete(&mut scheduler, &retry, 4);
    let decode = plan(&mut scheduler, 5);
    assert_eq!(decode.decode_items().len(), 1);
    assert!(matches!(
        scheduler.plan_iteration(6),
        Err(SchedulerError::IterationInFlight { .. })
    ));
    scheduler
        .abort_iteration(decode.iteration_id(), ExecutionAbort::NotDispatched, 6)
        .unwrap();
    let retry = plan(&mut scheduler, 7);
    assert_eq!(retry.decode_items().len(), 1);
    complete(&mut scheduler, &retry, 8);
    let prefill = plan(&mut scheduler, 9);
    assert_eq!(prefill.prefill_items().len(), 1);
    complete(&mut scheduler, &prefill, 10);
    close(scheduler, 11);
}

#[test]
fn rejected_output_vocabulary_cannot_enter_the_next_decode() {
    let mut scheduler = scheduler(2);
    submit(&mut scheduler, 1, 2, 0);
    let p = plan(&mut scheduler, 1);
    let bad = IterationResult::new(
        p.iteration_id(),
        vec![IterationOutput::new(p.output_slots()[0], 49_152, false)],
        0,
        0,
    )
    .unwrap();
    assert!(scheduler.complete_iteration(&bad, 2).is_err());
    assert_eq!(
        scheduler
            .request_snapshot(p.prefill_items()[0].request_id())
            .unwrap()
            .generated_tokens(),
        0
    );
    complete(&mut scheduler, &p, 2);
    let p = plan(&mut scheduler, 3);
    assert!(p.decode_items()[0].input_tokens()[0] < 49_152);
    complete(&mut scheduler, &p, 4);
    close(scheduler, 5);
}

#[test]
fn cancellation_of_one_inflight_decoder_keeps_other_rows_and_reuses_capacity() {
    let mut scheduler = scheduler(4);
    let ids: Vec<_> = (1..=4)
        .map(|id| submit(&mut scheduler, id, 32, 0))
        .collect();
    let mut now = 1;
    for _ in 0..7 {
        let p = plan(&mut scheduler, now);
        complete(&mut scheduler, &p, now + 1);
        now += 2;
    }
    let p = plan(&mut scheduler, now);
    assert_eq!(p.decode_items().len(), 4);
    let before = scheduler.pool_stats().allocated_block_count();
    assert!(
        scheduler
            .cancel(ids[1], now + 1)
            .unwrap()
            .deferred_until_iteration_settles()
    );
    assert_eq!(scheduler.pool_stats().allocated_block_count(), before);
    let updates = scheduler
        .complete_iteration(&result(&p, None), now + 2)
        .unwrap();
    assert_eq!(updates.token_events().len(), 3);
    assert!(
        updates
            .token_events()
            .iter()
            .all(|event| event.request_id() != ids[1])
    );
    assert_eq!(
        updates.completions()[0].reason(),
        RequestFinishReason::Cancelled
    );
    let p = plan(&mut scheduler, now + 3);
    assert_eq!(p.decode_items().len(), 3);
    complete(&mut scheduler, &p, now + 4);
    scheduler.cancel(ids[0], now + 5).unwrap();
    scheduler.cancel(ids[2], now + 5).unwrap();
    let p = plan(&mut scheduler, now + 6);
    assert_eq!(p.decode_items().len(), 1);
    complete(&mut scheduler, &p, now + 7);
    submit(&mut scheduler, 5, 32, now + 8);
    let p = plan(&mut scheduler, now + 9);
    assert_eq!(p.prefill_items().len(), 1);
    complete(&mut scheduler, &p, now + 10);
    let p = plan(&mut scheduler, now + 11);
    assert_eq!(p.decode_items().len(), 2);
    complete(&mut scheduler, &p, now + 12);
    close(scheduler, now + 13);
}

#[test]
fn output_limit32_never_plans_low_level_position159_or_output33() {
    let mut scheduler = scheduler(2);
    let id = submit(&mut scheduler, 1, 32, 0);
    for index in 0..32 {
        let p = plan(&mut scheduler, 2 * index + 1);
        let item = p
            .prefill_items()
            .first()
            .or_else(|| p.decode_items().first())
            .unwrap();
        assert_eq!(
            item.target_logical_length(),
            128 + usize::try_from(index).unwrap()
        );
        complete(&mut scheduler, &p, 2 * index + 2);
    }
    assert_eq!(scheduler.request_state(id), Some(RequestState::Finished));
    assert!(scheduler.plan_iteration(65).unwrap().plan().is_none());
    assert_eq!(scheduler.promised_kv_blocks(), 0);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    close(scheduler, 66);
}

#[test]
fn quiesced_failed_dispatch_poisons_only_its_rows_then_allows_survivors() {
    let mut scheduler = scheduler(4);
    let first = submit(&mut scheduler, 1, 32, 0);
    let second = submit(&mut scheduler, 2, 32, 0);
    let p = plan(&mut scheduler, 1);
    complete(&mut scheduler, &p, 2);
    let p = plan(&mut scheduler, 3);
    let updates = scheduler
        .abort_iteration(
            p.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            4,
        )
        .unwrap();
    assert_eq!(updates.completions()[0].request_id(), first);
    assert_eq!(
        updates.completions()[0].reason(),
        RequestFinishReason::ExecutorFailure
    );
    assert!(updates.settlement_failures().is_empty());
    let p = plan(&mut scheduler, 5);
    assert_eq!(p.prefill_items()[0].request_id(), second);
    complete(&mut scheduler, &p, 6);
    close(scheduler, 7);
}

#[test]
fn possibly_dispatched_prefill_failure_does_not_delay_a_healthy_decoder_again() {
    let mut scheduler = scheduler(4);
    let first = submit(&mut scheduler, 1, 32, 0);
    let second = submit(&mut scheduler, 2, 32, 0);
    let third = submit(&mut scheduler, 3, 32, 0);
    for now in [1, 3] {
        let p = plan(&mut scheduler, now);
        complete(&mut scheduler, &p, now + 1);
    }
    let p = plan(&mut scheduler, 5);
    assert_eq!(p.prefill_items()[0].request_id(), second);
    scheduler
        .abort_iteration(
            p.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            6,
        )
        .unwrap();
    let p = plan(&mut scheduler, 7);
    assert_eq!(p.decode_items()[0].request_id(), first);
    complete(&mut scheduler, &p, 8);
    let p = plan(&mut scheduler, 9);
    assert_eq!(p.prefill_items()[0].request_id(), third);
    complete(&mut scheduler, &p, 10);
    close(scheduler, 11);
}

#[test]
fn continuous_short_prefills_cannot_starve_an_aged_ready_decoder() {
    let mut scheduler = scheduler(2);
    let decoder = submit(&mut scheduler, 1, 32, 0);
    let first = plan(&mut scheduler, 1);
    complete(&mut scheduler, &first, 2);
    let mut now = 3;
    let mut previous_was_prefill = true;
    let mut generated = 1;
    let mut admitted_short = None;
    while scheduler.request_state(decoder) != Some(RequestState::Finished) {
        if admitted_short.is_none() {
            admitted_short = Some(submit(&mut scheduler, 2, 1, now));
        }
        let p = plan(&mut scheduler, now);
        if p.decode_items().is_empty() {
            assert!(!previous_was_prefill);
            assert_eq!(Some(p.prefill_items()[0].request_id()), admitted_short);
            admitted_short = None;
            previous_was_prefill = true;
        } else {
            assert_eq!(p.decode_items()[0].request_id(), decoder);
            generated += 1;
            previous_was_prefill = false;
        }
        complete(&mut scheduler, &p, now + 1);
        now += 2;
        assert!(now < 130, "ready decoder failed bounded class progress");
    }
    assert_eq!(generated, 32);
    close(scheduler, now);
}
