//! Deterministic, model-free integration coverage for the PR 13 scheduler.

use std::collections::{HashMap, HashSet};

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    ExecutionAbort, IterationId, IterationOutput, IterationPlan, IterationResult, OutputSlot,
    OverloadPolicy, RequestCompletion, RequestDescriptor, RequestFinishReason, RequestId,
    RequestState, Scheduler, SchedulerConfig, SchedulerError, WorkItem, WorkKind,
};

fn compact_config() -> SchedulerConfig {
    SchedulerConfig {
        max_waiting_requests: 8,
        max_waiting_prompt_tokens: 128,
        max_active_sequences: 4,
        max_sequence_tokens: 64,
        iteration_token_budget: 8,
        max_prefill_chunk_tokens: 4,
        aging_threshold_ns: 10,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(100),
        max_promised_kv_blocks: 16,
        metrics_window_samples: 4,
    }
}

fn scheduler_with(config: SchedulerConfig, physical_blocks: usize) -> Scheduler {
    let layout = KvLayout::checked(1, physical_blocks, 1, 8).expect("valid test KV layout");
    Scheduler::new(config, layout).expect("valid test scheduler")
}

fn prompt(first: u32, len: usize) -> Vec<u32> {
    (0..len)
        .map(|offset| first + u32::try_from(offset).expect("small prompt offset"))
        .collect()
}

fn take_plan(scheduler: &mut Scheduler, now_ns: u64) -> IterationPlan {
    let output = scheduler.plan_iteration(now_ns).expect("plan iteration");
    assert!(output.completions().is_empty(), "unexpected timed-out work");
    output.into_parts().0.expect("expected planned work")
}

fn fake_result(
    plan: &IterationPlan,
    output_bases: &HashMap<RequestId, u32>,
    generated_counts: &mut HashMap<RequestId, usize>,
    gpu_execution_ns: u64,
    gpu_idle_gap_ns: u64,
) -> IterationResult {
    let mut outputs = Vec::new();
    for item in plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items().iter())
    {
        let Some(slot) = item.output_slot() else {
            continue;
        };
        let generated_index = generated_counts.entry(item.request_id()).or_default();
        let base = output_bases
            .get(&item.request_id())
            .copied()
            .expect("output base for every request");
        let token_id = base
            .checked_mul(100)
            .and_then(|value| value.checked_add(u32::try_from(*generated_index).ok()?))
            .expect("small deterministic token ID");
        *generated_index += 1;
        outputs.push(IterationOutput::new(slot, token_id, false));
    }
    // Deliberately return a different order: routing is by slot, not vector index.
    outputs.reverse();
    IterationResult::new(
        plan.iteration_id(),
        outputs,
        gpu_execution_ns,
        gpu_idle_gap_ns,
    )
    .expect("valid fake executor result")
}

fn settle_fake(
    scheduler: &mut Scheduler,
    plan: &IterationPlan,
    output_bases: &HashMap<RequestId, u32>,
    generated_counts: &mut HashMap<RequestId, usize>,
    now_ns: u64,
) -> (Vec<(RequestId, u32)>, Vec<RequestCompletion>) {
    let result = fake_result(plan, output_bases, generated_counts, now_ns + 1, now_ns);
    let updates = scheduler
        .complete_iteration(&result, now_ns)
        .expect("complete fake iteration");
    let tokens = updates
        .token_events()
        .iter()
        .map(|event| (event.request_id(), event.token_id()))
        .collect();
    let (_, completions, failures) = updates.into_parts();
    assert!(failures.is_empty());
    (tokens, completions)
}

fn assert_nonzero_configuration_bounds() {
    for (field, mutate) in [
        (
            "max_waiting_requests",
            (|config: &mut SchedulerConfig| config.max_waiting_requests = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "max_waiting_prompt_tokens",
            (|config: &mut SchedulerConfig| config.max_waiting_prompt_tokens = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "max_active_sequences",
            (|config: &mut SchedulerConfig| config.max_active_sequences = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "max_sequence_tokens",
            (|config: &mut SchedulerConfig| config.max_sequence_tokens = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "iteration_token_budget",
            (|config: &mut SchedulerConfig| config.iteration_token_budget = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "max_prefill_chunk_tokens",
            (|config: &mut SchedulerConfig| config.max_prefill_chunk_tokens = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "max_promised_kv_blocks",
            (|config: &mut SchedulerConfig| config.max_promised_kv_blocks = 0)
                as fn(&mut SchedulerConfig),
        ),
        (
            "metrics_window_samples",
            (|config: &mut SchedulerConfig| config.metrics_window_samples = 0)
                as fn(&mut SchedulerConfig),
        ),
    ] {
        let mut config = compact_config();
        mutate(&mut config);
        assert!(matches!(
            config.validate(),
            Err(SchedulerError::InvalidConfiguration {
                field: actual,
                ..
            }) if actual == field
        ));
    }
}

#[test]
fn configuration_and_transport_contracts_fail_closed() {
    compact_config().validate().expect("valid compact config");
    assert_nonzero_configuration_bounds();

    let overflowing_capacity = SchedulerConfig {
        max_waiting_requests: usize::MAX,
        max_active_sequences: 1,
        ..compact_config()
    };
    let layout = KvLayout::checked(1, 16, 1, 8).expect("valid test KV layout");
    assert!(matches!(
        Scheduler::new(overflowing_capacity, layout),
        Err(SchedulerError::ArithmeticOverflow {
            field: "live request capacity"
        })
    ));

    let invalid_policy = SchedulerConfig {
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: Some(1),
        ..compact_config()
    };
    assert!(matches!(
        invalid_policy.validate(),
        Err(SchedulerError::InvalidConfiguration {
            field: "admission_timeout_ns",
            ..
        })
    ));

    assert!(RequestId::new(0).is_none());
    let iteration = IterationId::new(1).expect("non-zero iteration");
    assert!(matches!(
        IterationResult::from_version(2, iteration, Vec::new(), 0, 0),
        Err(SchedulerError::UnsupportedSchemaVersion {
            resource: "iteration result",
            expected: 1,
            actual: 2,
        })
    ));

    let duplicate = IterationOutput::new(OutputSlot::new(0), 7, false);
    assert!(matches!(
        IterationResult::new(iteration, vec![duplicate, duplicate], 1, 2),
        Err(SchedulerError::InvalidIterationResult {
            field: "outputs",
            ..
        })
    ));

    let request = RequestId::new(1).expect("non-zero request");
    assert!(matches!(
        WorkItem::new(request, WorkKind::Decode, vec![1, 2], 2, 0, None),
        Err(SchedulerError::InvalidPlan {
            field: "input_tokens",
            ..
        })
    ));
    assert!(matches!(
        IterationPlan::new(iteration, Vec::new(), Vec::new(), Vec::new()),
        Err(SchedulerError::InvalidPlan { field: "items", .. })
    ));

    let mut scheduler = scheduler_with(compact_config(), 16);
    assert!(matches!(
        scheduler.submit(RequestDescriptor::new(vec![1, 2], usize::MAX), 0),
        Err(SchedulerError::ArithmeticOverflow {
            field: "request maximum logical length"
        })
    ));
    assert_eq!(scheduler.active_sequence_count(), 0);
    assert_eq!(scheduler.waiting_request_count(), 0);
    assert_eq!(scheduler.promised_kv_blocks(), 0);
}

#[test]
fn oversized_caller_vec_capacity_does_not_change_bounded_accounting_or_plans() {
    let config = SchedulerConfig {
        max_active_sequences: 1,
        max_waiting_prompt_tokens: 4,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 16);

    let mut active_prompt = Vec::with_capacity(1 << 20);
    active_prompt.extend_from_slice(&[10, 11, 12]);
    let active_caller_capacity = active_prompt.capacity();
    assert!(active_caller_capacity > active_prompt.len());
    let active = scheduler
        .submit(RequestDescriptor::new(active_prompt, 2), 0)
        .expect("tiny active prompt with excess caller capacity");

    let mut waiting_prompt = Vec::with_capacity(1 << 20);
    waiting_prompt.extend_from_slice(&[20, 21]);
    let waiting_caller_capacity = waiting_prompt.capacity();
    assert!(waiting_caller_capacity > waiting_prompt.len());
    let waiting = scheduler
        .submit(RequestDescriptor::new(waiting_prompt, 1), 0)
        .expect("tiny waiting prompt with excess caller capacity");
    assert_eq!(waiting.state(), RequestState::Waiting);

    let active_snapshot = scheduler
        .request_snapshot(active.request_id())
        .expect("active snapshot");
    assert_eq!(active_snapshot.prompt_tokens(), 3);
    assert!(active_snapshot.retained_prompt_capacity_tokens() >= 3);
    assert!(active_snapshot.retained_prompt_capacity_tokens() < active_caller_capacity);
    let waiting_snapshot = scheduler
        .request_snapshot(waiting.request_id())
        .expect("waiting snapshot");
    assert_eq!(waiting_snapshot.prompt_tokens(), 2);
    assert!(waiting_snapshot.retained_prompt_capacity_tokens() >= 2);
    assert!(waiting_snapshot.retained_prompt_capacity_tokens() < waiting_caller_capacity);
    let gauges = scheduler
        .metrics_snapshot()
        .expect("capacity gauges")
        .gauges;
    assert_eq!(gauges.waiting_requests, 1);
    assert_eq!(gauges.waiting_prompt_tokens, 2);

    let plan = take_plan(&mut scheduler, 0);
    assert_eq!(plan.prefill_items().len(), 1);
    assert_eq!(plan.prefill_items()[0].request_id(), active.request_id());
    assert_eq!(plan.prefill_items()[0].input_tokens(), &[10, 11, 12]);
    scheduler
        .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 1)
        .expect("rollback capacity probe");
    let completions = scheduler.shutdown(1).expect("close capacity probe");
    assert_eq!(completions.len(), 2);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    assert_eq!(scheduler.promised_kv_blocks(), 0);
    assert_eq!(scheduler.pending_completion_count(), 0);
    assert!(scheduler.pop_pending_completion().is_none());
}

#[test]
fn mixed_prompts_are_fcfs_then_decode_first() {
    let config = SchedulerConfig {
        iteration_token_budget: 8,
        max_prefill_chunk_tokens: 4,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 16);
    let first = scheduler
        .submit(RequestDescriptor::new(prompt(10, 6), 2), 0)
        .expect("submit first");
    let second = scheduler
        .submit(RequestDescriptor::new(prompt(20, 2), 2), 0)
        .expect("submit second");
    let third = scheduler
        .submit(RequestDescriptor::new(prompt(30, 5), 2), 0)
        .expect("submit third");
    assert_eq!(first.state(), RequestState::Admitted);
    assert_eq!(second.state(), RequestState::Admitted);
    assert_eq!(third.state(), RequestState::Admitted);

    let plan = take_plan(&mut scheduler, 0);
    let prefill = plan.prefill_items();
    assert_eq!(prefill.len(), 3);
    assert_eq!(prefill[0].request_id(), first.request_id());
    assert_eq!(prefill[0].input_tokens(), prompt(10, 4));
    assert_eq!(prefill[1].request_id(), second.request_id());
    assert_eq!(prefill[1].input_tokens(), prompt(20, 2));
    assert_eq!(prefill[2].request_id(), third.request_id());
    assert_eq!(prefill[2].input_tokens(), prompt(30, 2));
    assert_eq!(plan.total_tokens(), 8);
    assert_eq!(plan.output_slots().len(), 1);

    let bases = HashMap::from([
        (first.request_id(), 10),
        (second.request_id(), 20),
        (third.request_id(), 30),
    ]);
    let mut generated = HashMap::new();
    settle_fake(&mut scheduler, &plan, &bases, &mut generated, 0);
    assert_eq!(
        scheduler.request_state(second.request_id()),
        Some(RequestState::Decoding)
    );

    let plan = take_plan(&mut scheduler, 1);
    assert_eq!(plan.decode_items().len(), 1);
    assert_eq!(plan.decode_items()[0].request_id(), second.request_id());
    assert_eq!(plan.decode_items()[0].input_tokens(), &[2_000]);
    assert_eq!(plan.prefill_items()[0].request_id(), first.request_id());
    assert!(plan.total_tokens() <= 8);
}

#[test]
fn aging_prevents_prefill_starvation_under_a_one_token_budget() {
    let config = SchedulerConfig {
        iteration_token_budget: 1,
        max_prefill_chunk_tokens: 1,
        aging_threshold_ns: 5,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 16);
    let decoder = scheduler
        .submit(RequestDescriptor::new(vec![10], 5), 0)
        .expect("submit decoder");
    let waiting_prefill = scheduler
        .submit(RequestDescriptor::new(prompt(20, 3), 1), 0)
        .expect("submit prefill");
    let bases = HashMap::from([
        (decoder.request_id(), 10),
        (waiting_prefill.request_id(), 20),
    ]);
    let mut generated = HashMap::new();

    let plan = take_plan(&mut scheduler, 0);
    assert_eq!(plan.prefill_items()[0].request_id(), decoder.request_id());
    settle_fake(&mut scheduler, &plan, &bases, &mut generated, 0);

    let plan = take_plan(&mut scheduler, 1);
    assert_eq!(plan.decode_items()[0].request_id(), decoder.request_id());
    settle_fake(&mut scheduler, &plan, &bases, &mut generated, 1);

    let plan = take_plan(&mut scheduler, 6);
    assert!(plan.decode_items().is_empty());
    assert_eq!(plan.prefill_items().len(), 1);
    assert_eq!(
        plan.prefill_items()[0].request_id(),
        waiting_prefill.request_id()
    );
    assert_eq!(plan.prefill_items()[0].input_tokens(), &[20]);
    scheduler
        .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 6)
        .expect("rollback aging probe");

    // The aging override alternates, preserving decode progress as well.
    let plan = take_plan(&mut scheduler, 7);
    assert_eq!(plan.decode_items()[0].request_id(), decoder.request_id());
}

#[test]
fn prefill_chunks_and_total_iteration_tokens_are_bounded() {
    let config = SchedulerConfig {
        iteration_token_budget: 5,
        max_prefill_chunk_tokens: 2,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 16);
    let first = scheduler
        .submit(RequestDescriptor::new(prompt(10, 5), 1), 0)
        .expect("first request");
    let second = scheduler
        .submit(RequestDescriptor::new(prompt(20, 5), 1), 0)
        .expect("second request");
    let bases = HashMap::from([(first.request_id(), 10), (second.request_id(), 20)]);
    let mut generated = HashMap::new();
    let mut chunks = Vec::new();
    let mut completions = Vec::new();

    for now_ns in 0..3 {
        let plan = take_plan(&mut scheduler, now_ns);
        assert!(plan.total_tokens() <= 5);
        let mut seen = HashSet::new();
        for item in plan.prefill_items() {
            assert!(item.input_tokens().len() <= 2);
            assert!(seen.insert(item.request_id()));
            chunks.push((item.request_id(), item.input_tokens().to_vec()));
        }
        completions.extend(settle_fake(&mut scheduler, &plan, &bases, &mut generated, now_ns).1);
    }
    assert_eq!(
        chunks,
        vec![
            (first.request_id(), vec![10, 11]),
            (second.request_id(), vec![20, 21]),
            (first.request_id(), vec![12, 13]),
            (second.request_id(), vec![22, 23]),
            (first.request_id(), vec![14]),
            (second.request_id(), vec![24]),
        ]
    );
    assert_eq!(completions.len(), 2);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
}

#[test]
fn kv_promises_gate_admission_before_physical_oom() {
    let reject_config = SchedulerConfig {
        max_active_sequences: 2,
        max_promised_kv_blocks: 3,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(reject_config, 3);
    let first = scheduler
        .submit(RequestDescriptor::new(prompt(10, 17), 1), 0)
        .expect("two-block request");
    assert_eq!(scheduler.promised_kv_blocks(), 2);
    assert!(matches!(
        scheduler.submit(RequestDescriptor::new(prompt(20, 17), 1), 0),
        Err(SchedulerError::KvCapacityExceeded {
            requested_blocks: 2,
            available_blocks: 1,
        })
    ));
    assert_eq!(scheduler.active_sequence_count(), 1);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    scheduler
        .cancel(first.request_id(), 0)
        .expect("cancel first");
    assert_eq!(scheduler.promised_kv_blocks(), 0);

    let wait_config = SchedulerConfig {
        max_active_sequences: 2,
        max_promised_kv_blocks: 3,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(wait_config, 3);
    let active = scheduler
        .submit(RequestDescriptor::new(prompt(10, 17), 1), 0)
        .expect("active request");
    let queued = scheduler
        .submit(RequestDescriptor::new(prompt(20, 17), 1), 0)
        .expect("bounded queue request");
    assert_eq!(queued.state(), RequestState::Waiting);
    scheduler
        .cancel(active.request_id(), 1)
        .expect("free promise");
    let plan = take_plan(&mut scheduler, 1);
    assert_eq!(plan.prefill_items()[0].request_id(), queued.request_id());
    assert_eq!(scheduler.promised_kv_blocks(), 2);
}

fn assert_idle_cancellation(config: &SchedulerConfig) {
    let mut scheduler = scheduler_with(config.clone(), 16);
    let admitted = scheduler
        .submit(RequestDescriptor::new(prompt(10, 3), 1), 0)
        .expect("admitted request");
    let waiting = scheduler
        .submit(RequestDescriptor::new(prompt(20, 3), 1), 0)
        .expect("waiting request");
    assert_eq!(waiting.state(), RequestState::Waiting);
    let outcome = scheduler
        .cancel(waiting.request_id(), 1)
        .expect("cancel waiting");
    assert_eq!(
        outcome.completion().expect("immediate completion").reason(),
        RequestFinishReason::Cancelled
    );
    assert_eq!(scheduler.waiting_request_count(), 0);
    scheduler
        .cancel(admitted.request_id(), 1)
        .expect("cancel admitted");
    assert_eq!(scheduler.active_sequence_count(), 0);
    assert_eq!(scheduler.promised_kv_blocks(), 0);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
}

fn assert_prefill_cancellation(config: &SchedulerConfig) {
    let mut scheduler = scheduler_with(config.clone(), 16);
    let request = scheduler
        .submit(RequestDescriptor::new(prompt(30, 3), 2), 0)
        .expect("prefill request");
    let plan = take_plan(&mut scheduler, 0);
    assert_eq!(
        scheduler.request_state(request.request_id()),
        Some(RequestState::Prefilling)
    );
    assert!(scheduler.pool_stats().allocated_block_count() > 0);
    let outcome = scheduler
        .cancel(request.request_id(), 1)
        .expect("defer prefill cancellation");
    assert!(outcome.deferred_until_iteration_settles());
    assert!(
        scheduler
            .request_snapshot(request.request_id())
            .expect("live deferred request")
            .cancellation_deferred()
    );
    let updates = scheduler
        .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 1)
        .expect("rollback cancelled prefill");
    assert_eq!(updates.completions().len(), 1);
    assert_eq!(
        updates.completions()[0].reason(),
        RequestFinishReason::Cancelled
    );
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
}

fn assert_decode_cancellation(config: &SchedulerConfig) {
    let mut scheduler = scheduler_with(config.clone(), 16);
    let request = scheduler
        .submit(RequestDescriptor::new(vec![40], 3), 0)
        .expect("decode request");
    let bases = HashMap::from([(request.request_id(), 40)]);
    let mut generated = HashMap::new();
    let plan = take_plan(&mut scheduler, 0);
    let first_updates = settle_fake(&mut scheduler, &plan, &bases, &mut generated, 0);
    assert_eq!(first_updates.0, vec![(request.request_id(), 4_000)]);
    assert_eq!(
        scheduler.request_state(request.request_id()),
        Some(RequestState::Decoding)
    );
    let plan = take_plan(&mut scheduler, 1);
    assert_eq!(plan.decode_items().len(), 1);
    assert!(
        scheduler
            .cancel(request.request_id(), 1)
            .expect("defer decode cancellation")
            .deferred_until_iteration_settles()
    );
    let result = fake_result(&plan, &bases, &mut generated, 2, 0);
    let updates = scheduler
        .complete_iteration(&result, 1)
        .expect("settle cancelled decode");
    assert!(updates.token_events().is_empty());
    assert_eq!(updates.completions()[0].generated_token_ids(), &[4_000]);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let repeated = scheduler
        .cancel(request.request_id(), 2)
        .expect("terminal cancel is idempotent");
    assert!(repeated.already_terminal());
}

fn assert_quiesced_failure_cleanup(config: SchedulerConfig) {
    let mut scheduler = scheduler_with(config, 16);
    let failed = scheduler
        .submit(RequestDescriptor::new(prompt(50, 3), 2), 0)
        .expect("failure request");
    let plan = take_plan(&mut scheduler, 0);
    let cancellation = scheduler
        .cancel(failed.request_id(), 1)
        .expect("defer cancellation before device failure");
    assert!(cancellation.deferred_until_iteration_settles());
    assert_eq!(
        scheduler.request_state(failed.request_id()),
        Some(RequestState::Prefilling)
    );
    let updates = scheduler
        .abort_iteration(
            plan.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            1,
        )
        .expect("quiesced failure cleanup");
    assert_eq!(
        updates.completions()[0].reason(),
        RequestFinishReason::ExecutorFailure
    );
    assert_eq!(
        scheduler.request_state(failed.request_id()),
        Some(RequestState::Failed)
    );
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    assert_eq!(scheduler.active_sequence_count(), 0);
    assert_eq!(scheduler.promised_kv_blocks(), 0);
}

#[test]
fn cancellation_is_safe_in_every_request_state() {
    let config = SchedulerConfig {
        max_active_sequences: 1,
        ..compact_config()
    };
    assert_idle_cancellation(&config);
    assert_prefill_cancellation(&config);
    assert_decode_cancellation(&config);
    assert_quiesced_failure_cleanup(config);
}

#[test]
fn inflight_shutdown_requires_one_of_the_two_explicit_abort_dispositions() {
    let config = SchedulerConfig {
        max_active_sequences: 1,
        ..compact_config()
    };

    let mut rollback_scheduler = scheduler_with(config.clone(), 16);
    let rollback_request = rollback_scheduler
        .submit(RequestDescriptor::new(prompt(60, 3), 2), 0)
        .expect("rollback request");
    let rollback_plan = take_plan(&mut rollback_scheduler, 0);
    let reserved_blocks = rollback_scheduler.pool_stats().allocated_block_count();
    assert!(reserved_blocks > 0);
    assert!(matches!(
        rollback_scheduler.shutdown(0),
        Err(SchedulerError::IterationInFlight { iteration_id })
            if iteration_id == rollback_plan.iteration_id()
    ));
    assert_eq!(
        rollback_scheduler.inflight_iteration_id(),
        Some(rollback_plan.iteration_id())
    );
    assert_eq!(
        rollback_scheduler.pool_stats().allocated_block_count(),
        reserved_blocks
    );
    assert_eq!(rollback_scheduler.pending_completion_count(), 0);

    let rollback_updates = rollback_scheduler
        .abort_iteration(
            rollback_plan.iteration_id(),
            ExecutionAbort::NotDispatched,
            1,
        )
        .expect("rollback undispatched plan");
    assert!(rollback_updates.completions().is_empty());
    assert_eq!(rollback_scheduler.inflight_iteration_id(), None);
    assert_eq!(
        rollback_scheduler.request_state(rollback_request.request_id()),
        Some(RequestState::Admitted)
    );
    assert_eq!(rollback_scheduler.pool_stats().allocated_block_count(), 0);
    assert_eq!(rollback_scheduler.active_sequence_count(), 1);
    assert_eq!(rollback_scheduler.promised_kv_blocks(), 1);
    let recovery = rollback_scheduler
        .shutdown(1)
        .expect("shutdown after rollback recovery");
    assert_eq!(recovery.len(), 1);
    assert_eq!(recovery[0].reason(), RequestFinishReason::Cancelled);
    assert_eq!(rollback_scheduler.active_sequence_count(), 0);
    assert_eq!(rollback_scheduler.promised_kv_blocks(), 0);
    assert_eq!(rollback_scheduler.pending_completion_count(), 0);

    let mut poison_scheduler = scheduler_with(config, 16);
    let poison_request = poison_scheduler
        .submit(RequestDescriptor::new(prompt(70, 3), 2), 0)
        .expect("poison request");
    let poison_plan = take_plan(&mut poison_scheduler, 0);
    assert!(matches!(
        poison_scheduler.shutdown(0),
        Err(SchedulerError::IterationInFlight { iteration_id })
            if iteration_id == poison_plan.iteration_id()
    ));
    let poison_updates = poison_scheduler
        .abort_iteration(
            poison_plan.iteration_id(),
            ExecutionAbort::DeviceQuiescedMutationUnknown,
            1,
        )
        .expect("poison quiesced plan");
    assert_eq!(poison_updates.completions().len(), 1);
    assert_eq!(
        poison_updates.completions()[0].reason(),
        RequestFinishReason::ExecutorFailure
    );
    assert_eq!(
        poison_scheduler.request_state(poison_request.request_id()),
        Some(RequestState::Failed)
    );
    assert_eq!(poison_scheduler.pool_stats().allocated_block_count(), 0);
    assert_eq!(poison_scheduler.active_sequence_count(), 0);
    assert_eq!(poison_scheduler.promised_kv_blocks(), 0);
    assert_eq!(poison_scheduler.pending_completion_count(), 0);
    assert!(poison_scheduler.pop_pending_completion().is_none());
    assert!(
        poison_scheduler
            .shutdown(1)
            .expect("empty shutdown")
            .is_empty()
    );
}

#[test]
fn consuming_close_requires_and_applies_an_inflight_disposition() {
    let config = SchedulerConfig {
        max_active_sequences: 1,
        ..compact_config()
    };

    let mut rollback_scheduler = scheduler_with(config.clone(), 16);
    let rollback_request = rollback_scheduler
        .submit(RequestDescriptor::new(prompt(80, 3), 2), 0)
        .expect("close rollback request");
    let rollback_plan = take_plan(&mut rollback_scheduler, 0);
    let close_failure = rollback_scheduler
        .close(0, None)
        .expect_err("in-flight close needs a disposition");
    assert!(matches!(
        close_failure.error(),
        SchedulerError::CloseDispositionRequired { iteration_id }
            if *iteration_id == rollback_plan.iteration_id()
    ));
    assert!(close_failure.settlement_failures().is_empty());
    let (_, mut rollback_scheduler, close_failures) = close_failure.into_parts();
    assert!(close_failures.is_empty());
    assert!(!rollback_scheduler.is_accepting());
    assert!(matches!(
        rollback_scheduler.submit(RequestDescriptor::new(vec![99], 1), 0),
        Err(SchedulerError::SchedulerClosed)
    ));
    let rollback_close = rollback_scheduler
        .close(1, Some(ExecutionAbort::NotDispatched))
        .expect("retry close with rollback disposition");
    assert!(rollback_close.settlement_failures().is_empty());
    assert_eq!(rollback_close.completions().len(), 1);
    assert_eq!(
        rollback_close.completions()[0].request_id(),
        rollback_request.request_id()
    );
    assert_eq!(
        rollback_close.completions()[0].reason(),
        RequestFinishReason::Cancelled
    );

    let mut poison_scheduler = scheduler_with(config, 16);
    let poison_request = poison_scheduler
        .submit(RequestDescriptor::new(prompt(90, 3), 2), 0)
        .expect("close poison request");
    take_plan(&mut poison_scheduler, 0);
    let poison_close = poison_scheduler
        .close(1, Some(ExecutionAbort::DeviceQuiescedMutationUnknown))
        .expect("close with quiesced poison disposition");
    assert!(poison_close.settlement_failures().is_empty());
    assert_eq!(poison_close.completions().len(), 1);
    assert_eq!(
        poison_close.completions()[0].request_id(),
        poison_request.request_id()
    );
    assert_eq!(
        poison_close.completions()[0].reason(),
        RequestFinishReason::ExecutorFailure
    );
}

fn run_oracle_workload(
    requests: &[(u32, usize, usize)],
) -> (
    HashMap<u32, Vec<u32>>,
    riley_scheduler::SchedulerMetricsSnapshot,
) {
    let config = SchedulerConfig {
        max_active_sequences: requests.len().max(1),
        max_waiting_requests: requests.len().max(1),
        iteration_token_budget: 7,
        max_prefill_chunk_tokens: 3,
        max_promised_kv_blocks: 64,
        admission_timeout_ns: None,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 64);
    let mut bases = HashMap::new();
    for &(base, prompt_len, max_new_tokens) in requests {
        let submission = scheduler
            .submit(
                RequestDescriptor::new(prompt(base, prompt_len), max_new_tokens),
                0,
            )
            .expect("oracle workload submission");
        bases.insert(submission.request_id(), base);
    }
    let mut generated_counts = HashMap::new();
    let mut completed_by_base = HashMap::new();
    for now_ns in 0..256 {
        let planning = scheduler.plan_iteration(now_ns).expect("oracle plan");
        let (plan, timed_out) = planning.into_parts();
        assert!(timed_out.is_empty());
        let Some(plan) = plan else {
            break;
        };
        let (_, completions) =
            settle_fake(&mut scheduler, &plan, &bases, &mut generated_counts, now_ns);
        for completion in completions {
            completed_by_base.insert(
                bases[&completion.request_id()],
                completion.generated_token_ids().to_vec(),
            );
        }
    }
    assert_eq!(completed_by_base.len(), requests.len());
    assert_eq!(scheduler.active_sequence_count(), 0);
    assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
    let metrics = scheduler.metrics_snapshot().expect("metrics snapshot");
    (completed_by_base, metrics)
}

#[test]
fn output_slots_match_independent_request_oracles() {
    let requests = [(11, 1, 3), (22, 5, 2), (33, 2, 4), (44, 7, 1)];
    let (batched, batched_metrics) = run_oracle_workload(&requests);
    assert!(batched_metrics.iteration_batch_size.p95().is_some());
    assert!(batched_metrics.iteration_batch_size.p95().unwrap() > 1);

    for request in requests {
        let (independent, _) = run_oracle_workload(&[request]);
        assert_eq!(batched[&request.0], independent[&request.0]);
        let expected = (0..request.2)
            .map(|index| request.0 * 100 + u32::try_from(index).expect("small index"))
            .collect::<Vec<_>>();
        assert_eq!(batched[&request.0], expected);
    }
}

#[test]
fn malformed_stale_and_replayed_feedback_never_mutates_state() {
    let mut scheduler = scheduler_with(compact_config(), 16);
    let request = scheduler
        .submit(RequestDescriptor::new(vec![10], 2), 0)
        .expect("request");
    let plan = take_plan(&mut scheduler, 0);
    let before_request = scheduler
        .request_snapshot(request.request_id())
        .expect("live snapshot");
    let before_pool = scheduler.pool_stats();
    let before_metrics = scheduler.metrics_snapshot().expect("metrics");

    let stale_id = IterationId::new(plan.iteration_id().get() + 1).expect("stale ID");
    let stale = IterationResult::new(
        stale_id,
        vec![IterationOutput::new(OutputSlot::new(0), 99, false)],
        1,
        2,
    )
    .expect("well-formed stale result");
    assert!(matches!(
        scheduler.complete_iteration(&stale, 1),
        Err(SchedulerError::UnexpectedIteration { .. })
    ));
    assert_eq!(
        scheduler.request_snapshot(request.request_id()),
        Some(before_request)
    );
    assert_eq!(scheduler.pool_stats(), before_pool);
    assert_eq!(
        scheduler.metrics_snapshot().expect("metrics"),
        before_metrics
    );
    assert_eq!(scheduler.inflight_iteration_id(), Some(plan.iteration_id()));

    let missing = IterationResult::new(plan.iteration_id(), Vec::new(), 1, 2)
        .expect("structurally valid empty result");
    assert!(matches!(
        scheduler.complete_iteration(&missing, 1),
        Err(SchedulerError::InvalidIterationResult {
            field: "outputs",
            ..
        })
    ));
    let unplanned = IterationResult::new(
        plan.iteration_id(),
        vec![IterationOutput::new(OutputSlot::new(9), 99, false)],
        1,
        2,
    )
    .expect("structurally valid unplanned slot");
    assert!(matches!(
        scheduler.complete_iteration(&unplanned, 1),
        Err(SchedulerError::InvalidIterationResult {
            field: "outputs",
            ..
        })
    ));
    assert_eq!(
        scheduler.request_snapshot(request.request_id()),
        Some(before_request)
    );
    assert_eq!(scheduler.pool_stats(), before_pool);
    assert_eq!(
        scheduler.metrics_snapshot().expect("metrics"),
        before_metrics
    );

    let bases = HashMap::from([(request.request_id(), 10)]);
    let mut generated = HashMap::new();
    let valid = fake_result(&plan, &bases, &mut generated, 3, 4);
    scheduler
        .complete_iteration(&valid, 1)
        .expect("valid result");
    let committed = scheduler
        .request_snapshot(request.request_id())
        .expect("decoding snapshot");
    let committed_pool = scheduler.pool_stats();
    let committed_metrics = scheduler.metrics_snapshot().expect("metrics");
    assert!(matches!(
        scheduler.complete_iteration(&valid, 2),
        Err(SchedulerError::NoIterationInFlight)
    ));
    assert_eq!(
        scheduler.request_snapshot(request.request_id()),
        Some(committed)
    );
    assert_eq!(scheduler.pool_stats(), committed_pool);
    assert_eq!(
        scheduler.metrics_snapshot().expect("metrics"),
        committed_metrics
    );
}

#[test]
fn admission_timeout_immediate_reject_and_queue_bounds_are_explicit() {
    let timeout_config = SchedulerConfig {
        max_active_sequences: 1,
        admission_timeout_ns: Some(10),
        ..compact_config()
    };
    let mut scheduler = scheduler_with(timeout_config, 16);
    let active = scheduler
        .submit(RequestDescriptor::new(prompt(10, 3), 2), 0)
        .expect("active request");
    let queued = scheduler
        .submit(RequestDescriptor::new(prompt(20, 3), 2), 0)
        .expect("queued request");
    assert_eq!(queued.state(), RequestState::Waiting);
    let planning = scheduler.plan_iteration(10).expect("timeout tick");
    assert_eq!(planning.completions().len(), 1);
    assert_eq!(planning.completions()[0].request_id(), queued.request_id());
    assert_eq!(
        planning.completions()[0].reason(),
        RequestFinishReason::AdmissionTimeout
    );
    assert_eq!(
        scheduler.request_state(queued.request_id()),
        Some(RequestState::Failed)
    );
    let plan = planning.into_parts().0.expect("active request still plans");
    assert_eq!(plan.prefill_items()[0].request_id(), active.request_id());
    scheduler
        .abort_iteration(plan.iteration_id(), ExecutionAbort::NotDispatched, 10)
        .expect("clean timeout test plan");
    let metrics = scheduler.metrics_snapshot().expect("timeout metrics");
    assert_eq!(metrics.admission_timeouts, 1);
    assert_eq!(metrics.requests_failed, 1);

    let reject_config = SchedulerConfig {
        max_active_sequences: 1,
        overload_policy: OverloadPolicy::RejectImmediately,
        admission_timeout_ns: None,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(reject_config, 16);
    scheduler
        .submit(RequestDescriptor::new(vec![1], 2), 0)
        .expect("fill active slot");
    assert!(matches!(
        scheduler.submit(RequestDescriptor::new(vec![2], 2), 0),
        Err(SchedulerError::ActiveSequenceLimit { limit: 1 })
    ));
    assert_eq!(
        scheduler
            .metrics_snapshot()
            .expect("rejection metrics")
            .requests_rejected,
        1
    );

    let queue_config = SchedulerConfig {
        max_active_sequences: 1,
        max_waiting_requests: 3,
        max_waiting_prompt_tokens: 5,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(queue_config, 16);
    scheduler
        .submit(RequestDescriptor::new(vec![1], 2), 0)
        .expect("fill active slot");
    scheduler
        .submit(RequestDescriptor::new(prompt(10, 2), 1), 0)
        .expect("queue two tokens");
    scheduler
        .submit(RequestDescriptor::new(prompt(20, 3), 1), 0)
        .expect("queue three tokens");
    assert!(matches!(
        scheduler.submit(RequestDescriptor::new(vec![30], 1), 0),
        Err(SchedulerError::WaitingTokenLimit {
            limit: 5,
            requested: 6,
        })
    ));
    let gauges = scheduler.metrics_snapshot().expect("queue gauges").gauges;
    assert_eq!(gauges.waiting_requests, 2);
    assert_eq!(gauges.waiting_prompt_tokens, 5);

    let count_config = SchedulerConfig {
        max_active_sequences: 1,
        max_waiting_requests: 2,
        max_waiting_prompt_tokens: 64,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(count_config, 16);
    for token in 1..=3 {
        scheduler
            .submit(RequestDescriptor::new(vec![token], 2), 0)
            .expect("active plus two queued requests");
    }
    assert!(matches!(
        scheduler.submit(RequestDescriptor::new(vec![4], 2), 0),
        Err(SchedulerError::WaitingQueueFull { limit: 2 })
    ));
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
}

struct TraceSimulation {
    scheduler: Scheduler,
    random: Lcg,
    trace: Vec<String>,
    known_ids: Vec<RequestId>,
    bases: HashMap<RequestId, u32>,
    generated_counts: HashMap<RequestId, usize>,
}

impl TraceSimulation {
    fn new(seed: u64) -> Self {
        let config = SchedulerConfig {
            max_waiting_requests: 6,
            max_waiting_prompt_tokens: 64,
            max_active_sequences: 3,
            max_sequence_tokens: 64,
            iteration_token_budget: 5,
            max_prefill_chunk_tokens: 3,
            aging_threshold_ns: 7,
            admission_timeout_ns: None,
            max_promised_kv_blocks: 32,
            metrics_window_samples: 8,
            ..compact_config()
        };
        Self {
            scheduler: scheduler_with(config, 32),
            random: Lcg(seed),
            trace: Vec::new(),
            known_ids: Vec::new(),
            bases: HashMap::new(),
            generated_counts: HashMap::new(),
        }
    }

    fn apply_host_event(&mut self, now_ns: u64) {
        let event = self.random.next() % 100;
        if event < 55 {
            let base = 100 + u32::try_from(self.random.next() % 500).expect("bounded base");
            let prompt_len = 1 + usize::try_from(self.random.next() % 6).expect("bounded prompt");
            let max_new = 1 + usize::try_from(self.random.next() % 4).expect("bounded output");
            match self.scheduler.submit(
                RequestDescriptor::new(prompt(base, prompt_len), max_new),
                now_ns,
            ) {
                Ok(submission) => {
                    self.known_ids.push(submission.request_id());
                    self.bases.insert(submission.request_id(), base);
                    self.trace.push(format!(
                        "submit:{}:{:?}",
                        submission.request_id().get(),
                        submission.state()
                    ));
                }
                Err(error) => self.trace.push(format!("submit-error:{error:?}")),
            }
        } else if event < 72 {
            let live = self
                .known_ids
                .iter()
                .copied()
                .filter(|request_id| self.scheduler.request_snapshot(*request_id).is_some())
                .collect::<Vec<_>>();
            if !live.is_empty() {
                let index =
                    usize::try_from(self.random.next()).expect("U64 fits usize") % live.len();
                let request_id = live[index];
                let outcome = self
                    .scheduler
                    .cancel(request_id, now_ns)
                    .expect("random idle cancellation");
                self.trace.push(format!(
                    "cancel:{}:{}",
                    request_id.get(),
                    outcome.deferred_until_iteration_settles()
                ));
            }
        }
    }

    fn run_planning_tick(&mut self, now_ns: u64) {
        let planning = self
            .scheduler
            .plan_iteration(now_ns)
            .expect("deterministic planning tick");
        let (plan, completions) = planning.into_parts();
        for completion in completions {
            self.trace.push(format!(
                "planning-complete:{}:{:?}",
                completion.request_id().get(),
                completion.reason()
            ));
        }
        if let Some(plan) = plan {
            let signature = plan
                .prefill_items()
                .iter()
                .chain(plan.decode_items().iter())
                .map(|item| {
                    format!(
                        "{}:{:?}:{}:{:?}",
                        item.request_id().get(),
                        item.kind(),
                        item.input_tokens().len(),
                        item.output_slot().map(OutputSlot::get)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            self.trace
                .push(format!("plan:{}:{signature}", plan.iteration_id().get()));
            let (tokens, completions) = settle_fake(
                &mut self.scheduler,
                &plan,
                &self.bases,
                &mut self.generated_counts,
                now_ns,
            );
            for (request_id, token_id) in tokens {
                self.trace
                    .push(format!("token:{}:{token_id}", request_id.get()));
            }
            for completion in completions {
                self.trace.push(format!(
                    "complete:{}:{:?}",
                    completion.request_id().get(),
                    completion.reason()
                ));
            }
        }
    }

    fn finish(mut self, now_ns: u64) -> Vec<String> {
        for completion in self
            .scheduler
            .shutdown(now_ns)
            .expect("deterministic shutdown")
        {
            self.trace.push(format!(
                "shutdown:{}:{:?}",
                completion.request_id().get(),
                completion.reason()
            ));
        }
        let metrics = self.scheduler.metrics_snapshot().expect("trace metrics");
        self.trace.push(format!(
            "metrics:{}:{}:{}:{}:{}:{}:{}:{}",
            metrics.requests_submitted,
            metrics.requests_finished,
            metrics.requests_failed,
            metrics.requests_rejected,
            metrics.requests_cancelled,
            metrics.iterations_completed,
            metrics.prefill_tokens,
            metrics.decode_tokens
        ));
        self.trace
    }
}

fn deterministic_event_trace(seed: u64) -> Vec<String> {
    let mut simulation = TraceSimulation::new(seed);
    for now_ns in 0..160_u64 {
        simulation.apply_host_event(now_ns);
        simulation.run_planning_tick(now_ns);
    }
    simulation.finish(160)
}

#[test]
fn fixed_lcg_simulation_is_reproducible() {
    let first = deterministic_event_trace(0x5eed_cafe_f00d_beef);
    let second = deterministic_event_trace(0x5eed_cafe_f00d_beef);
    assert_eq!(first, second);
    assert!(
        first.len() > 100,
        "trace should exercise a sustained workload"
    );
}

#[test]
fn long_run_reclaims_kv_and_bounds_tombstones_and_metric_windows() {
    let config = SchedulerConfig {
        max_waiting_requests: 2,
        max_active_sequences: 1,
        iteration_token_budget: 4,
        max_prefill_chunk_tokens: 2,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 16,
        metrics_window_samples: 4,
        ..compact_config()
    };
    let mut scheduler = scheduler_with(config, 16);
    let mut bases = HashMap::new();
    let mut generated_counts = HashMap::new();
    let mut now_ns = 0_u64;
    let mut first_request = None;
    let mut last_request = None;
    let mut recent_gpu_samples = Vec::new();

    for index in 0..128_u32 {
        let base = 100 + index;
        let submission = scheduler
            .submit(
                RequestDescriptor::new(
                    prompt(base, 1 + usize::try_from(index % 7).expect("small prompt")),
                    1 + usize::try_from(index % 4).expect("small output"),
                ),
                now_ns,
            )
            .expect("long-run submission");
        first_request.get_or_insert(submission.request_id());
        last_request = Some(submission.request_id());
        bases.insert(submission.request_id(), base);

        loop {
            let plan = take_plan(&mut scheduler, now_ns);
            recent_gpu_samples.push(now_ns + 1);
            if recent_gpu_samples.len() > 4 {
                recent_gpu_samples.remove(0);
            }
            let (_, completions) =
                settle_fake(&mut scheduler, &plan, &bases, &mut generated_counts, now_ns);
            now_ns += 1;
            if !completions.is_empty() {
                assert_eq!(completions.len(), 1);
                assert_eq!(completions[0].request_id(), submission.request_id());
                break;
            }
        }
        assert_eq!(scheduler.pool_stats().allocated_block_count(), 0);
        assert_eq!(scheduler.active_sequence_count(), 0);
        assert_eq!(scheduler.promised_kv_blocks(), 0);
    }

    let metrics = scheduler.metrics_snapshot().expect("long-run metrics");
    assert_eq!(metrics.requests_submitted, 128);
    assert_eq!(metrics.requests_finished, 128);
    assert_eq!(metrics.requests_rejected, 0);
    assert_eq!(metrics.requests_cancelled, 0);
    assert_eq!(metrics.gauges.waiting_requests, 0);
    assert_eq!(metrics.gauges.active_sequences, 0);
    assert_eq!(metrics.gauges.promised_kv_blocks, 0);
    assert_eq!(metrics.gauges.allocated_kv_blocks, 0);
    assert_eq!(metrics.gauges.outstanding_iterations, 0);
    assert_eq!(metrics.gauges.retained_terminal_requests, 3);
    assert_eq!(metrics.queue_wait_ns.sample_count(), 4);
    assert_eq!(metrics.queue_wait_ns.capacity(), 4);
    assert_eq!(metrics.queue_wait_ns.p95(), Some(0));
    assert_eq!(metrics.scheduler_cpu_ns.sample_count(), 4);
    assert_eq!(metrics.gpu_execution_ns.sample_count(), 4);
    assert_eq!(metrics.gpu_execution_ns.capacity(), 4);
    assert_eq!(
        metrics.gpu_execution_ns.p95(),
        recent_gpu_samples.iter().max().copied()
    );
    assert_eq!(metrics.gpu_idle_gap_ns.sample_count(), 4);
    assert_eq!(metrics.iteration_batch_size.sample_count(), 4);
    assert_eq!(metrics.batched_tokens.sample_count(), 4);
    assert_eq!(
        scheduler.request_state(first_request.expect("first ID")),
        None,
        "old tombstones must be evicted at the fixed capacity"
    );
    assert_eq!(
        scheduler.request_state(last_request.expect("last ID")),
        Some(RequestState::Finished)
    );
}
