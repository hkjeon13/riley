//! Model-free coverage for the prefill plateau policy.

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, OverloadPolicy, PrefillPlateauPolicy,
    RequestDescriptor, RequestId, Scheduler, SchedulerConfig, SchedulerError,
};

fn scheduler(policy: Option<PrefillPlateauPolicy>) -> Scheduler {
    let config = SchedulerConfig {
        max_waiting_requests: 8,
        max_waiting_prompt_tokens: 256,
        max_active_sequences: 8,
        max_sequence_tokens: 64,
        iteration_token_budget: 16,
        max_prefill_chunk_tokens: 16,
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 32,
        metrics_window_samples: 4,
    };
    let layout = KvLayout::checked(1, 32, 1, 8).expect("valid test KV layout");
    let mut scheduler = Scheduler::new(config, layout).expect("valid test scheduler");
    scheduler
        .set_prefill_plateau_policy(policy)
        .expect("valid plateau policy");
    scheduler
}

fn policy(max_deferred_tokens: usize, min_decode_rows: usize) -> PrefillPlateauPolicy {
    PrefillPlateauPolicy {
        boundaries: vec![8],
        max_deferred_tokens,
        min_decode_rows,
    }
}

fn submit(scheduler: &mut Scheduler, prompt_tokens: usize, now_ns: u64) -> RequestId {
    let prompt = (0..prompt_tokens)
        .map(|token| u32::try_from(token).expect("small token"))
        .collect();
    scheduler
        .submit(RequestDescriptor::new(prompt, 32), now_ns)
        .expect("submit request")
        .request_id()
}

fn plan(scheduler: &mut Scheduler, now_ns: u64) -> IterationPlan {
    scheduler
        .plan_iteration(now_ns)
        .expect("plan iteration")
        .into_parts()
        .0
        .expect("planned work")
}

fn settle(scheduler: &mut Scheduler, plan: &IterationPlan, now_ns: u64) {
    let outputs = plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items())
        .filter_map(|item| item.output_slot())
        .map(|slot| IterationOutput::new(slot, 7, false))
        .collect();
    let result = IterationResult::new(plan.iteration_id(), outputs, 1, 0).expect("fake result");
    let updates = scheduler
        .complete_iteration(&result, now_ns)
        .expect("complete iteration");
    assert!(updates.settlement_failures().is_empty());
}

fn prefill_tokens(plan: &IterationPlan, request_id: RequestId) -> usize {
    plan.prefill_items()
        .iter()
        .filter(|item| item.request_id() == request_id)
        .map(|item| item.input_tokens().len())
        .sum()
}

/// Two decoding requests, then prompts that push the iteration past eight rows.
fn two_decoders(policy: Option<PrefillPlateauPolicy>) -> Scheduler {
    let mut scheduler = scheduler(policy);
    submit(&mut scheduler, 1, 1);
    submit(&mut scheduler, 1, 1);
    let first = plan(&mut scheduler, 2);
    assert_eq!(first.total_tokens(), 2);
    settle(&mut scheduler, &first, 3);
    scheduler
}

#[test]
fn a_small_overflow_past_the_boundary_waits_one_iteration() {
    let mut scheduler = two_decoders(Some(policy(4, 2)));
    let long = submit(&mut scheduler, 9, 4);

    let trimmed = plan(&mut scheduler, 5);
    assert_eq!(trimmed.decode_items().len(), 2);
    assert_eq!(prefill_tokens(&trimmed, long), 6);
    assert_eq!(trimmed.total_tokens(), 8);
    settle(&mut scheduler, &trimmed, 6);

    let rest = plan(&mut scheduler, 7);
    assert_eq!(prefill_tokens(&rest, long), 3);
    assert_eq!(rest.total_tokens(), 5);
}

#[test]
fn a_request_is_trimmed_at_most_once() {
    let mut scheduler = two_decoders(Some(policy(8, 2)));
    let first = submit(&mut scheduler, 9, 4);
    let trimmed = plan(&mut scheduler, 5);
    assert_eq!(prefill_tokens(&trimmed, first), 6);
    settle(&mut scheduler, &trimmed, 6);

    // The carried remainder goes first; the newer prompt absorbs the overflow.
    let second = submit(&mut scheduler, 9, 7);
    let carried = plan(&mut scheduler, 8);
    assert_eq!(prefill_tokens(&carried, first), 3);
    assert_eq!(prefill_tokens(&carried, second), 3);
    assert_eq!(carried.total_tokens(), 8);
    settle(&mut scheduler, &carried, 9);

    // Three decoders plus the remaining six tokens end one row past the
    // boundary, but that prompt has already waited once.
    let crossing = plan(&mut scheduler, 10);
    assert_eq!(crossing.decode_items().len(), 3);
    assert_eq!(prefill_tokens(&crossing, second), 6);
    assert_eq!(crossing.total_tokens(), 9);
}

#[test]
fn a_large_overflow_or_too_few_decoders_keeps_the_default_selection() {
    let mut large = two_decoders(Some(policy(2, 2)));
    let long = submit(&mut large, 9, 4);
    assert_eq!(prefill_tokens(&plan(&mut large, 5), long), 9);

    let mut few = two_decoders(Some(policy(4, 3)));
    let long = submit(&mut few, 9, 4);
    assert_eq!(prefill_tokens(&plan(&mut few, 5), long), 9);

    let mut off = two_decoders(None);
    let long = submit(&mut off, 9, 4);
    assert_eq!(prefill_tokens(&plan(&mut off, 5), long), 9);
}

#[test]
fn an_invalid_policy_is_rejected() {
    let mut scheduler = scheduler(None);
    for policy in [
        PrefillPlateauPolicy {
            boundaries: vec![],
            max_deferred_tokens: 4,
            min_decode_rows: 0,
        },
        PrefillPlateauPolicy {
            boundaries: vec![8, 8],
            max_deferred_tokens: 4,
            min_decode_rows: 0,
        },
        PrefillPlateauPolicy {
            boundaries: vec![16],
            max_deferred_tokens: 4,
            min_decode_rows: 0,
        },
        PrefillPlateauPolicy {
            boundaries: vec![8],
            max_deferred_tokens: 0,
            min_decode_rows: 0,
        },
    ] {
        assert!(matches!(
            scheduler.set_prefill_plateau_policy(Some(policy)),
            Err(SchedulerError::InvalidConfiguration { .. })
        ));
    }
}
