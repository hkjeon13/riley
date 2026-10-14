//! Model-free coverage for the prefill coalescing policy.

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, OverloadPolicy, PrefillCoalescingPolicy,
    RequestDescriptor, RequestId, Scheduler, SchedulerConfig, SchedulerError,
};

fn scheduler(policy: Option<PrefillCoalescingPolicy>) -> Scheduler {
    let config = SchedulerConfig {
        max_waiting_requests: 8,
        max_waiting_prompt_tokens: 256,
        max_active_sequences: 8,
        max_sequence_tokens: 64,
        iteration_token_budget: 16,
        max_prefill_chunk_tokens: 4,
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 32,
        metrics_window_samples: 4,
    };
    let layout = KvLayout::checked(1, 32, 1, 8).expect("valid test KV layout");
    let mut scheduler = Scheduler::new(config, layout).expect("valid test scheduler");
    scheduler
        .set_prefill_coalescing_policy(policy)
        .expect("valid coalescing policy");
    scheduler
}

fn policy(min_tokens: usize, max_wait_ns: u64, min_decode_rows: usize) -> PrefillCoalescingPolicy {
    PrefillCoalescingPolicy {
        min_tokens,
        max_wait_ns,
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

/// Two decoding requests; later prompts meet two decode rows.
fn two_decoders(policy: Option<PrefillCoalescingPolicy>) -> Scheduler {
    let mut scheduler = scheduler(policy);
    submit(&mut scheduler, 1, 1);
    submit(&mut scheduler, 1, 1);
    let first = plan(&mut scheduler, 2);
    assert_eq!(first.total_tokens(), 2);
    settle(&mut scheduler, &first, 3);
    scheduler
}

#[test]
fn without_the_policy_a_prompt_starts_at_once() {
    let mut scheduler = two_decoders(None);
    let prompt = submit(&mut scheduler, 3, 4);
    let step = plan(&mut scheduler, 5);
    assert_eq!(step.decode_items().len(), 2);
    assert_eq!(prefill_tokens(&step, prompt), 3);
}

#[test]
fn a_lone_prompt_waits_until_enough_tokens_are_ready() {
    let mut scheduler = two_decoders(Some(policy(5, 1_000, 2)));
    let first = submit(&mut scheduler, 3, 4);

    let held = plan(&mut scheduler, 5);
    assert_eq!(held.decode_items().len(), 2);
    assert!(held.prefill_items().is_empty());
    settle(&mut scheduler, &held, 6);

    // A second prompt brings the ready tokens to the threshold: both start.
    let second = submit(&mut scheduler, 2, 7);
    let together = plan(&mut scheduler, 8);
    assert_eq!(prefill_tokens(&together, first), 3);
    assert_eq!(prefill_tokens(&together, second), 2);
}

#[test]
fn a_held_prompt_starts_when_it_has_waited_long_enough() {
    let mut scheduler = two_decoders(Some(policy(64, 10, 2)));
    let prompt = submit(&mut scheduler, 3, 4);

    let held = plan(&mut scheduler, 5);
    assert!(held.prefill_items().is_empty());
    settle(&mut scheduler, &held, 6);

    let late = plan(&mut scheduler, 20);
    assert_eq!(prefill_tokens(&late, prompt), 3);
}

#[test]
fn a_started_prompt_is_never_held() {
    // The prompt is longer than a chunk, so it starts and has tokens left.
    let mut scheduler = two_decoders(Some(policy(6, 1_000, 2)));
    let long = submit(&mut scheduler, 6, 4);
    let start = plan(&mut scheduler, 5);
    assert_eq!(prefill_tokens(&start, long), 4);
    settle(&mut scheduler, &start, 6);

    // Two tokens are left, below the threshold, but the prompt has started.
    let rest = plan(&mut scheduler, 7);
    assert_eq!(prefill_tokens(&rest, long), 2);
}

#[test]
fn few_decode_rows_do_not_hold_prompts() {
    let mut scheduler = two_decoders(Some(policy(64, 1_000, 3)));
    let prompt = submit(&mut scheduler, 3, 4);
    let step = plan(&mut scheduler, 5);
    assert_eq!(prefill_tokens(&step, prompt), 3);
}

#[test]
fn an_invalid_policy_is_rejected() {
    let mut scheduler = scheduler(None);
    for invalid in [policy(0, 10, 1), policy(8, 0, 1), policy(8, 10, 0)] {
        assert!(matches!(
            scheduler.set_prefill_coalescing_policy(Some(invalid)),
            Err(SchedulerError::InvalidConfiguration { .. })
        ));
    }
}
