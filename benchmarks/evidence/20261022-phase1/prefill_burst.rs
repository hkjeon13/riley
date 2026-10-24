//! Model-free coverage for the prefill burst policy.

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, OverloadPolicy, PrefillBurstPolicy,
    RequestDescriptor, RequestId, Scheduler, SchedulerConfig, SchedulerError,
};

fn scheduler(policy: Option<PrefillBurstPolicy>) -> Scheduler {
    let config = SchedulerConfig {
        max_waiting_requests: 8,
        max_waiting_prompt_tokens: 256,
        max_active_sequences: 8,
        max_sequence_tokens: 64,
        iteration_token_budget: 16,
        max_prefill_chunk_tokens: 8,
        aging_threshold_ns: u64::MAX,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 32,
        metrics_window_samples: 4,
    };
    let layout = KvLayout::checked(1, 32, 1, 8).expect("valid test KV layout");
    let mut scheduler = Scheduler::new(config, layout).expect("valid test scheduler");
    scheduler
        .set_prefill_burst_policy(policy)
        .expect("valid burst policy");
    scheduler
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

fn burst(steady_budget: usize, max_decode_rows: usize) -> PrefillBurstPolicy {
    PrefillBurstPolicy {
        steady_budget,
        max_decode_rows,
    }
}

#[test]
fn a_burst_with_few_decode_rows_uses_the_whole_budget() {
    let mut scheduler = scheduler(Some(burst(8, 2)));
    for _ in 0..4 {
        submit(&mut scheduler, 8, 1);
    }
    // No decode rows: two whole 8-token prompts fit the 16-token budget.
    let first = plan(&mut scheduler, 2);
    assert_eq!(first.decode_items().len(), 0);
    assert_eq!(first.total_tokens(), 16);
    settle(&mut scheduler, &first, 3);
    // Two decode rows are still at most `max_decode_rows`: 2 + 8 + 6.
    let second = plan(&mut scheduler, 4);
    assert_eq!(second.decode_items().len(), 2);
    assert_eq!(second.total_tokens(), 16);
}

#[test]
fn more_decode_rows_than_the_bound_keep_the_steady_budget() {
    let mut scheduler = scheduler(Some(burst(8, 2)));
    for _ in 0..3 {
        submit(&mut scheduler, 1, 1);
    }
    let first = plan(&mut scheduler, 2);
    assert_eq!(first.total_tokens(), 3);
    settle(&mut scheduler, &first, 3);
    submit(&mut scheduler, 8, 4);
    // Three decode rows exceed the bound: 3 + 5 prompt tokens = 8.
    let second = plan(&mut scheduler, 5);
    assert_eq!(second.decode_items().len(), 3);
    assert_eq!(second.total_tokens(), 8);
}

#[test]
fn without_the_policy_every_step_uses_the_whole_budget() {
    let mut scheduler = scheduler(None);
    for _ in 0..3 {
        submit(&mut scheduler, 1, 1);
    }
    let first = plan(&mut scheduler, 2);
    settle(&mut scheduler, &first, 3);
    submit(&mut scheduler, 8, 4);
    let second = plan(&mut scheduler, 5);
    assert_eq!(second.total_tokens(), 11);
}

#[test]
fn a_steady_budget_must_be_positive_and_below_the_iteration_budget() {
    let mut scheduler = scheduler(None);
    for invalid in [burst(0, 2), burst(16, 2), burst(17, 2)] {
        assert!(matches!(
            scheduler.set_prefill_burst_policy(Some(invalid)),
            Err(SchedulerError::InvalidConfiguration { .. })
        ));
    }
}
