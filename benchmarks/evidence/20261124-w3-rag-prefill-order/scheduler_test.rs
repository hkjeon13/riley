//! Verify the mixed-step order on the same real scheduler with each switch value.

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, RequestDescriptor, RequestId, Scheduler,
    SchedulerConfig, WorkItem,
};

fn admit(scheduler: &mut Scheduler, tokens: usize, time: u64) -> RequestId {
    scheduler
        .submit(RequestDescriptor::new(vec![1; tokens], 16), time)
        .expect("submit")
        .request_id()
}

fn plan(scheduler: &mut Scheduler, time: u64) -> IterationPlan {
    scheduler
        .plan_iteration(time)
        .expect("plan")
        .into_parts()
        .0
        .expect("work")
}

fn settle(scheduler: &mut Scheduler, plan: &IterationPlan, time: u64) {
    let outputs = plan
        .prefill_items()
        .iter()
        .chain(plan.decode_items())
        .filter_map(WorkItem::output_slot)
        .map(|slot| IterationOutput::new(slot, 7, false))
        .collect();
    let result = IterationResult::new(plan.iteration_id(), outputs, 1, 0).expect("executor result");
    let update = scheduler.complete_iteration(&result, time).expect("settle");
    assert!(update.settlement_failures().is_empty());
}

fn assert_mixed(plan: &IterationPlan, prompt: RequestId, decoder: RequestId, consumed: usize) {
    let [prefill] = plan.prefill_items() else {
        panic!("expected one prefill item");
    };
    let [decode] = plan.decode_items() else {
        panic!("expected one decode item");
    };
    assert_eq!(prefill.request_id(), prompt);
    assert_eq!(prefill.input_tokens().len(), 15);
    assert_eq!(prefill.target_logical_length(), consumed);
    assert!(prefill.output_slot().is_none());
    assert_eq!(decode.request_id(), decoder);
    assert_eq!(plan.total_tokens(), 16);
}

#[test]
fn mixed_prefill_order_preserves_decode_and_aging_override() {
    let config = SchedulerConfig {
        iteration_token_budget: 16,
        max_prefill_chunk_tokens: 16,
        max_sequence_tokens: 64,
        max_active_sequences: 3,
        max_promised_kv_blocks: 32,
        ..SchedulerConfig::default()
    };
    let mut scheduler =
        Scheduler::new(config, KvLayout::checked(1, 32, 1, 8).expect("layout")).expect("scheduler");
    let decoder = admit(&mut scheduler, 1, 0);
    let first = plan(&mut scheduler, 1);
    settle(&mut scheduler, &first, 2);

    let earlier = admit(&mut scheduler, 32, 1_000_000);
    let later = admit(&mut scheduler, 32, 2_000_000);
    let first_turn = plan(&mut scheduler, 3_000_000);
    assert_mixed(&first_turn, earlier, decoder, 15);
    settle(&mut scheduler, &first_turn, 30_000_000);

    let fifo = std::env::var_os("RILEY_W3_PREFILL_FIFO").is_some_and(|value| value == "1");
    let next_turn = plan(&mut scheduler, 31_000_000);
    if fifo {
        assert_mixed(&next_turn, earlier, decoder, 30);
    } else {
        assert_mixed(&next_turn, later, decoder, 15);
    }
    settle(&mut scheduler, &next_turn, 32_000_000);
    assert_eq!(Scheduler::mixed_prefill_reordered_steps(), u64::from(fifo));

    // The older *ready* timestamp still wins the original aging override,
    // even when it belongs to the later arrival with FIFO enabled.
    let aged = plan(&mut scheduler, 150_000_000);
    if fifo {
        assert_mixed(&aged, later, decoder, 15);
    } else {
        assert_mixed(&aged, earlier, decoder, 30);
    }
    assert_eq!(Scheduler::mixed_prefill_reordered_steps(), u64::from(fifo));
}
