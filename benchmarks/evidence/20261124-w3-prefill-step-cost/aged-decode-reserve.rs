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

#[test]
fn wide_aging_turn_retains_prefill_and_reserves_ready_decoders() {
    let enabled =
        std::env::var_os("RILEY_W3_RESERVE_AGED_DECODE").is_some_and(|value| value == "1");
    for (budget, long_prompt, atomic, aged) in [
        (1_024, 3_500, false, true),
        (1_024, 3_500, false, false),
        (1_024, 3_500, true, true),
        (512, 3_500, false, true),
        (1_024, 3_071, false, true),
    ] {
        let config = SchedulerConfig {
            iteration_token_budget: budget,
            max_prefill_chunk_tokens: budget,
            max_sequence_tokens: 4_096,
            max_active_sequences: 4,
            max_promised_kv_blocks: 1_024,
            ..SchedulerConfig::default()
        };
        let mut scheduler =
            Scheduler::new(config, KvLayout::checked(1, 1_024, 1, 8).expect("layout"))
                .expect("scheduler");
        if atomic {
            scheduler
                .set_atomic_prefill_chunks(true)
                .expect("atomic prefill");
        }
        let decoders = [
            admit(&mut scheduler, 1, 0),
            admit(&mut scheduler, 1, 0),
            admit(&mut scheduler, 1, 0),
        ];
        let first = plan(&mut scheduler, 1);
        settle(&mut scheduler, &first, 2);
        let prefill = admit(&mut scheduler, long_prompt, 1_000_000);
        let turn = plan(&mut scheduler, if aged { 150_000_000 } else { 30_000_000 });
        let expected_decodes =
            if !aged || enabled && budget == 1_024 && long_prompt >= 3_072 && !atomic {
                3
            } else if atomic {
                0
            } else {
                1
            };
        let observed: Vec<_> = turn
            .decode_items()
            .iter()
            .map(WorkItem::request_id)
            .collect();
        assert_eq!(observed, decoders[..expected_decodes]);
        assert_eq!(turn.total_tokens(), budget);
        let [prefill_item] = turn.prefill_items() else {
            panic!("expected the aged prefill");
        };
        assert_eq!(prefill_item.request_id(), prefill);
        assert_eq!(prefill_item.input_tokens().len(), budget - expected_decodes);
        settle(&mut scheduler, &turn, 160_000_000);

        if !atomic {
            let next = plan(&mut scheduler, 161_000_000);
            let mut observed: Vec<_> = next
                .decode_items()
                .iter()
                .map(WorkItem::request_id)
                .collect();
            observed.sort();
            assert_eq!(observed, decoders);
            assert_eq!(next.prefill_items()[0].request_id(), prefill);
        }
    }
}
