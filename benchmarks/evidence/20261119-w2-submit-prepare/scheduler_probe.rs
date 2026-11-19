use std::hint::black_box;
use std::time::Instant;

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationPlan, IterationResult, OverloadPolicy, OwnedBlockTable,
    RequestDescriptor, RequestId, RequestState, Scheduler, SchedulerConfig,
};

fn complete(scheduler: &mut Scheduler, plan: &IterationPlan, now: u64) {
    let outputs = plan
        .output_slots()
        .iter()
        .map(|&slot| IterationOutput::new(slot, 8, false))
        .collect();
    let result = IterationResult::new(plan.iteration_id(), outputs, 0, 0)
        .expect("deterministic tokens are valid");
    let updates = scheduler
        .complete_iteration(&result, now)
        .expect("iteration settles");
    assert!(updates.completions().is_empty());
    assert!(updates.settlement_failures().is_empty());
}

fn measure(context: usize) -> f64 {
    let config = SchedulerConfig {
        max_waiting_requests: 32,
        max_waiting_prompt_tokens: 32 * context,
        max_active_sequences: 32,
        max_sequence_tokens: context + 700,
        iteration_token_budget: 1024,
        max_prefill_chunk_tokens: context.min(1024),
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 12_000,
        metrics_window_samples: 16,
    };
    let layout = KvLayout::checked(1, 12_000, 1, 64).expect("valid host layout");
    let mut scheduler = Scheduler::new(config, layout).expect("valid scheduler");
    let mut requests = Vec::new();
    for _ in 0..32 {
        requests.push(
            scheduler
                .submit(RequestDescriptor::new(vec![7; context], 700), 0)
                .expect("fixture fits scheduler limits")
                .request_id(),
        );
    }
    let mut now = 1;
    while requests
        .iter()
        .any(|&request| scheduler.request_state(request) != Some(RequestState::Decoding))
    {
        let output = scheduler.plan_iteration(now).expect("prefill plans");
        let plan = output.plan().expect("prefill work remains");
        complete(&mut scheduler, plan, now + 1);
        now += 2;
    }
    let warmups = 48;
    let samples = 320;
    let mut planning_ns = 0;
    for sample in 0..warmups + samples {
        let started = Instant::now();
        let output = black_box(&mut scheduler)
            .plan_iteration(now)
            .expect("decode plans");
        let elapsed = started.elapsed().as_nanos();
        let plan = output.plan().expect("decode work remains");
        assert_eq!(plan.decode_items().len(), 32);
        assert!(plan.prefill_items().is_empty());
        assert!(plan.block_tables().iter().all(|table| {
            table
                .physical_block_ids()
                .windows(2)
                .all(|pair| pair[0] < pair[1])
        }));
        if sample >= warmups {
            planning_ns += elapsed;
        }
        complete(&mut scheduler, plan, now + 1);
        now += 2;
    }
    let closed = scheduler
        .close(now, None)
        .expect("scheduler closes without work in flight");
    assert_eq!(closed.completions().len(), 32);
    assert!(closed.settlement_failures().is_empty());
    planning_ns as f64 / f64::from(samples)
}

fn verify_external_tables() {
    let request = RequestId::new(1).expect("nonzero fixture request");
    let version = riley_runtime::paged_kv::BLOCK_TABLE_V1_VERSION;
    for ids in [vec![1, 7, 13], vec![13, 1, 7]] {
        let table = OwnedBlockTable::new(request, version, ids.clone(), vec![16, 16, 1], 33)
            .expect("unique IDs are accepted in arbitrary logical order");
        assert_eq!(table.physical_block_ids(), ids);
    }
    for ids in [vec![1, 1, 13], vec![13, 1, 13]] {
        let error = OwnedBlockTable::new(request, version, ids, vec![16, 16, 1], 33)
            .expect_err("duplicate IDs must still be rejected");
        assert!(
            error
                .to_string()
                .contains("physical block IDs must be unique")
        );
    }
    let error = OwnedBlockTable::new(request, version, vec![1, 7, 13], vec![16, 15, 2], 33)
        .expect_err("ordered IDs do not bypass canonical token-count validation");
    assert!(error.to_string().contains("one canonical tail"));
}

fn main() {
    verify_external_tables();
    for context in [129, 1025, 4097] {
        println!("decode32-{context} {:.3}", measure(context));
    }
}
