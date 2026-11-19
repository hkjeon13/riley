use std::collections::BTreeMap;
use std::io::{self, Read};

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationResult, OverloadPolicy, PrefillBurstPolicy, RequestDescriptor, Scheduler,
    SchedulerConfig,
};

fn values(line: &str) -> Vec<u64> {
    line.split_whitespace()
        .map(|field| field.parse().expect("unsigned replay field"))
        .collect()
}

fn main() {
    let max_age_adjustment = match std::env::args().nth(1).as_deref() {
        Some("send") => u64::MAX,
        Some("admit") => 0,
        Some(cap) => cap.parse().expect("unsigned nanosecond cap"),
        None => panic!("choose send, admit, or a nanosecond cap"),
    };
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("read fixture");
    let mut lines = input.lines();
    let mut requests: Vec<_> = lines
        .by_ref()
        .take(8)
        .map(|line| {
            let fields = values(line);
            let [index, send, admit, prompt]: [u64; 4] = fields.try_into().expect("timing record");
            assert!(send <= admit, "send must precede actual admission");
            (
                admit,
                index,
                send,
                usize::try_from(prompt).expect("prompt fits usize"),
            )
        })
        .collect();
    assert_eq!(requests.len(), 8);
    requests.sort_unstable();
    let mut requests = requests.into_iter().peekable();
    let mut ids = BTreeMap::new();
    let config = SchedulerConfig {
        max_waiting_requests: 64,
        max_waiting_prompt_tokens: 64 * 8_192,
        max_active_sequences: 32,
        max_sequence_tokens: 8_192,
        iteration_token_budget: 1_024,
        max_prefill_chunk_tokens: 1_024,
        aging_threshold_ns: 100_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: Some(30_000_000_000),
        max_promised_kv_blocks: 2_576,
        metrics_window_samples: 1_024,
    };
    let layout = KvLayout::checked(48, 2_576, 8, 128).expect("Qwen2.5-14B KV layout");
    let mut scheduler = Scheduler::new(config, layout).expect("valid scheduler");
    scheduler
        .set_prefill_burst_policy(Some(PrefillBurstPolicy {
            steady_budget: 256,
            max_decode_rows: 32,
            min_prompt_tokens: 512,
        }))
        .expect("valid burst policy");

    let mut last_end = 0;
    for line in lines {
        let fields = values(line);
        let [start, end]: [u64; 2] = fields.try_into().expect("step window");
        assert!(last_end <= start && start < end);
        while requests.peek().is_some_and(|&(at, _, _, _)| at <= start) {
            let (at, index, send, prompt) = requests.next().expect("next admission");
            assert!(at >= last_end);
            let submitted = scheduler
                .submit_with_initial_prefill_timestamp_for_replay(
                    RequestDescriptor::new(vec![1; prompt], 128),
                    at,
                    send.max(at.saturating_sub(max_age_adjustment)),
                )
                .expect("candidate admitted without KV pressure");
            ids.insert(submitted.request_id(), index);
        }
        let output = scheduler.plan_iteration(start).expect("prefill plans");
        let plan = output.plan().expect("a prefill step is available");
        if !plan.decode_items().is_empty() || !plan.output_slots().is_empty() {
            println!("non-prefill-only");
            return;
        }
        let [work] = plan.prefill_items() else {
            println!("non-single-prefill");
            return;
        };
        println!(
            "{} {} {}",
            ids[&work.request_id()],
            work.target_logical_length() - work.input_tokens().len(),
            work.target_logical_length(),
        );
        let results =
            IterationResult::new(plan.iteration_id(), Vec::new(), 0, 0).expect("step result");
        let updates = scheduler
            .complete_iteration(&results, end)
            .expect("prefill completion");
        assert!(updates.completions().is_empty() && updates.settlement_failures().is_empty());
        last_end = end;
    }
    assert!(requests.next().is_none());
    let released = scheduler.close(last_end, None).expect("scheduler closes");
    assert_eq!(released.completions().len(), 8);
}
