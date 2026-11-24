//! Replay prompt groups through the native scheduler, with one synthetic output token per request.
use std::collections::HashMap;
use std::error::Error;
use std::io::{self, BufRead};

use riley_runtime::paged_kv::KvLayout;
use riley_scheduler::{
    IterationOutput, IterationResult, OverloadPolicy, PrefillBurstPolicy, RequestDescriptor,
    Scheduler, SchedulerConfig, WorkItem,
};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Job {
    seed: u32,
    ids: Vec<u32>,
}

#[derive(Default, Serialize)]
struct Phase {
    rows: Vec<(u32, usize, usize)>,
    computed: usize,
    retained: usize,
    evictions: u64,
    cancelled: Vec<(u32, usize)>,
    completion_ticks: Vec<(u32, usize)>,
    pending_at_boundary: usize,
}

struct Pending {
    phase: usize,
    seed: u32,
    length: usize,
    executed: usize,
}

fn scheduler() -> Result<Scheduler, Box<dyn Error>> {
    let config = SchedulerConfig {
        max_waiting_requests: 64,
        max_waiting_prompt_tokens: 524_288,
        max_active_sequences: 8,
        max_sequence_tokens: 8192,
        iteration_token_budget: 1024,
        max_prefill_chunk_tokens: 1024,
        aging_threshold_ns: 1_000_000_000,
        overload_policy: OverloadPolicy::Wait,
        admission_timeout_ns: None,
        max_promised_kv_blocks: 4096,
        metrics_window_samples: 64,
    };
    let mut sched = Scheduler::new(config, KvLayout::checked(1, 4096, 1, 8)?)?;
    sched.enable_prefix_cache(1536)?;
    sched.set_prefill_burst_policy(Some(PrefillBurstPolicy {
        steady_budget: 256,
        max_decode_rows: 0,
        min_prompt_tokens: 512,
    }))?;
    Ok(sched)
}

fn replay(
    sched: &mut Scheduler,
    groups: Vec<Vec<Job>>,
    limit: Option<usize>,
    cancel: bool,
) -> Result<[Phase; 2], Box<dyn Error>> {
    assert_eq!(groups.len(), 2);
    let mut phases = [Phase::default(), Phase::default()];
    let mut pending = HashMap::new();
    let mut clock = 0;
    let mut tick = 0;
    let mut submitted_tick = [0; 2];
    for (stage, jobs) in groups.into_iter().enumerate() {
        submitted_tick[stage] = tick;
        for job in jobs {
            let len = job.ids.len();
            let id = sched
                .submit(RequestDescriptor::new(job.ids, 1), clock)?
                .request_id();
            pending.insert(
                id,
                Pending {
                    phase: stage,
                    seed: job.seed,
                    length: len,
                    executed: 0,
                },
            );
        }
        while !(pending.is_empty() || (stage == 0 && limit.is_some_and(|n| tick >= n))) {
            clock += 1;
            let (plan, finished) = sched.plan_iteration(clock)?.into_parts();
            assert!(finished.is_empty());
            let plan = plan.ok_or("unfinished requests with no scheduled work")?;
            assert!(plan.decode_items().is_empty());
            for item in plan.prefill_items() {
                let job = pending
                    .get_mut(&item.request_id())
                    .ok_or("unregistered request")?;
                job.executed += item.input_tokens().len();
                phases[job.phase].computed += item.input_tokens().len();
            }
            let outputs = plan
                .prefill_items()
                .iter()
                .filter_map(WorkItem::output_slot)
                .map(|slot| IterationOutput::new(slot, 1000, false))
                .collect();
            let result = IterationResult::new(plan.iteration_id(), outputs, 1, 0)?;
            clock += 1;
            tick += 1;
            let (_, finished, failures) = sched.complete_iteration(&result, clock)?.into_parts();
            assert!(failures.is_empty());
            for r in finished {
                let job = pending
                    .remove(&r.request_id())
                    .ok_or("unexpected completion")?;
                assert_eq!(job.executed, job.length - r.cached_prompt_tokens());
                phases[job.phase]
                    .rows
                    .push((job.seed, job.length, r.cached_prompt_tokens()));
                phases[job.phase]
                    .completion_ticks
                    .push((job.seed, tick - submitted_tick[job.phase]));
            }
        }
        phases[stage].pending_at_boundary = pending.len();
        if stage == 0 && cancel {
            // Match arrival order when cancelling. Cache recency is observable.
            let mut ids = pending.keys().copied().collect::<Vec<_>>();
            ids.sort_unstable_by_key(|id| id.get());
            for id in ids {
                let outcome = sched.cancel(id, clock)?;
                assert!(outcome.completion().is_some());
                let job = pending.remove(&id).unwrap();
                phases[job.phase].cancelled.push((job.seed, job.executed));
            }
        }
        let m = sched.metrics_snapshot()?;
        phases[stage].retained = m.gauges.prefix_cache_blocks;
        phases[stage].evictions = m.prefix_cache_evicted_blocks;
        if pending.is_empty() {
            assert_eq!(m.gauges.active_sequences, 0);
            assert_eq!(m.gauges.waiting_requests, 0);
        }
    }
    assert!(pending.is_empty());
    for p in &phases {
        assert_eq!(
            p.computed,
            p.rows.iter().map(|(_, n, h)| n - h).sum::<usize>()
                + p.cancelled.iter().map(|(_, n)| n).sum::<usize>()
        );
    }
    Ok(phases)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Without a step limit the two groups drain in sequence.
    let mut args = std::env::args().skip(1);
    let limit = args.next().map(|n| n.parse::<usize>()).transpose()?;
    let cancel = match args.next().as_deref() {
        None | Some("keep") => false,
        Some("cancel") => true,
        _ => return Err("expected keep or cancel".into()),
    };
    for line in io::stdin().lock().lines() {
        let groups: Vec<Vec<Job>> = serde_json::from_str(&line?)?;
        let mut sched = scheduler()?;
        let phases = replay(&mut sched, groups, limit, cancel)?;
        println!("{}", serde_json::to_string(&phases)?);
    }
    Ok(())
}
