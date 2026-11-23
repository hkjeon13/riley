//! Benchmark-only timeline. Materialize its records after serving stops.

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use riley_scheduler::IterationPlan;

const MAX_RECORDS: usize = 32_768;

enum Record {
    Admit(u64, u64, u64),
    Visible(String, u64, u64),
    Step {
        id: u64,
        start: u64,
        end: u64,
        commit: u64,
        gpu: u64,
        rows: usize,
        decode: usize,
        prefill: Vec<(u64, usize, usize)>,
    },
}

#[derive(Default)]
struct Trace {
    records: Vec<Record>,
    dropped: usize,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static EPOCH: OnceLock<Instant> = OnceLock::new();
static TRACE: Mutex<Trace> = Mutex::new(Trace {
    records: Vec::new(),
    dropped: 0,
});

fn enabled() -> bool {
    *ENABLED
        .get_or_init(|| std::env::var_os("RILEY_W3_RAG_STEPS").is_some_and(|value| value == "1"))
}

fn ns(at: Instant) -> u64 {
    let epoch = *EPOCH.get_or_init(|| at);
    u64::try_from(at.duration_since(epoch).as_nanos()).expect("bounded elapsed nanoseconds")
}

fn push(record: Record) {
    let mut trace = TRACE.lock().expect("trace lock poisoned");
    if trace.records.len() == MAX_RECORDS {
        trace.dropped += 1;
        return;
    }
    trace.records.push(record);
}

pub(crate) fn clock() -> Option<u64> {
    enabled().then(|| ns(Instant::now()))
}

pub(crate) fn admit(engine_id: u64, scheduler_id: u64) {
    if let Some(at) = clock() {
        push(Record::Admit(engine_id, scheduler_id, at));
    }
}

pub(crate) fn visible(id: &str, admitted: Instant, elapsed: Duration) {
    if enabled() {
        push(Record::Visible(
            id.to_owned(),
            ns(admitted),
            ns(admitted + elapsed),
        ));
    }
}

pub(crate) fn step(plan: &IterationPlan, start: u64, end: u64, gpu: u64) {
    let commit = ns(Instant::now());
    let prefill = plan
        .prefill_items()
        .iter()
        .map(|item| {
            (
                item.request_id().get(),
                item.input_tokens().len(),
                item.target_logical_length(),
            )
        })
        .collect();
    push(Record::Step {
        id: plan.iteration_id().get(),
        start,
        end,
        commit,
        gpu,
        rows: plan.total_tokens(),
        decode: plan.decode_items().len(),
        prefill,
    });
}

pub(crate) fn export() {
    if !enabled() {
        return;
    }
    let trace = std::mem::take(&mut *TRACE.lock().expect("trace lock poisoned"));
    eprintln!("W3-RAG-SNAPSHOT {} {}", trace.records.len(), trace.dropped);
    for row in trace.records {
        match row {
            Record::Admit(engine, scheduler, at) => {
                eprintln!("W3-RAG-ADMIT {engine} {scheduler} {at}")
            }
            Record::Visible(id, accepted, first) => {
                eprintln!("W3-RAG-VISIBLE {id} {accepted} {first}")
            }
            Record::Step {
                id,
                start,
                end,
                commit,
                gpu,
                rows,
                decode,
                prefill,
            } => {
                let items = prefill
                    .iter()
                    .map(|(id, count, target)| format!("{id}:{count}:{target}"))
                    .collect::<Vec<_>>()
                    .join(",");
                eprintln!("W3-RAG-STEP {id} {start} {end} {commit} {gpu} {rows} {decode} {items}");
            }
        }
    }
}
