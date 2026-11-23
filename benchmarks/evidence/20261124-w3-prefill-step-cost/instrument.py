"""Capture mixed-step choices and committed decode contexts in the diagnostic build."""

from pathlib import Path
import runpy
import sys


def instrument(root, broad, narrow, timeline):
    helpers = runpy.run_path(str(broad))
    edit = helpers["edit"]
    runpy.run_path(str(narrow))["instrument"](root, broad)
    server = root / "crates/riley-server/src"
    runpy.run_path(str(timeline))["instrument"](server)

    edit(root / "crates/riley-scheduler/src/scheduler.rs", [
        ("use std::sync::OnceLock;", "use std::sync::{Mutex, OnceLock};"),
        ("static MIXED_PREFILL_REORDERED: AtomicU64 = AtomicU64::new(0);",
         """static MIXED_PREFILL_REORDERED: AtomicU64 = AtomicU64::new(0);
static MIXED_PREFILL_TRACE: OnceLock<bool> = OnceLock::new();
static MIXED_PREFILL_CHOICES: Mutex<MixedPrefillChoices> = Mutex::new(MixedPrefillChoices {
    rows: Vec::new(),
    dropped: 0,
});

struct MixedPrefillChoice {
    iteration: u64,
    applied: bool,
    baseline: ReadyRequest,
    candidate: ReadyRequest,
    rows: usize,
}

struct MixedPrefillChoices {
    rows: Vec<MixedPrefillChoice>,
    dropped: usize,
}

fn trace_mixed_prefill() -> bool {
    *MIXED_PREFILL_TRACE.get_or_init(|| {
        std::env::var_os("RILEY_W3_RAG_STEPS").is_some_and(|value| value == "1")
    })
}"""),
        ("""            && *MIXED_PREFILL_FIFO.get_or_init(|| {
                std::env::var_os("RILEY_W3_PREFILL_FIFO").is_some_and(|value| value == "1")
            })
        {
            if let Some(head) = prefill.first().copied().filter(|item| {
                item.prompt_tokens >= 3_072 && item.remaining_tokens >= budget
            }) {""",
         """        {
            let fifo = *MIXED_PREFILL_FIFO.get_or_init(|| {
                std::env::var_os("RILEY_W3_PREFILL_FIFO").is_some_and(|value| value == "1")
            });
            let capture = trace_mixed_prefill();
            if let Some(head) = prefill.first().copied().filter(|item| {
                (fifo || capture) && item.prompt_tokens >= 3_072 && item.remaining_tokens >= budget
            }) {"""),
        ("""                if let Some((index, _)) = prefill.iter().enumerate().skip(1)""",
         """                if let Some((index, &candidate)) = prefill.iter().enumerate().skip(1)"""),
        ("""                    prefill.swap(0, index);
                    MIXED_PREFILL_REORDERED.fetch_add(1, Ordering::Relaxed);""",
         """                    if capture {
                        let mut choices = MIXED_PREFILL_CHOICES.lock().expect("mixed prefill trace");
                        if choices.rows.len() == 4_096 {
                            choices.dropped += 1;
                        } else {
                            choices.rows.push(MixedPrefillChoice {
                                iteration: self.next_iteration_id,
                                applied: fifo,
                                baseline: head,
                                candidate,
                                rows: budget,
                            });
                        }
                    }
                    if fifo {
                        prefill.swap(0, index);
                        MIXED_PREFILL_REORDERED.fetch_add(1, Ordering::Relaxed);
                    }"""),
        ("    fn drop_prefix_cache_entries(&mut self) -> SchedulerResult<()> {",
         """    /// Export benchmark-only mixed-prefill choices after serving stops.
    pub fn export_mixed_prefill_choices() {
        if !trace_mixed_prefill() {
            return;
        }
        let mut choices = MIXED_PREFILL_CHOICES.lock().expect("mixed prefill trace");
        eprintln!("W3-MIXED-CHOICES {} {}", choices.rows.len(), choices.dropped);
        for row in choices.rows.drain(..) {
            eprintln!(
                "W3-MIXED-CHOICE {} {} {} {} {} {} {} {} {}",
                row.iteration,
                u8::from(row.applied),
                row.baseline.request_id.get(),
                row.baseline.prompt_tokens,
                row.baseline.prompt_tokens - row.baseline.remaining_tokens,
                row.candidate.request_id.get(),
                row.candidate.prompt_tokens,
                row.candidate.prompt_tokens - row.candidate.remaining_tokens,
                row.rows,
            );
        }
        choices.dropped = 0;
    }

    fn drop_prefix_cache_entries(&mut self) -> SchedulerResult<()> {"""),
    ])
    edit(server / "engine.rs", [
        ('            eprintln!("W3-PREFILL-FIFO {}", Scheduler::mixed_prefill_reordered_steps());',
         '''            Scheduler::export_mixed_prefill_choices();
            eprintln!("W3-PREFILL-FIFO {}", Scheduler::mixed_prefill_reordered_steps());'''),
    ])
    edit(server / "rag_steps.rs", [
        ("        decode: usize,", "        decode: Vec<(u64, usize)>,"),
        ("        decode: plan.decode_items().len(),",
         """        decode: plan
            .decode_items()
            .iter()
            .map(|row| (row.request_id().get(), row.target_logical_length()))
            .collect(),"""),
        ('                eprintln!("W3-RAG-STEP {id} {start} {end} {commit} {gpu} {rows} {decode} {items}");',
         '''                eprintln!(
                    "W3-RAG-STEP {id} {start} {end} {commit} {gpu} {rows} {} {items}",
                    decode.len()
                );
                if !decode.is_empty() {
                    let contexts = decode
                        .iter()
                        .map(|(request, length)| format!("{request}:{length}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    eprintln!("W3-MIXED-DECODE {id} {contexts}");
                }'''),
    ])


if __name__ == "__main__":
    instrument(*map(Path, sys.argv[1:]))
