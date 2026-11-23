"""Add a benchmark switch for first-arrival order in mixed-step prefill."""

from pathlib import Path
import sys


def edit(path, replacements):
    source = path.read_text()
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"expected one pinned fragment in {path}: {before!r}")
        source = source.replace(before, after)
    path.write_text(source)


def instrument(root):
    edit(root / "crates/riley-scheduler/src/scheduler.rs", [
        ("use std::time::Instant;", """use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static MIXED_PREFILL_FIFO: OnceLock<bool> = OnceLock::new();
static MIXED_PREFILL_REORDERED: AtomicU64 = AtomicU64::new(0);"""),
        ("""        for item in &prefill {
            if budget == 0 {""",
         """        // Decode and the aging override have already selected their rows. Apply
        // first-arrival order only when ordinary prefill rows actually contend.
        if budget > 0
            && !decode.is_empty()
            && aged_request.is_none()
            && !hold_new_prompts
            && !self.atomic_prefill_chunks
            && *MIXED_PREFILL_FIFO.get_or_init(|| {
                std::env::var_os("RILEY_W3_PREFILL_FIFO").is_some_and(|value| value == "1")
            })
            && prefill.windows(2).any(|pair| pair[0].request_id > pair[1].request_id)
            && prefill.iter().fold(0_usize, |total, item| {
                total.saturating_add(item.remaining_tokens.min(self.config.max_prefill_chunk_tokens))
            }) > budget
        {
            prefill.sort_unstable_by_key(|item| item.request_id);
            MIXED_PREFILL_REORDERED.fetch_add(1, Ordering::Relaxed);
        }
        for item in &prefill {
            if budget == 0 {"""),
        ("    fn drop_prefix_cache_entries(&mut self) -> SchedulerResult<()> {",
         """    /// Count mixed steps whose optional prefill ordering was changed.
    pub fn mixed_prefill_reordered_steps() -> u64 {
        MIXED_PREFILL_REORDERED.load(Ordering::Relaxed)
    }

    fn drop_prefix_cache_entries(&mut self) -> SchedulerResult<()> {"""),
    ])
    edit(root / "crates/riley-server/src/engine.rs", [
        ("            let mut final_scheduler = None;",
         """            eprintln!("W3-PREFILL-FIFO {}", Scheduler::mixed_prefill_reordered_steps());
            let mut final_scheduler = None;"""),
    ])


if __name__ == "__main__":
    instrument(Path(sys.argv[1]))
