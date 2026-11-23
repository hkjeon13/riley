"""Restrict first-arrival preference to a single whole-budget long prefill."""

from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent


def instrument(root, base_instrument):
    helpers = runpy.run_path(str(base_instrument))
    helpers["instrument"](root)
    helpers["edit"](root / "crates/riley-scheduler/src/scheduler.rs", [
        ("""        // Decode and the aging override have already selected their rows. Apply
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
        }""",
         """        // Prefer one earlier long request that has already made progress. Both
        // inputs must fit the whole remaining budget so the replacement keeps the
        // decode rows, prefill rows, and number of prefill items in this step.
        if budget >= 1_016
            && budget <= self.config.max_prefill_chunk_tokens
            && (1..=8).contains(&decode.len())
            && aged_request.is_none()
            && !hold_new_prompts
            && !self.atomic_prefill_chunks
            && *MIXED_PREFILL_FIFO.get_or_init(|| {
                std::env::var_os("RILEY_W3_PREFILL_FIFO").is_some_and(|value| value == "1")
            })
        {
            if let Some(head) = prefill.first().copied().filter(|item| {
                item.prompt_tokens >= 3_072 && item.remaining_tokens >= budget
            }) {
                if let Some((index, _)) = prefill.iter().enumerate().skip(1)
                    .filter(|(_, item)| item.request_id < head.request_id && item.started
                        && item.prompt_tokens >= 3_072 && item.remaining_tokens >= budget)
                    .min_by_key(|(_, item)| item.request_id)
                {
                    prefill.swap(0, index);
                    MIXED_PREFILL_REORDERED.fetch_add(1, Ordering::Relaxed);
                }
            }
        }"""),
    ])


if __name__ == "__main__":
    instrument(Path(sys.argv[1]), Path(sys.argv[2]))
