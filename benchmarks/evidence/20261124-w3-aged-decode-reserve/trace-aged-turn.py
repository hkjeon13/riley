"""Record who received the wide aging turn and how much room was left for decode."""
from pathlib import Path
import sys

source = Path(sys.argv[1]) / "crates/riley-scheduler/src/scheduler.rs"
text = source.read_text()
changes = (
    ("    rows: Vec::new(),\n    dropped: 0,", "    rows: Vec::new(),\n    aged: Vec::new(),\n    dropped: 0,"),
    ("    rows: Vec<MixedPrefillChoice>,", """    rows: Vec<MixedPrefillChoice>,
    aged: Vec<AgedDecodeTurn>,"""),
    ("fn trace_mixed_prefill() -> bool {", """struct AgedDecodeTurn {
    iteration: u64,
    request: RequestId,
    prompt_tokens: usize,
    ready_decode: usize,
    prefill_rows: usize,
    reserve_all: bool,
}

fn trace_mixed_prefill() -> bool {"""),
    ("""                .min(usable);
            selected.push(Candidate {""", """                .min(usable);
            if trace_mixed_prefill()
                && budget == 1_024
                && !self.atomic_prefill_chunks
                && item.prompt_tokens >= 3_072
                && decode.len() > 1
            {
                let mut choices = MIXED_PREFILL_CHOICES.lock().expect("mixed prefill trace");
                if choices.aged.len() == 4_096 {
                    choices.dropped += 1;
                } else {
                    choices.aged.push(AgedDecodeTurn {
                        iteration: self.next_iteration_id,
                        request: request_id,
                        prompt_tokens: item.prompt_tokens,
                        ready_decode: decode.len(),
                        prefill_rows: token_count,
                        reserve_all,
                    });
                }
            }
            selected.push(Candidate {"""),
    ("        choices.dropped = 0;\n    }", """        eprintln!("W3-AGED-TURNS {}", choices.aged.len());
        for row in choices.aged.drain(..) {
            eprintln!(
                "W3-AGED-TURN {} {} {} {} {} {}",
                row.iteration, row.request.get(), row.prompt_tokens, row.ready_decode,
                row.prefill_rows, u8::from(row.reserve_all),
            );
        }
        choices.dropped = 0;
    }"""),
)
for old, new in changes:
    if text.count(old) != 1:
        raise ValueError(f"unexpected mixed trace source: {old!r}")
    text = text.replace(old, new)
source.write_text(text)
