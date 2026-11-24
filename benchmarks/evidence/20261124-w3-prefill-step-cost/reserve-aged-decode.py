"""Apply a diagnostic-only reserve for already active decodes on a wide aging turn."""
from pathlib import Path
import sys

source = Path(sys.argv[1]) / "crates/riley-scheduler/src/scheduler.rs"
text = source.read_text()
anchor = "            let reserved_for_decode = usize::from(budget > 1);"
replacement = """            // The normal aging override protects at least one decoder. This
            // trial leaves room for all active decoders on the wide long-prompt turn.
            static RESERVE_AGED_DECODE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let reserve_all = budget == 1_024
                && item.prompt_tokens >= 3_072
                && decode.len() > 1
                && !self.atomic_prefill_chunks
                && *RESERVE_AGED_DECODE.get_or_init(|| {
                    std::env::var_os("RILEY_W3_RESERVE_AGED_DECODE").is_some_and(|value| value == "1")
                });
            let reserved_for_decode = if reserve_all {
                decode.len().min(budget - 1)
            } else {
                usize::from(budget > 1)
            };"""
if text.count(anchor) != 1:
    raise ValueError("unexpected aging override source")
source.write_text(text.replace(anchor, replacement))
