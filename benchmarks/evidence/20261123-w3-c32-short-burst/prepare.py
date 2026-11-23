"""Add an experimental 512-row budget for short-only prefill to a pinned build."""
from hashlib import sha256
from pathlib import Path
import sys


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"missing scheduler anchor: {old}")
    return source.replace(old, new, 1)


root = Path(sys.argv[1])
path = root / "crates/riley-scheduler/src/scheduler.rs"
raw = path.read_bytes()
if sha256(raw).hexdigest() != "bed57e269c1d71e19ecbda3f7476dfb80ef4fca5c11d713b0fa82aec0f24c4fa":
    raise ValueError("wrong archived scheduler")
source = raw.decode()
source = replace_once(source, "    prefill_burst: Option<PrefillBurstPolicy>,",
                      "    prefill_burst: Option<PrefillBurstPolicy>,\n    short_prefill_burst_decode_limit: Option<usize>,")
source = replace_once(source, "        let config = config.validate()?;",
                      """        let config = config.validate()?;
        let short_prefill_burst_decode_limit = match std::env::var("RILEY_W3_SHORT_BURST_DECODE_LIMIT") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) if value == "0" => None,
            Ok(value) if value == "8" => Some(8),
            Ok(value) if value == "16" => Some(16),
            _ => return Err(SchedulerError::InvalidConfiguration {
                field: "RILEY_W3_SHORT_BURST_DECODE_LIMIT",
                reason: "expected 0, 8, or 16",
            }),
        };""")
source = replace_once(source, "            prefill_burst: None,",
                      "            prefill_burst: None,\n            short_prefill_burst_decode_limit,")
source = replace_once(source, "        let mut aged_request = None;",
                      """        if let (Some(decode_limit), Some(policy)) = (self.short_prefill_burst_decode_limit, self.prefill_burst) {
            if budget == policy.steady_budget
                && policy.min_prompt_tokens == 512
                && decode.len() <= decode_limit
                && prefill.iter().all(|item| item.prompt_tokens < policy.min_prompt_tokens)
                && prefill.iter().any(|item| item.remaining_tokens >= 64)
            {
                budget = budget.max(self.config.iteration_token_budget.min(512));
            }
        }
        let mut aged_request = None;""")
test = Path(__file__).with_name("short-burst-test.rs").read_text()
source = replace_once(source, "mod tests {\n", "mod tests {\n" + test + "\n")
path.write_text(source)
