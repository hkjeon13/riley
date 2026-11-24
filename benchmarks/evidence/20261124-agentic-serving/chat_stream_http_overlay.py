"""Add a test-only baseline selector and the HTTP timing test to an isolated source copy."""

import sys
from pathlib import Path

service = Path(sys.argv[1])
trial = Path(__file__).with_name("chat_stream_http_trial.rs")
src = service.read_text()
needle = "    let mut chat_shaper = delivery"
assert src.count(needle) == 1
src = src.replace(
    needle,
    '    let legacy = std::env::var_os("RILEY_STREAM_LEGACY").is_some();\n' + needle,
)
needle = (
    "                    Some(shaper) => shaper.encode_event(&mut encoder, &event),"
)
assert src.count(needle) == 1
src = src.replace(
    needle,
    "                    Some(shaper) if legacy => encoder.encode_event(&event).map(|frames| shaper.shape(&frames)),\n"
    + needle,
)
pos = src.rindex("\n}")
assert "mod tests" in src and not src[pos + 2 :].strip()
service.write_text(src[:pos] + "\n" + trial.read_text() + src[pos:])
