"""Select the narrow timing mode for each archived short-burst replay."""
from hashlib import sha256
from pathlib import Path
import sys


path = Path(sys.argv[1])
raw = path.read_bytes()
if sha256(raw).hexdigest() != "e31809d82369ef5fdb9c3b53cbf05a262fab3bd1fac5dd6974088abe412fedb7":
    raise ValueError("unexpected short-burst replay driver")
source = raw.decode()
for old, new in (
    ('    require(trace in ("control", "trace"), "unknown trace setting")',
     '    require(trace in ("control", "admission"), "unknown trace setting")'),
    ('    trace_env = f"RILEY_W3_STAGE_TRACE={int(trace == \'trace\')}"',
     '    trace_env = "RILEY_W3_STAGE_TRACE=" + ("admission" if trace == "admission" else "0")'),
    ('and trace == "control" and short_limit in ("0", "8", "16")', 'and short_limit in ("0", "8")'),
    ('arm = f"{case}-short-d{short_limit}-gpu{index}"', 'arm = f"{case}-short-d{short_limit}-{trace}-gpu{index}"'),
    ('"stage_trace": trace == "trace",', '"stage_trace": trace == "admission", "trace_mode": trace,'),
):
    if source.count(old) != 1:
        raise ValueError(f"unexpected driver anchor: {old}")
    source = source.replace(old, new, 1)
compile(source, str(path), "exec")
path.write_text(source)
