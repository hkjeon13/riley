"""Change the archived experiment's threshold for prefill-only wide steps."""

from hashlib import sha256
from pathlib import Path
import sys

SCHEDULER_SHA = "bf6063d09d310ddf6deb013926ae4340e3361a0aad58bf119a9e729e478b4d4d"
RUNNER_SHA = "acbf47befdb15c7453ad1df2cae9eb2568d77ea670528eeccbeadbb607c4b3fd"


def patch(path, expected, replacements):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"archived input differs: {path.name}")
    text = raw.decode()
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"expected one source anchor in {path.name}")
        text = text.replace(old, new, 1)
    path.write_text(text)


def prepare(source, runner):
    patch(source / "crates/riley-scheduler/src/scheduler.rs", SCHEDULER_SHA, (
        ('            Ok(value) if value == "8" => Some(8),',
         '            Ok(value) if value == "prefill-only" => Some(0),\n'
         '            Ok(value) if value == "8" => Some(8),'),
        ('reason: "expected 0, 8, or 16",', 'reason: "expected 0, prefill-only, 8, or 16",'),
        ('            (Some(8), 0, short.as_slice(), 512),',
         '            (Some(0), 0, short.as_slice(), 512),\n'
         '            (Some(0), 1, short.as_slice(), 256),\n'
         '            (Some(8), 0, short.as_slice(), 512),'),
    ))
    patch(runner / "run_case.py", RUNNER_SHA, (
        ('short_limit in ("0", "8")', 'short_limit in ("0", "prefill-only")'),
        ('"short_burst_decode_limit": int(short_limit)',
         '"short_burst_decode_limit": None if short_limit == "0" else 0'),
    ))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare-prefill.py SOURCE_DIRECTORY RUNNER_DIRECTORY")
    prepare(*map(Path, sys.argv[1:]))
