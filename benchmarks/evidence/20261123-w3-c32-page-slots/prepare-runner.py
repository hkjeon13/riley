"""Replay the archived load with an explicit native attention mode and slot limit."""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
for anchor, replacement in (
    ("workers, trace):", "workers, trace, attention, slots):"),
    ('    arm = f"{case}-{budget}-w{workers}-{trace}-gpu{index}"',
     '    require((budget, attention, slots) in {("128", "fixed", "32"), ("256", "fixed", "32"),\n'
     '                                            ("128", "page", "32"), ("128", "page", "48")}\n'
     '            and workers == "64" and trace == "control", "unexpected attention comparison")\n'
     '    attention_mode = {"fixed": "fixed-tile-unsplit-v1", "page": "page-v1"}[attention]\n'
     '    arm = f"{case}-s{budget}-{attention}-a{slots}-gpu{index}"'),
    ('    argv[argv.index("--prefill-burst-steady-budget") + 1] = budget',
     '    argv[argv.index("--prefill-burst-steady-budget") + 1] = budget\n'
     '    argv[argv.index("--max-active-sequences") + 1] = slots\n'
     '    argv += ["--native-attention-mode", attention_mode]'),
    ('"stage_trace": trace == "trace",',
     '"stage_trace": trace == "trace", "attention_mode": attention_mode, "active_slots": int(slots),'),
    ("*sys.argv[4:9]", "*sys.argv[4:11]"),
):
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected runner anchor: {anchor}")
    source = source.replace(anchor, replacement)
compile(source, str(path), "exec")
path.write_text(source)
