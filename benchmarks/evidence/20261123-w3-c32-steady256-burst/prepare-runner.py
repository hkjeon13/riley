"""Replay the same Coder workload with explicit prefill burst thresholds."""
from pathlib import Path
import sys


path = Path(sys.argv[1])
source = path.read_text()
for anchor, replacement in (
    ("workers, trace):", "workers, trace, decode_cap, prompt_floor):"),
    ('    arm = f"{case}-{budget}-w{workers}-{trace}-gpu{index}"',
     '    require((budget, decode_cap, prompt_floor) in {("256", "32", "512"),\n'
     '                                                       ("256", "8", "64"), ("256", "16", "64")}\n'
     '            and workers == "64" and trace == "control", "unexpected burst comparison")\n'
     '    arm = f"{case}-s{budget}-d{decode_cap}-p{prompt_floor}-gpu{index}"'),
    ('    argv[argv.index("--prefill-burst-steady-budget") + 1] = budget',
     '    argv[argv.index("--prefill-burst-steady-budget") + 1] = budget\n'
     '    argv[argv.index("--prefill-burst-max-decode-rows") + 1] = decode_cap\n'
     '    argv[argv.index("--prefill-burst-min-prompt-tokens") + 1] = prompt_floor'),
    ('"stage_trace": trace == "trace",',
     '"stage_trace": trace == "trace", "burst_decode_cap": int(decode_cap), "burst_prompt_floor": int(prompt_floor),'),
    ("*sys.argv[4:9]", "*sys.argv[4:11]"),
):
    if source.count(anchor) != 1:
        raise ValueError(f"unexpected runner anchor: {anchor}")
    source = source.replace(anchor, replacement)
compile(source, str(path), "exec")
path.write_text(source)
