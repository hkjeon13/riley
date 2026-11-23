"""Adapt the verified steady-256 comparison to the short-only scheduler policy."""
from pathlib import Path
import subprocess
import sys


here = Path(__file__).resolve().parent
parent = here.parent / "20261123-w3-c32-steady256-burst"
source = subprocess.check_output([sys.executable, "-B", str(parent / "compose-verifier.py")], text=True)
for old, new in (
    ('BASE = "s256-d32-p512"', 'BASE = "short-d0"'),
    ('CANDIDATES = ("s256-d8-p64", "s256-d16-p64")', 'CANDIDATES = ("short-d8", "short-d16")'),
    ('BINARY = "d9f7e375c4d2434e95e0a3b8ac9412919f777f5f4bd295548590620c2c8e809c"\n', ''),
    ('    require(build_checks["artifacts"]["target/release/riley"] == BINARY, "binary changed")',
     '''    require(digest(build / "target/release/riley") == build_checks["artifacts"]["target/release/riley"]
            and digest(build / "scheduler-test.log") == build_checks["artifacts"]["scheduler-test.log"],
            "measured binary or CPU proof changed")
    require("1 passed; 0 failed" in (build / "scheduler-test.log").read_text(),
            "short-only scheduler proof did not pass")'''),
    ('            steady, decode, prompt = (int(part[1:]) for part in layout.split("-"))',
     '''            steady, decode, prompt = 256, 32, 512
            short_limit = int(layout.removeprefix("short-d"))'''),
    ('                    and receipt["stage_trace"] is False, f"run configuration differs: {tag}")',
     '''                    and receipt["stage_trace"] is False and receipt["short_burst_decode_limit"] == short_limit,
                    f"run configuration differs: {tag}")'''),
    ('max(map(int, shapes)) <= (steady if canonical_control else 1024)',
     'max(map(int, shapes)) <= (steady if canonical_control else 512)'),
    ('                "steady_budget": steady, "burst_decode_cap": decode, "burst_prompt_floor": prompt,',
     '''                "steady_budget": steady, "burst_decode_cap": decode, "burst_prompt_floor": prompt,
                "short_burst_decode_limit": short_limit,'''),
):
    if source.count(old) != 1:
        raise ValueError(f"expected one archived verifier anchor: {old}")
    source = source.replace(old, new, 1)
compile(source, "short-only-burst-verifier", "exec")
print(source, end="")
