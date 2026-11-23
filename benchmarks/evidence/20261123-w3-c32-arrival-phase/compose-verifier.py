"""Reuse the archived identity, request, and timing checks for the two schedules."""
from pathlib import Path
import subprocess
import sys


def without_entry_point(source):
    return source.split('\nif __name__ == "__main__":\n', 1)[0]


here = Path(__file__).resolve().parent
previous = here.parent / "20261123-w3-c32-short-burst-admission"
archived = subprocess.check_output(
    [sys.executable, "-B", str(previous / "compose-verifier.py")], text=True
)
source = without_entry_point(archived)
for old, new in (
    ("\ndef verify(stage):\n", "\ndef verify(stage, expected_schedules):\n"),
    ('    require(digest(stage / "schedules.json") == "a368c89328bba1e8d629e15cc2b55079b16cce55065fcddca4aa370c58482df0", "schedule drift")',
     '    require(read(stage / "schedules.json") == expected_schedules, "schedule drift")'),
):
    if source.count(old) != 1:
        raise ValueError(f"expected one archived verifier anchor: {old}")
    source = source.replace(old, new, 1)

schedule = without_entry_point((here / "schedules.py").read_text())
boundaries = (previous / "queue-boundaries.py").read_text().replace("distribution(", "queue_distribution(")
report = (here / "verify.py").read_text()
composed = "\n\n".join((source, schedule, boundaries, report))
compile(composed, "arrival-phase-verifier", "exec")
print(composed, end="")
