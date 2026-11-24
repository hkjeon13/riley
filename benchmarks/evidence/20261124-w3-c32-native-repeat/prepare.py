"""Use distinct directories for repeated native runs without changing device identity."""
import ast
from pathlib import Path

arm = Path("arm.py")
source = arm.read_text()
for before, after in (
    ('out = OUT / HISTORY / os.environ["W3_PREFILL_ORDER"]',
     'out = OUT / HISTORY / os.environ["W3_REPEAT_PASS"] / os.environ["W3_PREFILL_ORDER"]'),
    ('        "--env", "RILEY_W3_EXPORT_OBSERVATIONS=1",\n', ''),
):
    if source.count(before) != 1:
        raise ValueError(f"parent arm differs at {before!r}")
    source = source.replace(before, after)
ast.parse(source)
arm.write_text(source)
