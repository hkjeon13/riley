"""Select replay cells for each independent server process."""
import ast
from pathlib import Path

arm = Path("arm.py")
source = arm.read_text()
scope = '''HISTORY = os.environ["W3_RAG_HISTORY"]
if HISTORY == "idle":
    CELLS = {"c32-riley-arrivals": CELLS["c32-riley-arrivals"]}
elif HISTORY != "rag":
    raise ValueError("unknown RAG history")
'''
for before, after in (
    ("OUT = STAGE", scope + "OUT = STAGE"),
    ('out = OUT / os.environ["W3_PREFILL_ORDER"]', 'out = OUT / HISTORY / os.environ["W3_PREFILL_ORDER"]'),
):
    if source.count(before) != 1:
        raise ValueError(f"previous arm changed at {before!r}")
    source = source.replace(before, after)
ast.parse(source)
arm.write_text(source)
