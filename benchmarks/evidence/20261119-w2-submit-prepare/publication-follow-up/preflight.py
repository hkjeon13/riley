"""Select explicit publication policy on the pinned W2 SmolLM2 runner."""

from pathlib import Path
import sys


MODES = ("immediate-first", "after-first", "after-second", "immediate-second")
TRACE_MODES = tuple(name + "-trace" for name in MODES)
ORDER = MODES + TRACE_MODES
BINARY = "306a4b25f35ddd531fea9b7219484d5c10c5752084e312140dcfd15f5b1902b8"


def rewrite(source):
    path = Path(source)
    original = path.read_text()
    old_order = 'MODES = ("base-first", "ordered-first", "ordered-second", "base-second", "base-trace", "ordered-trace")'
    old_args = '"--event-publication", "after-submit-v1",'
    if original.count(old_order) != 1 or original.count(old_args) != 1:
        raise ValueError("benchmark runner does not match the reviewed source")
    updated = original.replace(old_order, f"MODES = {ORDER!r}").replace(
        old_args, '"--event-publication", "after-submit-v1" if mode.startswith("after-") else "immediate-v1",')
    compile(updated, str(path), "exec")
    path.write_text(updated)


if __name__ == "__main__":
    rewrite(sys.argv[1])
