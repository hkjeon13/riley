"""Shift the later C32 cohort without changing either cohort's internal spacing."""
import json
from pathlib import Path
import sys


DELAYS_NS = {"original": 0, "late12": 12_000_000}


def shifted_schedules(reference, delay):
    if set(reference) != {"6", "7"}:
        raise ValueError("expected both pinned GPU schedules")
    shifted = {}
    for gpu, schedule in reference.items():
        offsets = schedule["offsets_ns"]
        order = sorted(range(len(offsets)), key=offsets.__getitem__)
        if len(offsets) != 96 or set(order[:32]) != set(range(32)) or min(offsets) != 0:
            raise ValueError("expected the pinned first 32 and later 64 requests")
        updated = [offset + (delay if i >= 32 else 0) for i, offset in enumerate(offsets)]
        if sorted(range(len(updated)), key=updated.__getitem__) != order:
            raise ValueError("shift would reorder client writes")
        shifted[gpu] = {**schedule, "offsets_ns": updated}
    return shifted


if __name__ == "__main__":
    reference_path, variant, output_path = sys.argv[1:]
    reference = json.loads(Path(reference_path).read_bytes())
    with Path(output_path).open("x") as handle:
        json.dump(shifted_schedules(reference, DELAYS_NS[variant]), handle, indent=2, allow_nan=False)
        handle.write("\n")
