"""Count rankable dispatch opportunities on the observed, unchanged round-robin timeline."""

import argparse
import json
from pathlib import Path


def analyze(rows):
    short = [r for r in rows if r["short"]]
    if len(rows) != 256 or len(short) != 192 or any(r["error"] for r in rows):
        raise ValueError("incomplete measured interval")
    p95 = sorted(r["e2e"] for r in short)[int(0.95 * len(short))]
    tallies = {}
    for ident, items in (
        ("short", short),
        ("long", [r for r in rows if not r["short"]]),
    ):
        seen = []
        for r in items:
            when = r["start"]
            active = [v for v in rows if v["start"] < when < v["start"] + v["e2e"]]
            before = [
                v for v in active if not v["short"] and when < v["start"] + v["ttft"]
            ]
            pick = r["replica"]
            peer = 1 - pick
            counts = [sum(v["replica"] == k for v in active) for k in (0, 1)]
            pending = [sum(v["replica"] == k for v in before) for k in (0, 1)]
            candidate = pending[pick] > 0 and not pending[peer]
            seen.append(
                {
                    "idx": r["index"],
                    "eligible": candidate,
                    "without_more_active": candidate and counts[peer] <= counts[pick],
                    "completion_ms": 1000 * r["e2e"],
                    "tail_short": r["short"] and r["e2e"] >= p95,
                }
            )
        tallies[ident] = {
            "requests": len(items),
            "other_replica_has_zero_waiting_long": sum(r["eligible"] for r in seen),
            "also_no_more_visible_active": sum(r["without_more_active"] for r in seen),
            "tail_short_opportunities": sum(
                r["without_more_active"] and r["tail_short"] for r in seen
            ),
            "tail_short_count": sum(r["tail_short"] for r in seen),
        }
    return tallies


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    root = p.parse_args().root
    print(
        json.dumps(
            {
                f.stem: analyze(json.loads(f.read_text()))
                for f in sorted(root.glob("arm-*-round-robin.json"))
            },
            indent=2,
        )
    )
