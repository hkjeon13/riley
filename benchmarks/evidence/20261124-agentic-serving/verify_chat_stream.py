"""Compare frame serialization and actual HTTP delivery separately on the recorded cases."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(path.read_text())


def nearest_rank(samples, p):
    return sorted(samples)[math.ceil(len(samples) * p) - 1]


def verify(root):
    cases = read(root / "cases.json")
    meta = read(root / "meta.json")
    report = {"normal_a100_coder_replays": meta, "cpu_ns_per_delta": {}, "http": {}}
    for env in ("mac", "linux"):
        result = read(root / f"{env}.json")
        rows = []
        for inputs, timing in zip(cases, result, strict=True):
            n = len(inputs["pieces"])
            assert n == timing["deltas"] and inputs["style"] == timing["style"]
            assert timing["bytes"] > 0 and len(timing["round_ns"]) == 10
            old, new = [
                median(col) / (n * timing["repeats"])
                for col in zip(*timing["round_ns"], strict=True)
            ]
            rows.append({"old": old, "new": new, "saving": old - new})
        report["cpu_ns_per_delta"][env] = rows
    bounds = [
        row["saving"] / (median(src["a100_normal_tpot_ms"]) * 1e6)
        for row, src in zip(report["cpu_ns_per_delta"]["linux"][:4], meta, strict=True)
    ]
    report["linux_cpu_saving_over_observed_a100_interval"] = bounds
    for run in ("0-old", "1-new", "2-new", "3-old"):
        rows = read(root / f"http-{run}.json")
        assert len(rows) == 250
        sizes = {}
        for size in (1, 8, 16):
            selected = [r for r in rows if r["clients"] == size]
            assert len(selected) == size * 10
            assert all(0 < r["first_ns"] <= r["last_ns"] for r in selected)
            sizes[size] = {
                "requests": len(selected),
                **{
                    f"{phase}_p95_ms": nearest_rank(
                        [r[f"{phase}_ns"] for r in selected], 0.95
                    )
                    / 1e6
                    for phase in ("first", "last")
                },
            }
        report["http"][run] = sizes
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(verify(args.raw), indent=2) + "\n")
