"""Measure all offered jobs, including time spent waiting for workflow admission."""

import argparse
from itertools import pairwise
import json
from pathlib import Path

from agent_workflow import percentile
from verify_coder_pressure import check_manifest, check_server, qualifies, work_latency
from verify_model_decode import natural, read


def summary(waves):
    rows = [r for w in waves for r in w["rows"]]
    readers = [r for r in rows if r["role"] == "reader"]
    targets = [r for r in rows if r["role"] == "target"]
    secs = sum(w["duration_s"] for w in waves)

    def tail(group, field):
        return percentile([r[field] for r in group if r.get(field) is not None], 0.95)

    n = sum(r["slo"] for r in rows)
    return {
        "valid": sum(r["valid"] for r in rows),
        "slo": n,
        "total": len(rows),
        "goodput": n * 3600 / secs,
        "completed_per_gpu_hour": sum(bool(r.get("completed")) for r in rows)
        * 3600
        / secs,
        "duration_s": secs,
        "reader_first_p95_ms": tail(readers, "first_ms"),
        "reader_gap_p95_of_p95_ms": tail(readers, "gap_p95_ms"),
        "last_p95_ms": tail(rows, "last_ms"),
        "last_max_ms": max(r["last_ms"] for r in rows if "last_ms" in r),
        "long_queue_p95_ms": tail(targets, "queue_ms"),
        "reader_slo": sum(r["slo"] for r in readers),
        "invalid": [r["seed"] for r in rows if not r["valid"]],
        "all_completed": all(r.get("completed") for r in rows),
    }


def verify(root):
    check_manifest(root)
    conf = read(root / "coder-window-registration.json")
    out = {}
    comparisons = {}
    for gpu, order in conf["order"].items():
        d = root / f"gpu{gpu}"
        check_server(
            d, "13b589df5c19a33c685ca1388398905f46a6cc566cbd1bed68f06b798080ca9f", gpu
        )
        assert read(d / "warm.json")["ok"]
        c = list(
            (mode, size, base, group)
            for group, (mode, specs) in enumerate(
                zip(order, conf["groups"], strict=True)
            )
            for size, base in specs
        )
        for (mode, size, base, group), wave in zip(
            c, read(d / "waves.json"), strict=True
        ):
            assert (wave["mode"], wave["size"], wave["base"]) == (mode, size, base)
            jobs = []
            events = []
            grants = []
            for i, j in zip(range(size), wave["rows"], strict=True):
                assert i == j["index"]
                work = j["work"]
                seed = base + i
                reader = i % 4 == 0
                scheduled = wave["origin"] + i * 30_000_000
                assert (
                    scheduled == j["scheduled"]
                    and scheduled
                    <= j["arrived"]
                    <= j["admitted"]
                    <= work["start_ns"]
                    <= work["end_ns"]
                    <= j["released"]
                    <= wave["drained"]
                )
                assert ("reader" if reader else "target") == j["role"]
                if not reader:
                    events.extend([(j["admitted"], 1), (j["released"], -1)])
                    grants.append(j["admitted"])
                facts = natural(
                    work,
                    seed,
                    2 if reader else 96,
                    0 if reader else 200,
                    delay=0 if reader else 0.5,
                )
                t = work_latency(work, scheduled)
                jobs.append(
                    {
                        "seed": seed,
                        "role": j["role"],
                        "valid": facts["valid"],
                        "calls": facts["verified_tool_calls"],
                        "slo": qualifies(facts["valid"], t),
                        "error": work.get("error"),
                        "queue_ms": (j["admitted"] - j["arrived"]) / 1e6,
                        "arrival_slip_ms": (j["arrived"] - scheduled) / 1e6,
                        **t,
                    }
                )
            peak = active = 0
            for _, delta in sorted(events):
                active += delta
                peak = max(peak, active)
            assert active == 0
            if mode == "window4":
                assert peak <= 4 and all(a <= b for a, b in pairwise(grants))
            w = {
                "size": size,
                "base": base,
                "max_active_long": peak,
                "duration_s": (wave["drained"] - wave["origin"]) / 1e9,
                "rows": jobs,
            }
            out.setdefault(group, {}).setdefault(mode, {"gpu": gpu, "waves": []})[
                "waves"
            ].append(w)
    for group, policies in out.items():
        for data in policies.values():
            data["small"] = summary(data["waves"][:1])
            data["high"] = summary(data["waves"][1:])
        a, b = [policies[k]["high"] for k in ("unlimited", "window4")]
        comparisons[group] = {
            "preserve_valid": set(b["invalid"]) <= set(a["invalid"]),
            "increase_slo": b["slo"] > a["slo"],
            "increase_goodput": b["goodput"] > a["goodput"],
            "protect_reader_first": b["reader_first_p95_ms"]
            <= 1.05 * a["reader_first_p95_ms"],
            "protect_reader_gap": b["reader_gap_p95_of_p95_ms"]
            <= 1.05 * a["reader_gap_p95_of_p95_ms"],
            "protect_all_tail": b["last_p95_ms"] <= 1.05 * a["last_p95_ms"],
            "drain_all": b["all_completed"],
        }
    return {
        "groups": out,
        "promotion_checks": comparisons,
        "passed": all(all(c.values()) for c in comparisons.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    print(json.dumps(verify(p.parse_args().directory), indent=2))
