"""Charge both allocated replicas for each joint wave and compare all original arrivals."""

import argparse
import hashlib
import json
from pathlib import Path

from agent_workflow import percentile
from join_trace import join
from verify_coder_pressure import (
    check_manifest,
    check_server,
    qualifies,
    reconcile,
    work_latency,
)
from verify_model_decode import natural, read


def aggregate(waves):
    rows = [r for w in waves for r in w["rows"]]
    duration = sum(w["duration_s"] for w in waves)
    n = sum(r["slo"] for r in rows)
    return {
        "slo": n,
        "jobs": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "wave_duration_s": duration,
        "gpu_hours": 2 * duration / 3600,
        "slo_per_gpu_hour": n * 3600 / (2 * duration),
        "completed_per_gpu_hour": sum(bool(r.get("completed")) for r in rows)
        * 3600
        / (2 * duration),
        "last_p95_ms": percentile([r["last_ms"] for r in rows if "last_ms" in r], 0.95),
        "last_max_ms": max(r["last_ms"] for r in rows if "last_ms" in r),
        "reader_first_p95_ms": percentile(
            [r["first_ms"] for r in rows if r["role"] == "reader" and "first_ms" in r],
            0.95,
        ),
        "target_first_p95_ms": percentile(
            [r["first_ms"] for r in rows if r["role"] == "target" and "first_ms" in r],
            0.95,
        ),
        "target_last_p95_ms": percentile(
            [r["last_ms"] for r in rows if r["role"] == "target" and "last_ms" in r],
            0.95,
        ),
        "stream_gap_p95_of_p95_ms": percentile(
            [r["gap_p95_ms"] for r in rows if r.get("gap_p95_ms") is not None], 0.95
        ),
        "invalid": [r["seed"] for r in rows if not r["valid"]],
        "reader_slo": sum(r["slo"] for r in rows if r["role"] == "reader"),
        "target_slo": sum(r["slo"] for r in rows if r["role"] == "target"),
    }


def verify(root, refpath):
    check_manifest(root)
    assert (root / "client.exit").read_text().strip() == "0"
    prior = read(refpath)
    sha = (root / "trace-server.sha256").read_text().split()[0]
    assert sha == "8db3d5c634bab3ef5a4d3178e680db0c3c67beb781ecd042067654f2871bd990"
    gw = (6, 7)
    logs = []
    args = set()
    lifetime = 0
    for gpu in gw:
        d = root / f"gpu{gpu}"
        check_server(d, sha, gpu)
        args.add((d / "server-arguments.txt").read_text())
        assert read(d / "warm.json")["ok"]
        logs.append((d / "server.log").read_bytes())
        lifetime += (
            int((root / f"gpu{gpu}.end").read_text())
            - int((root / f"gpu{gpu}.start").read_text())
        ) / 1e9
    assert len(args) == 1
    out = []
    for k, (raw, (size, base)) in enumerate(
        zip(
            read(root / "waves.json"),
            read(root / "coder-pressure-registration.json")["waves"],
            strict=True,
        )
    ):
        assert (raw["size"], raw["base"]) == (size, base)
        wave_steps = []
        slices = []
        for i in (0, 1):
            marks = {key: raw[key][i] for key in ("before", "after")}
            wave_steps.append(reconcile(marks, logs[i]))
            slices.append(
                logs[i][
                    marks["before"]["log_offset"] : marks["after"]["log_offset"]
                ].decode()
            )
        rows = []
        for i, j in zip(range(size), raw["rows"], strict=True):
            device = i % 2
            reader = i % 4 == 0
            seed = base + i
            work = j["work"]
            assert (j["index"], j["gpu"]) == (i, 6 + device)
            schedule = raw["origin"] + 30_000_000 * i
            assert j["scheduled"] == schedule and work["end_ns"] <= raw["drained"]
            f = natural(
                work,
                seed,
                2 if reader else 96,
                0 if reader else 200,
                delay=0 if reader else 0.5,
            )
            t = work_latency(work, schedule)
            cache = None
            if t.get("completed") and f["verified_tool_calls"] == 2:
                trace = join(slices[device], [work], details=True)
                assert [r["turn"] for r in trace] == [0, 1, 2]
                usage = work["turns"][-1]["usage"]
                cache = (usage.get("prompt_tokens_details") or {}).get(
                    "cached_tokens", 0
                )
                assert usage["prompt_tokens"] - cache == trace[-1]["prefill_rows"]
            same = {}
            for gpu in gw:
                baseline = prior[str(gpu)]["off"]["waves"][k]["rows"][i]
                assert baseline["seed"] == seed
                same[gpu] = baseline.get("text_hash") == t.get("text_hash")
            rows.append(
                {
                    "seed": seed,
                    "gpu": j["gpu"],
                    "role": "reader" if reader else "target",
                    "valid": f["valid"],
                    "slo": qualifies(f["valid"], t),
                    "calls": f["verified_tool_calls"],
                    "final_cached_tokens": cache,
                    "same_final_text_as_single": same,
                    "error": work.get("error"),
                    **t,
                }
            )
        out.append(
            {
                "size": size,
                "base": base,
                "duration_s": (raw["drained"] - raw["origin"]) / 1e9,
                "gpu_steps": wave_steps,
                "rows": rows,
            }
        )
    total = aggregate(out)
    total["inside_lease_gpu_seconds"] = lifetime
    total["including_startup_slo_per_gpu_hour"] = total["slo"] * 3600 / lifetime
    high = aggregate(out[2:])
    checks = {}
    for gpu in gw:
        a = prior[str(gpu)]["off"]["high"]
        checks[gpu] = {
            "preserved_valid": set(high["invalid"]) <= set(a["invalid_seeds"]),
            "more_slo": high["slo"] > a["slo"],
            "more_slo_per_gpu_hour": high["slo_per_gpu_hour"]
            > a["goodput_per_gpu_hour"],
        }
    return {
        "reference_sha256": hashlib.sha256(refpath.read_bytes()).hexdigest(),
        "waves": out,
        "all": total,
        "high": high,
        "by_wave": [aggregate([w]) for w in out],
        "high_checks": checks,
        "passed": all(all(v.values()) for v in checks.values()),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).with_name("coder-pressure-verified.json"),
    )
    args = p.parse_args()
    print(json.dumps(verify(args.directory, args.reference), indent=2))
