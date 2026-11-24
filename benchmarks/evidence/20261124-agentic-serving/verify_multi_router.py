"""Score cross-process agent sessions against their shared GPU wall-clock window."""

import argparse
import itertools
import json
from pathlib import Path

from agent_workflow import fixture, percentile


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    waves = []
    cohorts = {}
    allocated = []
    routes_by_seed = {}
    for block, directory in enumerate(["first", "second"]):
        pair = root / directory
        jobs = [read(pair / f"allocation-{i}.json") for i in (6, 7)]
        assert all(j["returncode"] == 0 for j in jobs)
        assert all(
            read(pair / f"gpu{i}/lease-after.json")["phase"] == "idle" for i in (6, 7)
        )
        for i in (6, 7):
            shutdown = read(pair / f"gpu{i}/shutdown.json")
            assert all(
                shutdown[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            assert all(v == 0 for v in shutdown["allocation"].values())
        busy_time = sum(j["elapsed_s"] for j in jobs)
        completed = 0
        for expected, row in enumerate(read(pair / "multi-router.json")):
            wave, policy = row["wave"], row["policy"]
            assert wave == expected and row["block"] == block
            assert policy == ["capacity", "observed-capacity"][(block + wave) % 2]
            clients = row["clients"]
            assert len({c["pid"] for c in clients}) == 2
            records = [r for c in clients for r in c["records"]]
            seeds = sorted((r["seed"], r["fixture_sha256"]) for r in records)
            assert len(seeds) == len(set(seeds)) == 8
            assert seeds == cohorts.setdefault(wave, seeds)
            start = min(c["scheduled_start_s"] for c in clients)
            duration = max(r["end_ns"] for r in records) / 1e9 - start
            for r in records:
                routes_by_seed.setdefault(r["seed"], []).append(
                    [t.get("route", {}).get("replica") for t in r["turns"]]
                )
            valid = [r for r in records if r["ok"]]
            good = sum(
                r["first_final_ms"] <= 10000
                and r["last_final_ms"] <= 18000
                and r["final_gap_p95_ms"] <= 150
                for r in valid
            )
            completed += good
            turns = list(itertools.chain.from_iterable(r["turns"] for r in records))
            waves.append(
                {
                    "policy": policy,
                    "block": block,
                    "wave": wave,
                    "submitted": len(records),
                    "correct": len(valid),
                    "slo_completed": good,
                    "shared_window_s": duration,
                    "slo_per_warm_gpu_hour": good * 3600 / (2 * duration),
                    "first_p95_ms": percentile(
                        [r["first_final_ms"] for r in valid], 0.95
                    ),
                    "last_p95_ms": percentile(
                        [r["last_final_ms"] for r in valid], 0.95
                    ),
                    "gap_p95_ms": percentile(
                        [r["final_gap_p95_ms"] for r in valid], 0.95
                    ),
                    "arrival_origin_skew_ms": abs(
                        clients[1]["scheduled_start_s"]
                        - clients[0]["scheduled_start_s"]
                    )
                    * 1000,
                    "http_calls": len(turns),
                    "http_200": sum(t.get("status") == 200 for t in turns),
                    "http_429": sum(t.get("status") == 429 for t in turns),
                    "by_replica": [
                        sum(t.get("route", {}).get("replica") == i for t in turns)
                        for i in range(2)
                    ],
                    "failures": [
                        {
                            "seed": r["seed"],
                            "error": r.get("error"),
                            "missing_source_ids": [
                                doc["id"]
                                for doc in fixture(r["seed"], r["document_pairs"])[2]
                                if doc["id"].lower()
                                not in r["turns"][-1].get("answer", "").lower()
                            ],
                        }
                        for r in records
                        if not r["ok"]
                    ],
                }
            )
        allocated.append(
            {
                "block": block,
                "allocated_gpu_seconds": busy_time,
                "slo_completed_per_allocated_gpu_hour": completed * 3600 / busy_time,
            }
        )
    assert len(waves) == 4 and all(len(c) == 8 for c in cohorts.values())
    gates = []
    for wave in (0, 1):
        policies = {r["policy"]: r for r in waves if r["wave"] == wave}
        a, b = policies["capacity"], policies["observed-capacity"]
        ratios = {
            k: b[k] / a[k]
            for k in (
                "first_p95_ms",
                "last_p95_ms",
                "gap_p95_ms",
                "slo_per_warm_gpu_hour",
            )
        }
        passed = (
            b["correct"] >= a["correct"]
            and b["slo_completed"] >= a["slo_completed"]
            and ratios["slo_per_warm_gpu_hour"] > 1
            and all(
                ratios[k] <= 1.05 for k in ("first_p95_ms", "last_p95_ms", "gap_p95_ms")
            )
        )
        gates.append({"wave": wave, "candidate_over_control": ratios, "passed": passed})
    return {
        "waves": waves,
        "paired_gates": gates,
        "allocated": allocated,
        "matched_sessions": len(routes_by_seed),
        "same_destinations_all_turns": sum(a == b for a, b in routes_by_seed.values()),
        "all_gates_passed": all(g["passed"] for g in gates),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(summarize(p.parse_args().root), indent=2))
