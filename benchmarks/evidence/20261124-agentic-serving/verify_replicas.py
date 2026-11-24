"""Reconcile per-workflow goodput, tenant cache isolation and observed burst outcomes."""

import argparse
import collections
import json
import math
from pathlib import Path


def read(folder, name):
    return json.loads((folder / name).read_text())


def summarize(folder):
    isolation = read(folder, "isolation.json")
    status = [r["status"] for r in isolation]
    hits = [
        (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        for r in isolation
        if r["status"] == 200
    ]
    if (
        status != [200, 200, 401, 200, 200]
        or hits[::2] != [0, 0]
        or min(hits[1::2]) <= 0
    ):
        raise ValueError("tenant isolation evidence incomplete")
    waves = []
    for cell in read(folder, "capacity.json"):
        records = cell["records"]
        correct = [r for r in records if r["ok"]]
        good = [
            r
            for r in correct
            if r["first_final_ms"] <= 10000
            and r["last_final_ms"] <= 18000
            and r["final_gap_p95_ms"] <= 150
        ]
        if len(records) != cell["submitted"] or len(good) != cell["slo_completed"]:
            raise ValueError("capacity denominator or SLO total disagrees")
        waves.append(
            {
                **{k: v for k, v in cell.items() if k != "records"},
                "fixture_hashes": [r["fixture_sha256"] for r in records],
                "errors": [
                    {"seed": r["seed"], "reason": r.get("error")}
                    for r in records
                    if not r["ok"]
                ],
                "visible_gap_p95_ms": sorted(r["final_gap_p95_ms"] for r in correct)[
                    math.ceil(len(correct) * 0.95) - 1
                ]
                if correct
                else None,
            }
        )
    burst = read(folder, "tenant-burst.json")
    after, before = burst["after"], burst["before"]
    protected = burst["protected"]
    outcomes = collections.Counter(
        r.get("status", "transport_error") for r in burst["requests"]
    )
    peer_events = protected["turns"][-1]["content_ns"]
    allocation_seconds = 0
    for gpu in (6, 7):
        alloc = read(folder, f"allocation-{gpu}.json")
        allocation_seconds += alloc["elapsed_s"]
        shut = read(folder / f"gpu{gpu}", "shutdown.json")
        if (
            alloc["returncode"]
            or any(
                shut[k]
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            )
            or any(shut["allocation"].values())
        ):
            raise ValueError("allocation or shutdown incomplete")
    return {
        "arm": folder.name,
        "isolation_status": status,
        "authorized_cached_tokens": hits,
        "waves": waves,
        "allocation_gpu_seconds": allocation_seconds,
        "slo_workflows_per_allocated_gpu_hour": 3600
        * sum(w["slo_completed"] for w in waves)
        / allocation_seconds,
        "burst": {
            "statuses": dict(outcomes),
            "protected_correct": protected["ok"],
            "started_during_peer_stream": peer_events[0]
            < burst["burst_start_ns"]
            < peer_events[-1],
            "peer_gap_p95_ms": protected.get("final_gap_p95_ms"),
            "real_client_drops": sum(
                "client_cancel_ns" in r for r in burst["requests"]
            ),
            "counter_delta": {
                k: after["counters"][k] - v for k, v in before["counters"].items()
            },
            "idle_after": all(
                after[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            ),
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    args = p.parse_args()
    results = [
        summarize(args.directory / a)
        for a in "abcdef"
        if (args.directory / a / "gpu7" / "lease-after.json").is_file()
    ]
    if not results:
        raise ValueError("no completed pair")
    for row in results:
        if [w["fixture_hashes"] for w in row["waves"]] != [
            w["fixture_hashes"] for w in results[0]["waves"]
        ]:
            raise ValueError("fixtures differ across policies")
    print(json.dumps({"completed_runs": len(results), "arms": results}, indent=2))


if __name__ == "__main__":
    main()
