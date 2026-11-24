"""Recompute all placement and protected-stream gates from HTTP/SSE receipts."""

import argparse
import itertools
import json
import statistics
from pathlib import Path

from agent_workflow import percentile


def load(path):
    return json.loads(path.read_text())


def measure(row):
    target = row["target"]
    initial = row["initial"]
    peers = row["peers"]
    assert len(peers) == row["active"] and row["router_pid"] != row["peer_pid"]
    assert all(
        r["status"] == 200 and r["done_ns"] and r["content_ns"] and r["finish"]
        for r in [initial, target, *peers]
    )
    assert target["input_sha256"] == initial["input_sha256"]
    assert initial["usage"]["prompt_tokens"] == target["usage"]["prompt_tokens"]
    assert all(r["usage"]["completion_tokens"] == 1 for r in [initial, target])
    assert all(r["usage"]["completion_tokens"] == 192 for r in peers)
    assert initial["route"]["replica"] == 0 and target["route"]["active"] == [0, 0]
    assert all(
        r["content_ns"][0] <= target["sent_ns"] < r["content_ns"][-1] for r in peers
    )
    gaps = [
        (b - a) / 1e6 for r in peers for a, b in itertools.pairwise(r["content_ns"])
    ]
    cache = target["usage"].get("prompt_tokens_details") or {}
    return {
        "warm_gpu": row["warm_gpu"],
        "policy": row["policy"],
        "external_streams": len(peers),
        "input_tokens": target["usage"]["prompt_tokens"],
        "selected_warm": target["route"]["replica"] == 0,
        "readiness": target.get("readiness"),
        "cached_tokens": cache.get("cached_tokens"),
        "first_ms": (target["content_ns"][0] - target["sent_ns"]) / 1e6,
        "done_ms": (target["done_ns"] - target["sent_ns"]) / 1e6,
        "peer_pooled_gap_p95_ms": percentile(gaps, 0.95),
        "peer_worst_stream_gap_p95_ms": max(
            percentile(
                [(b - a) / 1e6 for a, b in itertools.pairwise(r["content_ns"])], 0.95
            )
            for r in peers
        ),
    }


def summarize(pair):
    data = load(pair / "cross-router.json")
    rows = [measure(row) for row in data]
    assert len(rows) == 12 and len({r["tag"] for r in data}) == 12
    assert len({r["input_tokens"] for r in rows}) == 1
    order = [
        (gpu, n, policy)
        for gpu, loads, policies in [
            (6, (4, 7, 8), ("capacity", "observed-capacity")),
            (7, (8, 7, 4), ("observed-capacity", "capacity")),
        ]
        for n in loads
        for policy in policies
    ]
    assert [(r["warm_gpu"], r["external_streams"], r["policy"]) for r in rows] == order
    grouped = {}
    for row in rows:
        grouped.setdefault((row["warm_gpu"], row["external_streams"]), {})[
            row["policy"]
        ] = row
    gates = []
    for (gpu, count), arms in sorted(grouped.items()):
        control, candidate = arms["capacity"], arms["observed-capacity"]
        first = candidate["first_ms"] / control["first_ms"]
        peer = candidate["peer_pooled_gap_p95_ms"] / control["peer_pooled_gap_p95_ms"]
        gates.append(
            {
                "warm_gpu": gpu,
                "external_streams": count,
                "correct_destination": candidate["selected_warm"] == (count < 8),
                "target_ttft_ratio": first,
                "target_passed": first < 1 if count == 8 else first <= 1.05,
                "peer_gap_ratio": peer,
                "peer_passed": peer <= 1.05,
            }
        )
    assert len(gates) == 6
    fleet = load(pair / "inventory-replay.json")
    assert fleet["policy"] == "observed-capacity"
    replay = []
    for wave in fleet["waves"]:
        rec = wave["records"]
        correct = [r for r in rec if r["ok"]]
        good = sum(
            r["first_final_ms"] <= 10000
            and r["last_final_ms"] <= 18000
            and r["final_gap_p95_ms"] <= 150
            for r in correct
        )
        assert (len(rec), len(correct), good) == (
            wave["submitted"],
            wave["correct"],
            wave["slo_completed"],
        )
        replay.append(
            {
                k: wave[k]
                for k in (
                    "submitted",
                    "correct",
                    "slo_completed",
                    "first_p95_ms",
                    "last_p95_ms",
                )
            }
        )
    binaries = []
    for i in (6, 7):
        assert load(pair / f"allocation-{i}.json")["returncode"] == 0
        assert load(pair / f"gpu{i}/lease-after.json")["phase"] == "idle"
        assert load(pair / f"gpu{i}/metrics.json")["active_requests"] == 0
        shutdown = load(pair / f"gpu{i}/shutdown.json")
        assert all(
            shutdown[k] == 0
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        )
        assert all(v == 0 for v in shutdown["allocation"].values())
        binaries.append((pair / f"gpu{i}/binary.sha256").read_text().split()[0])
    assert len(set(binaries)) == 1
    return {
        "server_sha256": binaries[0],
        "placements": rows,
        "paired_gates": gates,
        "replay": replay,
        "all_placement_and_peer_gates_passed": all(
            g["correct_destination"] and g["target_passed"] and g["peer_passed"]
            for g in gates
        ),
        "peer_stream_count": sum(r["external_streams"] for r in rows),
        "observation_age_median_ms": statistics.median(
            s["age_ms"] for r in rows for s in (r["readiness"] or []) if s is not None
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pair", type=Path)
    print(json.dumps(summarize(p.parse_args().pair), indent=2))
