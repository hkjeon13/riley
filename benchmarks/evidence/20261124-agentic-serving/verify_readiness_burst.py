"""Verify all burst arrivals, sampled load, routing outcomes and protected final answers."""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

from agent_workflow import percentile


def load(path):
    return json.loads(path.read_text())


def completed(r):
    return r.get("status") == 200 and bool(
        r.get("done_ns") and r.get("finish") and r.get("content_ns")
    )


def summarize(r):
    burst = r["burst"]
    peers = r["incumbents"]
    release = r["release_ns"]
    decision = r["before_route_ns"]
    target = r["target"]
    assert r["count"] == len(burst) == len({x["index"] for x in burst})
    assert all(completed(p) and p["usage"]["completion_tokens"] == 384 for p in peers)
    assert len(peers) == 8
    assert completed(r["warm"]) and r["warm"]["route"]["replica"] == 0
    assert completed(r["recovery"]) and target["route"]["active"] == [0, 0]
    assert target["input_sha256"] == r["warm"]["input_sha256"]
    assert r["warm"]["usage"]["prompt_tokens"] == 5745
    assert all(x.get("sent_ns") and (completed(x) or x["error"]) for x in burst)
    true_busy = sum(
        p["content_ns"][0] <= decision <= p["content_ns"][-1] for p in peers
    )
    released_busy = sum(
        p["content_ns"][0] <= release <= p["content_ns"][-1] for p in peers
    )
    decisions = target["readiness"]
    polls = [p for p in r["polls"] if release / 1e9 <= p[1] <= decision / 1e9]
    poll_counts = [
        dict(Counter(p[4] for p in polls if p[0] == replica)) for replica in (0, 1)
    ]
    protected = r["protected"]
    final = protected["turns"][-1]
    assert protected["ok"] and len(protected["turns"]) == 3 and completed(final)
    assert final["content_ns"][0] <= release <= final["content_ns"][-1]
    assert all(x["end_ns"] >= x["sent_ns"] >= release for x in burst)
    for i, s in enumerate(target.get("initial_readiness", decisions)):
        if s is not None:
            assert 0 <= s["age_ms"] <= 500
        # Only replies delivered before route selection could influence its result.
        finished = [
            p
            for p in r["polls"]
            if p[0] == i and p[2] <= decision / 1e9 and p[4] == "sample"
        ]
        if s is None and finished:
            assert 1000 * (decision / 1e9 - max(p[1] for p in finished)) >= 499
    gaps = [
        (b - a) / 1e6 for p in peers for a, b in itertools.pairwise(p["content_ns"])
    ]
    attempts = Counter(str(x.get("status", "transport")) for x in burst)
    ok = completed(target)
    if ok:
        assert target["usage"]["completion_tokens"] == 1
    return {
        "gpu": r["busy_gpu"],
        "burst": r["count"],
        "policy": r["policy"],
        "incumbents_at_release": released_busy,
        "incumbents_at_decision": true_busy,
        "burst_sent_before_decision": sum(x["sent_ns"] <= decision for x in burst),
        "burst_http": dict(attempts),
        "burst_completed": sum(completed(x) for x in burst),
        "burst_errors": dict(
            Counter(x.get("error", "").split(":")[0] for x in burst if not completed(x))
        ),
        "last_arrival_ms": max(
            ((x["sent_ns"] - release) / 1e6 for x in burst), default=None
        ),
        "decision_offset_ms": (decision - release) / 1e6,
        "polls_started_before_decision": poll_counts,
        "busy_timeout_before_decision": sum(
            p[0] == 0 and p[4] == "timeout" and p[2] <= decision / 1e9 for p in polls
        ),
        "initial_busy_sample": target.get("initial_readiness", decisions)[0],
        "initial_other_sample": target.get("initial_readiness", decisions)[1],
        "routing_hold_ms": target.get("routing_hold_ms", 0),
        "route_ms": target["route"]["decision_ms"],
        "dispatch_after_protected_last_ms": (
            target["sent_ns"] - final["content_ns"][-1]
        )
        / 1e6,
        "incumbents_at_dispatch": sum(
            p["content_ns"][0] <= target["sent_ns"] <= p["content_ns"][-1]
            for p in peers
        ),
        "busy_sample": decisions[0],
        "other_sample": decisions[1],
        "destination": target["route"]["replica"],
        "http": target.get("status"),
        "target_ok": ok,
        "ttft_ms": (target["content_ns"][0] - target["sent_ns"]) / 1e6 if ok else None,
        "inclusive_ttft_ms": (target["content_ns"][0] - decision) / 1e6 if ok else None,
        "inclusive_end_ms": (target["end_ns"] - decision) / 1e6,
        "target_error": target.get("error"),
        "protected_gap_p95_ms": protected["final_gap_p95_ms"],
        "protected_done_ms": protected["last_final_ms"],
        "protected_ok": protected["ok"],
        "incumbent_gap_p95_ms": percentile(gaps, 0.95),
        "recovery_ttft_ms": (r["recovery"]["content_ns"][0] - r["recovery"]["sent_ns"])
        / 1e6,
    }


def report(root, guarded):
    candidate = "defer-cold-prefill" if guarded else "fresh-capacity"
    raw = load(root / "waves.json")
    rows = [summarize(r) for r in raw]
    order = [
        (g, n, p)
        for g, counts, pol in [
            (6, (0, 16, 64, 160), ("observed-capacity", candidate)),
            (7, (160, 64, 16, 0), (candidate, "observed-capacity")),
        ]
        for n in counts
        for p in pol
    ]
    assert [(r["gpu"], r["burst"], r["policy"]) for r in rows] == order
    pairs = []
    for gpu, n in itertools.product((6, 7), (0, 16, 64, 160)):
        a, b = [
            next(r for r in rows if (r["gpu"], r["burst"], r["policy"]) == (gpu, n, p))
            for p in ("observed-capacity", candidate)
        ]
        ratios = {
            k: b[k] / a[k]
            for k in (
                "protected_gap_p95_ms",
                "protected_done_ms",
                "incumbent_gap_p95_ms",
            )
        }
        peer_pass = all(v <= 1.05 for v in ratios.values())
        measured = all(
            r["incumbents_at_release"] == 8
            and r["incumbents_at_decision"] == 8
            and r["burst_sent_before_decision"] == n
            for r in [a, b]
        )
        placement = b["target_ok"] and b["destination"] == 1
        if n == 0:
            placement = placement and a["target_ok"] and a["destination"] == 1
            if guarded:
                placement = (
                    placement
                    and b["inclusive_ttft_ms"] <= 1.05 * a["inclusive_ttft_ms"]
                )
        elif (guarded or a["destination"] == 0) and a["target_ok"]:
            placement = placement and b["inclusive_ttft_ms"] < a["inclusive_ttft_ms"]
        pairs.append(
            {
                "gpu": gpu,
                "burst": n,
                "protected_ratios": ratios,
                "measured_overlap": measured,
                "peer_gate": peer_pass,
                "placement_gate": placement,
                "both_targets_completed": a["target_ok"] and b["target_ok"],
            }
        )
    for i in [6, 7]:
        assert (root / f"gpu{i}/binary.sha256").read_text().split()[
            0
        ] == "506b3e2519dd3c3dbb35c05f966b0fab339ae3326c5ecf21b0e8c118c1a015d6"
        assert (
            load(root / f"allocation-{i}.json")["returncode"] == 0
            and load(root / f"gpu{i}/lease-after.json")["phase"] == "idle"
        )
        shut = load(root / f"gpu{i}/shutdown.json")
        assert all(
            shut[k] == 0
            for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
        ) and not any(shut["allocation"].values())
    return {
        "waves": rows,
        "paired_gates": pairs,
        "submitted_burst": sum(r["burst"] for r in rows),
        "completed_burst": sum(r["burst_completed"] for r in rows),
        "protected_correct": sum(r["protected_ok"] for r in rows),
        "all_gates_passed": all(
            p["peer_gate"]
            and p["placement_gate"]
            and p["measured_overlap"]
            and (guarded or p["both_targets_completed"])
            for p in pairs
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--protect-reader", action="store_true")
    args = p.parse_args()
    print(json.dumps(report(args.run, args.protect_reader), indent=2))
