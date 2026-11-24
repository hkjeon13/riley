"""Reconstruct all mixed-workload turns and score default promotion on shared GPU time."""

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)

POLICIES = ["round-robin", "observed-capacity", "defer-cold-prefill"]


def check_work(work, policy, *, final_transform=None):
    seed = work["seed"]
    tools, prompt, docs = fixture(seed, work["document_pairs"])
    assert work["fixture_sha256"] == fingerprint([prompt, docs])
    history = history_messages(seed, 100) + [{"role": "user", "content": prompt}]
    assert work["history_entries"] == 100 and work["tool_delay_s"] == 0.1
    assert len(work["turns"]) == 3
    names = ["list_reports", "read_reports", None]
    for idx, (name, row) in enumerate(zip(names, work["turns"])):
        req = turn_request("m", history, tools, name)
        if name is None and final_transform is not None:
            req = final_transform(req)
        assert fingerprint(req) == row["input_sha256"]
        if policy == "round-robin":
            assert "readiness" not in row
        else:
            assert len(row["readiness"]) == 2
        if idx < 2:
            assert (
                row["status"] == 200
                and row["done_ns"]
                and len(row["observed_calls"]) == 1
            )
            call = row["observed_calls"][0]
            assert call["function"]["name"] == name and json.loads(
                call["function"]["arguments"]
            ) == ({} if idx == 0 else {"first": docs[0]["id"], "second": docs[1]["id"]})
            append_tool_result(history, [call], docs, idx)
    row = work["turns"][-1]
    text = row.get("answer", "").replace("*", "")
    values = [d["p95_ms"] for d in docs]
    correct = all(
        re.search(rf"\b{label}\s*:\s*{v}\s*ms\b", text, re.IGNORECASE)
        for label, v in zip(
            ("before", "after", "reduction"), [*values, values[0] - values[1]]
        )
    )
    missing = [d["id"] for d in docs if d["id"].lower() not in text.lower()]
    valid = bool(
        row.get("status") == 200
        and row.get("finish") == "stop"
        and row.get("done_ns")
        and row["usage"]["completion_tokens"] >= 64
        and not missing
        and correct
    )
    assert valid == work["ok"]
    if not valid:
        return {
            "valid": False,
            "slo": False,
            "missing_sources": missing,
            "error": work.get("error"),
        }
    events = row["content_ns"]
    first = (events[0] - work["start_ns"]) / 1e6
    last = (events[-1] - work["start_ns"]) / 1e6
    gap = percentile([(b - a) / 1e6 for a, b in itertools.pairwise(events)], 0.95)
    assert (first, last, gap) == (
        work["first_final_ms"],
        work["last_final_ms"],
        work["final_gap_p95_ms"],
    )
    return {
        "valid": True,
        "slo": first <= 10000 and last <= 18000 and gap <= 150,
        "first_ms": first,
        "last_ms": last,
        "gap_ms": gap,
    }


def tails(rows):
    return {
        k: percentile([row[k] for row in rows], 0.95)
        for k in ("first_ms", "last_ms", "gap_ms")
    }


def scores(root):
    waves = []
    allocations = []
    by_seed = defaultdict(dict)
    cohorts = {}
    for block in range(3):
        pair = root / f"run-{block}"
        allocation = 0
        for gpu in (6, 7):
            rc = json.loads((pair / f"allocation-{gpu}.json").read_text())
            assert rc["returncode"] == 0
            allocation += rc["elapsed_s"]
            assert (
                json.loads((pair / f"gpu{gpu}/lease-after.json").read_text())["phase"]
                == "idle"
            )
            assert (pair / f"gpu{gpu}/binary.sha256").read_text().split()[
                0
            ] == "506b3e2519dd3c3dbb35c05f966b0fab339ae3326c5ecf21b0e8c118c1a015d6"
            shut = json.loads((pair / f"gpu{gpu}/shutdown.json").read_text())
            assert not any(
                shut[k]
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            ) and not any(shut["allocation"].values())
        raw = json.loads((pair / "multi-router.json").read_text())
        assert len(raw) == 6
        for idx, w in enumerate(raw):
            policy = POLICIES[(block + [0, 1, 2, 2, 1, 0][idx]) % 3]
            assert (w["block"], w["wave"], w["policy"]) == (block, idx, policy)
            clients = w["clients"]
            assert len({c["pid"] for c in clients}) == 2
            work = []
            for j, c in enumerate(clients):
                assert [r["seed"] for r in c["records"]] == [
                    761 + 40 * idx + 20 * j + i for i in range(4)
                ]
                assert [r["document_pairs"] for r in c["records"]] == [2, 96, 2, 96]
                work.extend(c["records"])
            hashes = [r["fixture_sha256"] for r in work]
            assert cohorts.setdefault(idx, hashes) == hashes
            elapsed = max(r["end_ns"] for r in work) / 1e9 - min(
                c["scheduled_start_s"] for c in clients
            )
            metrics = []
            for r in work:
                metric = check_work(r, policy)
                by_seed[r["seed"]][policy] = metric
                metrics.append(metric)
            turns = [t for r in work for t in r["turns"]]
            good = sum(m["slo"] for m in metrics)
            waves.append(
                {
                    "block": block,
                    "wave": idx,
                    "policy": policy,
                    "submitted": len(work),
                    "http": dict(
                        Counter(str(t.get("status", "transport")) for t in turns)
                    ),
                    "correct": sum(m["valid"] for m in metrics),
                    "slo": good,
                    "warm_gpu_seconds": 2 * elapsed,
                    "goodput": good * 3600 / (2 * elapsed),
                    "tails": tails([m for m in metrics if m["valid"]]),
                    "destinations": [
                        sum(t["route"]["replica"] == i for t in turns) for i in (0, 1)
                    ],
                    "decisions_with_missing_sample": sum(
                        "readiness" in t and None in t["readiness"] for t in turns
                    ),
                    "hold_count": sum(t.get("routing_hold_ms", 0) > 0 for t in turns),
                    "missing_source_ids_by_seed": {
                        r["seed"]: m["missing_sources"]
                        for r, m in zip(work, metrics)
                        if not m["valid"]
                    },
                    "failed_seeds": [
                        r["seed"] for r, m in zip(work, metrics) if not m["valid"]
                    ],
                }
            )
        allocations.append(
            {
                "block": block,
                "gpu_seconds": allocation,
                "slo_per_allocated_gpu_hour": sum(
                    w["slo"] for w in waves if w["block"] == block
                )
                * 3600
                / allocation,
            }
        )
    assert len(by_seed) == 48 and all(len(p) == 3 for p in by_seed.values())
    groups = {}
    for p in POLICIES:
        ws = [w for w in waves if w["policy"] == p]
        agg = {
            k: sum(w[k] for w in ws)
            for k in (
                "submitted",
                "correct",
                "slo",
                "warm_gpu_seconds",
                "hold_count",
                "decisions_with_missing_sample",
            )
        }
        agg["slo_per_warm_gpu_hour"] = agg["slo"] * 3600 / agg["warm_gpu_seconds"]
        groups[p] = agg
    pairs = []
    candidate = POLICIES[-1]
    for control in POLICIES[:2]:
        for idx in range(6):
            a, b = [
                next(w for w in waves if w["wave"] == idx and w["policy"] == p)
                for p in (control, candidate)
            ]
            common = [
                v
                for s, v in by_seed.items()
                if 761 + 40 * idx <= s < 761 + 40 * idx + 24
                and v[control]["valid"]
                and v[candidate]["valid"]
            ]
            rc = {k: b["tails"][k] / v for k, v in a["tails"].items()}
            cm = {p: tails([r[p] for r in common]) for p in (control, candidate)}
            pairs.append(
                {
                    "control": control,
                    "wave": idx,
                    "tail_ratios": rc,
                    "common_correct": len(common),
                    "common_correct_tails": cm,
                    "tail_gate": max(rc.values()) <= 1.05,
                }
            )
    promoted = all(
        groups[candidate]["correct"] >= groups[p]["correct"]
        and groups[candidate]["slo"] >= groups[p]["slo"]
        and groups[candidate]["slo_per_warm_gpu_hour"]
        > groups[p]["slo_per_warm_gpu_hour"]
        for p in POLICIES[:2]
    ) and all(p["tail_gate"] for p in pairs)
    return {
        "waves": waves,
        "groups": groups,
        "paired": pairs,
        "blocks": allocations,
        "promotion_gate": promoted,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(scores(p.parse_args().root), indent=2))
