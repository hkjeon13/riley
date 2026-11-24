"""Reconstruct tool histories and audit natural final answers, overlap and paired user latency."""

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)
from verify_readiness_burst import completed, load


def measure(raw):
    w = raw["workflow"]
    seed = w["seed"]
    t = raw["target"]
    reader = raw["protected"]
    last = reader["turns"][-1]
    reqs = w["turns"]
    assert len(reqs) == 3 and t["input_sha256"] == reqs[-1]["input_sha256"]
    tools, ask, docs = fixture(seed, 96)
    messages = history_messages(seed, 200) + [{"role": "user", "content": ask}]
    canonical = list(messages)
    assert w["fixture_sha256"] == fingerprint([ask, docs])
    for i, name in enumerate(["list_reports", "read_reports", None]):
        turn = reqs[i]
        req = turn_request("m", messages, tools, name)
        assert fingerprint(req) == turn["input_sha256"]
        if name:
            assert completed_tool(turn)
            if i == 1:
                preview = {**req, "tool_choice": "none", "max_tokens": 1}
                assert raw["warm"]["input_sha256"] == fingerprint(
                    preview
                ) and completed(raw["warm"])
            call = turn["call"]
            assert call["function"]["name"] == name
            assert json.loads(call["function"]["arguments"]) == (
                {} if i == 0 else {"first": docs[0]["id"], "second": docs[1]["id"]}
            )
            append_tool_result(messages, [call], docs, i)
            append_tool_result(canonical, [{**call, "id": f"step-{i}"}], docs, i)
    assert raw["warm"]["route"]["replica"] == 0 and t["route"]["active"] == [0, 0]
    assert raw["warm"]["usage"]["prompt_tokens"] >= 2048
    user = reqs[-1].get("answer", "").replace("*", "")
    values = [d["p95_ms"] for d in docs]
    correct = all(
        re.search(rf"\b{k}\s*:\s*{v}\s*ms\b", user, re.IGNORECASE)
        for k, v in zip(
            ["before", "after", "reduction"], [*values, values[0] - values[1]]
        )
    ) and all(d["id"].lower() in user.lower() for d in docs)
    correct = bool(
        correct
        and completed(t)
        and t["finish"] == "stop"
        and t["usage"]["completion_tokens"] >= 64
    )
    assert correct == w["ok"]
    peers = raw["incumbents"]
    assert len(peers) == 8 and all(completed(x) for x in peers)
    release, begin = raw["release_ns"], raw["before_route_ns"]
    assert all(
        p["content_ns"][0] <= release <= begin <= p["content_ns"][-1] for p in peers
    )
    assert reader["ok"] and last["content_ns"][0] <= release <= last["content_ns"][-1]
    assert raw["count"] == len(raw["burst"]) and all(
        release <= r["sent_ns"] <= begin for r in raw["burst"]
    )
    assert (
        completed(raw["recovery"])
        and reqs[1]["tool_start_ns"] <= release <= reqs[1]["tool_ready_ns"]
    )
    gaps = [
        (b - a) / 1e6 for p in peers for a, b in itertools.pairwise(p["content_ns"])
    ]
    success = completed(t)
    first = (t["content_ns"][0] - w["start_ns"]) / 1e6 if success else None
    end = (t["content_ns"][-1] - w["start_ns"]) / 1e6 if success else None
    if correct:
        assert first == w["first_final_ms"] and end == w["last_final_ms"]
    usage = t.get("usage", {})
    return {
        "seed": seed,
        "answer_sha256": fingerprint(reqs[-1]["answer"])
        if reqs[-1].get("answer")
        else None,
        "gpu": raw["busy_gpu"],
        "burst": raw["count"],
        "policy": raw["policy"],
        "canonical_final_sha256": fingerprint(
            turn_request("m", canonical, tools, None)
        ),
        "preview_tokens": raw["warm"]["usage"]["prompt_tokens"],
        "final_tokens_in": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "final_tokens_out": usage.get("completion_tokens"),
        "initial_busy_sample": t.get("initial_readiness", t["readiness"])[0],
        "initial_other_sample": t.get("initial_readiness", t["readiness"])[1],
        "replica": t["route"]["replica"],
        "hold_ms": t.get("routing_hold_ms", 0),
        "after_tool_return_first_ms": (t["content_ns"][0] - reqs[1]["tool_ready_ns"])
        / 1e6
        if success
        else None,
        "after_tool_return_last_ms": (t["content_ns"][-1] - reqs[1]["tool_ready_ns"])
        / 1e6
        if success
        else None,
        "tool_wait_ms": (reqs[1]["tool_ready_ns"] - reqs[1]["tool_start_ns"]) / 1e6,
        "first_final_ms": first,
        "last_final_ms": end,
        "http": t.get("status"),
        "correct": correct,
        "gap_p95_ms": w.get("final_gap_p95_ms"),
        "error": w.get("error"),
        "slo": bool(
            correct and first <= 10000 and end <= 18000 and w["final_gap_p95_ms"] <= 150
        ),
        "protected_gap_ms": reader["final_gap_p95_ms"],
        "protected_done_ms": reader["last_final_ms"],
        "incumbent_gap_ms": percentile(gaps, 0.95),
        "burst_status": dict(
            Counter(str(x.get("status", "transport")) for x in raw["burst"])
        ),
    }


def completed_tool(t):
    return (
        t["status"] == 200
        and t["done_ns"]
        and t["finish"] in ("stop", "tool_calls")
        and len(t["observed_calls"]) == 1
    )


def audit(root):
    result = []
    for i, name in enumerate(["a", "b"]):
        pair = root / name
        rows = load(pair / "natural.json")
        assert len(rows) == 4
        for idx, r in enumerate(rows):
            assert (
                r["policy"]
                == ["observed-capacity", "defer-cold-prefill"][(idx + i) % 2]
            )
            assert (r["busy_gpu"], r["count"], r["workflow"]["seed"]) == [
                (6, 16, 761),
                (7, 160, 18081),
                (7, 16, 19959),
                (6, 160, 762),
            ][idx]
            result.append(measure(r))
        for gpu in (6, 7):
            assert (
                load(pair / f"allocation-{gpu}.json")["returncode"] == 0
                and load(pair / f"gpu{gpu}/lease-after.json")["phase"] == "idle"
            )
            shut = load(pair / f"gpu{gpu}/shutdown.json")
            assert all(
                shut[k] == 0
                for k in ("active_requests", "waiting_requests", "kv_allocated_blocks")
            ) and not any(shut["allocation"].values())
            assert (pair / f"gpu{gpu}/binary.sha256").read_text().split()[0] == (
                root / "server.sha256"
            ).read_text().split()[0]
    gates = []
    for seed in [761, 18081, 19959, 762]:
        a, b = [
            next(r for r in result if r["seed"] == seed and r["policy"] == p)
            for p in ["observed-capacity", "defer-cold-prefill"]
        ]
        assert (
            a["canonical_final_sha256"] == b["canonical_final_sha256"]
            and a["preview_tokens"] == b["preview_tokens"]
        )
        ratios = {
            k: b[k] / a[k]
            for k in ("protected_gap_ms", "protected_done_ms", "incumbent_gap_ms")
        }
        faster = bool(
            b["correct"]
            and (
                not a["correct"]
                or (
                    b["first_final_ms"] < a["first_final_ms"]
                    and b["last_final_ms"] < a["last_final_ms"]
                )
            )
        )
        gates.append(
            {
                "seed": seed,
                "peer_ratios": ratios,
                "target_gate": faster,
                "peer_gate": max(ratios.values()) <= 1.05,
                "candidate_slo": b["slo"],
            }
        )
    return {
        "runs": result,
        "gates": gates,
        "all_passed": all(
            g["target_gate"] and g["peer_gate"] and g["candidate_slo"] for g in gates
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    print(json.dumps(audit(p.parse_args().root), indent=2))
