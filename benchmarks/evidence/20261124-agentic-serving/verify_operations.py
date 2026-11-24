"""Verify application validity, interrupted SSE and lease release against separate receipts."""

import argparse
import json
import re
import sqlite3
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def result(work):
    last = work["turns"][-1]
    text = last.get("answer", "")
    plain = text.replace("*", "").lower()
    return {
        "seed": work["seed"],
        "workflow_ok": work["ok"],
        "valid_tool_calls": sum(r.get("call") is not None for r in work["turns"]),
        "http": last["status"],
        "chunks": len(last["content_ns"]),
        "sse_done": "done_ns" in last,
        "finish": last.get("finish"),
        "numeric_fields": dict(
            re.findall(r"\b(before|after|reduction)\s*:\s*(\d+)\s*ms\b", plain)
        ),
        "missing_sources": [
            doc for doc in ("report-A", "report-B") if doc.lower() not in plain
        ],
    }


def summarize(folder):
    raw = read(folder / "observations.json")
    checks = raw["checks"]
    outputs = {
        k: result(raw[k])
        for k in (
            "drained",
            "newly_routed",
            "partial",
            "recovered",
            "rollback_workflow",
        )
    }
    with sqlite3.connect(f"file:{folder / 'actions.sqlite'}?mode=ro", uri=True) as db:
        persisted = None
        columns = {r[1] for r in db.execute("PRAGMA table_info(action)")}
        if "result" in columns:
            saved = [
                (w, t, json.loads(v))
                for w, t, v in db.execute(
                    "SELECT workflow,turn,result FROM action ORDER BY turn"
                )
            ]
            ids = ["report-A", "report-B"]
            if [(w, t) for w, t, _ in saved] != [("logical-1", 0), ("logical-1", 1)]:
                raise ValueError("logical action identities changed")
            fetched = saved[1][2]
            expected_numbers = outputs["recovered"]["numeric_fields"]
            persisted = (
                saved[0][2] == ids
                and [doc["id"] for doc in fetched] == ids
                and [str(doc["p95_ms"]) for doc in fetched]
                == [expected_numbers.get("before"), expected_numbers.get("after")]
            )
        actions = [
            json.loads(s)
            for (s,) in db.execute("SELECT payload FROM action ORDER BY turn")
        ]
    gauges = ("active_requests", "waiting_requests", "kv_allocated_blocks")
    shutdown = {}
    for gen in ("old", "replacement"):
        val = read(folder / gen / "shutdown.json")
        shutdown[gen] = all(val[k] == 0 for k in gauges) and not any(
            val["allocation"].values()
        )
    leases = [
        read(folder / gen / "lease-after.json")
        for gen in ("old", "survivor", "replacement")
    ]
    if not all(s["phase"] == "idle" for s in leases):
        raise ValueError("a lease failed to return to idle")
    if not (
        int((folder / "survivor" / "fault-exit-code").read_text())
        == raw["fault_exit"]
        == 137
    ):
        raise ValueError("process failure missing")
    if [a["name"] for a in actions] != ["list_reports", "read_reports"]:
        raise ValueError("unexpected durable tool effects")
    route_ok = None
    if raw.get("explicit_citations"):
        assignments = {
            "drained": "old",
            "newly_routed": "survivor",
            "partial": "survivor",
            "recovered": "replacement",
            "rollback_workflow": "replacement",
        }
        route_ok = all(
            all(
                r["endpoint"] == (folder / gen / "address").read_text().strip()
                for r in raw[k]["turns"]
            )
            for k, gen in assignments.items()
        )
        if persisted is None:
            raise ValueError("missing durable tool results on retry")
    partial = outputs["partial"]
    if partial["sse_done"] or partial["finish"] is not None or partial["chunks"] < 1:
        raise ValueError("incomplete SSE contract did not occur")
    return {
        "workflows": outputs,
        "process_exit": raw["fault_exit"],
        "replacement_ready_s": raw["replacement_ready_s"],
        "fresh_workflow_after_fault_s": raw["successful_recovery_s"],
        "old_endpoint_closed": raw["old_now_closed"],
        "failed_endpoint_closed": raw["dead_before_generation"],
        "committed_actions": actions,
        "tool_attempts": raw["tool_attempts"],
        "shutdown_zero": shutdown,
        "engineering_checks": checks,
        "durable_results_match_answer": persisted,
        "turn_endpoints_match_generation": route_ok,
        "all_checks_passed": all(checks.values())
        and all(shutdown.values())
        and persisted is not False
        and route_ok is not False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    args = p.parse_args()
    print(json.dumps(summarize(args.folder), indent=2))
