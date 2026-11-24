"""Summarize a workflow arm, retaining unsuccessful work in the denominator."""

import argparse
import itertools
import json
from pathlib import Path

from agent_workflow import percentile


def final_latency_breakdown(record):
    """Partition client time through the last visible response content."""
    turns = record["turns"]
    if len(turns) < 1 or not turns[-1].get("content_ns"):
        return None
    final = turns[-1]
    start = record.get("start_ns")
    sent = final.get("sent_ns")
    first, last = final["content_ns"][0], final["content_ns"][-1]
    if start is None or sent is None or not start <= sent <= first <= last:
        return None
    ready = start
    tool_ns = 0
    for turn in turns[:-1]:
        began, ended = turn.get("tool_start_ns"), turn.get("tool_ready_ns")
        if began is None or ended is None or not ready <= began <= ended <= sent:
            return None
        tool_ns += ended - began
        ready = ended
    return {
        "prior_turns_and_client_overhead_ms": (sent - start - tool_ns) / 1e6,
        "tool_execution_ms": tool_ns / 1e6,
        "final_submit_to_first_content_ms": (first - sent) / 1e6,
        "final_first_to_last_content_ms": (last - first) / 1e6,
    }


def final_cache_usage(records):
    """Aggregate only completed final turns whose server reports cache credit."""
    known = 0
    prompt_total = 0
    cached_total = 0
    for record in records:
        usage = record["turns"][-1].get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None:
            continue
        prompt = usage.get("prompt_tokens")
        if (
            type(cached) is not int
            or type(prompt) is not int
            or not 0 <= cached <= prompt
        ):
            raise ValueError("invalid reported final prefix cache usage")
        known += 1
        prompt_total += prompt
        cached_total += cached
    return {
        "reported_workflows": known,
        "unreported_workflows": len(records) - known,
        "prompt_tokens": prompt_total if known else None,
        "cached_tokens": cached_total if known else None,
        "uncached_tokens": prompt_total - cached_total if known else None,
        "token_weighted_hit_fraction": cached_total / prompt_total
        if prompt_total
        else None,
    }


def summarize(records, slo):
    names = [r["workflow_id"] for r in records]
    if not records or len(names) != len(set(names)):
        raise ValueError("expected distinct workflow IDs")
    by_name = {r["workflow_id"]: r for r in records}
    passed = []
    metrics = {
        "first_final_ms": "workflow_first_final",
        "last_final_ms": "workflow_last_final",
        "final_gap_p95_ms": "final_content_gap_p95",
    }
    for r in records:
        if r["ok"]:
            turns = r["turns"]
            for previous, following in itertools.pairwise(turns):
                start_field = (
                    "dispatch_ns"
                    if following.get("dispatch") == "application"
                    else "sent_ns"
                )
                if previous["tool_ready_ns"] > following[start_field]:
                    raise ValueError(
                        "request started before its prerequisite tool result"
                    )
            if (
                len(turns) != 3
                or not turns[-1].get("response_id")
                or not all(
                    (t.get("planned_request_sha256") and t.get("call", {}).get("id"))
                    if t.get("dispatch") == "application"
                    else t.get("response_id")
                    for t in turns[:-1]
                )
            ):
                raise ValueError("missing workflow turn")
            passed.append(r)
    overlap = None
    if {"reader", "long-tool-result"} <= set(by_name) and all(
        by_name[n]["ok"] for n in ("reader", "long-tool-result")
    ):
        reader = by_name["reader"]["turns"][-1]["content_ns"]
        long_turn = by_name["long-tool-result"]["turns"][-1]
        overlap = reader[0] <= long_turn["sent_ns"] < reader[-1]
    breakdowns = {r["workflow_id"]: final_latency_breakdown(r) for r in passed}
    observed = [parts for parts in breakdowns.values() if parts is not None]
    return {
        "submitted": len(records),
        "completed_latency_breakdowns": len(observed),
        "reported_final_prefix_cache": final_cache_usage(passed),
        "final_latency_components_ms": {
            name: {
                p: percentile([parts[name] for parts in observed], q)
                for p, q in (("p50", 0.50), ("p95", 0.95))
            }
            for name in (observed[0] if observed else ())
        },
        "complete": len(passed),
        "failed": len(records) - len(passed),
        "met_slo": sum(
            all(
                r[key] is not None and r[key] <= slo[limit]
                for key, limit in metrics.items()
            )
            for r in passed
        ),
        "long_request_sent_during_final_answer": overlap,
        "latency_ms": {
            key: {
                p: percentile([r[key] for r in passed if r[key] is not None], q)
                for p, q in (("p50", 0.50), ("p95", 0.95))
            }
            for key in (*metrics, "final_tpot_ms")
        },
        "cells": {
            r["workflow_id"]: {
                "ok": r["ok"],
                "error": r.get("error"),
                "final_latency_components_ms": breakdowns.get(r["workflow_id"]),
                "turns": [
                    {
                        "request_id": t.get("response_id"),
                        "tokens": t.get("usage"),
                        "finish": t.get("finish"),
                        "submit_to_first_content_ms": (
                            t["content_ns"][0] - t["sent_ns"]
                        )
                        / 1e6
                        if t.get("content_ns")
                        else None,
                        "previous_tool_to_call_ms": (
                            t["last_call_fragment_ns"]
                            - r["turns"][i - 1]["tool_ready_ns"]
                        )
                        / 1e6
                        if i and "last_call_fragment_ns" in t
                        else None,
                    }
                    for i, t in enumerate(r["turns"])
                ],
                **{k: r.get(k) for k in (*metrics, "final_tpot_ms")},
            }
            for r in records
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("registration", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    slo = json.loads(args.registration.read_text())["engineering_slo_ms"]
    if isinstance(raw, dict) and "waves" in raw:
        result = {"waves": [summarize(w["records"], slo) for w in raw["waves"]]}
    else:
        result = summarize(raw, slo)
    print(json.dumps(result, indent=2))
