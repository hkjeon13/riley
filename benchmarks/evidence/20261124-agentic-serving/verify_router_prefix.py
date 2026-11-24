"""Replay recorded routing inputs and time incremental history-key construction."""

import json
import platform
import tarfile
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter_ns
from unittest.mock import patch

import replica_workflow
from agent_workflow import (
    append_tool_result,
    fingerprint,
    fixture,
    history_messages,
    percentile,
    turn_request,
)
from replica_workflow import Router, history_credit_keys

HERE = Path(__file__).resolve().parent


def original(layout, messages):
    return [fingerprint([layout, messages[:n]]) for n in range(1, len(messages) + 1)]


def layout(req):
    return req["model"], req["tool_choice"] == "none", req["tools"]


def recorded():
    with tarfile.open(HERE / "replicas-raw.tar.gz") as archive:
        for arm in "abcdef":
            waves = json.load(archive.extractfile(f"replicas/{arm}/capacity.json"))
            for wave in waves:
                for session, task in enumerate(wave["records"]):
                    seed = task["seed"]
                    tools, ask, docs = fixture(seed, task["document_pairs"])
                    messages = history_messages(seed, task["history_entries"])
                    messages.append({"role": "user", "content": ask})
                    for turn, name in zip(
                        task["turns"],
                        ("list_reports", "read_reports", None),
                        strict=True,
                    ):
                        req = turn_request("m", deepcopy(messages), tools, name)
                        if fingerprint(req) != turn["input_sha256"]:
                            raise ValueError("archived request is not reproducible")
                        yield req, turn, session
                        if name:
                            append_tool_result(
                                messages, turn["observed_calls"], docs, turn["turn"]
                            )


def compare_routes(events, policy):
    left = Router([("a", None), ("b", None)], policy, [8, 8])
    right = Router(left.endpoints, policy, left.active_limits)

    def transport(address, req, row, _, bearer):
        row["usage"] = turn["usage"]
        return address

    with patch.object(replica_workflow, "stream", transport):
        for req, turn, session in events:
            observed = []
            for router, key_fn in ((left, original), (right, history_credit_keys)):
                router.active = turn["route"]["active"].copy()
                result = {}
                with patch.object(replica_workflow, "history_credit_keys", key_fn):
                    addr = router.sender(session)(None, req, result, None)
                route = result["route"]
                route.pop("decision_ms")
                observed.append((addr, route))
            if observed[0] != observed[1] or left.credits != right.credits:
                raise ValueError("route, credited prefix, or score differed")


def timing(req):
    sample = {}
    for n in (4, 100, 400):
        # Each synthetic history entry is a separate chat message. The archived
        # 100-line history uses one user message and is measured separately.
        messages = [
            {
                "role": "assistant" if i % 2 else "user",
                "content": f'요청 {i}: "검토"\n' + "read the record. " * 12,
            }
            for i in range(n)
        ]
        candidate = {**req, "messages": messages}
        sample[str(n)] = candidate
    sample["recorded-final"] = req
    metrics = {}
    for label, request in sample.items():
        times = {
            k: {"keys_ms": [], "decision_ms": []} for k in ("original", "incremental")
        }
        if original(layout(request), request["messages"]) != history_credit_keys(
            layout(request), request["messages"]
        ):
            raise ValueError("long input prefix keys changed")

        def transport(_, __, row, ___, bearer):
            row["usage"] = {"prompt_tokens": 1000}

        with patch.object(replica_workflow, "stream", transport):
            for rep in range(20):
                order = [("original", original), ("incremental", history_credit_keys)]
                for name, fn in order if rep % 2 == 0 else reversed(order):
                    start = perf_counter_ns()
                    fn(layout(request), request["messages"])
                    t = (perf_counter_ns() - start) / 1e6
                    router = Router([("a", None), ("b", None)], "cost")
                    row = {}
                    with patch.object(replica_workflow, "history_credit_keys", fn):
                        router.sender(0)(None, request, row, None)
                    times[name]["keys_ms"].append(t)
                    times[name]["decision_ms"].append(row["route"]["decision_ms"])
        metrics[label] = {
            "messages": len(request["messages"]),
            "request_bytes": len(json.dumps(request).encode()),
            "timings": {
                name: {
                    key: {
                        "samples_ms": values,
                        "p50_ms": median(values),
                        "p95_ms": percentile(values, 0.95),
                    }
                    for key, values in arms.items()
                }
                for name, arms in times.items()
            },
        }
    return metrics


def run():
    events = list(recorded())
    boundaries = 0
    for req, _, _ in events:
        messages = req["messages"]
        if original(layout(req), messages) != history_credit_keys(
            layout(req), messages
        ):
            raise ValueError("key mismatch")
        boundaries += len(messages)
    layout_a = ("모형", False, [{"properties": {"z": {}, "a": {"value": '"\n'}}}])
    edges = [
        [],
        [{"role": "assistant", "content": None}],
        [{"content": [None, {"b": True, "a": "Ω🚀"}], "role": "tool"}, {}],
    ]
    for messages in edges:
        for current in (layout_a, (layout_a[0], True, layout_a[2]), ("m", False, [])):
            if original(current, messages) != history_credit_keys(current, messages):
                raise ValueError("escaped or nested message mismatch")
    for policy in ("cost", "capacity", "session", "round-robin"):
        compare_routes(events, policy)
    times = timing(events[-1][0])
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wire_request_instances": len(events),
        "prefix_boundaries": boundaries,
        "real_message_counts": dict(Counter(len(r["messages"]) for r, _, _ in events)),
        "matched_route_decisions": len(events) * 4,
        "edge_comparisons": len(edges) * 3,
        "driver_sha256": sha256(
            Path(replica_workflow.__file__).read_bytes()
        ).hexdigest(),
        "local_cpu_only": times,
        "registered_speed_gate": all(
            times[str(n)]["timings"]["incremental"]["keys_ms"]["p50_ms"]
            < times[str(n)]["timings"]["original"]["keys_ms"]["p50_ms"]
            for n in (100, 400)
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
