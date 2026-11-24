"""Replay dependent tool calls and time the user's final answer over streaming HTTP."""

import argparse
import concurrent.futures
import hashlib
import http.client
import itertools
import json
import math
import os
import random
import re
import socket
import threading
import time
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * fraction) - 1]


def tool(name, description, properties):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def fixture(seed, pairs):
    rng = random.Random(seed)
    before = rng.randrange(45, 85)
    after = rng.randrange(12, 33)
    ids = ["report-A", "report-B"]
    values = [before, after]
    docs = [
        {
            "id": doc,
            "p95_ms": value,
            "statistic": "service response latency",
            "notes": [
                f"Observation {idx}: {rng.choice(['nominal', 'stable', 'expected'])} during routine sampling."
                for idx in range(pairs)
            ],
        }
        for doc, value in zip(ids, values)
    ]
    tools = [
        tool("list_reports", "Return the two report identifiers.", {}),
        tool(
            "read_reports",
            "Read the report pair returned by list_reports.",
            {
                "first": {"type": "string", "enum": ids[:1]},
                "second": {"type": "string", "enum": ids[1:]},
            },
        ),
    ]
    prompt = (
        f"Review case {seed}. List the reports, then read them. Compare the before (report-A) and "
        "after (report-B) latency. Start your final answer with 'Before: X ms. After: Y ms. "
        "Reduction: Z ms.' Fill in all three numbers. Then give a 90-word explanation of the "
        "comparison, citing both reports and explaining what p95 means. Do not invent measurements."
    )
    return tools, prompt, docs


def history_messages(seed, count):
    if not count:
        return []
    background = " ".join(
        f"Entry {i}: prior audit completed for region {(i + seed) % 13}."
        for i in range(count)
    )
    return [
        {
            "role": "user",
            "content": "Keep this historical audit log as context. " + background,
        },
        {
            "role": "assistant",
            "content": "The audit log is available for the next comparison.",
        },
    ]


def turn_request(model, messages, tools, name):
    choice = {"type": "function", "function": {"name": name}} if name else "none"
    return {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": choice,
        "temperature": 0,
        "max_tokens": 96 if name else 384,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def answer_format(content, docs):
    """Report literal task format separately from historic value and source checks."""
    before, after = (doc["p95_ms"] for doc in docs)
    opening = f"Before: {before} ms. After: {after} ms. Reduction: {before - after} ms."
    text = content.replace("**", "").lstrip()
    paragraphs = re.split(r"\n\s*\n", text, maxsplit=1)
    explanation = paragraphs[1] if len(paragraphs) == 2 else ""
    explanation = re.sub(r"^\s*Explanation:\s*", "", explanation, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9]+(?:[-’'][A-Za-z0-9]+)*", explanation)
    return {
        "exact_numeric_opening": text.startswith(opening),
        "explanation_words": len(words),
    }


def fixture_tool_result(index, docs):
    return [d["id"] for d in docs] if index == 0 else docs


def append_tool_result(messages, calls, docs, index, execute=None):
    result = (
        execute(index, calls[0], docs) if execute else fixture_tool_result(index, docs)
    )
    messages.extend(
        [
            {"role": "assistant", "content": None, "tool_calls": calls},
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": json.dumps(result),
            },
        ]
    )


class StreamCancellation:
    """Interrupt one optional HTTP request, including a preview waiting for first content."""

    def __init__(self):
        self.lock = threading.Lock()
        self.sock = None
        self.cancelled = False

    def attach(self, sock):
        with self.lock:
            if self.cancelled:
                raise ConnectionAbortedError("preview cancelled before request")
            self.sock = sock

    def detach(self):
        with self.lock:
            self.sock = None

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    # The server may already have closed the transport.
                    pass


def stream(
    address,
    request,
    row,
    first_visible=None,
    *,
    bearer=None,
    on_content=None,
    timeout_s=90,
    cancellation=None,
):
    url = urlsplit(address if "://" in address else "http://" + address)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "expected an HTTP(S) service URL or host:port, without credentials or query"
        )
    transport = (
        http.client.HTTPSConnection
        if url.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = transport(url.hostname, url.port, timeout=timeout_s)
    base = url.path.rstrip("/")
    api = base if base.endswith("/v1") else base + "/v1"
    calls = {}
    fragments = []
    event_times = row["content_ns"] = []
    row.update(sent_ns=time.monotonic_ns(), input_sha256=fingerprint(request))
    try:
        if cancellation is not None:
            connection.connect()
            cancellation.attach(connection.sock)
        headers = {"Content-Type": "application/json"}
        if bearer:
            headers["Authorization"] = "Bearer " + bearer
        try:
            connection.request(
                "POST",
                api + "/chat/completions",
                json.dumps(request),
                headers,
            )
        except BrokenPipeError as write_error:
            # A full HTTP listener can send 429 and close before the POST body
            # is written. Read any early rejection on this same connection.
            row["request_write_error"] = "broken_pipe"
            if connection.sock is None:
                raise
            try:
                connection.sock.settimeout(min(timeout_s, 0.25))
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as read_error:
                raise write_error from read_error
            if response.status < 400:
                raise
        else:
            row["request_written_ns"] = time.monotonic_ns()
            response = connection.getresponse()
        row["status"] = response.status
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {response.read(1024)!r}")
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            now = time.monotonic_ns()
            if raw == b"[DONE]":
                row["done_ns"] = now
                break
            msg = json.loads(raw)
            if "error" in msg:
                row["stream_error"] = msg["error"]
                raise ValueError(f"server streaming error: {msg['error']}")
            row["response_id"] = msg["id"]
            if msg.get("prompt_token_ids") is not None:
                row["prompt_ids_sha256"] = fingerprint(msg["prompt_token_ids"])
            if msg.get("usage"):
                row["usage"] = msg["usage"]
            for choice in msg["choices"]:
                if choice.get("prompt_token_ids") is not None:
                    row["prompt_ids_sha256"] = fingerprint(choice["prompt_token_ids"])
                if choice.get("token_ids"):
                    row.setdefault("generated_ids", []).extend(choice["token_ids"])
                if choice.get("finish_reason"):
                    row["finish"] = choice["finish_reason"]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    fragments.append(delta["content"])
                    event_times.append(now)
                    if on_content is not None:
                        on_content(len(event_times))
                    if len(event_times) == 1 and first_visible:
                        first_visible.set()
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(
                        part["index"],
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    call["id"] += part.get("id") or ""
                    for field in ("name", "arguments"):
                        call["function"][field] += (
                            part.get("function", {}).get(field) or ""
                        )
                    row["last_call_fragment_ns"] = now
    finally:
        if cancellation is not None:
            cancellation.detach()
        connection.close()
    if "done_ns" not in row or "usage" not in row or not row.get("finish"):
        raise ValueError("incomplete stream, finish reason or usage")
    return "".join(fragments), list(calls.values())


def replay(
    address,
    model,
    seed,
    pairs,
    delay,
    workflow_id,
    first_visible=None,
    before_final=None,
    history=0,
    send=stream,
    on_tool_start=None,
    execute_tool=None,
    dispatch_fixed_tools=False,
):
    tools, prompt, docs = fixture(seed, pairs)
    messages = history_messages(seed, history) + [{"role": "user", "content": prompt}]
    record = {
        "workflow_id": workflow_id,
        "start_ns": time.monotonic_ns(),
        "seed": seed,
        "document_pairs": pairs,
        "history_entries": history,
        "tool_delay_s": delay,
        "fixture_sha256": fingerprint([prompt, docs]),
        "turns": [],
        "ok": False,
    }
    record["epoch_offset_ns"] = time.time_ns() - time.monotonic_ns()
    try:
        for index, name in enumerate(("list_reports", "read_reports", None)):
            if name is None and before_final and not before_final.wait(60):
                raise ValueError("peer did not begin streaming")
            row = {"turn": index}
            record["turns"].append(row)
            request = turn_request(model, messages, tools, name)
            resolved = None
            if dispatch_fixed_tools and name:
                from agent_tool_plan import resolve_named_step

                resolved = resolve_named_step(tools, request["tool_choice"])
            if resolved is None:
                content, calls = send(
                    address, request, row, first_visible if not name else None
                )
            else:
                from uuid import uuid4

                function_name, arguments = resolved
                if index == 1:
                    listed = json.loads(messages[-1]["content"])
                    if not isinstance(listed, list) or not all(
                        value in listed for value in arguments.values()
                    ):
                        raise ValueError(
                            "named reports are missing from the preceding tool result"
                        )
                row["dispatch"] = "application"
                row["dispatch_ns"] = time.monotonic_ns()
                row["planned_request_sha256"] = fingerprint(request)
                calls = [
                    {
                        "type": "function",
                        "id": f"call-{uuid4().hex}",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments, separators=(",", ":")),
                        },
                    }
                ]
            if name:
                # With an explicitly named tool some servers return stop.
                # In both cases require one complete schema-valid invocation.
                if resolved is None:
                    row["observed_calls"] = calls
                if (
                    resolved is None and row["finish"] not in ("tool_calls", "stop")
                ) or len(calls) != 1:
                    raise ValueError(f"expected one completed {name} invocation")
                call = calls[0]
                expected = (
                    {}
                    if index == 0
                    else {"first": docs[0]["id"], "second": docs[1]["id"]}
                )
                if (
                    not call["id"]
                    or call["function"]["name"] != name
                    or json.loads(call["function"]["arguments"]) != expected
                ):
                    raise ValueError(f"invalid tool payload: {call}")
                row["call"] = call
                row["tool_start_ns"] = time.monotonic_ns()
                if on_tool_start is not None:
                    on_tool_start(index, request, row)
                if delay:
                    time.sleep(delay)
                append_tool_result(messages, calls, docs, index, execute_tool)
                row["tool_ready_ns"] = time.monotonic_ns()
            else:
                row["answer"] = content
                row["answer_format"] = answer_format(content, docs)
                expected_values = [d["p95_ms"] for d in docs]
                expected_values.append(expected_values[0] - expected_values[1])
                plain = content.replace("*", "")
                labels = ("before", "after", "reduction")
                correct = all(
                    re.search(rf"\b{label}\s*:\s*{value}\s*ms\b", plain, re.IGNORECASE)
                    for label, value in zip(labels, expected_values)
                )
                if (
                    calls
                    or row["finish"] != "stop"
                    or row["usage"]["completion_tokens"] < 64
                    or not correct
                    or not all(d["id"].lower() in plain.lower() for d in docs)
                ):
                    raise ValueError(
                        f"final answer failed values, citations, length or finish: {expected_values}"
                    )
                events = row["content_ns"]
                record["first_final_ms"] = (events[0] - record["start_ns"]) / 1e6
                record["last_final_ms"] = (events[-1] - record["start_ns"]) / 1e6
                record["final_tpot_ms"] = (
                    (events[-1] - events[0])
                    / 1e6
                    / (row["usage"]["completion_tokens"] - 1)
                )
                record["final_gap_p95_ms"] = percentile(
                    [(b - a) / 1e6 for a, b in itertools.pairwise(events)], 0.95
                )
                record["ok"] = True
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        http.client.HTTPException,
    ) as error:
        record["error"] = str(error)
    record["end_ns"] = time.monotonic_ns()
    record["end_epoch_offset_ns"] = time.time_ns() - time.monotonic_ns()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("address")
    p.add_argument("model")
    p.add_argument(
        "--bearer-env", help="environment variable containing the bearer token"
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--history-entries", type=int, default=0)
    p.add_argument("--long-pairs", type=int, default=96)
    p.add_argument(
        "--cohort",
        action="store_true",
        help="release three long final inputs while a peer streams",
    )
    p.add_argument(
        "--dispatch-fixed-tools",
        action="store_true",
        help="let the application execute named tool steps with a single possible argument object",
    )
    args = p.parse_args()
    bearer = os.environ.get(args.bearer_env) if args.bearer_env else None
    if args.bearer_env and not bearer:
        p.error("requested bearer token environment variable is unset or empty")
    run_workflow = partial(
        replay,
        send=partial(stream, bearer=bearer),
        dispatch_fixed_tools=args.dispatch_fixed_tools,
    )
    if args.cohort:
        # Hold the reader until the other sessions have real tool results. Release
        # their final requests on its first content event, not a guessed delay.
        ready = threading.Barrier(4, timeout=90)
        visible = threading.Event()

        class FinalGate:
            def __init__(self, reader):
                self.reader = reader

            def wait(self, timeout):
                try:
                    ready.wait(timeout)
                except threading.BrokenBarrierError:
                    return False
                return self.reader or visible.wait(timeout)

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            jobs = [
                pool.submit(
                    run_workflow,
                    args.address,
                    args.model,
                    args.seed + i,
                    (2, args.long_pairs, args.long_pairs // 2, args.long_pairs)[i],
                    0,
                    f"cohort-{i}",
                    visible if i == 0 else None,
                    FinalGate(i == 0),
                    args.history_entries,
                )
                for i in range(4)
            ]
            records = [job.result() for job in jobs]
        args.out.write_text(json.dumps(records, indent=2) + "\n")
        return int(not all(r["ok"] for r in records))
    records = []
    for name, delay in (("cold", 0), ("immediate", 0), ("delayed", 0.1)):
        records.append(
            run_workflow(
                args.address,
                args.model,
                args.seed,
                2,
                delay,
                name,
                history=args.history_entries,
            )
        )
        if args.smoke or not records[-1]["ok"]:
            break
    if not args.smoke and all(r["ok"] for r in records):
        visible = threading.Event()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            reader = pool.submit(
                run_workflow,
                args.address,
                args.model,
                args.seed + 1,
                2,
                0,
                "reader",
                visible,
                None,
                args.history_entries,
            )
            writer = pool.submit(
                run_workflow,
                args.address,
                args.model,
                args.seed + 2,
                args.long_pairs,
                0,
                "long-tool-result",
                None,
                visible,
                args.history_entries,
            )
            records.extend([reader.result(), writer.result()])
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(
        [
            (r["workflow_id"], r["ok"], r.get("error"), r.get("last_final_ms"))
            for r in records
        ]
    )
    return int(not all(r["ok"] for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
