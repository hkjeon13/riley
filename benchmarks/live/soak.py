#!/usr/bin/env python3
"""Validate a synthetic, mixed, open-loop workload against a Riley server."""
import argparse
import asyncio
from functools import partial
from io import StringIO
import json
import math
import random
import re
import subprocess
import time
import urllib.request

from load import document, percentile

SCHEMA = {"type": "object", "properties": {"title": {"type": "string", "maxLength": 40},
                                           "year": {"type": "integer", "minimum": 1000, "maximum": 2100},
                                           "tags": {"type": "array", "items": {"enum": ["a", "b", "c"]}, "maxItems": 3}},
          "required": ["title", "year"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {
    "type": "object", "properties": {"city": {"type": "string", "maxLength": 30}}, "required": ["city"]}}}]
KINDS = ("plain", "schema", "logprobs", "tools", "long", "reason", "cancel")
WEIGHTS = (55, 12, 8, 5, 10, 5, 5)
IDLE_GAUGES = ("active_requests", "waiting_requests", "kv_allocated_blocks")
WORDS = "the river market station garden winter engine letter mountain copper signal harbor forest".split()


def text(rng, words):
    return " ".join(rng.choice(WORDS) for _ in range(words))


def request(model, rng, shared_prefixes):
    kind = rng.choices(KINDS, WEIGHTS)[0]
    body = {"model": model, "temperature": 0, "max_tokens": 64, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Write about {text(rng, rng.randint(3, 40))}."}]}
    if kind == "schema":
        body.update(max_tokens=96, response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": SCHEMA}})
    elif kind == "logprobs":
        body.update(logprobs=True, top_logprobs=3)
    elif kind == "tools":
        body.update(tools=TOOLS, tool_choice="required", max_tokens=96)
    elif kind == "long":
        paragraph = text(rng, rng.choice([800, 1500, 3000]))
        if shared_prefixes:
            prefix = shared_prefixes[rng.randrange(len(shared_prefixes))]
            paragraph = f"{prefix}\n\n{paragraph}"
        body["messages"][0]["content"] = f"{paragraph}\nSummarize the text above."
        body["max_tokens"] = 96
    elif kind == "reason":
        body["max_tokens"] = 512
    return kind, body


def validate_completion(kind, finish, content, calls, logprobs, usage, limit):
    if finish not in ("stop", "length", "tool_calls"):
        raise ValueError("missing or invalid SSE finish")
    if (any(type(usage.get(k)) is not int or usage[k] < 0
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
            or not 0 < usage["completion_tokens"] <= limit
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
            or (finish == "length" and usage["completion_tokens"] != limit)):
        raise ValueError("invalid SSE usage")
    if kind == "schema" and finish == "stop":
        value = json.loads(content)
        if (not isinstance(value, dict) or not {"title", "year"} <= set(value) <= {"title", "year", "tags"}
                or not isinstance(value["title"], str) or len(value["title"]) > 40
                or type(value["year"]) is not int or not 1000 <= value["year"] <= 2100
                or ("tags" in value and (not isinstance(value["tags"], list)
                    or len(value["tags"]) > 3 or any(tag not in ("a", "b", "c") for tag in value["tags"])))):
            raise ValueError("invalid completed schema")
    if kind == "tools":
        if finish == "stop":
            raise ValueError("required tool call is missing")
        if finish == "tool_calls":
            if set(calls) != {0} or calls[0]["name"] != "get_weather":
                raise ValueError("unexpected required tool call")
            args = json.loads(calls[0]["arguments"])
            if (not isinstance(args, dict) or not isinstance(args.get("city"), str) or len(args["city"]) > 30):
                raise ValueError("invalid required tool arguments")
    elif finish == "tool_calls":
        raise ValueError("unexpected tool terminal")
    if kind == "logprobs" and not logprobs:
        raise ValueError("missing streamed logprobs")


async def read_completion(reader, kind, limit, started):
    status = int((await reader.readline()).split()[1])
    while (await reader.readline()) != b"\r\n":
        if reader.at_eof():
            raise ValueError("incomplete HTTP headers")
    if kind == "tools_unsupported":
        if status == 400:
            error = json.loads(await reader.read())["error"]
            if (error.get("type"), error.get("code"), error.get("param"), error.get("message")) == (
                "invalid_request_error", "unsupported_parameter", "tools", "this model's chat template does not support tools",
            ):
                return {"outcome": "unsupported"}
        return {"outcome": "error", "http_status": status, "detail": "expected the template tool rejection"}
    if status != 200:
        return {"outcome": "rejected" if status == 429 else "error", "http_status": status}

    content, calls, usage = [], {}, {}
    first_visible = finish = None
    visible_chunks = logprobs = 0
    while True:
        line = await reader.readline()
        if not line:
            raise ValueError("SSE closed without [DONE]")
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            break
        event = json.loads(payload)
        if "error" in event:
            raise ValueError("server sent an SSE error")
        if event.get("usage") is not None:
            if usage:
                raise ValueError("duplicate SSE usage")
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice["delta"]
            fragment = delta.get("content") or ""
            pieces = delta.get("tool_calls") or []
            if fragment or pieces:
                if finish is not None:
                    raise ValueError("content after SSE finish")
                if first_visible is None:
                    first_visible = time.perf_counter()
                visible_chunks += 1
            content.append(fragment)
            for part in pieces:
                fields = calls.setdefault(part["index"], {"name": "", "arguments": ""})
                for key in fields:
                    fields[key] += part["function"].get(key) or ""
            for row in (choice.get("logprobs") or {}).get("content") or []:
                if not math.isfinite(row["logprob"]):
                    raise ValueError("non-finite streamed logprob")
                logprobs += 1
            terminal = choice.get("finish_reason")
            if terminal is not None:
                if finish is not None:
                    raise ValueError("duplicate SSE finish")
                finish = terminal
        if kind == "cancel" and visible_chunks >= 4 and finish is None:
            return {"outcome": "disconnected"}

    finished = time.perf_counter()
    validate_completion(kind, finish, "".join(content), calls, logprobs, usage, limit)
    return {"outcome": "completed", "finish": finish,
            "ttft_ms": None if first_visible is None else (first_visible - started) * 1000,
            "e2e_ms": (finished - started) * 1000}


async def exchange(host, port, kind, body, started):
    reader, writer = await asyncio.open_connection(host, port, limit=1 << 22)
    try:
        payload = json.dumps(body).encode()
        head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
        writer.write(head + payload)
        await writer.drain()
        return await read_completion(reader, kind, body["max_tokens"], started)
    finally:
        writer.close()


async def run_request(host, port, kind, body, scheduled):
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(exchange(host, port, kind, body, started), timeout=120)
    except (asyncio.TimeoutError, OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result = {"outcome": "error", "detail": str(error)[:100] or type(error).__name__}
    return {"kind": kind, "scheduled": scheduled, "dispatch_ms": (started - scheduled) * 1000, **result}


def open_host_proc(pid, filename):
    return open(f"/proc/{pid}/{filename}", encoding="utf-8")


def open_container_proc(container, pid, filename):
    """Read PID 1 with the server's uid while binding it to the host PID."""
    inspect = ["docker", "container", "inspect", "--format", "{{.Id}} {{.State.Pid}} {{.RestartCount}}", container]
    command = ["docker", "exec", container, "cat", f"/proc/1/{filename}"]
    try:
        before = subprocess.check_output(inspect, text=True, timeout=10).strip()
        expected = f"{container} {pid} 0"
        if before != expected:
            raise ValueError("container process identity differs from the soak target")
        content = subprocess.check_output(command, text=True, encoding="utf-8", timeout=10)
        after = subprocess.check_output(inspect, text=True, timeout=10).strip()
        if after != expected:
            raise ValueError("container process identity changed during the proc read")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise OSError(f"cannot read container PID 1 {filename}: {type(error).__name__}") from error
    return StringIO(content)


def read_smaps_rollup(pid, *, open_proc=open_host_proc):
    fields = {"Rss": "rss_kib", "Anonymous": "anonymous_kib"}
    values = {}
    started = time.perf_counter()
    with open_proc(pid, "smaps_rollup") as info:
        for line in info:
            key, _, body = line.partition(":")
            field = fields.get(key)
            if field is None:
                continue
            count, unit = body.split()
            if unit != "kB" or field in values:
                raise ValueError(f"invalid smaps_rollup field {key}")
            values[field] = int(count)
    rss_kib = values.get("rss_kib", 0)
    if rss_kib <= 0 or not 0 <= values.get("anonymous_kib", -1) <= rss_kib:
        raise ValueError("missing or inconsistent smaps_rollup RSS")
    values["read_ms"] = (time.perf_counter() - started) * 1000
    return values


def record_smaps_rollup(sample, pid, *, open_proc=open_host_proc):
    try:
        sample["smaps_rollup"] = read_smaps_rollup(pid, open_proc=open_proc)
    except (OSError, ValueError) as error:
        sample["smaps_rollup"] = None
        sample["smaps_rollup_error"] = str(error)


def snapshot(pid, address, start, smaps_after_seconds=None, *, open_proc=open_host_proc):
    with open_proc(pid, "status") as info:
        rss = next((int(line.split()[1]) / 1024 for line in info if line.startswith("VmRSS:")), None)
    if rss is None:
        raise ValueError("missing VmRSS in process status")
    with urllib.request.urlopen(f"http://{address}/metrics", timeout=5) as response:
        metrics = json.load(response)
    sample = {"at_seconds": time.perf_counter() - start, "rss_mib": rss, "metrics": metrics}
    if smaps_after_seconds is not None and sample["at_seconds"] >= smaps_after_seconds:
        record_smaps_rollup(sample, pid, open_proc=open_proc)
    return sample


def idle(metrics):
    return all(type(metrics.get(key)) is int and metrics[key] == 0 for key in IDLE_GAUGES)


def latency_windows(rows, start, duration):
    windows = {}
    for label, minimum, maximum in (("early", .20, .50), ("late", .70, 1.0)):
        eligible = [row for row in rows if row["outcome"] == "completed"
                    and minimum * duration <= row["scheduled"] - start < maximum * duration]
        windows[label] = {"completed": len(eligible)}
        for metric in ("ttft_ms", "e2e_ms"):
            values = [row[metric] for row in eligible if row[metric] is not None]
            windows[label][metric + "_count"] = len(values)
            windows[label][metric + "_p95"] = percentile(values, .95)
    changes = {}
    for metric in ("ttft_ms", "e2e_ms"):
        early, late = (windows[label][metric + "_p95"] for label in ("early", "late"))
        changes[metric] = (late / early - 1) * 100 if early and late else None
    return windows, changes


def first_second_half_sample(samples, duration):
    return next((sample for sample in samples if sample["at_seconds"] >= duration / 2), {})


def rss_growth_second_half(samples, final, duration):
    baseline = first_second_half_sample(samples, duration).get("rss_mib")
    return final["rss_mib"] - baseline if baseline is not None else None


def smaps_growth_second_half(samples, final, duration):
    baseline = first_second_half_sample(samples, duration)
    start, end = baseline.get("smaps_rollup"), final.get("smaps_rollup")
    comparison = {"baseline": start, "drain": end, "rss_growth_kib": None, "anonymous_growth_kib": None}
    if start is not None and end is not None:
        comparison["rss_growth_kib"] = end["rss_kib"] - start["rss_kib"]
        comparison["anonymous_growth_kib"] = end["anonymous_kib"] - start["anonymous_kib"]
    return comparison


async def soak(args, *, open_proc=open_host_proc):
    host, port = args.address.split(":")
    rng = random.Random(11)
    prefix_rng = random.Random(12)
    shared_prefixes = [document(prefix_rng, 512) for _ in range(8)] if args.shared_prefix else []
    start = time.perf_counter()
    initial = await asyncio.to_thread(snapshot, args.pid, args.address, start, open_proc=open_proc)
    if not idle(initial["metrics"]):
        raise ValueError("soak requires a dedicated idle server")
    start = time.perf_counter()
    scheduled = rng.expovariate(args.rate)
    samples = []
    rows, tasks, next_sample = [], [], 0.0
    poll = None
    smaps_after_seconds = args.seconds / 2 if args.require_zero_rss_growth else None
    expect_tools_unsupported = getattr(args, "expect_tools_unsupported", False)
    while scheduled < args.seconds:
        await asyncio.sleep(max(0, start + scheduled - time.perf_counter()))
        kind, body = request(args.model, rng, shared_prefixes)
        if kind == "tools" and expect_tools_unsupported:
            kind = "tools_unsupported"
        tasks.append(asyncio.create_task(run_request(host, int(port), kind, body, start + scheduled)))
        if scheduled >= next_sample:
            if poll is not None:
                sample = await poll
                samples.append(sample)
                if "smaps_rollup" in sample:
                    smaps_after_seconds = None
            poll = asyncio.create_task(asyncio.to_thread(
                snapshot, args.pid, args.address, start, smaps_after_seconds=smaps_after_seconds,
                open_proc=open_proc,
            ))
            running = []
            for task in tasks:
                if task.done():
                    rows.append(task.result())
                else:
                    running.append(task)
            tasks = running
            print(f"{scheduled:.0f}s completed_records={len(rows)} outstanding={len(tasks)}", flush=True)
            next_sample += 60
        scheduled += rng.expovariate(args.rate)
    await asyncio.sleep(max(0, start + args.seconds - time.perf_counter()))
    rows.extend(await asyncio.gather(*tasks))
    if poll is not None:
        samples.append(await poll)
    deadline = time.monotonic() + 10
    while True:
        final = await asyncio.to_thread(snapshot, args.pid, args.address, start, open_proc=open_proc)
        if idle(final["metrics"]) or time.monotonic() >= deadline:
            break
        await asyncio.sleep(.5)

    smaps_comparison = None
    if args.require_zero_rss_growth:
        record_smaps_rollup(final, args.pid, open_proc=open_proc)
        smaps_comparison = smaps_growth_second_half(samples, final, args.seconds)

    kinds = tuple("tools_unsupported" if kind == "tools" and expect_tools_unsupported else kind for kind in KINDS)
    counts = {kind: {} for kind in kinds}
    problems = []
    for row in rows:
        outcomes = counts[row["kind"]]
        outcome = row["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome in ("error", "rejected"):
            problems.append({"kind": row["kind"], "http_status": row.get("http_status"), "detail": row.get("detail")})
        elif outcome == "completed" and row["kind"] in ("schema", "tools"):
            terminal = row["finish"]
            outcomes[terminal] = outcomes.get(terminal, 0) + 1

    before, after = (item["metrics"]["counters"] for item in (initial, final))
    counter_changes = {key: after[key] - before[key] for key in before}
    windows, drift = latency_windows(rows, start, args.seconds)
    tool_covered = (counts["tools_unsupported"].get("unsupported", 0) > 0 if expect_tools_unsupported
                    else counts["tools"].get("tool_calls", 0) > 0)
    checks = {
        "no_unexpected_http_or_stream_error": not problems and counter_changes["overloads"] == 0,
        "completed_schema_and_tool_covered": counts["schema"].get("stop", 0) > 0 and tool_covered,
        "disconnects_observed": counts["cancel"].get("disconnected", 0) > 0
        and counter_changes["disconnects"] > 0 and counter_changes["cancellations"] > 0,
        "idle_after_drain": idle(final["metrics"]),
        "native_allocation_plateau": initial["metrics"]["allocation"] == final["metrics"]["allocation"],
        "workload_coverage": all(counts[kind].get("unsupported" if kind == "tools_unsupported" else "completed", 0) > 0
                                 for kind in kinds if kind != "cancel"),
    }
    if not args.smoke:
        checks["p95_drift_at_most_5_percent"] = all(value is not None and value <= 5 for value in drift.values())
    rss_growth = rss_growth_second_half(samples, final, args.seconds)
    if smaps_comparison is not None:
        precise_growth = smaps_comparison["rss_growth_kib"]
        checks["rss_growth_second_half_at_most_zero"] = precise_growth is not None and precise_growth <= 0
    result = {"status": "passed" if all(checks.values()) else "failed",
              "mode": "smoke" if args.smoke else "mixed_soak", "seconds": args.seconds, "rate": args.rate,
              "shared_prefixes": len(shared_prefixes), "checks": checks, "counts": counts,
              "problem_count": len(problems), "problems": problems[:20],
              "latency_windows": windows, "p95_change_percent": drift,
              "client_dispatch_p95_ms": percentile([row["dispatch_ms"] for row in rows], .95),
              "rss_growth_second_half_mib": rss_growth,
              "rss_samples": samples, "initial": initial, "final": final, "counter_changes": counter_changes}
    if smaps_comparison is not None:
        result["smaps_rollup_second_half"] = smaps_comparison
    if expect_tools_unsupported:
        result["expect_tools_unsupported"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("address")
    parser.add_argument("model")
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--rate", type=float, default=8)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--docker-container", help="read proc entries inside this full container ID using its user")
    parser.add_argument("--out")
    parser.add_argument("--shared-prefix", action="store_true", help="reuse eight fixed prefixes in the long-prompt traffic")
    parser.add_argument("--smoke", action="store_true", help="short runtime check without a latency-drift verdict")
    parser.add_argument("--expect-tools-unsupported", action="store_true",
                        help="validate the explicit template tool rejection instead of generating tool calls in smoke")
    parser.add_argument("--require-zero-rss-growth", action="store_true",
                        help="fail if Linux smaps_rollup RSS grows from the first second-half sample to drain, or either reading is missing")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("seconds and rate must be finite and positive")
    if args.expect_tools_unsupported and not args.smoke:
        parser.error("--expect-tools-unsupported requires --smoke")
    open_proc = open_host_proc
    if args.docker_container is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", args.docker_container) or args.pid <= 0:
            parser.error("--docker-container requires a full container ID and its positive host PID")
        open_proc = partial(open_container_proc, args.docker_container)
    result = asyncio.run(soak(args, open_proc=open_proc))
    if args.docker_container is not None:
        result["rss_source"] = {"container_id": args.docker_container, "host_pid": args.pid, "container_pid": 1}
    print(json.dumps({key: result[key] for key in ("status", "checks", "counts", "p95_change_percent")}), flush=True)
    if args.out:
        with open(args.out, "x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
