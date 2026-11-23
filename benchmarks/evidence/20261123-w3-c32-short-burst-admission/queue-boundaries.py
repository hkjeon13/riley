"""Decompose recorded queue waits by when the overlapping engine step began."""
from bisect import bisect_right
from private_reference import STARTUP


def read_boundaries(log, sample, observed, warmup_requests):
    confirmed, _ = engine_trace(log, sample, observed, warmup_requests)
    sends, workers, spans = {}, {}, []
    for line in log.splitlines():
        if line.startswith("W3-REQUEST-PHASE "):
            _, request, phase, ns = line.split()
            if phase == "send":
                sends[int(request)] = int(ns)
            elif phase == "worker":
                workers[int(request)] = int(ns)
        elif line.startswith("W3-ENGINE-SPAN "):
            spans.append(tuple(map(int, line.split()[1:])))
    starts = [start for start, *_ in spans]
    records = []
    for reply, verified in zip(sample["short"]["rows"], confirmed["short"], strict=True):
        request = int(reply["response_id"].removeprefix("cmpl-"), 16)
        sent, worker = sends[request], workers[request]
        preceding = bisect_right(starts, sent) - 1
        incumbent = spans[preceding] if preceding >= 0 and sent < spans[preceding][1] else None
        incumbent_ns, elapsed, duration, shape = 0, None, None, None
        if incumbent:
            start, end, rows, decode = incumbent
            require(end <= worker, "worker began admission before the incumbent execution ended")
            incumbent_ns, elapsed, duration, shape = end - sent, sent - start, end - start, (rows, decode)
        later_ns = later_steps = 0
        for start, end, _, _ in spans[preceding + 1:]:
            if start >= worker:
                break
            require(end <= worker, "worker began admission while a later execution was still active")
            later_ns += end - start
            later_steps += 1
        outside_ns = worker - sent - incumbent_ns - later_ns
        require(outside_ns >= 0 and abs(outside_ns / 1_000_000 - verified["queue_outside_execution_ms"]) < 0.000001,
                "queue intervals do not match the initial overlap verification")
        records.append({
            "queue_wait_ms": (worker - sent) / 1_000_000,
            "incumbent_remaining_ms": incumbent_ns / 1_000_000,
            "later_execution_ms": later_ns / 1_000_000,
            "outside_execution_ms": outside_ns / 1_000_000,
            "incumbent_duration_ms": duration / 1_000_000 if incumbent else None,
            "incumbent_elapsed_ms": elapsed / 1_000_000 if incumbent else None,
            "incumbent_shape": shape,
            "later_steps": later_steps,
        })
    return records


def distribution(values):
    return {"p50": p(values, .5), "p95": p(values, .95), "mean": sum(values) / len(values)}


QUEUE_FIELDS = ("queue_wait_ms", "incumbent_remaining_ms", "later_execution_ms", "outside_execution_ms")


def summarize_run(records):
    return {
        "requests": len(records),
        "sent_during_execution": sum(r["incumbent_shape"] is not None for r in records),
        "with_later_execution": sum(r["later_steps"] > 0 for r in records),
        **{field: distribution([r[field] for r in records]) for field in QUEUE_FIELDS},
    }


def summarize_pairs(base, candidate):
    pairs = list(zip(base, candidate, strict=True))
    arms = {label: summarize_run(rows) for label, rows in (("baseline", base), ("candidate", candidate))}
    both = [(left, right) for left, right in pairs
            if left["incumbent_shape"] is not None and right["incumbent_shape"] is not None]
    same_shape = [(left, right) for left, right in both if left["incumbent_shape"] == right["incumbent_shape"]]
    paired = {field: distribution([right[field] - left[field] for left, right in pairs]) for field in QUEUE_FIELDS}
    decomposition = {}
    for group, matching in (("both_during_execution", both), ("same_incumbent_shape", same_shape)):
        decomposition[group] = {"requests": len(matching)}
        if matching:
            decomposition[group].update({
                field: distribution([right[field] - left[field] for left, right in matching])
                for field in ("incumbent_remaining_ms", "incumbent_duration_ms", "incumbent_elapsed_ms")
            })
    return {**arms, "paired_delta_ms": paired, "incumbent_pairs": decomposition}


def queue_boundary_report(stage, checksums):
    schedules = read(stage / "schedules.json")
    measurements = {}
    for gpu in ("6", "7"):
        arms = {}
        for cap in (0, 8):
            folder = stage / f"private-short-d{cap}-admission-gpu{gpu}"
            sample = read(folder / "workload.json")
            receipt = read(folder / "receipt.json")
            require(receipt["trace_mode"] == "admission" and receipt["short_burst_decode_limit"] == cap,
                    "unexpected timing mode")
            parsed = check_replay(sample["short"], schedules[gpu])
            require(sorted(parsed["first_32_indices"]) == list(range(32)), "first cohort changed")
            log = (folder / "server.log").read_text()
            # The original replay verifier obtained this count from the startup log.
            startup = STARTUP.findall(log)
            require(len(startup) == 1, "missing or duplicate warmup count")
            arms[cap] = read_boundaries(log, sample, bind_observations(log, sample), int(startup[0][0]))
        measurements[gpu] = {
            name: summarize_pairs(arms[0][cohort], arms[8][cohort])
            for name, cohort in (("all", slice(96)), ("first_32", slice(32)), ("later_64", slice(32, 96)))
        }
    return {"capture_checksums_sha256": checksums, "paired_trace_queue_boundaries": measurements}
