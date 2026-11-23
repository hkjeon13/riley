"""Bind engine timestamps to request receipts and compare paired phase durations."""
from observation_reference import ADMISSION

TRACE_PHASES = ("submit", "send", "worker", "admitted")
EXECUTION_KINDS = ("pure_decode", "mixed_or_prefill_up_to_256", "mixed_or_prefill_above_256")
TRACE_FIELDS = (
    "submit_to_send_ms", "send_to_worker_ms", "worker_to_admitted_ms",
    "total_engine_submit_ms", "outer_submit_remainder_ms", "queue_outside_execution_ms",
    *EXECUTION_KINDS,
)


def engine_trace(log, sample, observations, warmup_count):
    phases, spans, dropped = {}, [], []
    for line in log.splitlines():
        if line.startswith("W3-REQUEST-PHASE "):
            _, request, phase, value = line.split()
            known = phases.setdefault(int(request), {})
            require(phase in TRACE_PHASES and phase not in known, "unexpected or duplicate request trace")
            known[phase] = int(value)
        elif line.startswith("W3-ENGINE-SPAN "):
            start, end, rows, decode = map(int, line.split()[1:])
            require(start < end and 0 <= decode <= rows <= 1024 and rows > 0, "invalid engine execution span")
            require(not spans or start >= spans[-1][1], "engine execution spans overlap")
            kind = EXECUTION_KINDS[0 if rows == decode else 1 if rows <= 256 else 2]
            spans.append((start, end, kind))
        elif line.startswith("W3-DROPPED "):
            dropped.append(int(line.split()[1]))
    require(dropped == [0] and spans, "trace is missing or incomplete")
    records = {}
    first_id = first_submit = None
    for cell in REQUEST_COUNTS:
        records[cell] = []
        for response, observed in zip(sample[cell]["rows"], observations[cell], strict=True):
            request = int(response["response_id"].removeprefix("cmpl-"), 16)
            require(request in phases and set(phases[request]) == set(TRACE_PHASES), "missing request timepoints")
            timepoints = phases.pop(request)
            submit, sent, worker, admitted = (timepoints[name] for name in TRACE_PHASES)
            require(submit <= sent <= worker <= admitted, "request timepoints out of order")
            if first_id is None:
                first_id, first_submit = request, submit
            else:
                first_id, first_submit = min(first_id, request), min(first_submit, submit)
            engine_ns = admitted - submit
            remaining = observed[ADMISSION] - engine_ns / 1_000_000
            require(remaining >= -0.001, "engine submit escaped the outer request measurement")
            overlap = dict.fromkeys(EXECUTION_KINDS, 0)
            for start, end, kind in spans:
                if start >= worker:
                    break
                overlap[kind] += max(0, min(worker, end) - max(sent, start))
            queue_ns = worker - sent
            non_execution_ns = queue_ns - sum(overlap.values())
            require(non_execution_ns >= 0, "execution overlap exceeds queue wait")
            records[cell].append({
                "submit_to_send_ms": (sent - submit) / 1_000_000,
                "send_to_worker_ms": queue_ns / 1_000_000,
                "worker_to_admitted_ms": (admitted - worker) / 1_000_000,
                "total_engine_submit_ms": engine_ns / 1_000_000,
                "outer_submit_remainder_ms": remaining,
                "queue_outside_execution_ms": non_execution_ns / 1_000_000,
                **{kind: value / 1_000_000 for kind, value in overlap.items()},
                "client_ttft_ms": observed["client_ttft_ms"],
            })
    require(len(phases) == warmup_count and first_id is not None, "unmatched requests differ from the warmup count")
    for request, times in phases.items():
        require(request < first_id and set(times) == set(TRACE_PHASES), "invalid pre-replay warmup request")
        submit, sent, worker, admitted = (times[name] for name in TRACE_PHASES)
        require(submit <= sent <= worker <= admitted < first_submit, "warmup overlaps measured requests")
    stats = {
        "spans_by_kind": {kind: sum(span[2] == kind for span in spans) for kind in EXECUTION_KINDS},
        "warmup_requests": warmup_count,
        "measured_requests": sum(map(len, records.values())),
    }
    return records, stats


def phase_pairs(before, after):
    phases = (*TRACE_FIELDS, "client_ttft_ms")
    result = {}
    for group, indices in (("all", range(96)), ("first_32", range(32)), ("later_64", range(32, 96))):
        local = {}
        for phase in phases:
            base = [before["short"][index][phase] for index in indices]
            new = [after["short"][index][phase] for index in indices]
            delta = [b - a for a, b in zip(base, new, strict=True)]
            local[phase] = {"base_p50_ms": p(base, .5), "base_p95_ms": p(base, .95),
                            "candidate_p50_ms": p(new, .5), "candidate_p95_ms": p(new, .95),
                            "paired_delta_p50_ms": p(delta, .5),
                            "paired_delta_mean_ms": sum(delta) / len(delta)}
        local["with_wide_span_overlap"] = {
            key: sum(arm["short"][index][EXECUTION_KINDS[2]] > 0 for index in indices)
            for key, arm in (("baseline", before), ("candidate", after))
        }
        result[group] = local
    return result


def ordinary_phase_pairs(before, after):
    phases = ("client_ttft_ms", "client_before_write_ms", ADMISSION,
              "admitted_to_first_visible_ms", "other_after_write_ms")
    return {
        group: {
            phase: p([after[index][phase] - before[index][phase] for index in indices], .5)
            for phase in phases
        }
        for group, indices in (("all", range(96)), ("first_32", range(32)), ("later_64", range(32, 96)))
    }
