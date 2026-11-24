"""Replay fixed prefill capacity and certify a deadline bound on the pinned RAG trace."""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import runpy
import tarfile


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "20261124-w3-c32-prefill-tail"
ACCOUNTING = ROOT / "20261124-w3-rag-tail-accounting" / "verified.json"


@dataclass(frozen=True)
class Request:
    index: int
    request_id: int
    tokens: int
    release: int
    measured_ms: float
    completion_offset_ns: int


@dataclass(frozen=True)
class Slot:
    start: int
    end: int
    tokens: int


def percentile_95(values):
    ordered = sorted(values)
    return ordered[int(len(ordered) * .95)]


def replay(jobs, slots, rank):
    """Allocate each preserved batch capacity, using only requests already admitted."""
    pending = [job.tokens for job in jobs]
    finished = [None] * len(jobs)
    for index, slot in enumerate(slots):
        ready = sorted((i for i, job in enumerate(jobs) if pending[i] and job.release <= index),
                       key=lambda i: rank(i, pending[i]))
        budget = slot.tokens
        for i in ready:
            used = min(budget, pending[i])
            pending[i] -= used
            budget -= used
            if pending[i] == 0:
                finished[i] = index
            if budget == 0:
                break
        if budget:
            raise ValueError("no admitted prompt can use a slot reserved for measured prompts")
    if any(pending):
        raise ValueError("preserved slots failed to serve every prompt")
    return finished


def constraints(jobs, slots, deadlines):
    """Hall constraints for jobs with contiguous sets of eligible slots."""
    prefix = [0]
    for slot in slots:
        prefix.append(prefix[-1] + slot.tokens)
    forced_late = 0
    limits = {}
    for i, job in enumerate(jobs):
        if deadlines[i] < job.release:
            forced_late |= 1 << i
    for left in sorted({job.release for job in jobs}):
        for right in sorted({d for d in deadlines if left <= d < len(slots) - 1}):
            subset = 0
            demand = 0
            for i, job in enumerate(jobs):
                if not forced_late & (1 << i) and left <= job.release and deadlines[i] <= right:
                    subset |= 1 << i
                    demand += job.tokens
            shortage = demand - (prefix[right + 1] - prefix[left])
            if shortage > 0:
                limits[subset] = max(limits.get(subset, 0), shortage)
    # Intervals ending at the last slot contain every released job whether or
    # not its deadline is relaxed. The recorded feasible allocation proves
    # those constraints, so they need not enter the late-request search.
    return forced_late, sorted(limits.items(), key=lambda item: (-item[1], item[0]))


def bit_indexes(mask):
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def deadline_plan(jobs, slots, target_latency_ns, late_budget):
    ends = [slot.end for slot in slots]
    deadlines = [bisect_right(ends, target_latency_ns - job.completion_offset_ns) - 1 for job in jobs]
    forced, limits = constraints(jobs, slots, deadlines)

    @lru_cache(maxsize=None)
    def search(relaxed):
        if relaxed.bit_count() > late_budget:
            return None
        for mask, shortage in limits:
            removed = sum(jobs[i].tokens for i in bit_indexes(mask & relaxed))
            if removed >= shortage:
                continue
            choices = sorted(bit_indexes(mask & ~relaxed), key=lambda i: (-jobs[i].tokens, i))
            remaining = late_budget - relaxed.bit_count()
            if sum(jobs[i].tokens for i in choices[:remaining]) < shortage - removed:
                return None
            for i in choices:
                answer = search(relaxed | (1 << i))
                if answer is not None:
                    return answer
            return None
        return relaxed

    relaxed = search(forced)
    if relaxed is None:
        return None
    effective = [len(slots) - 1 if relaxed & (1 << i) else deadline
                 for i, deadline in enumerate(deadlines)]
    finish = replay(jobs, slots, lambda i, _: (effective[i], jobs[i].release, jobs[i].index))
    if any(done > deadline for done, deadline in zip(finish, effective, strict=True)):
        raise ValueError("EDF replay contradicts interval feasibility")
    return finish


def read_workload(rows, admissions, executions, prefill):
    by_id = {int(row["response_id"].removeprefix("cmpl-"), 16): row for row in rows}
    if len(by_id) != len(rows) or len(rows) != 64:
        raise ValueError("expected 64 distinct measured requests")
    slots = []
    finish = {}
    consumed = dict.fromkeys(by_id, 0)
    for start, (span, _, _) in sorted(executions.items()):
        measured = [(rid, *work) for rid, work in prefill.get(start, {}).items() if rid in by_id]
        if not measured:
            continue
        slots.append(Slot(start, span.end, sum(tokens for _, _, tokens, _ in measured)))
        for rid, position, tokens, last in measured:
            if (start < admissions[rid][-1] or rid in finish or consumed[rid] + tokens != position
                    or bool(last) != (position == by_id[rid]["prompt_tokens"])):
                raise ValueError("recorded prompt allocation violates release or token accounting")
            consumed[rid] = position
            if last:
                finish[rid] = span.end
    if finish.keys() != by_id.keys():
        raise ValueError("measured prompts are missing terminal prefill")
    starts = [slot.start for slot in slots]
    jobs = []
    for rid, row in sorted(by_id.items(), key=lambda pair: pair[1]["request_index"]):
        release = bisect_left(starts, admissions[rid][-1])
        # For a moved completion, hold fixed the observed client-to-admission
        # and terminal-prefill-to-first-token portions of this exact request.
        jobs.append(Request(row["request_index"], rid, row["prompt_tokens"], release,
                            row["ttft"] * 1000, round(row["ttft"] * 1e9) - finish[rid]))
    if sum(job.tokens for job in jobs) != sum(slot.tokens for slot in slots):
        raise ValueError("measured prefill work differs from fixed slot capacity")
    return jobs, slots


def evaluate(rows, admissions, executions, prefill, reference):
    jobs, slots = read_workload(rows, admissions, executions, prefill)
    measured = percentile_95(job.measured_ms for job in jobs)
    if measured != reference["ttft_ms"]:
        raise ValueError("request latency differs from earlier RAG tail accounting")

    def latencies(finish):
        return [(job.completion_offset_ns + slots[step].end) / 1e6
                for job, step in zip(jobs, finish, strict=True)]

    late_budget = len(jobs) - int(len(jobs) * .95) - 1
    low, high = 0, round(measured * 1000)
    if deadline_plan(jobs, slots, high * 1000, late_budget) is None:
        # Rounding the measured p95 to microseconds may round down.
        high += 1
    while high - low > 1:
        mid = (low + high) // 2
        if deadline_plan(jobs, slots, mid * 1000, late_budget) is None:
            low = mid
        else:
            high = mid
    witness = deadline_plan(jobs, slots, high * 1000, late_budget)
    floor = percentile_95(latencies(witness))
    if not low / 1000 < floor <= high / 1000:
        raise ValueError("reported allocation does not certify the computed interval")
    fifo = replay(jobs, slots, lambda i, _: jobs[i].request_id)
    srpt = replay(jobs, slots, lambda i, remaining: (remaining, jobs[i].request_id))
    fifo_p95 = percentile_95(latencies(fifo))
    target = reference.get("frozen_vllm", {}).get("target_ms")
    return {
        "requests": len(jobs), "slots": len(slots), "tokens": sum(job.tokens for job in jobs),
        "measured_p95_ms": measured, "fifo_replay_p95_ms": fifo_p95,
        "fifo_replay_delta_pct": (fifo_p95 / measured - 1) * 100,
        "srpt_replay_p95_ms": percentile_95(latencies(srpt)),
        "oracle_p95_interval_ms": [low / 1000, high / 1000],
        "p95_target_ms": target,
        "oracle_at_final_target_feasible": None if target is None else (
            deadline_plan(jobs, slots, round(target * 1e6), late_budget) is not None
        ),
        "oracle_prefill_finish_changes_ms": {
            str(job.index): (step.end + job.completion_offset_ns) / 1e6 - job.measured_ms
            for job, step in zip(jobs, (slots[i] for i in witness), strict=True)
        },
    }


def verify():
    previous = json.loads(ACCOUNTING.read_bytes())
    path = TRACE / "raw-results.tar.gz"
    raw_hash = sha256(path.read_bytes()).hexdigest()
    if previous["trace_archive_sha256"] != raw_hash:
        raise ValueError("trace differs from validated RAG accounting")
    trace = runpy.run_path(str(TRACE / "verify.py"))
    stages = runpy.run_path(str(ROOT / trace["STEP"] / "stages.py"))
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for gpu, arms in previous["gpus"].items():
            result[gpu] = {}
            for mode, workloads in arms.items():
                prefix = f"gpu{gpu}/rag/{mode}"
                log = archive.extractfile(f"{prefix}/server.log").read().decode()
                _, admissions, executions, prefill = trace["parse_trace"](log, stages)
                result[gpu][mode] = {}
                for label, reference in workloads.items():
                    rows = json.load(archive.extractfile(f"{prefix}/{label}.json"))["measured"]
                    result[gpu][mode][label] = evaluate(rows, admissions, executions, prefill, reference)
    return {"trace_sha256": raw_hash, "accounting_sha256": sha256(ACCOUNTING.read_bytes()).hexdigest(),
            "gpus": result}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, allow_nan=False))
