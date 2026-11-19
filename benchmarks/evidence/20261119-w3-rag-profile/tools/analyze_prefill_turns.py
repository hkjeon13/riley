"""Reconstruct prefill steps and identify which turns overlap measured TTFT tails."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import NamedTuple

from analyze_prefill_wait_shapes import merge_spans
from summarize_request_stages import prefill_union, read_phases, require, summarize


AGE_NS = 100_000_000


class Chunk(NamedTuple):
    request: str
    start: int
    end: int
    first_token: int
    next_token: int
    final: int
    batch_rows: int
    decode_rows: int
    gpu_ns: int


class Step(NamedTuple):
    start: int
    end: int
    chunks: list[Chunk]


def reconstruct(events):
    chunks = []
    admitted = {}
    prompt_lengths = {}
    for request, phases in events.items():
        (admission, _), = phases["admitted"]
        admitted[request] = admission
        finished = 0
        for (begin, fields), (end, receipt) in zip(phases["prefill_start"], phases["prefill_return"], strict=True):
            first, following, final, batch_rows, decode_rows = fields
            completed, gpu_ns = receipt
            require(admission <= begin < end and first == finished < following == completed
                    and 0 <= decode_rows < batch_rows and 0 <= gpu_ns <= end - begin, "invalid chunk trace")
            chunks.append(Chunk(request, begin, end, first, following, final, batch_rows, decode_rows, gpu_ns))
            finished = following
        require(bool(finished) and phases["prefill_start"][-1][1][2] == 1, "request did not record its final prompt chunk")
        prompt_lengths[request] = finished
    require(len(set(admitted.values())) == len(admitted), "admission ordering is ambiguous")
    steps = []
    for chunk in sorted(chunks, key=lambda item: item.start):
        if steps and chunk.start < steps[-1].end:
            previous = steps[-1]
            previous.chunks.append(chunk)
            steps[-1] = Step(previous.start, max(previous.end, chunk.end), previous.chunks)
        else:
            steps.append(Step(chunk.start, chunk.end, [chunk]))
    turns = defaultdict(list)
    for index, step in enumerate(steps):
        members = step.chunks
        shapes = {(chunk.batch_rows, chunk.decode_rows, chunk.gpu_ns) for chunk in members}
        require(len(shapes) == 1 and len({chunk.request for chunk in members}) == len(members),
                "overlapping spans do not identify a unique step")
        batch, decode, _ = next(iter(shapes))
        require(sum(chunk.next_token - chunk.first_token for chunk in members) == batch - decode,
                "the grouped prompt chunks do not account for the step")
        for chunk in members:
            turns[chunk.request].append((index, chunk))
    require(merge_spans((step.start, step.end) for step in steps) == prefill_union(events),
            "reconstructed steps differ from the traced prefill timeline")
    return steps, turns, admitted, prompt_lengths


def request_wait(row, steps, turns, admitted, lengths, cap):
    request = row["request_id"]
    timings = dict.fromkeys(("only_earlier_admissions", "only_later_admissions", "both_admission_orders",
                             "with_finishing_competitor", "while_at_most_cap_tokens_remained",
                             "later_only_after_100ms_with_decode",
                             "other_prefill_step_total", "step_boundary_edges"), 0)
    for (previous, chunk), (following, next_chunk) in zip(turns[request], turns[request][1:]):
        timings["step_boundary_edges"] += steps[previous].end - chunk.end + next_chunk.start - steps[following].start
        remaining = lengths[request] - chunk.next_token
        for step in steps[previous + 1:following]:
            duration = step.end - step.start
            direction = {admitted[item.request] < admitted[request] for item in step.chunks}
            label = "both_admission_orders" if len(direction) > 1 else (
                "only_earlier_admissions" if True in direction else "only_later_admissions"
            )
            timings[label] += duration
            timings["other_prefill_step_total"] += duration
            if (label == "only_later_admissions" and step.start - steps[previous].end >= AGE_NS
                and step.chunks[0].decode_rows > 0):
                timings["later_only_after_100ms_with_decode"] += duration
            if any(item.final for item in step.chunks):
                timings["with_finishing_competitor"] += duration
            if remaining <= cap:
                timings["while_at_most_cap_tokens_remained"] += duration
    old_overlap_ns = round(row["other_prefill_during_wait_ms"]["between_steps"] * 1e6)
    require(timings["other_prefill_step_total"] + timings["step_boundary_edges"] == old_overlap_ns,
            "attributed waiting time disagrees with the original request record")
    other = timings["other_prefill_step_total"]
    return {"request_index": row["request_index"], "prompt_tokens": lengths[request],
            "span_ms": {name: duration / 1e6 for name, duration in timings.items()},
            "later_only_fraction": timings["only_later_admissions"] / other if other else None,
            "near_completion_fraction": timings["while_at_most_cap_tokens_remained"] / other if other else None}


def analyze(root, cap):
    original = summarize(root)
    log = root / "qwen14-stage-trace/server.log"
    events = read_phases(log)
    if not log.exists():
        log = log.with_suffix(".log.gz")
    steps, turns, admitted, lengths = reconstruct(events)
    cells = {}
    for name, cell in original["cells"].items():
        cohorts = set(cell["at_or_above_client_ttft_p95"]["request_indices"])
        rows = [request_wait(row, steps, turns, admitted, lengths, cap) for row in cell["rows"]
                if row["request_index"] in cohorts]
        require(len(rows) == len(cohorts) > 0, "tail requests are incomplete")
        cells[name] = {"requests": rows,
                       "median_span_ms": {name: median(row["span_ms"][name] for row in rows) for name in rows[0]["span_ms"]}}
    return {"source_log_sha256": sha256(log.read_bytes()).hexdigest(), "build_sha256": original["build_sha256"],
            "physical_gpu": original["physical_gpu"], "traced_requests": len(turns), "prefill_steps": len(steps),
            "requests_with_multiple_prefill_steps": sum(len(request_turns) > 1 for request_turns in turns.values()),
            "max_remaining_rows_for_near_completion": cap, "tail_cohorts": cells,
            "scope": "Recorded host prefill interval overlap between completed turns, classified by engine admission time. "
                     "The 100 ms interval is elapsed since the traced step return, without scheduler readiness state."}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(analyze(args.root, 512), handle, indent=2, allow_nan=False)
        handle.write("\n")
