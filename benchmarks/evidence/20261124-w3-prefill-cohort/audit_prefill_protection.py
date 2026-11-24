"""Estimate the new guard's exposure on two earlier, unchanged GPU schedules."""

from bisect import bisect_left
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import runpy
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
TRACE = ROOT / "20261124-w3-prefill-step-cost/raw-results.tar.gz"
TRACE_SHA256 = "e7fdf13a5d246dd3c154ba0d12cde6550cdfb6a0c2804cb6f5dd27415de6ebd3"
CELLS = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "wave-sync", "wave-50ms")


def assess(steps, admitted, history, estimated_completion):
    starts = [step.start for step in steps]
    groups = {request: bisect_left(starts, timestamp) for request, timestamp in admitted.items()}
    last = {request: points[-1][0].start for request, points in history.items()}
    lengths = {request: points[-1][2] for request, points in history.items()}
    if history.keys() != admitted.keys():
        raise ValueError("a recorded admission lacks prefill work")
    totals = Counter()
    for step in steps:
        if not (step.rows > 256 and 0 < step.decode <= 32
                and any(request in estimated_completion for request, _, _ in step.prefill)):
            continue
        ready = [request for request, timestamp in admitted.items() if timestamp <= step.start <= last[request]]
        if any(lengths[request] < 512 for request in ready):
            continue
        totals["eligible_mixed_steps"] += 1
        cohorts = Counter(groups[request] for request in ready)
        totals["steps_with_multiple_ready_in_one_cohort"] += max(cohorts.values()) > 1
        totals["steps_with_three_ready_in_one_cohort"] += max(cohorts.values()) >= 3
        totals["steps_with_four_ready_across_three_adjacent_cohorts"] += any(
            sum(cohorts[index + offset] for offset in range(3)) >= 4 for index in cohorts
        )
        live = Counter(groups[request] for request, timestamp in admitted.items()
                       if timestamp <= step.start <= max(last[request], estimated_completion.get(request, 0))
                       and lengths[request] >= 512)
        totals["steps_with_four_long_live_across_two_adjacent_cohorts"] += any(
            live[index] + live[index + 1] >= 4 for index in live
        )
    return dict(totals)


def audit():
    with TRACE.open("rb") as file:
        if file_digest(file, "sha256").hexdigest() != TRACE_SHA256:
            raise ValueError("trace is not the earlier verified baseline")
    parse = runpy.run_path(str(ROOT / "20261124-w3-rag-prefill-steps/verify.py"))["read_trace"]
    gpus = {}
    with tarfile.open(TRACE, "r:gz") as archive, TemporaryDirectory() as temporary:
        for gpu in (6, 7):
            prefix = f"gpu{gpu}/control"
            path = Path(temporary) / f"gpu{gpu}.log"
            raw = archive.extractfile(f"{prefix}/server.log").read()
            path.write_bytes(raw)
            visible = {line.split()[1] for line in raw.decode().splitlines() if line.startswith("W3-RAG-VISIBLE ")}
            ids, first_tokens, steps, history, admitted, _, _ = parse(path, visible)
            cells = {}
            for label in CELLS:
                measurements = json.load(archive.extractfile(f"{prefix}/{label}.json"))["measured"]
                estimated_completion = {
                    ids[int(row["response_id"][5:], 16)][0]: (
                        first_tokens[row["response_id"]][1] + round((row["e2e"] - row["ttft"]) * 1e9)
                    )
                    for row in measurements
                }
                if len(estimated_completion) != (64 if label.startswith("rag-") else 40):
                    raise ValueError(f"unexpected request count in {prefix}/{label}")
                cells[label] = assess(steps, admitted, history, estimated_completion)
            gpus[str(gpu)] = cells
    return {"trace_sha256": TRACE_SHA256, "gpus": gpus}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
