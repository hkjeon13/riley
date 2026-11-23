"""Count candidate placements in the frozen RAG trace before applying aging checks."""

from collections import Counter
import json
from pathlib import Path
import runpy
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "20261124-w3-rag-prefill-steps"


def placements(steps, submitted, prompt_lengths, activation):
    consumed = Counter()
    minimum = activation["long_prompt_tokens_min"]
    for step in steps:
        if (step.rows == 1024 and len(step.prefill) == 1
                and activation["decode_rows_min"] <= step.decode <= activation["decode_rows_max"]):
            head, budget, _ = step.prefill[0]
            if (prompt_lengths.get(head, 0) >= minimum and budget == 1024 - step.decode
                    and budget >= activation["prefill_budget_min"]):
                eligible = sorted(req for req, length in prompt_lengths.items()
                                  if req < head and submitted[req] <= step.start and length >= minimum
                                  and 0 < consumed[req] <= length - budget)
                if eligible:
                    yield step, head, budget, consumed[head], [(req, consumed[req]) for req in eligible]
        for req, count, _ in step.prefill:
            consumed[req] += count


def count_placements(steps, submitted, prompt_lengths, activation):
    counts = Counter(step.decode for step, *_ in placements(steps, submitted, prompt_lengths, activation))
    return dict(sorted(counts.items()))


def summarize(summarize_placements):
    activation = json.loads((HERE / "registration.json").read_bytes())["activation"]
    reader = runpy.run_path(str(SOURCE / "verify.py"))
    prior = reader["PRIOR"]
    arm = reader["ARM"]
    counts = {}
    with tarfile.open(SOURCE / "raw-results.tar.gz", "r:gz") as archive, tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "server.log"
        log.write_bytes(archive.extractfile(f"{arm}/server.log").read())
        observations = prior["observations"](log)
        admits, _, steps, by_scheduler, submitted, *_ = reader["read_trace"](log, observations.keys())
        for label in prior["LABELS"]:
            data = json.load(archive.extractfile(f"{arm}/{label}.json"))
            ids = data["warmup_response_ids"] + [row["response_id"] for row in data["measured"]]
            schedulers = [admits[int(request[5:], 16)][0] for request in ids]
            # These completed requests committed the full input by their final prefill step.
            prompt_lengths = {request: by_scheduler[request][-1][2] for request in schedulers}
            counts[label] = summarize_placements(steps, submitted, prompt_lengths, activation)
    return {
        "gpu": 7,
        "source_committed_steps": len(steps),
        "includes_http_warmup": True,
        "accounts_for_aging_override": False,
        "placements_by_decode_rows": counts,
    }


if __name__ == "__main__":
    print(json.dumps(summarize(count_placements), indent=2))
