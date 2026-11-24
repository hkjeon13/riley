"""Relate successive C32 first-token delays to measured client completion overlap."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
from statistics import median
import tarfile


PRIOR = Path(__file__).resolve().parents[1] / "20261124-w3-prefill-deadline-capacity/raw-results.tar.gz"
PRIOR_SHA256 = "8695103d7e6df8cafa53a0abb2ca9e4ee94834eb2487b946c2ece750ddc41b30"
IDENTITY = ("request_index", "scheduled_write_ns", "input_sha256", "output_sha256", "prompt_tokens",
            "completion_tokens", "finish")


def client_phases(row):
    before = row["sent_at"] - row["start"]
    after = row["ttft"] - before
    if not 0 <= before <= row["ttft"] < row["e2e"]:
        raise ValueError("invalid client latency phases")
    return before * 1000, after * 1000


def prior_completions(previous, current):
    first_send = min(row["sent_at"] for row in current)
    offset_ms = [(row["start"] + row["e2e"] - first_send) * 1000 for row in previous]
    return {"prior_responses_still_open": sum(value > 0 for value in offset_ms),
            "prior_completion_p50_from_first_send_ms": median(offset_ms),
            "prior_completion_last_from_first_send_ms": max(offset_ms)}


def wave_summary(control, candidate):
    if len(control) != 192 or any(any(a[k] != b[k] for k in IDENTITY)
                                    for a, b in zip(control, candidate, strict=True)):
        raise ValueError("C32 request identities differ")
    scheduled = [row["scheduled_write_ns"] for row in control]
    # Each frozen C32 group has 32 sends. The spacing between groups is at
    # least one second, while all within-group gaps are shorter than one second.
    if [index for index, (a, b) in enumerate(zip(scheduled, scheduled[1:]), 1)
            if b - a >= 1_000_000_000] != list(range(32, 192, 32)):
        raise ValueError("the frozen six C32 groups changed")
    groups = []
    for start in range(0, 192, 32):
        baseline = control[start:start + 32]
        trial = candidate[start:start + 32]
        displacements = [tuple(b - a for a, b in zip(client_phases(x), client_phases(y), strict=True))
                         for x, y in zip(baseline, trial, strict=True)]
        row = {
            "group": start // 32 + 1,
            "before_send_paired_p50_delta_ms": median(x[0] for x in displacements),
            "after_send_paired_p50_delta_ms": median(x[1] for x in displacements),
        }
        if start:
            row["prior_group"] = {
                "control": prior_completions(control[start - 32:start], baseline),
                "candidate": prior_completions(candidate[start - 32:start], trial),
            }
        groups.append(row)
    return groups


def summarize(archive):
    with archive.open("rb") as file:
        fingerprint = file_digest(file, "sha256").hexdigest()
    if archive.resolve() == PRIOR and fingerprint != PRIOR_SHA256:
        raise ValueError("the earlier reference archive changed")
    result = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for gpu in (6, 7):
            captures = [json.load(bundle.extractfile(f"gpu{gpu}/{mode}/c32-riley-arrivals.json"))["measured"]
                        for mode in ("control", "candidate")]
            result[str(gpu)] = wave_summary(*captures)
    return {"source_sha256": fingerprint, "gpus": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=PRIOR)
    args = parser.parse_args()
    print(json.dumps(summarize(args.archive), indent=2, allow_nan=False))
