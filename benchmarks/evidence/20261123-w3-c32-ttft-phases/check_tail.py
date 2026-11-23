"""Attribute the recorded C32 percentile changes to paired request phases and rank shifts."""

from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys

MANIFESTS = {
    "original": "81a874d787c117ce20eb93baf0e0fb9f61ab9c0027d0ea0135b37d8a011b78be",
    "late12": "9109f0f65df21c37d5ca1b1fadfacb6058ccf5d6c95abcb491d3c228ac0e9c23",
}
PARTS = ("client_before_write", "backend_submit", "admitted_to_first_visible", "other_after_write")
IDENTITY = ("input_sha256", "prompt_ids_sha256", "token_ids_sha256", "output_sha256", "finish",
            "prompt_tokens", "completion_tokens")


def read_pinned(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError(f"archived evidence changed at {path.name}")
    return data


def per_request(root, manifest, tag):
    observed, queues = {}, {}
    workload, log = (read_pinned(root / tag / file, manifest[f"{tag}/{file}"])
                     for file in ("workload.json", "server.log"))
    snapshot, workers = [], []
    for line in log.decode().splitlines():
        if line.startswith("W3-OBS-SNAPSHOT "):
            snapshot.append(tuple(map(int, line.split()[1:])))
        elif line.startswith("W3-HTTP-WORKERS "):
            workers.append(int(line.split()[1]))
        elif line.startswith("W3-HTTP-QUEUE "):
            _, request, duration = line.split()
            if request in queues or int(duration) < 0:
                raise ValueError(f"unexpected HTTP queue observation in {tag}")
            queues[request] = int(duration) / 1_000_000
        elif line.startswith("W3-OBS "):
            _, request, submit, visible, tokens, status = line.split()
            if request in observed or status != "Completed":
                raise ValueError(f"unexpected terminal observation in {tag}")
            observed[request] = tuple(map(int, (submit, visible, tokens)))
    if (snapshot != [(164, 0)] or workers != [64] or len(observed) != 164
            or queues.keys() != observed.keys()):
        raise ValueError(f"unexpected observation buffer counts in {tag}")
    rows = json.loads(workload)["short"]["rows"]
    if len(rows) != 96 or len({row["response_id"] for row in rows}) != len(rows):
        raise ValueError(f"unexpected short workload in {tag}")
    measurements = []
    for row in rows:
        submit, visible, tokens = observed[row["response_id"]]
        ttft = 1000 * row["ttft"]
        early = 1000 * (row["sent_at"] - row["start"])
        submit_ms, visible_ms = (value / 1_000_000 for value in (submit, visible))
        remaining_ms = ttft - early - submit_ms - visible_ms
        if row["status"] != 200 or row["error"] or tokens != row["completion_tokens"] or min(
                early, submit_ms, visible_ms, remaining_ms) < 0:
            raise ValueError(f"latency or completion disagrees with the archived request in {tag}")
        measurements.append({
            "identity": tuple(row[key] for key in IDENTITY),
            "ttft": ttft,
            "http_queue": queues[row["response_id"]],
            **dict(zip(PARTS, (early, submit_ms, visible_ms, remaining_ms), strict=True)),
        })
    return measurements


def compare(base, candidate, q):
    size = len(base)
    rank = min(int(size * q), size - 1)
    old_rank, new_rank = (sorted(range(size), key=lambda index: arm[index]["ttft"])[rank]
                          for arm in (base, candidate))
    old, new, paired = base[old_rank], candidate[new_rank], base[new_rank]
    changes = {key: new[key] - paired[key] for key in PARTS}
    rank_shift = paired["ttft"] - old["ttft"]
    difference = new["ttft"] - old["ttft"]
    if not isclose(difference, sum(changes.values()) + rank_shift, abs_tol=1e-8):
        raise ValueError("latency and rank decomposition disagree")
    return {
        "baseline_ms": old["ttft"], "candidate_ms": new["ttft"], "difference_ms": difference,
        "baseline_rank_request": old_rank + 1, "candidate_rank_request": new_rank + 1,
        "paired_request_phase_deltas_ms": changes, "baseline_rank_shift_ms": rank_shift,
        "paired_request_http_queue_ms": [paired["http_queue"], new["http_queue"]],
        "paired_request_other_after_write_ms": [paired["other_after_write"], new["other_after_write"]],
    }


def report(root):
    comparisons = {}
    for schedule, expected in MANIFESTS.items():
        schedule_root = root.with_name(f"{root.name}-{schedule}")
        manifest = {}
        for line in read_pinned(schedule_root / "result.sha256", expected).decode().splitlines():
            digest, name = line.split("  ", 1)
            manifest[name] = digest
        comparisons[schedule] = {}
        for gpu in (6, 7):
            comparisons[schedule][str(gpu)] = {}
            for trace in ("control", "admission"):
                base, candidate = (per_request(schedule_root, manifest, f"private-short-d{limit}-{trace}-gpu{gpu}")
                                   for limit in (0, 8))
                if any(a["identity"] != b["identity"] for a, b in zip(base, candidate, strict=True)):
                    raise ValueError("candidate changed the input or output for a paired request")
                comparisons[schedule][str(gpu)][trace] = {
                    "matched_requests": len(base), "p50": compare(base, candidate, 0.50),
                    "p95": compare(base, candidate, 0.95),
                }
    return {"source": str(root), "result_manifest_sha256": MANIFESTS, "comparisons": comparisons}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_tail.py ARRIVAL_PHASE_DIRECTORY")
    print(json.dumps(report(Path(sys.argv[1])), indent=2, allow_nan=False))
