"""Count measured admissions that share the first eligible recorded step."""

from bisect import bisect_left
from collections import Counter
from hashlib import file_digest
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "20261124-w3-prefill-step-cost/raw-results.tar.gz"
CELLS = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "wave-sync", "wave-50ms")


def summarize():
    with SOURCE.open("rb") as handle:
        fingerprint = file_digest(handle, "sha256").hexdigest()
    manifest = json.loads((ROOT / "20261119-improvement-qualification/manifest.json").read_bytes())
    pending, pins = [manifest], []
    while pending:
        row = pending.pop()
        if isinstance(row, dict):
            if row.get("path") == str(SOURCE.relative_to(ROOT.parents[1])):
                pins.append(row["sha256"])
            pending.extend(row.values())
        elif isinstance(row, list):
            pending.extend(row)
    if not pins or set(pins) != {fingerprint}:
        raise ValueError("admission source differs from the qualified trace")
    gpus = {}
    with tarfile.open(SOURCE, "r:gz") as archive:
        for gpu in (6, 7):
            path = f"gpu{gpu}/control"
            admissions, starts = {}, []
            log = archive.extractfile(f"{path}/server.log").read().decode()
            for line in log.splitlines():
                if line.startswith("W3-RAG-ADMIT "):
                    engine, _, at = map(int, line.split()[1:])
                    if engine in admissions:
                        raise ValueError("duplicate admission")
                    admissions[engine] = at
                elif line.startswith("W3-RAG-STEP "):
                    starts.append(int(line.split()[2]))
            if not starts or any(a >= b for a, b in zip(starts, starts[1:])):
                raise ValueError("missing or unordered steps")
            cells = {}
            for label in CELLS:
                measured = json.load(archive.extractfile(f"{path}/{label}.json"))["measured"]
                releases = Counter(bisect_left(starts, admissions[int(row["response_id"][5:], 16)])
                                   for row in measured)
                if sum(releases.values()) != (64 if label.startswith("rag-") else 40) or len(starts) in releases:
                    raise ValueError("incomplete request cohort")
                cells[label] = {
                    "admitted_alone": sum(size == 1 for size in releases.values()),
                    "admitted_with_other_measured_requests": sum(size for size in releases.values() if size > 1),
                    "largest_cohort": max(releases.values()),
                }
            gpus[str(gpu)] = cells
    return {"source_sha256": fingerprint, "gpus": gpus}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, allow_nan=False))
