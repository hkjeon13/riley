"""Recompute closed-C32 request phases and freeze the two arrival schedules."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import runpy
from statistics import median
import tarfile

SOURCE = Path(__file__).resolve().parent.parent / "20261124-w3-package-coder14"
BASE = runpy.run_path(str(SOURCE / "verify.py"))
ARCHIVE_SHA = "5c9ee54d9a39b48e75a483116b64b099ac180a0dbe0ad18215c57cca5d05671e"
ENGINES = ("riley", "vllm")
LABEL = "closed-c32"


def require(valid, message):
    if not valid:
        raise ValueError(message)


def percentile(values, q):
    return sorted(values)[min(int(len(values) * q), len(values) - 1)]


def phase_tpot(row):
    return 1e3 * (row["e2e"] - row["ttft"]) / (row["completion_tokens"] - 1)


def read_cell(archive, gpu, engine, spec):
    relative = f"gpu{gpu}/coder14-{engine}"
    raw = archive.extractfile(f"{relative}/{LABEL}.json").read()
    receipt = json.load(archive.extractfile(f"{relative}/receipt.json"))
    require(receipt["status"] == "passed" and receipt["gpu"] == BASE["GPUS"][gpu]
            and receipt["results"][LABEL] == sha256(raw).hexdigest(), "C32 data differs from the arm receipt")
    data = json.loads(raw)
    rows, stats = data["measured"], data["statistics"]
    require(stats["config"] == spec and len(rows) == stats["ok"] == stats["requests"] == 192
            and stats["warmup_ok"] == 32 and stats["errors"] == stats["rejected_429"] == 0
            and [row["request_index"] for row in rows] == list(range(32, 224)), "C32 request coverage differs")
    for row in rows:
        require(row["completion_tokens"] == 64 and row["finish"] == "length"
                and all(isfinite(row[field]) for field in ("start", "sent_at", "ttft", "e2e"))
                and row["start"] <= row["sent_at"] <= row["start"] + row["ttft"] < row["start"] + row["e2e"],
                "invalid C32 request timing or token count")
    require(abs(percentile([phase_tpot(row) for row in rows], .50) - stats["tpot_p50_ms"]) < 1e-9,
            "C32 TPOT does not reproduce the reported benchmark")
    return rows, sha256(raw).hexdigest()


def compare(riley, vllm):
    riley_tpot, vllm_tpot = ([phase_tpot(row) for row in rows] for rows in (riley, vllm))
    first_deltas, last_deltas, directions = [], [], []
    for a, b, at, bt in zip(riley, vllm, riley_tpot, vllm_tpot, strict=True):
        first_ms = 1e3 * (a["ttft"] - b["ttft"])
        last_ms = 1e3 * (a["e2e"] - b["e2e"])
        first_deltas.append(first_ms)
        last_deltas.append(last_ms)
        directions.append((first_ms < 0, last_ms < 0, at > bt))
    return {
        "paired_requests": len(riley),
        "riley_over_vllm_tpot_p50": percentile(riley_tpot, .50) / percentile(vllm_tpot, .50),
        "riley_minus_vllm_first_content_ms_median": median(first_deltas),
        "riley_minus_vllm_last_content_ms_median": median(last_deltas),
        "riley_first_content_earlier": sum(first for first, _, _ in directions),
        "riley_last_content_earlier": sum(last for _, last, _ in directions),
        "riley_higher_tpot": sum(tpot for _, _, tpot in directions),
        "riley_earlier_first_and_last_with_higher_tpot": directions.count((True, True, True)),
    }


def freeze(rows, digest, spec, origin):
    epoch = min(row["sent_at"] for row in rows)
    schedule = [{**{key: row[key] for key in BASE["IDENTITY"]}, "offset_ns": round(1e9 * (row["sent_at"] - epoch))}
                for row in rows]
    return {"cell": spec, "requests": spec["requests"], "origin": origin, "source_sha256": digest,
            "schedule": schedule}


def prepare(path):
    require(sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA, "archived comparison has changed")
    observed, schedules = {}, {}
    with tarfile.open(path, "r:gz") as archive:
        spec = json.load(archive.extractfile("cells.json"))[LABEL]
        require(spec == {"prompt_words": [30], "max_tokens": 64, "concurrency": 32, "requests": 192,
                         "warmup": 32, "seed": 1}, "unexpected C32 request settings")
        for gpu in BASE["GPUS"]:
            arms = {engine: read_cell(archive, gpu, engine, spec) for engine in ENGINES}
            riley, vllm = (arms[engine][0] for engine in ENGINES)
            require(all(all(a[key] == b[key] for key in BASE["IDENTITY"])
                        for a, b in zip(riley, vllm, strict=True)), "C32 engine inputs or token counts differ")
            timelines = {engine: freeze(rows, digest, spec, engine) for engine, (rows, digest) in arms.items()}
            skew_ms = [abs(a["offset_ns"] - b["offset_ns"]) / 1e6 for a, b in
                       zip(timelines["riley"]["schedule"], timelines["vllm"]["schedule"], strict=True)]
            cohorts = {f"{32+i}-{32+i+31}": compare(riley[i:i+32], vllm[i:i+32]) for i in range(0, len(riley), 32)}
            observed[gpu] = {"overall": compare(riley, vllm), "ordinal_cohorts": cohorts,
                             "abs_paired_write_time_ms_p95": percentile(skew_ms, .95),
                             "source_arm_hashes": {engine: arm[1] for engine, arm in arms.items()}}
            schedules[gpu] = {f"c32-{engine}-arrivals": timeline for engine, timeline in timelines.items()}
    return {"schedules": schedules, "paired_origin_differences": observed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", default=SOURCE / "raw-results.tar.gz")
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive), indent=2, allow_nan=False))
