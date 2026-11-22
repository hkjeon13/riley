"""Compare existing server-side request observations from two reversed repetitions."""

import json
from pathlib import Path
import re
import sys

from comparison_base import compare, p
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import REQUEST_COUNTS, make_workloads

ORDER = ("off-warm-r1", "private-r1", "private-r2", "off-warm-r2")
ORDERS = {"6": tuple(ORDER[i] for i in (1, 0, 3, 2)), "7": ORDER}
REQUEST_ID = re.compile(r"cmpl-[a-f0-9]{16}")
TTFT = "client_ttft_ms"
BEFORE_WRITE = "client_before_write_ms"
ADMISSION = "backend_submit_ms"
AFTER_ADMISSION = "admitted_to_first_visible_ms"
OTHER = "other_after_write_ms"
PARTS = (BEFORE_WRITE, ADMISSION, AFTER_ADMISSION, OTHER)


def bind_observations(log, raw):
    observations = {}
    snapshot = []
    for line in log.splitlines():
        if line.startswith("W3-OBS-SNAPSHOT "):
            _, retained, dropped = line.split()
            snapshot.append((int(retained), int(dropped)))
        elif line.startswith("W3-OBS "):
            _, request, admit, visible, tokens, status = line.split()
            require(REQUEST_ID.fullmatch(request) and request not in observations and status == "Completed",
                    f"invalid or duplicate observation {request}")
            observations[request] = (int(admit), int(visible), int(tokens))
    count = sum(REQUEST_COUNTS.values())
    require(snapshot == [(count, 0)] and len(observations) == count, "partial observation export")
    timings = {}
    for name, expected in REQUEST_COUNTS.items():
        rows = raw[name]["rows"]
        require(len(rows) == expected, f"{name} request count differs")
        phases = []
        for row in rows:
            request = row["response_id"]
            require(request in observations, f"missing server observation for {request}")
            admit, visible, tokens = observations.pop(request)
            total = row["ttft"] * 1_000
            before_write = (row["sent_at"] - row["start"]) * 1_000
            admit_ms = admit / 1_000_000
            visible_ms = visible / 1_000_000
            rest = total - before_write - admit_ms - visible_ms
            require(admit > 0 and visible > 0 and tokens == row["completion_tokens"]
                    and before_write >= 0 and rest >= 0,
                    f"observation total or usage differs for {request}")
            phases.append({TTFT: total, BEFORE_WRITE: before_write, ADMISSION: admit_ms,
                           AFTER_ADMISSION: visible_ms, OTHER: rest})
        timings[name] = phases
    require(not observations, "request observation not tied to client evidence")
    return timings


def compare_cohort(before, after, indices):
    rows = list((before[index], after[index]) for index in indices)
    changes = {}
    for key in (TTFT, *PARTS):
        base, candidate = ([pair[side][key] for pair in rows] for side in (0, 1))
        delta = [new - old for old, new in zip(base, candidate, strict=True)]
        changes[key] = {
            "off_warm_p50": p(base, .50), "private_p50": p(candidate, .50),
            "off_warm_p75": p(base, .75), "private_p75": p(candidate, .75),
            "paired_delta_p50": p(delta, .50), "paired_delta_mean": sum(delta) / len(delta),
            "slower_requests": sum(value > 0 for value in delta),
        }
    return {
        "requests": len(rows),
        "phase_ms": changes,
        # This is a recomputation using paired measured values, not a causal prediction.
        "private_p50_replacing_one_phase_with_off_warm_ms": {
            key: p([other[TTFT] - other[key] + base[key] for base, other in rows], .50)
            for key in PARTS
        },
    }


def verify(stage):
    evidence = stage.parent
    build = Path((stage / "build-path.txt").read_text().strip())
    parent = evidence / "w5-prefix-metrics-package-20261121.nEFyze5I/cpu-evidence/report.json"
    stock = read(parent)
    original = evidence / "w3-coder-rag-cache-20261121.kXvzFrJJ/off-first/workload.json"
    reference_hash = digest(original)
    require(reference_hash == "a77aa2092bd229a40038622d1a2de6768d2a8a6fdca159fbff42c20d9d6cfb7b",
            "reference output changed")
    require(stock["status"] == "passed" and stock["source_revision"] == "3951d90c8df381daef2ab88fd18ac9e8a83c4554",
            "parent report changed")
    toolchain = (build / "toolchain.txt").read_text()
    require(toolchain.startswith("rustc 1.85.0") and "V12.4." in toolchain, "unexpected toolchain")
    build_checks = {
        "input_sha256": digest(build / "build-inputs.sha256"),
        "result_sha256": digest(build / "build-result.sha256"),
        "artifacts": {name: sha for sha, name in
                      (line.split("  ", 1) for line in (build / "build-result.sha256").read_text().splitlines())},
    }
    data, plans, workloads = read(original), read(stage / "schedules.json"), make_workloads()
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "campaign sequence incomplete")
    runs, reservations, comparisons = {}, {}, {}
    for index, order in ORDERS.items():
        events = [row for row in timeline if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, f"{label}-gpu{index}"]
                                               for label in order for phase in ("START", "END")]
                and [row[2] for row in events] == sorted(row[2] for row in events),
                "GPU-local sequence differs")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(checked_reservation(stage / f"gpu{index}-after.json", index) == owner, "GPU owner changed")
        reservations.update(owner)
        raw, arms, timings = {}, {}, {}
        for label in order:
            condition, replica = label.rsplit("-", 1)
            tag = f"{label}-gpu{index}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["replica"] == replica, f"repetition differs for {tag}")
            raw[label], arms[label] = check_case(stage, condition, index, owner, stock, parent, build_checks,
                                                data, workloads, label=tag)
            timings[label] = bind_observations((folder / "server.log").read_text(), raw[label])
            arms[label]["observations"] = {name: len(rows) for name, rows in timings[label].items()}
            arms[label]["observation_source_sha256"] = receipt["artifacts_sha256"]["server.log"]
            arms[label]["fidelity"] = check_replay(raw[label]["short"], plans[index])
            runs[tag] = arms[label]
        comparisons[index] = {}
        for replica in ("r1", "r2"):
            before, after = f"off-warm-{replica}", f"private-{replica}"
            short_before, short_after = (raw[label]["short"]["rows"] for label in (before, after))
            errors = [abs(left["relative_write_error_ms"] - right["relative_write_error_ms"])
                      for left, right in zip(short_before, short_after, strict=True)]
            paired_error = p(errors, .95)
            require(paired_error <= 5, "paired client timing diverged")
            indices = arms[before]["fidelity"]["first_32_indices"]
            remaining = sorted(set(range(96)) - set(indices))
            timings_before, timings_after = (timings[label]["short"] for label in (before, after))
            comparisons[index][replica] = {
                "candidate_ran_first": order.index(after) < order.index(before),
                "short_and_rag": compare(raw[before], raw[after], arms[before], arms[after]),
                "paired_write_error_p95_ms": paired_error,
                "short_cohorts": {name: compare_cohort(timings_before, timings_after, group)
                                  for name, group in (("all", range(96)), ("first_32", indices), ("remaining_64", remaining))},
            }
    return {
        "status": "passed", "remote_evidence": str(stage), "remote_build": str(build),
        "source_revision": stock["source_revision"], "image_id": stock["image_id"], "build": build_checks,
        "stock_workload_sha256": reference_hash, "schedules_sha256": digest(stage / "schedules.json"),
        "reservations": reservations, "order": timeline, "runs": runs, "comparisons": comparisons,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
