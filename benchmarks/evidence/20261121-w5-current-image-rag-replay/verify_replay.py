"""Verify matched timing, output, and resources for the fixed RAG schedules."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean

from fixed_arrival import MEASURED, bodies, read_schedule
from load import percentile
from compare_allocator_performance import ORDER, SERVER_ARGS, expected_image
from current_binding import PROFILE, binding, reservation
from verify_allocator_image import LIB_SHA
from verify_a100_candidate import digest, gpu_evidence, read, require


ORIGINS = ("parent-first", "packaged-first")
METRICS = ("ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
FIELDS = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")


def output_keys(rows):
    return [tuple(row[key] for key in FIELDS) for row in rows]


def checked_scope(package, evidence, stage, name):
    folder = stage / name
    parent = name.startswith("parent-")
    _, expected = expected_image(package, evidence, stage, name)
    receipt = gpu_evidence(folder, expected, int((folder / "exit").read_text()))
    require(receipt["status"] == "passed" and receipt["api_checks"] == 40, "runtime API failed")
    process = read(folder / "process-after.json")
    require(process == read(folder / "process-before.json") and process["image_id"] == expected["image_id"]
            and tuple(process["server_arguments"]) == SERVER_ARGS,
            "different process or image during replay")
    allocator = process["allocator"]
    require((allocator == "glibc" if parent else allocator["library_sha256"] == LIB_SHA
            and allocator["malloc_conf"] == PROFILE and allocator["mounts"] == ["/models"]),
            "runtime allocator differs")
    holders = reservation(folder)
    marks = [line.split()[1] for line in (folder / "gpu-evidence/api.log").read_text().splitlines()
             if line.startswith("FIXED_ARRIVAL_COMPLETED ")]
    ordered = ORIGINS if name.endswith("-first") else tuple(reversed(ORIGINS))
    require(tuple(marks) == ordered, "fixed schedule execution order changed")
    return {"image_id": expected["image_id"], "api_checks": receipt["api_checks"], "allocator": allocator, "server_arguments": process["server_arguments"],
            "report_sha256": receipt["gpu_receipt_sha256"], "artifacts_sha256": receipt["artifacts_sha256"],
            "reservations_after": holders}


def checked_replay(stage, name, origin, schedule, source_sha):
    folder = stage / name / "gpu-evidence"
    summary = read(folder / f"{origin}.summary.json")
    rows = read(folder / f"{origin}.records.json")
    require(summary["config"]["schedule_sha256"] == source_sha and summary["ok"] == summary["requests"] == MEASURED
            and summary["errors"] == summary["rejected_429"] == 0, "incomplete replay")
    require([row["request_index"] for row in rows] == list(schedule), "replay request IDs differ")
    errors = []
    for row in rows:
        require(row["scheduled_write"] == schedule[row["request_index"]]
                and isfinite(row["relative_write_error_ms"])
                and 0 <= row["pre_write"] <= row["ttft"] <= row["e2e"],
                "replayed relative schedule or request timing differs")
        errors.append(abs(row["relative_write_error_ms"]))
    delay = percentile(errors, .95)
    require(isclose(delay, summary["relative_write_error_abs_p95_ms"], abs_tol=1e-12)
            and max(errors) == summary["relative_write_error_abs_max_ms"], "write timing receipt differs")
    metrics = {metric: summary[metric] for metric in METRICS}
    require(all(isfinite(number) and number > 0 for number in metrics.values()), "invalid replay metric")
    require(isclose(metrics["ttft_p95_ms"], percentile([1000 * row["ttft"] for row in rows], .95), abs_tol=1e-12),
            "TTFT p95 differs from the request timings")
    long_rows = [row for row in rows if row["prompt_tokens"] >= 2500]
    require(len(long_rows) == 129, "long-prompt cohort differs")
    metrics.update(
        pre_write_p95_ms=percentile([1000 * row["pre_write"] for row in rows], .95),
        write_to_first_text_p95_ms=percentile([1000 * (row["ttft"] - row["pre_write"]) for row in rows], .95),
        long_prompt_ttft_p95_ms=percentile([1000 * row["ttft"] for row in long_rows], .95),
    )
    return rows, {**metrics, "write_error_abs_p95_ms": delay, "write_error_abs_max_ms": max(errors),
                  "summary_sha256": digest(folder / f"{origin}.summary.json"),
                  "records_sha256": digest(folder / f"{origin}.records.json")}



def checked_schedule(reference, origin):
    require(digest(reference / "verified.json") == "80b3e7545c58a13293924b2d9bf4a707d45231bab90b60ff513e0ffd65b358bf",
            "closed-loop source receipt differs")
    receipt = read(reference / "verified.json")
    cell = receipt["cells"]["rag-c8-seed2"]
    require(receipt["all_tested_streams_match"] is True and cell["aligned_samples_per_arm"] == MEASURED,
            "closed-loop source did not verify")
    source = reference / origin / "gpu-evidence/rag-c8-seed2.records.json"
    schedule, checksum = read_schedule(source, bodies("m"))
    require(checksum == cell["request_timing"][origin]["records_sha256"], "closed-loop source capture differs")
    return source, schedule, checksum


def verify(package, evidence, stage, reference):
    require(read(stage / "gpu-checker-input.json") == binding(package, evidence), "image evidence changed")
    marks = [line.split() for line in (stage / "campaign.log").read_text().splitlines()
             if line.startswith(("START ", "END "))]
    require(len(marks) == 2 * len(ORDER), "incomplete replay chronology")
    last = ""
    for name, start, end in zip(ORDER, marks[::2], marks[1::2], strict=True):
        require(start[:2] == ["START", name] and end[:2] == ["END", name]
                and len(start) == 3 and len(end) == 4 and end[3] == "exit=0" and last <= start[2] < end[2],
                f"{name}: incomplete or overlapping replay")
        last = end[2]
    arms = {name: checked_scope(package, evidence, stage, name) for name in ORDER}
    require(all(arm["reservations_after"] == arms[ORDER[0]]["reservations_after"] for arm in arms.values()),
            "original holders changed between arms")
    require(len({tuple(arm["server_arguments"]) for arm in arms.values()}) == 1, "arm arguments differ")
    cells = {}
    for origin in ORIGINS:
        source, schedule, checksum = checked_schedule(reference, origin)
        expected = output_keys(read(source))
        runs = {name: checked_replay(stage, name, origin, schedule, checksum) for name in ORDER}
        pair_metrics = {}
        for order in ("first", "second"):
            (left_rows, left), (right_rows, right) = (runs[f"{kind}-{order}"] for kind in ("parent", "packaged"))
            p95 = (right["ttft_p95_ms"] / left["ttft_p95_ms"] - 1) * 100
            write_difference = [abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                                for a, b in zip(left_rows, right_rows, strict=True)]
            pair_metrics[order] = {"ttft_p95_change_percent": p95,
                                   "write_to_first_text_p95_difference_ms":
                                       right["write_to_first_text_p95_ms"] - left["write_to_first_text_p95_ms"],
                                   "long_prompt_ttft_p95_difference_ms":
                                       right["long_prompt_ttft_p95_ms"] - left["long_prompt_ttft_p95_ms"],
                                   "paired_relative_write_difference_abs_p95_ms": percentile(write_difference, .95)}
        means = {kind: fmean(runs[f"{kind}-{order}"][1]["ttft_p95_ms"] for order in ("first", "second"))
                 for kind in ("parent", "packaged")}
        cells[origin] = {"source_sha256": checksum, "requests_per_arm": MEASURED,
                         "scheduled_span_seconds": max(schedule.values()),
                         "streams_match_original": all(output_keys(rows) == expected for rows, _ in runs.values()),
                         "all_p95_relative_write_errors_at_most_5ms": all(metrics["write_error_abs_p95_ms"] <= 5
                                                                         for _, metrics in runs.values()),
                         "metrics": {name: values[1] for name, values in runs.items()},
                         "paired_deltas": pair_metrics,
                         "ttft_p95_change_percent_by_arm_means": (means["packaged"] / means["parent"] - 1) * 100}
    require(all(cell["streams_match_original"] for cell in cells.values()), "replay output changed")
    return {"reference_receipt_sha256": digest(reference / "verified.json"), "remote_stage": str(stage),
            "scope": "Qwen2.5-7B-Instruct A100 6 RAG C8 seed 2, two observed write schedules; diagnostic comparison.",
            "order": ORDER, "arms": arms, "schedules": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("package", "evidence", "stage", "reference"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.evidence, args.stage, args.reference), indent=2, allow_nan=False))
