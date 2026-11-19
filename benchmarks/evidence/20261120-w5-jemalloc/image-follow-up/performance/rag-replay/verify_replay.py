"""Verify matched timing, output, and resources for the fixed RAG schedules."""

from argparse import ArgumentParser
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import fmean

from fixed_arrival import MEASURED, bodies, read_schedule
from load import percentile
from qualify_allocator import check_reservations
from qualify_packaged_gpu import PROFILE, binding
from verify_allocator_image import LIB_SHA
from verify_a100_candidate import CHECKPOINT, GPU, RILEY, ROOTS, digest, gpu_evidence, read, require


ORDER = ("parent-first", "packaged-first", "packaged-second", "parent-second")
ORIGINS = ("parent-first", "packaged-first")
METRICS = ("ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
FIELDS = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")


def output_keys(rows):
    return [tuple(row[key] for key in FIELDS) for row in rows]


def checked_scope(evidence, stage, name):
    folder = stage / name
    parent = name.startswith("parent-")
    selected = evidence / ROOTS["base"] / "cpu-evidence/report.json" if parent else stage / "gpu-checker-input.json"
    cpu = read(selected)
    fields = {key: cpu[key] for key in ("image_id", "source_revision", "source_archive_sha256")}
    fields.update(riley_binary_sha256=cpu["binaries"][RILEY], cpu_report_sha256=digest(selected),
                  checkpoint_sha256=CHECKPOINT, gpu_uuid=GPU)
    receipt = gpu_evidence(folder, fields, int((folder / "exit").read_text()))
    require(receipt["status"] == "passed" and receipt["api_checks"] == 40, "runtime API failed")
    process = read(folder / "process-after.json")
    require(process == read(folder / "process-before.json") and process["image_id"] == cpu["image_id"],
            "different process or image during replay")
    allocator = process["allocator"]
    require((allocator == "glibc" if parent else allocator["library_sha256"] == LIB_SHA
            and allocator["malloc_conf"] == PROFILE and allocator["mounts"] == ["/models"]),
            "runtime allocator differs")
    holders = check_reservations(folder, "after")
    require(holders == check_reservations(folder, "before"), "GPU reservation did not recover")
    marks = [line.split()[1] for line in (folder / "gpu-evidence/api.log").read_text().splitlines()
             if line.startswith("FIXED_ARRIVAL_COMPLETED ")]
    ordered = ORIGINS if name.endswith("-first") else tuple(reversed(ORIGINS))
    require(tuple(marks) == ordered, "fixed schedule execution order changed")
    return {"image_id": cpu["image_id"], "allocator": allocator, "server_arguments": process["server_arguments"],
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
                and isfinite(row["relative_write_error_ms"]), "replayed relative schedule differs")
        errors.append(abs(row["relative_write_error_ms"]))
    delay = percentile(errors, .95)
    require(isclose(delay, summary["relative_write_error_abs_p95_ms"], abs_tol=1e-12)
            and max(errors) == summary["relative_write_error_abs_max_ms"], "write timing receipt differs")
    metrics = {metric: summary[metric] for metric in METRICS}
    require(all(isfinite(number) and number > 0 for number in metrics.values()), "invalid replay metric")
    require(isclose(metrics["ttft_p95_ms"], percentile([1000 * row["ttft"] for row in rows], .95), abs_tol=1e-12),
            "TTFT p95 differs from the request timings")
    return rows, {**metrics, "write_error_abs_p95_ms": delay, "write_error_abs_max_ms": max(errors)}


def verify(package, evidence, stage, reference):
    require(read(stage / "gpu-checker-input.json") == binding(package, evidence), "image evidence changed")
    arms = {name: checked_scope(evidence, stage, name) for name in ORDER}
    require(len({tuple(arm["server_arguments"]) for arm in arms.values()}) == 1, "arm arguments differ")
    cells = {}
    for origin in ORIGINS:
        source = reference / origin / "gpu-evidence/rag-c8-seed2.records.json"
        schedule, checksum = read_schedule(source, bodies("m"))
        expected = output_keys(read(source))
        runs = {name: checked_replay(stage, name, origin, schedule, checksum) for name in ORDER}
        pair_metrics = {}
        for order in ("first", "second"):
            (left_rows, left), (right_rows, right) = (runs[f"{kind}-{order}"] for kind in ("parent", "packaged"))
            p95 = (right["ttft_p95_ms"] / left["ttft_p95_ms"] - 1) * 100
            write_difference = [abs(a["relative_write_error_ms"] - b["relative_write_error_ms"])
                                for a, b in zip(left_rows, right_rows, strict=True)]
            pair_metrics[order] = {"ttft_p95_change_percent": p95,
                                   "paired_relative_write_difference_abs_p95_ms": percentile(write_difference, .95)}
        means = {kind: fmean(runs[f"{kind}-{order}"][1]["ttft_p95_ms"] for order in ("first", "second"))
                 for kind in ("parent", "packaged")}
        cells[origin] = {"source_sha256": checksum, "requests_per_arm": MEASURED,
                         "streams_match_original": all(output_keys(rows) == expected for rows, _ in runs.values()),
                         "all_p95_relative_write_errors_at_most_5ms": all(metrics["write_error_abs_p95_ms"] <= 5
                                                                         for _, metrics in runs.values()),
                         "metrics": {name: values[1] for name, values in runs.items()},
                         "paired_deltas": pair_metrics,
                         "ttft_p95_change_percent_by_arm_means": (means["packaged"] / means["parent"] - 1) * 100}
    return {"scope": "Qwen2.5-7B-Instruct A100 6 RAG C8 seed 2, two observed write schedules; diagnostic comparison.",
            "order": ORDER, "arms": arms, "schedules": cells}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("package", "evidence", "stage", "reference"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.package, args.evidence, args.stage, args.reference), indent=2, allow_nan=False))
