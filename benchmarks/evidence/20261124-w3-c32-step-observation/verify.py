"""Check a C32 replay and decompose its admitted requests' worker queue latency."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile
from tempfile import TemporaryDirectory

from stages import c32_execution_spans, reconcile


HISTORY = "20261124-w3-c32-rag-history"
OBSERVATION = "20261124-w3-aged-c32-observation"
SHORT = "c32-riley-arrivals"
MODES = ("control", "reserve")
FIELDS = (
    "admission_ms", "before_enqueue_ms", "command_queue_ms", "worker_step_remaining_ms",
    "later_worker_steps_ms", "cuda_span_overlap_ms", "queue_outside_steps_ms",
    "dispatch_to_ack_ms", "admission_remainder_ms",
)
MANIFESTS = ("dispatch.sha256", "build-result.sha256", "runtime.sha256", "result.sha256")
EXTRA = {
    "build.exit", "build.started", "build.pid", "build.log", "campaign.exit", "campaign.started",
    "campaign.pid", "campaign.log", "gpu6-staged-start.json", "gpu7-staged-start.json",
}
CHECKSUM_LINE = re.compile(r"([a-f0-9]{64})  ((?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*)")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def check_archive(archive, root, reg):
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = {member.name for member in members}
        if len(names) != len(members) or not all(member.isfile() for member in members):
            raise ValueError("duplicate or nonregular archive member")
        bundle.extractall(root, filter="data")
    if digest(root / "dispatch.sha256") != reg["dispatch_sha256"]:
        raise ValueError("remote dispatch does not match launch")
    expected = EXTRA | set(MANIFESTS)
    binary = None
    for manifest in MANIFESTS:
        declared = set()
        for line in (root / manifest).read_text().splitlines():
            match = CHECKSUM_LINE.fullmatch(line)
            if not match:
                raise ValueError(f"invalid archive entry in {manifest}")
            signature, name = match.groups()
            if name in declared:
                raise ValueError(f"duplicate entry in {manifest}")
            declared.add(name)
            if (manifest, name) == ("build-result.sha256", "riley-diagnostic"):
                binary = signature
            else:
                if digest(root / name) != signature:
                    raise ValueError(f"checksum mismatch: {name}")
                expected.add(name)
    if names != expected or not binary or binary == reg["previous_diagnostic_binary"]:
        raise ValueError("diagnostic binary or archive membership differs")
    for phase in ("build", "campaign"):
        if ((root / f"{phase}.exit").read_text().strip() != "0"
                or int((root / f"{phase}.pid").read_text()) != reg[f"{phase}_pid"]
                or (root / f"{phase}.started").read_text().strip() != reg[f"{phase}_started_at"]):
            raise ValueError(f"wrong process identity or exit in {phase}")
    return binary


def compare(test, arms, baseline, reg):
    results = {}
    before, after = (arms[mode][0] for mode in MODES)
    for label in before:
        row = baseline["compare_cells"](before[label], after[label], label, reg)
        row["reserve"] = row.pop("fifo")
        row["abs_write_error_p95_ms"]["reserve"] = row["abs_write_error_p95_ms"].pop("fifo")
        results[label] = row
    first, second = (arms[mode][1] for mode in MODES)
    exact = test["phase_effect"](first, second, baseline["percentile"])
    parsed_before, parsed_after = (arms[mode][3] for mode in MODES)

    def stats(a, b):
        pairs = list(zip(a, b, strict=True))
        if not all(left["request_index"] == right["request_index"] for left, right in pairs):
            raise ValueError("admission trace changed request order")
        rank = baseline["percentile"]
        return {
            "requests": len(pairs),
            "paired_delta_p50_ms": {key: rank([y[key] - x[key] for x, y in pairs], .5) for key in FIELDS},
            "control_p50_ms": {key: rank([x[key] for x, _ in pairs], .5) for key in FIELDS},
            "reserve_p50_ms": {key: rank([y[key] for _, y in pairs], .5) for key in FIELDS},
            "control_shape_count": sum(x["incumbent_shape"] is not None for x, _ in pairs),
            "reserve_shape_count": sum(y["incumbent_shape"] is not None for _, y in pairs),
        }

    chosen = exact["reserve_p50_request"]["request_index"]
    (a,) = [row for row in parsed_before if row["request_index"] == chosen]
    (b,) = [row for row in parsed_after if row["request_index"] == chosen]
    rank = baseline["percentile"]
    same_shape = [(x, y) for x, y in zip(parsed_before, parsed_after, strict=True)
                  if x["incumbent_shape"] is not None and x["incumbent_shape"] == y["incumbent_shape"]]
    matched = {"requests": len(same_shape)}
    if same_shape:
        matched["paired_delta_p50_ms"] = {
            field: rank([y[field] - x[field] for x, y in same_shape], .5)
            for field in ("incumbent_cuda_start_offset_ms", "incumbent_cuda_duration_ms")
        }
    sequences = {}
    execution = {}
    for mode in MODES:
        spans, measured_start = arms[mode][4]
        shapes = [(rows, decode) for _, rows, decode in spans]
        sequences[mode] = shapes
        execution[mode] = {
            "shape_sequence_sha256": sha256(json.dumps(shapes).encode()).hexdigest(),
            "steps_with_warmup": len(spans),
            "steps_started_after_first_measured_submission": sum(span.start >= measured_start for span, _, _ in spans),
            "max_rows": max(rows for rows, _ in shapes),
            "prefill_or_mixed_steps": sum(rows != decode for rows, decode in shapes),
        }
    return {
        "cells": results, "arm_receipts": {mode: arms[mode][2] for mode in MODES},
        "c32_server_rank": exact, "c32_p50_request_engine_delta_ms": {key: b[key] - a[key] for key in FIELDS},
        "submission": stats(parsed_before, parsed_after),
        "first_32": stats(parsed_before[:32], parsed_after[:32]),
        "later_160": stats(parsed_before[32:], parsed_after[32:]),
        "same_incumbent_shape": matched,
        "c32_execution": {"same_shape_sequence": sequences["control"] == sequences["reserve"], "arms": execution},
    }


def verify(archive, registration, evidence, root):
    reg = read(registration)
    recent = evidence / HISTORY
    observed = evidence / OBSERVATION
    history_reg = read(recent / "registration.json")
    if (reg["recent_stage"] != history_reg["stage"]
            or reg["recent_archive_sha256"] != digest(recent / "raw-results.tar.gz")
            or reg["previous_diagnostic_binary"] != history_reg["diagnostic_binary_sha256"]
            or any(reg[name] != history_reg[name] for name in (
                "source_revision", "package_image_id", "package_cpu_report_sha256", "mode_flag", "schedules_sha256",
                "holdout_requests_per_arm", "write_error_p95_limit_ms", "decode_tpot_p50_max_regression_pct_exclusive",
                "short_ttft_p50_max_regression_pct_exclusive", "short_throughput_max_drop_pct_exclusive",
                "short_start_target_s", "rag_start_target_s", "history_release_tolerance_s",
            ))):
        raise ValueError("parent experiment, serving settings or gates changed")
    binary = check_archive(archive, root, reg)
    if (digest(root / "engine-admission.patch") != reg["patch_sha256"]
            or (root / "parent-sources.sha256").read_bytes() != (observed / "patched-source.sha256").read_bytes()):
        raise ValueError("parent engine sources or trace patch differ")
    history = runpy.run_path(str(recent / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    native = runpy.run_path(str(evidence / history["NATIVE"] / "verify.py"))
    observe = runpy.run_path(str(observed / "verify.py"))
    events = re.findall(r"^(START|END) gpu([67]) rag (reserve|control) \S+(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    with tarfile.open(recent / "raw-results.tar.gz", "r:gz") as gold:
        for name in ("arm.py", "client.py", "schedules.json", "cpu-placement.json", "replay.py", "holdout.py"):
            if (root / name).read_bytes() != gold.extractfile(name).read():
                raise ValueError(f"planned request replay changed at {name}")
        reports = {}
        for gpu, order in (("6", MODES[::-1]), ("7", MODES)):
            expected = [(kind, mode, status) for mode in order for kind, status in (("START", ""), ("END", "0"))]
            if [(kind, mode, code) for kind, index, mode, code in events if index == gpu] != expected:
                raise ValueError(f"run order or terminal status differs on GPU {gpu}")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for path in (root / f"gpu{gpu}-staged-start.json", *(root / f"gpu{gpu}/{s}.json" for s in ("before", "between", "after"))):
                row = read(path)
                if (row["phase"] != "idle" or row["other_processes"] or
                        row["allocation_bytes"] != row["torch_reserved_bytes"] or row["allocation_bytes"] != 20 * 2**30 or
                        any(row[k] != old_holder[k] for k in ("physical_index", "gpu", "pid", "start_ticks"))):
                    raise ValueError(f"reservation did not recover on GPU {gpu}")
            def previous(name):
                return json.load(gold.extractfile(f"gpu{gpu}/rag/control/{name}.json"))
            arms = {}
            for mode in MODES:
                arm = root / f"gpu{gpu}/rag/{mode}"
                samples, phases, summary = history["evaluate_arm"](
                    arm, previous, reg, "rag", mode, int(gpu) + 38, binary, baseline, native, observe,
                )
                log = (arm / "server.log").read_text()
                stages = reconcile(log, phases)
                spans = c32_execution_spans(log, read(arm / f"{SHORT}.json"), reg["holdout_requests_per_arm"])
                arms[mode] = samples, phases, summary, stages, spans
            reports[gpu] = compare(observe, arms, baseline, reg)
    return {"binary_sha256": binary, "gpus": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    with TemporaryDirectory() as destination:
        report = verify(args.archive, args.registration, args.evidence_root, Path(destination))
        print(json.dumps(report, indent=2, allow_nan=False))
