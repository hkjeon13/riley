"""Identify which verified CUDA iteration produces each selected C32 response."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import sys
import tarfile
from tempfile import TemporaryDirectory


STEP = "20261124-w3-c32-step-observation"
NATIVE_REPEAT = "20261124-w3-c32-native-repeat"
HISTORY = "20261124-w3-c32-rag-history"
RECEIPT = ("physical_index", "gpu", "pid", "start_ticks")
TARGETS = {"6": [210, 211], "7": [207, 219]}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def parse_trace(log, stages):
    original = []
    prefill = defaultdict(dict)
    snapshot = None
    for line in log.splitlines():
        if line.startswith("W3-PREFILL "):
            start, request, end_position, rows, output = map(int, line.split()[1:])
            group = prefill[start]
            if request in group or not 0 < rows <= end_position or output not in (0, 1):
                raise ValueError("invalid or duplicate prefill request in an iteration")
            group[request] = (end_position, rows, output)
        elif line.startswith("W3-ADMISSION-TRACE "):
            if snapshot is not None:
                raise ValueError("duplicate trace snapshot")
            snapshot = tuple(map(int, line.split()[1:]))
        else:
            original.append(line)
    count = sum(len(group) for group in prefill.values())
    if snapshot is None or not count or snapshot[0] <= count or snapshot[1] != 0:
        raise ValueError("missing or incomplete bounded trace")
    original.append(f"W3-ADMISSION-TRACE {snapshot[0] - count} 0")
    normalized = "\n".join(original)
    admissions, _, _, cuda = stages["parse"](normalized)
    executions = {span.start: (span, rows, decode) for span, rows, decode in cuda.values()}
    if any(start not in executions for start in prefill):
        raise ValueError("prefill record has no matching execution")
    for start, (_, rows, decode) in executions.items():
        if sum(item[1] for item in prefill.get(start, {}).values()) + decode != rows:
            raise ValueError("prefill and decode rows do not sum to iteration shape")
    return normalized, admissions, executions, prefill


def select_request(row, observed, admissions, executions, prefill, admission_trace):
    response = int(row["response_id"].removeprefix("cmpl-"), 16)
    _, _, _, ack = admissions[response]
    seen = []
    last_position = 0
    for start, (span, total_rows, decode_rows) in executions.items():
        work = prefill.get(start, {}).get(response)
        if work is None:
            continue
        position, tokens, output = work
        if start < ack or position - tokens != last_position:
            raise ValueError("prefill starts before admission or skips input rows")
        last_position = position
        seen.append({
            "position": position, "rows": tokens, "produces_output": bool(output),
            "start_after_ack_ms": (start - ack) / 1e6,
            "end_after_ack_ms": (span.end - ack) / 1e6,
            "iteration_rows": total_rows, "decode_rows": decode_rows,
        })
    if not seen or last_position != row["prompt_tokens"] or not seen[-1]["produces_output"] or any(
            stage["produces_output"] for stage in seen[:-1]):
        raise ValueError("request lacks a complete prefill and terminal prefill output")
    before_first_prefill = [
        {"start_after_ack_ms": (span.start - ack) / 1e6, "end_after_ack_ms": (span.end - ack) / 1e6,
         "rows": rows, "decode": decode}
        for span, rows, decode in executions.values()
        if ack <= span.start and (span.end - ack) / 1e6 <= seen[0]["start_after_ack_ms"]
    ]
    return {
        "request_index": row["request_index"], "input_tokens": row["prompt_tokens"],
        "ttft_ms": observed["ttft_ms"], "backend_admission_ms": observed["backend_admission_ms"],
        "command_queue_ms": admission_trace["command_queue_ms"],
        "admitted_to_first_content_ms": observed["admitted_to_first_visible_ms"],
        "iterations_before_first_prefill": before_first_prefill, "prefill": seen,
    }


def native_tail(archive, result, gpu, label, reference_rows):
    def rank(rows):
        return sorted(rows, key=lambda row: row["ttft"])[int(len(rows) * .95)]
    identity = ("request_index", "scheduled_write_ns", "input_sha256", "output_sha256")
    summary = {}
    for name, pair in result["gpus"][gpu]["comparisons"].items():
        runs = []
        for which in ("first", "second"):
            filename = f"gpu{gpu}/rag/{pair[f'{which}_pass']}/{pair[f'{which}_mode']}/{label}.json"
            rows = json.load(archive.extractfile(filename))["measured"]
            if any(any(a[key] != b[key] for key in identity)
                   for a, b in zip(reference_rows, rows, strict=True)):
                raise ValueError("native replay and traced request sequence differs")
            runs.append(rows)
        a, b = map(rank, runs)
        same = next(row for row in runs[0] if row["request_index"] == b["request_index"])
        write_a = same["sent_at"] - same["start"]
        write_b = b["sent_at"] - b["start"]
        summary[name] = {
            "first_tail_request": a["request_index"], "second_tail_request": b["request_index"],
            "ttft_p95_delta_ms": (b["ttft"] - a["ttft"]) * 1000,
            "rank_shift_ms": (same["ttft"] - a["ttft"]) * 1000,
            "matched_before_write_delta_ms": (write_b - write_a) * 1000,
            "matched_after_write_delta_ms": (b["ttft"] - write_b - same["ttft"] + write_a) * 1000,
        }
    return summary


def verify(archive, registration, evidence, scratch):
    reg = read(registration)
    step = evidence / STEP
    history_path = evidence / HISTORY
    old_reg = read(step / "registration.json")
    if (reg["target_request_indexes"] != TARGETS
            or reg["prior_step_archive_sha256"] != digest(step / "raw-results.tar.gz")
            or reg["native_repeat_archive_sha256"] != digest(evidence / NATIVE_REPEAT / "raw-results.tar.gz")
            or reg["recent_archive_sha256"] != digest(history_path / "raw-results.tar.gz")
            or reg["previous_diagnostic_binary"] != (step / "build-result.sha256").read_text().split()[0]
            or any(reg[k] != old_reg[k] for k in (
                "source_revision", "package_image_id", "package_cpu_report_sha256", "mode_flag",
                "schedules_sha256", "holdout_requests_per_arm", "write_error_p95_limit_ms",
                "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
                "short_throughput_max_drop_pct_exclusive", "short_start_target_s", "rag_start_target_s",
                "history_release_tolerance_s",
            ))):
        raise ValueError("experiment inputs or frozen request identity changed")
    sys.path.insert(0, str(step))
    step_verify = runpy.run_path(str(step / "verify.py"))
    stages = runpy.run_path(str(step / "stages.py"))
    history = runpy.run_path(str(history_path / "verify.py"))
    baseline = runpy.run_path(str(evidence / history["BASELINE"] / "verify.py"))
    native = runpy.run_path(str(evidence / history["NATIVE"] / "verify.py"))
    observe = runpy.run_path(str(evidence / step_verify["OBSERVATION"] / "verify.py"))
    repeats = runpy.run_path(str(evidence / NATIVE_REPEAT / "verify.py"))
    binary = step_verify["check_archive"](archive, scratch, reg)
    if digest(scratch / "prefill-tail.patch") != reg["patch_sha256"]:
        raise ValueError("transmitted patch differs from registration")
    prior_source = (step / "diagnostic-sources.sha256").read_text().splitlines()
    patched_receipt = (scratch / "parent-tail-sources.sha256").read_text().splitlines()
    if len(patched_receipt) != 2 or any(line not in prior_source for line in patched_receipt):
        raise ValueError("patched source does not match preceding trace")
    events = re.findall(
        r"^(START|END) gpu([67]) rag (reserve|control) \S+(?: exit=(\d+))?$",
        (scratch / "campaign.log").read_text(), re.MULTILINE,
    )
    results = {}
    native_result = read(evidence / NATIVE_REPEAT / "verified.json")
    with (tarfile.open(history_path / "raw-results.tar.gz") as gold,
          tarfile.open(evidence / NATIVE_REPEAT / "raw-results.tar.gz") as repeat):
        for filename in ("arm.py", "client.py", "holdout.py", "replay.py", "schedules.json", "cpu-placement.json"):
            if (scratch / filename).read_bytes() != gold.extractfile(filename).read():
                raise ValueError(f"frozen runner differs at {filename}")
        for gpu, order in (("6", ("reserve", "control")), ("7", ("control", "reserve"))):
            if [(k, mode, code) for k, index, mode, code in events if index == gpu] != [
                (k, mode, code) for mode in order for k, code in (("START", ""), ("END", "0"))
            ]:
                raise ValueError(f"invalid execution order or exit on GPU {gpu}")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for path in [scratch / f"gpu{gpu}-staged-start.json", *(
                scratch / f"gpu{gpu}/{phase}.json" for phase in ("before", "between", "after")
            )]:
                holder = read(path)
                if (holder["phase"] != "idle" or holder["other_processes"]
                        or holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30
                        or any(holder[k] != old_holder[k] for k in RECEIPT)):
                    raise ValueError(f"original reservation did not recover on GPU {gpu}")
            def previous(name):
                return json.load(gold.extractfile(f"gpu{gpu}/rag/control/{name}.json"))
            reports = {}
            for mode in order:
                path = scratch / f"gpu{gpu}/rag/{mode}"
                workload, phases, profile = history["evaluate_arm"](
                    path, previous, reg, "rag", mode, int(gpu) + 38, binary, baseline, native, observe,
                )
                repeats["verify_metrics"](path, workload, baseline["percentile"])
                log = (path / "server.log").read_text()
                normalized, admissions, executions, prefill = parse_trace(log, stages)
                admissions_by_request = {r["request_index"]: r for r in stages["reconcile"](normalized, phases)}
                phase_by_request = {r["request_index"]: r for r in phases}
                short = workload[history["SHORT"]][0]
                by_request = {r["request_index"]: r for r in short}
                ranked = sorted(short, key=lambda row: row["ttft"])
                tail_rank = int(len(ranked) * .95)
                tail = ranked[tail_rank]["request_index"]
                tail_requests = [row["request_index"] for row in ranked[tail_rank:]]
                requests = {
                    str(idx): select_request(by_request[idx], phase_by_request[idx], admissions, executions,
                                             prefill, admissions_by_request[idx])
                    for idx in sorted(set([*TARGETS[gpu], *tail_requests]))
                }
                reports[mode] = {
                    "tail_request": tail, "tail_requests": tail_requests,
                    "requests": requests, "profile": profile,
                }
            results[gpu] = {
                "traced_runs": reports,
                "native_tail": native_tail(
                    repeat, native_result, gpu, history["SHORT"], previous(history["SHORT"])["measured"],
                ),
            }
    return {"binary_sha256": binary, "gpus": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    with TemporaryDirectory() as scratch:
        print(json.dumps(verify(args.archive, args.registration, args.evidence_root, Path(scratch)),
                         indent=2, allow_nan=False))
