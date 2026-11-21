"""Join decoded HTTP response IDs to server timestamps and verify the paired replay."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

from comparison_base import compare, p, short_distribution
from private_reference import check_case
from replay_verifier import check_replay
from verify_a100_candidate import digest, read, require
from warm_reference import checked_reservation
from workload import make_workloads

ORDER = ("off-warm-control", "private-control", "off-warm-trace", "private-trace")
ORDERS = {"6": ORDER[::-1], "7": ORDER}
PHASES = ("http_parsed", "submit", "send", "worker", "admitted", "visible")
RESPONSE_ID = re.compile(r"cmpl-([a-f0-9]{16})")


def trace_data(log):
    phases, prefill = defaultdict(dict), defaultdict(list)
    dropped = []
    for line in log.splitlines():
        if line.startswith("W3-REQUEST-PHASE "):
            _, request, phase, ns = line.split()
            record = phases[int(request)]
            require(phase in PHASES and phase not in record, "unrecognized or repeated phase")
            record[phase] = int(ns)
        elif line.startswith("W3-PREFILL-PHASE "):
            _, *fields = line.split()
            request, *step = map(int, fields)
            require(len(step) == 7, "bad prefill record")
            prefill[request].append(step)
        elif line.startswith("W3-DROPPED "):
            dropped.append(int(line.split()[1]))
    return phases, prefill, dropped


def join_short(rows, phases, records):
    iteration_spans = {}
    for steps in records.values():
        for iteration, start, end, *_ in steps:
            span = (start, end)
            require(iteration_spans.setdefault(iteration, span) == span, "shared prefill interval differs")
    spans = sorted(iteration_spans.values())
    require(all(end <= next_start for (_, end), (next_start, _) in zip(spans, spans[1:])),
            "prefill intervals overlap")
    joined, ids = [], set()
    for row in rows:
        match = RESPONSE_ID.fullmatch(row["response_id"])
        require(match is not None, "response ID is not a native engine ID")
        request = int(match[1], 16)
        require(request not in ids and set(phases[request]) == set(PHASES) and records[request],
                f"incomplete trace for request {request}")
        ids.add(request)
        parsed, submit, send, worker, admitted, visible = (phases[request][name] for name in PHASES)
        write = round(row["sent_at"] * 1e9)
        content = round((row["start"] + row["ttft"]) * 1e9)
        steps = records[request]
        first_start = steps[0][1]
        last_end = steps[-1][2]
        require(all(a <= b for a, b in zip(
            (write, parsed, submit, send, worker, admitted, last_end, visible),
            (parsed, submit, send, worker, admitted, first_start, visible, content), strict=True)),
            f"client and server clocks or phase order disagree for request {request}")
        elapsed = cuda = tokens = 0
        for j, (iteration, start, end, gpu_ns, rows_in_step, decode, input_tokens) in enumerate(steps):
            require((j == 0 or (steps[j - 1][0] < iteration and steps[j - 1][2] <= start))
                    and 0 < gpu_ns <= end - start and rows_in_step >= input_tokens + decode > 0,
                    f"invalid prefill interval for request {request}")
            elapsed += end - start
            cuda += gpu_ns
            tokens += input_tokens
        require(tokens == row["prompt_tokens"] and row["cached_tokens"] == 0, "short prompt accounting differs")
        queued_during_prefill = sum(max(0, min(worker, end) - max(send, start)) for start, end in spans)
        regions = {
            "write_to_http_parsed": parsed - write,
            "http_parsed_to_submit": submit - parsed,
            "submit_to_enqueue": send - submit,
            "enqueue_during_prefill_batch": queued_during_prefill,
            "enqueue_outside_prefill_batch": worker - send - queued_during_prefill,
            "worker_admission": admitted - worker,
            "admitted_to_first_prefill": first_start - admitted,
            "prefill_batch_cuda": cuda,
            "prefill_batch_host_remainder": elapsed - cuda,
            "between_prefill_batches": last_end - first_start - elapsed,
            "last_prefill_to_visible": visible - last_end,
            "visible_to_client_content": content - visible,
        }
        require(sum(regions.values()) == content - write, "timing partition does not reconcile")
        joined.append({
            "request_id": row["response_id"], "ttft_ms": row["ttft"] * 1000,
            "phases_ms": {**{key: ns / 1e6 for key, ns in regions.items()},
                          "enqueue_to_worker": (worker - send) / 1e6, "write_to_content": (content - write) / 1e6},
            "prefill_steps": len(steps), "first_prefill_rows": steps[0][4], "first_prefill_decode_rows": steps[0][5],
        })
    return joined


def paired_timing(before, after, indices):
    keys = before[0]["phases_ms"].keys()
    phases = {}
    for key in keys:
        baseline = [before[i]["phases_ms"][key] for i in indices]
        candidate = [after[i]["phases_ms"][key] for i in indices]
        change = [new - old for old, new in zip(baseline, candidate, strict=True)]
        phases[key] = {
            "off_warm_p50_ms": p(baseline, .5), "private_p50_ms": p(candidate, .5),
            "paired_p50_change_ms": p(change, .5), "paired_p95_change_ms": p(change, .95),
            "paired_mean_change_ms": sum(change) / len(change),
            "slower_requests": sum(value > 0 for value in change),
        }
    return {
        "requests": len(indices), "phases": phases,
        "prefill_steps": {case: dict(sorted(Counter(rows[i]["prefill_steps"] for i in indices).items()))
                          for case, rows in (("off_warm", before), ("private", after))},
        "first_prefill_rows": {case: dict(sorted(Counter(rows[i]["first_prefill_rows"] for i in indices).items()))
                               for case, rows in (("off_warm", before), ("private", after))},
        "mixed_first_prefill": {case: sum(rows[i]["first_prefill_decode_rows"] > 0 for i in indices)
                                for case, rows in (("off_warm", before), ("private", after))},
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
    data = read(original)
    plans = read(stage / "schedules.json")
    workloads = make_workloads()
    timeline = [row.split() for row in (stage / "campaign.log").read_text().splitlines()
                if row.startswith(("START ", "END "))]
    require(len(timeline) == 16 and all(row[-1] == "exit=0" for row in timeline if row[0] == "END"),
            "campaign sequence incomplete")
    runs, reservations, timing, comparisons, replay_errors = {}, {}, {}, {}, {}
    for index, labels in ORDERS.items():
        events = [row for row in timeline if row[1].endswith(f"-gpu{index}")]
        require([row[:2] for row in events] == [[phase, f"{label}-gpu{index}"]
                                               for label in labels for phase in ("START", "END")],
                "GPU-local sequence differs")
        owner = checked_reservation(stage / f"gpu{index}-before.json", index)
        require(checked_reservation(stage / f"gpu{index}-after.json", index) == owner, "GPU owner changed")
        reservations.update(owner)
        raw, arms, joined = {}, {}, {}
        for label in labels:
            condition, variant = label.rsplit("-", 1)
            tag = f"{label}-gpu{index}"
            folder = stage / tag
            receipt = read(folder / "receipt.json")
            require(receipt["stage_trace"] is (variant == "trace"), f"unexpected trace switch for {tag}")
            raw[label], arms[label] = check_case(stage, condition, index, owner, stock, parent, build_checks,
                                                  data, workloads, label=tag)
            phases, prefill, dropped = trace_data((folder / "server.log").read_text())
            if variant == "trace":
                require(dropped == [0], f"trace ended with missing records for {tag}")
                joined[condition] = join_short(raw[label]["short"]["rows"], phases, prefill)
            else:
                require(not phases and not prefill and not dropped, f"control emitted trace for {tag}")
            arms[label]["trace"] = {"phase_records": sum(map(len, phases.values())),
                                   "prefill_records": sum(map(len, prefill.values())),
                                   "dropped": dropped[0] if dropped else None}
            arms[label]["fidelity"] = check_replay(raw[label]["short"], plans[index])
            runs[tag] = arms[label]
        indices = arms["off-warm-trace"]["fidelity"]["first_32_indices"]
        require(sorted(indices) == list(range(32)), "first C32 burst no longer matches the closed client cohort")
        rest = sorted(set(range(96)) - set(indices))
        timing[index] = {name: paired_timing(joined["off-warm"], joined["private"], cohort)
                         for name, cohort in (("all", range(96)), ("first_32", indices), ("remaining_64", rest))}
        pairs = {
            "private_vs_off_warm_control": ("off-warm-control", "private-control"),
            "private_vs_off_warm_trace": ("off-warm-trace", "private-trace"),
            "off_warm_trace_vs_control": ("off-warm-control", "off-warm-trace"),
            "private_trace_vs_control": ("private-control", "private-trace"),
        }
        comparisons[index], replay_errors[index] = {}, {}
        for name, (before, after) in pairs.items():
            comparisons[index][name] = compare(raw[before], raw[after], arms[before], arms[after])
            cohorts = short_distribution(raw[before], raw[after])
            for label, which in ((before, "off"), (after, "on")):
                rows = raw[label]["short"]["rows"]
                cohorts["remaining_64"][f"{which}_p75_ms"] = p([1000 * rows[i]["ttft"] for i in rest], .75)
            comparisons[index][name]["short_cohorts"] = cohorts
            errors = [abs(left["relative_write_error_ms"] - right["relative_write_error_ms"])
                      for left, right in zip(raw[before]["short"]["rows"], raw[after]["short"]["rows"], strict=True)]
            replay_errors[index][name] = p(errors, .95)
            require(replay_errors[index][name] <= 5, "paired client timing diverged")
    return {
        "status": "passed", "remote_evidence": str(stage), "remote_build": str(build),
        "source_revision": stock["source_revision"], "image_id": stock["image_id"], "build": build_checks,
        "stock_workload_sha256": reference_hash, "schedules_sha256": digest(stage / "schedules.json"),
        "reservations": reservations, "order": timeline, "runs": runs,
        "comparisons": comparisons, "paired_write_error_p95_ms": replay_errors, "timing": timing,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(sys.argv[1])), indent=2, allow_nan=False))
