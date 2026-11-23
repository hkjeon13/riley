"""Bind archived request IDs, committed GPU steps, and baseline/candidate prefill choices."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import runpy
from statistics import median
import tarfile
from typing import NamedTuple


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "20261124-w3-rag-prefill-order"
TIMELINE = HERE.parent / "20261124-w3-rag-prefill-steps"
UNTRACED = HERE.parent / "20261124-w3-wide-prefill-order"


class Choice(NamedTuple):
    step: int
    applied: int
    baseline: int
    baseline_prompt: int
    baseline_prefix: int
    candidate: int
    candidate_prompt: int
    candidate_prefix: int
    rows: int


def read_events(lines, unique):
    choices, decodes = {}, {}
    snapshots = []
    for line in lines:
        if line.startswith("W3-MIXED-CHOICE "):
            choice = Choice(*map(int, line.split()[1:]))
            unique(choices, choice.step, choice)
        elif line.startswith("W3-MIXED-CHOICES "):
            snapshots.append(tuple(map(int, line.split()[1:])))
        elif line.startswith("W3-MIXED-DECODE "):
            _, step, contexts = line.split()
            pairs = [tuple(map(int, pair.split(":"))) for pair in contexts.split(",")]
            unique(decodes, int(step), pairs)
    if snapshots != [(len(choices), 0)]:
        raise ValueError("incomplete mixed-prefill choices")
    return choices, decodes


def validate_choice(choice, step, cursor, requested, admitted, enabled):
    picked = choice.candidate if enabled else choice.baseline
    prefix = choice.candidate_prefix if enabled else choice.baseline_prefix
    if (choice.applied != int(enabled) or choice.baseline <= choice.candidate
            or any(admitted[req] > step.start for req in (choice.baseline, choice.candidate))
            or not 1 <= step.decode <= 8 or choice.rows < 1016 or step.rows != 1024
            or step.prefill != [(picked, choice.rows, prefix + choice.rows)]):
        raise ValueError(f"choice and committed iteration differ at {step.id}")
    for req, prompt, before in (
        (choice.baseline, choice.baseline_prompt, choice.baseline_prefix),
        (choice.candidate, choice.candidate_prompt, choice.candidate_prefix),
    ):
        if prompt < 3072 or before != cursor[req] or before + choice.rows > prompt:
            raise ValueError(f"invalid whole-budget prefill cursor at {step.id}")
        if req in requested and requested[req]["prompt_tokens"] != prompt:
            raise ValueError(f"prefill choice differs from the request length at {step.id}")
    if choice.candidate_prefix <= 0:
        raise ValueError(f"selected candidate had not started at {step.id}")
    return picked, prefix


def check_steps(steps, choices, decodes, admitted, requested, labels, enabled):
    """Check all committed contexts before computing statistics for mixed choices."""
    cursor = defaultdict(int)
    last = {}
    seen_decode = set()
    observed = []
    for step in steps:
        dec = decodes.pop(step.id, [])
        if (len(dec) != step.decode or len({req for req, _ in dec}) != step.decode
                or {req for req, _ in dec} & {req for req, _, _ in step.prefill}):
            raise ValueError(f"decode context coverage differs at {step.id}")
        gaps = []
        for req, target in dec:
            if (req not in last or target != cursor[req] + 1 or admitted[req] > step.start
                    or (req in requested and cursor[req] < requested[req]["prompt_tokens"])):
                raise ValueError(f"discontinuous decode cursor at {step.id} for request {req}")
            gaps.append(step.commit - last[req])
        choice = choices.pop(step.id, None)
        if choice is not None:
            picked, prefix = validate_choice(choice, step, cursor, requested, admitted, enabled)
            measured = requested.get(picked)
            observed.append({
                "step": step.id, "label": labels.get(picked),
                "selected_request": measured["request_index"] if measured else None,
                "baseline_request": requested.get(choice.baseline, {}).get("request_index"),
                "candidate_request": requested.get(choice.candidate, {}).get("request_index"),
                "selected_prefix_tokens": prefix, "baseline_prefix_tokens": choice.baseline_prefix,
                "candidate_prefix_tokens": choice.candidate_prefix,
                "prefill_rows": choice.rows, "gpu_ms": step.gpu / 1e6,
                "logical_pair_delta": choice.rows * (choice.candidate_prefix - choice.baseline_prefix),
                "decode_kv_tokens_total": sum(length for _, length in dec),
                "decode": [
                    {"scheduler_request": req, "kv_tokens": length,
                     "request_index": requested[req]["request_index"] if req in requested else None,
                     "commit_gap_ms": gap / 1e6}
                    for (req, length), gap in zip(dec, gaps, strict=True)
                ],
            })
        for req, rows, target in step.prefill:
            if req in seen_decode or target != cursor[req] + rows:
                raise ValueError(f"discontinuous prefill cursor at {step.id} for request {req}")
            if req in requested and target > requested[req]["prompt_tokens"]:
                raise ValueError(f"prefill cursor beyond input at {step.id} for request {req}")
            cursor[req], last[req] = target, step.commit
        for req, target in dec:
            cursor[req], last[req] = target, step.commit
            seen_decode.add(req)
    if choices or decodes:
        raise ValueError("choice or decode record missing its committed step")
    for req, row in requested.items():
        if cursor[req] < row["prompt_tokens"] or req not in seen_decode and row["completion_tokens"] > 1:
            raise ValueError(f"truncated scheduler trace for measured request {req}")
    return observed


def verify_trace(arm, cells, holdout, enabled, timeline):
    read = timeline["read"]
    plan = read(arm / "plan.json")
    if [arg for arg in plan["docker"] if arg.startswith("RILEY_W3_RAG_STEPS=")] != ["RILEY_W3_RAG_STEPS=1"]:
        raise ValueError("diagnostic trace switch was not enabled")
    raw_requests, label_by_response, expected_ids = {}, {}, set()
    unique = timeline["put_unique"]
    for label in cells:
        client = read(arm / f"{label}.json")
        for request in client["warmup_response_ids"]:
            unique(label_by_response, request, label)
            expected_ids.add(request)
        for row in client["measured"]:
            response = row["response_id"]
            unique(raw_requests, response, row)
            unique(label_by_response, response, label)
            expected_ids.add(response)
    log = arm / "server.log"
    lines = log.read_text().splitlines()
    visible = set(re.findall(r"^W3-RAG-VISIBLE (cmpl-[0-9a-f]+) \d+ \d+$", "\n".join(lines), re.MULTILINE))
    if expected_ids - visible or len(visible - expected_ids) != holdout:
        raise ValueError("timeline does not cover the measured and warmup requests and token holdout")
    admits, _, steps, _, admitted, startup_requests, startup_steps = timeline["read_trace"](log, visible)
    labels = {admits[int(response[5:], 16)][0]: label for response, label in label_by_response.items()}
    requested = {admits[int(response[5:], 16)][0]: row for response, row in raw_requests.items()}
    choices, decodes = read_events(lines, unique)
    (swap_count,) = [int(line.split()[1]) for line in lines if line.startswith("W3-PREFILL-FIFO ")]
    if (swap_count != sum(choice.applied for choice in choices.values())
            or not choices
            or any(choice.applied != int(enabled) for choice in choices.values())):
        raise ValueError("trace disagrees with the measured reorder count")
    rows = check_steps(steps, choices, decodes, admitted, requested, labels, enabled)
    by_label = defaultdict(list)
    for row in rows:
        by_label[row["label"] or "outside_measured_schedules"].append(row)
    return {
        "committed_steps": len(steps), "startup_steps": startup_steps, "startup_requests": startup_requests,
        "choice_steps_by_schedule": {
            label: {"steps": len(group), "gpu_ms_median": median(row["gpu_ms"] for row in group),
                    "decode_commit_gap_ms_median": median(item["commit_gap_ms"] for row in group for item in row["decode"])}
            for label, group in by_label.items()
        },
        "choices": rows,
    }


def verify(root, registration):
    prior = runpy.run_path(str(PRIOR / "verify.py"))
    timeline = runpy.run_path(str(TIMELINE / "verify.py"))
    reg = prior["read"](registration)
    paired = prior["verify"](root, registration)
    with tarfile.open(UNTRACED / "raw-results.tar.gz", "r:gz") as reference:
        for gpu, row in paired["gpus"].items():
            row["trace"] = {}
            for mode in row["order"]:
                relative = f"gpu{gpu}/{mode}"
                arm = root / relative
                result = verify_trace(arm, row["cells"], reg["holdout_requests_per_arm"], mode == "fifo", timeline)
                result["untraced_comparison"] = {}
                if json.load(reference.extractfile(f"{relative}/token-id-holdout.json")) != prior["read"](
                        arm / "token-id-holdout.json"):
                    raise ValueError(f"holdout differs from the untraced run at {relative}")
                for label in row["cells"]:
                    recorded = json.load(reference.extractfile(f"{relative}/{label}.json"))
                    current = prior["read"](arm / f"{label}.json")
                    comparisons = [(data["measured"], {key: data["statistics"][key] for key in prior["METRICS"]})
                                   for data in (recorded, current)]
                    matched = prior["compare_cells"](*comparisons, label, reg)
                    if not matched["gates"]["output"]:
                        raise ValueError(f"measured output differs from the untraced run at {relative}/{label}")
                    result["untraced_comparison"][label] = {
                        "write_error_p95_ms": matched["abs_write_error_p95_ms"],
                        "timing_valid": matched["gates"]["timing"], "delta_percent": matched["delta_percent"],
                    }
                row["trace"][mode] = result
    return paired


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("registration", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage, args.registration), indent=2, allow_nan=False))
