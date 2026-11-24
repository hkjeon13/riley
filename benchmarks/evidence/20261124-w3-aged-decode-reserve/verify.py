"""Verify Coder-14B wide aging turns, outputs and performance against fixed workloads."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile


REFERENCE = "20261124-w3-prefill-step-cost"
RESERVATION = "20261124-w3-aged-decode-reserve"
MATCH_KEYS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")
RUN_ID = re.compile(r"^W3-RAG-VISIBLE (cmpl-[0-9a-f]+) \d+ \d+$", re.MULTILINE)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def aging_steps(steps, lines, enabled):
    snapshot, turns = [], {}
    for line in lines:
        if line.startswith("W3-AGED-TURNS "):
            snapshot.append(int(line.split()[1]))
        elif line.startswith("W3-AGED-TURN "):
            index, *turn = map(int, line.split()[1:])
            if index in turns or len(turn) != 5:
                raise ValueError(f"invalid or duplicate aging turn {index}")
            turns[index] = turn
    if snapshot != [len(turns)] or not turns:
        raise ValueError("missing or incomplete aging trace")
    cursor = defaultdict(int)
    rows = []
    for step in steps:
        if step.id in turns:
            request, prompt, ready, prefill, reserve = turns.pop(step.id)
            expected = min(prompt - cursor[request], 1024 - (ready if enabled else 1))
            if (not 2 <= ready <= 32 or not 0 < expected == prefill or prompt < 3072
                    or reserve != int(enabled) or not step.prefill
                    or step.prefill[0] != (request, prefill, cursor[request] + prefill)
                    or step.decode != min(ready, 1024 - prefill)):
                raise ValueError(f"aging trace disagrees with committed plan at step {step.id}")
            rows.append({
                "step": step.id, "scheduler_request": request, "prompt_tokens": prompt,
                "prefill_rows": prefill, "ready_decode_rows": ready, "executed_decode_rows": step.decode,
                "deferred_decode_rows": ready - step.decode, "gpu_ms": step.gpu / 1e6,
            })
        for request, count, target in step.prefill:
            if target != cursor[request] + count:
                raise ValueError(f"invalid prefill cursor at {step.id}")
            cursor[request] = target
    if turns or any(cursor[row["scheduler_request"]] != row["prompt_tokens"] for row in rows):
        raise ValueError("aging turn does not match a committed step and complete prefill")
    return rows


def verify_plan(arm, gold, gpu, mode, placement, binary, stage, reference_mode="control"):
    plan, receipt = read(arm / "plan.json"), read(arm / "receipt.json")
    old_plan, old_receipt = (json.load(gold.extractfile(f"gpu{gpu}/{reference_mode}/{file}.json"))
                             for file in ("plan", "receipt"))
    if (set(receipt) != set(old_receipt) or receipt["riley_binary_sha256"] != binary
            or any(receipt[key] != old_receipt[key] for key in set(receipt) - {
                "riley_binary_sha256", "holdout_sha256", "results",
            })):
        raise ValueError(f"server or checkpoint identity differs at GPU {gpu}/{mode}")
    if any(plan[key] != old_plan[key] for key in ("server", "cells", "server_cpu_affinity")):
        raise ValueError(f"schedule, server flags or CPU placement differs at GPU {gpu}/{mode}")
    if set(receipt["results"]) != set(plan["cells"]) or plan["server_cpu_affinity"] != placement["server_cpus"][gpu]:
        raise ValueError(f"serving plan coverage or affinity differs at GPU {gpu}/{mode}")
    for name, value in (
        ("RILEY_W3_RAG_STEPS", 1), ("RILEY_W3_PREFILL_FIFO", int(mode == "combined")),
        ("RILEY_W3_RESERVE_AGED_DECODE", int(mode != "control")),
    ):
        if [arg for arg in plan["docker"] if arg.startswith(f"{name}=")] != [f"{name}={value}"]:
            raise ValueError(f"{name} differs at GPU {gpu}/{mode}")
    read_only = f"type=bind,source={stage}/riley-diagnostic,target=/opt/riley/bin/riley,readonly"
    if plan["docker"].count(read_only) != 1:
        raise ValueError(f"diagnostic binary mount differs at GPU {gpu}/{mode}")
    return plan, receipt


def arm_results(arm, plan, receipt, mode, cpu, reg, baseline, timeline, trace_tool, focus):
    workload = {}
    for label, cell in plan["cells"].items():
        workload[label] = baseline["check_cell"](arm / f"{label}.json", receipt, cell, MATCH_KEYS, cpu)
    holdout_file = arm / "token-id-holdout.json"
    tokens = read(holdout_file)
    if (digest(holdout_file) != receipt["holdout_sha256"] or set(tokens) != {"c1", "c8"}
            or any(len(row) * 2 != reg["holdout_requests_per_arm"] for row in tokens.values())):
        raise ValueError("holdout artifact is missing or incomplete")
    trace = trace_tool["verify_trace"](
        arm, plan["cells"], reg["holdout_requests_per_arm"], mode == "combined", timeline, focus,
    )
    log = arm / "server.log"
    text = log.read_text()
    admits, _, steps, *_ = timeline["read_trace"](log, set(RUN_ID.findall(text)))
    labels = {}
    for label in workload:
        client = read(arm / f"{label}.json")
        receipts = client["warmup_response_ids"] + [row["response_id"] for row in client["measured"]]
        labels.update((admits[int(response[5:], 16)][0], label) for response in receipts)
    turns = aging_steps(steps, text.splitlines(), mode != "control")
    by_schedule = defaultdict(Counter)
    for row in turns:
        label = labels.get(row["scheduler_request"], "outside_measured_schedules")
        row["label"] = label
        by_schedule[label].update({"turns": 1, "ready_decode": row["ready_decode_rows"],
                                   "deferred_decode": row["deferred_decode_rows"]})
    result = {
        "batch_shapes": baseline["batch_summary"](arm / "shutdown.json"),
        "committed_steps": trace["committed_steps"], "focus": trace["focus"],
        "aging_by_schedule": by_schedule, "aging_turns": turns,
    }
    if mode == "combined":
        result["prefill_reorders"] = trace["choices"]
    return workload, tokens, result


def verify(root, registration, evidence_root, *, combined=False):
    reg = read(registration)
    control_mode, trial_mode = ("reserve", "combined") if combined else ("control", "reserve")
    orders = {"6": (trial_mode, control_mode), "7": (control_mode, trial_mode)}
    reference = evidence_root / (RESERVATION if combined else REFERENCE)
    prior = read(reference / "registration.json")
    matching = ("base_revision", "physical_gpus", "schedules_sha256", "measured_requests_per_arm",
                "holdout_requests_per_arm", "write_error_p95_limit_ms",
                "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
                "short_throughput_max_drop_pct_exclusive")
    if (any(reg[name] != prior[name] for name in matching) or reg["parent_stage"] != prior["stage"]
            or reg["parent_binary_sha256"] != (reference / "build-result.sha256").read_text().split()[0]):
        raise ValueError("registration differs from the prior serving conditions or gates")
    if (reg["orders"] != {gpu: list(order) for gpu, order in orders.items()}
            or reg["physical_gpus"] != [6, 7]
            or any((root / f"{phase}.exit").read_text().strip() != "0" for phase in ("build", "campaign"))):
        raise ValueError("incomplete experiment or different GPU order")
    for filename, key in (
        ("build-dispatch.sha256", "build_dispatch_sha256"), ("build-result.sha256", "build_result_sha256"),
        ("campaign-dispatch.sha256", "campaign_dispatch_sha256"), ("schedules.json", "schedules_sha256"),
    ):
        if digest(root / filename) != reg[key]:
            raise ValueError(f"registered fingerprint differs: {filename}")
    for phase in ("build", "campaign"):
        if (int((root / f"{phase}.pid").read_text()) != reg[f"{phase}_pid"]
                or (root / f"{phase}.started").read_text().strip() != reg[f"{phase}_started_at"]):
            raise ValueError(f"raw launch receipt differs: {phase}")
    binary, file = (root / "build-result.sha256").read_text().splitlines()[0].split()
    if file != "riley-diagnostic" or (binary == reg["parent_binary_sha256"]) != combined:
        raise ValueError("diagnostic binary reuse differs from the experiment")
    if combined and reg["build_result_sha256"] != prior["build_result_sha256"]:
        raise ValueError("combined run must reuse the entire verified build receipt")
    placement = read(root / "cpu-placement.json")
    baseline = runpy.run_path(str(evidence_root / "20261124-w3-rag-prefill-order/verify.py"))
    timeline = runpy.run_path(str(evidence_root / "20261124-w3-rag-prefill-steps/verify.py"))
    trace_tool = runpy.run_path(str(evidence_root / REFERENCE / "verify.py"))
    focus = {name: set(indexes) for name, indexes in reg["focus_gpu7"].items()}
    events = re.findall(r"^(START|END) gpu([67]) (reserve|control|combined) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    gpus = {}
    with (tarfile.open(reference / "raw-results.tar.gz", "r:gz") as gold,
          tarfile.open(evidence_root / "20261124-w3-separated-client/initial-results.tar.gz", "r:gz") as engine):
        for filename in ("schedules.json", "cpu-placement.json"):
            if (root / filename).read_bytes() != gold.extractfile(filename).read():
                raise ValueError(f"frozen schedules or CPU placement differs: {filename}")
        for gpu, order in orders.items():
            expected = [(when, mode, code) for mode in order for when, code in (("START", ""), ("END", "0"))]
            if [(when, mode, code) for when, index, mode, _, code in events if index == gpu] != expected:
                raise ValueError(f"order or child exit differs on GPU {gpu}")
            previous = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for phase, snapshot in (
                ("staged-start", root / f"gpu{gpu}-staged-start.json"),
                *((phase, root / f"gpu{gpu}/{phase}.json") for phase in ("before", "between", "after")),
            ):
                holder = read(snapshot)
                if (holder["phase"] != "idle" or holder["other_processes"] or holder["physical_index"] != int(gpu)
                        or holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30
                        or any(holder[key] != previous[key] for key in ("gpu", "pid", "start_ticks"))):
                    raise ValueError(f"holder identity or reservation differs on GPU {gpu}/{phase}")
            workloads, tokens, arms = {}, {}, {}
            for mode in order:
                arm = root / f"gpu{gpu}/{mode}"
                plan, receipt = verify_plan(arm, gold, gpu, mode, placement, binary, reg["stage"], control_mode)
                workloads[mode], tokens[mode], arms[mode] = arm_results(
                    arm, plan, receipt, mode, placement["client_cpus"][gpu], reg, baseline,
                    timeline, trace_tool, focus if gpu == "7" else {},
                )

            cells = {}
            for label, control in workloads[control_mode].items():
                trial = workloads[trial_mode][label]
                cell = baseline["compare_cells"](control, trial, label, reg)
                cell[trial_mode] = cell.pop("fifo")
                cell["abs_write_error_p95_ms"][trial_mode] = cell["abs_write_error_p95_ms"].pop("fifo")
                if combined:
                    cell[control_mode] = cell.pop("control")
                    cell["abs_write_error_p95_ms"][control_mode] = cell["abs_write_error_p95_ms"].pop("control")
                cell["outputs_equal_previous"] = {}
                previous_cell = json.load(gold.extractfile(f"gpu{gpu}/{control_mode}/{label}.json"))
                for mode in order:
                    same = baseline["compare_cells"](
                        (previous_cell["measured"], previous_cell["statistics"]), workloads[mode][label], label, reg,
                    )
                    cell["outputs_equal_previous"][mode] = same["gates"]["output"]
                if gpu == "7" and label.startswith("rag-"):
                    previous_cell = json.load(engine.extractfile(f"gpu7/coder14-vllm/{label}.json"))
                    against_vllm = {}
                    for mode in order:
                        rows, metrics = workloads[mode][label]
                        errors = [abs(a["write_error_ms"] - b["write_error_ms"])
                                  for a, b in zip(rows, previous_cell["measured"], strict=True)]
                        timing_error = baseline["percentile"](errors, .95)
                        against_vllm[mode] = {
                            "write_error_p95_ms": timing_error,
                            "ttft_p95_ratio": metrics["ttft_p95_ms"] / previous_cell["statistics"]["ttft_p95_ms"]
                            if cell["gates"]["timing"] and timing_error <= reg["write_error_p95_limit_ms"] else None,
                        }
                    cell["frozen_vllm"] = against_vllm
                cells[label] = cell
            if sum(row["requests"] for row in cells.values()) != reg["measured_requests_per_arm"]:
                raise ValueError(f"frozen replay coverage differs on GPU {gpu}")
            prior_tokens = json.load(gold.extractfile(f"gpu{gpu}/{control_mode}/token-id-holdout.json"))
            gpus[gpu] = {
                "order": order, "cells": cells, "arms": arms,
                "holdout_matches_c1_c8": {mode: rows["c1"] == rows["c8"] for mode, rows in tokens.items()},
                "holdout_matches_previous": {mode: rows == prior_tokens for mode, rows in tokens.items()},
            }
    return {"base_revision": reg["base_revision"], "binary_sha256": binary, "gpus": gpus}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--combined", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.stage, args.registration, args.evidence_root, combined=args.combined),
                     indent=2, allow_nan=False))
