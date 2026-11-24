"""Bind the native serve flag experiment to its build, package and frozen requests."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
from statistics import median
import tarfile


REFERENCE = "20261124-w3-aged-decode-reserve"
NATIVE = "20261124-w3-aged-decode-native"
PACKAGE = "20261124-w4-uri-reference-fix/a100/cpu-evidence/report.json"
MATCH_KEYS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")
GATES = ("orders", "physical_gpus", "schedules_sha256", "measured_requests_per_arm",
         "holdout_requests_per_arm", "write_error_p95_limit_ms",
         "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
         "short_throughput_max_drop_pct_exclusive")
SHORT_CELLS = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals", "c32-riley-arrivals")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_flag(plan, old_plan, stage, image, mode, flag):
    expected_server = old_plan["server"] + ([flag] if mode == "reserve" else [])
    if plan["server"] != expected_server or any(plan[name] != old_plan[name]
                                                 for name in ("cells", "server_cpu_affinity")):
        raise ValueError(f"serving arguments or frozen replay changed in {mode}")
    command = plan["docker"]
    mount = f"type=bind,source={stage}/riley-diagnostic,target=/opt/riley/bin/riley,readonly"
    if (command[-(len(expected_server) + 1):] != [image, *expected_server]
            or command.count(mount) != 1 or "127.0.0.1::8080" not in command
            or any(arg.startswith("RILEY_W3_") for arg in command)):
        raise ValueError(f"native package, mounted binary or argv differs in {mode}")


def verify_arm(path, old_plan, old_receipt, reg, binary, mode, client_cpu, baseline):
    plan, receipt = (read(path / f"{file}.json") for file in ("plan", "receipt"))
    check_flag(plan, old_plan, reg["stage"], reg["package_image_id"], mode, reg["mode_flag"])
    changed = {
        "riley_binary_sha256": binary, "source_revision": reg["source_revision"],
        "image_id": reg["package_image_id"], "cpu_report_sha256": reg["package_cpu_report_sha256"],
    }
    invariant = set(old_receipt) - set(changed) - {"holdout_sha256", "results"}
    if (set(receipt) != set(old_receipt) or any(receipt[k] != v for k, v in changed.items())
            or any(receipt[k] != old_receipt[k] for k in invariant)
            or set(receipt["results"]) != set(plan["cells"])):
        raise ValueError(f"native runtime, device, or checkpoint differs in {path}")
    captured = {
        name: baseline["check_cell"](path / f"{name}.json", receipt, cell, MATCH_KEYS, client_cpu)
        for name, cell in plan["cells"].items()
    }
    token_file = path / "token-id-holdout.json"
    tokens = read(token_file)
    if (digest(token_file) != receipt["holdout_sha256"] or set(tokens) != {"c1", "c8"}
            or any(len(rows) * 2 != reg["holdout_requests_per_arm"] for rows in tokens.values())):
        raise ValueError(f"token ID holdout missing in {path}")
    return captured, tokens, baseline["batch_summary"](path / "shutdown.json")


def verify(root, registration, evidence, *, short_retry=False):
    reg = read(registration)
    reference = evidence / (NATIVE if short_retry else REFERENCE)
    prior = read(reference / "registration.json")
    package_path = evidence / PACKAGE
    package = read(package_path)
    scoped_fields = {"orders", "schedules_sha256", "measured_requests_per_arm"}
    matching = [key for key in GATES if key not in scoped_fields] if short_retry else GATES
    if (any(reg[key] != prior[key] for key in matching) or reg["mode_flag"] != "--prefill-aging-reserve-decode"
            or reg["source_revision"] != package["source_revision"]
            or reg["package_image_id"] != package["image_id"]
            or reg["package_cpu_report_sha256"] != digest(package_path)
            or package["status"] != "passed"):
        raise ValueError("registration differs from the calibrated workload or verified package")
    if short_retry and (
        reg["orders"] != {gpu: list(reversed(order)) for gpu, order in prior["orders"].items()}
        or tuple(reg["cells"]) != SHORT_CELLS or reg["measured_requests_per_arm"] != 320
        or reg["parent_stage"] != prior["stage"] or reg["parent_schedules_sha256"] != prior["schedules_sha256"]
        or reg["candidate_patch_sha256"] != prior["candidate_patch_sha256"]
        or reg["build_result_sha256"] != prior["build_result_sha256"]
    ):
        raise ValueError("C32 retry must reuse the verified binary and original request prefix in reverse order")
    for name, key in (
        ("build-dispatch.sha256", "build_dispatch_sha256"), ("build-result.sha256", "build_result_sha256"),
        ("campaign-dispatch.sha256", "campaign_dispatch_sha256"), ("schedules.json", "schedules_sha256"),
    ):
        if digest(root / name) != reg[key]:
            raise ValueError(f"native build input or captured data differs: {name}")
    if not short_retry and digest(root / "aged-decode-option.patch") != reg["candidate_patch_sha256"]:
        raise ValueError("native option patch differs from registration")
    for phase in ("build", "campaign"):
        if ((root / f"{phase}.exit").read_text().strip() != "0"
                or int((root / f"{phase}.pid").read_text()) != reg[f"{phase}_pid"]
                or (root / f"{phase}.started").read_text().strip() != reg[f"{phase}_started_at"]):
            raise ValueError(f"stage exit or process identity differs: {phase}")
    binary, filename = (root / "build-result.sha256").read_text().splitlines()[0].split()
    if (filename != "riley-diagnostic" or not re.fullmatch(r"[a-f0-9]{64}", binary)
            or binary == package["binaries"]["/opt/riley/bin/riley"]
            or (binary == (reference / "build-result.sha256").read_text().split()[0]) != short_retry
            or reg["mode_flag"] not in (root / "help.txt").read_text()):
        raise ValueError("candidate binary or native help does not match the new option")

    baseline = runpy.run_path(str(evidence / "20261124-w3-rag-prefill-order/verify.py"))
    events = re.findall(r"^(START|END) gpu([67]) (reserve|control) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    gpus = {}
    with (tarfile.open(reference / "raw-results.tar.gz", "r:gz") as gold,
          tarfile.open(evidence / "20261124-w3-separated-client/initial-results.tar.gz", "r:gz") as engine):
        schedules = gold.extractfile("schedules.json").read()
        if short_retry:
            original = json.loads(schedules)["schedules"]
            if any(tuple(cells)[:len(SHORT_CELLS)] != SHORT_CELLS for cells in original.values()):
                raise ValueError("C32 retry did not preserve both preceding RAG workloads")
            prefix = {gpu: {label: cells[label] for label in SHORT_CELLS} for gpu, cells in original.items()}
            schedules = (json.dumps({"schedules": prefix}, indent=2) + "\n").encode()
        if ((root / "schedules.json").read_bytes() != schedules
                or (root / "cpu-placement.json").read_bytes() != gold.extractfile("cpu-placement.json").read()):
            raise ValueError("captured schedules or CPU placement changed")
        placement = read(root / "cpu-placement.json")
        for gpu, order in reg["orders"].items():
            expected = [(kind, mode, status) for mode in order for kind, status in (("START", ""), ("END", "0"))]
            observed = [(kind, mode, status) for kind, index, mode, _, status in events if index == gpu]
            if expected != observed:
                raise ValueError(f"GPU {gpu} ran a different order or exited early")
            original = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            paths = [root / f"gpu{gpu}-staged-start.json",
                     *(root / f"gpu{gpu}/{phase}.json" for phase in ("before", "between", "after"))]
            for path in paths:
                holder = read(path)
                if (holder["phase"] != "idle" or holder["other_processes"] or holder["physical_index"] != int(gpu)
                        or holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30
                        or any(holder[k] != original[k] for k in ("gpu", "pid", "start_ticks"))):
                    raise ValueError(f"original 20GiB reservation differs at {path}")
            old_mode = "control" if short_retry else "reserve"
            old_plan, old_receipt, old_tokens = (json.load(gold.extractfile(f"gpu{gpu}/{old_mode}/{name}.json"))
                                                  for name in ("plan", "receipt", "token-id-holdout"))
            if short_retry:
                old_plan["cells"] = {label: old_plan["cells"][label] for label in SHORT_CELLS}
            workload, tokens, shapes = {}, {}, {}
            for mode in order:
                arm = root / f"gpu{gpu}/{mode}"
                workload[mode], tokens[mode], shapes[mode] = verify_arm(
                    arm, old_plan, old_receipt, reg, binary, mode, placement["client_cpus"][gpu], baseline,
                )
            cells = {}
            for label, control in workload["control"].items():
                cell = baseline["compare_cells"](control, workload["reserve"][label], label, reg)
                cell["reserve"] = cell.pop("fifo")
                errors = cell["abs_write_error_p95_ms"]
                errors["reserve"] = errors.pop("fifo")
                previous = json.load(gold.extractfile(f"gpu{gpu}/{old_mode}/{label}.json"))
                reference_rows = previous["measured"]
                cell["outputs_equal_previous"] = {
                    mode: all(a["output_sha256"] == b["output_sha256"]
                              for a, b in zip(workload[mode][label][0], reference_rows, strict=True))
                    for mode in order
                }
                if gpu == "7" and label.startswith("rag-"):
                    reference_vllm = json.load(engine.extractfile(f"gpu7/coder14-vllm/{label}.json"))
                    ratios = {}
                    for mode in order:
                        rows, stats = workload[mode][label]
                        timing = baseline["percentile"]([
                            abs(a["write_error_ms"] - b["write_error_ms"])
                            for a, b in zip(rows, reference_vllm["measured"], strict=True)
                        ], .95)
                        ratios[mode] = {
                            "write_error_p95_ms": timing,
                            "ttft_p95_ratio": stats["ttft_p95_ms"] / reference_vllm["statistics"]["ttft_p95_ms"]
                            if cell["gates"]["timing"] and timing <= reg["write_error_p95_limit_ms"] else None,
                        }
                    cell["frozen_vllm"] = ratios
                if short_retry and label == "c32-riley-arrivals":
                    cell["client_phases"] = {}
                    for mode in order:
                        rows, _ = workload[mode][label]
                        before = [(row["sent_at"] - row["start"]) * 1000 for row in rows]
                        after = [row["ttft"] * 1000 - elapsed for row, elapsed in zip(rows, before, strict=True)]
                        if any(elapsed < 0 for elapsed in after):
                            raise ValueError("client first token precedes send completion")
                        cell["client_phases"][mode] = {
                            "before_write_p50_ms": median(before), "after_write_p50_ms": median(after),
                            "prompt_tokens_min": min(row["prompt_tokens"] for row in rows),
                            "prompt_tokens_max": max(row["prompt_tokens"] for row in rows),
                        }
                cells[label] = cell
            if sum(cell["requests"] for cell in cells.values()) != reg["measured_requests_per_arm"]:
                raise ValueError(f"GPU {gpu} omitted a scheduled response")
            gpus[gpu] = {
                "order": order, "cells": cells, "batch_shapes": shapes,
                "holdout_matches_c1_c8": {mode: rows["c1"] == rows["c8"] for mode, rows in tokens.items()},
                "holdout_matches_previous": {mode: rows == old_tokens for mode, rows in tokens.items()},
            }
    return {"source_revision": reg["source_revision"], "image_id": reg["package_image_id"],
            "candidate_patch_sha256": reg["candidate_patch_sha256"], "binary_sha256": binary, "gpus": gpus}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--short-retry", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.capture, args.registration, args.evidence_root, short_retry=args.short_retry),
                     indent=2, allow_nan=False))
