"""Bind the native serve flag experiment to its build, package and frozen requests."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile


REFERENCE = "20261124-w3-aged-decode-reserve"
PACKAGE = "20261124-w4-uri-reference-fix/a100/cpu-evidence/report.json"
MATCH_KEYS = ("request_index", "input_sha256", "prompt_tokens", "completion_tokens", "finish")
GATES = ("orders", "physical_gpus", "schedules_sha256", "measured_requests_per_arm",
         "holdout_requests_per_arm", "write_error_p95_limit_ms",
         "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
         "short_throughput_max_drop_pct_exclusive")


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


def verify(root, registration, evidence):
    reg = read(registration)
    reference = evidence / REFERENCE
    prior = read(reference / "registration.json")
    package_path = evidence / PACKAGE
    package = read(package_path)
    if (any(reg[key] != prior[key] for key in GATES) or reg["mode_flag"] != "--prefill-aging-reserve-decode"
            or reg["source_revision"] != package["source_revision"]
            or reg["package_image_id"] != package["image_id"]
            or reg["package_cpu_report_sha256"] != digest(package_path)
            or package["status"] != "passed"):
        raise ValueError("registration differs from the calibrated workload or verified package")
    for name, key in (
        ("build-dispatch.sha256", "build_dispatch_sha256"), ("build-result.sha256", "build_result_sha256"),
        ("campaign-dispatch.sha256", "campaign_dispatch_sha256"), ("schedules.json", "schedules_sha256"),
        ("aged-decode-option.patch", "candidate_patch_sha256"),
    ):
        if digest(root / name) != reg[key]:
            raise ValueError(f"native build input or captured data differs: {name}")
    for phase in ("build", "campaign"):
        if ((root / f"{phase}.exit").read_text().strip() != "0"
                or int((root / f"{phase}.pid").read_text()) != reg[f"{phase}_pid"]
                or (root / f"{phase}.started").read_text().strip() != reg[f"{phase}_started_at"]):
            raise ValueError(f"stage exit or process identity differs: {phase}")
    binary, filename = (root / "build-result.sha256").read_text().splitlines()[0].split()
    if (filename != "riley-diagnostic" or not re.fullmatch(r"[a-f0-9]{64}", binary)
            or binary in (package["binaries"]["/opt/riley/bin/riley"],
                          (reference / "build-result.sha256").read_text().split()[0])
            or reg["mode_flag"] not in (root / "help.txt").read_text()):
        raise ValueError("candidate binary or native help does not match the new option")

    baseline = runpy.run_path(str(evidence / "20261124-w3-rag-prefill-order/verify.py"))
    events = re.findall(r"^(START|END) gpu([67]) (reserve|control) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    gpus = {}
    with (tarfile.open(reference / "raw-results.tar.gz", "r:gz") as gold,
          tarfile.open(evidence / "20261124-w3-separated-client/initial-results.tar.gz", "r:gz") as engine):
        for name in ("schedules.json", "cpu-placement.json"):
            if (root / name).read_bytes() != gold.extractfile(name).read():
                raise ValueError(f"captured {name} is outside the frozen reference")
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
            old_plan, old_receipt, old_tokens = (json.load(gold.extractfile(f"gpu{gpu}/reserve/{name}.json"))
                                                  for name in ("plan", "receipt", "token-id-holdout"))
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
                previous = json.load(gold.extractfile(f"gpu{gpu}/reserve/{label}.json"))
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
    args = parser.parse_args()
    print(json.dumps(verify(args.capture, args.registration, args.evidence_root), indent=2, allow_nan=False))
