"""Verify the isolated A100 admission-order experiment against frozen inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
PARENT = EVIDENCE / "20261124-w3-aged-decode-native"
REFERENCE = EVIDENCE / "20261124-w3-separated-client" / "initial-results.tar.gz"
PACKAGE = EVIDENCE / "20261124-w4-pointer-syntax/a100/cpu-evidence/report.json"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def check_manifests(stage, reg, package):
    for name, key in (
        ("dispatch.sha256", "build_dispatch_sha256"),
        ("campaign-dispatch.sha256", "campaign_dispatch_sha256"),
        ("prefill-admission-fifo.patch", "candidate_patch_sha256"),
        ("schedules.json", "schedules_sha256"),
        ("src/crates/riley-scheduler/src/scheduler.rs", "patched_scheduler_sha256"),
    ):
        expect(digest(stage / name) == reg[key], f"registered input differs: {name}")
    for phase in ("build", "campaign"):
        expect((stage / f"{phase}.exit").read_text().strip() == "0"
               and int((stage / f"{phase}.pid").read_text()) == reg[f"{phase}_pid"]
               and (stage / f"{phase}.started").read_text().strip() == reg[f"{phase}_started_at"],
               f"{phase} process or exit differs")
    binary = None
    for name in ("dispatch", "campaign-dispatch", "build-result", "runtime", "result"):
        manifest = (stage / f"{name}.sha256").read_text().splitlines()
        expect(bool(manifest), f"empty checksum list: {name}")
        seen = set()
        for line in manifest:
            match = re.fullmatch(r"([0-9a-f]{64})  ((?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*)", line)
            expect(match is not None, f"invalid checksum in {name}")
            fingerprint, path = match.groups()
            expect(path not in seen, f"duplicate checksum entry: {path}")
            seen.add(path)
            if path == "riley-diagnostic":
                expect(name == "build-result" and binary is None, "unexpected binary checksum")
                binary = fingerprint
            else:
                expect(digest(stage / path) == fingerprint, f"artifact does not match checksum: {path}")
    expect(binary and binary != package["binaries"]["/opt/riley/bin/riley"], "missing diagnostic binary hash")
    return binary


def paired_latency(control, candidate, percentile):
    """Describe how much work remains after the earlier first-token boundary."""
    deltas = [(b["ttft"] - a["ttft"], (b["e2e"] - b["ttft"]) - (a["e2e"] - a["ttft"]),
               b["e2e"] - a["e2e"]) for a, b in zip(control, candidate, strict=True)]
    return {
        "faster_first_token_requests": sum(first < 0 for first, _, _ in deltas),
        "faster_first_token_and_longer_generation_requests": sum(
            first < 0 and decode > 0 for first, decode, _ in deltas
        ),
        "faster_first_token_and_later_completion_requests": sum(
            first < 0 and complete > 0 for first, _, complete in deltas
        ),
        "complete_p95_ms": {name: percentile([row["e2e"] * 1e3 for row in rows], .95)
                            for name, rows in (("control", control), ("candidate", candidate))},
    }


def summary(stage, registration=HERE / "registration.json"):
    reg = read(registration)
    package = read(PACKAGE)
    expect(package["status"] == "passed" and package["image_id"] == reg["package_image_id"]
           and package["source_revision"] == reg["source_revision"]
           and digest(PACKAGE) == reg["package_cpu_report_sha256"], "packaged image differs")
    binary = check_manifests(stage, reg, package)
    baseline = runpy.run_path(str(EVIDENCE / "20261124-w3-rag-prefill-order/verify.py"))
    native = runpy.run_path(str(PARENT / "verify.py"))
    events = re.findall(r"^(START|END) gpu([67]) (control|candidate) \S+(?: exit=(\d+))?$",
                        (stage / "campaign.log").read_text(), re.MULTILINE)
    gpus = {}
    with (tarfile.open(PARENT / "raw-results.tar.gz", "r:gz") as gold,
          tarfile.open(REFERENCE, "r:gz") as reference):
        for path in ("schedules.json", "cpu-placement.json"):
            expect((stage / path).read_bytes() == gold.extractfile(path).read(), f"frozen {path} changed")
        placement = read(stage / "cpu-placement.json")
        for gpu, order in reg["orders"].items():
            parent = stage / f"gpu{gpu}"
            expected = [(kind, arm, code) for arm in order for kind, code in (("START", ""), ("END", "0"))]
            actual = [(kind, arm, code) for kind, index, arm, code in events if index == gpu]
            expect(actual == expected, f"GPU {gpu} run order or exit differs")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for phase in ("before", "between", "after"):
                idle = read(parent / f"{phase}.json")
                expect(idle["phase"] == "idle" and not idle["other_processes"]
                       and idle["allocation_bytes"] == idle["torch_reserved_bytes"] == 20 * 2**30
                       and all(idle[key] == old_holder[key] for key in ("gpu", "pid", "start_ticks", "physical_index")),
                       f"GPU {gpu} holder identity or restoration differs at {phase}")
            old_plan, old_receipt, old_tokens = (
                json.load(gold.extractfile(f"gpu{gpu}/control/{name}.json"))
                for name in ("plan", "receipt", "token-id-holdout")
            )
            captures, tokens, batches = {}, {}, {}
            for mode in order:
                arm = parent / mode
                command = read(arm / "plan.json")["docker"]
                option = f'{reg["mode_env"]}={int(mode == "candidate")}'
                expect([arg for arg in command if arg.startswith(reg["mode_env"] + "=")] == [option]
                       and command[command.index(option) - 1] == "--env", f"unexpected switch on GPU {gpu}/{mode}")
                captures[mode], tokens[mode], batches[mode] = native["verify_arm"](
                    arm, old_plan, old_receipt, dict(reg, mode_flag=reg["mode_env"]),
                    binary, "control", placement["client_cpus"][gpu], baseline,
                )
                expect(tokens[mode] == old_tokens and tokens[mode]["c1"] == tokens[mode]["c8"],
                       f"token IDs changed on GPU {gpu}/{mode}")
            cells = {}
            for label, control in captures["control"].items():
                candidate = captures["candidate"][label]
                cell = baseline["compare_cells"](control, candidate, label, reg)
                cell["candidate"] = cell.pop("fifo")
                errors = cell["abs_write_error_p95_ms"]
                errors["candidate"] = errors.pop("fifo")
                previous = json.load(gold.extractfile(f"gpu{gpu}/control/{label}.json"))["measured"]
                keys = ("request_index", "input_sha256", "output_sha256", "prompt_tokens", "completion_tokens", "finish")
                cell["outputs_equal_previous"] = {
                    mode: all(all(a[k] == b[k] for k in keys)
                              for a, b in zip(captures[mode][label][0], previous, strict=True))
                    for mode in order
                }
                cell["paired_latency"] = paired_latency(control[0], candidate[0], baseline["percentile"])
                if gpu == "7" and label.startswith("rag-"):
                    vllm = json.load(reference.extractfile(f"gpu7/coder14-vllm/{label}.json"))
                    ratios = {}
                    for mode in order:
                        rows, stats = captures[mode][label]
                        identity = ("request_index", "input_sha256", "prompt_tokens", "scheduled_write_ns")
                        expect(all(all(a[k] == b[k] for k in identity)
                                   for a, b in zip(rows, vllm["measured"], strict=True)),
                               f"historical vLLM inputs changed in {label}")
                        error = baseline["percentile"]([
                            abs(a["write_error_ms"] - b["write_error_ms"])
                            for a, b in zip(rows, vllm["measured"], strict=True)
                        ], .95)
                        valid = cell["gates"]["timing"] and error <= reg["write_error_p95_limit_ms"]
                        ratio = stats["ttft_p95_ms"] / vllm["statistics"]["ttft_p95_ms"] if valid else None
                        ratios[mode] = {"write_error_p95_ms": error, "ttft_p95_ratio": ratio}
                        for name in ("phase_one", "final"):
                            ratios[mode][name + "_target"] = ratio is not None and ratio <= reg[f"rag_ttft_p95_vllm_{name}_ratio"]
                    cell["frozen_vllm"] = ratios
                cells[label] = cell
            expect(sum(cell["requests"] for cell in cells.values()) == reg["measured_requests_per_arm"],
                   f"GPU {gpu} has an incomplete workload")
            gpus[gpu] = {"order": order, "batch_shapes": batches, "cells": cells}
    return {"source_revision": reg["source_revision"], "binary_sha256": binary, "gpus": gpus}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(summary(args.stage), indent=2, allow_nan=False))
