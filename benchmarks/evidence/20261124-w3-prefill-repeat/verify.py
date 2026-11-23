"""Verify the same-binary A/A and A/B prefill scheduling replay."""

import argparse
import json
from pathlib import Path
import re
import runpy
import sys
import tarfile


HERE = Path(__file__).resolve().parent
BROAD = HERE.parent / "20261124-w3-rag-prefill-order"
PRIOR = HERE.parent / "20261124-w3-wide-prefill-order"
FROZEN = HERE.parent / "20261124-w3-separated-client"
PHASES = ("before", "between-first", "between-second", "after")


def check_manifest(root, filename, sha, omitted_binary):
    for entry in (root / filename).read_text().splitlines():
        digest, relative = entry.split("  ", 1)
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unexpected manifest path: {path}")
        actual = omitted_binary if relative == "riley-diagnostic" else sha(root / path)
        if actual != digest:
            raise ValueError(f"manifest mismatch: {path}")


def verify(root):
    sys.path[0:0] = [str(root), str(root / "tools"), str(PRIOR)]
    common = runpy.run_path(str(BROAD / "verify.py"))
    config = runpy.run_path(str(BROAD / "configure.py"))
    audit = runpy.run_path(str(PRIOR / "audit_cost.py"))
    read, sha = common["read"], common["sha"]
    reg, prior_reg = read(HERE / "registration.json"), read(PRIOR / "registration.json")
    checks = (("campaign.exit", "0"), ("campaign.pid", str(reg["campaign_pid"])),
              ("campaign.started", reg["campaign_started_at"]))
    for filename, expected in checks:
        if (root / filename).read_text().strip() != expected:
            raise ValueError(f"unexpected campaign receipt: {filename}")
    for filename, key in (("dispatch.sha256", "dispatch_sha256"), ("schedules.json", "schedules_sha256")):
        if sha(root / filename) != reg[key]:
            raise ValueError(f"registered input differs: {filename}")
    for filename in ("dispatch.sha256", "runtime.sha256", "result.sha256"):
        check_manifest(root, filename, sha, reg["binary_sha256"])
    schedules = read(root / "schedules.json")["schedules"]

    events = re.findall(r"^(START|END) gpu([67]) (control-first|control-second|fifo) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    results = {}
    with (tarfile.open(PRIOR / "raw-results.tar.gz", "r:gz") as prior,
          tarfile.open(FROZEN / "initial-results.tar.gz", "r:gz") as reference):
        source = lambda name: prior.extractfile(name).read()
        sha256 = common["sha256"]
        if sha256(source("result.sha256")).hexdigest() != reg["source_result_sha256"]:
            raise ValueError("source run manifest differs")
        expected_arm = source("arm.py").replace(b'out = OUT / os.environ["W3_PREFILL_ORDER"]',
                                                b'out = OUT / os.environ["W3_PREFILL_LABEL"]')
        if (root / "arm.py").read_bytes() != expected_arm:
            raise ValueError("case label must be the only arm change")
        for name in ("client.py", "replay.py", "holder.py", "holdout.py", "schedules.json", "cpu-placement.json"):
            if (root / name).read_bytes() != source(name):
                raise ValueError(f"same-binary replay changed {name}")
        for gpu, order in reg["orders"].items():
            actual = [(kind, mode, code) for kind, index, mode, _, code in events if index == gpu]
            if actual != [(kind, mode, code) for mode in order for kind, code in (("START", ""), ("END", "0"))]:
                raise ValueError(f"registered order or exit differs on GPU {gpu}")
            parent = root / f"gpu{gpu}"
            holder = json.loads(source(f"gpu{gpu}/after.json"))
            for phase in PHASES:
                row = read(parent / f"{phase}.json")
                if (row["physical_index"] != int(gpu) or row["phase"] != "idle" or row["other_processes"]
                        or row["allocation_bytes"] != row["torch_reserved_bytes"] or row["allocation_bytes"] != 20 * 2**30
                        or any(row[key] != holder[key] for key in ("pid", "start_ticks", "gpu"))):
                    raise ValueError(f"GPU {gpu} reservation changed")
            original = json.loads(source(f"gpu{gpu}/control/receipt.json"))
            server = json.loads(source(f"gpu{gpu}/control/plan.json"))["server"]
            expected_holdout = source(f"gpu{gpu}/control/token-id-holdout.json")
            if reg["binary_sha256"] != original["riley_binary_sha256"]:
                raise ValueError("registered binary does not match the previous control")

            captures, counts, batches = {}, {}, {}
            for mode in order:
                arm = parent / mode
                plan, receipt = read(arm / "plan.json"), read(arm / "receipt.json")
                marker = f"RILEY_W3_PREFILL_FIFO={int(mode == 'fifo')}"
                switches = [arg for arg in plan["docker"] if arg.startswith("RILEY_W3_PREFILL_FIFO=")]
                if (plan["server"] != server or plan["cells"] != schedules[gpu]
                        or plan["server_cpu_affinity"] != config["SERVER_CPUS"][gpu] or switches != [marker]
                        or any(receipt[key] != original[key] for key in set(original) - {"results"})):
                    raise ValueError(f"same-binary execution inputs differ on GPU {gpu}/{mode}")
                holdout = arm / "token-id-holdout.json"
                if sha(holdout) != receipt["holdout_sha256"] or json.loads(expected_holdout) != read(holdout):
                    raise ValueError("holdout or generated token IDs differ")
                (count,) = re.findall(r"^W3-PREFILL-FIFO (\d+)$", (arm / "server.log").read_text(), re.MULTILINE)
                counts[mode] = int(count)
                if (counts[mode] > 0) != (mode == "fifo"):
                    raise ValueError("scheduler switch did not match the registered mode")
                batches[mode] = common["batch_summary"](arm / "shutdown.json")
                captures[mode] = {}
                for label, cell in schedules[gpu].items():
                    path = arm / f"{label}.json"
                    capture = common["check_cell"](path, receipt, cell, config["MATCHED_KEYS"], config["CLIENT_CPUS"][gpu])
                    audit["check_statistics"](read(path), common["percentile"])
                    measured, _ = capture
                    previous = json.loads(source(f"gpu{gpu}/control/{label}.json"))["measured"]
                    if any(a["output_sha256"] != b["output_sha256"] for a, b in zip(measured, previous, strict=True)):
                        raise ValueError("output hash differs from the same-binary prior control")
                    captures[mode][label] = capture
                if sum(len(capture[0]) for capture in captures[mode].values()) != prior_reg["measured_requests_per_arm"]:
                    raise ValueError("incomplete workload coverage")

            cells = {}
            for label in schedules[gpu]:
                arms = {mode: captures[mode][label] for mode in order}
                comparisons = {"control_repeat": ("control-first", "control-second"),
                               "candidate_vs_first": ("control-first", "fifo"),
                               "candidate_vs_second": ("control-second", "fifo")}
                cells[label] = {name: common["compare_cells"](arms[a], arms[b], label, prior_reg)
                                for name, (a, b) in comparisons.items()}
                if gpu == "7" and label in config["FIXED"]:
                    prior_vllm = json.load(reference.extractfile(f"gpu7/coder14-vllm/{label}.json"))
                    metric = "ttft_p95_ms" if label.startswith("rag-") else "tpot_p50_ms"
                    cells[label]["frozen_vllm_reference"] = {}
                    for mode, (rows, metrics) in arms.items():
                        paired = list(zip(rows, prior_vllm["measured"], strict=True))
                        if any(a["request_index"] != b["request_index"] or a["input_sha256"] != b["input_sha256"]
                               for a, b in paired):
                            raise ValueError(f"frozen vLLM request differs for {label}")
                        p95 = common["percentile"]([abs(a["write_error_ms"] - b["write_error_ms"])
                                                   for a, b in paired], .95)
                        cells[label]["frozen_vllm_reference"][mode] = {
                            "metric": metric, "between_write_error_p95_ms": p95,
                            "ratio": metrics[metric] / prior_vllm["statistics"][metric]
                            if p95 <= prior_reg["write_error_p95_limit_ms"] else None,
                        }
            results[gpu] = {"order": order, "reordered_steps": counts, "batch_shapes": batches, "cells": cells}

    return {"base_revision": prior_reg["base_revision"], "binary_sha256": reg["binary_sha256"], "gpus": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.stage), indent=2, allow_nan=False))
