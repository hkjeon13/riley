"""Check the first-arrival candidate against matched Riley arms and frozen requests."""

import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import re
import runpy
import sys
import tarfile


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "20261124-w3-separated-client"
WAVE_SOURCE = HERE.parent / "20261124-w3-chunk-pair"
METRICS = ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "itl_p99_ms", "output_tok_s")
sys.path.insert(0, str(HERE.parents[2] / "ci"))
from check_distribution_gpu import current_batch_shapes


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def percentile(values, q):
    return sorted(values)[int(len(values) * q)]


def check_cell(path, receipt, plan, identity, cpu):
    if sha(path) != receipt["results"][path.stem]:
        raise ValueError(f"receipt does not bind {path}")
    data = read(path)
    stats, measured = data["statistics"], data["measured"]
    count = len(plan["schedule"])
    if (stats["requests"] != stats["ok"] or stats["ok"] != count or stats["errors"] or stats["rejected_429"]
            or stats["warmup_ok"] != plan["cell"]["warmup"] or stats["config"] != plan["cell"]
            or data["client_timing"]["cpu_affinity"] != [cpu] or data["length_mismatches"]):
        raise ValueError(f"incomplete or unmatched measurements in {path}")
    origin = min(row["sent_at"] for row in measured)
    for row, frozen in zip(measured, plan["schedule"], strict=True):
        expected_error = (row["sent_at"] - origin - frozen["offset_ns"] / 1e9) * 1e3
        if (any(row[key] != frozen[key] for key in identity) or row["scheduled_write_ns"] != frozen["offset_ns"]
                or not row["wake_at"] <= row["start"] <= row["sent_at"]
                or not 0 < row["ttft"] < row["e2e"] or not isclose(row["write_error_ms"], expected_error, abs_tol=0.000_05)):
            raise ValueError(f"invalid request body or timing in {path}")
    return measured, {key: stats[key] for key in METRICS}


def compare_cells(control, fifo, label, reg):
    baseline_rows, baseline = control
    candidate_rows, candidate = fifo
    delta = {key: 100 * (candidate[key] / baseline[key] - 1) for key in METRICS}
    errors = {
        "control": percentile([abs(row["write_error_ms"]) for row in baseline_rows], .95),
        "fifo": percentile([abs(row["write_error_ms"]) for row in candidate_rows], .95),
        "between": percentile([abs(a["write_error_ms"] - b["write_error_ms"])
                                for a, b in zip(baseline_rows, candidate_rows, strict=True)], .95),
    }
    output_changes = [a["request_index"] for a, b in zip(baseline_rows, candidate_rows, strict=True)
                      if a["output_sha256"] != b["output_sha256"]]
    gates = {"timing": max(errors.values()) <= reg["write_error_p95_limit_ms"], "output": not output_changes,
             "decode": delta["tpot_p50_ms"] < reg["decode_tpot_p50_max_regression_pct_exclusive"]}
    if label.startswith("c32-"):
        gates["short_ttft"] = delta["ttft_p50_ms"] < reg["short_ttft_p50_max_regression_pct_exclusive"]
        gates["short_throughput"] = delta["output_tok_s"] > -reg["short_throughput_max_drop_pct_exclusive"]
    return {"requests": len(candidate_rows), "abs_write_error_p95_ms": errors, "changed_output_indexes": output_changes,
            "control": baseline, "fifo": candidate, "delta_percent": delta, "gates": gates}


def batch_summary(path):
    shutdown = read(path)
    if (any(shutdown[key] for key in ("active_requests", "waiting_requests", "kv_allocated_blocks"))
            or any(shutdown["allocation"].values())):
        raise ValueError("server retained resources at shutdown")
    shapes = current_batch_shapes({"metrics": shutdown})
    wide_hits, wide_ns, _ = shapes[1024]
    return {"total_steps": sum(hits for hits, _, _ in shapes.values()), "wide_1024_steps": wide_hits,
            "wide_1024_gpu_time_ms": wide_ns / 1e6}


def verify(root):
    reg = read(HERE / "registration.json")
    for marker in ("build.exit", "campaign.exit"):
        if (root / marker).read_text().strip() != "0":
            raise ValueError(f"incomplete campaign: {marker}")
    for filename, key in (("build-dispatch.sha256", "build_dispatch_sha256"),
                          ("build-result.sha256", "build_result_sha256"),
                          ("campaign-dispatch.sha256", "campaign_dispatch_sha256"), ("schedules.json", "schedules_sha256")):
        if sha(root / filename) != reg[key]:
            raise ValueError(f"registered input differs: {filename}")
    binary, name = (root / "build-result.sha256").read_text().splitlines()[0].split()
    if name != "riley-diagnostic":
        raise ValueError("diagnostic build checksum is missing")
    sys.path[0:0] = [str(root), str(root / "tools")]
    config = runpy.run_path(str(HERE / "configure.py"))
    document = read(root / "schedules.json")
    with (tarfile.open(BASE / "initial-results.tar.gz", "r:gz") as fixed,
          tarfile.open(WAVE_SOURCE / "raw-results.tar.gz", "r:gz") as waves):
        expected = config["schedules"](json.load(fixed.extractfile("schedules.json")),
                                       lambda gpu, name: json.load(waves.extractfile(f"gpu{gpu}/chunk-1024/{name}.json")),
                                       json.load(waves.extractfile("cells.json")))
        holders = {gpu: json.load(fixed.extractfile(f"gpu{gpu}/after.json")) for gpu in ("6", "7")}
        old_arms = {gpu: json.load(fixed.extractfile(f"gpu{gpu}/coder14-riley/receipt.json")) for gpu in ("6", "7")}
        vllm = {label: json.load(fixed.extractfile(f"gpu7/coder14-vllm/{label}.json")) for label in config["FIXED"]}
    if document != expected or read(root / "cpu-placement.json") != {
            "server_cpus": config["SERVER_CPUS"], "client_cpus": config["CLIENT_CPUS"]}:
        raise ValueError("frozen workloads or CPU placement differ")
    log = (root / "campaign.log").read_text()
    events = re.findall(r"^(START|END) gpu([67]) (fifo|control) (\S+?)(?: exit=(\d+))?$", log, re.MULTILINE)
    results = {}
    for gpu, order in config["ORDER"].items():
        actual = [(kind, arm, code) for kind, index, arm, _, code in events if index == gpu]
        if actual != [(kind, arm, code) for arm in order for kind, code in (("START", ""), ("END", "0"))]:
            raise ValueError(f"run order or exit differs on GPU {gpu}")
        parent = root / f"gpu{gpu}"
        for phase in ("before", "between", "after"):
            row = read(parent / f"{phase}.json")
            if (row["physical_index"] != int(gpu) or row["phase"] != "idle" or row["other_processes"]
                    or row["allocation_bytes"] != row["torch_reserved_bytes"] or row["allocation_bytes"] != 20 * 2**30
                    or any(row[key] != holders[gpu][key] for key in ("pid", "start_ticks", "gpu"))):
                raise ValueError(f"GPU {gpu} did not return to its initial reservation")
        plans, receipts, holdouts, counts, batches = {}, {}, {}, {}, {}
        for mode in order:
            arm = parent / mode
            plan, receipt = read(arm / "plan.json"), read(arm / "receipt.json")
            keys = set(old_arms[gpu]) - {"riley_binary_sha256", "results"}
            if (receipt["riley_binary_sha256"] != binary or receipt["source_revision"] != reg["base_revision"]
                    or any(receipt[key] != old_arms[gpu][key] for key in keys)
                    or plan["cells"] != expected["schedules"][gpu] or plan["server_cpu_affinity"] != config["SERVER_CPUS"][gpu]
                    or f"RILEY_W3_PREFILL_FIFO={int(mode == 'fifo')}" not in plan["docker"]):
                raise ValueError(f"server identity or mode differs for GPU {gpu}/{mode}")
            holdout = arm / "token-id-holdout.json"
            block = read(holdout)
            if sha(holdout) != receipt["holdout_sha256"] or block.keys() != {"c1", "c8"} or any(len(v) != 8 for v in block.values()):
                raise ValueError(f"incomplete generated ID holdout for GPU {gpu}/{mode}")
            (count,) = re.findall(r"^W3-PREFILL-FIFO (\d+)$", (arm / "server.log").read_text(), re.MULTILINE)
            counts[mode] = int(count)
            if (counts[mode] > 0) != (mode == "fifo"):
                raise ValueError(f"scheduler switch did not change a mixed step for GPU {gpu}/{mode}")
            batches[mode] = batch_summary(arm / "shutdown.json")
            plans[mode], receipts[mode], holdouts[mode] = plan, receipt, block
        if plans["control"]["server"] != plans["fifo"]["server"]:
            raise ValueError(f"server flags changed between GPU {gpu} arms")
        cells = {}
        for label, plan in expected["schedules"][gpu].items():
            captures = [check_cell(parent / mode / f"{label}.json", receipts[mode], plan,
                                   config["MATCHED_KEYS"], config["CLIENT_CPUS"][gpu]) for mode in ("control", "fifo")]
            cells[label] = compare_cells(*captures, label, reg)
            if gpu == "7" and label in vllm:
                baselines = vllm[label]["measured"]
                against_vllm = {mode: percentile([abs(a["write_error_ms"] - b["write_error_ms"])
                                                  for a, b in zip(rows, baselines, strict=True)], .95)
                                for mode, (rows, _) in zip(("control", "fifo"), captures, strict=True)}
                metric = "ttft_p95_ms" if label.startswith("rag-") else "tpot_p50_ms"
                matched = cells[label]["gates"]["timing"] and all(error <= reg["write_error_p95_limit_ms"]
                                                                 for error in against_vllm.values())
                cells[label]["frozen_vllm_reference"] = {"metric": metric, "write_error_p95_ms": against_vllm,
                                                          "valid": matched,
                                                          "ratios": {mode: cells[label][mode][metric] / vllm[label]["statistics"][metric]
                                                                     for mode in ("control", "fifo")} if matched else {}}
        if sum(cell["requests"] for cell in cells.values()) != reg["measured_requests_per_arm"]:
            raise ValueError(f"GPU {gpu} workload coverage differs")
        results[gpu] = {"order": order, "reordered_steps": counts, "full_arm_batch_shapes": batches, "cells": cells,
                        "holdout_c1_c8_invariant": {mode: holdouts[mode]["c1"] == holdouts[mode]["c8"] for mode in order},
                        "holdout_same_across_policies": holdouts["control"] == holdouts["fifo"]}
    return {"base_revision": reg["base_revision"], "binary_sha256": binary, "gpus": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
