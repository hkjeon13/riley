"""Join the frozen RAG clients to the observations collected at server shutdown."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

from configure import LABELS


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "20261124-w3-separated-client"
REQUEST = re.compile(r"cmpl-[a-f0-9]{16}")
PHASES = ("client_before_write_ms", "backend_submit_ms", "admitted_to_first_visible_ms", "other_after_write_ms")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def p(values, percentile):
    return sorted(values)[int(len(values) * percentile)]


def observations(path):
    records = {}
    snapshots = []
    for line in path.read_text().splitlines():
        if line.startswith("W3-OBS-SNAPSHOT "):
            snapshots.append(tuple(map(int, line.split()[1:])))
        elif line.startswith("W3-OBS "):
            _, request, admit, visible, tokens, status = line.split()
            if not REQUEST.fullmatch(request) or request in records or status != "Completed":
                raise ValueError(f"invalid or duplicate server observation for {request}")
            records[request] = tuple(map(int, (admit, visible, tokens)))
    if snapshots != [(144, 0)] or len(records) != 144:
        raise ValueError("expected all measured and warmup observations without evictions")
    return records


def analyze_cell(data, riley_reference, vllm_reference, focus, records):
    if len(data["warmup_response_ids"]) != 8 or data["client_timing"]["cpu_affinity"] != [45]:
        raise ValueError("unexpected client affinity or warmup count")
    for request in data["warmup_response_ids"]:
        records.pop(request)
    result = []
    errors = {key: [] for key in ("diagnostic", "against_shipped_riley", "against_vllm")}
    for row, prior, reference in zip(data["measured"], riley_reference["measured"], vllm_reference["measured"], strict=True):
        keys = ("request_index", "input_sha256", "scheduled_write_ns", "prompt_tokens", "completion_tokens", "finish")
        if any(row[key] != prior[key] or row[key] != reference[key] for key in keys):
            raise ValueError(f"frozen schedule or token count changed for request {row['request_index']}")
        admit, visible, tokens = records.pop(row["response_id"])
        ttft = row["ttft"] * 1e3
        prewrite = (row["sent_at"] - row["start"]) * 1e3
        subtotal = (admit + visible) / 1e6
        remainder = ttft - prewrite - subtotal
        if tokens != row["completion_tokens"] or admit <= 0 or visible <= 0 or prewrite < 0 or remainder < -0.05:
            raise ValueError(f"inconsistent client and server phases for request {row['request_index']}")
        errors["diagnostic"].append(abs(row["write_error_ms"]))
        errors["against_shipped_riley"].append(abs(row["write_error_ms"] - prior["write_error_ms"]))
        errors["against_vllm"].append(abs(row["write_error_ms"] - reference["write_error_ms"]))
        result.append({"request_index": row["request_index"], "response_id": row["response_id"],
                       "prompt_tokens": row["prompt_tokens"], "client_ttft_ms": ttft,
                       "client_before_write_ms": prewrite, "backend_submit_ms": admit / 1e6,
                       "admitted_to_first_visible_ms": visible / 1e6, "other_after_write_ms": remainder,
                       "shipped_riley_ttft_ms": prior["ttft"] * 1e3, "vllm_ttft_ms": reference["ttft"] * 1e3,
                       "riley_post_admission_minus_vllm_total_ms": visible / 1e6 - reference["ttft"] * 1e3,
                       "output_matches_shipped_riley": row["output_sha256"] == prior["output_sha256"]})
    if len(result) != 64 or len({row["request_index"] for row in result}) != 64:
        raise ValueError("expected 64 distinct measured requests")
    write_p95 = {key: p(values, .95) for key, values in errors.items()}
    preserved = sum(row["output_matches_shipped_riley"] for row in result)
    tail = sorted(result, key=lambda row: row["client_ttft_ms"])[-4:]
    p95_ms = tail[0]["client_ttft_ms"]
    if p95_ms != data["statistics"]["ttft_p95_ms"]:
        raise ValueError("recomputed p95 differs from the diagnostic client")
    return {"requests": len(result), "matched_output_hashes": preserved, "write_error_p95_ms": write_p95,
            "valid_against_frozen_vllm": preserved == len(result) and max(write_p95.values()) <= 3,
            "diagnostic_ttft_p95_ms": p95_ms,
            "shipped_riley_ttft_p95_ms": riley_reference["statistics"]["ttft_p95_ms"],
            "frozen_vllm_ttft_p95_ms": vllm_reference["statistics"]["ttft_p95_ms"],
            "phase_p50_ms": {key: p([row[key] for row in result], .50) for key in PHASES},
            "target_request": next(row for row in result if row["request_index"] == focus),
            "diagnostic_top_four": tail, "all_requests": result}


def verify(root):
    reg = read(HERE / "registration.json")
    if any((root / marker).read_text().strip() != "0" for marker in ("campaign.exit", "build.exit")):
        raise ValueError("build or replay did not complete")
    for name, key in (("build-dispatch.sha256", "build_dispatch_sha256"),
                      ("campaign-dispatch.sha256", "campaign_dispatch_sha256")):
        if sha(root / name) != reg[key]:
            raise ValueError(f"the registered {name} differs")
    toolchain = (root / "toolchain.txt").read_text()
    if not toolchain.startswith("rustc 1.85.0") or "V12.4." not in toolchain:
        raise ValueError("diagnostic toolchain differs")
    expected_binary, name = (root / "build-result.sha256").read_text().splitlines()[0].split()
    arm = root / "gpu7/coder14-riley"
    receipt = read(arm / "receipt.json")
    if name != "riley-diagnostic" or receipt["riley_binary_sha256"] != expected_binary:
        raise ValueError("diagnostic was not served with the hashed binary")
    records = observations(arm / "server.log")
    with tarfile.open(PREVIOUS / "initial-results.tar.gz", "r:gz") as baseline:
        old = lambda name: json.load(baseline.extractfile(name))
        old_arm = "gpu7/coder14-riley"
        old_receipt = old(f"{old_arm}/receipt.json")
        compare_keys = set(old_receipt) - {"riley_binary_sha256", "results"}
        if expected_binary == old_receipt["riley_binary_sha256"] or any(receipt[k] != old_receipt[k] for k in compare_keys):
            raise ValueError("diagnostic server identity differs from the frozen arm")
        if receipt["source_revision"] != reg["source_revision"]:
            raise ValueError("diagnostic base source is not the registered package")
        before = old("gpu7/after.json")
        for phase in ("before", "after"):
            holder = read(root / f"gpu7/{phase}.json")
            if (any(holder[key] != before[key] for key in ("pid", "start_ticks", "gpu"))
                    or holder["physical_index"] != 7 or holder["phase"] != "idle" or holder["other_processes"]
                    or holder["allocation_bytes"] != holder["torch_reserved_bytes"] or holder["allocation_bytes"] != 20 * 2**30):
                raise ValueError("GPU 7 reservation differs")
        plan = read(arm / "plan.json")
        prior_plan = old(f"{old_arm}/plan.json")
        if (plan["server"] != prior_plan["server"] or plan["server_cpu_affinity"] != prior_plan["server_cpu_affinity"]
                or plan["cells"] != {label: prior_plan["cells"][label] for label in LABELS}):
            raise ValueError("serving flags, CPU affinity or matched request schedule differs")
        results = {}
        for seed, label in enumerate(LABELS, start=1):
            path = arm / f"{label}.json"
            if receipt["results"][label] != sha(path):
                raise ValueError(f"receipt does not bind {label}")
            results[label] = analyze_cell(read(path), old(f"{old_arm}/{label}.json"),
                                          old(f"gpu7/coder14-vllm/{label}.json"),
                                          reg["focus_requests_by_seed"][str(seed)], records)
    if records:
        raise ValueError("unmatched server observations remain")
    return {"source_revision": receipt["source_revision"], "gpu": 7, "completed_observations": 144,
            "diagnostic_binary_sha256": expected_binary, "cells": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    print(json.dumps(verify(parser.parse_args().stage), indent=2, allow_nan=False))
