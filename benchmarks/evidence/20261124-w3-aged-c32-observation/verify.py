"""Join the C32 client replay to request-level server observations."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
import re
import runpy
import tarfile


NATIVE = "20261124-w3-aged-decode-native"
REFERENCE = "20261124-w3-aged-native-short"
REQUEST = re.compile(r"cmpl-[a-f0-9]{16}")
PHASES = ("client_before_write_ms", "backend_admission_ms", "admitted_to_first_visible_ms", "other_after_write_ms")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def bind_observations(log, captures, holdout_count):
    table = {}
    snapshots = []
    for line in log.splitlines():
        if line.startswith("W3-OBS-SNAPSHOT "):
            _, retained, dropped = line.split()
            snapshots.append((int(retained), int(dropped)))
        elif line.startswith("W3-OBS "):
            _, request, admission, first, tokens, status = line.split()
            if not REQUEST.fullmatch(request) or request in table or status != "Completed":
                raise ValueError(f"unexpected server observation for {request}")
            table[request] = tuple(map(int, (admission, first, tokens)))
    if snapshots != [(len(table), 0)]:
        raise ValueError("server observation snapshot was dropped or incomplete")
    count = len(table)
    result = {}
    for label, capture in captures.items():
        for request in capture["warmup_response_ids"]:
            if request not in table:
                raise ValueError(f"warmup request {request} has no server observation")
            table.pop(request)
        rows = []
        for row in capture["measured"]:
            request = row["response_id"]
            if request not in table:
                raise ValueError(f"measured request {request} has no unique server observation")
            admission, first, tokens = table.pop(request)
            total = row["ttft"] * 1000
            before = (row["sent_at"] - row["start"]) * 1000
            admit, visible = admission / 1e6, first / 1e6
            if (not all(map(isfinite, (total, before, admit, visible))) or not 0 <= before <= total
                    or not (admission > 0 and first > 0 and admit + visible <= total)
                    or tokens != row["completion_tokens"]):
                raise ValueError(f"client and server timing or usage differs for {request}")
            rows.append({
                "request_index": row["request_index"], "response_id": request, "ttft_ms": total,
                "client_before_write_ms": before, "backend_admission_ms": admit,
                "admitted_to_first_visible_ms": visible, "other_after_write_ms": total - before - admit - visible,
            })
        result[label] = rows
    if len(table) != holdout_count:
        raise ValueError("extra observations do not match the registered token ID holdout")
    return count, result


def phase_effect(control, reserve, p):
    if [row["request_index"] for row in control] != [row["request_index"] for row in reserve]:
        raise ValueError("phase comparison reordered the measured requests")
    rank = int(len(control) * .5)
    before_rank = sorted(control, key=lambda row: row["ttft_ms"])[rank]
    after_rank = sorted(reserve, key=lambda row: row["ttft_ms"])[rank]
    same_request = next(row for row in control if row["request_index"] == after_rank["request_index"])
    phase_deltas = {key: after_rank[key] - same_request[key] for key in PHASES}
    rank_shift = same_request["ttft_ms"] - before_rank["ttft_ms"]
    if not isclose(after_rank["ttft_ms"] - before_rank["ttft_ms"], rank_shift + sum(phase_deltas.values()),
                   rel_tol=0, abs_tol=1e-8):
        raise ValueError("phases and rank shift do not account for the p50 change")
    return {
        "control_p50_request": before_rank, "reserve_p50_request": after_rank,
        "control_request_at_reserve_rank": same_request, "p50_request_phase_delta_ms": phase_deltas,
        "control_rank_shift_ms": rank_shift,
        "paired_delta_p50_ms": {
            key: p([b[key] - a[key] for a, b in zip(control, reserve, strict=True)], .5)
            for key in ("ttft_ms", *PHASES)
        },
        "p50_replacing_one_measured_phase_ms": {
            key: p([b["ttft_ms"] - b[key] + a[key] for a, b in zip(control, reserve, strict=True)], .5)
            for key in PHASES
        },
    }


def verify(root, registration, evidence):
    reg = read(registration)
    reference = evidence / REFERENCE
    native = evidence / NATIVE
    prior = read(reference / "registration.json")
    native_tool = runpy.run_path(str(native / "verify.py"))
    matching = (*native_tool["GATES"], "cells", "candidate_patch_sha256", "mode_flag",
                "source_revision", "package_image_id", "package_cpu_report_sha256")
    if (any(reg[name] != prior[name] for name in matching) or reg["parent_stage"] != prior["stage"]
            or reg["parent_build_result_sha256"] != prior["build_result_sha256"]):
        raise ValueError("diagnostic workload or parent differs from the short retry")
    for path, name in (
        ("build-dispatch.sha256", "build_dispatch_sha256"), ("build-result.sha256", "build_result_sha256"),
        ("campaign-dispatch.sha256", "campaign_dispatch_sha256"), ("schedules.json", "schedules_sha256"),
        ("export-observations.patch", "export_observations_sha256"),
    ):
        if digest(root / path) != reg[name]:
            raise ValueError(f"diagnostic artifact differs from registration: {path}")
    for path, parent, receipt in (
        ("native-build.sha256", native, "build-result.sha256"),
        ("short-result.sha256", reference, "result.sha256"),
    ):
        if (root / path).read_text() != f"{digest(parent / receipt)}  {receipt}\n":
            raise ValueError(f"diagnostic predecessor fingerprint differs: {path}")
    for phase in ("build", "campaign"):
        if ((root / f"{phase}.exit").read_text().strip() != "0"
                or int((root / f"{phase}.pid").read_text()) != reg[f"{phase}_pid"]
                or (root / f"{phase}.started").read_text().strip() != reg[f"{phase}_started_at"]):
            raise ValueError(f"process exit or registered identity differs: {phase}")
    binary, name = (root / "build-result.sha256").read_text().splitlines()[0].split()
    old_binary = (native / "build-result.sha256").read_text().split()[0]
    old_sources = (native / "patched-source.sha256").read_text().splitlines()
    new_sources = (root / "patched-source.sha256").read_text().splitlines()
    if (name != "riley-diagnostic" or binary == old_binary or new_sources[:len(old_sources)] != old_sources
            or len(new_sources) != len(old_sources) + 1 or not new_sources[-1].endswith("  crates/riley-server/src/service.rs")):
        raise ValueError("diagnostic build changed native patched inputs besides the observation exporter")

    baseline = runpy.run_path(str(evidence / "20261124-w3-rag-prefill-order/verify.py"))
    events = re.findall(r"^(START|END) gpu([67]) (reserve|control) (\S+?)(?: exit=(\d+))?$",
                        (root / "campaign.log").read_text(), re.MULTILINE)
    gpus = {}
    with tarfile.open(reference / "raw-results.tar.gz", "r:gz") as gold:
        for path in ("schedules.json", "cpu-placement.json"):
            if (root / path).read_bytes() != gold.extractfile(path).read():
                raise ValueError(f"diagnostic replay changed {path}")
        placement = read(root / "cpu-placement.json")
        for gpu, order in reg["orders"].items():
            expected = [(kind, mode, code) for mode in order for kind, code in (("START", ""), ("END", "0"))]
            if [(kind, mode, code) for kind, index, mode, _, code in events if index == gpu] != expected:
                raise ValueError(f"invalid diagnostic arm order or exit on GPU {gpu}")
            old_holder = json.load(gold.extractfile(f"gpu{gpu}/after.json"))
            for path in (root / f"gpu{gpu}-staged-start.json",
                         *(root / f"gpu{gpu}/{phase}.json" for phase in ("before", "between", "after"))):
                holder = read(path)
                if (holder["phase"] != "idle" or holder["other_processes"] or holder["physical_index"] != int(gpu)
                        or holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30
                        or any(holder[key] != old_holder[key] for key in ("gpu", "pid", "start_ticks"))):
                    raise ValueError(f"original GPU holder differs at {path}")
            old_plan, old_receipt, old_tokens = (json.load(gold.extractfile(f"gpu{gpu}/control/{file}.json"))
                                                  for file in ("plan", "receipt", "token-id-holdout"))
            workloads, arms, phases = {}, {}, {}
            for mode in order:
                arm = root / f"gpu{gpu}/{mode}"
                workload, tokens, shape = native_tool["verify_arm"](
                    arm, old_plan, old_receipt, reg, binary, mode, placement["client_cpus"][gpu], baseline,
                    observations=True,
                )
                captures = {label: read(arm / f"{label}.json") for label in workload}
                count, phases[mode] = bind_observations(
                    (arm / "server.log").read_text(), captures, reg["holdout_requests_per_arm"],
                )
                workloads[mode] = workload
                arms[mode] = {
                    "server_observations": count, "batch_shapes": shape,
                    "holdout_matches_c1_c8": tokens["c1"] == tokens["c8"],
                    "holdout_matches_previous": tokens == old_tokens,
                }
            cells = {}
            c32 = "c32-riley-arrivals"
            for label, before in workloads["control"].items():
                cell = baseline["compare_cells"](before, workloads["reserve"][label], label, reg)
                cell["reserve"] = cell.pop("fifo")
                errors = cell["abs_write_error_p95_ms"]
                errors["reserve"] = errors.pop("fifo")
                cell["against_previous_same_mode"] = {}
                for mode in order:
                    old = json.load(gold.extractfile(f"gpu{gpu}/{mode}/{label}.json"))
                    comparison = baseline["compare_cells"](
                        (old["measured"], old["statistics"]), workloads[mode][label], label, reg,
                    )
                    cell["against_previous_same_mode"][mode] = {
                        "output_match": comparison["gates"]["output"],
                        "timing_valid": comparison["gates"]["timing"],
                        "delta_percent": comparison["delta_percent"],
                    }
                cells[label] = cell
            if sum(cell["requests"] for cell in cells.values()) != reg["measured_requests_per_arm"]:
                raise ValueError(f"diagnostic request coverage differs on GPU {gpu}")
            effect = phase_effect(phases["control"][c32], phases["reserve"][c32], baseline["percentile"])
            for mode in order:
                rank = effect[f"{mode}_p50_request"]["ttft_ms"]
                if not isclose(rank, cells[c32][mode]["ttft_p50_ms"], rel_tol=0, abs_tol=1e-7):
                    raise ValueError("phase analysis does not reproduce the reported C32 median")
            gpus[gpu] = {"order": order, "arms": arms, "cells": cells, "c32_p50_phases": effect,
                         "c32_observations": {mode: phases[mode][c32] for mode in order}}
    return {"source_revision": reg["source_revision"], "image_id": reg["package_image_id"],
            "diagnostic_binary_sha256": binary, "native_binary_sha256": old_binary, "gpus": gpus}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.capture, args.registration, args.evidence_root), indent=2, allow_nan=False))
