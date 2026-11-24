"""Verify a four-arm contrast of prior RAG activity versus equally long idle."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import runpy
import tarfile
from tempfile import TemporaryDirectory


PARENT = "20261124-w3-aged-c32-observation"
NATIVE = "20261124-w3-aged-decode-native"
BASELINE = "20261124-w3-rag-prefill-order"
SHORT = "c32-riley-arrivals"
RAG = ("rag-c8-s1-vllm-arrivals", "rag-c8-s2-vllm-arrivals")
MODES = ("control", "reserve")
ORDER = {
    "6": (("idle", "control"), ("rag", "reserve"), ("rag", "control"), ("idle", "reserve")),
    "7": (("idle", "reserve"), ("rag", "control"), ("rag", "reserve"), ("idle", "control")),
}
MANIFESTS = ("dispatch.sha256", "runtime.sha256", "result.sha256")
RECEIPTS = {"campaign.exit", "campaign.pid", "campaign.started", "campaign.log", *MANIFESTS}
MEMBERSHIP = re.compile(r"([a-f0-9]{64})  ((?:[A-Za-z0-9][A-Za-z0-9_.-]*/)*[A-Za-z0-9][A-Za-z0-9_.-]*)")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_archive(archive, root, registration):
    """Accept only regular files listed by a captured checksum or process receipt."""
    with tarfile.open(archive, "r:gz") as bundle:
        names = set()
        for member in bundle:
            if not member.isfile() or member.name in names:
                raise ValueError("archive contains nonregular or duplicate entries")
            names.add(member.name)
        bundle.extractall(root, filter="data")
    if digest(root / "dispatch.sha256") != registration["campaign_dispatch_sha256"]:
        raise ValueError("launch dispatch fingerprint does not match")
    expected = set(RECEIPTS)
    archived_binary = False
    for manifest in MANIFESTS:
        declared = set()
        for line in (root / manifest).read_text().splitlines():
            match = MEMBERSHIP.fullmatch(line)
            if match is None:
                raise ValueError(f"invalid registered path in {manifest}")
            fingerprint, name = match.groups()
            if name in declared:
                raise ValueError(f"duplicate registered path in {manifest}: {name}")
            declared.add(name)
            if manifest == "runtime.sha256" and name == "riley-diagnostic":
                archived_binary = True
                if fingerprint != registration["diagnostic_binary_sha256"] or name in names:
                    raise ValueError("diagnostic binary fingerprint differs")
            elif digest(root / name) != fingerprint:
                raise ValueError(f"captured file differs: {name}")
            else:
                expected.add(name)
    if not archived_binary or names != expected:
        raise ValueError("archive membership differs from checksum manifests")


def check_history(path, labels, registration):
    record = read(path)
    launches, finishes = record["launch_s"], record["finish_s"]
    targets = (record["short_start_target_s"], record["rag_start_target_s"])
    if (targets != (registration["short_start_target_s"], registration["rag_start_target_s"])
            or tuple(launches) != labels or tuple(finishes) != labels):
        raise ValueError("history workload or wait target differs")
    previous = 0
    for name in labels:
        if not previous <= launches[name] < finishes[name]:
            raise ValueError(f"invalid replay timeline in {path}")
        previous = finishes[name]
    tolerance = registration["history_release_tolerance_s"]
    return {
        "c32_release_s": launches[SHORT],
        "release_timed": abs(launches[SHORT] - targets[0]) <= tolerance,
        "rag_finish_s": finishes[RAG[-1]] if labels[0] == RAG[0] else None,
    }


def evaluate_arm(path, previous, reg, history, mode, cpu, binary, baseline, native, observe):
    old_plan, old_receipt, old_tokens = [previous(name) for name in ("plan", "receipt", "token-id-holdout")]
    labels = (SHORT,) if history == "idle" else (*RAG, SHORT)
    if tuple(old_plan["cells"]) != (*RAG, SHORT):
        raise ValueError("published parent workload differs")
    old_plan["cells"] = {label: old_plan["cells"][label] for label in labels}
    workload, tokens, shape = native["verify_arm"](
        path, old_plan, old_receipt, reg, binary, mode, cpu, baseline, observations=True,
    )
    captures = {label: read(path / f"{label}.json") for label in labels}
    count, phases = observe["bind_observations"](
        (path / "server.log").read_text(), captures, reg["holdout_requests_per_arm"],
    )
    if count != sum(len(data["measured"]) + len(data["warmup_response_ids"]) for data in captures.values()) + 16:
        raise ValueError("server observation coverage differs")
    for name, (rows, _) in workload.items():
        reference = previous(name)["measured"]
        expected = [(item["request_index"], item["output_sha256"]) for item in reference]
        if [(item["request_index"], item["output_sha256"]) for item in rows] != expected:
            raise ValueError(f"response output differs from the published baseline in {path}/{name}")
    if tokens != old_tokens or tokens["c1"] != tokens["c8"]:
        raise ValueError(f"token-ID holdout differs from the parent in {path}")
    return workload, phases[SHORT], {
        "server_observations": count, "measured_requests": sum(len(rows) for rows, _ in workload.values()),
        "batch_shapes": shape, "schedule": check_history(path / "history.json", labels, reg),
    }


def evaluate_gpu(root, parent, gpu, baseline, native, observe, reg, binary, cpu):
    old_holder = json.load(parent.extractfile(f"gpu{gpu}/after.json"))
    for name in ("before", "before-rag", "between-rag", "before-last", "after"):
        holder = read(root / f"gpu{gpu}/{name}.json")
        if (holder["phase"] != "idle" or holder["other_processes"] or
                holder["allocation_bytes"] != 20 * 2**30 or holder["torch_reserved_bytes"] != 20 * 2**30 or
                any(holder[k] != old_holder[k] for k in ("gpu", "physical_index", "pid", "start_ticks"))):
            raise ValueError(f"initial reservation did not return on GPU {gpu}/{name}")
    def previous(name):
        return json.load(parent.extractfile(f"gpu{gpu}/control/{name}.json"))

    arms = {
        (history, mode): evaluate_arm(
            root / f"gpu{gpu}/{history}/{mode}", previous, reg, history, mode, cpu, binary, baseline, native, observe,
        )
        for history, mode in ORDER[gpu]
    }
    results = {}
    for history in ("idle", "rag"):
        (a, a_phases, a_summary), (b, b_phases, b_summary) = [arms[history, mode] for mode in MODES]
        cells = {}
        for label, workload in a.items():
            cell = baseline["compare_cells"](workload, b[label], label, reg)
            cell["reserve"] = cell.pop("fifo")
            errors = cell["abs_write_error_p95_ms"]
            errors["reserve"] = errors.pop("fifo")
            cells[label] = cell
        results[history] = {
            "cells": cells, "arms": {"control": a_summary, "reserve": b_summary},
            "phases": observe["phase_effect"](a_phases, b_phases, baseline["percentile"]),
        }

    idle, rag = results["idle"], results["rag"]
    cross_history = {
        mode: baseline["compare_cells"](arms["idle", mode][0][SHORT], arms["rag", mode][0][SHORT], SHORT, reg)
        for mode in MODES
    }
    results["rag_minus_idle_contrast"] = {
        "send_timing_matched": all(history["cells"][SHORT]["gates"]["timing"] for history in (idle, rag))
        and all(cell["gates"]["timing"] for cell in cross_history.values()),
        "server_age_matched": all(
            arm["schedule"]["release_timed"] for history in (idle, rag) for arm in history["arms"].values()
        ),
        "cross_history_send_error_p95_ms": {mode: cell["abs_write_error_p95_ms"] for mode, cell in cross_history.items()},
        "effect_ms": {
            "ttft_p50": (
                rag["cells"][SHORT]["reserve"]["ttft_p50_ms"] - rag["cells"][SHORT]["control"]["ttft_p50_ms"]
                - (idle["cells"][SHORT]["reserve"]["ttft_p50_ms"] - idle["cells"][SHORT]["control"]["ttft_p50_ms"])
            ),
            "median_paired_admission": (
                rag["phases"]["paired_delta_p50_ms"]["backend_admission_ms"]
                - idle["phases"]["paired_delta_p50_ms"]["backend_admission_ms"]
            ),
        },
    }
    return results


def verify(archive, registration, evidence, root):
    reg = read(registration)
    parent_path = evidence / PARENT
    parent_registration = read(parent_path / "registration.json")
    parent_archive = parent_path / "raw-results.tar.gz"
    binary = (parent_path / "build-result.sha256").read_text().split()[0]
    if (reg["parent_stage"] != parent_registration["stage"] or reg["parent_archive_sha256"] != digest(parent_archive)
            or reg["diagnostic_binary_sha256"] != binary or
            any(reg[key] != parent_registration[key] for key in (
                "source_revision", "package_image_id", "package_cpu_report_sha256", "mode_flag",
                "schedules_sha256", "holdout_requests_per_arm", "write_error_p95_limit_ms",
                "decode_tpot_p50_max_regression_pct_exclusive", "short_ttft_p50_max_regression_pct_exclusive",
                "short_throughput_max_drop_pct_exclusive",
            ))):
        raise ValueError("registered source, binary, workload or gates differ from parent")
    check_archive(archive, root, reg)
    if ((root / "campaign.exit").read_text().strip() != "0" or
            int((root / "campaign.pid").read_text()) != reg["campaign_pid"] or
            (root / "campaign.started").read_text().strip() != reg["campaign_started_at"]):
        raise ValueError("campaign identity or exit differs")
    events = re.findall(
        r"^(START|END) gpu([67]) (idle|rag) (reserve|control) \S+(?: exit=(\d+))?$",
        (root / "campaign.log").read_text(), re.MULTILINE,
    )
    for gpu, runs in ORDER.items():
        expected = [(action, history, mode, code) for history, mode in runs
                    for action, code in (("START", ""), ("END", "0"))]
        if [(action, history, mode, code) for action, index, history, mode, code in events if index == gpu] != expected:
            raise ValueError(f"registered order or exits differ on GPU {gpu}")

    baseline = runpy.run_path(str(evidence / BASELINE / "verify.py"))
    native = runpy.run_path(str(evidence / NATIVE / "verify.py"))
    observe = runpy.run_path(str(parent_path / "verify.py"))
    with tarfile.open(parent_archive, "r:gz") as parent:
        for name in ("schedules.json", "cpu-placement.json", "replay.py", "holder.py", "holdout.py"):
            if (root / name).read_bytes() != parent.extractfile(name).read():
                raise ValueError(f"frozen workload or harness changed: {name}")
        placement = read(root / "cpu-placement.json")
        gpus = {
            gpu: evaluate_gpu(root, parent, gpu, baseline, native, observe, reg, binary, cpu)
            for gpu, cpu in placement["client_cpus"].items()
        }
    return {"diagnostic_binary_sha256": binary, "gpus": gpus}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    with TemporaryDirectory() as work:
        print(json.dumps(verify(args.archive, args.registration, args.evidence_root, Path(work)), indent=2, allow_nan=False))
