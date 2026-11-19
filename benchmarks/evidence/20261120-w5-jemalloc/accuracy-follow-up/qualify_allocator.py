"""Gate the published two-mode soak and qualify long output with the same allocator."""

from argparse import ArgumentParser
import json
from pathlib import Path
from unittest.mock import patch

import run_gpu_check as allocator
from verify_a100_candidate import (CHECKPOINT, GOLDENS, GPU, HOLDOUT, ROOTS, digest, gpu_evidence,
                                  read, require, validate_accuracy)
from verify_zerodecay import MODES, compare, holder, identity


def golden_inputs(evidence):
    files = evidence / ROOTS["base"] / "src/benchmarks/evidence"
    legacy = files / "20261119-w4-long/riley/capture.json.gz"
    workload = files / "20261119-w4-holdout/holdout.json.gz"
    golden = files / "20261119-w4-holdout/riley/capture.json.gz"
    require((digest(legacy), digest(workload), digest(golden))
            == (GOLDENS["baseline"], HOLDOUT, GOLDENS["holdout"]), "accuracy input role changed")
    return legacy, workload, golden


def decision(candidate, evidence):
    if (candidate / "campaign.exit").read_text().strip() != "0":
        raise ValueError("allocator soak campaign did not finish")
    replay = compare(candidate, evidence, MODES)
    saved = candidate / "verified-comparison.json"
    require(replay == read(saved), "the paired strict-soak receipt does not match replay")
    return {
        "run_accuracy": replay["both_modes_pass_strict_soak"], "identity": replay["identity"],
        "paired_comparison_sha256": digest(saved),
        "soak_modes": {mode: {key: replay["results"][mode][key]
                              for key in ("status", "failed_checks", "candidate_report_sha256")}
                       for mode in MODES},
    }


def check_reservations(status, time):
    return {
        "gpu6": holder(status / f"idle-{time}.json", 6, 2431228),
        "gpu7": holder(status / f"gpu7-{time}.json", 7, 2431237),
    }


def run_accuracy(candidate, evidence, stage):
    require(read(stage / "decision.json")["run_accuracy"], "the two strict soaks did not pass")
    inputs = golden_inputs(evidence)
    baseline, workload, golden = inputs
    original = allocator.checker.check_gpu
    base = evidence / ROOTS["base"]

    def check(*args, mixed_soak):
        require(mixed_soak is None, "accuracy wrapper received a mixed-soak mode")
        return original(*args, accuracy_baseline=baseline, accuracy_holdout=(workload, golden))

    with patch.object(allocator.checker, "check_gpu", check):
        return allocator.run(base / "cpu-evidence/report.json", GPU, base / "model", CHECKPOINT,
                             stage / "gpu-evidence", None, candidate)


def verify(candidate, evidence, stage):
    gate = decision(candidate, evidence)
    require(gate == read(stage / "decision.json") and gate["run_accuracy"], "accuracy gate changed")
    result = gpu_evidence(stage, gate["identity"], int((stage / "gpu-exit").read_text()))
    require(result["status"] == "passed", "allocator accuracy failed")
    validate_accuracy(stage, result)
    before, after = (read(stage / f"allocator-{side}.json") for side in ("before", "after"))
    require(before == after and before["image_id"] == gate["identity"]["image_id"]
            and before["library"] == allocator.LIBRARY and before["library_sha256"] == allocator.LIBRARY_SHA
            and before["malloc_conf"] == allocator.MALLOC_CONF and before["library_mapping_present"]
            and before["exec_helper_uses_default_environment"], "accuracy allocator differed from soak")
    reservations = check_reservations(stage, "after")
    require(reservations == check_reservations(stage, "before"), "GPU reservations changed during accuracy")
    return {"decision": gate, "accuracy_suites": result["accuracy_suites"], "api_checks": result["api_checks"],
            "allocator": before, "reservations_after": reservations, "shutdown_metrics": result["shutdown_metrics"],
            "accuracy_report_sha256": result["gpu_receipt_sha256"], "artifacts_sha256": result["artifacts_sha256"]}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inputs", "decision", "admission", "run", "verify"))
    parser.add_argument("candidate", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    if args.action == "inputs":
        golden_inputs(args.evidence)
        prior = args.evidence / ROOTS["accuracy"]
        require((prior / "campaign.exit").read_text().strip() == "0", "original accuracy did not finish")
        original = gpu_evidence(prior, identity(args.evidence), 0)
        validate_accuracy(prior, original)
        print(json.dumps({key: original["accuracy_suites"][key]["verified_requests"]
                          for key in ("baseline", "holdout")}))
    elif args.action == "admission":
        check_reservations(args.stage, "before")
    elif args.action == "run":
        raise SystemExit(run_accuracy(args.candidate, args.evidence, args.stage))
    else:
        proof = decision(args.candidate, args.evidence) if args.action == "decision" else verify(
            args.candidate, args.evidence, args.stage)
        print(json.dumps(proof, indent=2, allow_nan=False))
