"""Gate a same-image long-context run on the framing candidate's strict RSS pair."""

from argparse import ArgumentParser
import json
from pathlib import Path

from verify_a100_candidate import CHECKPOINT, GPU, RILEY, REVISION, digest, gpu_evidence, require, validate_accuracy

MODES = {"cache-on", "cache-off"}


def read(path):
    return json.loads(path.read_bytes())


def image(root):
    cpu_path = root / "cpu-evidence/report.json"
    cpu = read(cpu_path)
    require(cpu["schema_version"] == "riley.distribution-image-check.v1" and cpu["status"] == "passed"
            and cpu["source_revision"] == REVISION, "unverified framing image")
    return {"image_id": cpu["image_id"], "source_revision": cpu["source_revision"],
            "source_archive_sha256": cpu["source_archive_sha256"], "riley_binary_sha256": cpu["binaries"][RILEY],
            "cpu_report_sha256": digest(cpu_path), "checkpoint_sha256": CHECKPOINT, "gpu_uuid": GPU}


def gate(root):
    require((root / "campaign.exit").read_text().strip() == "0", "paired W5 workload did not complete")
    identity = image(root)
    ready = read(root / "cpu-comparison.json")["image"]
    require(all(ready[f"candidate_{key}"] == identity[key] for key in
                ("image_id", "source_archive_sha256", "cpu_report_sha256"))
            and ready["candidate_binary_sha256"] == identity["riley_binary_sha256"], "package receipt differs")
    complete = read(root / "cache-off/comparison.json")
    decisions = complete["modes"]
    require(decisions.keys() == MODES and complete["image"] == ready, "incomplete candidate comparison")
    verdicts = {}
    for mode in sorted(MODES):
        folder = root / mode
        report = read(folder / "gpu-evidence/report.json")
        measured = report["mixed_soak"]
        result = decisions[mode]
        exit_code = int((folder / "exit").read_text())
        require(all(report[key] == value for key, value in identity.items())
                and measured["mode"] == mode and measured["status"] == report["status"]
                and measured["failed_checks"] == result["candidate_failed_checks"]
                and report["status"] == result["candidate_status"] == ("failed" if measured["failed_checks"] else "passed")
                and exit_code == int(report["status"] == "failed")
                and result["candidate_report_sha256"] == digest(folder / "gpu-evidence/report.json")
                and result["same_20gib_reservation_restored"] is True
                and result["identical_outcome_counts"] is True,
                f"{mode}: candidate or soak result changed")
        verdicts[mode] = {"status": report["status"], "failed_checks": measured["failed_checks"],
                          "report_sha256": result["candidate_report_sha256"]}
    eligible = all(mode["status"] == "passed" for mode in verdicts.values())
    return {"run_accuracy": eligible, "image": identity, "soak_modes": verdicts}


def verify(root, framing):
    decision = gate(framing)
    require(decision["run_accuracy"], "framing soak did not allow an accuracy run")
    candidate = gpu_evidence(root, decision["image"], 0)
    validate_accuracy(root, candidate)
    suites = {name: item["verified_requests"] for name, item in candidate["accuracy_suites"].items()}
    return {"scope": "A100 package and pinned long-context consistency for the W5 diagnostic image.",
            **decision, "api_cases": candidate["api_checks"], "verified_requests": sum(suites.values()),
            "suites": suites, "same_20gib_reservation_restored": True,
            "accuracy_report_sha256": candidate["gpu_receipt_sha256"],
            "accuracy_artifact_sha256": {
                path: candidate["artifacts_sha256"][path] for path in (
                    "accuracy-baseline.json", "accuracy-baseline-comparison.json",
                    "accuracy-holdout.json", "accuracy-holdout-comparison.json",
                )
            }}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("framing", type=Path)
    parser.add_argument("--accuracy", type=Path)
    args = parser.parse_args()
    result = gate(args.framing) if args.accuracy is None else verify(args.accuracy, args.framing)
    print(json.dumps(result, indent=2, allow_nan=False))
