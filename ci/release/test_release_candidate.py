#!/usr/bin/env python3
"""CPU-only tests for the final release-candidate evidence gate."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from build_release_bundle import build_bundle  # noqa: E402
from check_release_candidate import (  # noqa: E402
    ATTESTATION_VERSION,
    CUDA_FAULT_CHECKS,
    MANIFEST_VERSION,
    PYTHON_FREE_CHECKS,
    evaluate,
)
from test_release import EPOCH, fixture_elf  # noqa: E402


REVISION = "1a2b3c4d5e6f78901234567890abcdef12345678"


def digest(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


class CandidateFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "LicenseRef-Test-Fixture"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_text(
            "Owner-approved fixture license for release contract tests.\n"
            "Permission is granted only inside this temporary unit-test fixture.\n",
            encoding="utf-8",
        )
        self.paths = {
            "source": root / "source.tar",
            "binary": root / "rustinfer",
            "bundle": root / "rustinfer.tar.gz",
            "python_raw": root / "python-free-evidence.tar",
            "cuda_raw": root / "cuda-fault-evidence.tar",
            "python_report": root / "python-free-report.json",
            "cuda_report": root / "cuda-fault-report.json",
            "correctness": root / "correctness-report.json",
            "performance": root / "performance-report.json",
            "soak": root / "soak-report.json",
        }
        self.paths["source"].write_bytes(b"exact source archive fixture")
        self.paths["binary"].write_bytes(fixture_elf())
        self.paths["binary"].chmod(0o755)
        build_bundle(
            binary_path=self.paths["binary"],
            output=self.paths["bundle"],
            repository_root=repository,
            source_revision=REVISION,
            source_date_epoch=EPOCH,
        )
        self.paths["python_raw"].write_bytes(b"python-free raw evidence")
        self.paths["cuda_raw"].write_bytes(b"cuda fault raw evidence")
        self.image_sha = digest(b"release image")
        self.documents: dict[str, dict[str, object]] = {}
        self._build_documents()
        self.write_reports()
        self.manifest_path = root / "release-candidate.json"
        self.refresh_manifest()

    def _binding(self) -> dict[str, object]:
        return {
            "git_revision": REVISION,
            "git_dirty": False,
            "source_archive_sha256": digest(self.paths["source"].read_bytes()),
            "release_binary_sha256": digest(self.paths["binary"].read_bytes()),
            "release_bundle_sha256": digest(self.paths["bundle"].read_bytes()),
            "release_image_sha256": self.image_sha,
        }

    def _attestation(self, gate: str, raw: str, checks: set[str]) -> dict[str, object]:
        return {
            "schema_version": ATTESTATION_VERSION,
            "gate": gate,
            "status": "passed",
            "source": self._binding(),
            "raw_evidence_sha256": digest(self.paths[raw].read_bytes()),
            "checks": [{"id": check, "passed": True} for check in sorted(checks)],
        }

    def _build_documents(self) -> None:
        self.documents["python_report"] = self._attestation(
            "python-free-clean-runtime-e2e", "python_raw", PYTHON_FREE_CHECKS
        )
        self.documents["cuda_report"] = self._attestation(
            "cuda-fault-injection", "cuda_raw", CUDA_FAULT_CHECKS
        )
        summary_variant = {
            "case_count": 31,
            "failure_count": 0,
            "numeric_pass": True,
            "semantic_pass": True,
            "aggregate_numeric": {},
            "pass": True,
        }
        metric = {"metrics": {}, "pass": True}
        case_variant = {
            "numeric": {
                "first_layer_hidden": copy.deepcopy(metric),
                "final_logits": copy.deepcopy(metric),
                "final_log_probs": copy.deepcopy(metric),
            },
            "semantic": {"pass": True},
            "pass": True,
        }
        self.documents["correctness"] = {
            "schema_version": "1.0.0",
            "gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "created_at": "2026-08-26T00:00:00Z",
            "status": "pass",
            "roles": {},
            "gate_contract": {},
            "inputs": {},
            "bindings": {
                "candidate_git_revision": REVISION,
                "candidate_git_status_sha256": hashlib.sha256(b"").hexdigest(),
                "candidate_executable_sha256": digest(b"correctness executable"),
            },
            "summary": {
                "case_count": 31,
                "candidate_variant_count": 2,
                "failure_count": 0,
                "numeric_pass": True,
                "semantic_pass": True,
                "variants": {
                    "canonical-v1": copy.deepcopy(summary_variant),
                    "fixed-contiguous-37-balanced-v1": copy.deepcopy(summary_variant),
                },
            },
            "cases": [
                {
                    "prompt_id": f"prompt-{index:02d}",
                    "variants": {
                        "canonical-v1": copy.deepcopy(case_variant),
                        "fixed-contiguous-37-balanced-v1": copy.deepcopy(case_variant),
                    },
                    "pass": True,
                }
                for index in range(31)
            ],
        }
        correctness_sha = digest(
            (json.dumps(self.documents["correctness"], sort_keys=True, indent=2) + "\n").encode()
        )
        self.documents["performance"] = {
            "schema_version": "rustinfer.release-performance-report.v1",
            "status": "passed",
            "passed": True,
            "baseline": {},
            "candidate": {
                "candidate_id": "fixture",
                "recorded_at_utc": "2026-08-26T00:00:00Z",
                "source": {
                    "git_commit": REVISION,
                    "git_dirty": False,
                    "source_archive_sha256": self._binding()["source_archive_sha256"],
                    "profile_binary_sha256": digest(b"profile binary"),
                    "release_binary_sha256": self._binding()["release_binary_sha256"],
                    "profile_image_sha256": digest(b"profile image"),
                    "release_image_sha256": self.image_sha,
                    "semantic_class": "E0",
                    "correctness_gate_id": "pr16-release-correctness-v1",
                    "correctness_report_sha256": correctness_sha,
                },
                "metrics": {},
                "run_summary": {},
                "raw_runs": [],
            },
            "ratios": {},
            "checks": [{"name": "all_regressions", "passed": True}],
            "errors": [],
        }
        self.documents["soak"] = {
            "schema_version": "rustinfer.reliability-soak-report.v1",
            "status": "passed",
            "passed": True,
            "bindings": {
                "manifest_sha256": digest(b"soak manifest"),
                "binding_sha256": digest(b"soak binding"),
                "source": {
                    "git_commit": REVISION,
                    "git_dirty": False,
                    "source_archive_sha256": self._binding()["source_archive_sha256"],
                    "binary_sha256": self._binding()["release_binary_sha256"],
                    "image_sha256": self.image_sha,
                    "model_sha256": digest(b"model"),
                    "model_id": "HuggingFaceTB/SmolLM2-135M",
                    "model_revision": "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
                },
            },
            "scenario_summaries": [{}],
            "observations": {},
            "checks": [{"name": "all_scenarios", "passed": True}],
            "errors": [],
        }

    def write_reports(self) -> None:
        for name in ("python_report", "cuda_report", "correctness", "performance", "soak"):
            self.paths[name].write_text(
                json.dumps(self.documents[name], sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

    def refresh_manifest(self) -> None:
        self.write_reports()

        def artifact(name: str) -> dict[str, str]:
            path = self.paths[name]
            return {"path": path.relative_to(self.root).as_posix(), "sha256": digest(path.read_bytes())}

        self.manifest = {
            "schema_version": MANIFEST_VERSION,
            "candidate_id": "rustinfer-0.1.0-rc1",
            "source": {
                "git_revision": REVISION,
                "git_dirty": False,
                "archive": artifact("source"),
            },
            "release": {
                "binary": artifact("binary"),
                "bundle": artifact("bundle"),
                "image_digest": f"sha256:{self.image_sha}",
            },
            "evidence": {
                "python_free_e2e": {"report": artifact("python_report"), "raw_evidence": artifact("python_raw")},
                "cuda_fault": {"report": artifact("cuda_report"), "raw_evidence": artifact("cuda_raw")},
                "correctness": {"report": artifact("correctness")},
                "performance": {"report": artifact("performance")},
                "reliability_soak": {"report": artifact("soak")},
            },
        }
        self.write_manifest()

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate(self) -> dict[str, object]:
        return evaluate(self.manifest_path, self.root)


class ReleaseCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = CandidateFixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_source_bound_candidate_passes(self) -> None:
        report = self.fixture.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["bindings"]["git_revision"], REVISION)
        self.assertEqual(len(report["bindings"]["evidence_sha256"]), 7)

    def test_failed_or_missing_gate_fails_closed(self) -> None:
        del self.fixture.documents["performance"]["status"]
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("closed object mismatch", report["errors"][0])

    def test_cross_binding_mismatch_fails_closed(self) -> None:
        source = self.fixture.documents["performance"]["candidate"]["source"]
        source["release_binary_sha256"] = digest(b"different binary")
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("release_binary_sha256", report["errors"][0])

    def test_tampered_hashed_artifact_fails_closed(self) -> None:
        self.fixture.paths["cuda_raw"].write_bytes(b"tampered after manifest")
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("artifact digest mismatch", report["errors"][0])

    def test_path_traversal_fails_before_file_access(self) -> None:
        self.fixture.manifest["release"]["binary"]["path"] = "../rustinfer"
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("path traversal", report["errors"][0])

    def test_symlink_artifact_is_rejected(self) -> None:
        link = self.fixture.root / "binary-link"
        link.symlink_to(self.fixture.paths["binary"])
        self.fixture.manifest["release"]["binary"] = {
            "path": link.name,
            "sha256": digest(self.fixture.paths["binary"].read_bytes()),
        }
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("symlink", report["errors"][0])

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = self.fixture.manifest_path.read_text(encoding="utf-8")
        raw = raw.replace(
            '"candidate_id": "rustinfer-0.1.0-rc1",',
            '"candidate_id": "first", "candidate_id": "second",',
            1,
        )
        self.fixture.manifest_path.write_text(raw, encoding="utf-8")
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("duplicate JSON key", report["errors"][0])

    def test_placeholder_is_rejected(self) -> None:
        self.fixture.manifest["candidate_id"] = "replace-me"
        self.fixture.write_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("placeholder", report["errors"][0])

    def test_bundle_and_standalone_binary_must_match(self) -> None:
        self.fixture.paths["binary"].write_bytes(
            self.fixture.paths["binary"].read_bytes() + b"changed"
        )
        self.fixture.paths["binary"].chmod(0o755)
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("standalone binary differs", report["errors"][0])

    def test_required_attestation_check_set_is_closed(self) -> None:
        checks = self.fixture.documents["cuda_report"]["checks"]
        checks.pop()
        self.fixture.refresh_manifest()
        report = self.fixture.evaluate()
        self.assertFalse(report["passed"])
        self.assertIn("required check set mismatch", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
