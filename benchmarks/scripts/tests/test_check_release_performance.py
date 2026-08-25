from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_release_performance.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
BASELINE = REPOSITORY_ROOT / "benchmarks/release/performance-baseline-v1.json"
SPEC = importlib.util.spec_from_file_location("check_release_performance", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ReleaseFixture:
    def __init__(self, root: Path) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.root = root
        self.paths = {
            "source_archive": root / "source.tar",
            "binary": root / "rustinfer",
            "weights": root / "model.safetensors",
            "tokenizer": root / "tokenizer.json",
            "correctness_report": root / "correctness.json",
        }
        self.digests = {name: digest(name) for name in self.paths}
        self.image_digest = digest("image")
        self.candidate = {
            "schema_version": "rustinfer.release-performance-candidate.v1",
            "baseline_sha256": checker.BASELINE_SHA256,
            "candidate_id": "pr16-release-candidate-fixture",
            "recorded_at_utc": "2026-08-26T12:34:56Z",
            "status": "success",
            "source": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "source_archive_sha256": self.digests["source_archive"],
                "binary_sha256": self.digests["binary"],
                "image_sha256": self.image_digest,
                "semantic_class": "E0",
                "correctness_gate_id": "pr16-release-correctness-v1",
                "correctness_report_sha256": self.digests["correctness_report"],
            },
            "model": copy.deepcopy(baseline["model"]),
            "environment": copy.deepcopy(baseline["environment"]),
            "workload": copy.deepcopy(baseline["workload"]),
            "run_summary": {
                "independent_runs": 5,
                "warmups_per_run": 5,
                "measured_iterations_per_run": 30,
                "failure_count": 0,
                "dropped_trace_records": 0,
            },
            "metrics": {
                field: value * (0.98 if field.startswith(("ttft", "tpot", "e2e")) else 1.02)
                for field, value in baseline["metrics"].items()
            },
        }
        self.digests["weights"] = self.candidate["model"]["weights_sha256"]
        self.digests["tokenizer"] = self.candidate["model"]["tokenizer_sha256"]
        self.candidate_path = root / "candidate.json"
        for path in self.paths.values():
            path.write_bytes(b"fixture artifact")
        self.write()

    def write(self) -> None:
        self.candidate_path.write_text(
            json.dumps(self.candidate, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def digest_for(self, path: Path, _label: str) -> str:
        for name, expected_path in self.paths.items():
            if path == expected_path:
                return self.digests[name]
        raise AssertionError(f"unexpected artifact path: {path}")

    def evaluate(self) -> dict[str, object]:
        with mock.patch.object(checker, "_digest_file", side_effect=self.digest_for):
            return checker.evaluate(
                BASELINE,
                self.candidate_path,
                source_archive=self.paths["source_archive"],
                binary=self.paths["binary"],
                weights=self.paths["weights"],
                tokenizer=self.paths["tokenizer"],
                correctness_report=self.paths["correctness_report"],
                image_id=f"sha256:{self.image_digest}",
            )


class ReleasePerformanceTests(unittest.TestCase):
    def test_reviewed_baseline_digest_is_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(BASELINE.read_bytes()).hexdigest(), checker.BASELINE_SHA256
        )
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "baseline.json"
            changed.write_bytes(BASELINE.read_bytes() + b"\n")
            fixture = ReleaseFixture(Path(directory))
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for
            ):
                report = checker.evaluate(
                    changed,
                    fixture.candidate_path,
                    source_archive=fixture.paths["source_archive"],
                    binary=fixture.paths["binary"],
                    weights=fixture.paths["weights"],
                    tokenizer=fixture.paths["tokenizer"],
                    correctness_report=fixture.paths["correctness_report"],
                    image_id=f"sha256:{fixture.image_digest}",
                )
            self.assertEqual(report["status"], "error")
            self.assertIn("not the reviewed v1 baseline", report["errors"][0])

    def test_passing_candidate_binds_artifacts_and_reports_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report = fixture.evaluate()
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["checks"]), 4)
            self.assertEqual(
                report["candidate"]["source"]["binary_sha256"],
                fixture.digests["binary"],
            )

    def test_threshold_regression_fails_without_becoming_incomparable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.candidate["metrics"]["ttft_p95_ms"] *= 1.2
            fixture.write()
            report = fixture.evaluate()
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            failed = [check["name"] for check in report["checks"] if not check["passed"]]
            self.assertEqual(failed, ["ttft_p95_regression"])

    def test_model_or_environment_drift_is_incomparable(self) -> None:
        for field, value in [("model", "other/model"), ("environment", "8.0")]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = ReleaseFixture(Path(directory))
                if field == "model":
                    fixture.candidate["model"]["model_id"] = value
                else:
                    fixture.candidate["environment"]["compute_capability"] = value
                fixture.write()
                report = fixture.evaluate()
                self.assertEqual(report["status"], "incomparable")
                self.assertFalse(report["passed"])

    def test_artifact_mismatch_and_unknown_fields_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.digests["binary"] = digest("different binary")
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("does not match artifact", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.candidate["unexpected"] = True
            fixture.write()
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("unknown fields", report["errors"][0])

    def test_cli_refuses_to_overwrite_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report_path = fixture.root / "report.json"
            report_path.write_text("occupied", encoding="utf-8")
            argv = [
                "--baseline", str(BASELINE),
                "--candidate", str(fixture.candidate_path),
                "--source-archive", str(fixture.paths["source_archive"]),
                "--binary", str(fixture.paths["binary"]),
                "--weights", str(fixture.paths["weights"]),
                "--tokenizer", str(fixture.paths["tokenizer"]),
                "--correctness-report", str(fixture.paths["correctness_report"]),
                "--image-id", f"sha256:{fixture.image_digest}",
                "--report", str(report_path),
            ]
            stderr = io.StringIO()
            with mock.patch.object(checker, "_digest_file", side_effect=fixture.digest_for):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    exit_code = checker.main(argv)
            self.assertEqual(exit_code, 2)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "occupied")
            self.assertIn("refusing to overwrite", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
