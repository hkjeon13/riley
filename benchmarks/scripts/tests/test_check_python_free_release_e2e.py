from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "check_python_free_release_e2e.py"
REPOSITORY_ROOT = SCRIPT.parents[2]
DRIVER = REPOSITORY_ROOT / "ci/run_python_free_release_e2e.sh"
RELEASE_DIR = REPOSITORY_ROOT / "ci/release"
sys.path.insert(0, str(RELEASE_DIR))
from build_release_bundle import build_bundle  # noqa: E402
from test_release import EPOCH, fixture_elf, rewrite_archive  # noqa: E402

SPEC = importlib.util.spec_from_file_location("check_python_free_release_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class E2EFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.revision = "a" * 40
        self.image_id = f"sha256:{'b' * 64}"
        self.source_archive = root / "source.tar"
        self.release_binary = root / "rustinfer"
        self.release_bundle = root / "rustinfer.tar.gz"
        self.model_dir = root / "model"
        self.model_dir.mkdir()
        self.tokenizer = self.model_dir / "tokenizer.json"
        self.weights = self.model_dir / "model.safetensors"
        self.golden = root / "golden.json"
        self.correctness_report = root / "correctness-report.json"
        self.shutdown_metrics = root / "shutdown-metrics.json"
        self.repeat_shutdown_metrics = root / "repeat-shutdown-metrics.json"
        self.evidence = root / "raw.json"
        self.source_archive.write_bytes(b"source archive fixture")
        self.release_binary.write_bytes(fixture_elf())
        self.release_binary.chmod(0o755)
        self.tokenizer.write_bytes(b'{"tokenizer":"fixture"}\n')
        self.weights.write_bytes(b"safetensors fixture")
        self.expected_text_sha256 = sha_bytes(b"fixture completion")
        self.model_revision = "model-revision-fixture"
        self.correctness_report.write_text(
            json.dumps(
                {
                    "gate_id": checker.CORRECTNESS_GATE,
                    "status": "pass",
                    "bindings": {
                        "candidate_git_revision": self.revision,
                        "candidate_git_status_sha256": sha_bytes(b""),
                        "model_id": "fixture-model",
                        "model_revision": self.model_revision,
                        "weights_sha256": sha_bytes(self.weights.read_bytes()),
                        "tokenizer_sha256": sha_bytes(self.tokenizer.read_bytes()),
                    },
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        correctness_report_sha256 = sha_bytes(self.correctness_report.read_bytes())
        self.golden.write_text(
            json.dumps(
                {
                    "schema_version": checker.GOLDEN_SCHEMA,
                    "correctness_gate_id": checker.CORRECTNESS_GATE,
                    "correctness_report_sha256": correctness_report_sha256,
                    "source_revision": self.revision,
                    "model_id": "fixture-model",
                    "model_revision": self.model_revision,
                    "weights_sha256": sha_bytes(self.weights.read_bytes()),
                    "tokenizer_sha256": sha_bytes(self.tokenizer.read_bytes()),
                    "prompt": "A bounded release probe",
                    "max_tokens": 8,
                    "expected_greedy_text_sha256": self.expected_text_sha256,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        repository = root / "repository"
        repository.mkdir()
        (repository / "Cargo.toml").write_text(
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0"\n'
            'license = "LicenseRef-Test-Fixture"\n',
            encoding="utf-8",
        )
        (repository / "LICENSE").write_text(
            "Owner-approved fixture license for release contract unit tests.\n"
            "Permission is granted only inside this temporary test fixture.\n",
            encoding="utf-8",
        )
        build_bundle(
            binary_path=self.release_binary,
            output=self.release_bundle,
            repository_root=repository,
            source_revision=self.revision,
            source_date_epoch=EPOCH,
        )
        self.hashes = {
            "archive": sha_bytes(self.source_archive.read_bytes()),
            "binary": sha_bytes(self.release_binary.read_bytes()),
            "bundle": sha_bytes(self.release_bundle.read_bytes()),
            "model": checker.model_tree_sha256(self.model_dir),
            "weights": sha_bytes(self.weights.read_bytes()),
            "tokenizer": sha_bytes(self.tokenizer.read_bytes()),
            "golden": sha_bytes(self.golden.read_bytes()),
            "correctness": sha_bytes(self.correctness_report.read_bytes()),
        }
        self.raw = self._raw()
        self.shutdown_metrics.write_text(
            json.dumps(self.raw["observations"]["shutdown"]["metrics"], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.repeat_shutdown_metrics.write_text(
            json.dumps(self.raw["observations"]["shutdown"]["repeat_metrics"], sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.raw["observations"]["shutdown"]["metrics_sha256"] = sha_bytes(
            self.shutdown_metrics.read_bytes()
        )
        self.raw["observations"]["shutdown"]["repeat_metrics_sha256"] = sha_bytes(
            self.repeat_shutdown_metrics.read_bytes()
        )
        self.write()

    def _raw(self) -> dict[str, object]:
        return {
            "schema_version": checker.RAW_SCHEMA,
            "run_id": "python-free-e2e-fixture",
            "recorded_at_utc": "2026-08-26T12:34:56Z",
            "status": "success",
            "source": {
                "git_revision": self.revision,
                "git_dirty": False,
                "source_archive_sha256": self.hashes["archive"],
            },
            "release": {
                "binary_sha256": self.hashes["binary"],
                "bundle_sha256": self.hashes["bundle"],
                "image_sha256": self.image_id.removeprefix("sha256:"),
            },
            "model": {
                "model_id": "fixture-model",
                "model_revision": self.model_revision,
                "model_tree_sha256": self.hashes["model"],
                "weights_sha256": self.hashes["weights"],
                "tokenizer_sha256": self.hashes["tokenizer"],
                "correctness_gate_id": checker.CORRECTNESS_GATE,
                "correctness_report_sha256": self.hashes["correctness"],
                "correctness_golden_sha256": self.hashes["golden"],
            },
            "runtime": {
                "container_ids": ["c" * 64, "e" * 64],
                "network_mode": "none",
                "image_id": self.image_id,
                "image_binary_sha256": self.hashes["binary"],
            },
            "observations": {
                "readyz": {"http_status": 200, "ready": True, "accepting": True},
                "models": {"http_status": 200, "model_ids": ["fixture-model"]},
                "greedy": {
                    "non_stream_http_status": 200,
                    "stream_http_status": 200,
                    "non_stream_text_sha256": self.expected_text_sha256,
                    "stream_text_sha256": self.expected_text_sha256,
                    "approved_text_sha256": self.expected_text_sha256,
                    "completion_tokens": 8,
                    "stream_token_events": 8,
                    "finish_reason": "length",
                    "stream_done": True,
                    "prompt_sha256": sha_bytes(b"A bounded release probe"),
                    "max_tokens": 8,
                },
                "sampling": {
                    "seed": 424242,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "first_http_status": 200,
                    "second_http_status": 200,
                    "first_completion_tokens": 16,
                    "second_completion_tokens": 16,
                    "first_finish_reason": "length",
                    "second_finish_reason": "length",
                    "first_text_sha256": "d" * 64,
                    "second_text_sha256": "d" * 64,
                },
                "cancellation": {
                    "disconnect_probe_sent": True,
                    "cancellations_before": 0,
                    "cancellations_after": 1,
                    "disconnects_before": 0,
                    "disconnects_after": 1,
                    "active_requests_after": 0,
                    "waiting_requests_after": 0,
                },
                "shutdown": {
                    "signal": "SIGTERM",
                    "exit_code": 0,
                    "repeat_exit_code": 0,
                    "metrics_sha256": "0" * 64,
                    "repeat_metrics_sha256": "0" * 64,
                    "metrics": {
                        "active_requests": 0,
                        "waiting_requests": 0,
                        "kv_allocated_blocks": 0,
                        "allocation": {
                            "device_live_count": 0,
                            "device_live_bytes": 0,
                            "pinned_live_count": 0,
                            "pinned_live_bytes": 0,
                        },
                        "counters": {
                            "cancellations": 1,
                            "disconnects": 1,
                            "overloads": 0,
                            "dropped_observations": 0,
                        },
                    },
                    "repeat_metrics": {
                        "active_requests": 0,
                        "waiting_requests": 0,
                        "kv_allocated_blocks": 0,
                        "allocation": {
                            "device_live_count": 0,
                            "device_live_bytes": 0,
                            "pinned_live_count": 0,
                            "pinned_live_bytes": 0,
                        },
                        "counters": {
                            "cancellations": 0,
                            "disconnects": 0,
                            "overloads": 0,
                            "dropped_observations": 0,
                        },
                    },
                },
                "python_free": {
                    "forbidden_executables": [],
                    "forbidden_artifact_count": 0,
                    "processes": [
                        {"pid": 101, "ppid": 0, "comm": "rustinfer", "args": "rustinfer serve"}
                    ],
                    "manifest_dependencies": sorted(checker.REQUIRED_DEPENDENCIES),
                    "loader_dependencies": [
                        "libcuda.so.1 => /usr/lib/libcuda.so.1",
                        "libcudart.so.12 => /usr/local/cuda/lib64/libcudart.so.12",
                    ],
                    "unresolved_dependencies": [],
                    "forbidden_dependency_matches": [],
                },
            },
        }

    def write(self) -> None:
        self.evidence.write_text(
            json.dumps(self.raw, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate(self) -> tuple[dict[str, object], str | None]:
        return checker.evaluate(
            self.evidence,
            source_revision=self.revision,
            source_archive=self.source_archive,
            release_binary=self.release_binary,
            release_bundle=self.release_bundle,
            image_id=self.image_id,
            model_dir=self.model_dir,
            expected_model_tree_sha256=self.hashes["model"],
            weights=self.weights,
            expected_weights_sha256=self.hashes["weights"],
            tokenizer=self.tokenizer,
            expected_tokenizer_sha256=self.hashes["tokenizer"],
            correctness_golden=self.golden,
            expected_correctness_golden_sha256=self.hashes["golden"],
            correctness_report=self.correctness_report,
            expected_correctness_report_sha256=self.hashes["correctness"],
            shutdown_metrics=self.shutdown_metrics,
            repeat_shutdown_metrics=self.repeat_shutdown_metrics,
        )

    def argv(self, report: Path) -> list[str]:
        return [
            "--evidence", str(self.evidence),
            "--source-revision", self.revision,
            "--source-archive", str(self.source_archive),
            "--release-binary", str(self.release_binary),
            "--release-bundle", str(self.release_bundle),
            "--image-id", self.image_id,
            "--model-dir", str(self.model_dir),
            "--model-tree-sha256", self.hashes["model"],
            "--weights", str(self.weights),
            "--weights-sha256", self.hashes["weights"],
            "--tokenizer", str(self.tokenizer),
            "--tokenizer-sha256", self.hashes["tokenizer"],
            "--correctness-golden", str(self.golden),
            "--correctness-golden-sha256", self.hashes["golden"],
            "--correctness-report", str(self.correctness_report),
            "--correctness-report-sha256", self.hashes["correctness"],
            "--shutdown-metrics", str(self.shutdown_metrics),
            "--repeat-shutdown-metrics", str(self.repeat_shutdown_metrics),
            "--report", str(report),
        ]


class PythonFreeReleaseE2ETests(unittest.TestCase):
    def test_complete_bound_real_runtime_evidence_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            report, diagnostic = fixture.evaluate()
            self.assertIsNone(diagnostic)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["gate"], checker.GATE)
            self.assertEqual(
                {check["id"] for check in report["checks"]}, set(checker.CHECK_IDS)
            )
            self.assertEqual(
                report["raw_evidence_sha256"], sha_bytes(fixture.evidence.read_bytes())
            )

    def test_self_asserted_golden_or_model_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            fixture.raw["observations"]["greedy"]["approved_text_sha256"] = "e" * 64
            fixture.raw["observations"]["greedy"]["non_stream_text_sha256"] = "e" * 64
            fixture.raw["observations"]["greedy"]["stream_text_sha256"] = "e" * 64
            fixture.write()
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("correctness golden", diagnostic)

        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            fixture.weights.write_bytes(b"tampered")
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("model tree binding mismatch", diagnostic)

        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            golden = json.loads(fixture.golden.read_text(encoding="utf-8"))
            golden["source_revision"] = "f" * 40
            fixture.golden.write_text(
                json.dumps(golden, sort_keys=True, indent=2) + "\n", encoding="utf-8"
            )
            fixture.hashes["golden"] = sha_bytes(fixture.golden.read_bytes())
            fixture.raw["model"]["correctness_golden_sha256"] = fixture.hashes["golden"]
            fixture.write()
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("correctness golden", diagnostic)

    def test_robust_release_bundle_verifier_and_nonempty_sampling_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            invalid = fixture.root / "extra.tar.gz"

            def add_extra(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
                root = entries[0][0].name.split("/", 1)[0]
                member = tarfile.TarInfo(f"{root}/extra.txt")
                member.size = 1
                member.mode = 0o644
                member.mtime = EPOCH
                entries.append((member, b"x"))

            rewrite_archive(fixture.release_bundle, invalid, add_extra)
            fixture.release_bundle = invalid
            fixture.hashes["bundle"] = sha_bytes(invalid.read_bytes())
            fixture.raw["release"]["bundle_sha256"] = fixture.hashes["bundle"]
            fixture.write()
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("unreviewed extra", diagnostic)

        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            empty_sha256 = sha_bytes(b"")
            sampling = fixture.raw["observations"]["sampling"]
            sampling["first_text_sha256"] = empty_sha256
            sampling["second_text_sha256"] = empty_sha256
            fixture.write()
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("empty output", diagnostic)

    def test_network_python_cancellation_and_shutdown_claims_fail_closed(self) -> None:
        mutations = {
            "network_mode": lambda raw: raw["runtime"].__setitem__("network_mode", "bridge"),
            "python process": lambda raw: raw["observations"]["python_free"]["processes"][0].__setitem__("comm", "python3"),
            "cancellation": lambda raw: raw["observations"]["cancellation"].__setitem__("cancellations_after", 0),
            "shutdown": lambda raw: raw["observations"]["shutdown"]["metrics"]["allocation"].__setitem__("device_live_count", 1),
        }
        for expected, mutate in mutations.items():
            with self.subTest(expected), tempfile.TemporaryDirectory() as directory:
                fixture = E2EFixture(Path(directory))
                mutate(fixture.raw)
                fixture.write()
                report, diagnostic = fixture.evaluate()
                self.assertEqual(report["status"], "error")
                self.assertIn(expected.split()[0], diagnostic.lower())

    def test_unknown_fields_and_raw_digest_change_are_rejected_or_rebound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            fixture.raw["unexpected"] = True
            fixture.write()
            report, diagnostic = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("unknown fields", diagnostic)

        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            report, diagnostic = fixture.evaluate()
            self.assertIsNone(diagnostic)
            old_digest = report["raw_evidence_sha256"]
            fixture.raw["run_id"] = "python-free-e2e-fixture-2"
            fixture.write()
            rebound, diagnostic = fixture.evaluate()
            self.assertIsNone(diagnostic)
            self.assertNotEqual(old_digest, rebound["raw_evidence_sha256"])

    def test_cli_report_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = E2EFixture(Path(directory))
            report = fixture.root / "attestation.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(checker.main(fixture.argv(report)), 0)
            original = report.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main(fixture.argv(report)), 2)
            self.assertEqual(report.read_bytes(), original)

    def test_remote_driver_contract_is_network_none_and_runtime_python_free(self) -> None:
        source = DRIVER.read_text(encoding="utf-8")
        for required in [
            "--network none",
            "--read-only",
            "RUSTINFER_SHUTDOWN_METRICS_PATH",
            "docker kill --signal TERM",
            "/readyz",
            "/v1/models",
            "/v1/completions",
            "command_name in python python3 pip pip3",
            "docker top",
            "ldd /opt/rustinfer/bin/rustinfer",
        ]:
            self.assertIn(required, source)
        self.assertNotIn("--network host", source)


if __name__ == "__main__":
    unittest.main()
