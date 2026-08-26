from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import stat
import sys
import tarfile
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

PROFILE_FIXTURE_SCRIPT = Path(__file__).with_name("test_check_native_profile_pair.py")
PROFILE_SPEC = importlib.util.spec_from_file_location(
    "release_native_profile_fixture", PROFILE_FIXTURE_SCRIPT
)
assert PROFILE_SPEC is not None and PROFILE_SPEC.loader is not None
profile_fixture_module = importlib.util.module_from_spec(PROFILE_SPEC)
sys.modules[PROFILE_SPEC.name] = profile_fixture_module
PROFILE_SPEC.loader.exec_module(profile_fixture_module)

PACKAGE_SCRIPT = SCRIPT.with_name("package_release_performance_evidence.py")
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_release_performance_evidence", PACKAGE_SCRIPT
)
assert PACKAGE_SPEC is not None and PACKAGE_SPEC.loader is not None
packager = importlib.util.module_from_spec(PACKAGE_SPEC)
sys.modules[PACKAGE_SPEC.name] = packager
PACKAGE_SPEC.loader.exec_module(packager)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class ReleaseFixture:
    def __init__(self, root: Path) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        self.root = root
        self.paths = {
            "source_archive": root / "source.tar",
            "profile_binary": root / "rustinfer-profile",
            "release_binary": root / "rustinfer",
            "weights": root / "model.safetensors",
            "tokenizer": root / "tokenizer.json",
            "correctness_report": root / "correctness.json",
        }
        self.digests = {name: digest(name) for name in self.paths}
        self.profile_image_digest = digest("profile image")
        self.release_image_digest = digest("release image")

        raw_root = root / "raw"
        raw_root.mkdir()
        self.profile_fixture = profile_fixture_module.ProfilePairFixture(raw_root)
        for run_index, run in enumerate(self.profile_fixture.candidate):
            run["source"] = {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "executable_sha256": self.digests["profile_binary"],
                "implementation_id": "native-iteration-command-batch",
                "runtime_flag": {
                    "name": "execution_completion",
                    "value": "iteration-batch",
                },
                "semantic_class": "E0",
                "correctness_gate_id": checker.CORRECTNESS_GATE_ID,
                "correctness_report_sha256": self.digests["correctness_report"],
            }
            run["environment"]["gpu"]["uuid"] = baseline["environment"][
                "gpu_uuid"
            ]
            run["environment"]["gpu"]["compute_capability"] = baseline[
                "environment"
            ]["compute_capability"]
            run["environment"]["host"]["environment_id"] = baseline[
                "environment"
            ]["environment_id"]
            software = run["environment"]["software"]
            software["nvidia_driver_version"] = baseline["environment"][
                "driver_version"
            ]
            software["cuda_runtime_version"] = baseline["environment"][
                "cuda_runtime_version"
            ]
            software["cuda_toolkit_version"] = baseline["environment"][
                "cuda_toolkit_version"
            ]
            software["container_image_sha256"] = self.profile_image_digest

            workload = run["workload"]
            workload.update(
                {
                    "workload_id": baseline["workload"]["workload_id"],
                    "model_id": baseline["model"]["model_id"],
                    "model_revision": baseline["model"]["model_revision"],
                    "weights_sha256": baseline["model"]["weights_sha256"],
                    "tokenizer_sha256": baseline["model"]["tokenizer_sha256"],
                    "dtype": baseline["model"]["dtype"],
                    "concurrency": baseline["workload"]["concurrency"],
                    "prompt_tokens": baseline["workload"]["prompt_tokens"],
                    "output_tokens": baseline["workload"]["output_tokens"],
                    "warmups": baseline["workload"]["warmups_per_run"],
                    "measured_iterations": baseline["workload"][
                        "measured_iterations_per_run"
                    ],
                    "sampling_id": baseline["workload"]["sampling"],
                    "seed": None,
                }
            )
            run_factor = 1.0 + (run_index - 2) * 0.002
            for request_index, request in enumerate(run["requests"]):
                request_factor = 1.0 + request_index * 0.0005
                request["ttft_ms"] = 5.4 * run_factor * request_factor
                request["tpot_ms"] = 7.0 * run_factor * request_factor
                request["e2e_ms"] = 225.0 * run_factor * request_factor
            run["aggregate"]["throughput_output_tokens_per_second"] = (
                140.0 / run_factor
            )

        self.candidate = {
            "schema_version": "rustinfer.release-performance-candidate.v1",
            "baseline_sha256": checker.BASELINE_SHA256,
            "candidate_id": "rustinfer-0.1.0-rc1",
            "recorded_at_utc": "2026-08-26T12:34:56Z",
            "status": "success",
            "source": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "source_archive_sha256": self.digests["source_archive"],
                "profile_binary_sha256": self.digests["profile_binary"],
                "release_binary_sha256": self.digests["release_binary"],
                "profile_image_sha256": self.profile_image_digest,
                "release_image_sha256": self.release_image_digest,
                "semantic_class": "E0",
                "correctness_gate_id": checker.CORRECTNESS_GATE_ID,
                "correctness_report_sha256": self.digests["correctness_report"],
            },
            "model": copy.deepcopy(baseline["model"]),
            "environment": copy.deepcopy(baseline["environment"]),
            "workload": copy.deepcopy(baseline["workload"]),
            "run_summary": {},
            "metrics": {},
            "raw_runs": [],
        }
        self.digests["weights"] = self.candidate["model"]["weights_sha256"]
        self.digests["tokenizer"] = self.candidate["model"]["tokenizer_sha256"]
        self.candidate_path = root / "candidate.json"
        for path in self.paths.values():
            path.write_bytes(b"fixture artifact")
        self.write_correctness_report()
        self.refresh()

    def optimization_correctness_report(self) -> dict[str, object]:
        source = self.candidate["source"]
        model = self.candidate["model"]
        environment = self.candidate["environment"]
        return {
            "schema_version": 1,
            "gate_id": checker.CORRECTNESS_GATE_ID,
            "recorded_at_utc": "2026-08-26T12:30:00Z",
            "status": "passed",
            "semantic_class": "E0",
            "source": {
                "git_commit": source["git_commit"],
                "git_dirty": False,
                "archive_sha256": source["source_archive_sha256"],
            },
            "model": {
                "model_id": model["model_id"],
                "revision": model["model_revision"],
                "dtype": model["dtype"],
                "manifest_sha256": digest("model manifest"),
                "weights_sha256": model["weights_sha256"],
                "tokenizer_sha256": model["tokenizer_sha256"],
            },
            "gpu": {
                "model": "NVIDIA GeForce RTX 4090",
                "uuid": environment["gpu_uuid"],
                "pci_bus_id": "00000000:01:00.0",
                "compute_capability": environment["compute_capability"],
                "vram_mib": 24564,
                "driver_version": environment["driver_version"],
            },
            "build": {
                "rustc": "1.85.0",
                "cuda_toolkit": environment["cuda_toolkit_version"],
                "cuda_architecture": environment["cuda_architecture"],
                "container_image_sha256": source["profile_image_sha256"],
                "network": "none",
                "cargo_locked": True,
                "cargo_offline": True,
            },
            "implementations": {
                "baseline": "per-operation",
                "candidate": "iteration-batch",
                "residual_rmsnorm": "separate",
                "rollback": "--execution-completion per-operation",
            },
            "tests": [
                {
                    "id": "cuda-compile-only",
                    "result": "passed",
                    "log_sha256": digest("cuda compile log"),
                },
                {
                    "id": "workspace-all-features-all-targets",
                    "result": "passed",
                    "log_sha256": digest("workspace test log"),
                },
                {
                    "id": "command-batch-lifecycle",
                    "result": "passed",
                    "log_sha256": digest("lifecycle log"),
                    "one_shot_finish": True,
                    "drop_restores_stream": True,
                },
                {
                    "id": "command-batch-resource-ledger",
                    "result": "passed",
                    "log_sha256": digest("resource ledger log"),
                    "queued_chain_raw_byte_mismatches": 0,
                    "cuda_live_allocation_delta": 0,
                    "owner_close_live_allocation_count": 0,
                    "validation_fail_closed": True,
                    "stream_reuse_after_finish": True,
                },
                {
                    "id": "smollm2-multi-step-greedy-exact",
                    "result": "passed",
                    "log_sha256": digest("smollm2 exact log"),
                    "decode_steps": 16,
                    "committed_iterations": 16,
                    "generated_token_ids": list(
                        checker.OPTIMIZATION_GOLDEN_TOKEN_IDS
                    ),
                    "raw_logit_mismatches": 0,
                    "token_id_mismatches": 0,
                    "cuda_live_allocation_delta": 0,
                    "owner_close_live_allocation_count": 0,
                },
            ],
        }

    def write_correctness_report(self) -> None:
        self.paths["correctness_report"].write_text(
            json.dumps(
                self.optimization_correctness_report(), sort_keys=True, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    def refresh(self) -> None:
        self.profile_fixture.write()
        runs = self.profile_fixture.candidate
        self.raw_paths = self.profile_fixture.candidate_paths
        self.candidate["raw_runs"] = [
            {
                "pair_index": run["pair_index"],
                "run_id": run["run_id"],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path, run in zip(self.raw_paths, runs, strict=True)
        ]
        workload = runs[0]["workload"]
        self.candidate["run_summary"] = {
            "independent_runs": len(runs),
            "warmups_per_run": workload["warmups"],
            "measured_iterations_per_run": workload["measured_iterations"],
            "failure_count": sum(run["failure_count"] for run in runs),
            "dropped_trace_records": sum(
                run["trace"]["dropped_records"] for run in runs
            ),
        }
        requests = [request for run in runs for request in run["requests"]]
        self.candidate["metrics"] = {
            "ttft_p95_ms": checker.native_profile.r7(
                [request["ttft_ms"] for request in requests], 0.95
            ),
            "tpot_p95_ms": checker.native_profile.r7(
                [request["tpot_ms"] for request in requests], 0.95
            ),
            "e2e_median_ms": checker.native_profile.r7(
                [request["e2e_ms"] for request in requests], 0.50
            ),
            "throughput_median_output_tokens_per_second": checker.native_profile.r7(
                [
                    run["aggregate"]["throughput_output_tokens_per_second"]
                    for run in runs
                ],
                0.50,
            ),
        }
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
        if path in self.raw_paths:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        raise AssertionError(f"unexpected artifact path: {path}")

    def digest_for_package(self, path: Path, label: str) -> str:
        try:
            return self.digest_for(path, label)
        except AssertionError:
            return hashlib.sha256(path.read_bytes()).hexdigest()

    def evaluate(self, *, baseline: Path = BASELINE) -> dict[str, object]:
        with mock.patch.object(checker, "_digest_file", side_effect=self.digest_for):
            return checker.evaluate(
                baseline,
                self.candidate_path,
                source_archive=self.paths["source_archive"],
                profile_binary=self.paths["profile_binary"],
                release_binary=self.paths["release_binary"],
                weights=self.paths["weights"],
                tokenizer=self.paths["tokenizer"],
                correctness_report=self.paths["correctness_report"],
                profile_image_id=f"sha256:{self.profile_image_digest}",
                release_image_id=f"sha256:{self.release_image_digest}",
                run_paths=self.raw_paths,
            )


class ReleasePerformanceTests(unittest.TestCase):
    def test_reviewed_baseline_digest_is_pinned(self) -> None:
        self.assertEqual(
            hashlib.sha256(BASELINE.read_bytes()).hexdigest(), checker.BASELINE_SHA256
        )
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            changed = Path(directory) / "baseline.json"
            changed.write_bytes(BASELINE.read_bytes() + b"\n")
            report = fixture.evaluate(baseline=changed)
            self.assertEqual(report["status"], "error")
            self.assertIn("not the reviewed v1 baseline", report["errors"][0])

    def test_passing_candidate_binds_producer_release_and_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report = fixture.evaluate()
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["checks"]), 4)
            source = report["candidate"]["source"]
            self.assertEqual(
                source["profile_binary_sha256"], fixture.digests["profile_binary"]
            )
            self.assertEqual(
                source["release_binary_sha256"], fixture.digests["release_binary"]
            )
            self.assertEqual(len(report["candidate"]["raw_runs"]), 5)

    def test_threshold_regression_is_derived_from_raw_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            report = fixture.evaluate()
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "failed")
            failed = [check["name"] for check in report["checks"] if not check["passed"]]
            self.assertEqual(failed, ["ttft_p95_regression"])

    def test_self_asserted_metric_or_raw_digest_tamper_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.candidate["metrics"]["ttft_p95_ms"] *= 0.5
            fixture.write()
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("raw-derived R7 metrics", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            fixture.raw_paths[0].write_bytes(fixture.raw_paths[0].read_bytes() + b" ")
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("raw run binding", report["errors"][0])

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
            fixture.digests["release_binary"] = digest("different release binary")
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

    def test_correctness_report_is_semantically_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            correctness = fixture.optimization_correctness_report()
            correctness["gate_id"] = "self-asserted-gate"
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("correctness_report.gate_id", report["errors"][0])

        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            correctness = fixture.optimization_correctness_report()
            correctness["tests"][-1]["generated_token_ids"][-1] += 1
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            report = fixture.evaluate()
            self.assertEqual(report["status"], "error")
            self.assertIn("generated_token_ids", report["errors"][0])

    def test_cli_refuses_to_overwrite_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            report_path = fixture.root / "report.json"
            report_path.write_text("occupied", encoding="utf-8")
            argv = [
                "--baseline", str(BASELINE),
                "--candidate", str(fixture.candidate_path),
                "--source-archive", str(fixture.paths["source_archive"]),
                "--profile-binary", str(fixture.paths["profile_binary"]),
                "--release-binary", str(fixture.paths["release_binary"]),
                "--weights", str(fixture.paths["weights"]),
                "--tokenizer", str(fixture.paths["tokenizer"]),
                "--correctness-report", str(fixture.paths["correctness_report"]),
                "--profile-image-id", f"sha256:{fixture.profile_image_digest}",
                "--release-image-id", f"sha256:{fixture.release_image_digest}",
                "--run", *(str(path) for path in fixture.raw_paths),
                "--report", str(report_path),
            ]
            stderr = io.StringIO()
            with mock.patch.object(checker, "_digest_file", side_effect=fixture.digest_for):
                with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                    exit_code = checker.main(argv)
            self.assertEqual(exit_code, 2)
            self.assertEqual(report_path.read_text(encoding="utf-8"), "occupied")
            self.assertIn("refusing to overwrite", stderr.getvalue())


class ReleasePerformancePackagingTests(unittest.TestCase):
    def package(
        self,
        fixture: ReleaseFixture,
        output: Path,
        *,
        candidate_id: str = "rustinfer-0.1.0-rc1",
    ) -> dict[str, object]:
        with mock.patch.object(
            checker, "_digest_file", side_effect=fixture.digest_for_package
        ):
            return checker.package_release_performance_evidence(
                BASELINE,
                output,
                candidate_id=candidate_id,
                recorded_at_utc="2026-08-26T12:34:56Z",
                source_archive=fixture.paths["source_archive"],
                profile_binary=fixture.paths["profile_binary"],
                release_binary=fixture.paths["release_binary"],
                weights=fixture.paths["weights"],
                tokenizer=fixture.paths["tokenizer"],
                correctness_report=fixture.paths["correctness_report"],
                profile_image_id=f"sha256:{fixture.profile_image_digest}",
                release_image_id=f"sha256:{fixture.release_image_digest}",
                run_paths=fixture.raw_paths,
            )

    @staticmethod
    def cli_argv(
        fixture: ReleaseFixture,
        output: Path,
        *,
        candidate_id: str = "rustinfer-0.1.0-rc1",
    ) -> list[str]:
        return [
            "--baseline", str(BASELINE),
            "--candidate-id", candidate_id,
            "--recorded-at-utc", "2026-08-26T12:34:56Z",
            "--source-archive", str(fixture.paths["source_archive"]),
            "--profile-binary", str(fixture.paths["profile_binary"]),
            "--release-binary", str(fixture.paths["release_binary"]),
            "--weights", str(fixture.paths["weights"]),
            "--tokenizer", str(fixture.paths["tokenizer"]),
            "--correctness-report", str(fixture.paths["correctness_report"]),
            "--profile-image-id", f"sha256:{fixture.profile_image_digest}",
            "--release-image-id", f"sha256:{fixture.release_image_digest}",
            "--run", *(str(path) for path in fixture.raw_paths),
            "--output-directory", str(output),
        ]

    def test_raw_derivation_does_not_require_a_candidate_document(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = ReleaseFixture(Path(directory))
            payloads = [
                (str(path), path.read_bytes())
                for path in reversed(fixture.raw_paths)
            ]
            derived = checker.derive_raw_run_payloads(payloads)
            self.assertEqual(
                [name for name, _ in derived["payloads"]],
                list(checker.RAW_EVIDENCE_FILES),
            )
            self.assertEqual(derived["model"], fixture.candidate["model"])
            self.assertEqual(
                derived["environment"], fixture.candidate["environment"]
            )
            self.assertEqual(derived["workload"], fixture.candidate["workload"])
            self.assertEqual(
                derived["run_summary"], fixture.candidate["run_summary"]
            )
            self.assertEqual(derived["metrics"], fixture.candidate["metrics"])
            self.assertEqual(derived["raw_runs"], fixture.candidate["raw_runs"])

    def test_raw_archive_is_deterministic_canonical_ustar_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            payloads = [
                (str(path), path.read_bytes())
                for path in reversed(fixture.raw_paths)
            ]
            first = root / "first.tar"
            second = root / "second.tar"
            first_sha = checker.write_raw_evidence_archive(first, payloads)
            second_sha = checker.write_raw_evidence_archive(second, payloads)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(stat.S_IMODE(first.stat().st_mode), 0o644)
            self.assertEqual(stat.S_IMODE(second.stat().st_mode), 0o644)
            self.assertEqual(first_sha, second_sha)
            self.assertEqual(first_sha, hashlib.sha256(first.read_bytes()).hexdigest())
            with tarfile.open(first, "r:") as archive:
                members = archive.getmembers()
                self.assertEqual(
                    [member.name for member in members],
                    list(checker.RAW_EVIDENCE_FILES),
                )
                for member in members:
                    self.assertEqual(member.mode, 0o644)
                    self.assertEqual(member.uid, 0)
                    self.assertEqual(member.gid, 0)
                    self.assertEqual(member.uname, "")
                    self.assertEqual(member.gname, "")
                    self.assertEqual(member.mtime, 0)
            replay = checker.replay_raw_evidence_archive(first)
            self.assertEqual(replay["archive_sha256"], first_sha)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )

    def test_raw_archive_rejects_noncanonical_metadata_compression_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            payloads = checker.derive_raw_run_payloads(
                [(str(path), path.read_bytes()) for path in fixture.raw_paths]
            )["payloads"]

            wrong_mode = root / "wrong-mode.tar"
            with tarfile.open(
                wrong_mode, "w:", format=tarfile.USTAR_FORMAT
            ) as archive:
                for index, (name, raw) in enumerate(payloads):
                    info = checker._canonical_tar_info(name, len(raw))
                    if index == 0:
                        info.mode = 0o600
                    archive.addfile(info, io.BytesIO(raw))
            with self.assertRaisesRegex(checker.InputError, "non-canonical metadata"):
                checker.load_raw_evidence_archive(wrong_mode)

            compressed = root / "compressed.tar.gz"
            with tarfile.open(
                compressed, "w:gz", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, raw in payloads:
                    archive.addfile(
                        checker._canonical_tar_info(name, len(raw)),
                        io.BytesIO(raw),
                    )
            with self.assertRaisesRegex(checker.InputError, "uncompressed USTAR"):
                checker.load_raw_evidence_archive(compressed)

            tailed = root / "tailed.tar"
            checker.write_raw_evidence_archive(tailed, payloads)
            tailed.write_bytes(tailed.read_bytes() + b"unexpected trailing bytes")
            with self.assertRaisesRegex(checker.InputError, "canonical deterministic"):
                checker.load_raw_evidence_archive(tailed)

    def test_raw_archive_writer_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "existing.tar"
            output.write_bytes(b"owner data")
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            with self.assertRaises(FileExistsError):
                checker.write_raw_evidence_archive(output, payloads)
            self.assertEqual(output.read_bytes(), b"owner data")

    def test_raw_archive_writer_loses_publish_race_without_unlinking_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "raced.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]

            def competing_publish(_source: Path, target: Path) -> None:
                self.assertEqual(target, output)
                target.write_bytes(b"owner data created by the winner")
                raise FileExistsError(target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=competing_publish
            ):
                with self.assertRaises(FileExistsError):
                    checker.write_raw_evidence_archive(output, payloads)
            self.assertEqual(output.read_bytes(), b"owner data created by the winner")
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_file())
            self.assertEqual(stat.S_IMODE(residues[0].stat().st_mode), 0o644)
            replay = checker.replay_raw_evidence_archive(residues[0])
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )

    def test_raw_archive_writer_does_not_unlink_replaced_published_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "published.tar"
            displaced = root / "displaced-generated.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            fsync_directory = checker._fsync_directory

            def replace_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    output.rename(displaced)
                    output.write_bytes(b"owner replacement")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "held inode"):
                    checker.write_raw_evidence_archive(output, payloads)
            self.assertEqual(output.read_bytes(), b"owner replacement")
            self.assertTrue(displaced.is_file())
            self.assertEqual(list(root.glob(f".{output.name}.staging-*")), [])

    def test_raw_archive_writer_detects_same_inode_tamper_after_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "tampered-after-publish.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            rename_noreplace = checker._rename_noreplace

            def tamper_after_publish(source: Path, target: Path) -> None:
                rename_noreplace(source, target)
                if target == output:
                    target.write_bytes(b"same inode replacement bytes")

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=tamper_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "digest changed"):
                    checker.write_raw_evidence_archive(output, payloads)
            self.assertEqual(output.read_bytes(), b"same inode replacement bytes")

    def test_raw_archive_writer_preserves_complete_publish_on_parent_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "fsync-failed.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected parent fsync failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failure"):
                    checker.write_raw_evidence_archive(output, payloads)
            replay = checker.replay_raw_evidence_archive(output)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )

    def test_raw_archive_replay_digest_and_payload_share_one_open_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            archive = root / "evidence.tar"
            moved = root / "opened-snapshot.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            expected_sha = checker.write_raw_evidence_archive(archive, payloads)
            stable_snapshot = checker._stable_fd_snapshot
            replaced = False

            def replace_after_open(
                descriptor: int, label: str, maximum: int
            ) -> tuple[bytes, str, os.stat_result]:
                nonlocal replaced
                if not replaced:
                    archive.rename(moved)
                    archive.write_bytes(b"replacement path contents")
                    replaced = True
                return stable_snapshot(descriptor, label, maximum)

            with mock.patch.object(
                checker, "_stable_fd_snapshot", side_effect=replace_after_open
            ):
                replay = checker.replay_raw_evidence_archive(archive)
            self.assertEqual(replay["archive_sha256"], expected_sha)
            self.assertEqual(
                replay["derived"]["raw_runs"], fixture.candidate["raw_runs"]
            )
            self.assertEqual(archive.read_bytes(), b"replacement path contents")
            self.assertEqual(hashlib.sha256(moved.read_bytes()).hexdigest(), expected_sha)

    def test_raw_archive_writer_preserves_only_private_failed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "failed.tar"
            payloads = [
                (str(path), path.read_bytes()) for path in fixture.raw_paths
            ]
            partial_bytes = b"partial archive bytes"

            def write_partial_then_fail(handle: object, _payloads: object) -> None:
                handle.write(partial_bytes)  # type: ignore[attr-defined]
                raise OSError("injected archive write failure")

            with mock.patch.object(
                checker,
                "_write_canonical_raw_archive",
                side_effect=write_partial_then_fail,
            ):
                with self.assertRaisesRegex(OSError, "injected archive write failure"):
                    checker.write_raw_evidence_archive(output, payloads)
            self.assertFalse(output.exists())
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_file())
            self.assertEqual(residues[0].read_bytes(), partial_bytes)
            self.assertEqual(stat.S_IMODE(residues[0].stat().st_mode), 0o600)

    def test_packager_emits_only_candidate_report_and_raw_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            first = root / "package-one"
            second = root / "package-two"
            result = self.package(fixture, first)
            self.package(fixture, second)
            self.assertTrue(result["report"]["passed"], result)
            expected_names = {
                checker.PACKAGE_CANDIDATE_NAME,
                checker.PACKAGE_REPORT_NAME,
                checker.PACKAGE_RAW_EVIDENCE_NAME,
            }
            self.assertEqual({path.name for path in first.iterdir()}, expected_names)
            for name in expected_names:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
                self.assertEqual(stat.S_IMODE((first / name).stat().st_mode), 0o644)
            candidate = json.loads(
                (first / checker.PACKAGE_CANDIDATE_NAME).read_text(encoding="utf-8")
            )
            report = json.loads(
                (first / checker.PACKAGE_REPORT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(candidate["raw_runs"], fixture.candidate["raw_runs"])
            self.assertEqual(report, result["report"])
            replay = checker.replay_raw_evidence_archive(
                first / checker.PACKAGE_RAW_EVIDENCE_NAME
            )
            checker.validate_raw_run_payloads(
                replay["payloads"], checker._validate_candidate(candidate)
            )

    def test_packager_is_create_only_and_preserves_structurally_failed_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            existing = root / "existing"
            existing.mkdir()
            owner = existing / "owner.txt"
            owner.write_text("owner data", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                self.package(fixture, existing)
            self.assertEqual(owner.read_text(encoding="utf-8"), "owner data")

            correctness = fixture.optimization_correctness_report()
            correctness["gate_id"] = "invalid-gate"
            fixture.paths["correctness_report"].write_text(
                json.dumps(correctness), encoding="utf-8"
            )
            invalid = root / "invalid"
            with self.assertRaisesRegex(checker.InputError, "structurally invalid"):
                self.package(fixture, invalid)
            self.assertFalse(invalid.exists())
            residues = list(root.glob(f".{invalid.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_dir())
            self.assertEqual(
                {path.name for path in residues[0].iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_preserves_comparable_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            output = root / "regression-evidence"
            result = self.package(fixture, output)
            self.assertFalse(result["report"]["passed"])
            self.assertEqual(result["report"]["status"], "failed")
            self.assertEqual(result["report"]["errors"], [])
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_loses_atomic_publish_race_without_touching_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "raced-package"
            original_rename = checker._rename_noreplace

            def competing_publish(source: Path, target: Path) -> None:
                if target == output:
                    target.mkdir()
                    (target / "owner.txt").write_text("owner data", encoding="utf-8")
                    raise FileExistsError(target)
                original_rename(source, target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=competing_publish
            ):
                with self.assertRaises(FileExistsError):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"), "owner data"
            )
            residues = list(root.glob(f".{output.name}.staging-*"))
            self.assertEqual(len(residues), 1)
            self.assertTrue(residues[0].is_dir())
            self.assertEqual(
                {path.name for path in residues[0].iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_cleanup_skips_replaced_post_publish_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "published-package"
            displaced = root / "displaced-package"
            fsync_directory = checker._fsync_directory

            def replace_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    output.rename(displaced)
                    output.mkdir()
                    (output / "owner.txt").write_text(
                        "owner replacement", encoding="utf-8"
                    )

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_after_publish
            ):
                with self.assertRaisesRegex(checker.InputError, "changed"):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "owner.txt").read_text(encoding="utf-8"),
                "owner replacement",
            )
            self.assertEqual(
                {path.name for path in displaced.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )

    def test_packager_preserves_triplet_on_parent_fsync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "fsync-failed-package"
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected parent fsync failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failure"):
                    self.package(fixture, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )
            checker.replay_raw_evidence_archive(
                output / checker.PACKAGE_RAW_EVIDENCE_NAME
            )
            candidate = json.loads(
                (output / checker.PACKAGE_CANDIDATE_NAME).read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (output / checker.PACKAGE_REPORT_NAME).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(candidate["raw_runs"], fixture.candidate["raw_runs"])
            self.assertTrue(report["passed"], report)

    def test_packager_never_deletes_post_publish_child_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "owner-replaced-package"
            owner_bytes = b"owner replacement after directory publish"
            fsync_directory = checker._fsync_directory

            def replace_child_and_fail(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    candidate = output / checker.PACKAGE_CANDIDATE_NAME
                    candidate.unlink()
                    candidate.write_bytes(owner_bytes)
                    raise OSError("injected post-publish failure")

            with mock.patch.object(
                checker, "_fsync_directory", side_effect=replace_child_and_fail
            ):
                with self.assertRaisesRegex(OSError, "post-publish failure"):
                    self.package(fixture, output)
            self.assertEqual(
                (output / checker.PACKAGE_CANDIDATE_NAME).read_bytes(), owner_bytes
            )
            self.assertTrue((output / checker.PACKAGE_REPORT_NAME).is_file())
            self.assertTrue((output / checker.PACKAGE_RAW_EVIDENCE_NAME).is_file())

    def test_packager_revalidates_every_child_after_atomic_publish(self) -> None:
        for child_name in (
            checker.PACKAGE_CANDIDATE_NAME,
            checker.PACKAGE_REPORT_NAME,
            checker.PACKAGE_RAW_EVIDENCE_NAME,
        ):
            with self.subTest(child_name=child_name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    fixture = ReleaseFixture(root)
                    output = root / "tampered-package"
                    rename_noreplace = checker._rename_noreplace

                    def tamper_then_publish(source: Path, target: Path) -> None:
                        if target == output:
                            (source / child_name).write_bytes(b"tampered child bytes")
                        rename_noreplace(source, target)

                    with mock.patch.object(
                        checker,
                        "_rename_noreplace",
                        side_effect=tamper_then_publish,
                    ):
                        with self.assertRaisesRegex(
                            checker.InputError, "changed"
                        ):
                            self.package(fixture, output)
                    self.assertEqual(
                        (output / child_name).read_bytes(), b"tampered child bytes"
                    )

    def test_packager_rejects_post_check_extra_inventory_without_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "extra-child-package"
            rename_noreplace = checker._rename_noreplace

            def add_extra_then_publish(source: Path, target: Path) -> None:
                if target == output:
                    (source / "unexpected-owner-file").write_bytes(b"owner data")
                rename_noreplace(source, target)

            with mock.patch.object(
                checker, "_rename_noreplace", side_effect=add_extra_then_publish
            ):
                with self.assertRaisesRegex(
                    checker.InputError, "exact three-file inventory"
                ):
                    self.package(fixture, output)
            self.assertEqual(
                (output / "unexpected-owner-file").read_bytes(), b"owner data"
            )

    def test_invalid_candidate_id_fails_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for index, candidate_id in enumerate(
                (
                    "pr16-candidate",
                    "rustinfer-00.1.0-rc1",
                    "rustinfer-0.01.0-rc1",
                    "rustinfer-0.1.00-rc1",
                    "rustinfer-0.1.0-rc01",
                )
            ):
                with self.subTest(candidate_id=candidate_id):
                    output = root / f"invalid-id-{index}"
                    with mock.patch.object(
                        checker,
                        "_read_raw_run_paths",
                        side_effect=AssertionError("raw inputs were read"),
                    ):
                        with self.assertRaisesRegex(
                            checker.InputError, "candidate_id"
                        ):
                            self.package(
                                fixture, output, candidate_id=candidate_id
                            )
                    self.assertFalse(output.exists())


    def test_packager_cli_uses_the_same_fail_closed_producer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "cli-package"
            argv = self.cli_argv(fixture, output)
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                self.assertEqual(packager.main(argv), 0)
            self.assertTrue(
                json.loads(
                    (output / checker.PACKAGE_REPORT_NAME).read_text(
                        encoding="utf-8"
                    )
                )["passed"]
            )

    def test_packager_cli_returns_one_and_publishes_threshold_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            for run in fixture.profile_fixture.candidate:
                for request in run["requests"]:
                    request["ttft_ms"] *= 1.2
            fixture.refresh()
            output = root / "cli-regression"
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                self.assertEqual(packager.main(self.cli_argv(fixture, output)), 1)
            report = json.loads(
                (output / checker.PACKAGE_REPORT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])

    def test_packager_cli_returns_two_without_overwriting_or_invalid_publish(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)

            invalid = root / "invalid-cli"
            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(
                            self.cli_argv(
                                fixture, invalid, candidate_id="not-a-release-candidate"
                            )
                        ),
                        2,
                    )
            self.assertFalse(invalid.exists())
            self.assertIn("candidate_id", stderr.getvalue())

            existing = root / "existing-cli"
            existing.mkdir()
            owner = existing / "owner.txt"
            owner.write_text("owner data", encoding="utf-8")
            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(self.cli_argv(fixture, existing)), 2
                    )
            self.assertEqual(owner.read_text(encoding="utf-8"), "owner data")
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_packager_cli_returns_two_but_preserves_post_publish_triplet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = ReleaseFixture(root)
            output = root / "cli-post-publish-failure"
            fsync_directory = checker._fsync_directory

            def fail_after_publish(path: Path) -> None:
                fsync_directory(path)
                if path == root and output.exists():
                    raise OSError("injected CLI parent fsync failure")

            stderr = io.StringIO()
            with mock.patch.object(
                checker, "_digest_file", side_effect=fixture.digest_for_package
            ), mock.patch.object(
                checker, "_fsync_directory", side_effect=fail_after_publish
            ):
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        packager.main(self.cli_argv(fixture, output)), 2
                    )
            self.assertIn("parent fsync failure", stderr.getvalue())
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    checker.PACKAGE_CANDIDATE_NAME,
                    checker.PACKAGE_REPORT_NAME,
                    checker.PACKAGE_RAW_EVIDENCE_NAME,
                },
            )


if __name__ == "__main__":
    unittest.main()
