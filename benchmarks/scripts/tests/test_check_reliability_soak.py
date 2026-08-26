from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "check_reliability_soak.py"
SPEC = importlib.util.spec_from_file_location("check_reliability_soak", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def write_raw_tar(path: Path, payloads: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:", format=tarfile.USTAR_FORMAT) as archive:
        for name in sorted(payloads):
            contents = payloads[name]
            member = tarfile.TarInfo(name)
            member.size = len(contents)
            member.mode = 0o644
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            member.mtime = 0
            archive.addfile(member, io.BytesIO(contents))


def read_raw_tar(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


class SoakFixture:
    KINDS = [
        ("steady", "steady", "iteration-batch"),
        ("burst-idle", "burst-idle", "iteration-batch"),
        ("mixed-short-long", "mixed", "iteration-batch"),
        ("invalid", "invalid", "iteration-batch"),
        ("overload", "overload", "iteration-batch"),
        ("cancellation-disconnect", "cancellation-disconnect", "iteration-batch"),
        ("near-kv", "near-kv", "iteration-batch"),
        ("graceful-restart", "graceful-restart", "iteration-batch"),
        ("rollback-iteration-batch", "rollback", "iteration-batch"),
        ("rollback-per-operation", "rollback", "per-operation"),
    ]

    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_directory = root / "run"
        self.run_directory.mkdir()
        self.golden = digest("golden completion")
        self.source = {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "source_archive_sha256": "b" * 64,
            "binary_sha256": "c" * 64,
            "image_sha256": "d" * 64,
            "model_sha256": "e" * 64,
            "model_id": "fixture/model",
            "model_revision": "f" * 40,
        }
        self.config_sha256 = digest("config")
        self.weights_sha256 = digest("weights")
        self.tokenizer_aggregate_sha256 = digest("tokenizer aggregate")
        self.tokenizer_json_sha256 = digest("tokenizer json")
        self.native_correctness_report = self._native_correctness_report()
        self.native_correctness_report_path = root / "native-correctness-report.json"
        self.native_correctness_report_path.write_text(
            json.dumps(self.native_correctness_report, sort_keys=True) + "\n"
        )
        self.native_correctness_report_sha256 = hashlib.sha256(
            self.native_correctness_report_path.read_bytes()
        ).hexdigest()
        self.correctness_golden = self._correctness_golden()
        self.correctness_golden_path = root / "correctness-golden.json"
        self.correctness_golden_path.write_text(
            json.dumps(self.correctness_golden, sort_keys=True) + "\n"
        )
        self.manifest = self._manifest()
        self.events: list[dict[str, object]] = []
        self.binding = checker._canonical_sha256(self.source)
        self._build_events()
        self.write()

    def _native_correctness_report(self) -> dict[str, object]:
        return {
            "schema_version": "1.0.0",
            "gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "status": "pass",
            "bindings": {
                "candidate_git_revision": self.source["git_commit"],
                "candidate_git_status_sha256": hashlib.sha256(b"").hexdigest(),
                "model_id": self.source["model_id"],
                "model_revision": self.source["model_revision"],
                "config_sha256": self.config_sha256,
                "weights_sha256": self.weights_sha256,
                "tokenizer_sha256": self.tokenizer_aggregate_sha256,
            },
        }

    def _correctness_golden(self) -> dict[str, object]:
        return {
            "schema_version": "rustinfer.python-free-release-e2e-golden.v1",
            "correctness_gate_id": "smollm2-fp32-bf16-native-e0-v2",
            "correctness_report_sha256": self.native_correctness_report_sha256,
            "source_revision": self.source["git_commit"],
            "model_id": self.source["model_id"],
            "model_revision": self.source["model_revision"],
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
            "tokenizer_aggregate_sha256": self.tokenizer_aggregate_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "prompt": "fixture prompt",
            "max_tokens": 8,
            "expected_greedy_text_sha256": self.golden,
        }

    def _manifest(self) -> dict[str, object]:
        scenarios = []
        for scenario_id, kind, mode in self.KINDS:
            request_profile = {
                "invalid": "invalid",
                "overload": "long",
                "cancellation-disconnect": "long",
                "near-kv": "near_kv",
            }.get(kind, "short")
            scenario = {
                "id": scenario_id, "kind": kind, "required": True,
                "duration_seconds": 2, "concurrency": 2,
                "cycle_interval_ms": 1000 if kind == "burst-idle" else 0,
                "request_profile": request_profile, "execution_completion": mode,
            }
            if kind == "mixed":
                scenario["secondary_request_profile"] = "long"
            scenarios.append(scenario)
        return {
            "schema_version": "rustinfer.reliability-soak-manifest.v1",
            "contract_id": "fixture",
            "target": {
                "kind": "process", "binary": "/bin/rustinfer", "model_path": "/models/fixture",
                "bind": "127.0.0.1:18080", "completion_path": "/v1/completions",
                "health_path": "/readyz", "metrics_path": "/metrics",
                "shutdown_signal": "TERM",
                "launch_arguments": ["serve", "--model", "{model_path}"],
            },
            "thresholds": {
                "sample_interval_ms": 1000, "maximum_sample_gap_ms": 1500,
                "minimum_samples_per_scenario": 2, "plateau_tail_fraction": 1.0,
                "maximum_rss_plateau_growth_bytes": 1024,
                "maximum_rss_slope_bytes_per_hour": 1024,
                "maximum_vram_plateau_growth_bytes": 1024,
                "maximum_vram_slope_bytes_per_hour": 1024,
                "minimum_cancellations": 1, "minimum_disconnects": 1,
                "minimum_overloads": 1, "graceful_shutdown_deadline_ms": 30000,
            },
            "requests": {
                "short": {
                    "model": self.source["model_id"],
                    "prompt": "fixture prompt",
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
                "long": {},
                "near_kv": {},
                "invalid": {},
            },
            "golden": {
                "request_profile": "short", "digest_domain": "completion-text-utf8",
                "generated_sha256": self.golden,
                "provenance_sha256": self.native_correctness_report_sha256,
            },
            "scenarios": scenarios,
        }

    def _event(self, kind: str, scenario_id: str | None, **values: object) -> None:
        self.events.append({
            "schema_version": "rustinfer.reliability-soak-event.v1",
            "sequence": len(self.events) + 1,
            "monotonic_ns": (len(self.events) + 1) * 1_000_000_000,
            "kind": kind, "scenario_id": scenario_id,
            "binding_sha256": self.binding, **values,
        })

    @staticmethod
    def metrics(*, active: int = 0) -> dict[str, object]:
        return {
            "active_requests": active, "waiting_requests": 0, "kv_allocated_blocks": 0,
            "allocation": {"device_live_count": 0, "device_live_bytes": 0, "pinned_live_count": 0, "pinned_live_bytes": 0},
            "counters": {"cancellations": 1, "disconnects": 1, "overloads": 1, "dropped_observations": 0},
        }

    def _sample(self, scenario_id: str | None) -> None:
        self._event(
            "sample", scenario_id,
            process={"pid": 123, "rss_bytes": 1000, "hwm_bytes": 1200, "fd_count": 8, "thread_count": 4, "children": []},
            gpu={"vram_bytes": 2000}, metrics=self.metrics(), sample_dropped=False,
        )

    def _request(self, scenario_id: str, outcome: str) -> None:
        generated = self.golden if outcome == "success" else None
        status = {"success": 200, "invalid": 400, "overload": 429, "cancelled": 0, "disconnected": 0}[outcome]
        self._event("request", scenario_id, request_id=f"request-{len(self.events)}", outcome=outcome, http_status=status, latency_ms=1.0, generated_sha256=generated)

    def _build_events(self) -> None:
        self._event("run_start", None)
        for scenario_id, kind, mode in self.KINDS:
            self._event("scenario_start", scenario_id, execution_completion=mode)
            self._sample(scenario_id)
            self._sample(scenario_id)
            if kind == "invalid":
                self._request(scenario_id, "invalid")
            elif kind == "overload":
                self._request(scenario_id, "overload")
            elif kind == "cancellation-disconnect":
                self._request(scenario_id, "cancelled")
                self._request(scenario_id, "disconnected")
            else:
                self._request(scenario_id, "success")
            if kind == "graceful-restart":
                self._event("restart", scenario_id, graceful=True, exit_code=0, elapsed_ms=100.0, before_generated_sha256=self.golden, after_generated_sha256=self.golden)
            self._event("scenario_end", scenario_id, status="success")
        self._sample(None)
        self._event("run_end", None, status="success")

    def write(self) -> None:
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest, sort_keys=True) + "\n")
        manifest_sha = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        run = {
            "schema_version": "rustinfer.reliability-soak-run.v1", "run_id": "fixture-run",
            "manifest_sha256": manifest_sha, "binding_sha256": self.binding, "source": self.source,
            "target": {"kind": "process", "pid": 123, "image_id": "sha256:" + "d" * 64, "command_sha256": "1" * 64},
            "started_at_utc": "2026-08-26T00:00:00Z",
        }
        (self.run_directory / "run.json").write_text(json.dumps(run, sort_keys=True) + "\n")
        (self.run_directory / "events.jsonl").write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in self.events))

    def renumber(self) -> None:
        for index, event in enumerate(self.events, 1):
            event["sequence"] = index
            event["monotonic_ns"] = index * 1_000_000_000

    def trusted_arguments(self) -> dict[str, Path]:
        return {
            "correctness_golden": self.correctness_golden_path,
            "native_correctness_report": self.native_correctness_report_path,
        }


class ReliabilitySoakCheckerTests(unittest.TestCase):
    def evaluate(self, mutate=None):  # type: ignore[no-untyped-def]
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        if mutate is not None:
            mutate(fixture)
            fixture.renumber()
            fixture.write()
        fixture_contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            fixture_contract,
        ):
            return checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )

    def test_short_complete_fixture_passes(self) -> None:
        report = self.evaluate()
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(
            report["schema_version"],
            "rustinfer.reliability-soak-report.v2",
        )
        self.assertEqual(len(report["scenario_summaries"]), 10)
        self.assertEqual(
            report["bindings"]["trusted_correctness"]["generated_text_sha256"],
            digest("golden completion"),
        )

    def test_trusted_correctness_inputs_are_mandatory(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        raw_evidence = Path(directory.name) / "soak.evidence.tar"
        fixture_contract = checker._normalized_manifest_sha256(fixture.manifest)
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            fixture_contract,
        ):
            report = checker.evaluate(fixture.manifest_path, fixture.run_directory)
            with self.assertRaisesRegex(checker.InputError, "--correctness-golden"):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    raw_evidence,
                )
            self.assertFalse(raw_evidence.exists())
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                raw_evidence,
                **fixture.trusted_arguments(),
            )
            replay = checker.replay_raw_evidence_archive(raw_evidence)
        self.assertFalse(report["passed"])
        self.assertIn("--correctness-golden", report["errors"][0])
        self.assertFalse(replay["report"]["passed"])
        self.assertIn("--correctness-golden", replay["report"]["errors"][0])

    def test_internally_consistent_generated_hash_cannot_self_authorize(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            substituted = digest("self-authorized completion")
            fixture.manifest["golden"]["generated_sha256"] = substituted
            for event in fixture.events:
                if event["kind"] == "request" and event["outcome"] == "success":
                    event["generated_sha256"] = substituted
                elif event["kind"] == "restart":
                    event["before_generated_sha256"] = substituted
                    event["after_generated_sha256"] = substituted

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("trusted E2E correctness golden", report["errors"][0])

    def test_provenance_must_hash_submitted_native_report(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["golden"]["provenance_sha256"] = digest(
                "self-authorized report"
            )

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("submitted native correctness report", report["errors"][0])

    def test_golden_request_identity_is_cross_bound(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["requests"]["short"]["prompt"] = "different prompt"

        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("trusted E2E correctness golden", report["errors"][0])

    def test_missing_required_scenario_end_fails(self) -> None:
        report = self.evaluate(lambda fixture: fixture.events.pop(next(i for i, event in enumerate(fixture.events) if event["kind"] == "scenario_end")))
        self.assertFalse(report["passed"])
        self.assertTrue(any(check["name"].endswith(".complete") and not check["passed"] for check in report["checks"]))

    def test_failure_event_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.events.insert(-1, {
                "schema_version": "rustinfer.reliability-soak-event.v1", "sequence": 0,
                "monotonic_ns": 0, "kind": "failure", "scenario_id": None,
                "binding_sha256": fixture.binding, "stage": "fixture", "message": "forced",
            })
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_failure_events")["passed"])

    def test_execution_completion_claim_must_match_manifest(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            event = next(event for event in fixture.events if event["scenario_id"] == "rollback-per-operation" and event["kind"] == "scenario_start")
            event["execution_completion"] = "iteration-batch"
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "rollback-per-operation.execution_completion")["passed"])

    def test_python_child_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            sample = next(event for event in fixture.events if event["kind"] == "sample")
            sample["process"]["children"] = [{"pid": 77, "comm": "python3", "executable": "/usr/bin/python3"}]
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_python_children")["passed"])

    def test_final_nonzero_allocation_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["kind"] == "sample"]
            samples[-1]["metrics"]["allocation"]["device_live_count"] = 1
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "final_quiescence")["passed"])

    def test_dropped_sample_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["scenario_id"] == "steady" and event["kind"] == "sample"]
            samples[1]["sample_dropped"] = True
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "no_dropped_samples")["passed"])

    def test_bounded_request_observation_eviction_is_not_sample_loss(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [
                event
                for event in fixture.events
                if event["scenario_id"] == "steady" and event["kind"] == "sample"
            ]
            samples[1]["metrics"]["counters"]["dropped_observations"] = 10_000

        report = self.evaluate(mutate)
        self.assertTrue(
            next(
                check
                for check in report["checks"]
                if check["name"] == "no_dropped_samples"
            )["passed"]
        )

    def test_sample_gap_fails(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.manifest["thresholds"]["maximum_sample_gap_ms"] = 500
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "steady.sample_gap_ms")["passed"])

    def test_truncated_scenario_cannot_claim_planned_duration(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            scenario = next(
                scenario
                for scenario in fixture.manifest["scenarios"]
                if scenario["id"] == "steady"
            )
            scenario["duration_seconds"] = 60

        report = self.evaluate(mutate)
        failed = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("steady.duration_seconds", failed)
        self.assertIn("steady.sample_coverage_seconds", failed)

    def test_manifest_contract_is_pinned_except_materialized_golden(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        fixture.manifest["thresholds"]["minimum_overloads"] = 2
        fixture.write()
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            report = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(report["status"], "error")
        self.assertIn("reviewed PR16 soak contract", report["errors"][0])

    def test_checked_in_template_digest_is_reviewed(self) -> None:
        template = json.loads(
            (SCRIPT.parents[1] / "soak/reliability-soak-v1.json").read_text()
        )
        self.assertEqual(
            checker._normalized_manifest_sha256(template),
            checker.REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256,
        )

    def test_cancellation_and_overload_must_be_observed(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            for event in fixture.events:
                if event["kind"] == "request" and event["outcome"] in {"cancelled", "disconnected", "overload"}:
                    event["outcome"] = "success"
                    event["http_status"] = 200
                    event["generated_sha256"] = fixture.golden
        report = self.evaluate(mutate)
        names = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertTrue({"cancellations_observed", "disconnects_observed", "overloads_observed"} <= names)

    def test_restart_and_rollback_must_match_bound_golden(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            restart = next(event for event in fixture.events if event["kind"] == "restart")
            restart["after_generated_sha256"] = digest("wrong")
            request = next(event for event in fixture.events if event["scenario_id"] == "rollback-per-operation" and event["kind"] == "request")
            request["generated_sha256"] = digest("also wrong")
        report = self.evaluate(mutate)
        names = {check["name"] for check in report["checks"] if not check["passed"]}
        self.assertIn("graceful_restart_golden_parity", names)
        self.assertIn("rollback_golden_parity", names)

    def test_steady_short_requests_must_match_bound_golden(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            request = next(
                event
                for event in fixture.events
                if event["scenario_id"] == "steady" and event["kind"] == "request"
            )
            request["generated_sha256"] = digest("drifted completion")

        report = self.evaluate(mutate)
        self.assertFalse(
            next(
                check
                for check in report["checks"]
                if check["name"] == "steady.golden_parity"
            )["passed"]
        )

    def test_source_binding_mismatch_is_an_input_error(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            fixture.source["binary_sha256"] = "9" * 64
        report = self.evaluate(mutate)
        self.assertEqual(report["status"], "error")
        self.assertIn("binding_sha256", report["errors"][0])

    def test_rss_slope_is_bounded(self) -> None:
        def mutate(fixture: SoakFixture) -> None:
            samples = [event for event in fixture.events if event["scenario_id"] == "steady" and event["kind"] == "sample"]
            samples[1]["process"]["rss_bytes"] = 1_000_000
        report = self.evaluate(mutate)
        self.assertFalse(next(check for check in report["checks"] if check["name"] == "steady.rss_slope_per_hour")["passed"])

    def test_raw_packager_is_deterministic_and_replays_exact_report(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        first = fixture.root / "first.tar"
        second = fixture.root / "second.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            expected = checker.evaluate(
                fixture.manifest_path,
                fixture.run_directory,
                **fixture.trusted_arguments(),
            )
            first_replay = checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                first,
                **fixture.trusted_arguments(),
            )
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                second,
                **fixture.trusted_arguments(),
            )
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_replay["report"], expected)
        self.assertEqual(
            first_replay["archive_sha256"],
            hashlib.sha256(first.read_bytes()).hexdigest(),
        )
        with tarfile.open(first, "r:") as archive:
            self.assertEqual(
                [member.name for member in archive.getmembers()],
                sorted(checker.RAW_ARCHIVE_MEMBERS),
            )

    def test_raw_replay_rejects_checksum_tampering(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        original = fixture.root / "original.tar"
        tampered = fixture.root / "tampered.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["events.jsonl"] += b"\n"
            write_raw_tar(tampered, payloads)
            with self.assertRaisesRegex(checker.InputError, "SHA256SUMS"):
                checker.replay_raw_evidence_archive(
                    tampered,
                    **fixture.trusted_arguments(),
                )

    def test_raw_replay_rejects_extra_member(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        original = fixture.root / "original.tar"
        expanded = fixture.root / "expanded.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            checker.package_raw_evidence(
                fixture.manifest_path,
                fixture.run_directory,
                original,
                **fixture.trusted_arguments(),
            )
            payloads = read_raw_tar(original)
            payloads["self-asserted-report.json"] = b'{"passed":true}\n'
            write_raw_tar(expanded, payloads)
            with self.assertRaisesRegex(checker.InputError, "exact ordered inventory"):
                checker.replay_raw_evidence_archive(
                    expanded,
                    **fixture.trusted_arguments(),
                )

    def test_raw_packager_is_create_only(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "existing.tar"
        output.write_bytes(b"owner data")
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            with self.assertRaises(FileExistsError):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertEqual(output.read_bytes(), b"owner data")

    def test_raw_packager_refuses_failed_run(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = SoakFixture(Path(directory.name))
        fixture.events[-1]["status"] = "failure"
        fixture.write()
        reviewed_digest = checker._normalized_manifest_sha256(fixture.manifest)
        output = fixture.root / "failed.tar"
        with mock.patch.object(
            checker,
            "REVIEWED_MANIFEST_TEMPLATE_CANONICAL_SHA256",
            reviewed_digest,
        ):
            with self.assertRaisesRegex(checker.InputError, "non-passing"):
                checker.package_raw_evidence(
                    fixture.manifest_path,
                    fixture.run_directory,
                    output,
                    **fixture.trusted_arguments(),
                )
        self.assertFalse(output.exists())


class SoakRunnerStaticTests(unittest.TestCase):
    def test_runner_recomputes_the_canonical_model_tree_binding(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        runner = (repository / "ci/run_release_soak.sh").read_text(encoding="utf-8")
        self.assertIn('test -d "$model_path"', runner)
        self.assertIn('test ! -L "$model_path"', runner)
        self.assertIn('find "$model_path" -mindepth 1 ! -type d ! -type f', runner)
        self.assertIn("-type f -print0 | sort -z", runner)
        self.assertIn("printf '%s  %s\\n'", runner)
        self.assertIn(
            'test "$computed_model_sha256" = "$RUSTINFER_MODEL_SHA256"',
            runner,
        )


if __name__ == "__main__":
    unittest.main()
