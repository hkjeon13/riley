#!/usr/bin/env python3
"""CPU-only hostile tests for the C02 soak v2 semantic replay."""

from __future__ import annotations

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v4 as v4_binder
import check_c02_provenance_v2 as raw
import check_soak_v2_receipt_v2 as semantic
import provenance_v2_common as common
import test_bind_raw_c02_soak_v4 as v4_fixtures
import test_bind_raw_c02_soak_v5 as v5_fixtures


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class CheckSoakV2ReceiptV2Tests(unittest.TestCase):
    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _v4_manifest(self, name: str = "serial-v4.json") -> tuple[v4_fixtures.BindRawC02SoakV4Tests, dict]:
        fixture = v4_fixtures.BindRawC02SoakV4Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        return fixture, v4_binder.bind_raw_soak_manifest(fixture.root, request_path, name)

    def _v5_manifest(self, name: str = "fallback-v5.json") -> tuple[v5_fixtures.BindRawC02SoakV5Tests, dict]:
        fixture = v5_fixtures.BindRawC02SoakV5Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        return fixture, fixture._bind(request_path, name)

    def _empty_root(self, document: object, name: str = "manifest.json") -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve() / "evidence"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        root.joinpath(name).write_bytes(
            document if isinstance(document, bytes) else common.canonical_json_bytes(document)
        )
        return root

    def _target(self, fixture: v4_fixtures.BindRawC02SoakV4Tests) -> raw.ObservedTarget:
        return raw.ObservedTarget(
            target=raw.TargetTuple(
                pid=fixture.pid,
                start_ticks=fixture.ticks,
                gpu_index=0,
                gpu_uuid=fixture.gpu_uuid,
            ),
            listener_port=fixture.port,
            listener_inode=fixture.inode,
        )

    def _bindings(self, fixture: v4_fixtures.BindRawC02SoakV4Tests) -> raw.ReplayedSoakSemanticBindings:
        return raw.ReplayedSoakSemanticBindings(
            freeze_sha256="a" * 64,
            base_release_candidate_report_sha256="b" * 64,
            configuration_profile=fixture.profile,
            configuration_sha256=fixture.configuration_sha256,
        )

    def test_replays_v4_leaves_as_narrow_semantic_diagnostic_without_writing(self) -> None:
        fixture, raw_report = self._v4_manifest()
        before = sorted(path.relative_to(fixture.root).as_posix() for path in fixture.root.rglob("*"))

        report = semantic.check_soak_v2_receipt_v2(fixture.root, "serial-v4.json")

        self.assertEqual(report["schema_version"], semantic.SEMANTIC_REPORT_VERSION)
        self.assertEqual(report["scope"], semantic.SEMANTIC_SCOPE)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], semantic.SEMANTIC_AUTHORITY)
        self.assertEqual(report["candidate_id"], raw_report["candidate_id"])
        self.assertEqual(report["raw_manifest_version"], raw.SOAK_V4_MANIFEST_VERSION)
        scenario = report["derived_facts"]["scenarios"][0]
        self.assertEqual(scenario["scenario_id"], "smoke")
        self.assertEqual(scenario["interval"]["scope"], "per-observation-session")
        self.assertTrue(scenario["interval"]["strictly_increasing"])
        self.assertTrue(scenario["metrics"]["cumulative_request_counters_monotonic"])
        self.assertTrue(scenario["typed_sampling"]["all_committed"])
        self.assertFalse(scenario["typed_sampling"]["fallback_projection_replayed"])
        self.assertEqual(report["reason_codes"], [])
        self.assertTrue(all(item["passed"] for item in report["checks"]))
        self.assertEqual(report["not_established"]["qualification"], "not-established")
        self.assertEqual(report["not_established"]["cross_session_interval_order"], "not-established")
        after = sorted(path.relative_to(fixture.root).as_posix() for path in fixture.root.rglob("*"))
        self.assertEqual(after, before)

    def test_replays_v5_typed_fallback_from_audit_and_event_leaves(self) -> None:
        fixture, raw_report = self._v5_manifest()

        report = semantic.check_soak_v2_receipt_v2(fixture.fixture.root, "fallback-v5.json")

        self.assertEqual(report["raw_manifest_version"], raw.SOAK_V5_MANIFEST_VERSION)
        self.assertEqual(report["candidate_id"], raw_report["candidate_id"])
        scenarios = report["derived_facts"]["scenarios"]
        self.assertEqual(len(scenarios), 1)
        scenario = scenarios[0]
        self.assertEqual(scenario["scenario_id"], raw.FALLBACK_SCENARIO_ID)
        self.assertIn("fallback_event", scenario)
        self.assertTrue(scenario["typed_sampling"]["fallback_projection_replayed"])
        self.assertEqual(
            scenario["typed_sampling"]["derived_selected_backend_counts"],
            {"cpu-normative": 1, "gpu-greedy": 0},
        )
        self.assertEqual(
            scenario["typed_sampling"]["derived_ineligibility_reason_counts"]["nonzero-temperature"],
            1,
        )

    def test_historical_versions_are_rejected_before_semantic_input_replay(self) -> None:
        for version, reason in (
            ("riley.soak-v2-receipt.v1", "historical-soak-v1-rejected"),
            ("riley.soak-v2-raw-provenance.v1", "historical-soak-v1-rejected"),
            ("riley.soak-v2-raw-provenance.v2", "historical-soak-v2-rejected"),
            ("riley.soak-v2-raw-provenance.v3", "historical-soak-v3-rejected"),
        ):
            with self.subTest(version=version):
                root = self._empty_root({"schema_version": version})
                with mock.patch.object(
                    raw,
                    "replay_completed_soak_v4_semantic_inputs_fd",
                    side_effect=AssertionError("historical input reached v4 replay"),
                ), mock.patch.object(
                    raw,
                    "replay_completed_soak_v5_semantic_inputs_fd",
                    side_effect=AssertionError("historical input reached v5 replay"),
                ):
                    with self.assertRaises(semantic.SoakV2SemanticReplayError) as raised:
                        semantic.check_soak_v2_receipt_v2(root, "manifest.json")
                self.assert_reason(raised, reason)

    def test_rejects_noncanonical_and_nonroot_manifest_inputs(self) -> None:
        root = self._empty_root(b'{"schema_version":"riley.soak-v2-raw-provenance.v4"}\n')
        with self.assertRaises(semantic.SoakV2SemanticReplayError) as noncanonical:
            semantic.check_soak_v2_receipt_v2(root, "manifest.json")
        self.assert_reason(noncanonical, "noncanonical-json")
        for name in ("nested/manifest.json", "../manifest.json", ".manifest.json", "manifest.complete"):
            with self.subTest(name=name):
                with self.assertRaises(semantic.SoakV2SemanticReplayError) as raised:
                    semantic.check_soak_v2_receipt_v2(root, name)
                self.assert_reason(raised, "raw-manifest-must-be-direct-root-leaf")

    def test_source_audit_semantics_rejects_token_accounting_and_iteration_drift(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        audit_path = fixture.root / "source-audit/cmpl-1.json"
        original = json.loads(audit_path.read_text(encoding="utf-8"))
        for mutate, reason in (
            (lambda document: document["usage"].update({"completion_tokens": 2}), "semantic-audit-token-accounting"),
            (lambda document: document["sampling_selections"][0].update({"iteration_id": 0}), "invalid-semantic-integer"),
            (lambda document: document["sampling_selections"][0].update({"committed": False}), "semantic-audit-uncommitted-selection"),
            (lambda document: document["sampling_selections"][0].update({"configured_backend": []}), "semantic-audit-unknown-backend"),
            (lambda document: document["sampling_selections"][0].update({"ineligibility_reason": {}}), "semantic-audit-unknown-ineligibility"),
            (lambda document: document.update({"finish_reason": []}), "semantic-audit-terminal-reason"),
        ):
            with self.subTest(reason=reason):
                document = copy.deepcopy(original)
                mutate(document)
                with self.assertRaises(semantic.SoakV2SemanticReplayError) as raised:
                    semantic._reconstruct_generation_audit(  # noqa: SLF001
                        common.canonical_json_bytes(document),
                        candidate_id=fixture.candidate_id,
                        bindings=self._bindings(fixture),
                        target=self._target(fixture),
                        request_id="cmpl-1",
                        label="hostile audit",
                    )
                self.assert_reason(raised, reason)

    def test_observation_semantics_rejects_counter_regression_and_interval_inversion(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        metric = json.loads((fixture.root / "obs-0/raw/metrics.json").read_text(encoding="utf-8"))
        regressed = copy.deepcopy(metric)
        regressed["request_states"]["completed"] = 0
        first = common.canonical_json_bytes(metric)
        second = common.canonical_json_bytes(regressed)
        session = common.descriptor_for_bytes("session.json", b"{}", "session")
        first_descriptor = common.descriptor_for_bytes("first.json", first, "first")
        second_descriptor = common.descriptor_for_bytes("second.json", second, "second")
        observation = raw.ReplayedObservationSessionSemanticInputs(
            session=session,
            observed_target=self._target(fixture),
            samples=(
                raw.ReplayedObservationSampleSemanticInput(
                    sample=first_descriptor,
                    sequence=0,
                    elapsed_monotonic_millis=0,
                    metrics=first_descriptor,
                    metrics_bytes=first,
                ),
                raw.ReplayedObservationSampleSemanticInput(
                    sample=second_descriptor,
                    sequence=1,
                    elapsed_monotonic_millis=1,
                    metrics=second_descriptor,
                    metrics_bytes=second,
                ),
            ),
        )
        with self.assertRaises(semantic.SoakV2SemanticReplayError) as regression:
            semantic._reconstruct_observation_semantics(observation, label="hostile observation")  # noqa: SLF001
        self.assert_reason(regression, "semantic-observation-cumulative-counter-regression")

        inverted = raw.ReplayedObservationSessionSemanticInputs(
            session=session,
            observed_target=self._target(fixture),
            samples=(
                observation.samples[0],
                raw.ReplayedObservationSampleSemanticInput(
                    sample=second_descriptor,
                    sequence=1,
                    elapsed_monotonic_millis=0,
                    metrics=second_descriptor,
                    metrics_bytes=first,
                ),
            ),
        )
        with self.assertRaises(semantic.SoakV2SemanticReplayError) as interval:
            semantic._reconstruct_observation_semantics(inverted, label="hostile observation")  # noqa: SLF001
        self.assert_reason(interval, "semantic-observation-interval-order")

    def test_rejects_semantic_input_drift_between_two_held_fd_passes(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        original = semantic._replay_once  # noqa: SLF001
        calls = 0

        def drifting(*args: object) -> dict:
            nonlocal calls
            calls += 1
            replayed = original(*args)  # type: ignore[arg-type]
            if calls == 2:
                replayed = copy.deepcopy(replayed)
                replayed["derived_facts"]["scenarios"][0]["request_id"] = "cmpl-drift"
            return replayed

        with mock.patch.object(semantic, "_replay_once", side_effect=drifting):
            with self.assertRaises(semantic.SoakV2SemanticReplayError) as raised:
                semantic.check_soak_v2_receipt_v2(fixture.root, "serial-v4.json")
        self.assert_reason(raised, "semantic-replay-drift")

    def test_holds_shared_lock_for_all_semantic_replays(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        original = raw.replay_completed_soak_v4_semantic_inputs_fd
        contender = (
            "import errno,fcntl,os,sys\n"
            "fd=os.open(sys.argv[1],os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)\n"
            "try:\n"
            "    try:\n"
            "        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "    except OSError as error:\n"
            "        if error.errno in {errno.EACCES,errno.EAGAIN}: raise SystemExit(0)\n"
            "        raise\n"
            "    raise SystemExit(1)\n"
            "finally:\n"
            "    os.close(fd)\n"
        )

        def assert_shared_lock(root_fd: int, manifest_name: str) -> raw.ReplayedSoakSemanticInputs:
            completed = subprocess.run(
                ["/usr/bin/python3", "-B", "-S", "-c", contender, str(fixture.root)],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return original(root_fd, manifest_name)

        with mock.patch.object(raw, "replay_completed_soak_v4_semantic_inputs_fd", side_effect=assert_shared_lock):
            report = semantic.check_soak_v2_receipt_v2(fixture.root, "serial-v4.json")
        self.assertEqual(report["status"], "passed")

    def test_cli_emits_canonical_json_and_failure_has_no_stdout(self) -> None:
        fixture, _raw_report = self._v5_manifest()
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = semantic.main(
                ["--evidence-root", str(fixture.fixture.root), "--raw-manifest", "fallback-v5.json"]
            )
        self.assertEqual(exit_code, 0, stderr.getvalue())
        document = common.parse_canonical_json(stdout.buffer.getvalue().rstrip(b"\n"), "semantic CLI")
        self.assertEqual(document["status"], "passed")
        self.assertEqual(document["authority"], semantic.SEMANTIC_AUTHORITY)

        root = self._empty_root([])
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = semantic.main(["--evidence-root", str(root), "--raw-manifest", "manifest.json"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.buffer.getvalue(), b"")
        self.assertIn("semantic replay failed", stderr.getvalue())

    def test_static_surface_uses_only_the_fd_semantic_input_api(self) -> None:
        source = Path(semantic.__file__).read_text(encoding="utf-8")
        for required in (
            "sys.dont_write_bytecode = True",
            "open_private_evidence_directory",
            "fcntl.LOCK_SH | fcntl.LOCK_NB",
            "replay_completed_soak_v4_semantic_inputs_fd",
            "replay_completed_soak_v5_semantic_inputs_fd",
            "per-observation-session",
            "semantic-replay-drift",
        ):
            self.assertIn(required, source)
        self.assertLess(source.index("sys.dont_write_bytecode = True"), source.index("import check_c02_provenance_v2 as raw"))
        for forbidden in (
            "import check_soak_v2_receipt",
            "import subprocess",
            "import socket",
            "urllib",
            "import requests",
            "nvidia",
            "docker",
            "podman",
            "ssh ",
            "O_CREAT",
            "os.link",
            "common.write_create_only_json",
            "common.publish_create_only_hardlink",
            "os.mkdir",
            "--output",
            "Path.read_text",
            "Path.read_bytes",
        ):
            self.assertNotIn(forbidden, source)

    def test_schema_reserves_the_narrow_v2_authority(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/soak-v2-semantic-replay-v2.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["schema_version"]["const"], semantic.SEMANTIC_REPORT_VERSION)
        self.assertEqual(schema["properties"]["status"]["const"], "passed")
        self.assertEqual(schema["properties"]["qualification_status"]["const"], "not-run")
        self.assertEqual(schema["properties"]["authority"]["const"], semantic.SEMANTIC_AUTHORITY)
        self.assertEqual(schema["$defs"]["notEstablished"]["required"], sorted(semantic._NOT_ESTABLISHED))  # noqa: SLF001
        variants = schema["oneOf"]
        self.assertEqual(len(variants), 2)
        self.assertEqual(
            variants[0]["properties"]["raw_manifest_version"]["const"], raw.SOAK_V4_MANIFEST_VERSION
        )
        self.assertEqual(
            variants[1]["properties"]["raw_manifest_version"]["const"], raw.SOAK_V5_MANIFEST_VERSION
        )
        description = schema["description"].lower()
        for denied_claim in ("qualification", "gate e", "campaign threshold", "cross-session"):
            self.assertIn(denied_claim, description)


if __name__ == "__main__":
    unittest.main()
