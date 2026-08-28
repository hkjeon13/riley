#!/usr/bin/env python3
"""CPU-only integration tests for the closed C02 native-fallback binder v5."""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_c02_soak_v5 as binder
import capture_c02_raw_soak_scenarios_v1 as capture
import check_c02_provenance_v2 as checker
import effective_runtime_config_contract as runtime_config
import provenance_v2_common as common
import test_bind_raw_c02_soak_v4 as v4_tests


class BindRawC02SoakV5Tests(unittest.TestCase):
    """Use the v4 CPU fixture only to construct raw process/GPU evidence."""

    def setUp(self) -> None:
        self.fixture = v4_tests.BindRawC02SoakV4Tests(methodName="runTest")
        self.fixture.setUp()
        self.fixture.profile = checker.MAX_PERFORMANCE_EXACT_PROFILE

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _bind(self, request_path: str, manifest_name: str) -> dict:
        """Exercise the FD entry point; this copied fixture lives under /private."""

        root_fd = common.open_private_evidence_directory(self.fixture.root, "test root")
        try:
            return binder.bind_raw_soak_manifest_fd(root_fd, request_path, manifest_name)
        finally:
            os.close(root_fd)

    def _fallback_audit(self, request_id: str) -> None:
        fixture = self.fixture
        selections = [
            {
                "iteration_id": 1,
                "configured_backend": "gpu-greedy",
                "selected_backend": "cpu-normative",
                "ineligibility_reason": "nonzero-temperature",
                "committed": True,
            }
        ]
        audit = {
            "schema_version": capture.AUDIT_VERSION,
            "candidate_id": fixture.candidate_id,
            "runtime_identity": {
                "configuration_profile": fixture.profile,
                "configuration_sha256": fixture.configuration_sha256,
            },
            "process_identity": {"pid": fixture.pid, "start_ticks": fixture.ticks},
            "server_request_id": request_id,
            "delivery_mode": "non-stream",
            "prompt_token_ids": [1],
            "committed_output_tokens": [{"emitted_text_delta": "x", "token_id": 2}],
            "sampling_selections": selections,
            "finish_reason": "length",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        audit_raw = common.canonical_json_bytes(audit)
        audit_name = f"{request_id}.json"
        (fixture.audit_dir / audit_name).write_bytes(audit_raw)
        (fixture.audit_dir / f"{audit_name}.complete").write_bytes(
            common.canonical_json_bytes(
                {
                    "schema_version": capture.AUDIT_COMPLETION_VERSION,
                    "artifact_filename": audit_name,
                    "artifact_sha256": hashlib.sha256(audit_raw).hexdigest(),
                }
            )
        )
        event_name = f"{request_id}.fallback.json"
        event = {
            "schema_version": capture.FALLBACK_EVENT_VERSION,
            "candidate_id": fixture.candidate_id,
            "runtime_identity": {
                "configuration_profile": fixture.profile,
                "configuration_sha256": fixture.configuration_sha256,
            },
            "process_identity": {"pid": fixture.pid, "start_ticks": fixture.ticks},
            "server_request_id": request_id,
            "generation_audit": {
                "artifact_filename": audit_name,
                "artifact_sha256": hashlib.sha256(audit_raw).hexdigest(),
            },
            "fallback_selections": selections,
        }
        event_raw = common.canonical_json_bytes(event)
        (fixture.audit_dir / event_name).write_bytes(event_raw)
        (fixture.audit_dir / f"{event_name}.complete").write_bytes(
            common.canonical_json_bytes(
                {
                    "schema_version": capture.FALLBACK_EVENT_COMPLETION_VERSION,
                    "artifact_filename": event_name,
                    "artifact_sha256": hashlib.sha256(event_raw).hexdigest(),
                }
            )
        )

    def _fallback_capture(self) -> str:
        fixture = self.fixture
        contract = {
            "schema_version": capture.FALLBACK_CONTRACT_VERSION,
            "candidate_id": fixture.candidate_id,
            "configuration_profile": fixture.profile,
            "scenarios": [
                {
                    "scenario_id": capture.FALLBACK_SCENARIO_ID,
                    "completion_request": {
                        "model": "fixture-model",
                        "prompt": "fallback fixture",
                        "max_tokens": 1,
                        "temperature": 1,
                        "top_p": 1,
                        "seed": 1,
                        "stream": False,
                    },
                }
            ],
        }
        contract_path = fixture.base / "fallback-contract.json"
        contract_path.write_bytes(common.canonical_json_bytes(contract))
        request_id = "cmpl-1"
        self._fallback_audit(request_id)
        body = common.canonical_json_bytes({"id": request_id, "object": "text_completion"})
        head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        listener = capture.Listener(
            tcp=v4_tests._proc_tcp(fixture.port, fixture.inode),
            inode=fixture.inode,
            sockets=(fixture.inode,),
        )
        request = capture.CaptureRequest(
            endpoint=capture.parse_endpoint(f"http://127.0.0.1:{fixture.port}/v1/completions"),
            server_pid=fixture.pid,
            candidate_id=fixture.candidate_id,
            configuration_profile=fixture.profile,
            configuration_sha256=fixture.configuration_sha256,
            evidence_root=fixture.root,
            capture_name="fallback-capture",
            audit_dir_name="source-audit",
            scenario_contract=contract_path,
            audit_wait_seconds=0.2,
        )
        socket = v4_tests.FakeSocket([head + body, b""])
        with mock.patch.object(capture.socket, "create_connection", return_value=socket), mock.patch.object(
            capture,
            "_server_stat",
            return_value=(v4_tests._proc_stat(fixture.pid, fixture.ticks), fixture.ticks),
        ), mock.patch.object(capture, "_bound_listener", return_value=listener):
            capture.capture_raw_scenarios(request, repository_root=fixture.repository)
        return "fallback-capture/session.json"

    def _request(self) -> tuple[str, dict]:
        fixture = self.fixture
        request = {
            "schema_version": binder.BIND_REQUEST_VERSION,
            "candidate_id": fixture.candidate_id,
            "binding_inputs": {
                "freeze_sha256": "a" * 64,
                "base_release_candidate_report_sha256": "b" * 64,
                "configuration_profile": fixture.profile,
            },
            "configuration_evidence": fixture._configuration_evidence(),
            "scenario_capture_session_path": self._fallback_capture(),
            "scenarios": [
                {
                    "scenario_id": capture.FALLBACK_SCENARIO_ID,
                    "observation_session_path": fixture._observation_session("fallback-observation"),
                }
            ],
        }
        path = "requests/v5-bind-request.json"
        fixture.tree.put(path, request)
        return path, request

    def _set_effective_sampling_backend(self, backend: str) -> None:
        """Rebind the complete config endpoint/bridge fixture to one backend."""

        fixture = self.fixture
        endpoint_path = fixture.root / "config/endpoint.json"
        endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
        endpoint["effective_config"]["sampling_backend"] = backend
        endpoint["effective_config_sha256"] = hashlib.sha256(
            runtime_config.canonical_json_bytes(endpoint["effective_config"])
        ).hexdigest()
        endpoint_raw = fixture.tree.put("config/endpoint.json", endpoint)
        startup_path = fixture.root / "config/startup.json"
        startup = json.loads(startup_path.read_text(encoding="utf-8"))
        startup["endpoint_payload"] = endpoint
        startup["endpoint_payload_sha256"] = hashlib.sha256(endpoint_raw).hexdigest()
        fixture.tree.put("config/startup.json", startup)
        head_raw = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(endpoint_raw)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        session_path = "config-bridge/session.json"
        session = json.loads((fixture.root / session_path).read_text(encoding="utf-8"))
        session["endpoint"]["body_sha256"] = hashlib.sha256(endpoint_raw).hexdigest()
        session["endpoint"]["body_byte_length"] = len(endpoint_raw)
        session["endpoint"]["response_head"] = fixture.tree.descriptor(
            "config-bridge/raw/config-response-head.http",
            fixture.tree.put("config-bridge/raw/config-response-head.http", head_raw),
        )
        fixture.tree.put(session_path, session)

    def _rewrite_fallback_event(self, mutate: callable) -> None:
        """Rebind the capture-v2 index/session after a hostile event edit."""

        fixture = self.fixture
        event_path = "source-audit/cmpl-1.fallback.json"
        event = json.loads((fixture.root / event_path).read_text(encoding="utf-8"))
        mutate(event)
        event_raw = fixture.tree.put(event_path, event)
        marker_path = f"{event_path}.complete"
        fixture.tree.put(
            marker_path,
            {
                "schema_version": capture.FALLBACK_EVENT_COMPLETION_VERSION,
                "artifact_filename": Path(event_path).name,
                "artifact_sha256": hashlib.sha256(event_raw).hexdigest(),
            },
        )
        index_path = "fallback-capture/000000-exact-backend-fallback.generation-audit-index.json"
        index = json.loads((fixture.root / index_path).read_text(encoding="utf-8"))
        index["fallback_event"] = fixture.tree.descriptor(event_path, event_raw)
        marker_raw = (fixture.root / marker_path).read_bytes()
        index["fallback_completion_marker"] = fixture.tree.descriptor(marker_path, marker_raw)
        index_raw = fixture.tree.put(index_path, index)
        session_path = "fallback-capture/session.json"
        session = json.loads((fixture.root / session_path).read_text(encoding="utf-8"))
        session["scenarios"][0]["fallback_event_log"] = fixture.tree.descriptor(event_path, event_raw)
        session["scenarios"][0]["generation_audit_index"] = fixture.tree.descriptor(index_path, index_raw)
        fixture.tree.put(session_path, session)

    def _rewrite_fallback_marker(self, mutate: callable) -> None:
        fixture = self.fixture
        marker_path = "source-audit/cmpl-1.fallback.json.complete"
        marker = json.loads((fixture.root / marker_path).read_text(encoding="utf-8"))
        mutate(marker)
        marker_raw = fixture.tree.put(marker_path, marker)
        index_path = "fallback-capture/000000-exact-backend-fallback.generation-audit-index.json"
        index = json.loads((fixture.root / index_path).read_text(encoding="utf-8"))
        index["fallback_completion_marker"] = fixture.tree.descriptor(marker_path, marker_raw)
        index_raw = fixture.tree.put(index_path, index)
        session_path = "fallback-capture/session.json"
        session = json.loads((fixture.root / session_path).read_text(encoding="utf-8"))
        session["scenarios"][0]["generation_audit_index"] = fixture.tree.descriptor(index_path, index_raw)
        fixture.tree.put(session_path, session)

    def test_binds_v2_fallback_capture_and_publishes_v5_marker(self) -> None:
        request_path, _request = self._request()
        report = self._bind(request_path, "fallback-v5.json")
        manifest_raw = (self.fixture.root / "fallback-v5.json").read_bytes()
        manifest = common.parse_canonical_json(manifest_raw, "v5 manifest")
        marker = common.parse_canonical_json(
            (self.fixture.root / "fallback-v5.json.complete").read_bytes(), "v5 marker"
        )
        final_stat = os.lstat(self.fixture.root / "fallback-v5.json.complete")
        intent_stat = os.lstat(self.fixture.root / "fallback-v5.json.intent")
        self.assertEqual(manifest["schema_version"], checker.SOAK_V5_MANIFEST_VERSION)
        self.assertEqual(manifest["bindings"]["configuration_profile"], checker.MAX_PERFORMANCE_EXACT_PROFILE)
        self.assertEqual(manifest["scenarios"][0]["scenario_id"], capture.FALLBACK_SCENARIO_ID)
        self.assertEqual(
            manifest["scenarios"][0]["fallback_event_log"]["path"],
            "source-audit/cmpl-1.fallback.json",
        )
        self.assertEqual(marker["schema_version"], checker.SOAK_V5_COMPLETION_MARKER_VERSION)
        self.assertEqual(marker["artifact_sha256"], hashlib.sha256(manifest_raw).hexdigest())
        self.assertEqual((final_stat.st_dev, final_stat.st_ino), (intent_stat.st_dev, intent_stat.st_ino))
        self.assertEqual(final_stat.st_nlink, 2)
        self.assertEqual(report["schema_version"], checker.SOAK_V5_REPORT_VERSION)
        self.assertEqual(
            checker.verify_completed_soak_provenance_v5(self.fixture.root, "fallback-v5.json"),
            report,
        )

    def test_rejects_cpu_effective_backend_before_manifest_publication(self) -> None:
        request_path, _request = self._request()
        self._set_effective_sampling_backend("cpu")
        with self.assertRaises(binder.RawSoakBindError) as raised:
            self._bind(request_path, "cpu-v5.json")
        self.assert_reason(raised, "effective-sampling-backend-mismatch")
        self.assertFalse((self.fixture.root / "cpu-v5.json").exists())

    def test_rejects_historical_capture_version_and_retained_incomplete_marker(self) -> None:
        request_path, _request = self._request()
        session_path = self.fixture.root / "fallback-capture/session.json"
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["schema_version"] = capture.CAPTURE_VERSION
        self.fixture.tree.put("fallback-capture/session.json", session)
        with self.assertRaises(binder.RawSoakBindError) as raised:
            self._bind(request_path, "historical-v5.json")
        self.assert_reason(raised, "historical-fallback-scenario-capture-version-rejected")
        session["schema_version"] = capture.FALLBACK_CAPTURE_VERSION
        self.fixture.tree.put("fallback-capture/session.json", session)
        self.fixture.tree.put(
            "fallback-capture/capture-incomplete.json",
            {
                "schema_version": capture.FALLBACK_INCOMPLETE_MARKER_VERSION,
                "capture_name": "fallback-capture",
            },
        )
        with self.assertRaises(binder.RawSoakBindError) as raised:
            self._bind(request_path, "incomplete-v5.json")
        self.assert_reason(raised, "incomplete-capture")

    def test_rejects_rebound_source_event_selection_drift(self) -> None:
        request_path, _request = self._request()
        self._rewrite_fallback_event(
            lambda event: event["fallback_selections"][0].update(
                {"ineligibility_reason": "other"}
            )
        )
        with self.assertRaises(binder.RawSoakBindError) as raised:
            self._bind(request_path, "event-drift-v5.json")
        self.assert_reason(raised, "source-fallback-selection-mismatch")
        self.assertFalse((self.fixture.root / "event-drift-v5.json").exists())

    def test_rejects_rebound_source_audit_sha_and_marker_hash_drift(self) -> None:
        request_path, _request = self._request()
        self._rewrite_fallback_event(
            lambda event: event["generation_audit"].update({"artifact_sha256": "e" * 64})
        )
        with self.assertRaises(binder.RawSoakBindError) as audit_raised:
            self._bind(request_path, "audit-sha-drift-v5.json")
        self.assert_reason(audit_raised, "source-fallback-audit-binding-mismatch")
        self.assertFalse((self.fixture.root / "audit-sha-drift-v5.json").exists())

        # Create a new fixture because the first hostile edit intentionally
        # leaves capture evidence nonterminal and nonreplaceable.
        self.tearDown()
        self.setUp()
        request_path, _request = self._request()
        self._rewrite_fallback_marker(
            lambda marker: marker.update({"artifact_sha256": "f" * 64})
        )
        with self.assertRaises(binder.RawSoakBindError) as marker_raised:
            self._bind(request_path, "marker-sha-drift-v5.json")
        self.assert_reason(marker_raised, "source-fallback-marker-mismatch")
        self.assertFalse((self.fixture.root / "marker-sha-drift-v5.json").exists())

    def test_rejects_manifest_fallback_descriptor_drift_and_output_collision(self) -> None:
        request_path, _request = self._request()
        self._bind(request_path, "drift-v5.json")
        manifest_path = self.fixture.root / "drift-v5.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["scenarios"][0]["fallback_event_log"] = manifest["scenarios"][0][
            "generation_audit_index"
        ]
        self.fixture.tree.put("drift-v5.json", manifest)
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_soak_provenance_v5(self.fixture.root, "drift-v5.json")
        self.assert_reason(raised, "scenario-capture-derived-leaf-mismatch")
        self.fixture.tree.put("taken-v5.json", {"occupied": True})
        with self.assertRaises(binder.RawSoakBindError) as collision:
            self._bind(request_path, "taken-v5.json")
        self.assert_reason(collision, "output-name-collision")

    def test_rejects_hardlinked_source_marker_and_terminal_third_link(self) -> None:
        request_path, _request = self._request()
        source_marker = self.fixture.root / "source-audit/cmpl-1.fallback.json.complete"
        os.link(source_marker, self.fixture.root / "source-audit/fallback-marker-alias.json")
        with self.assertRaises(binder.RawSoakBindError) as source_raised:
            self._bind(request_path, "source-link-v5.json")
        self.assert_reason(source_raised, "nonunique-evidence-inode")

        # A fresh fixture is unnecessary: the failed pre-publication bind
        # created no output and source evidence is unrelated to the terminal
        # pair checked below.
        os.unlink(self.fixture.root / "source-audit/fallback-marker-alias.json")
        self._bind(request_path, "third-link-v5.json")
        os.link(
            self.fixture.root / "third-link-v5.json.complete",
            self.fixture.root / "third-link-v5.json.marker-alias",
        )
        with self.assertRaises(checker.C02ProvenanceError) as terminal_raised:
            checker.verify_completed_soak_provenance_v5(
                self.fixture.root, "third-link-v5.json"
            )
        self.assert_reason(terminal_raised, "invalid-paired-hardlink")

    def test_rejects_symlinked_terminal_completion_marker(self) -> None:
        request_path, _request = self._request()
        self._bind(request_path, "symlink-v5.json")
        final_marker = self.fixture.root / "symlink-v5.json.complete"
        final_marker.unlink()
        final_marker.symlink_to("symlink-v5.json.intent")
        with self.assertRaises(checker.C02ProvenanceError) as raised:
            checker.verify_completed_soak_provenance_v5(self.fixture.root, "symlink-v5.json")
        self.assert_reason(raised, "invalid-paired-hardlink")

    def test_v5_verifier_rejects_unpaired_and_content_mismatched_markers(self) -> None:
        request_path, _request = self._request()
        name = "marker-shape-v5.json"
        self._bind(request_path, name)
        final = self.fixture.root / f"{name}.complete"
        intent = self.fixture.root / f"{name}.intent"
        intent.unlink()
        with self.assertRaises(checker.C02ProvenanceError) as missing:
            checker.verify_completed_soak_provenance_v5(self.fixture.root, name)
        self.assert_reason(missing, "missing-soak-v5-completion-marker")

        final.unlink()
        marker = {
            "schema_version": checker.SOAK_V5_COMPLETION_MARKER_VERSION,
            "artifact_filename": name,
            "artifact_sha256": hashlib.sha256((self.fixture.root / name).read_bytes()).hexdigest(),
        }
        self.fixture._write_marker_pair(name, marker)
        os.unlink(final)
        os.unlink(intent)
        self.fixture._write_marker_pair(
            name,
            {**marker, "artifact_sha256": "f" * 64},
        )
        with self.assertRaises(checker.C02ProvenanceError) as mismatched:
            checker.verify_completed_soak_provenance_v5(self.fixture.root, name)
        self.assert_reason(mismatched, "soak-v5-completion-marker-mismatch")

    def test_v5_final_marker_directory_sync_failure_is_ambiguous(self) -> None:
        request_path, _request = self._request()
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "v5 soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(binder.RawSoakBindError) as raised:
                self._bind(request_path, "ambiguous-v5.json")
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.fixture.root / "ambiguous-v5.json.complete").is_file())
        self.assertTrue((self.fixture.root / "ambiguous-v5.json.intent").is_file())
        self.assertEqual(
            checker.verify_completed_soak_provenance_v5(
                self.fixture.root, "ambiguous-v5.json"
            )["status"],
            "bound",
        )

    def test_keeps_v4_and_v5_capture_manifest_versions_isolated(self) -> None:
        request_path, request = self._request()
        v4_request = {
            **request,
            "schema_version": v4_tests.binder.BIND_REQUEST_VERSION,
        }
        self.fixture.tree.put("requests/v4-with-v2-capture.json", v4_request)
        with self.assertRaises(v4_tests.binder.RawSoakBindError) as v4_raised:
            v4_tests.binder.bind_raw_soak_manifest(
                self.fixture.root,
                "requests/v4-with-v2-capture.json",
                "v4-cannot-widen.json",
            )
        self.assert_reason(v4_raised, "historical-scenario-capture-version-rejected")
        self.assertFalse((self.fixture.root / "v4-cannot-widen.json").exists())

        configuration = self.fixture._configuration_evidence()
        v1_capture = self.fixture._serial_capture()
        observation = self.fixture._observation_session("legacy-v4-observation")
        v4_only_request = {
            "schema_version": v4_tests.binder.BIND_REQUEST_VERSION,
            "candidate_id": self.fixture.candidate_id,
            "binding_inputs": request["binding_inputs"],
            "configuration_evidence": configuration,
            "scenario_capture_session_path": v1_capture,
            "scenarios": [{"scenario_id": "smoke", "observation_session_path": observation}],
        }
        self.fixture.tree.put("requests/valid-v4-only.json", v4_only_request)
        v4_tests.binder.bind_raw_soak_manifest(
            self.fixture.root,
            "requests/valid-v4-only.json",
            "legacy-v4.json",
        )
        with self.assertRaises(checker.C02ProvenanceError) as v5_raised:
            checker.verify_soak_provenance_v5(self.fixture.root, "legacy-v4.json")
        self.assert_reason(v5_raised, "historical-soak-v5-manifest-version-rejected")

    def test_static_binder_and_wrapper_remain_local_only(self) -> None:
        source = Path(binder.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
        ):
            self.assertNotIn(forbidden, source)
        wrapper = Path(__file__).with_name("run_bind_raw_c02_soak_v5.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("bind_raw_c02_soak_v5.py", wrapper)
        for forbidden in ("nvidia-smi", "docker", "podman", "ssh", "curl", "wget", "systemctl"):
            self.assertNotIn(forbidden, wrapper)


if __name__ == "__main__":
    unittest.main()
