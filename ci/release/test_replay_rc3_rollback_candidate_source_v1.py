#!/usr/bin/env python3
"""CPU-only hostile tests for the fixed RC3 candidate/source join."""

from __future__ import annotations

import hashlib
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import capture_rc3_rollback_phase_v1 as phase_capture
import check_c02_provenance_v2 as c02
import check_rc3_rollback_provenance_v3 as rollback
import check_rc3_static_effective_config_v1 as static_effective
import prepare_rc3_rollback_evidence_v1 as preparation
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import test_bind_raw_c02_soak_v4 as c02_fixtures
from test_check_reconstructed_prior_baseline_v2 import BaselineV2Fixture


class CandidateSourceJoinTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = self._new_environment("primary")
        self.addCleanup(self._close_environment, self.environment)

    def _close_environment(self, environment: dict[str, object]) -> None:
        fixture = environment["fixture"]
        assert isinstance(fixture, c02_fixtures.BindRawC02SoakV4Tests)
        fixture.tearDown()

    def _effective_config(self) -> dict[str, object]:
        return {
            "execution_completion_mode": "iteration-batch",
            "batch_shape": {"policy": "power-of-two", "buckets": [1, 8, 64]},
            "metadata_transport": "packed-async",
            "sampling_backend": "gpu-greedy",
            "attention_backend": {
                "prefill": "riley.attention.prefill-v1",
                "decode": "riley.attention.decode-v1",
            },
            "gemm_reduction_policy": "strict-no-split-v1",
            "experimental_flags": {},
            "fallback_policy": {
                "cross_profile_fallback": "forbidden",
                "runtime_selection": "exact-fallback-allowed",
            },
            "batch_token_budget": 64,
            "kv_geometry": {
                "layout": "paged",
                "block_tokens": 16,
                "physical_blocks": 512,
            },
        }

    def _prepare_static(self, fixture: c02_fixtures.BindRawC02SoakV4Tests) -> None:
        BaselineV2Fixture(
            fixture.root,
            target_commit_sha1=rollback.RECONSTRUCTED_ROLLBACK_TARGET,
            tag_object_sha1=rollback.RECONSTRUCTED_ROLLBACK_TAG_OBJECT,
        )
        inputs = fixture.base / "static-inputs"
        inputs.mkdir(mode=0o700)
        os.chmod(inputs, 0o700)
        freeze = inputs / "freeze.raw"
        base_report = inputs / "base-report.raw"
        configuration = inputs / "stable-default-config.json"
        freeze.write_bytes(b'{"freeze":"future"}\n')
        base_report.write_bytes(b'{"base_report":"future"}\n')
        expected = self._effective_config()
        configuration.write_bytes(
            common.canonical_json_bytes(
                {
                    "schema_version": static_effective.STATIC_EFFECTIVE_CONFIG_VERSION,
                    "candidate_id": fixture.candidate_id,
                    "configuration_profile": "stable-default",
                    "expected_effective_config": expected,
                    "expected_effective_config_sha256": hashlib.sha256(
                        common.canonical_json_bytes(expected)
                    ).hexdigest(),
                }
            )
        )
        for source in (freeze, base_report, configuration):
            os.chmod(source, 0o644)
        preparation.prepare_rollback_evidence(
            preparation.EvidencePreparationRequest(
                evidence_root=fixture.root,
                baseline_manifest_path="baseline.json",
                candidate_id=fixture.candidate_id,
                freeze_input=freeze,
                base_release_candidate_report_input=base_report,
                stable_default_configuration_input=configuration,
            )
        )

    def _capture_candidate_phase(
        self,
        fixture: c02_fixtures.BindRawC02SoakV4Tests,
        *,
        generation: bool,
        port: int | None = None,
        inode: int | None = None,
    ) -> None:
        phase_port = fixture.port if port is None else port
        phase_inode = fixture.inode if inode is None else inode
        endpoint = phase_capture.parse_endpoint(f"http://127.0.0.1:{phase_port}")
        generation_body = (
            common.canonical_json_bytes(
                {
                    "model": "fixture-model",
                    "prompt": "candidate",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 3,
                    "stream": False,
                }
            )
            if generation
            else None
        )
        request = phase_capture.CaptureRequest(
            endpoint=endpoint,
            server_pid=fixture.pid,
            gpu_index=0,
            evidence_root=fixture.root,
            capture_name=candidate_source.CANDIDATE_PHASE_CAPTURE_NAME,
            generation_body=generation_body,
        )
        target = phase_capture.TargetIdentity(
            fixture.pid,
            fixture.ticks,
            phase_port,
            phase_inode,
            0,
            fixture.gpu_uuid,
        )
        socket_snapshot = common.canonical_json_bytes(
            {
                "schema_version": phase_capture.c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": fixture.pid,
                "socket_inodes": [phase_inode],
            }
        )
        listener = phase_capture.c02.BoundListener(
            proc_net_tcp=c02_fixtures._proc_tcp(phase_port, phase_inode),
            socket_inode=phase_inode,
            server_socket_inodes=(phase_inode,),
        )
        health_body = b"ready\n"
        health_head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(health_body)}\r\n"
            "Content-Type: text/plain\r\n\r\n"
        ).encode("ascii")
        exchanges: list[tuple[bytes, bytes, bytes]] = [
            (
                phase_capture._request_bytes("GET", endpoint, "/readyz", b""),
                health_head,
                health_body,
            )
        ]
        if generation_body is not None:
            generation_response = common.canonical_json_bytes({"id": "cmpl-candidate"})
            generation_head = (
                f"HTTP/1.1 200 OK\r\nContent-Length: {len(generation_response)}\r\n"
                "Content-Type: application/json\r\n\r\n"
            ).encode("ascii")
            exchanges.append(
                (
                    phase_capture._request_bytes(
                        "POST", endpoint, "/v1/completions", generation_body
                    ),
                    generation_head,
                    generation_response,
                )
            )
        with mock.patch.object(phase_capture, "_preflight_target", return_value=target), mock.patch.object(
            phase_capture.c02,
            "_capture_server_stat",
            side_effect=[
                (c02_fixtures._proc_stat(fixture.pid, fixture.ticks), fixture.ticks),
                (c02_fixtures._proc_stat(fixture.pid, fixture.ticks), fixture.ticks),
            ],
        ), mock.patch.object(
            phase_capture.c02,
            "_capture_bound_listener",
            side_effect=[listener, listener],
        ), mock.patch.object(
            phase_capture.c02,
            "_socket_snapshot_raw",
            side_effect=[socket_snapshot, socket_snapshot],
        ), mock.patch.object(
            phase_capture.c02,
            "_capture_gpu",
            return_value=(
                f"0, {fixture.gpu_uuid}\n".encode("ascii"),
                f"{fixture.pid}, 42\n".encode("ascii"),
                fixture.gpu_uuid,
            ),
        ), mock.patch.object(
            phase_capture.c02,
            "_capture_server_status",
            return_value=f"Name:\triley\nPid:\t{fixture.pid}\n".encode("ascii"),
        ), mock.patch.object(
            phase_capture,
            "_capture_exchange",
            side_effect=exchanges,
        ):
            phase_capture.capture_phase(request)

    def _write_shutdown(
        self,
        fixture: c02_fixtures.BindRawC02SoakV4Tests,
        *,
        start_ticks: int | None = None,
    ) -> None:
        artifact = {
            "schema_version": c02.SHUTDOWN_VERSION,
            "capture_status": "captured",
            "qualification_status": "not-run",
            "server_pid": fixture.pid,
            "server_start_ticks": fixture.ticks if start_ticks is None else start_ticks,
            "worker_ready": False,
            "final_metrics": c02_fixtures._metrics(),
        }
        raw = fixture.tree.put(candidate_source.SHUTDOWN_ARTIFACT_PATH, artifact)
        fixture.tree.put(
            candidate_source.SHUTDOWN_MARKER_PATH,
            {
                "schema_version": c02.SHUTDOWN_MARKER_VERSION,
                "artifact_filename": "shutdown.json",
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            },
        )

    def _new_environment(
        self,
        name: str,
        *,
        scenario_ids: tuple[str, ...] = ("smoke",),
        candidate_generation: bool = False,
        candidate_port: int | None = None,
        shutdown_ticks: int | None = None,
    ) -> dict[str, object]:
        fixture = c02_fixtures.BindRawC02SoakV4Tests()
        fixture.setUp()
        self._prepare_static(fixture)
        fixture._request(scenario_ids=scenario_ids)
        self._capture_candidate_phase(
            fixture,
            generation=candidate_generation,
            port=candidate_port,
        )
        self._write_shutdown(fixture, start_ticks=shutdown_ticks)
        return {"name": name, "fixture": fixture}

    def _join(
        self,
        environment: dict[str, object] | None = None,
    ) -> candidate_source.ReplayedCandidateSourceJoin:
        environment = self.environment if environment is None else environment
        fixture = environment["fixture"]
        assert isinstance(fixture, c02_fixtures.BindRawC02SoakV4Tests)
        root_fd = common.open_private_evidence_directory(fixture.root, "candidate source test root")
        try:
            return candidate_source._replay_candidate_source_join_on_held_root_fd(root_fd)
        finally:
            os.close(root_fd)

    def assert_reason(self, code: str, call: object) -> None:
        with self.assertRaises(candidate_source.CandidateSourceJoinError) as raised:
            call()  # type: ignore[operator]
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def test_replays_fixed_candidate_source_and_shutdown_join(self) -> None:
        replayed = self._join()
        self.assertIsNone(replayed.candidate_phase.generation)
        self.assertEqual(replayed.static_effective.candidate_id, "riley-0.1.0-rc3")
        self.assertEqual(replayed.source_scenario.scenario_id, "smoke")
        self.assertEqual(
            replayed.generation.request,
            replayed.source_scenario.request,
        )
        self.assertEqual(
            replayed.generation.response_head,
            replayed.source_scenario.response_head,
        )
        self.assertEqual(
            replayed.generation.response_body,
            replayed.source_scenario.response_body,
        )
        self.assertEqual(
            replayed.generation.generation_audit_index,
            replayed.source_scenario.generation_audit_index,
        )
        self.assertEqual(replayed.shutdown.artifact.path, candidate_source.SHUTDOWN_ARTIFACT_PATH)
        self.assertEqual(replayed.shutdown.marker.path, candidate_source.SHUTDOWN_MARKER_PATH)
        self.assertIsInstance(replayed.consumed_paths, frozenset)
        for descriptor in (
            replayed.static_effective.static_bindings.reconstructed_baseline,
            replayed.static_effective.static_bindings.freeze,
            replayed.static_effective.static_bindings.base_release_candidate_report,
            replayed.static_effective.static_bindings.configuration,
            replayed.static_effective.config_bridge.endpoint,
            replayed.static_effective.config_bridge.startup_artifact,
            replayed.static_effective.config_bridge.endpoint_observation,
            replayed.source_capture.session,
            replayed.source_capture.contract,
            replayed.source_scenario.request_ledger,
            replayed.source_scenario.runtime_event_log,
        ):
            self.assertIn(descriptor.path, replayed.consumed_paths)

    def test_rejects_local_generation_multiple_sources_target_and_shutdown_drift(self) -> None:
        local_generation = self._new_environment("local-generation", candidate_generation=True)
        self.addCleanup(self._close_environment, local_generation)
        self.assert_reason(
            "candidate-local-generation-forbidden",
            lambda: self._join(local_generation),
        )

        multiple = self._new_environment("multiple", scenario_ids=("first", "second"))
        self.addCleanup(self._close_environment, multiple)
        self.assert_reason(
            "candidate-source-scenario-count",
            lambda: self._join(multiple),
        )

        target_drift = self._new_environment("target-drift", candidate_port=18081)
        self.addCleanup(self._close_environment, target_drift)
        self.assert_reason(
            "candidate-config-bridge-target-mismatch",
            lambda: self._join(target_drift),
        )

        shutdown_drift = self._new_environment("shutdown-drift", shutdown_ticks=2223)
        self.addCleanup(self._close_environment, shutdown_drift)
        self.assert_reason(
            "shutdown-target-mismatch",
            lambda: self._join(shutdown_drift),
        )

    def test_rejects_redirected_source_audit_and_has_no_operational_surface(self) -> None:
        fixture = self.environment["fixture"]
        assert isinstance(fixture, c02_fixtures.BindRawC02SoakV4Tests)
        fixture._redirect_source_audit(0, "other-source-audit")
        self.assert_reason(
            "candidate-source-audit-directory-mismatch",
            self._join,
        )

        source = Path(candidate_source.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "def _replay_candidate_source_join_on_held_root_fd",
            source,
        )
        self.assertEqual(
            list(
                inspect.signature(
                    candidate_source._replay_candidate_source_join_on_held_root_fd
                ).parameters
            ),
            ["root_fd"],
        )
        for forbidden in (
            "import argparse",
            "def main(",
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
