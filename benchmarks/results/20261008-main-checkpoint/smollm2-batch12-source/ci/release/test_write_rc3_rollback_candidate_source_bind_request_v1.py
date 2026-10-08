#!/usr/bin/env python3
"""CPU-only hostile tests for the fixed RC3 candidate/source v3 request writer."""

from __future__ import annotations

import ast
import dataclasses
import fcntl
import hashlib
import inspect
import os
import unittest
from pathlib import Path
from unittest import mock

import bind_raw_rc3_rollback_capture as v3_binder
import capture_rc3_rollback_atomic_switch_v1 as atomic
import capture_rc3_rollback_phase_v1 as phase_capture
import check_rc3_rollback_provenance_v3 as v3_checker
import check_rc3_static_effective_config_v1 as static_effective
import prepare_rc3_rollback_artifacts_v1 as artifact_prepare
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import test_bind_raw_c02_soak_v4 as c02_fixtures
import test_replay_rc3_rollback_candidate_source_v1 as candidate_fixtures
import write_rc3_rollback_candidate_source_bind_request_v1 as writer
import capture_rc3_rollback_atomic_transaction_v1 as transaction


def _fake_exchange(directory_fd: int) -> None:
    """Portable test double; production capture remains renameat2-only."""

    temporary = "writer-test-exchange-temporary"
    os.rename(
        artifact_prepare.ACTIVE_NAME,
        temporary,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(
        artifact_prepare.ROLLBACK_STAGED_NAME,
        artifact_prepare.ACTIVE_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )
    os.rename(
        temporary,
        artifact_prepare.ROLLBACK_STAGED_NAME,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
    )


class WriteRollbackCandidateSourceBindRequestTests(unittest.TestCase):
    rollback_pid = 3333
    rollback_ticks = 4444
    rollback_port = 18081
    rollback_inode = 7002

    def setUp(self) -> None:
        self.candidate_fixture = candidate_fixtures.CandidateSourceJoinTests(
            methodName="runTest"
        )
        self.candidate_fixture.setUp()
        environment = self.candidate_fixture.environment
        fixture = environment["fixture"]
        assert isinstance(fixture, c02_fixtures.BindRawC02SoakV4Tests)
        self.fixture = fixture
        self.root = fixture.root
        self._prepare_artifacts_and_transaction()
        self._capture_rollback_phase()

    def tearDown(self) -> None:
        self.candidate_fixture.doCleanups()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def _artifact_input(self, name: str, raw: bytes, mode: int) -> Path:
        inputs = self.fixture.base / "writer-artifact-inputs"
        inputs.mkdir(mode=0o700, exist_ok=True)
        os.chmod(inputs, 0o700)
        path = inputs / name
        path.write_bytes(raw)
        os.chmod(path, mode)
        return path

    def _prepare_artifacts_and_transaction(self) -> None:
        baseline = self.root / "reproductions" / "a"
        rollback_binary = baseline / "riley-server"
        rollback_bundle = baseline / "riley.bundle.tar.zst"
        rollback_image = baseline / "docker-image-inspect.json"
        request = artifact_prepare.PreparationRequest(
            evidence_root=self.root,
            candidate_binary=self._artifact_input(
                "candidate-binary",
                b"candidate writer binary\n",
                0o700,
            ),
            candidate_bundle=self._artifact_input(
                "candidate-bundle",
                b"candidate writer bundle\n",
                0o600,
            ),
            candidate_image_inspect=self._artifact_input(
                "candidate-image-inspect.json",
                rollback_image.read_bytes(),
                0o600,
            ),
            rollback_binary=self._artifact_input(
                "rollback-binary",
                rollback_binary.read_bytes(),
                0o700,
            ),
            rollback_bundle=self._artifact_input(
                "rollback-bundle",
                rollback_bundle.read_bytes(),
                0o600,
            ),
            rollback_image_inspect=self._artifact_input(
                "rollback-image-inspect.json",
                rollback_image.read_bytes(),
                0o600,
            ),
        )
        artifact_prepare.prepare_artifacts(request)
        with mock.patch.object(atomic, "_rename_exchange", side_effect=_fake_exchange):
            transaction.capture_atomic_transaction(self.root)

    def _capture_rollback_phase(self) -> None:
        endpoint = phase_capture.parse_endpoint(
            f"http://127.0.0.1:{self.rollback_port}"
        )
        generation_body = common.canonical_json_bytes(
            {
                "model": "fixture-model",
                "prompt": "rollback",
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": 4,
                "stream": False,
            }
        )
        request = phase_capture.CaptureRequest(
            endpoint=endpoint,
            server_pid=self.rollback_pid,
            gpu_index=0,
            evidence_root=self.root,
            capture_name=writer.ROLLBACK_PHASE_CAPTURE_NAME,
            generation_body=generation_body,
        )
        target = phase_capture.TargetIdentity(
            self.rollback_pid,
            self.rollback_ticks,
            self.rollback_port,
            self.rollback_inode,
            0,
            self.fixture.gpu_uuid,
        )
        socket_snapshot = common.canonical_json_bytes(
            {
                "schema_version": phase_capture.c02.SOCKET_SNAPSHOT_VERSION,
                "server_pid": self.rollback_pid,
                "socket_inodes": [self.rollback_inode],
            }
        )
        listener = phase_capture.c02.BoundListener(
            proc_net_tcp=c02_fixtures._proc_tcp(
                self.rollback_port,
                self.rollback_inode,
            ),
            socket_inode=self.rollback_inode,
            server_socket_inodes=(self.rollback_inode,),
        )
        health_body = b"ready\n"
        health_head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(health_body)}\r\n"
            "Content-Type: text/plain\r\n\r\n"
        ).encode("ascii")
        generation_body_response = common.canonical_json_bytes({"id": "cmpl-rollback"})
        generation_head = (
            f"HTTP/1.1 200 OK\r\nContent-Length: {len(generation_body_response)}\r\n"
            "Content-Type: application/json\r\n\r\n"
        ).encode("ascii")
        exchanges = [
            (
                phase_capture._request_bytes("GET", endpoint, "/readyz", b""),
                health_head,
                health_body,
            ),
            (
                phase_capture._request_bytes(
                    "POST",
                    endpoint,
                    "/v1/completions",
                    generation_body,
                ),
                generation_head,
                generation_body_response,
            ),
        ]
        with mock.patch.object(
            phase_capture,
            "_preflight_target",
            return_value=target,
        ), mock.patch.object(
            phase_capture.c02,
            "_capture_server_stat",
            side_effect=[
                (
                    c02_fixtures._proc_stat(
                        self.rollback_pid,
                        self.rollback_ticks,
                    ),
                    self.rollback_ticks,
                ),
                (
                    c02_fixtures._proc_stat(
                        self.rollback_pid,
                        self.rollback_ticks,
                    ),
                    self.rollback_ticks,
                ),
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
                f"0, {self.fixture.gpu_uuid}\n".encode("ascii"),
                f"{self.rollback_pid}, 42\n".encode("ascii"),
                self.fixture.gpu_uuid,
            ),
        ), mock.patch.object(
            phase_capture.c02,
            "_capture_server_status",
            return_value=(
                f"Name:\triley\nPid:\t{self.rollback_pid}\n"
            ).encode("ascii"),
        ), mock.patch.object(
            phase_capture,
            "_capture_exchange",
            side_effect=exchanges,
        ):
            phase_capture.capture_phase(request)

    def _write_on_held_fds(
        self,
    ) -> writer.WrittenCandidateSourceBindRequest:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "fixed candidate-source writer test root",
        )
        switch_fd: int | None = None
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            switch_fd = common.open_private_child_directory(
                root_fd,
                artifact_prepare.SWITCH_DIRECTORY_NAME,
                "fixed candidate-source writer test switch",
            )
            fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds(  # noqa: SLF001
                root_fd,
                switch_fd,
            )
        finally:
            if switch_fd is not None:
                fcntl.flock(switch_fd, fcntl.LOCK_UN)
                os.close(switch_fd)
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def test_writes_fixed_request_from_actual_replays_and_v3_binds_it(self) -> None:
        written = self._write_on_held_fds()
        request = written.request
        request_path = self.root / writer.BIND_REQUEST_NAME
        self.assertTrue(request_path.is_file())
        self.assertEqual(request_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(request_path.stat().st_nlink, 1)
        self.assertFalse((self.root / f"{writer.BIND_REQUEST_NAME}.intent").exists())
        self.assertFalse((self.root / f"{writer.BIND_REQUEST_NAME}.complete").exists())
        self.assertEqual(
            request["candidate"]["generation"]["request_path"],
            written.candidate_source.generation.request.path,
        )
        self.assertEqual(
            request["candidate"]["generation"]["response_head_path"],
            written.candidate_source.generation.response_head.path,
        )
        self.assertEqual(
            request["candidate"]["generation"]["response_body_path"],
            written.candidate_source.generation.response_body.path,
        )
        self.assertEqual(
            request["candidate"]["generation_audit_index_path"],
            written.candidate_source.generation.generation_audit_index.path,
        )
        self.assertIsNone(written.candidate_source.candidate_phase.generation)
        self.assertIsNotNone(written.rollback_phase.generation)
        self.assertEqual(
            request["binding_evidence"]["configuration_path"],
            written.static_bindings.configuration.path,
        )
        static_configuration = (self.root / written.static_bindings.configuration.path).read_bytes()
        self.assertNotEqual(
            hashlib.sha256(static_configuration).hexdigest(),
            written.candidate_source.static_effective.runtime_configuration_sha256,
        )

        root_fd = common.open_private_evidence_directory(
            self.root,
            "fixed candidate-source writer v3 bind root",
        )
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = v3_binder._bind_raw_rollback_manifest_held_locked_fd(  # noqa: SLF001
                root_fd,
                writer.BIND_REQUEST_NAME,
                "writer-v3-manifest.json",
            )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(
            report["bindings"]["configuration_sha256"],
            hashlib.sha256(static_configuration).hexdigest(),
        )

    def test_static_checkpoint_failure_leaves_no_fixed_request(self) -> None:
        error = static_effective.StaticEffectiveConfigError(
            "fixture static preparation descriptor drift"
        )
        error.reason_code = "static-preparation-descriptor-drift"  # type: ignore[attr-defined]
        original = static_effective._recheck_static_preparation_bindings_on_held_root_fd
        calls = 0

        def fail_terminal_recheck(
            root_fd: int,
            expected: static_effective.StaticPreparationBindings,
        ) -> static_effective.StaticPreparationBindings:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise error
            return original(root_fd, expected)

        with mock.patch.object(
            writer.static_effective,
            "_recheck_static_preparation_bindings_on_held_root_fd",
            side_effect=fail_terminal_recheck,
        ):
            with self.assertRaises(
                writer.RollbackCandidateSourceBindRequestError
            ) as raised:
                self._write_on_held_fds()
        self.assert_reason(raised, "static-preparation-descriptor-drift")
        self.assertFalse((self.root / writer.BIND_REQUEST_NAME).exists())

    def test_rejects_missing_rollback_generation_before_request_publication(self) -> None:
        original = phase_capture.replay_rc3_rollback_phase_v1_fd

        def no_rollback_generation(
            root_fd: int,
            capture_name: str,
        ) -> phase_capture.ReplayedPhaseCapture:
            replayed = original(root_fd, capture_name)
            if capture_name == writer.ROLLBACK_PHASE_CAPTURE_NAME:
                return dataclasses.replace(replayed, generation=None)
            return replayed

        with mock.patch.object(
            writer.phase_capture,
            "replay_rc3_rollback_phase_v1_fd",
            side_effect=no_rollback_generation,
        ):
            with self.assertRaises(
                writer.RollbackCandidateSourceBindRequestError
            ) as raised:
                self._write_on_held_fds()
        self.assert_reason(raised, "rollback-generation-required")
        self.assertFalse((self.root / writer.BIND_REQUEST_NAME).exists())

    def test_rejects_candidate_source_inventory_alias_with_atomic_artifact(self) -> None:
        original = candidate_source._replay_candidate_source_join_on_held_root_fd

        def forged_inventory(
            root_fd: int,
        ) -> candidate_source.ReplayedCandidateSourceJoin:
            replayed = original(root_fd)
            return dataclasses.replace(
                replayed,
                consumed_paths=frozenset(
                    set(replayed.consumed_paths)
                    | {f"{artifact_prepare.ARTIFACT_DIRECTORY_NAME}/candidate-binary"}
                ),
            )

        with mock.patch.object(
            writer.candidate_source,
            "_replay_candidate_source_join_on_held_root_fd",
            side_effect=forged_inventory,
        ):
            with self.assertRaises(
                writer.RollbackCandidateSourceBindRequestError
            ) as raised:
                self._write_on_held_fds()
        self.assert_reason(raised, "duplicate-evidence-path")
        self.assertFalse((self.root / writer.BIND_REQUEST_NAME).exists())

    def test_fixed_output_collision_fails_before_input_replay(self) -> None:
        (self.root / writer.BIND_REQUEST_NAME).write_bytes(b"occupied\n")
        os.chmod(self.root / writer.BIND_REQUEST_NAME, 0o600)
        with mock.patch.object(
            candidate_source,
            "_replay_candidate_source_join_on_held_root_fd",
            side_effect=AssertionError("input replay must not run after output collision"),
        ):
            with self.assertRaises(
                writer.RollbackCandidateSourceBindRequestError
            ) as raised:
                self._write_on_held_fds()
        self.assert_reason(raised, "output-name-collision")

    def test_held_fds_are_not_reopened_or_relocked(self) -> None:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "fixed candidate-source writer held FD root",
        )
        switch_fd = common.open_private_child_directory(
            root_fd,
            artifact_prepare.SWITCH_DIRECTORY_NAME,
            "fixed candidate-source writer held FD switch",
        )
        real_flock = fcntl.flock
        try:
            real_flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            real_flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(
                common,
                "open_private_evidence_directory",
                side_effect=AssertionError("writer must not reopen the root path"),
            ), mock.patch.object(
                transaction.fcntl,
                "flock",
                side_effect=AssertionError("writer must not change caller-owned locks"),
            ):
                written = writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                )
        finally:
            real_flock(switch_fd, fcntl.LOCK_UN)
            real_flock(root_fd, fcntl.LOCK_UN)
            os.close(switch_fd)
            os.close(root_fd)
        self.assertEqual(written.request_descriptor.path, writer.BIND_REQUEST_NAME)

    def test_private_surface_has_no_cli_or_injected_inputs(self) -> None:
        source = Path(writer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        public_functions = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
        }
        self.assertEqual(public_functions, set())
        for forbidden in (
            "import argparse",
            "def main(",
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "publish_create_only_hardlink",
        ):
            self.assertNotIn(forbidden, source)
        self.assertEqual(
            list(
                inspect.signature(
                    writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds  # noqa: SLF001
                ).parameters
            ),
            ["root_fd", "switch_fd"],
        )
        self.assertNotIn("target", writer._write_fixed_candidate_source_bind_request_on_held_root_switch_fds.__annotations__)


if __name__ == "__main__":
    unittest.main()
