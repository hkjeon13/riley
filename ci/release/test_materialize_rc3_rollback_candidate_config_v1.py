#!/usr/bin/env python3
"""CPU-only hostile tests for RC3 fixed candidate config materialization."""

from __future__ import annotations

import ast
import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import materialize_rc3_rollback_candidate_config_v1 as materializer
import test_c02_provenance_v2 as c02_fixtures


class RollbackCandidateConfigMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve() / "evidence"
        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)
        self.c02 = c02_fixtures.C02ProvenanceV2Tests()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def _seed_bridge(self) -> tuple[bytes, bytes]:
        materializer._initialize_candidate_config_directory(self.root)  # noqa: SLF001
        tree = c02_fixtures.EvidenceTree(self.root)
        bridge = self.c02.configuration_evidence(
            tree,
            self.c02.bindings,
            bridge_prefix=materializer.CONFIG_BRIDGE_DIRECTORY_NAME,
        )
        endpoint_path = bridge["endpoint"]["path"]
        startup_path = bridge["startup_artifact"]["path"]
        assert isinstance(endpoint_path, str)
        assert isinstance(startup_path, str)
        endpoint_raw = (self.root / endpoint_path).read_bytes()
        startup_raw = (self.root / startup_path).read_bytes()
        captured = self.root / materializer.CAPTURED_ENDPOINT_PATH
        captured.write_bytes(endpoint_raw)
        captured.chmod(0o600)
        startup = self.root / materializer.CONFIG_STARTUP_PATH
        startup.write_bytes(startup_raw)
        startup.chmod(0o600)
        return endpoint_raw, startup_raw

    def _project(self) -> Any:
        return materializer._materialize_candidate_config_bridge(  # noqa: SLF001
            self.root,
            candidate_id=self.c02.candidate,
            configuration_profile="stable-default",
        )

    def test_initializes_and_projects_a_replayed_fixed_config_bridge(self) -> None:
        endpoint_raw, startup_raw = self._seed_bridge()
        replayed = self._project()
        config_directory = self.root / materializer.CONFIG_DIRECTORY_NAME
        endpoint = self.root / materializer.CONFIG_ENDPOINT_PATH
        startup = self.root / materializer.CONFIG_STARTUP_PATH
        self.assertEqual(stat.S_IMODE(config_directory.stat().st_mode), 0o700)
        self.assertEqual(endpoint.read_bytes(), endpoint_raw)
        self.assertEqual(startup.read_bytes(), startup_raw)
        self.assertEqual(replayed.endpoint.path, materializer.CONFIG_ENDPOINT_PATH)
        self.assertEqual(replayed.startup_artifact.path, materializer.CONFIG_STARTUP_PATH)
        self.assertEqual(
            replayed.endpoint_observation.path,
            materializer.CONFIG_BRIDGE_SESSION_PATH,
        )
        self.assertEqual(replayed.candidate_id, self.c02.candidate)
        self.assertEqual(replayed.configuration_profile, "stable-default")

    def test_existing_config_directory_cannot_be_reused(self) -> None:
        materializer._initialize_candidate_config_directory(self.root)  # noqa: SLF001
        with self.assertRaises(
            materializer.RollbackCandidateConfigMaterializationError
        ) as raised:
            materializer._initialize_candidate_config_directory(self.root)  # noqa: SLF001
        self.assert_reason(raised, "output-name-collision")

    def test_fixed_endpoint_collision_stops_before_bridge_replay(self) -> None:
        self._seed_bridge()
        endpoint = self.root / materializer.CONFIG_ENDPOINT_PATH
        endpoint.write_bytes(b"occupied\n")
        endpoint.chmod(0o600)
        with mock.patch.object(
            materializer.config_bridge,
            "replay_config_bridge_v1_fd",
            side_effect=AssertionError("bridge replay must not run after endpoint collision"),
        ):
            with self.assertRaises(
                materializer.RollbackCandidateConfigMaterializationError
            ) as raised:
                self._project()
        self.assert_reason(raised, "output-name-collision")

    def test_config_sidecar_stops_before_bridge_replay(self) -> None:
        self._seed_bridge()
        sidecar = self.root / materializer.CONFIG_DIRECTORY_NAME / "stale.json"
        sidecar.write_bytes(b"{}")
        sidecar.chmod(0o600)
        with mock.patch.object(
            materializer.config_bridge,
            "replay_config_bridge_v1_fd",
            side_effect=AssertionError("bridge replay must not run with a config sidecar"),
        ):
            with self.assertRaises(
                materializer.RollbackCandidateConfigMaterializationError
            ) as raised:
                self._project()
        self.assert_reason(raised, "unexpected-config-inventory")
        self.assertFalse((self.root / materializer.CONFIG_ENDPOINT_PATH).exists())

    def test_invalid_captured_endpoint_leaves_no_fixed_endpoint(self) -> None:
        self._seed_bridge()
        captured = self.root / materializer.CAPTURED_ENDPOINT_PATH
        captured.write_bytes(b"{}")
        captured.chmod(0o600)
        with self.assertRaises(materializer.RollbackCandidateConfigMaterializationError):
            self._project()
        self.assertFalse((self.root / materializer.CONFIG_ENDPOINT_PATH).exists())

    def test_nonstable_profile_is_rejected_before_evidence_access(self) -> None:
        with mock.patch.object(
            materializer.common,
            "open_private_evidence_directory",
            side_effect=AssertionError("nonstable profile must fail before opening evidence"),
        ):
            with self.assertRaises(
                materializer.RollbackCandidateConfigMaterializationError
            ) as raised:
                materializer._materialize_candidate_config_bridge(  # noqa: SLF001
                    self.root,
                    candidate_id=self.c02.candidate,
                    configuration_profile="max-performance-exact",
                )
        self.assert_reason(raised, "invalid-configuration-profile")

    def test_second_held_fd_replay_detects_startup_drift(self) -> None:
        self._seed_bridge()
        original = materializer.config_bridge.replay_config_bridge_v1_fd
        calls = 0

        def mutate_after_preflight(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            replayed = original(*args, **kwargs)
            if calls == 1:
                startup = self.root / materializer.CONFIG_STARTUP_PATH
                startup.write_bytes(b"{}")
                startup.chmod(0o600)
            return replayed

        with mock.patch.object(
            materializer.config_bridge,
            "replay_config_bridge_v1_fd",
            side_effect=mutate_after_preflight,
        ):
            with self.assertRaises(materializer.RollbackCandidateConfigMaterializationError):
                self._project()
        self.assertTrue((self.root / materializer.CONFIG_ENDPOINT_PATH).is_file())

    def test_private_surface_has_no_cli_or_operational_imports(self) -> None:
        source = Path(materializer.__file__).read_text(encoding="utf-8")
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
            "systemctl ",
            "def resume",
            "def retry",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
