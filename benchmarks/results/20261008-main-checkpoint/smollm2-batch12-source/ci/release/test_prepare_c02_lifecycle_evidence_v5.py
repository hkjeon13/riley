#!/usr/bin/env python3
"""CPU-only hostile tests for the closed native-fallback lifecycle-v5 preparer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare_c02_lifecycle_evidence_v5 as prepare
import provenance_v2_common as common


class PrepareC02LifecycleEvidenceV5Tests(unittest.TestCase):
    candidate_id = "riley-0.1.0-rc3"
    profile = prepare.PROFILE

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.contract_path = self.base / "fallback-scenario.json"
        self.contract_path.write_bytes(self._contract())
        self.root = self.base / "evidence"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _contract(self, scenarios: list[dict] | None = None) -> bytes:
        return common.canonical_json_bytes(
            {
                "schema_version": prepare.CONTRACT_VERSION,
                "candidate_id": self.candidate_id,
                "configuration_profile": self.profile,
                "scenarios": scenarios
                if scenarios is not None
                else [
                    {
                        "scenario_id": prepare.FALLBACK_SCENARIO_ID,
                        "completion_request": {
                            "model": "fixture-model",
                            "prompt": "hello",
                            "max_tokens": 1,
                            "temperature": 1.0,
                            "top_p": 1.0,
                            "seed": 1,
                            "stream": False,
                        },
                    }
                ],
            }
        )

    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def test_creates_private_root_source_audit_and_frozen_one_scenario_copy(self) -> None:
        result = prepare.prepare_lifecycle_evidence(
            self.root,
            scenario_contract=self.contract_path,
            candidate_id=self.candidate_id,
            configuration_profile=self.profile,
        )
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["qualification_status"], "not-run")
        self.assertEqual(result["schema_version"], "riley.c02-lifecycle-evidence-preparation.v5")
        self.assertEqual(result["scenario_id"], prepare.FALLBACK_SCENARIO_ID)
        self.assertEqual((self.root.stat().st_mode & 0o777), 0o700)
        self.assertEqual(((self.root / "source-audit").stat().st_mode & 0o777), 0o700)
        copied = self.root / prepare.DEFAULT_CONTRACT_COPY_NAME
        self.assertEqual(copied.read_bytes(), self.contract_path.read_bytes())
        self.assertEqual(result["scenario_contract"]["path"], prepare.DEFAULT_CONTRACT_COPY_NAME)
        root_fd = common.open_private_evidence_directory(self.root, "evidence root")
        os.close(root_fd)

    def test_rejects_multi_scenario_before_creating_evidence_root(self) -> None:
        two = [
            {
                "scenario_id": "first",
                "completion_request": {
                    "model": "m", "prompt": "p", "max_tokens": 1,
                    "temperature": 0.0, "top_p": 1.0, "seed": 1, "stream": False,
                },
            },
            {
                "scenario_id": "second",
                "completion_request": {
                    "model": "m", "prompt": "p", "max_tokens": 1,
                    "temperature": 0.0, "top_p": 1.0, "seed": 2, "stream": False,
                },
            },
        ]
        self.contract_path.write_bytes(self._contract(two))
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "lifecycle-scenario-count")
        self.assertFalse(self.root.exists())

    def test_rejects_oversized_valid_contract_before_creating_evidence_root(self) -> None:
        document = json.loads(self._contract().decode("utf-8"))
        document["scenarios"][0]["completion_request"]["prompt"] = "p" * (1024 * 1024)
        oversized = common.canonical_json_bytes(document)
        self.assertGreater(len(oversized), prepare.MAX_CONTRACT_BYTES)
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as direct_raised:
            prepare.validate_one_fallback_scenario_contract(
                oversized,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(direct_raised, "invalid-json-byte-length")
        self.contract_path.write_bytes(oversized)

        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "input-too-large")
        self.assertFalse(self.root.exists())

    def test_rejects_historical_profile_and_exact_fallback_request_drift_before_root(self) -> None:
        cases: tuple[tuple[str, str, object], ...] = (
            ("schema_version", "unsupported-contract-version", "riley.c02-raw-soak-runner-contract.v1"),
            ("configuration_profile", "contract-profile-mismatch", "stable-default"),
            ("max_tokens", "invalid-contract-value", 2),
            ("temperature", "invalid-contract-value", 0.0),
            ("top_p", "invalid-contract-value", 0.5),
            ("stream", "streaming-not-supported", True),
            ("model", "invalid-contract-value", "fixture-model\x00"),
            ("prompt", "invalid-contract-value", "hello\x00"),
        )
        for field, reason, value in cases:
            document = json.loads(self._contract().decode("utf-8"))
            if field in {"schema_version", "configuration_profile"}:
                document[field] = value
            else:
                document["scenarios"][0]["completion_request"][field] = value
            self.contract_path.write_bytes(common.canonical_json_bytes(document))
            with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
                prepare.prepare_lifecycle_evidence(
                    self.root,
                    scenario_contract=self.contract_path,
                    candidate_id=self.candidate_id,
                    configuration_profile=self.profile,
                )
            self.assert_reason(raised, reason)
            self.assertFalse(self.root.exists())

        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile="stable-default",
            )
        self.assert_reason(raised, "invalid-configuration-profile")
        self.assertFalse(self.root.exists())

    def test_rejects_nonfallback_contract_and_source_tree_roots(self) -> None:
        self.contract_path.write_bytes(
            self._contract(
                [
                    {
                        "scenario_id": "smoke",
                        "completion_request": {
                            "model": "m", "prompt": "p", "max_tokens": 1,
                            "temperature": 1.0, "top_p": 1.0, "seed": 1, "stream": False,
                        },
                    }
                ]
            )
        )
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "invalid-scenario-id")

        inside = Path(__file__).resolve().parents[2] / "temporary-lifecycle-evidence"
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                inside,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

    def test_rejects_contract_symlink_and_existing_root(self) -> None:
        linked = self.base / "linked-contract.json"
        linked.symlink_to(self.contract_path.name)
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=linked,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "unsafe-evidence-path")
        self.assertFalse(self.root.exists())

        self.root.mkdir(mode=0o700)
        self.root.chmod(0o700)
        with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "create-only-collision")

    def test_rejects_source_audit_replacement_after_initial_name_check(self) -> None:
        actual_open = os.open
        replaced = False

        def replace_source_audit_before_open(
            path: str,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal replaced
            if (
                not replaced
                and path == prepare.DEFAULT_AUDIT_DIRECTORY_NAME
                and dir_fd is not None
            ):
                replaced = True
                os.rename(
                    prepare.DEFAULT_AUDIT_DIRECTORY_NAME,
                    "source-audit-original",
                    src_dir_fd=dir_fd,
                    dst_dir_fd=dir_fd,
                )
                os.mkdir(prepare.DEFAULT_AUDIT_DIRECTORY_NAME, 0o700, dir_fd=dir_fd)
            return actual_open(path, flags, mode, dir_fd=dir_fd)

        with mock.patch.object(common.os, "open", side_effect=replace_source_audit_before_open):
            with self.assertRaises(prepare.LifecycleV5EvidencePreparationError) as raised:
                prepare.prepare_lifecycle_evidence(
                    self.root,
                    scenario_contract=self.contract_path,
                    candidate_id=self.candidate_id,
                    configuration_profile=self.profile,
                )
        self.assertTrue(replaced)
        self.assert_reason(raised, "raced-output")
        self.assertTrue(self.root.is_dir())
        self.assertFalse((self.root / prepare.DEFAULT_CONTRACT_COPY_NAME).exists())

    def test_has_a_fixed_nonoperational_cli_surface(self) -> None:
        source = Path(prepare.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "import socket",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "publish_create_only_hardlink",
            "--audit-directory-name",
            "--contract-copy-name",
            "--output-name",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
