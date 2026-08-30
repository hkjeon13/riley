#!/usr/bin/env python3
"""CPU-only tests for the narrow C02 lifecycle evidence-root preparation."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import prepare_c02_lifecycle_evidence_v1 as prepare
import provenance_v2_common as common


class PrepareC02LifecycleEvidenceV1Tests(unittest.TestCase):
    candidate_id = "riley-0.1.0-rc3"
    profile = "stable-default"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve()
        self.contract_path = self.base / "one-scenario.json"
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
                        "scenario_id": "smoke",
                        "completion_request": {
                            "model": "fixture-model",
                            "prompt": "hello",
                            "max_tokens": 2,
                            "temperature": 0.0,
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
        self.assertEqual(result["scenario_id"], "smoke")
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
        with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "lifecycle-scenario-count")
        self.assertFalse(self.root.exists())

    def test_rejects_fallback_and_source_tree_roots(self) -> None:
        self.contract_path.write_bytes(
            self._contract(
                [
                    {
                        "scenario_id": "exact-backend-fallback",
                        "completion_request": {
                            "model": "m", "prompt": "p", "max_tokens": 1,
                            "temperature": 0.0, "top_p": 1.0, "seed": 1, "stream": False,
                        },
                    }
                ]
            )
        )
        with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "invalid-scenario-id")

        inside = Path(__file__).resolve().parents[2] / "temporary-lifecycle-evidence"
        with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
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
        with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
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
        with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
            prepare.prepare_lifecycle_evidence(
                self.root,
                scenario_contract=self.contract_path,
                candidate_id=self.candidate_id,
                configuration_profile=self.profile,
            )
        self.assert_reason(raised, "create-only-collision")

    def test_missing_directory_open_flags_fail_before_creating_child(self) -> None:
        root_fd = common.create_private_evidence_directory(self.root, "evidence root")
        try:
            for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"):
                with self.subTest(flag=flag), mock.patch.object(prepare.common.os, flag, 0):
                    with self.assertRaises(prepare.LifecycleEvidencePreparationError) as raised:
                        prepare._new_private_child(root_fd, "source-audit", "source audit directory")
                self.assert_reason(raised, "missing-open-safety-flag")
                self.assertFalse((self.root / "source-audit").exists())

        finally:
            os.close(root_fd)

        source = Path(prepare.__file__).read_text(encoding="utf-8")
        for fallback in (
            'getattr(os, "O_CLOEXEC", 0)',
            'getattr(os, "O_NOFOLLOW", 0)',
            'getattr(os, "O_DIRECTORY", 0)',
        ):
            self.assertNotIn(fallback, source)


if __name__ == "__main__":
    unittest.main()
