#!/usr/bin/env python3
"""CPU-only tests for workspace roles and the reviewed MIT license."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))

from check_workspace_boundaries import (  # noqa: E402
    BoundaryError,
    EXPECTED_DEVELOPMENT_CRATES,
    EXPECTED_FEATURES,
    EXPECTED_LICENSE_TEXT,
    EXPECTED_PRODUCTION_CRATES,
    validate_features,
    validate_license_text,
    validate_package_manifests,
    validate_root_manifest,
    validate_runtime_source_text,
)


def _reviewed_packages() -> dict[str, dict[str, object]]:
    packages = {
        name: {
            "features": copy.deepcopy(features),
            "targets": [{"name": name.replace("-", "_"), "kind": ["lib"]}],
        }
        for name, features in EXPECTED_FEATURES.items()
    }
    packages["rustinfer-server"]["targets"] = [
        {
            "name": "rustinfer",
            "kind": ["bin"],
            "required-features": ["server"],
        },
        {
            "name": "rustinfer-profile",
            "kind": ["bin"],
            "required-features": ["bench", "cuda"],
        },
        {"name": "rustinfer_server", "kind": ["lib"]},
    ]
    return packages


class WorkspaceRoleContractTests(unittest.TestCase):
    def test_checked_in_workspace_and_license_pass(self) -> None:
        validate_root_manifest(ROOT)
        validate_package_manifests(ROOT)
        validate_license_text((ROOT / "LICENSE").read_text(encoding="utf-8"))

    def test_roles_are_disjoint_and_default_free(self) -> None:
        production = set(EXPECTED_PRODUCTION_CRATES.values())
        development = set(EXPECTED_DEVELOPMENT_CRATES.values())
        self.assertEqual(len(production), 7)
        self.assertEqual(development, {"rustinfer-native"})
        self.assertTrue(production.isdisjoint(development))
        self.assertEqual(EXPECTED_FEATURES["rustinfer-native"]["default"], [])

    def test_pr_a_library_only_target_passes(self) -> None:
        validate_features(_reviewed_packages())

    def test_pr_a_rejects_a_native_stub_binary(self) -> None:
        packages = _reviewed_packages()
        packages["rustinfer-native"]["targets"].append(
            {"name": "rustinfer-native", "kind": ["bin"]}
        )
        with self.assertRaisesRegex(BoundaryError, "feature-off library with no binary"):
            validate_features(packages)

    def test_native_development_source_cannot_launch_python_or_subprocesses(self) -> None:
        forbidden = (
            'std::process::Command::new("python3")',
            'use std::process::Command;',
            'use std::{fs, process};',
        )
        for source in forbidden:
            with self.subTest(source=source):
                with self.assertRaisesRegex(BoundaryError, "external processes"):
                    validate_runtime_source_text(source, "rustinfer-native/src/main.rs")
        validate_runtime_source_text(
            "use std::process::ExitCode;\nfn main() -> ExitCode { ExitCode::SUCCESS }",
            "rustinfer-native/src/main.rs",
        )

    def test_license_text_fails_closed(self) -> None:
        mutations = {
            "missing": "",
            "wrong-owner": EXPECTED_LICENSE_TEXT.replace(
                "rustinfer contributors", "another owner", 1
            ),
            "wrong-license": EXPECTED_LICENSE_TEXT.replace("MIT License", "Other", 1),
            "trailing-data": EXPECTED_LICENSE_TEXT + "NOTICE\n",
        }
        for name, contents in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(BoundaryError):
                    validate_license_text(contents)


if __name__ == "__main__":
    unittest.main()
