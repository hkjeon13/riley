#!/usr/bin/env python3
"""CPU-only tests for workspace roles and the reviewed MIT license."""

from __future__ import annotations

import copy
import sys
import json
import unittest
from unittest import mock
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
    cargo_metadata,
)


def _reviewed_packages() -> dict[str, dict[str, object]]:
    packages = {
        name: {
            "features": copy.deepcopy(features),
            "targets": [{"name": name.replace("-", "_"), "kind": ["lib"]}],
        }
        for name, features in EXPECTED_FEATURES.items()
    }
    packages["riley-server"]["targets"] = [
        {
            "name": "riley",
            "kind": ["bin"],
            "required-features": ["server"],
        },
        {
            "name": "riley-profile",
            "kind": ["bin"],
            "required-features": ["bench", "cuda"],
        },
        {"name": "riley_server", "kind": ["lib"]},
    ]
    packages["riley-native"]["targets"] = [
        {"name": "riley_native", "kind": ["lib"]},
        {
            "name": "riley-native",
            "kind": ["bin"],
            "required-features": ["cuda"],
        },
    ]
    return packages


class WorkspaceRoleContractTests(unittest.TestCase):
    def test_resolved_metadata_includes_optional_serving_and_probe_dependencies(self) -> None:
        process = mock.Mock(returncode=0, stdout=json.dumps({"packages": []}))
        with mock.patch("check_workspace_boundaries.subprocess.run", return_value=process) as run:
            cargo_metadata(ROOT, True, no_deps=False)
        self.assertIn("--all-features", run.call_args.args[0])
        self.assertIn("--locked", run.call_args.args[0])

    def test_probe_features_cannot_become_defaults_or_acquire_unknown_edges(self) -> None:
        for name in ("riley-cuda", "riley-runtime", "riley-server"):
            packages = _reviewed_packages()
            packages[name]["features"]["default"] = ["cuda"]
            with self.subTest(name=name):
                with self.assertRaisesRegex(BoundaryError, "feature propagation drifted"):
                    validate_features(packages)
        packages = _reviewed_packages()
        packages["riley-server"]["features"]["unknown-probe"] = ["cuda"]
        with self.assertRaisesRegex(BoundaryError, "feature propagation drifted"):
            validate_features(packages)

    def test_gpu_target_cannot_silently_enter_cpu_validation(self) -> None:
        packages = _reviewed_packages()
        target = {"name": "new_attention_gpu", "kind": ["test"]}
        packages["riley-cuda"]["targets"].append(target)
        with self.assertRaisesRegex(BoundaryError, "explicit CUDA feature"):
            validate_features(packages)
        target["required-features"] = ["cuda"]
        validate_features(packages)
    def test_checked_in_workspace_and_license_pass(self) -> None:
        validate_root_manifest(ROOT)
        validate_package_manifests(ROOT)
        validate_license_text((ROOT / "LICENSE").read_text(encoding="utf-8"))

    def test_roles_are_disjoint_and_default_free(self) -> None:
        production = set(EXPECTED_PRODUCTION_CRATES.values())
        development = set(EXPECTED_DEVELOPMENT_CRATES.values())
        self.assertEqual(len(production), 7)
        self.assertEqual(development, {"riley-native"})
        self.assertTrue(production.isdisjoint(development))
        self.assertEqual(EXPECTED_FEATURES["riley-native"]["default"], [])

    def test_cuda_gated_native_calibration_target_passes(self) -> None:
        validate_features(_reviewed_packages())

    def test_native_calibration_binary_must_be_cuda_gated(self) -> None:
        packages = _reviewed_packages()
        packages["riley-native"]["targets"][1]["required-features"] = []
        with self.assertRaisesRegex(BoundaryError, "require exactly `cuda`"):
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
                    validate_runtime_source_text(source, "riley-native/src/main.rs")
        validate_runtime_source_text(
            "use std::process::ExitCode;\nfn main() -> ExitCode { ExitCode::SUCCESS }",
            "riley-native/src/main.rs",
        )

    def test_license_text_fails_closed(self) -> None:
        mutations = {
            "missing": "",
            "wrong-owner": EXPECTED_LICENSE_TEXT.replace(
                "Riley contributors", "another owner", 1
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
