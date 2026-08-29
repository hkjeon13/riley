#!/usr/bin/env python3
"""CPU-only hostile tests for the C02 soak semantic-replay input precheck."""

from __future__ import annotations

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
import bind_raw_c02_soak_v5 as v5_binder
import check_c02_provenance_v2 as raw
import check_soak_v2_receipt as precheck
import provenance_v2_common as common
import test_bind_raw_c02_soak_v4 as v4_fixtures
import test_bind_raw_c02_soak_v5 as v5_fixtures


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class CheckSoakV2ReceiptTests(unittest.TestCase):
    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _v4_manifest(self, name: str = "serial-v4.json") -> tuple[v4_fixtures.BindRawC02SoakV4Tests, dict]:
        fixture = v4_fixtures.BindRawC02SoakV4Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        report = v4_binder.bind_raw_soak_manifest(fixture.root, request_path, name)
        return fixture, report

    def _v5_manifest(self, name: str = "fallback-v5.json") -> tuple[v5_fixtures.BindRawC02SoakV5Tests, dict]:
        fixture = v5_fixtures.BindRawC02SoakV5Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        report = fixture._bind(request_path, name)
        return fixture, report

    def _empty_root(self, document: object, name: str = "manifest.json") -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve() / "evidence"
        root.mkdir(mode=0o700)
        root.chmod(0o700)
        raw_bytes = (
            document
            if isinstance(document, bytes)
            else common.canonical_json_bytes(document)
        )
        (root / name).write_bytes(raw_bytes)
        return root

    def test_replays_completed_v4_as_raw_structural_only_precheck(self) -> None:
        fixture, raw_report = self._v4_manifest()
        before = sorted(path.relative_to(fixture.root).as_posix() for path in fixture.root.rglob("*"))

        report = precheck.check_soak_v2_receipt(fixture.root, "serial-v4.json")

        self.assertEqual(report["schema_version"], precheck.PRECHECK_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], precheck.RAW_STRUCTURAL_ONLY_AUTHORITY)
        self.assertEqual(report["raw_manifest_version"], raw.SOAK_V4_MANIFEST_VERSION)
        self.assertEqual(report["candidate_id"], raw_report["candidate_id"])
        self.assertEqual(report["raw_manifest"], raw_report["raw_manifest"])
        self.assertEqual(report["checks"][0], {"name": "completed-v4-raw-provenance", "bound": True})
        self.assertEqual(report["reason_codes"], [])
        self.assertNotIn("passed", report)
        self.assertNotIn("qualified", report)
        after = sorted(path.relative_to(fixture.root).as_posix() for path in fixture.root.rglob("*"))
        self.assertEqual(after, before)

    def test_replays_completed_v5_as_raw_structural_only_precheck(self) -> None:
        fixture, raw_report = self._v5_manifest()

        report = precheck.check_soak_v2_receipt(fixture.fixture.root, "fallback-v5.json")

        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], "raw-structural-only")
        self.assertEqual(report["raw_manifest_version"], raw.SOAK_V5_MANIFEST_VERSION)
        self.assertEqual(report["candidate_id"], raw_report["candidate_id"])
        self.assertEqual(report["checks"][0], {"name": "completed-v5-raw-provenance", "bound": True})

    def test_historical_raw_versions_are_rejected_before_raw_replay(self) -> None:
        for version, reason in (
            ("riley.soak-v2-receipt.v1", "historical-soak-v1-rejected"),
            ("riley.soak-v2-raw-provenance.v1", "historical-soak-v1-rejected"),
            ("riley.soak-v2-raw-provenance.v2", "historical-soak-v2-rejected"),
            ("riley.soak-v2-raw-provenance.v3", "historical-soak-v3-rejected"),
            (raw.SOAK_V4_REPORT_VERSION, "unsupported-soak-raw-manifest-version"),
            (raw.SOAK_V5_REPORT_VERSION, "unsupported-soak-raw-manifest-version"),
            (
                raw.SOAK_V4_COMPLETION_MARKER_VERSION,
                "unsupported-soak-raw-manifest-version",
            ),
            (
                raw.SOAK_V5_COMPLETION_MARKER_VERSION,
                "unsupported-soak-raw-manifest-version",
            ),
            (v4_binder.BIND_REQUEST_VERSION, "unsupported-soak-raw-manifest-version"),
            (v5_binder.BIND_REQUEST_VERSION, "unsupported-soak-raw-manifest-version"),
        ):
            with self.subTest(version=version):
                root = self._empty_root({"schema_version": version})
                with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as raised:
                    precheck.check_soak_v2_receipt(root, "manifest.json")
                self.assert_reason(raised, reason)

    def test_rejects_unknown_noncanonical_and_duplicate_header_shapes(self) -> None:
        non_object = self._empty_root([])
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as non_object_raised:
            precheck.check_soak_v2_receipt(non_object, "manifest.json")
        self.assert_reason(non_object_raised, "invalid-json-root")

        unknown = self._empty_root({"schema_version": "riley.soak-v2-raw-provenance.v99"})
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as unknown_raised:
            precheck.check_soak_v2_receipt(unknown, "manifest.json")
        self.assert_reason(unknown_raised, "unsupported-soak-raw-manifest-version")

        noncanonical = self._empty_root(
            b'{"schema_version":"riley.soak-v2-raw-provenance.v4"}\n'
        )
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as noncanonical_raised:
            precheck.check_soak_v2_receipt(noncanonical, "manifest.json")
        self.assert_reason(noncanonical_raised, "noncanonical-json")

        duplicate = self._empty_root(
            b'{"schema_version":"riley.soak-v2-raw-provenance.v4","schema_version":"riley.soak-v2-raw-provenance.v5"}'
        )
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as duplicate_raised:
            precheck.check_soak_v2_receipt(duplicate, "manifest.json")
        self.assert_reason(duplicate_raised, "duplicate-json-key")

    def test_requires_a_direct_nonhidden_root_manifest_leaf(self) -> None:
        root = self._empty_root({"schema_version": "riley.soak-v2-raw-provenance.v4"})
        for name in ("nested/manifest.json", "../manifest.json", ".manifest.json", "manifest.complete"):
            with self.subTest(name=name):
                with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as raised:
                    precheck.check_soak_v2_receipt(root, name)
                self.assert_reason(raised, "raw-manifest-must-be-direct-root-leaf")

    def test_rejects_symlink_hardlink_and_missing_completion_marker(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        root = fixture.root
        (root / "symlink.json").symlink_to("serial-v4.json")
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as symlink_raised:
            precheck.check_soak_v2_receipt(root, "symlink.json")
        self.assert_reason(symlink_raised, "unsafe-evidence-path")
        (root / "symlink.json").unlink()

        os.link(root / "serial-v4.json", root / "hardlink.json")
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as hardlink_raised:
            precheck.check_soak_v2_receipt(root, "hardlink.json")
        self.assert_reason(hardlink_raised, "nonunique-evidence-inode")
        (root / "hardlink.json").unlink()

        (root / "serial-v4.json.complete").unlink()
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as marker_raised:
            precheck.check_soak_v2_receipt(root, "serial-v4.json")
        self.assert_reason(marker_raised, "missing-soak-v4-completion-marker")

    def test_rejects_unsafe_or_source_checkout_evidence_root(self) -> None:
        root = self._empty_root({"schema_version": "riley.soak-v2-raw-provenance.v4"})
        root.chmod(0o755)
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as unsafe_raised:
            precheck.check_soak_v2_receipt(root, "manifest.json")
        self.assert_reason(unsafe_raised, "unsafe-evidence-root-mode")

        source_child = Path(precheck.__file__).resolve().parents[2] / "not-evidence.json"
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as source_raised:
            precheck.check_soak_v2_receipt(source_child, "manifest.json")
        self.assert_reason(source_raised, "evidence-root-inside-source-checkout")

    def test_holds_a_shared_root_lock_for_the_completed_replay(self) -> None:
        fixture, _raw_report = self._v4_manifest()
        original = raw.verify_completed_soak_provenance_v4_fd
        contender = (
            "import errno,fcntl,os,sys\n"
            "flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC\n"
            "fd=os.open(sys.argv[1],flags)\n"
            "try:\n"
            "    try:\n"
            "        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "    except OSError as error:\n"
            "        if error.errno in {errno.EACCES,errno.EAGAIN}:\n"
            "            raise SystemExit(0)\n"
            "        raise\n"
            "    raise SystemExit(1)\n"
            "finally:\n"
            "    os.close(fd)\n"
        )

        def assert_shared_lock(root_fd: int, manifest_name: str) -> dict:
            completed = subprocess.run(
                ["/usr/bin/python3", "-B", "-S", "-c", contender, str(fixture.root)],
                text=True,
                capture_output=True,
                check=False,
                env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return original(root_fd, manifest_name)

        with mock.patch.object(
            raw,
            "verify_completed_soak_provenance_v4_fd",
            side_effect=assert_shared_lock,
        ):
            report = precheck.check_soak_v2_receipt(fixture.root, "serial-v4.json")
        self.assertEqual(report["status"], "bound")

    def test_refuses_a_report_descriptor_that_differs_from_the_header(self) -> None:
        fixture, raw_report = self._v4_manifest()
        mutated = json.loads(common.canonical_json_bytes(raw_report))
        mutated["raw_manifest"]["sha256"] = "f" * 64
        with mock.patch.object(
            raw,
            "verify_completed_soak_provenance_v4_fd",
            return_value=mutated,
        ):
            with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as raised:
                precheck.check_soak_v2_receipt(fixture.root, "serial-v4.json")
        self.assert_reason(raised, "raw-manifest-changed-during-version-dispatch")

    def test_ambiguous_v4_completion_pair_remains_raw_structural_only(self) -> None:
        fixture = v4_fixtures.BindRawC02SoakV4Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(v4_binder.RawSoakBindError) as raised:
                v4_binder.bind_raw_soak_manifest(
                    fixture.root,
                    request_path,
                    "ambiguous-v4.json",
                )
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((fixture.root / "ambiguous-v4.json.complete").is_file())

        report = precheck.check_soak_v2_receipt(fixture.root, "ambiguous-v4.json")

        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], "raw-structural-only")
        self.assertNotIn("passed", report)
        self.assertNotIn("completed", report)

    def test_ambiguous_v5_completion_pair_remains_raw_structural_only(self) -> None:
        fixture = v5_fixtures.BindRawC02SoakV5Tests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        request_path, _request = fixture._request()
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "v5 soak raw manifest completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(v5_binder.RawSoakBindError) as raised:
                fixture._bind(request_path, "ambiguous-v5.json")
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((fixture.fixture.root / "ambiguous-v5.json.complete").is_file())

        report = precheck.check_soak_v2_receipt(
            fixture.fixture.root,
            "ambiguous-v5.json",
        )

        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], "raw-structural-only")
        self.assertNotIn("passed", report)
        self.assertNotIn("completed", report)

    def test_dispatcher_preserves_v5_raw_replay_rejection(self) -> None:
        fixture, _raw_report = self._v5_manifest()
        fixture._set_effective_sampling_backend("cpu")

        with self.assertRaises(raw.C02ProvenanceError) as raw_raised:
            raw.verify_completed_soak_provenance_v5(
                fixture.fixture.root,
                "fallback-v5.json",
            )
        with self.assertRaises(precheck.SoakV2ReceiptPrecheckError) as precheck_raised:
            precheck.check_soak_v2_receipt(fixture.fixture.root, "fallback-v5.json")
        self.assertEqual(
            getattr(precheck_raised.exception, "reason_code", None),
            getattr(raw_raised.exception, "reason_code", None),
        )

    def test_cli_emits_only_canonical_structural_precheck_json(self) -> None:
        fixture, _raw_report = self._v5_manifest()
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = precheck.main(
                ["--evidence-root", str(fixture.fixture.root), "--raw-manifest", "fallback-v5.json"]
            )
        self.assertEqual(exit_code, 0, stderr.getvalue())
        document = common.parse_canonical_json(stdout.buffer.getvalue().rstrip(b"\n"), "CLI precheck")
        self.assertEqual(document["authority"], "raw-structural-only")
        self.assertEqual(document["qualification_status"], "not-run")

    def test_cli_failure_emits_no_stdout(self) -> None:
        root = self._empty_root([])
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = precheck.main(
                ["--evidence-root", str(root), "--raw-manifest", "manifest.json"]
            )
        self.assertEqual(exit_code, 2)
        self.assertEqual(stdout.buffer.getvalue(), b"")
        self.assertIn("semantic-replay precheck failed", stderr.getvalue())

    def test_static_surface_cannot_issue_a_semantic_or_operational_result(self) -> None:
        source = Path(precheck.__file__).read_text(encoding="utf-8")
        for required in (
            "open_private_evidence_directory",
            "fcntl.LOCK_SH | fcntl.LOCK_NB",
            "verify_completed_soak_provenance_v4_fd",
            "verify_completed_soak_provenance_v5_fd",
            "raw-structural-only",
            "raw-manifest-must-be-direct-root-leaf",
            "evidence-root-inside-source-checkout",
        ):
            self.assertIn(required, source)
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("import check_c02_provenance_v2 as raw"),
        )
        for forbidden in (
            "import subprocess",
            "import socket",
            "urllib",
            "requests",
            "nvidia",
            "docker",
            "podman",
            "ssh ",
            "O_CREAT",
            "os.link",
            "common.write_create_only_json",
            "common.publish_create_only_hardlink",
            "os.O_CREAT",
            "os.mkdir",
            "os.link",
            "--output",
            "--freeze",
            "--gate-e",
            "\"passed\"",
            "\"qualified\"",
        ):
            self.assertNotIn(forbidden, source)

    def test_precheck_schema_reserves_only_raw_structural_authority(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/soak-v2-semantic-replay-precheck-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            precheck.PRECHECK_REPORT_VERSION,
        )
        self.assertEqual(
            schema["properties"]["authority"]["const"],
            precheck.RAW_STRUCTURAL_ONLY_AUTHORITY,
        )
        self.assertEqual(schema["properties"]["qualification_status"]["const"], "not-run")
        self.assertNotIn("passed", json.dumps(schema, sort_keys=True))

    def test_precheck_schema_binds_version_to_exact_primary_check(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/soak-v2-semantic-replay-precheck-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        variants = schema["oneOf"]
        self.assertEqual(len(variants), 2)
        variants_by_version = {
            variant["properties"]["raw_manifest_version"]["const"]: variant
            for variant in variants
        }
        v4 = variants_by_version[raw.SOAK_V4_MANIFEST_VERSION]
        v5 = variants_by_version[raw.SOAK_V5_MANIFEST_VERSION]
        self.assertEqual(
            v4["properties"]["checks"]["prefixItems"],
            [
                {"$ref": "#/$defs/v4PrimaryCheck"},
                {"$ref": "#/$defs/headerDescriptorCheck"},
            ],
        )
        self.assertEqual(
            v5["properties"]["checks"]["prefixItems"],
            [
                {"$ref": "#/$defs/v5PrimaryCheck"},
                {"$ref": "#/$defs/headerDescriptorCheck"},
            ],
        )
        self.assertFalse(v4["properties"]["checks"]["items"])
        self.assertFalse(v5["properties"]["checks"]["items"])
        self.assertEqual(v5["properties"]["targets"]["maxItems"], 1)
        self.assertEqual(
            v5["properties"]["bindings"]["properties"]["configuration_profile"][
                "const"
            ],
            "max-performance-exact",
        )
        self.assertEqual(
            v5["properties"]["targets"]["prefixItems"][0]["properties"][
                "scenario_id"
            ]["const"],
            "exact-backend-fallback",
        )
        self.assertFalse(v5["properties"]["targets"]["items"])
        self.assertEqual(
            v4["properties"]["targets"]["not"]["contains"]["properties"][
                "scenario_id"
            ]["const"],
            "exact-backend-fallback",
        )


if __name__ == "__main__":
    unittest.main()
