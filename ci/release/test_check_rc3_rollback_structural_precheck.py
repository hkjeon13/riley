#!/usr/bin/env python3
"""CPU-only hostile tests for the RC3 rollback raw-structural precheck."""

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

import bind_raw_rc3_rollback_capture as v3_binder
import bind_raw_rc3_rollback_terminal_v4 as v4_binder
import check_rc3_rollback_provenance_v4 as raw
import check_rc3_rollback_structural_precheck as precheck
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
import test_bind_raw_rc3_rollback_terminal_v4 as v4_fixtures


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class CheckRc3RollbackStructuralPrecheckTests(unittest.TestCase):
    def assert_reason(self, raised: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _completed_v4(
        self,
    ) -> tuple[v4_fixtures.RollbackV4TerminalTests, dict]:
        fixture = v4_fixtures.RollbackV4TerminalTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        return fixture, fixture._compose()

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

    def test_replays_completed_v4_as_raw_structural_only(self) -> None:
        fixture, raw_report = self._completed_v4()
        before = sorted(
            path.relative_to(fixture.root).as_posix()
            for path in fixture.root.rglob("*")
        )

        report = precheck.check_rc3_rollback_structural_precheck(
            fixture.root,
            fixture.v4_name,
        )

        self.assertEqual(report["schema_version"], precheck.PRECHECK_REPORT_VERSION)
        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], precheck.RAW_STRUCTURAL_ONLY_AUTHORITY)
        self.assertEqual(report["raw_manifest_version"], raw.ROLLBACK_V4_MANIFEST_VERSION)
        self.assertEqual(report["candidate_id"], raw_report["candidate_id"])
        self.assertEqual(report["bindings"], raw_report["bindings"])
        self.assertEqual(report["raw_manifest"], raw_report["raw_manifest"])
        self.assertEqual(
            report["checks"][0],
            {"name": "completed-v4-rollback-raw-provenance", "bound": True},
        )
        self.assertEqual(report["reason_codes"], [])
        self.assertNotIn("passed", report)
        self.assertNotIn("completed", report)
        self.assertNotIn("rollback_success", report)
        after = sorted(
            path.relative_to(fixture.root).as_posix()
            for path in fixture.root.rglob("*")
        )
        self.assertEqual(after, before)

    def test_rejects_historical_nonterminal_and_nonmanifest_headers(self) -> None:
        for version, reason in (
            ("riley.rc3-rollback-receipt.v1", "historical-rollback-v1-rejected"),
            (
                "riley.rc3-rollback-raw-provenance.v1",
                "historical-rollback-v1-rejected",
            ),
            ("riley.rc3-rollback-raw-provenance.v2", "historical-rollback-v2-rejected"),
            (
                "riley.rc3-rollback-raw-provenance.v3",
                "nonterminal-rollback-v3-not-admissible",
            ),
            (raw.ROLLBACK_V4_REPORT_VERSION, "unsupported-rollback-raw-manifest-version"),
            (raw.ROLLBACK_V4_COMPLETION_VERSION, "unsupported-rollback-raw-manifest-version"),
            (v3_binder.BIND_REQUEST_VERSION, "unsupported-rollback-raw-manifest-version"),
        ):
            with self.subTest(version=version):
                root = self._empty_root({"schema_version": version})
                with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
                    precheck.check_rc3_rollback_structural_precheck(root, "manifest.json")
                self.assert_reason(raised, reason)

    def test_rejects_nonobject_unknown_noncanonical_and_duplicate_headers(self) -> None:
        non_object = self._empty_root([])
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(non_object, "manifest.json")
        self.assert_reason(raised, "invalid-json-root")

        unknown = self._empty_root({"schema_version": "riley.rc3-rollback-provenance.v99"})
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(unknown, "manifest.json")
        self.assert_reason(raised, "unsupported-rollback-raw-manifest-version")

        noncanonical = self._empty_root(
            b'{"schema_version":"riley.rc3-rollback-terminal-provenance.v4"}\n'
        )
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(noncanonical, "manifest.json")
        self.assert_reason(raised, "noncanonical-json")

        duplicate = self._empty_root(
            b'{"schema_version":"riley.rc3-rollback-terminal-provenance.v4",'
            b'"schema_version":"riley.rc3-rollback-raw-provenance.v3"}'
        )
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(duplicate, "manifest.json")
        self.assert_reason(raised, "duplicate-json-key")

    def test_requires_a_direct_nonhidden_root_manifest_leaf(self) -> None:
        root = self._empty_root(
            {"schema_version": raw.ROLLBACK_V4_MANIFEST_VERSION}
        )
        for name in (
            "nested/manifest.json",
            "../manifest.json",
            ".manifest.json",
            "manifest.complete",
            "a" * 124 + ".json",
        ):
            with self.subTest(name=name):
                with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
                    precheck.check_rc3_rollback_structural_precheck(root, name)
                self.assert_reason(raised, "raw-manifest-must-be-direct-root-leaf")

    def test_rejects_aliases_and_propagates_missing_completion_replay_failure(self) -> None:
        fixture, _raw_report = self._completed_v4()
        root = fixture.root
        (root / "symlink.json").symlink_to(fixture.v4_name)
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(root, "symlink.json")
        self.assert_reason(raised, "unsafe-evidence-path")
        (root / "symlink.json").unlink()

        os.link(root / fixture.v4_name, root / "hardlink.json")
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(root, "hardlink.json")
        self.assert_reason(raised, "nonunique-evidence-inode")
        (root / "hardlink.json").unlink()

        (root / f"{fixture.v4_name}.complete").unlink()
        with self.assertRaises(raw.RollbackV4ProvenanceError) as raw_raised:
            raw.verify_rollback_provenance_v4(
                root,
                fixture.v4_name,
                require_completion=True,
            )
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as precheck_raised:
            precheck.check_rc3_rollback_structural_precheck(root, fixture.v4_name)
        self.assertEqual(
            getattr(precheck_raised.exception, "reason_code", None),
            getattr(raw_raised.exception, "reason_code", None),
        )

    def test_rejects_unsafe_source_and_missing_switch_roots(self) -> None:
        unsafe = self._empty_root({"schema_version": raw.ROLLBACK_V4_MANIFEST_VERSION})
        unsafe.chmod(0o755)
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(unsafe, "manifest.json")
        self.assert_reason(raised, "unsafe-evidence-root-mode")

        source_child = Path(precheck.__file__).resolve().parents[2] / "not-evidence.json"
        with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
            precheck.check_rc3_rollback_structural_precheck(source_child, "manifest.json")
        self.assert_reason(raised, "evidence-root-inside-source-checkout")

        fixture, _raw_report = self._completed_v4()
        moved_switch = fixture.root / "moved-switch"
        os.rename(fixture.root / prepare.SWITCH_DIRECTORY_NAME, moved_switch)
        with self.assertRaises(precheck.RollbackStructuralPrecheckError):
            precheck.check_rc3_rollback_structural_precheck(
                fixture.root,
                fixture.v4_name,
            )

    def test_holds_shared_root_and_switch_locks_for_completed_replay(self) -> None:
        fixture, _raw_report = self._completed_v4()
        original = raw.verify_completed_rollback_provenance_v4_on_held_switch_fd
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

        def assert_shared_locks(
            root_fd: int,
            switch_fd: int,
            manifest_name: str,
        ) -> dict:
            for directory in (
                fixture.root,
                fixture.root / prepare.SWITCH_DIRECTORY_NAME,
            ):
                completed = subprocess.run(
                    ["/usr/bin/python3", "-B", "-S", "-c", contender, str(directory)],
                    text=True,
                    capture_output=True,
                    check=False,
                    env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            return original(root_fd, switch_fd, manifest_name)

        with mock.patch.object(
            raw,
            "verify_completed_rollback_provenance_v4_on_held_switch_fd",
            side_effect=assert_shared_locks,
        ):
            report = precheck.check_rc3_rollback_structural_precheck(
                fixture.root,
                fixture.v4_name,
            )
        self.assertEqual(report["status"], "bound")

    def test_refuses_a_replayed_descriptor_that_differs_from_the_header(self) -> None:
        fixture, raw_report = self._completed_v4()
        mutated = json.loads(common.canonical_json_bytes(raw_report))
        mutated["raw_manifest"]["sha256"] = "f" * 64
        with mock.patch.object(
            raw,
            "verify_completed_rollback_provenance_v4_on_held_switch_fd",
            return_value=mutated,
        ):
            with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
                precheck.check_rc3_rollback_structural_precheck(
                    fixture.root,
                    fixture.v4_name,
                )
        self.assert_reason(raised, "raw-manifest-changed-during-version-dispatch")

    def test_ambiguous_v4_pair_remains_raw_structural_only(self) -> None:
        fixture = v4_fixtures.RollbackV4TerminalTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        original = common._fsync_checked

        def fail_v4_parent(descriptor: int, label: str) -> None:
            if label == "rollback v4 completion marker parent directory":
                error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                raise error
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_v4_parent):
            with self.assertRaises(v4_binder.RollbackV4TerminalBindError) as raised:
                fixture._compose()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((fixture.root / f"{fixture.v4_name}.complete").is_file())

        report = precheck.check_rc3_rollback_structural_precheck(
            fixture.root,
            fixture.v4_name,
        )

        self.assertEqual(report["status"], "bound")
        self.assertEqual(report["qualification_status"], "not-run")
        self.assertEqual(report["authority"], "raw-structural-only")
        self.assertNotIn("passed", report)
        self.assertNotIn("completed", report)
        self.assertNotIn("rollback_success", report)

    def test_cli_emits_canonical_raw_structural_json_only(self) -> None:
        fixture, _raw_report = self._completed_v4()
        stdout = _CapturedStdout()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            exit_code = precheck.main(
                ["--evidence-root", str(fixture.root), "--raw-manifest", fixture.v4_name]
            )
        self.assertEqual(exit_code, 0, stderr.getvalue())
        document = common.parse_canonical_json(
            stdout.buffer.getvalue().rstrip(b"\n"),
            "rollback precheck CLI",
        )
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
        self.assertIn("raw-structural precheck failed", stderr.getvalue())

    def test_refuses_to_operate_without_bytecode_cache_suppression(self) -> None:
        fixture, _raw_report = self._completed_v4()
        with mock.patch.object(precheck, "_BYTECODE_DISABLED_AT_STARTUP", False):
            with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
                precheck.check_rc3_rollback_structural_precheck(
                    fixture.root,
                    fixture.v4_name,
                )
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

    def test_refuses_to_operate_when_embedding_changed_the_entry_flag(self) -> None:
        fixture, _raw_report = self._completed_v4()
        with mock.patch.object(precheck, "_BYTECODE_DISABLED_ON_MODULE_ENTRY", False):
            with self.assertRaises(precheck.RollbackStructuralPrecheckError) as raised:
                precheck.check_rc3_rollback_structural_precheck(
                    fixture.root,
                    fixture.v4_name,
                )
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")

    def test_direct_cli_requires_bytecode_cache_suppression_before_evidence_access(self) -> None:
        completed = subprocess.run(
            [
                "/usr/bin/python3",
                "-S",
                str(Path(precheck.__file__).resolve()),
                "--evidence-root",
                "/tmp/rollback-precheck-not-opened",
                "--raw-manifest",
                "manifest.json",
            ],
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("invoke this precheck with python3 -B", completed.stderr)

    def test_static_surface_keeps_raw_structural_read_only_authority(self) -> None:
        source = Path(precheck.__file__).read_text(encoding="utf-8")
        for required in (
            "open_private_evidence_directory",
            "open_private_child_directory",
            "fcntl.LOCK_SH | fcntl.LOCK_NB",
            "verify_completed_rollback_provenance_v4_on_held_switch_fd",
            "raw-structural-only",
            "nonterminal-rollback-v3-not-admissible",
            "bytecode-cache-write-not-permitted",
            "_BYTECODE_DISABLED_ON_MODULE_ENTRY",
            "raw-manifest-must-be-direct-root-leaf",
            "evidence-root-inside-source-checkout",
        ):
            self.assertIn(required, source)
        self.assertLess(
            source.index("sys.dont_write_bytecode = True"),
            source.index("import check_rc3_rollback_provenance_v4 as raw"),
        )
        for forbidden in (
            "import subprocess",
            "import socket",
            "urllib",
            "requests",
            "docker",
            "podman",
            "ssh ",
            "nvidia",
            "bind_raw_rc3_rollback_terminal_v4",
            "prepare_artifacts(",
            "write_create_only_json",
            "publish_create_only_hardlink",
            "os.O_CREAT",
            "os.rename",
            "--output",
            "--freeze",
            "--gate-e",
            "verify_rollback_provenance_v4(",
            "\"passed\"",
            "\"qualified\"",
        ):
            self.assertNotIn(forbidden, source)

    def test_schema_reserves_exact_v4_raw_structural_contract(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rollback-raw-structural-precheck-v1.schema.json"
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
        self.assertEqual(
            schema["properties"]["raw_manifest_version"]["const"],
            raw.ROLLBACK_V4_MANIFEST_VERSION,
        )
        self.assertEqual(
            schema["$defs"]["bindings"]["properties"]["configuration_profile"]["const"],
            "stable-default",
        )
        self.assertEqual(
            schema["$defs"]["fixedTransactionDescriptor"]["allOf"][1]["properties"]["path"]["const"],
            "rollback-v3-atomic-transaction/session.json",
        )
        self.assertEqual(
            schema["properties"]["checks"]["prefixItems"],
            [
                {"$ref": "#/$defs/v4CompletedCheck"},
                {"$ref": "#/$defs/headerDescriptorCheck"},
            ],
        )
        self.assertFalse(schema["properties"]["checks"]["items"])
        self.assertNotIn("passed", json.dumps(schema, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
