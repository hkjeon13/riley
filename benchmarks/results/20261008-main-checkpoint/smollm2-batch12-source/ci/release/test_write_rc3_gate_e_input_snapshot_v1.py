#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Gate E opaque input snapshot."""

from __future__ import annotations

import fcntl
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import check_rc3_prefreeze as prefreeze  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import rc3_frozen_candidate_common as frozen  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import write_rc3_frozen_candidate_v1 as frozen_writer  # noqa: E402
import write_rc3_gate_e_input_snapshot_v1 as writer  # noqa: E402
from test_check_rc3_freeze_input_admission import (  # noqa: E402
    AdmissionFixture,
    CANDIDATE_ID,
    _sha256,
)


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class Rc3GateEInputSnapshotV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve(strict=True)
        self.fixture = AdmissionFixture(self.base)
        self.frozen_root = self.base / "frozen-candidate"
        self.gate_root = self.base / "gate-e-inputs"
        self.sources_root = self.base / "opaque-sources"
        self.sources_root.mkdir(mode=0o700)
        self.sources: dict[str, Path] = {}
        for index, spec in enumerate(writer.SOURCE_SPECS, start=1):
            source = self.sources_root / f"{index:02d}-{spec.snapshot_name}"
            source.write_bytes(f"opaque source for {spec.group}.{spec.field}\n".encode("utf-8"))
            source.chmod(0o700 if spec.require_owner_executable else 0o600)
            self.sources[spec.attribute] = source
        self._write_frozen()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext,
        reason: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), reason)

    def _defaults(self):
        return mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(self.fixture.defaults),
        )

    def _write_frozen(self) -> None:
        with self._defaults():
            frozen_writer.write_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
                expected_revision=self.fixture.revision,
                candidate_id=CANDIDATE_ID,
                request_name=self.fixture.request_name,
            )

    def _source_request(self) -> writer.GateEInputSnapshotSources:
        return writer.GateEInputSnapshotSources(**self.sources)

    def _write(self, root: Path | None = None) -> dict[str, object]:
        with self._defaults():
            return writer.write_rc3_gate_e_input_snapshot_v1(
                self.gate_root if root is None else root,
                frozen_candidate_root=self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
                sources=self._source_request(),
            )

    def _source_snapshot(self) -> dict[str, bytes]:
        return {
            path.name: path.read_bytes()
            for path in sorted(self.sources_root.iterdir())
        }

    def test_writer_creates_exact_opaque_snapshot_and_structural_self_replay(self) -> None:
        source_before = self._source_snapshot()
        input_before = self.fixture.evidence_snapshot()
        frozen_before = (self.frozen_root / frozen.MANIFEST_NAME).read_bytes()

        result = self._write()

        expected_entries = {
            gate_inputs.INVENTORY_NAME,
            *(spec.snapshot_name for spec in writer.SOURCE_SPECS),
        }
        self.assertEqual({entry.name for entry in self.gate_root.iterdir()}, expected_entries)
        self.assertEqual(stat.S_IMODE(self.gate_root.stat().st_mode), 0o700)
        for entry in self.gate_root.iterdir():
            metadata = entry.stat()
            self.assertTrue(stat.S_ISREG(metadata.st_mode))
            self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
            self.assertEqual(metadata.st_nlink, 1)
        self.assertEqual(self._source_snapshot(), source_before)
        self.assertEqual(self.fixture.evidence_snapshot(), input_before)
        self.assertEqual((self.frozen_root / frozen.MANIFEST_NAME).read_bytes(), frozen_before)
        self.assertEqual(self.fixture.source_status(), b"")

        raw_inventory = (self.gate_root / gate_inputs.INVENTORY_NAME).read_bytes()
        inventory = json.loads(raw_inventory)
        self.assertEqual(raw_inventory, common.canonical_json_bytes(inventory))
        self.assertEqual(inventory["schema_version"], gate_inputs.INVENTORY_VERSION)
        self.assertEqual(inventory["candidate_id"], CANDIDATE_ID)
        self.assertEqual(inventory["source_revision"], self.fixture.revision)
        self.assertNotIn(os.fspath(self.sources_root), raw_inventory.decode("utf-8"))
        self.assertNotIn("passed", inventory)
        self.assertNotIn("qualification", inventory)
        self.assertNotIn("receipt", inventory)
        self.assertNotIn("normal_return", inventory)

        replay = result["replay"]
        self.assertIsInstance(replay, dict)
        assert isinstance(replay, dict)
        self.assertEqual(replay["status"], "bound")
        self.assertEqual(replay["candidate_status"], "frozen")
        self.assertEqual(replay["qualification_status"], "not-run")
        self.assertEqual(replay["authority"], gate_inputs.AUTHORITY)
        self.assertNotIn("passed", replay)
        self.assertEqual(result["gate_e_input_inventory"], replay["gate_e_input_inventory"])

    def test_inventory_is_written_after_all_fourteen_snapshot_leaves(self) -> None:
        original = common.write_create_only_json
        expected_snapshots = {spec.snapshot_name for spec in writer.SOURCE_SPECS}

        def inspect_then_write(directory_fd, name, value, label):
            self.assertEqual(name, gate_inputs.INVENTORY_NAME)
            with os.scandir(directory_fd) as entries:
                self.assertEqual({entry.name for entry in entries}, expected_snapshots)
            return original(directory_fd, name, value, label)

        with mock.patch.object(common, "write_create_only_json", side_effect=inspect_then_write):
            self._write()

    def test_reserves_total_budget_before_each_snapshot_copy(self) -> None:
        original = common.snapshot_absolute_regular_create_only
        observed: list[tuple[Path, int]] = []

        def record_budget(source, destination_parent_fd, destination_name, label, **kwargs):
            observed.append((source, kwargs["maximum_bytes"]))
            return original(source, destination_parent_fd, destination_name, label, **kwargs)

        with mock.patch.object(
            common,
            "snapshot_absolute_regular_create_only",
            side_effect=record_budget,
        ):
            self._write()
        self.assertEqual(len(observed), len(writer.SOURCE_SPECS))
        remaining = writer.MAX_SNAPSHOT_TOTAL_BYTES
        for (source, maximum), spec in zip(observed, writer.SOURCE_SPECS):
            self.assertEqual(source, self.sources[spec.attribute])
            self.assertEqual(maximum, remaining)
            remaining -= source.stat().st_size
        self.assertGreaterEqual(remaining, 0)

    def test_budget_exhaustion_precedes_inventory_publication(self) -> None:
        for spec in writer.SOURCE_SPECS:
            source = self.sources[spec.attribute]
            source.write_bytes(b"x")
            source.chmod(0o700 if spec.require_owner_executable else 0o600)
        with mock.patch.object(writer, "MAX_SNAPSHOT_TOTAL_BYTES", len(writer.SOURCE_SPECS) - 1), self.assertRaises(
            writer.GateEInputSnapshotError
        ) as raised:
            self._write()
        self.assert_reason(raised, "external-evidence-byte-budget-exceeded")
        self.assertTrue(self.gate_root.is_dir())
        self.assertEqual(len(tuple(self.gate_root.iterdir())), len(writer.SOURCE_SPECS) - 1)
        self.assertFalse((self.gate_root / gate_inputs.INVENTORY_NAME).exists())

    def test_cli_emits_canonical_ephemeral_result_without_expanding_inventory_schema(self) -> None:
        stdout = _CapturedStdout()
        arguments = [
            "--gate-e-evidence-root",
            os.fspath(self.gate_root),
            "--frozen-candidate-root",
            os.fspath(self.frozen_root),
            "--input-evidence-root",
            os.fspath(self.fixture.evidence),
            "--repository-root",
            os.fspath(self.fixture.root),
        ]
        for spec in writer.SOURCE_SPECS:
            arguments.extend((spec.option, os.fspath(self.sources[spec.attribute])))
        with self._defaults(), mock.patch.object(writer.sys, "stdout", stdout):
            status = writer.main(arguments)
        self.assertEqual(status, 0)
        raw = stdout.buffer.getvalue()
        document = json.loads(raw)
        self.assertEqual(raw, common.canonical_json_bytes(document) + b"\n")
        self.assertEqual(document["replay"]["status"], "bound")
        self.assertNotIn("passed", document)
        self.assertNotIn("receipt", document)

    def test_missing_late_source_retains_partial_root_and_refuses_resume(self) -> None:
        missing = self.sources["soak_raw_evidence"]
        missing.unlink()
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "missing-input")
        self.assertTrue(self.gate_root.is_dir())
        self.assertNotIn(
            gate_inputs.INVENTORY_NAME,
            {entry.name for entry in self.gate_root.iterdir()},
        )
        self.assertEqual(len(tuple(self.gate_root.iterdir())), len(writer.SOURCE_SPECS) - 1)

        missing.write_bytes(b"replacement opaque soak raw\n")
        missing.chmod(0o600)
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "create-only-collision")

    def test_rejects_unsafe_or_wrong_mode_source_before_inventory_publication(self) -> None:
        target = self.sources["performance_report"]
        target.unlink()
        target.symlink_to(self.sources["soak_report"])
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "unsafe-evidence-path")
        self.assertFalse((self.gate_root / gate_inputs.INVENTORY_NAME).exists())

        other_root = self.base / "other-gate-root"
        target.unlink()
        target.write_bytes(b"opaque performance report\n")
        target.chmod(0o600)
        profile = self.sources["release_profile_binary"]
        profile.chmod(0o600)
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write(other_root)
        self.assert_reason(raised, "unsafe-source-mode")
        self.assertFalse((other_root / gate_inputs.INVENTORY_NAME).exists())

    def test_rejects_duplicate_source_path_and_root_overlap_before_output_creation(self) -> None:
        duplicate_sources = dict(self.sources)
        duplicate_sources["soak_raw_evidence"] = duplicate_sources["performance_raw_evidence"]
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            with self._defaults():
                writer.write_rc3_gate_e_input_snapshot_v1(
                    self.gate_root,
                    frozen_candidate_root=self.frozen_root,
                    input_evidence_root=self.fixture.evidence,
                    repository_root=self.fixture.root,
                    sources=writer.GateEInputSnapshotSources(**duplicate_sources),
                )
        self.assert_reason(raised, "duplicate-gate-e-input-source")
        self.assertFalse(self.gate_root.exists())

        nested_root = self.fixture.evidence / "nested-gate-e"
        with self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write(nested_root)
        self.assert_reason(raised, "frozen-candidate-root-overlap")
        self.assertFalse(nested_root.exists())

    def test_bytecode_and_open_safety_guards_precede_root_creation(self) -> None:
        with mock.patch.object(writer, "_BYTECODE_DISABLED_AT_STARTUP", False), mock.patch.object(
            common,
            "create_private_evidence_directory",
            side_effect=AssertionError("bytecode guard must precede output creation"),
        ), self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")
        self.assertFalse(self.gate_root.exists())

        with mock.patch.object(common.os, "O_NOFOLLOW", None), mock.patch.object(
            common,
            "create_private_evidence_directory",
            side_effect=AssertionError("open-safety guard must precede output creation"),
        ), self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "missing-open-safety-flag")
        self.assertFalse(self.gate_root.exists())

    def test_rechecks_private_snapshot_modes_after_structural_self_replay(self) -> None:
        original = gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds
        for index, leaf in enumerate(
            (writer.SOURCE_SPECS[0].snapshot_name, gate_inputs.INVENTORY_NAME),
        ):
            with self.subTest(leaf=leaf):
                root = self.base / f"mode-mutated-{index}"

                def mutate_mode(gate_fd, frozen_fd, input_fd, repository_root, repository_fd):
                    result = original(gate_fd, frozen_fd, input_fd, repository_root, repository_fd)
                    (root / leaf).chmod(0o644)
                    return result

                with mock.patch.object(
                    gate_inputs,
                    "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
                    side_effect=mutate_mode,
                ), self.assertRaises(writer.GateEInputSnapshotError) as raised:
                    self._write(root)
                self.assert_reason(raised, "unsafe-output-mode")

    def test_detects_visible_output_root_replacement_after_structural_self_replay(self) -> None:
        original = gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds
        displaced = self.base / "displaced-gate-e-root"

        def replace_visible_root(gate_fd, frozen_fd, input_fd, repository_root, repository_fd):
            result = original(gate_fd, frozen_fd, input_fd, repository_root, repository_fd)
            os.rename(self.gate_root, displaced)
            self.gate_root.mkdir(mode=0o700)
            self.gate_root.chmod(0o700)
            return result

        with mock.patch.object(
            gate_inputs,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            side_effect=replace_visible_root,
        ), self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "raced-output")

    def test_rechecks_exact_entry_set_after_structural_self_replay(self) -> None:
        original = gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds

        def add_extra_leaf(gate_fd, frozen_fd, input_fd, repository_root, repository_fd):
            result = original(gate_fd, frozen_fd, input_fd, repository_root, repository_fd)
            extra = self.gate_root / "unexpected-after-replay"
            extra.write_bytes(b"raced extra leaf\n")
            extra.chmod(0o600)
            return result

        with mock.patch.object(
            gate_inputs,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            side_effect=add_extra_leaf,
        ), self.assertRaises(writer.GateEInputSnapshotError) as raised:
            self._write()
        self.assert_reason(raised, "unexpected-evidence-entry")

    def test_writer_does_not_add_semantic_or_process_execution_surface(self) -> None:
        source = Path(writer.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "Path.read_bytes",
            "Path.write_bytes",
            "shutil",
            "subprocess",
            "check_release_candidate",
            "publish_create_only_hardlink",
            "docker",
            "nvidia-smi",
            "ssh ",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertIn("snapshot_absolute_regular_create_only", source)
        self.assertIn("write_create_only_json", source)
        self.assertIn("_replay_rc3_gate_e_input_inventory_v1_on_held_fds", source)

    def test_private_self_replay_retains_root_and_input_locks(self) -> None:
        original = gate_inputs._replay_rc3_gate_e_input_inventory_v1_on_held_fds

        def conflicting_lock_is_blocked(path: Path, operation: int) -> None:
            program = (
                "import fcntl, os, sys\n"
                "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)\n"
                "try:\n"
                "    fcntl.flock(fd, int(sys.argv[2]) | fcntl.LOCK_NB)\n"
                "except BlockingIOError:\n"
                "    raise SystemExit(0)\n"
                "else:\n"
                "    raise SystemExit(1)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-B", "-c", program, os.fspath(path), str(operation)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

        def assert_locks(gate_fd, frozen_fd, input_fd, repository_root, repository_fd):
            conflicting_lock_is_blocked(self.gate_root, fcntl.LOCK_SH)
            conflicting_lock_is_blocked(self.frozen_root, fcntl.LOCK_EX)
            conflicting_lock_is_blocked(self.fixture.evidence, fcntl.LOCK_EX)
            return original(gate_fd, frozen_fd, input_fd, repository_root, repository_fd)

        with mock.patch.object(
            gate_inputs,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            side_effect=assert_locks,
        ):
            result = self._write()
        self.assertEqual(result["replay"]["status"], "bound")


if __name__ == "__main__":
    unittest.main()
