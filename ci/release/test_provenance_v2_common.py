#!/usr/bin/env python3
"""Focused hostile-path tests for provenance_v2_common."""

from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import provenance_v2_common as common


class ProvenanceV2CommonTests(unittest.TestCase):
    def open_root(self, root: Path) -> int:
        # macOS exposes /tmp and /var through compatibility symlinks.  The
        # primitive correctly refuses those aliases, so tests hand it the
        # physical, already-resolved temporary-directory path.
        return common.open_absolute_directory(root.resolve(strict=True), "test evidence root")

    def assert_reason(self, context: unittest.case._AssertRaisesContext, reason: str) -> None:
        self.assertEqual(getattr(context.exception, "reason_code", None), reason)

    def test_canonical_json_round_trip_and_rejects_aliases(self) -> None:
        raw = common.canonical_json_bytes({"é": [1, True], "a": {"z": None}})
        self.assertEqual(raw, b'{"a":{"z":null},"\xc3\xa9":[1,true]}')
        self.assertEqual(common.parse_canonical_json(raw, "receipt"), {"a": {"z": None}, "é": [1, True]})

        for malformed, reason in (
            (b'{"a": 1}', "noncanonical-json"),
            (b'{"a":1,"a":1}', "duplicate-json-key"),
            (b'{"a":NaN}', "non-finite-json-number"),
            (b'[1,2]', "invalid-json-root"),
        ):
            with self.subTest(malformed=malformed), self.assertRaises(common.ProvenanceV2Error) as raised:
                common.parse_canonical_json(malformed, "receipt")
            self.assert_reason(raised, reason)

    def test_strict_raw_json_accepts_docker_whitespace_but_rejects_duplicate_or_nonfinite(self) -> None:
        raw_docker = b'[\n  {"Id": "sha256:abc"}\n]\n'
        self.assertEqual(
            common.parse_strict_json(raw_docker, "docker inspect", require_object=False),
            [{"Id": "sha256:abc"}],
        )
        for malformed, reason in (
            (b'[{"Id":"one","Id":"two"}]', "duplicate-json-key"),
            (b'[{"Id":NaN}]', "non-finite-json-number"),
        ):
            with self.subTest(malformed=malformed), self.assertRaises(common.ProvenanceV2Error) as raised:
                common.parse_strict_json(malformed, "docker inspect", require_object=False)
            self.assert_reason(raised, reason)

    def test_missing_nofollow_fails_closed_before_opening(self) -> None:
        with mock.patch.object(common.os, "O_NOFOLLOW", 0):
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.require_safe_open_flags()
        self.assert_reason(raised, "missing-open-safety-flag")

    def test_missing_directory_flag_fails_closed_before_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(common.os, "O_DIRECTORY", 0):
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.open_absolute_directory(Path(temporary), "evidence root")
        self.assert_reason(raised, "missing-open-safety-flag")

    def test_private_evidence_root_requires_0700_euid_and_safe_ancestors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(strict=True)
            root = parent / "evidence"
            root.mkdir(mode=0o700)
            root.chmod(0o700)
            descriptor = common.open_private_evidence_directory(root, "evidence root")
            os.close(descriptor)

            root.chmod(0o755)
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.open_private_evidence_directory(root, "evidence root")
            self.assert_reason(raised, "unsafe-evidence-root-mode")
            direct_fd = common.open_absolute_directory(root, "direct evidence root")
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.require_private_evidence_directory_fd(
                        direct_fd,
                        "direct evidence root",
                    )
                self.assert_reason(raised, "unsafe-evidence-root-mode")
            finally:
                os.close(direct_fd)
            root.chmod(0o700)

            with mock.patch.object(common.os, "geteuid", return_value=os.geteuid() + 1):
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.open_private_evidence_directory(root, "evidence root")
            self.assert_reason(raised, "unsafe-evidence-root-owner")

            link = parent / "evidence-link"
            link.symlink_to(root.name, target_is_directory=True)
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.open_private_evidence_directory(link, "evidence root")
            self.assert_reason(raised, "unsafe-evidence-directory")

            parent.chmod(0o777)
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.open_private_evidence_directory(root, "evidence root")
                self.assert_reason(raised, "unsafe-evidence-ancestor")
            finally:
                parent.chmod(0o700)

    def test_create_private_evidence_directory_is_durable_private_and_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(strict=True)
            target = parent / "new-evidence"
            parent_metadata = parent.stat()
            synchronized: list[int] = []
            real_fsync = os.fsync

            def record_fsync(descriptor: int) -> None:
                synchronized.append(os.fstat(descriptor).st_ino)
                real_fsync(descriptor)

            with mock.patch.object(common.os, "fsync", side_effect=record_fsync):
                descriptor = common.create_private_evidence_directory(target, "new evidence root")
            try:
                metadata = os.fstat(descriptor)
                self.assertTrue(stat.S_ISDIR(metadata.st_mode))
                self.assertEqual(metadata.st_uid, os.geteuid())
                self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o700)
                self.assertEqual(metadata.st_nlink, 2)
                self.assertIn(metadata.st_ino, synchronized)
                self.assertIn(parent_metadata.st_ino, synchronized)

                moved = parent / "moved-evidence"
                os.rename(target, moved)
                target.mkdir(mode=0o700)
                os.chmod(target, 0o700)
                os.mkdir("through-held-fd", 0o700, dir_fd=descriptor)
                self.assertTrue((moved / "through-held-fd").is_dir())
                self.assertFalse((target / "through-held-fd").exists())
            finally:
                os.close(descriptor)

    def test_create_private_evidence_directory_rejects_hostile_paths_and_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside_temporary:
            parent = Path(temporary).resolve(strict=True)
            outside = Path(outside_temporary).resolve(strict=True)
            target = parent / "evidence"
            target.mkdir(mode=0o700)
            target.chmod(0o700)
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.create_private_evidence_directory(target, "collision")
            self.assert_reason(raised, "create-only-collision")

            target.rmdir()
            target.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.create_private_evidence_directory(target, "symlink collision")
            self.assert_reason(raised, "create-only-collision")
            self.assertFalse((outside / "evidence").exists())
            target.unlink()

            link = parent / "linked-parent"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(common.ProvenanceV2Error) as raised:
                common.create_private_evidence_directory(link / "child", "linked parent")
            self.assert_reason(raised, "unsafe-evidence-directory")
            self.assertFalse((outside / "child").exists())

            unsafe_parent = parent / "unsafe-parent"
            unsafe_parent.mkdir(mode=0o700)
            unsafe_parent.chmod(0o777)
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.create_private_evidence_directory(unsafe_parent / "child", "unsafe parent")
                self.assert_reason(raised, "unsafe-evidence-ancestor")
                self.assertFalse((unsafe_parent / "child").exists())
            finally:
                unsafe_parent.chmod(0o700)

            with mock.patch.object(common.os, "O_NOFOLLOW", 0):
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.create_private_evidence_directory(parent / "missing-flag", "missing flag")
            self.assert_reason(raised, "missing-open-safety-flag")
            self.assertFalse((parent / "missing-flag").exists())

    def test_create_private_evidence_directory_allows_owned_sticky_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(strict=True)
            parent.chmod(0o1777)
            try:
                descriptor = common.create_private_evidence_directory(parent / "evidence", "sticky parent")
                try:
                    metadata = os.fstat(descriptor)
                    self.assertEqual(metadata.st_uid, os.geteuid())
                    self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o700)
                finally:
                    os.close(descriptor)
            finally:
                parent.chmod(0o700)

    def test_create_private_evidence_directory_fails_closed_on_parent_sync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(strict=True)
            parent_inode = parent.stat().st_ino
            target = parent / "unsynced-evidence"
            real_fsync = os.fsync

            def fail_parent_sync(descriptor: int) -> None:
                if os.fstat(descriptor).st_ino == parent_inode:
                    raise OSError("fixture parent fsync failure")
                real_fsync(descriptor)

            with mock.patch.object(common.os, "fsync", side_effect=fail_parent_sync):
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.create_private_evidence_directory(target, "unsynced evidence root")
            self.assert_reason(raised, "durability-failure")
            self.assertTrue(target.is_dir())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)

    def test_final_symlink_is_rejected_without_reading_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_bytes(common.canonical_json_bytes({"target": "unchanged"}))
            (root / "receipt.json").symlink_to(target.name)
            root_fd = self.open_root(root)
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_regular_relative(root_fd, "receipt.json", "receipt")
            finally:
                os.close(root_fd)
        self.assert_reason(raised, "unsafe-evidence-path")

    def test_intermediate_symlink_is_rejected_without_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            external = Path(outside)
            (external / "receipt.json").write_bytes(common.canonical_json_bytes({"outside": True}))
            (root / "raw").symlink_to(external, target_is_directory=True)
            root_fd = self.open_root(root)
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_regular_relative(root_fd, "raw/receipt.json", "receipt")
            finally:
                os.close(root_fd)
        self.assert_reason(raised, "unsafe-evidence-directory")

    def test_reader_enforces_bound_single_link_and_path_stability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = common.canonical_json_bytes({"state": "stable"})
            receipt = root / "receipt.json"
            receipt.write_bytes(payload)
            root_fd = self.open_root(root)
            try:
                self.assertEqual(
                    common.read_bounded_regular_relative(
                        root_fd,
                        "receipt.json",
                        "receipt",
                        maximum_bytes=len(payload),
                    ),
                    payload,
                )
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_regular_relative(
                        root_fd,
                        "receipt.json",
                        "receipt",
                        maximum_bytes=len(payload) - 1,
                    )
                self.assert_reason(raised, "input-too-large")

                alias = root / "receipt-alias.json"
                os.link(receipt, alias)
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_regular_relative(root_fd, "receipt.json", "receipt")
                self.assert_reason(raised, "nonunique-evidence-inode")
            finally:
                os.close(root_fd)

    def test_reader_rejects_path_swap_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "receipt.json"
            receipt.write_bytes(common.canonical_json_bytes({"state": "before"}))
            replacement = root / "replacement.json"
            replacement.write_bytes(common.canonical_json_bytes({"state": "after"}))
            root_fd = self.open_root(root)
            real_read = common.os.read
            swapped = False

            def read_then_swap(descriptor: int, count: int) -> bytes:
                nonlocal swapped
                result = real_read(descriptor, count)
                if not swapped:
                    swapped = True
                    os.replace(replacement, receipt)
                return result

            try:
                with mock.patch.object(common.os, "read", side_effect=read_then_swap):
                    with self.assertRaises(common.ProvenanceV2Error) as raised:
                        common.read_bounded_regular_relative(root_fd, "receipt.json", "receipt")
            finally:
                os.close(root_fd)
        self.assertIn(getattr(raised.exception, "reason_code", None), {"raced-input", "mutated-input", "nonunique-evidence-inode"})

    def test_descriptors_are_strict_and_unique_by_path(self) -> None:
        first = common.descriptor_for_bytes("raw/first.json", b"first", "first")
        second = common.descriptor_for_bytes("raw/second.json", b"first", "second")
        self.assertEqual(common.require_unique_descriptors([first, second], "inputs"), (first, second))
        with self.assertRaises(common.ProvenanceV2Error) as raised:
            common.require_unique_descriptors([first, first], "inputs")
        self.assert_reason(raised, "duplicate-evidence-path")
        with self.assertRaises(common.ProvenanceV2Error) as raised:
            common.parse_descriptor({"path": "raw/first.json", "sha256": "0" * 64}, "input")
        self.assert_reason(raised, "invalid-descriptor")

    def test_create_only_json_and_marker_reject_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                created = common.write_create_only_json(root_fd, "receipt.json", {"result": "ok"}, "receipt")
                self.assertEqual(created.name, "receipt.json")
                self.assertEqual((root / "receipt.json").stat().st_mode & 0o777, 0o600)
                self.assertEqual((root / "receipt.json").read_bytes(), b'{"result":"ok"}')
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.write_create_only_json(root_fd, "receipt.json", {"result": "new"}, "receipt")
                self.assert_reason(raised, "create-only-collision")
                self.assertEqual((root / "receipt.json").read_bytes(), b'{"result":"ok"}')

                marker = common.create_incomplete_marker(
                    root_fd,
                    "capture-incomplete.json",
                    {"capture_status": "incomplete", "schema_version": "v2"},
                )
                self.assertEqual(marker.name, "capture-incomplete.json")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.create_incomplete_marker(
                        root_fd,
                        "capture-incomplete.json",
                        {"capture_status": "replacement", "schema_version": "v2"},
                    )
                self.assert_reason(raised, "create-only-collision")
            finally:
                os.close(root_fd)

    def test_create_only_does_not_follow_an_existing_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_bytes(b"original")
            (root / "receipt.json").symlink_to(target.name)
            root_fd = self.open_root(root)
            try:
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.write_create_only_json(root_fd, "receipt.json", {"result": "new"}, "receipt")
            finally:
                os.close(root_fd)
            self.assert_reason(raised, "create-only-collision")
            self.assertEqual(target.read_bytes(), b"original")

    def test_read_descriptor_json_binds_digest_length_and_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                raw = common.canonical_json_bytes({"sample": 1})
                common.write_create_only(root_fd, "sample.json", raw, "sample")
                descriptor = common.descriptor_for_bytes("sample.json", raw, "sample")
                received_raw, document = common.read_descriptor_json(root_fd, descriptor, "sample")
                self.assertEqual(received_raw, raw)
                self.assertEqual(document, {"sample": 1})
            finally:
                os.close(root_fd)

    def test_read_descriptor_bytes_binds_noncanonical_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                raw = b'[\n {"Id":"sha256:abc"}\n]\n'
                common.write_create_only(root_fd, "docker-inspect.json", raw, "docker inspect")
                descriptor = common.descriptor_for_bytes("docker-inspect.json", raw, "docker inspect")
                self.assertEqual(
                    common.read_descriptor_bytes(root_fd, descriptor, "docker inspect"), raw
                )
            finally:
                os.close(root_fd)

    def test_verify_descriptor_file_streams_digest_length_without_json_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                payload = b"artifact-bytes" * 4096
                common.write_create_only(root_fd, "artifact.bin", payload, "artifact")
                descriptor = common.descriptor_for_bytes("artifact.bin", payload, "artifact")
                common.verify_descriptor_file(root_fd, descriptor, "artifact")

                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.verify_descriptor_file(
                        root_fd,
                        {**descriptor.as_json(), "byte_length": len(payload) - 1},
                        "artifact",
                    )
                self.assert_reason(raised, "evidence-length-mismatch")
            finally:
                os.close(root_fd)

    def test_describe_regular_relative_streams_a_safe_artifact_without_materializing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                payload = b"artifact-bytes" * (2 * 1024 * 1024)
                common.write_create_only(root_fd, "artifact.bin", payload, "artifact")
                with mock.patch.object(
                    common,
                    "_read_exact_bounded",
                    side_effect=AssertionError("descriptor derivation must stream"),
                ):
                    descriptor = common.describe_regular_relative(
                        root_fd,
                        "artifact.bin",
                        "artifact",
                    )
                self.assertEqual(descriptor.path, "artifact.bin")
                self.assertEqual(descriptor.byte_length, len(payload))
                self.assertEqual(descriptor.sha256, common.descriptor_for_bytes(
                    "artifact.bin", payload, "artifact"
                ).sha256)
                common.verify_descriptor_file(root_fd, descriptor, "artifact")

                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.describe_regular_relative(
                        root_fd,
                        "artifact.bin",
                        "artifact",
                        maximum_bytes=len(payload) - 1,
                    )
                self.assert_reason(raised, "input-too-large")

                os.link(root / "artifact.bin", root / "artifact-alias.bin")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.describe_regular_relative(root_fd, "artifact.bin", "artifact")
                self.assert_reason(raised, "nonunique-evidence-inode")
            finally:
                os.close(root_fd)

    def test_paired_hardlink_publication_and_reader_remain_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            raw = common.canonical_json_bytes({"artifact": "marker"})
            try:
                common.write_create_only(root_fd, "marker.intent", raw, "marker intent")
                common.publish_create_only_hardlink(
                    root_fd,
                    "marker.intent",
                    "marker.complete",
                    "marker",
                )
                self.assertEqual(
                    common.read_bounded_paired_hardlink(
                        root_fd,
                        "marker.complete",
                        "marker.intent",
                        "marker",
                    ),
                    raw,
                )
                common.write_create_only(root_fd, "collision.intent", raw, "collision intent")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.publish_create_only_hardlink(
                        root_fd,
                        "collision.intent",
                        "marker.complete",
                        "marker",
                    )
                self.assert_reason(raised, "create-only-collision")

                os.link(root / "marker.complete", root / "marker-third-link")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_paired_hardlink(
                        root_fd,
                        "marker.complete",
                        "marker.intent",
                        "marker",
                    )
                self.assert_reason(raised, "invalid-paired-hardlink")
                os.unlink(root / "marker-third-link")

                os.unlink(root / "marker.complete")
                os.unlink(root / "marker.intent")
                (root / "first").write_bytes(raw)
                (root / "second").write_bytes(raw)
                os.chmod(root / "first", 0o600)
                os.chmod(root / "second", 0o600)
                os.link(root / "first", root / "marker.complete")
                os.link(root / "first", root / "first-peer")
                os.link(root / "second", root / "marker.intent")
                os.link(root / "second", root / "second-peer")
                os.unlink(root / "first")
                os.unlink(root / "second")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_bounded_paired_hardlink(
                        root_fd,
                        "marker.complete",
                        "marker.intent",
                        "marker",
                    )
                self.assert_reason(raised, "invalid-paired-hardlink")
            finally:
                os.close(root_fd)

    def test_paired_hardlink_publication_rejects_unsafe_source_modes_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            try:
                common.write_create_only(root_fd, "mode.intent", b"mode", "mode intent")
                os.chmod(root / "mode.intent", 0o644)
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.publish_create_only_hardlink(
                        root_fd,
                        "mode.intent",
                        "mode.complete",
                        "mode marker",
                    )
                self.assert_reason(raised, "unsafe-output-mode")

                common.write_create_only(root_fd, "linked.intent", b"linked", "linked intent")
                os.link(root / "linked.intent", root / "linked-peer")
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.publish_create_only_hardlink(
                        root_fd,
                        "linked.intent",
                        "linked.complete",
                        "linked marker",
                    )
                self.assert_reason(raised, "nonunique-evidence-inode")
            finally:
                os.close(root_fd)

    def test_paired_hardlink_post_link_sync_failure_is_explicitly_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            original = common._fsync_checked

            def fail_final_parent(descriptor: int, label: str) -> None:
                if label == "marker parent directory":
                    error = common.ProvenanceV2Error("fixture final marker directory sync failure")
                    error.reason_code = "durability-failure"  # type: ignore[attr-defined]
                    raise error
                original(descriptor, label)

            try:
                common.write_create_only(root_fd, "marker.intent", b"marker", "marker intent")
                with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
                    with self.assertRaises(common.ProvenanceV2Error) as raised:
                        common.publish_create_only_hardlink(
                            root_fd,
                            "marker.intent",
                            "marker.complete",
                            "marker",
                        )
                self.assert_reason(raised, "ambiguous-terminal-publication")
                self.assertTrue((root / "marker.intent").is_file())
                self.assertTrue((root / "marker.complete").is_file())
                self.assertEqual(
                    (root / "marker.intent").stat().st_ino,
                    (root / "marker.complete").stat().st_ino,
                )
            finally:
                os.close(root_fd)

    def test_held_child_rebinding_and_private_rebased_json_replay_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root_fd = self.open_root(root)
            child_fd: int | None = None
            try:
                child_fd = common.create_private_child_directory(root_fd, "capture", "capture directory")
                raw = common.canonical_json_bytes({"result": "captured"})
                common.write_create_only(child_fd, "session.json", raw, "session")
                root_descriptor = common.descriptor_for_bytes("capture/session.json", raw, "session")
                held_descriptor = common.rebase_descriptor_to_held_leaf(
                    root_descriptor,
                    expected_root_relative_path="capture/session.json",
                    leaf_name="session.json",
                    label="session",
                )
                self.assertEqual(held_descriptor.path, "session.json")
                received, document = common.read_private_descriptor_json_leaf(
                    child_fd,
                    held_descriptor,
                    "session",
                )
                self.assertEqual(received, raw)
                self.assertEqual(document, {"result": "captured"})
                self.assertEqual(
                    common.read_private_canonical_json_leaf(child_fd, "session.json", "session"),
                    {"result": "captured"},
                )

                for expected, leaf in (
                    ("other/session.json", "session.json"),
                    ("capture/session.json", "other.json"),
                ):
                    with self.subTest(expected=expected, leaf=leaf), self.assertRaises(common.ProvenanceV2Error) as raised:
                        common.rebase_descriptor_to_held_leaf(
                            root_descriptor,
                            expected_root_relative_path=expected,
                            leaf_name=leaf,
                            label="session",
                        )
                    self.assert_reason(raised, "invalid-descriptor")

                os.chmod(root / "capture" / "session.json", 0o644)
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.read_private_descriptor_json_leaf(child_fd, held_descriptor, "session")
                self.assert_reason(raised, "unsafe-output-mode")
                os.chmod(root / "capture" / "session.json", 0o600)

                moved = root / "capture-old"
                os.rename(root / "capture", moved)
                (root / "capture").mkdir(mode=0o700)
                os.chmod(root / "capture", 0o700)
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.require_private_child_directory_fd(
                        root_fd,
                        child_fd,
                        "capture",
                        "capture directory",
                    )
                self.assert_reason(raised, "raced-input")
            finally:
                if child_fd is not None:
                    os.close(child_fd)
                os.close(root_fd)

    def test_held_snapshot_consumer_binds_one_private_large_leaf_without_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary).resolve(strict=True)
            root_fd = common.create_private_evidence_directory(parent / "evidence", "evidence root")
            child_fd: int | None = None
            try:
                child_fd = common.create_private_child_directory(root_fd, "capture", "capture directory")
                raw = b"large immutable artifact" * 4096
                common.write_create_only(child_fd, "archive.tar", raw, "archive")
                root_descriptor = common.descriptor_for_bytes("capture/archive.tar", raw, "archive")
                held_descriptor = common.rebase_descriptor_to_held_leaf(
                    root_descriptor,
                    expected_root_relative_path="capture/archive.tar",
                    leaf_name="archive.tar",
                    label="archive",
                )
                self.assertEqual(
                    common.consume_private_snapshot_descriptor_file(
                        child_fd,
                        held_descriptor,
                        "archive",
                        lambda source: source.read(17),
                        maximum_bytes=len(raw),
                    ),
                    raw[:17],
                )
                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.consume_private_snapshot_descriptor_file(
                        child_fd,
                        {**held_descriptor.as_json(), "byte_length": len(raw) - 1},
                        "archive",
                        lambda source: source.read(1),
                        maximum_bytes=len(raw),
                    )
                self.assert_reason(raised, "evidence-length-mismatch")

                replacement = parent / "replacement.tar"
                replacement.write_bytes(raw)
                os.chmod(replacement, 0o600)

                def read_then_swap(source: object) -> bytes:
                    result = source.read(1)  # type: ignore[union-attr]
                    os.replace(replacement, parent / "evidence" / "capture" / "archive.tar")
                    return result

                with self.assertRaises(common.ProvenanceV2Error) as raised:
                    common.consume_private_snapshot_descriptor_file(
                        child_fd,
                        held_descriptor,
                        "archive",
                        read_then_swap,
                        maximum_bytes=len(raw),
                    )
                self.assertIn(
                    getattr(raised.exception, "reason_code", None),
                    {"raced-input", "mutated-input", "nonunique-evidence-inode"},
                )
            finally:
                if child_fd is not None:
                    os.close(child_fd)
                os.close(root_fd)


if __name__ == "__main__":
    unittest.main()
