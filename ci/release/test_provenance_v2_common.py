#!/usr/bin/env python3
"""Focused hostile-path tests for provenance_v2_common."""

from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
