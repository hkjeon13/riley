#!/usr/bin/env python3
"""CPU-only hostile-path tests for the private Gate E aggregate replay record."""

from __future__ import annotations

import ast
import copy
import fcntl
import hashlib
import inspect
import json
import os
import stat
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, TypeVar
from unittest import mock


sys.path.insert(0, os.fspath(Path(__file__).parent))

import provenance_v2_common as common  # noqa: E402
import rc3_frozen_candidate_topology as topology  # noqa: E402
import replay_rc3_gate_e_aggregate_v1 as aggregate  # noqa: E402
import replay_rc3_gate_e_input_inventory_v1 as gate_inputs  # noqa: E402
import test_replay_rc3_gate_e_aggregate_v1 as aggregate_tests  # noqa: E402
import write_rc3_gate_e_aggregate_replay_receipt_v1 as receipt  # noqa: E402


T = TypeVar("T")


class GateEAggregateReplayReceiptV1Tests(unittest.TestCase):
    """Drive the private terminal writer with mocked semantic child cores."""

    def setUp(self) -> None:
        self.aggregate_fixture = aggregate_tests.Rc3GateEAggregateV1Tests("runTest")
        self.aggregate_fixture.setUp()
        self.receipt_root = self.aggregate_fixture.gate.base / "aggregate-replay-record"
        self.receipt_root.mkdir(mode=0o700)
        self.receipt_root.chmod(0o700)

    def tearDown(self) -> None:
        self.aggregate_fixture.tearDown()

    def assert_reason(
        self,
        raised: unittest.case._AssertRaisesContext[BaseException],
        code: str,
    ) -> None:
        self.assertEqual(getattr(raised.exception, "reason_code", None), code)

    def _release_image(self) -> str:
        return self.aggregate_fixture._release_image()  # noqa: SLF001

    def _golden_sha256(self) -> str:
        return self.aggregate_fixture._golden_sha256()  # noqa: SLF001

    def _snapshot_tree(self, root: Path) -> dict[str, bytes]:
        return self.aggregate_fixture._snapshot_tree(root)  # noqa: SLF001

    def _with_held_fds(
        self,
        call: Callable[[int, int, int, int, int], T],
        *,
        receipt_root: Path | None = None,
    ) -> T:
        root = self.receipt_root if receipt_root is None else receipt_root
        source_fd = common.open_absolute_directory(
            self.aggregate_fixture.gate.fixture.root,  # noqa: SLF001
            "aggregate receipt fixture source",
        )
        input_fd = common.open_private_evidence_directory(
            self.aggregate_fixture.gate.fixture.evidence,  # noqa: SLF001
            "aggregate receipt fixture input",
        )
        frozen_fd = common.open_private_evidence_directory(
            self.aggregate_fixture.gate.frozen_root,  # noqa: SLF001
            "aggregate receipt fixture frozen candidate",
        )
        gate_fd = common.open_private_evidence_directory(
            self.aggregate_fixture.gate.gate_root,  # noqa: SLF001
            "aggregate receipt fixture Gate E",
        )
        receipt_fd = common.open_private_evidence_directory(
            root,
            "aggregate receipt fixture output",
        )
        try:
            # This is the exact outer-session lock order the private writer
            # requires.  It must neither relock nor upgrade these descriptors.
            for descriptor, mode in (
                (input_fd, fcntl.LOCK_SH),
                (frozen_fd, fcntl.LOCK_SH),
                (gate_fd, fcntl.LOCK_EX),
                (receipt_fd, fcntl.LOCK_EX),
            ):
                fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
            return call(receipt_fd, gate_fd, frozen_fd, input_fd, source_fd)
        finally:
            for descriptor in (receipt_fd, gate_fd, frozen_fd, input_fd):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
            os.close(source_fd)

    def _call(
        self,
        receipt_fd: int,
        gate_fd: int,
        frozen_fd: int,
        input_fd: int,
        source_fd: int,
        *,
        receipt_root: Path | None = None,
        gate_e_evidence_root: Path | None = None,
    ) -> dict[str, Any]:
        return receipt._replay_and_publish_gate_e_aggregate_receipt_on_held_fds(  # noqa: SLF001
            self.receipt_root if receipt_root is None else receipt_root,
            receipt_fd,
            (
                self.aggregate_fixture.gate.gate_root  # noqa: SLF001
                if gate_e_evidence_root is None
                else gate_e_evidence_root
            ),
            gate_fd,
            self.aggregate_fixture.gate.frozen_root,  # noqa: SLF001
            frozen_fd,
            self.aggregate_fixture.gate.fixture.evidence,  # noqa: SLF001
            input_fd,
            self.aggregate_fixture.gate.fixture.root,  # noqa: SLF001
            source_fd,
            aggregate.EXTERNAL_SCRATCH_PARENT,
            self._release_image(),
            aggregate_tests.OPTIMIZER_IMAGE,
            self._golden_sha256(),
        )

    def _publish(self) -> dict[str, Any]:
        results = self.aggregate_fixture._component_results()  # noqa: SLF001
        patches = self.aggregate_fixture._component_patches(results)  # noqa: SLF001
        with self.aggregate_fixture.gate._defaults(), mock.patch.object(  # noqa: SLF001
            gate_inputs,
            "_replay_rc3_gate_e_input_inventory_v1_on_held_fds",
            return_value=self.aggregate_fixture._structural(),  # noqa: SLF001
        ), patches[0], patches[1], patches[2], patches[3], patches[4]:
            return self._with_held_fds(self._call)

    def _valid_aggregate_result(self) -> dict[str, Any]:
        return self.aggregate_fixture._replay(  # noqa: SLF001
            self.aggregate_fixture._component_results()  # noqa: SLF001
        )

    def test_publishes_one_closed_replay_only_record_in_a_separate_root(self) -> None:
        before = {
            "gate": self._snapshot_tree(self.aggregate_fixture.gate.gate_root),  # noqa: SLF001
            "frozen": self._snapshot_tree(self.aggregate_fixture.gate.frozen_root),  # noqa: SLF001
            "input": self._snapshot_tree(self.aggregate_fixture.gate.fixture.evidence),  # noqa: SLF001
            "source": self._snapshot_tree(self.aggregate_fixture.gate.fixture.root),  # noqa: SLF001
        }
        original = receipt.aggregate._replay_rc3_gate_e_aggregate_v1_on_held_fds  # noqa: SLF001
        with mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=original,
        ) as replay:
            document = self._publish()
        self.assertEqual(replay.call_count, 2)
        self.assertEqual(document["schema_version"], receipt.RECEIPT_VERSION)
        self.assertEqual(document["scope"], receipt.SCOPE)
        self.assertEqual(document["authority"], aggregate.AUTHORITY)
        self.assertEqual(document["status"], "bound")
        self.assertEqual(document["candidate_status"], "frozen")
        self.assertEqual(document["qualification_status"], "not-run")
        self.assertNotIn("gate_e_status", document)
        self.assertNotIn("semantic_receipt", document)
        self.assertEqual(document["not_established"], receipt.NOT_ESTABLISHED)
        aggregate_record = document["aggregate_replay"]
        self.assertIsInstance(aggregate_record, dict)
        self.assertEqual(len(aggregate_record["report_sha256"]), 64)
        self.assertGreater(aggregate_record["report_byte_length"], 0)
        self.assertEqual(
            self._snapshot_tree(self.aggregate_fixture.gate.gate_root),  # noqa: SLF001
            before["gate"],
        )
        self.assertEqual(
            self._snapshot_tree(self.aggregate_fixture.gate.frozen_root),  # noqa: SLF001
            before["frozen"],
        )
        self.assertEqual(
            self._snapshot_tree(self.aggregate_fixture.gate.fixture.evidence),  # noqa: SLF001
            before["input"],
        )
        self.assertEqual(
            self._snapshot_tree(self.aggregate_fixture.gate.fixture.root),  # noqa: SLF001
            before["source"],
        )
        self.assertEqual(
            (self.receipt_root / receipt.RECEIPT_NAME).read_bytes(),
            common.canonical_json_bytes(document),
        )
        intent = self.receipt_root / f"{receipt.RECEIPT_NAME}.intent"
        complete = self.receipt_root / f"{receipt.RECEIPT_NAME}.complete"
        completion = json.loads(intent.read_text(encoding="utf-8"))
        self.assertEqual(
            completion,
            {
                "schema_version": receipt.RECEIPT_COMPLETION_VERSION,
                "artifact_filename": receipt.RECEIPT_NAME,
                "artifact_sha256": hashlib.sha256(
                    common.canonical_json_bytes(document)
                ).hexdigest(),
            },
        )
        self.assertEqual(complete.read_bytes(), intent.read_bytes())
        self.assertEqual(
            {path.name for path in self.receipt_root.iterdir()},
            {receipt.RECEIPT_NAME, intent.name, complete.name},
        )
        self.assertEqual(stat.S_IMODE(intent.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(complete.stat().st_mode), 0o600)
        self.assertEqual(intent.stat().st_nlink, 2)
        self.assertEqual(complete.stat().st_nlink, 2)
        self.assertEqual(
            (intent.stat().st_dev, intent.stat().st_ino),
            (complete.stat().st_dev, complete.stat().st_ino),
        )

    def test_rejects_a_nonempty_receipt_root_before_any_aggregate_replay(self) -> None:
        (self.receipt_root / "foreign").write_bytes(b"occupied\n")
        with mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=AssertionError("aggregate must not start after receipt-root collision"),
        ):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._with_held_fds(self._call)
        self.assert_reason(raised, "receipt-root-not-empty")

    def test_rejects_receipt_root_aliasing_gate_e_root_before_aggregate_replay(self) -> None:
        gate_root = self.aggregate_fixture.gate.gate_root  # noqa: SLF001
        source_fd = common.open_absolute_directory(
            self.aggregate_fixture.gate.fixture.root,  # noqa: SLF001
            "aggregate receipt alias source",
        )
        input_fd = common.open_private_evidence_directory(
            self.aggregate_fixture.gate.fixture.evidence,  # noqa: SLF001
            "aggregate receipt alias input",
        )
        frozen_fd = common.open_private_evidence_directory(
            self.aggregate_fixture.gate.frozen_root,  # noqa: SLF001
            "aggregate receipt alias frozen",
        )
        gate_fd = common.open_private_evidence_directory(gate_root, "aggregate receipt alias Gate E")
        try:
            for descriptor, mode in (
                (input_fd, fcntl.LOCK_SH),
                (frozen_fd, fcntl.LOCK_SH),
                (gate_fd, fcntl.LOCK_EX),
            ):
                fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
            with mock.patch.object(
                receipt.aggregate,
                "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
                side_effect=AssertionError("aggregate must not run for an aliased receipt root"),
            ):
                with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                    self._call(
                        gate_fd,
                        gate_fd,
                        frozen_fd,
                        input_fd,
                        source_fd,
                        receipt_root=gate_root,
                        gate_e_evidence_root=gate_root,
                    )
            self.assert_reason(raised, "frozen-candidate-root-overlap")
        finally:
            for descriptor in (gate_fd, frozen_fd, input_fd):
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
                os.close(descriptor)
            os.close(source_fd)

    def test_rejects_receipt_root_nested_under_gate_e_root_before_aggregate_replay(self) -> None:
        nested = self.aggregate_fixture.gate.gate_root / "receipt-child"  # noqa: SLF001
        nested.mkdir(mode=0o700)
        nested.chmod(0o700)
        with mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=AssertionError("aggregate must not run for nested receipt root"),
        ):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._with_held_fds(
                    lambda receipt_fd, gate_fd, frozen_fd, input_fd, source_fd: self._call(
                        receipt_fd,
                        gate_fd,
                        frozen_fd,
                        input_fd,
                        source_fd,
                        receipt_root=nested,
                    ),
                    receipt_root=nested,
                )
        self.assert_reason(raised, "frozen-candidate-root-overlap")

    def test_rejects_visible_mount_topology_failure_before_aggregate_replay(self) -> None:
        error = topology.FrozenCandidateTopologyError("fixture mount alias")
        error.reason_code = "frozen-candidate-mount-alias"  # type: ignore[attr-defined]
        with mock.patch.object(
            receipt.topology,
            "assert_existing_roots_disjoint",
            side_effect=error,
        ), mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=AssertionError("aggregate must not run after mount topology failure"),
        ):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._with_held_fds(self._call)
        self.assert_reason(raised, "frozen-candidate-mount-alias")
        self.assertEqual(list(self.receipt_root.iterdir()), [])

    def test_drift_between_the_two_aggregate_replays_creates_no_record(self) -> None:
        first = self._valid_aggregate_result()
        second = copy.deepcopy(first)
        second["source_revision"] = "e" * 40
        with mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=[first, second],
        ):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._with_held_fds(self._call)
        self.assert_reason(raised, "aggregate-replay-drift")
        self.assertEqual(list(self.receipt_root.iterdir()), [])

    def test_aggregate_failure_creates_no_record(self) -> None:
        error = aggregate.AggregateReplayError("fixture aggregate failure")
        error.reason_code = "fixture-aggregate-failure"  # type: ignore[attr-defined]
        with mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=error,
        ):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._with_held_fds(self._call)
        self.assert_reason(raised, "fixture-aggregate-failure")
        self.assertEqual(list(self.receipt_root.iterdir()), [])

    def test_final_completion_sync_ambiguity_never_returns_success(self) -> None:
        original = common._fsync_checked

        def fail_final_parent(descriptor: int, label: str) -> None:
            if label == "Gate E aggregate-replay record completion marker parent directory":
                common._fail("durability-failure", "fixture final marker sync failure")
            original(descriptor, label)

        with mock.patch.object(common, "_fsync_checked", side_effect=fail_final_parent):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                self._publish()
        self.assert_reason(raised, "ambiguous-terminal-publication")
        self.assertTrue((self.receipt_root / receipt.RECEIPT_NAME).is_file())
        self.assertTrue((self.receipt_root / f"{receipt.RECEIPT_NAME}.intent").is_file())
        self.assertTrue((self.receipt_root / f"{receipt.RECEIPT_NAME}.complete").is_file())

    def test_successful_publish_has_no_post_publish_aggregate_replay(self) -> None:
        original_publish = common.publish_create_only_hardlink
        original_aggregate = receipt.aggregate._replay_rc3_gate_e_aggregate_v1_on_held_fds  # noqa: SLF001
        published = False

        def mark_publish(*args: Any, **kwargs: Any) -> None:
            nonlocal published
            original_publish(*args, **kwargs)
            destination = args[2] if len(args) >= 3 else kwargs.get("destination_name")
            if destination == f"{receipt.RECEIPT_NAME}.complete":
                published = True

        def reject_post_publish_aggregate(*args: Any, **kwargs: Any) -> Any:
            if published:
                raise AssertionError("aggregate replay must not run after terminal publication")
            return original_aggregate(*args, **kwargs)

        with mock.patch.object(
            common,
            "publish_create_only_hardlink",
            side_effect=mark_publish,
        ), mock.patch.object(
            receipt.aggregate,
            "_replay_rc3_gate_e_aggregate_v1_on_held_fds",
            side_effect=reject_post_publish_aggregate,
        ):
            document = self._publish()
        self.assertEqual(document["status"], "bound")

    def test_bytecode_guard_and_private_surface_are_closed(self) -> None:
        with mock.patch.object(receipt, "_BYTECODE_DISABLED_AT_STARTUP", False):
            with self.assertRaises(receipt.GateEAggregateReplayReceiptError) as raised:
                receipt._replay_and_publish_gate_e_aggregate_receipt_on_held_fds(  # noqa: SLF001
                    self.receipt_root,
                    -1,
                    self.aggregate_fixture.gate.gate_root,  # noqa: SLF001
                    -1,
                    self.aggregate_fixture.gate.frozen_root,  # noqa: SLF001
                    -1,
                    self.aggregate_fixture.gate.fixture.evidence,  # noqa: SLF001
                    -1,
                    self.aggregate_fixture.gate.fixture.root,  # noqa: SLF001
                    -1,
                    aggregate.EXTERNAL_SCRATCH_PARENT,
                    self._release_image(),
                    aggregate_tests.OPTIMIZER_IMAGE,
                    self._golden_sha256(),
                )
        self.assert_reason(raised, "bytecode-cache-write-not-permitted")
        source = Path(receipt.__file__).read_text(encoding="utf-8")
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
            "replay_rc3_gate_e_aggregate_v1(",
            "write_rc3_gate_e_input_snapshot_v1",
            "import subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "def resume",
            "def retry",
            "import fcntl",
            "flock(",
            "open_private_evidence_directory(",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("assert_existing_roots_disjoint", source)
        self.assertIn("require_distinct_root_fds", source)
        self.assertEqual(
            list(
                inspect.signature(
                    receipt._replay_and_publish_gate_e_aggregate_receipt_on_held_fds  # noqa: SLF001
                ).parameters
            ),
            [
                "receipt_root",
                "receipt_root_fd",
                "gate_e_evidence_root",
                "gate_e_evidence_root_fd",
                "frozen_candidate_root",
                "frozen_candidate_root_fd",
                "input_evidence_root",
                "input_evidence_root_fd",
                "repository_root",
                "repository_root_fd",
                "scratch_parent",
                "expected_release_image_id",
                "expected_optimizer_build_image_id",
                "expected_correctness_golden_sha256",
            ],
        )

    def test_schema_keeps_the_record_replay_only(self) -> None:
        schema_path = (
            Path(receipt.__file__).resolve().parents[2]
            / "benchmarks/release/candidates"
            / "rc3-gate-e-aggregate-replay-record-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        properties = schema["properties"]
        self.assertEqual(properties["schema_version"]["const"], receipt.RECEIPT_VERSION)
        self.assertEqual(properties["scope"]["const"], receipt.SCOPE)
        self.assertEqual(properties["authority"]["const"], aggregate.AUTHORITY)
        self.assertNotIn("gate_e_status", properties)
        self.assertEqual(
            set(schema["$defs"]["notEstablished"]["required"]),
            set(receipt.NOT_ESTABLISHED),
        )
        self.assertIn(
            "not actual GPU capture",
            schema["description"],
        )
        completion_schema_path = (
            Path(receipt.__file__).resolve().parents[2]
            / "benchmarks/release/candidates"
            / "rc3-gate-e-aggregate-replay-record-completion-v1.schema.json"
        )
        completion_schema = json.loads(completion_schema_path.read_text(encoding="utf-8"))
        completion_properties = completion_schema["properties"]
        self.assertEqual(
            completion_properties["schema_version"]["const"],
            receipt.RECEIPT_COMPLETION_VERSION,
        )
        self.assertEqual(
            completion_properties["artifact_filename"]["const"],
            receipt.RECEIPT_NAME,
        )
        self.assertTrue(completion_schema["additionalProperties"] is False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
