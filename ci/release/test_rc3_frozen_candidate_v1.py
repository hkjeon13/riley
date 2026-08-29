#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 frozen-candidate boundary."""

from __future__ import annotations

import copy
import fcntl
import io
import inspect
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys_path = Path(__file__).parent
import sys

sys.path.insert(0, os.fspath(sys_path))

import check_rc3_prefreeze as prefreeze  # noqa: E402
import provenance_v2_common as common  # noqa: E402
import rc3_frozen_candidate_common as frozen  # noqa: E402
import rc3_frozen_candidate_topology as topology  # noqa: E402
import replay_rc3_frozen_candidate_v1 as replayer  # noqa: E402
import write_rc3_frozen_candidate_v1 as writer  # noqa: E402
from test_check_rc3_freeze_input_admission import (  # noqa: E402
    AdmissionFixture,
    CANDIDATE_ID,
    _sha256,
)


class _CapturedStdout:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


class Rc3FrozenCandidateV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name).resolve(strict=True)
        self.fixture = AdmissionFixture(self.base)
        self.frozen_root = self.base / "frozen-candidate"

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

    def _write(self) -> dict[str, object]:
        with self._defaults():
            return writer.write_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
                expected_revision=self.fixture.revision,
                candidate_id=CANDIDATE_ID,
                request_name=self.fixture.request_name,
            )

    def _replay(self) -> dict[str, object]:
        with self._defaults():
            return replayer.replay_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
            )

    def test_writer_creates_one_identity_only_manifest_and_self_replays(self) -> None:
        input_before = self.fixture.evidence_snapshot()
        result = self._write()

        self.assertEqual(self.fixture.evidence_snapshot(), input_before)
        self.assertEqual(self.fixture.source_status(), b"")
        self.assertEqual(
            {entry.name for entry in self.frozen_root.iterdir()},
            {frozen.MANIFEST_NAME},
        )
        manifest_path = self.frozen_root / frozen.MANIFEST_NAME
        metadata = manifest_path.stat()
        self.assertEqual(stat.S_IMODE(metadata.st_mode), 0o600)
        self.assertEqual(metadata.st_nlink, 1)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest, result["manifest"])
        self.assertEqual(manifest["schema_version"], frozen.MANIFEST_VERSION)
        self.assertEqual(manifest["status"], "frozen")
        self.assertEqual(manifest["candidate_status"], "frozen")
        self.assertEqual(manifest["qualification_status"], "not-run")
        self.assertEqual(manifest["authority"], frozen.MANIFEST_AUTHORITY)
        self.assertEqual(manifest["not_established"]["writer_normal_return"], "not-established")
        self.assertEqual(manifest["not_established"]["input_root_immutability"], "not-established")
        self.assertNotIn("passed", manifest)
        self.assertNotIn("freeze_sha256", manifest)
        self.assertNotIn("gate_e_report_sha256", manifest)
        self.assertEqual(
            manifest["freeze_input_request"]["path"],
            self.fixture.request_name,
        )
        self.assertEqual(
            manifest["bound_inputs"],
            self.fixture.request,
        )
        self.assertEqual(
            manifest["reconstructed_baseline"],
            {
                "baseline_id": "reconstructed-riley-0.1.0-rc2",
                "tag_name": "riley-0.1.0-rc2",
                "target_commit_sha1": "b" * 40,
                "relationship": "immediately-prior-rc-same-semver",
                "provenance_class": "reconstructed-from-source",
                "historical_distribution": "not-attested",
            },
        )
        replay = result["replay"]
        self.assertIsInstance(replay, dict)
        assert isinstance(replay, dict)
        self.assertEqual(replay["status"], "bound")
        self.assertEqual(replay["qualification_status"], "not-run")
        self.assertNotIn("passed", replay)
        self.assertEqual(result["manifest_descriptor"], replay["frozen_candidate_manifest"])

    def test_public_and_private_held_fd_replay_are_read_only(self) -> None:
        self._write()
        frozen_before = (self.frozen_root / frozen.MANIFEST_NAME).read_bytes()
        input_before = self.fixture.evidence_snapshot()
        with self._defaults(), mock.patch.object(
            common,
            "write_create_only",
            side_effect=AssertionError("replayer must not write"),
        ), mock.patch.object(
            common,
            "write_create_only_json",
            side_effect=AssertionError("replayer must not write JSON"),
        ), mock.patch.object(
            common,
            "create_private_evidence_directory",
            side_effect=AssertionError("replayer must not create a root"),
        ):
            public = replayer.replay_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
            )
        self.assertEqual((self.frozen_root / frozen.MANIFEST_NAME).read_bytes(), frozen_before)
        self.assertEqual(self.fixture.evidence_snapshot(), input_before)

        source_fd = common.open_absolute_directory(self.fixture.root, "fixture source")
        input_fd = common.open_private_evidence_directory(self.fixture.evidence, "fixture input")
        frozen_fd = common.open_private_evidence_directory(self.frozen_root, "fixture frozen")
        try:
            fcntl.flock(input_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(frozen_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with self._defaults():
                private = replayer._replay_rc3_frozen_candidate_v1_on_held_fds(
                    frozen_fd,
                    input_fd,
                    self.fixture.root,
                    source_fd,
                )
        finally:
            fcntl.flock(frozen_fd, fcntl.LOCK_UN)
            fcntl.flock(input_fd, fcntl.LOCK_UN)
            os.close(frozen_fd)
            os.close(input_fd)
            os.close(source_fd)
        self.assertEqual(public, private)
        self.assertEqual((self.frozen_root / frozen.MANIFEST_NAME).read_bytes(), frozen_before)
        self.assertEqual(self.fixture.evidence_snapshot(), input_before)

    def test_writer_cli_emits_canonical_result_after_create_only_self_replay(self) -> None:
        stdout = _CapturedStdout()
        with self._defaults(), mock.patch.object(writer.sys, "stdout", stdout):
            result = writer.main(
                [
                    "--frozen-candidate-root",
                    os.fspath(self.frozen_root),
                    "--input-evidence-root",
                    os.fspath(self.fixture.evidence),
                    "--repository-root",
                    os.fspath(self.fixture.root),
                    "--expected-revision",
                    self.fixture.revision,
                    "--candidate-id",
                    CANDIDATE_ID,
                    "--request",
                    self.fixture.request_name,
                ]
            )
        self.assertEqual(result, 0)
        raw = stdout.buffer.getvalue()
        self.assertTrue(raw.endswith(b"\n"))
        document = json.loads(raw)
        self.assertEqual(raw[:-1], common.canonical_json_bytes(document))
        self.assertEqual(document["manifest"]["status"], "frozen")
        self.assertEqual(document["replay"]["status"], "bound")

        replay_stdout = _CapturedStdout()
        with self._defaults(), mock.patch.object(replayer.sys, "stdout", replay_stdout):
            replay_result = replayer.main(
                [
                    "--frozen-candidate-root",
                    os.fspath(self.frozen_root),
                    "--input-evidence-root",
                    os.fspath(self.fixture.evidence),
                    "--repository-root",
                    os.fspath(self.fixture.root),
                ]
            )
        self.assertEqual(replay_result, 0)
        replay_raw = replay_stdout.buffer.getvalue()
        replay_document = json.loads(replay_raw)
        self.assertEqual(replay_raw, common.canonical_json_bytes(replay_document) + b"\n")
        self.assertEqual(replay_document["status"], "bound")

    def test_replayer_rejects_mutated_request_and_manifest_extra_leaf(self) -> None:
        self._write()
        self.fixture.request["toolchain"]["driver_version"] = "550.55"
        self.fixture.write_request()
        with self.assertRaises(replayer.FrozenCandidateReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "frozen-manifest-replay-mismatch")

        clean = AdmissionFixture(self.base / "extra-leaf")
        extra_root = self.base / "extra-frozen"
        with mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(clean.defaults),
        ):
            writer.write_rc3_frozen_candidate_v1(
                extra_root,
                input_evidence_root=clean.evidence,
                repository_root=clean.root,
                expected_revision=clean.revision,
                candidate_id=CANDIDATE_ID,
                request_name=clean.request_name,
            )
        (extra_root / "unexpected.json").write_bytes(b"{}")
        with mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(clean.defaults),
        ), self.assertRaises(replayer.FrozenCandidateReplayError) as raised:
            replayer.replay_rc3_frozen_candidate_v1(
                extra_root,
                input_evidence_root=clean.evidence,
                repository_root=clean.root,
            )
        self.assert_reason(raised, "unexpected-evidence-entry")

    def test_cross_role_baseline_descriptor_alias_is_rejected_before_output(self) -> None:
        baseline_archive = self.fixture.baseline_fixture.archive
        self.fixture.request["source"]["archive"] = baseline_archive
        self.fixture.write_request()
        with mock.patch.object(
            frozen.baseline,
            "_verify_raw_descriptor",
            side_effect=AssertionError("cross-role alias must fail before baseline raw streaming"),
        ), self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            self._write()
        self.assert_reason(raised, "duplicate-evidence-path")
        self.assertFalse(self.frozen_root.exists())

        recipe_alias = AdmissionFixture(self.base / "recipe-alias")
        recipe_alias.request["release"]["elf"] = recipe_alias.baseline_fixture.a_recipe
        recipe_alias.write_request()
        recipe_root = self.base / "recipe-alias-frozen"
        with mock.patch.object(
            prefreeze,
            "SERVER_DEFAULTS_SOURCE_SHA256",
            _sha256(recipe_alias.defaults),
        ), mock.patch.object(
            frozen.baseline,
            "_verify_raw_descriptor",
            side_effect=AssertionError("cross-role alias must fail before baseline raw streaming"),
        ), self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            writer.write_rc3_frozen_candidate_v1(
                recipe_root,
                input_evidence_root=recipe_alias.evidence,
                repository_root=recipe_alias.root,
                expected_revision=recipe_alias.revision,
                candidate_id=CANDIDATE_ID,
                request_name=recipe_alias.request_name,
            )
        self.assert_reason(raised, "duplicate-evidence-path")
        self.assertFalse(recipe_root.exists())

    def test_full_closure_budget_fails_before_baseline_raw_streaming(self) -> None:
        recipe_path = self.fixture.evidence / "reproductions/a/recipe-inspect.json"
        recipe_document = json.loads(recipe_path.read_text(encoding="utf-8"))
        recipe_document["recipe"]["byte_length"] = frozen.MAX_FROZEN_INPUT_CLOSURE_BYTES + 1
        recipe_descriptor = self.fixture.put(
            "reproductions/a/recipe-inspect.json",
            common.canonical_json_bytes(recipe_document),
        )
        receipt = copy.deepcopy(self.fixture.baseline_fixture.a_receipt)
        receipt["recipe_inspect"] = recipe_descriptor
        receipt_descriptor = self.fixture.put(
            self.fixture.baseline_fixture.a_receipt_path,
            common.canonical_json_bytes(receipt),
        )
        self.fixture.baseline_document["reproductions"]["a"] = receipt_descriptor
        baseline_descriptor = self.fixture.write_baseline()
        self.fixture.request["rollback"]["reconstructed_baseline_manifest"] = baseline_descriptor
        self.fixture.write_request()

        with mock.patch.object(
            frozen.baseline,
            "_verify_raw_descriptor",
            side_effect=AssertionError("closure budget must precede baseline raw streaming"),
        ), mock.patch.object(
            frozen.freeze_inputs,
            "_verify_opaque_inputs",
            side_effect=AssertionError("closure budget must precede candidate raw streaming"),
        ), self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            self._write()
        self.assert_reason(raised, "external-evidence-byte-budget-exceeded")
        self.assertFalse(self.frozen_root.exists())

    def test_canonical_request_leaf_participates_in_cross_role_closure(self) -> None:
        request_descriptor = common.EvidenceDescriptor(
            path=self.fixture.request_name,
            sha256="a" * 64,
            byte_length=1,
        )
        baseline_descriptor = common.EvidenceDescriptor(
            path=self.fixture.request_name,
            sha256="b" * 64,
            byte_length=1,
        )
        with self.assertRaises(frozen.FrozenCandidateError) as raised:
            frozen._bound_frozen_input_closure(  # noqa: SLF001
                (request_descriptor,),
                (baseline_descriptor,),
            )
        self.assert_reason(raised, "duplicate-evidence-path")

    def test_full_closure_descriptor_budget_is_global(self) -> None:
        descriptors = tuple(
            common.EvidenceDescriptor(
                path=f"budget/{index}",
                sha256="a" * 64,
                byte_length=1,
            )
            for index in range(frozen.MAX_FROZEN_INPUT_CLOSURE_DESCRIPTORS + 1)
        )
        with self.assertRaises(frozen.FrozenCandidateError) as raised:
            frozen._bound_frozen_input_closure(descriptors, ())  # noqa: SLF001
        self.assert_reason(raised, "too-many-external-descriptors")

    def test_output_collision_and_path_overlap_fail_without_replacement(self) -> None:
        self.frozen_root.mkdir(mode=0o700)
        self.frozen_root.chmod(0o700)
        sentinel = self.frozen_root / "sentinel"
        sentinel.write_bytes(b"retain")
        with self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            self._write()
        self.assert_reason(raised, "create-only-collision")
        self.assertEqual(sentinel.read_bytes(), b"retain")

        nested = self.fixture.evidence / "new-frozen"
        with self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            with self._defaults():
                writer.write_rc3_frozen_candidate_v1(
                    nested,
                    input_evidence_root=self.fixture.evidence,
                    repository_root=self.fixture.root,
                    expected_revision=self.fixture.revision,
                    candidate_id=CANDIDATE_ID,
                    request_name=self.fixture.request_name,
                )
        self.assert_reason(raised, "frozen-candidate-root-overlap")
        self.assertFalse(nested.exists())

    def test_writer_rejects_visible_root_replacement_after_self_replay(self) -> None:
        displaced = self.base / "displaced-frozen-candidate"
        original_replay = replayer._replay_rc3_frozen_candidate_v1_on_held_fds

        def replace_visible_root(*args, **kwargs):
            os.rename(self.frozen_root, displaced)
            self.frozen_root.mkdir(mode=0o700)
            self.frozen_root.chmod(0o700)
            return original_replay(*args, **kwargs)

        with self._defaults(), mock.patch.object(
            writer.replayer,
            "_replay_rc3_frozen_candidate_v1_on_held_fds",
            side_effect=replace_visible_root,
        ), self.assertRaises(writer.FrozenCandidateWriterError) as raised:
            writer.write_rc3_frozen_candidate_v1(
                self.frozen_root,
                input_evidence_root=self.fixture.evidence,
                repository_root=self.fixture.root,
                expected_revision=self.fixture.revision,
                candidate_id=CANDIDATE_ID,
                request_name=self.fixture.request_name,
            )
        self.assert_reason(raised, "raced-output")
        self.assertEqual(
            {entry.name for entry in displaced.iterdir()},
            {frozen.MANIFEST_NAME},
        )
        self.assertEqual(tuple(self.frozen_root.iterdir()), ())

    def test_topology_rejects_fd_and_linux_mount_backing_aliases(self) -> None:
        ancestor = self.base / "topology-ancestor"
        child = ancestor / "child"
        child.mkdir(parents=True)
        ancestor_fd = common.open_absolute_directory(ancestor, "topology ancestor")
        child_fd = common.open_absolute_directory(child, "topology child")
        try:
            with self.assertRaises(topology.FrozenCandidateTopologyError) as raised:
                topology.assert_existing_roots_disjoint(
                    {
                        "topology ancestor": (ancestor, ancestor_fd),
                        "topology child": (child, child_fd),
                    }
                )
        finally:
            os.close(child_fd)
            os.close(ancestor_fd)
        self.assert_reason(raised, "frozen-candidate-root-overlap")

        source = self.base / "topology-source"
        input_root = self.base / "topology-input"
        source.mkdir()
        input_root.mkdir()
        source_fd = common.open_absolute_directory(source, "topology source")
        input_fd = common.open_absolute_directory(input_root, "topology input")
        displaced = self.base / "topology-source-displaced"
        os.rename(source, displaced)
        source.mkdir()
        try:
            with self.assertRaises(topology.FrozenCandidateTopologyError) as raised:
                topology.assert_existing_roots_disjoint(
                    {
                        "source checkout": (source, source_fd),
                        "freeze-input evidence root": (input_root, input_fd),
                    }
                )
        finally:
            os.close(input_fd)
            os.close(source_fd)
        self.assert_reason(raised, "raced-output")

        mountinfo = (
            "36 25 0:42 /checkout /visible/source rw - ext4 /dev/x rw\n"
            "37 25 0:42 /checkout/output /visible/frozen rw - ext4 /dev/x rw\n"
        )
        with mock.patch.object(topology.sys, "platform", "linux"), mock.patch.object(
            topology.Path,
            "read_text",
            return_value=mountinfo,
        ), self.assertRaises(topology.FrozenCandidateTopologyError) as raised:
            topology._assert_mount_regions_disjoint(  # noqa: SLF001
                Path("/visible/frozen"),
                "frozen candidate root",
                Path("/visible/source"),
                "source checkout",
            )
        self.assert_reason(raised, "frozen-candidate-mount-alias")

        ambiguous_mountinfo = (
            "36 25 0:42 /one /visible/shared rw - ext4 /dev/x rw\n"
            "37 25 0:42 /two /visible/shared rw - ext4 /dev/y rw\n"
        )
        with self.assertRaises(topology.FrozenCandidateTopologyError) as raised:
            topology._mount_for_path(  # noqa: SLF001
                topology._mount_records(ambiguous_mountinfo),  # noqa: SLF001
                Path("/visible/shared/child"),
            )
        self.assert_reason(raised, "unsafe-evidence-directory")

    def test_manifest_identity_cannot_steer_a_different_request_name(self) -> None:
        self._write()
        path = self.frozen_root / frozen.MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        document["freeze_input_request"]["path"] = "nested/request.json"
        path.write_bytes(common.canonical_json_bytes(document))
        with self.assertRaises(replayer.FrozenCandidateReplayError) as raised:
            self._replay()
        self.assert_reason(raised, "request-must-be-direct-root-leaf")

    def test_schema_and_private_core_surface_remain_narrow(self) -> None:
        schema_path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-frozen-candidate-v1.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["schema_version"]["const"],
            frozen.MANIFEST_VERSION,
        )
        self.assertEqual(schema["properties"]["checks"]["minItems"], len(frozen.CHECK_NAMES))
        self.assertFalse(schema["properties"]["checks"]["items"])
        self.assertNotIn("passed", json.dumps(schema, sort_keys=True))
        self.assertNotIn("freeze_sha256", json.dumps(schema, sort_keys=True))
        self.assertIn(
            "writer_normal_return",
            schema["$defs"]["notEstablished"]["required"],
        )

        core_source = inspect.getsource(replayer._replay_rc3_frozen_candidate_v1_on_held_fds)
        for forbidden in (
            "fcntl.",
            "os.open",
            "os.close",
            "os.mkdir",
            "write_create",
            "socket",
            "docker",
            "nvidia-smi",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, core_source)
        core_doc = inspect.getdoc(replayer._replay_rc3_frozen_candidate_v1_on_held_fds) or ""
        self.assertIn("trusted", core_doc)
        self.assertIn("Git", core_doc)
        self.assertNotIn("request_name", inspect.signature(
            replayer._replay_rc3_frozen_candidate_v1_on_held_fds
        ).parameters)


if __name__ == "__main__":
    unittest.main()
