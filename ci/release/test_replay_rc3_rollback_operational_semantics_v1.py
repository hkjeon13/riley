#!/usr/bin/env python3
"""CPU-only tests for the private RC3 rollback operational semantics core."""

from __future__ import annotations

import ast
from dataclasses import replace
import fcntl
import hashlib
import inspect
import json
import os
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import capture_rc3_rollback_atomic_switch_v1 as atomic
import check_c02_provenance_v2 as c02
import compose_rc3_rollback_finalizer_receipt_v1 as composer
import finalize_rc3_rollback_candidate_source_v4 as fixed_finalizer
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common
import replay_rc3_rollback_candidate_source_v1 as candidate_source
import replay_rc3_rollback_operational_semantics_v1 as semantics
import test_write_rc3_rollback_candidate_source_bind_request_v1 as writer_fixtures
import write_rc3_rollback_finalizer_receipt_v1 as receipt
from test_compose_rc3_rollback_finalizer_receipt_v1 import RollbackFinalizerReceiptFixture


class ReplayRollbackOperationalSemanticsTests(
    unittest.TestCase,
    RollbackFinalizerReceiptFixture,
):
    def setUp(self) -> None:
        if self._testMethodName == "test_rejects_shared_candidate_rollback_listener_port":
            with mock.patch.object(
                writer_fixtures.WriteRollbackCandidateSourceBindRequestTests,
                "rollback_port",
                18080,
            ):
                self._set_up_rollback_receipt_fixture()
        else:
            self._set_up_rollback_receipt_fixture()

    def tearDown(self) -> None:
        self._tear_down_rollback_receipt_fixture()

    def _compose(self) -> dict[str, Any]:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback operational semantics compositor test root",
        )
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with mock.patch.object(
                atomic,
                "_rename_exchange",
                side_effect=writer_fixtures._fake_exchange,
            ):
                return composer._prepare_transaction_and_write_fixed_receipt_on_held_root_fd(  # noqa: SLF001
                    root_fd,
                    self.request,
                )
        finally:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def _with_held_fds(self, call: Callable[[int, int], dict[str, Any]]) -> dict[str, Any]:
        root_fd = common.open_private_evidence_directory(
            self.root,
            "rollback operational semantics test root",
        )
        switch_fd: int | None = None
        try:
            fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            switch_fd = common.open_private_child_directory(
                root_fd,
                prepare.SWITCH_DIRECTORY_NAME,
                "rollback operational semantics test switch",
            )
            fcntl.flock(switch_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return call(root_fd, switch_fd)
        finally:
            if switch_fd is not None:
                fcntl.flock(switch_fd, fcntl.LOCK_UN)
                os.close(switch_fd)
            fcntl.flock(root_fd, fcntl.LOCK_UN)
            os.close(root_fd)

    def _replay(self) -> dict[str, Any]:
        return self._with_held_fds(
            semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds  # noqa: SLF001
        )

    def _rewrite_shutdown(self, mutate: Callable[[dict[str, Any]], None]) -> None:
        artifact_path = self.root / candidate_source.SHUTDOWN_ARTIFACT_PATH
        document = json.loads(artifact_path.read_text(encoding="utf-8"))
        mutate(document)
        raw = common.canonical_json_bytes(document)
        artifact_path.write_bytes(raw)
        marker = {
            "schema_version": c02.SHUTDOWN_MARKER_VERSION,
            "artifact_filename": "shutdown.json",
            "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        }
        (self.root / candidate_source.SHUTDOWN_MARKER_PATH).write_bytes(
            common.canonical_json_bytes(marker)
        )

    def _rewrite_rollback_response(self, response: dict[str, Any]) -> None:
        phase_root = self.root / "rollback-phase"
        body_path = phase_root / "raw" / "generation-response-body.bin"
        head_path = phase_root / "raw" / "generation-response-head.http"
        session_path = phase_root / "session.json"
        body = common.canonical_json_bytes(response)
        head = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        )
        body_path.write_bytes(body)
        head_path.write_bytes(head)
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["generation"]["response_body"] = common.descriptor_for_bytes(
            "rollback-phase/raw/generation-response-body.bin",
            body,
            "rollback generation response body",
        ).as_json()
        session["generation"]["response_head"] = common.descriptor_for_bytes(
            "rollback-phase/raw/generation-response-head.http",
            head,
            "rollback generation response head",
        ).as_json()
        session_path.write_bytes(common.canonical_json_bytes(session))

    def _rewrite_rollback_gpu(self, gpu_uuid: str) -> None:
        phase_root = self.root / "rollback-phase"
        selection_path = phase_root / "raw" / "gpu-selection.csv"
        session_path = phase_root / "session.json"
        raw = f"0, {gpu_uuid}\n".encode("ascii")
        selection_path.write_bytes(raw)
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session["target"]["gpu_uuid"] = gpu_uuid
        session["process_evidence"]["gpu_selection"] = common.descriptor_for_bytes(
            "rollback-phase/raw/gpu-selection.csv",
            raw,
            "rollback GPU selection",
        ).as_json()
        session_path.write_bytes(common.canonical_json_bytes(session))

    def _remove_nonsemantic_terminal_pairs(self) -> None:
        for name in (
            f"{fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME}.intent",
            f"{fixed_finalizer.ROLLBACK_V4_MANIFEST_NAME}.complete",
            receipt.RECEIPT_NAME,
            f"{receipt.RECEIPT_NAME}.intent",
            f"{receipt.RECEIPT_NAME}.complete",
        ):
            (self.root / name).unlink()

    def test_replays_narrow_operational_facts_without_publishing(self) -> None:
        self._compose()
        # Deliberately re-open test FDs after the compositor returns.  This
        # proves only the core's FD-native raw diagnostic; it is not evidence
        # of a finalizer normal-return capability.
        before = set(os.listdir(self.root))
        with (
            mock.patch.object(
                common,
                "write_create_only",
                side_effect=AssertionError("operational semantic replay must not write"),
            ),
            mock.patch.object(
                common,
                "write_create_only_json",
                side_effect=AssertionError("operational semantic replay must not write"),
            ),
            mock.patch.object(
                common,
                "publish_create_only_hardlink",
                side_effect=AssertionError("operational semantic replay must not publish"),
            ),
        ):
            document = self._replay()
        self.assertEqual(before, set(os.listdir(self.root)))
        self.assertEqual(document["schema_version"], semantics.SEMANTICS_VERSION)
        self.assertEqual(document["status"], "passed")
        self.assertEqual(document["qualification_status"], "not-run")
        self.assertEqual(document["authority"], semantics.SEMANTICS_AUTHORITY)
        self.assertEqual(document["candidate_id"], "riley-0.1.0-rc3")
        facts = document["derived_facts"]
        self.assertEqual(facts["candidate_source_response_id"], "cmpl-1")
        self.assertEqual(facts["rollback_generation_response_id"], "cmpl-rollback")
        self.assertTrue(facts["candidate_shutdown_drained"])
        self.assertTrue(facts["isolated_artifact_exchange"])
        self.assertEqual(facts["candidate_target"]["listener_port"], 18080)
        self.assertEqual(facts["rollback_target"]["listener_port"], 18081)
        self.assertEqual(
            facts["candidate_target"]["gpu_uuid"],
            facts["rollback_target"]["gpu_uuid"],
        )
        self.assertEqual(document["reason_codes"], [])
        self.assertEqual(len(document["checks"]), 9)
        self.assertTrue(all(item["passed"] for item in document["checks"]))
        self.assertEqual(
            [item["name"] for item in document["checks"]],
            list(semantics._CHECK_NAMES),  # noqa: SLF001
        )
        for forbidden in ("frozen_candidate", "gate_e", "promotion", "deployment"):
            self.assertNotIn(forbidden, document)
        self.assertEqual(
            common.parse_canonical_json(
                common.canonical_json_bytes(document),
                "operational semantic test diagnostic",
            ),
            document,
        )

    def test_raw_v4_manifest_replay_does_not_consume_terminal_pairs_or_receipt(self) -> None:
        self._compose()
        self._remove_nonsemantic_terminal_pairs()
        document = self._replay()
        self.assertEqual(document["status"], "passed")
        self.assertEqual(document["authority"], semantics.SEMANTICS_AUTHORITY)

    def test_rejects_shared_candidate_rollback_listener_port(self) -> None:
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(
            getattr(raised.exception, "reason_code", None),
            "candidate-rollback-listener-port-reused",
        )

    def test_rejects_privileged_listener_port_before_reporting_schema_facts(self) -> None:
        self._compose()

        def invoke(root_fd: int, switch_fd: int) -> dict[str, Any]:
            original = semantics.writer._replay_inputs(root_fd, switch_fd)  # noqa: SLF001
            forged_phase = replace(
                original.rollback_phase,
                target=replace(original.rollback_phase.target, listener_port=80),
            )
            forged_inputs = replace(original, rollback_phase=forged_phase)
            with mock.patch.object(
                semantics.writer,
                "_replay_inputs",
                return_value=forged_inputs,
            ):
                return semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                )

        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._with_held_fds(invoke)
        self.assertEqual(getattr(raised.exception, "reason_code", None), "invalid-listener-port")

    def test_rejects_non_drained_candidate_shutdown_after_full_raw_composition(self) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["final_metrics"]["request_states"]["active"] = 1

        self._rewrite_shutdown(mutate)
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "candidate-shutdown-not-drained")

    def test_rejects_nonzero_candidate_shutdown_allocation_after_full_raw_composition(self) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["final_metrics"]["allocation"]["device_live_bytes"] = 1

        self._rewrite_shutdown(mutate)
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "candidate-shutdown-not-drained")

    def test_rejects_accepting_candidate_shutdown_worker_after_full_raw_composition(self) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["final_metrics"]["quiescence"]["worker_accepting"] = True

        self._rewrite_shutdown(mutate)
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "candidate-shutdown-not-drained")

    def test_rejects_accepting_candidate_shutdown_scheduler_after_full_raw_composition(self) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["final_metrics"]["quiescence"]["scheduler_accepting"] = True

        self._rewrite_shutdown(mutate)
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "candidate-shutdown-not-drained")

    def test_rejects_rollback_generation_id_reused_from_candidate_source(self) -> None:
        self._rewrite_rollback_response({"id": "cmpl-1", "object": "text_completion"})
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "rollback-generation-id-reused")

    def test_rejects_typed_source_response_audit_identity_mismatch(self) -> None:
        self._compose()

        def invoke(root_fd: int, switch_fd: int) -> dict[str, Any]:
            original = semantics.writer._replay_inputs(root_fd, switch_fd)  # noqa: SLF001
            forged_scenario = replace(
                original.candidate_source.source_scenario,
                request_id="cmpl-different-audit-id",
            )
            forged_join = replace(
                original.candidate_source,
                source_scenario=forged_scenario,
            )
            forged_inputs = replace(original, candidate_source=forged_join)
            with mock.patch.object(
                semantics.writer,
                "_replay_inputs",
                return_value=forged_inputs,
            ):
                return semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds(  # noqa: SLF001
                    root_fd,
                    switch_fd,
                )

        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._with_held_fds(invoke)
        self.assertEqual(
            getattr(raised.exception, "reason_code", None),
            "candidate-source-response-audit-mismatch",
        )

    def test_rejects_rollback_generation_response_without_a_completion_id(self) -> None:
        self._rewrite_rollback_response({"id": "not-a-completion", "object": "text_completion"})
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "invalid-completion-response-id")

    def test_rejects_rollback_gpu_identity_drift_after_full_raw_composition(self) -> None:
        self._rewrite_rollback_gpu("GPU-87654321-abcd-efab-cdef-1234567890ab")
        self._compose()
        with self.assertRaises(semantics.RollbackOperationalSemanticsError) as raised:
            self._replay()
        self.assertEqual(getattr(raised.exception, "reason_code", None), "candidate-rollback-gpu-mismatch")

    def test_static_surface_is_private_held_fd_read_only(self) -> None:
        source = inspect.getsource(semantics)
        tree = ast.parse(source)
        public_functions = [
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        self.assertEqual(public_functions, [])
        signature = inspect.signature(
            semantics._replay_rc3_rollback_operational_semantics_on_held_root_switch_fds  # noqa: SLF001
        )
        self.assertEqual(list(signature.parameters), ["root_fd", "switch_fd"])
        for forbidden in (
            "import argparse",
            "def main(",
            "socket",
            "subprocess",
            "nvidia-smi",
            "docker",
            "podman",
            "ssh ",
            "import fcntl",
            "flock(",
            "write_create_only",
            "publish_create_only_hardlink",
            "write_rc3_rollback_finalizer_receipt_v1",
            "check_soak_v2_receipt",
            "check_rc3_rollback_structural_precheck",
            "verify_completed_rollback_provenance_v4_on_held_switch_fd",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("do not prove", source)
        self.assertIn("invocation lineage", source)

    def test_schema_reserves_only_raw_operational_semantic_authority(self) -> None:
        path = (
            Path(__file__).resolve().parents[2]
            / "benchmarks/release/candidates/rc3-rollback-operational-semantics-v1.schema.json"
        )
        schema = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["status"], {"const": "passed"})
        self.assertEqual(schema["properties"]["qualification_status"], {"const": "not-run"})
        self.assertEqual(
            schema["properties"]["authority"],
            {"const": semantics.SEMANTICS_AUTHORITY},
        )
        checks = schema["properties"]["checks"]
        self.assertFalse(checks["items"])
        self.assertEqual(
            [item["$ref"].rsplit("/", 1)[-1] for item in checks["prefixItems"]],
            [
                "heldFdRawTopologyReplayCheck",
                "candidateRollbackProcessIdentityDistinctCheck",
                "candidateRollbackGpuIdentityEqualCheck",
                "candidateRollbackListenerPortDistinctCheck",
                "candidateSourceResponseAuditIdentityCheck",
                "rollbackGenerationResponseIdDistinctCheck",
                "candidateShutdownDrainedCheck",
                "isolatedArtifactInodeExchangeCheck",
                "v4RawClosureCrossBindingCheck",
            ],
        )
        self.assertEqual(
            [
                schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]["allOf"][1]["properties"][
                    "name"
                ]["const"]
                for item in checks["prefixItems"]
            ],
            list(semantics._CHECK_NAMES),  # noqa: SLF001
        )
        self.assertEqual(schema["$defs"]["target"]["properties"]["listener_port"]["minimum"], 1024)
        description = schema["description"].lower()
        for denied_claim in ("deployment rollback", "frozen-candidate", "gate e", "qualification"):
            self.assertIn(denied_claim, description)


if __name__ == "__main__":
    unittest.main()
