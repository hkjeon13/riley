#!/usr/bin/env python3
"""CPU-only hostile-path tests for the Gate E guardian/lease v1 model.

The fixture is intentionally synthetic.  It must not create a cgroup or lock,
start a process, contact Docker, query a GPU, or write a release artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(RELEASE_DIRECTORY))

import rc3_gate_e_guardian_lease_contract_v1 as contract  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def identity(pid: int, label: str, *, uid: int = 0, gid: int = 0, starttime: int | None = None) -> dict[str, object]:
    return {
        "pid": pid,
        "starttime_ticks": pid * 100 if starttime is None else starttime,
        "pidfd_token": digest(f"pidfd:{label}"),
        "uid": uid,
        "gid": gid,
    }


def identity_from_model(value: contract.ProcessIdentity) -> dict[str, object]:
    return {
        "pid": value.pid,
        "starttime_ticks": value.starttime_ticks,
        "pidfd_token": value.pidfd_token,
        "uid": value.uid,
        "gid": value.gid,
    }


def root_regular(mode: str) -> dict[str, object]:
    return {
        "uid": 0,
        "gid": 0,
        "mode": mode,
        "regular": True,
        "single_link": True,
        "posix_acl": "absent",
    }


def cgroup(*, inode: int = 1002, token_label: str = "cgroup") -> dict[str, object]:
    return {
        "st_dev": 1001,
        "st_ino": inode,
        "held_fd_token": digest(f"held-fd:{token_label}"),
        "non_delegated": True,
    }


def valid_session() -> dict[str, object]:
    controller = identity(1, "controller", starttime=111)
    guardian = identity(101, "guardian")
    warden = identity(102, "warden")
    worker = identity(201, "worker", uid=65532, gid=65532)
    return {
        "schema_version": contract.SCHEMA_VERSION,
        "scope": contract.SCOPE,
        "authority": contract.AUTHORITY,
        "installation_status": contract.INSTALLATION_STATUS,
        "boot_id": digest("boot-id"),
        "generation": 7,
        "lease_id": digest("lease-id"),
        "nonce": digest("nonce"),
        "guardian_contract_sha256": digest("guardian-contract"),
        "anchor_manifest_sha256": digest("anchor-manifest"),
        "runtime_closure_sha256": digest("runtime-closure"),
        "anchor_manifest": {
            "fixed_path": contract.FUTURE_ANCHOR_MANIFEST_AUDIT_PATH,
            "sha256": digest("anchor-manifest"),
            "byte_length": 1700,
            "owner": root_regular("0644"),
            "filesystem_types": ["ext4", "xfs"],
            "source": "held-fd-no-path-reresolution",
            "path_reresolution": False,
            "authenticated_pre_python": True,
            "bootstrap_sha256": digest("bootstrap"),
            "bootstrap_byte_length": 2200,
            "bootstrap_held_fd_token": digest("bootstrap-held-fd"),
            "core_sha256": digest("core"),
            "core_byte_length": 3300,
            "core_held_fd_token": digest("core-held-fd"),
            "python_sha256": contract.PYTHON_SHA256,
            "python_held_fd_token": digest("python-held-fd"),
            "runtime_closure_sha256": digest("runtime-closure"),
        },
        "bootstrap": {
            "fixed_path": contract.FUTURE_BOOTSTRAP_AUDIT_PATH,
            "sha256": digest("bootstrap"),
            "byte_length": 2200,
            "owner": root_regular("0755"),
            "filesystem_types": ["ext4", "xfs"],
            "source": "sealed-memfd-derived-from-held-fd",
            "path_reresolution": False,
            "authenticated_pre_python": True,
            "held_fd_token": digest("bootstrap-held-fd"),
        },
        "core": {
            "fixed_path": contract.FUTURE_CORE_AUDIT_PATH,
            "sha256": digest("core"),
            "byte_length": 3300,
            "source": "sealed-memfd-derived-from-held-fd",
            "path_reresolution": False,
            "held_fd_token": digest("core-held-fd"),
        },
        "python": {
            "path": contract.PYTHON_PATH,
            "sha256": contract.PYTHON_SHA256,
            "exec_object": "same-verified-fd",
            "runtime_closure_sha256": digest("runtime-closure"),
            "secure_exec": True,
            "execveat": True,
            "held_fd_token": digest("python-held-fd"),
        },
        "guardian": {
            "identity": guardian,
            "role": "native-root-guardian",
            "native_static": True,
            "pre_python_verifier": True,
            "exec_handoff": "same-verified-fd",
        },
        "warden": {
            "identity": warden,
            "role": "native-lease-warden",
            "holds_private_lease": True,
            "controls_non_delegated_cgroup": True,
        },
        "controller": {
            "identity": controller,
            "role": "pid1-system-manager",
            "sole_admission": True,
            "survives_warden": True,
            "non_delegated_cgroup": True,
        },
        "worker": worker,
        "host_context": {
            "initial_uid_map": contract.FULL_INITIAL_ID_MAP,
            "initial_gid_map": contract.FULL_INITIAL_ID_MAP,
            "user_namespace_inode": 301,
            "mount_namespace_inode": 302,
            "cgroup_namespace_inode": 303,
        },
        "launch": {
            "argv": list(contract.FUTURE_BOOTSTRAP_ARGV),
            "environment": {},
            "bootstrap_inherited_fds": list(contract.FUTURE_BOOTSTRAP_EXEC_FDS),
            "worker_inherited_fds": list(contract.FUTURE_CORE_EXEC_FDS),
            "bootstrap_fd_number": contract.FUTURE_SEALED_BOOTSTRAP_FD,
            "bootstrap_fd_token": digest("bootstrap-held-fd"),
            "bootstrap_fd_is_sealed": True,
            "bootstrap_fd_carries_lease": False,
            "bootstrap_fd_carries_cgroup_control": False,
            "core_fd_number": contract.FUTURE_SEALED_CORE_FD,
            "core_fd_token": digest("core-held-fd"),
            "core_fd_is_sealed": True,
            "core_fd_consumed_before_worker": True,
            "core_fd_inherited_by_worker": False,
            "lease_fd_inherited": False,
            "cgroup_control_fd_inherited": False,
            "no_new_privs": True,
            "capabilities": [],
        },
    }


def preflight_event(session: dict[str, object], held_cgroup: dict[str, object]) -> dict[str, object]:
    return {
        "kind": "NATIVE_PREFLIGHT_OK",
        "guardian": copy.deepcopy(session["guardian"]["identity"]),
        "warden": copy.deepcopy(session["warden"]["identity"]),
        "controller": copy.deepcopy(session["controller"]["identity"]),
        "cgroup": copy.deepcopy(held_cgroup),
        "populated": False,
    }


class GuardianLeaseContractTests(unittest.TestCase):
    maxDiff = None

    def enter_leased(self) -> tuple[contract.ControllerState, dict[str, object], dict[str, object], contract.Session]:
        document = valid_session()
        parsed = contract.parse_session_contract(document)
        state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
        self.assertEqual(state.phase, contract.PHASE_PREFLIGHT)
        held_cgroup = cgroup()
        state = contract.transition(state, preflight_event(document, held_cgroup))
        self.assertEqual(state.phase, contract.PHASE_LEASED_EMPTY)
        return state, document, held_cgroup, parsed

    def enter_live(self) -> tuple[contract.ControllerState, dict[str, object], dict[str, object], contract.Session]:
        state, document, held_cgroup, parsed = self.enter_leased()
        state = contract.transition(
            state,
            {
                "kind": "BOOTSTRAP_EXECED",
                "worker": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
            },
        )
        self.assertEqual(state.phase, contract.PHASE_BOOTSTRAP_STARTING)
        state = contract.transition(
            state,
            {
                "kind": "READY",
                "packet": contract.control_packet(parsed, "ready"),
                "credentials": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
                "ancillary_fds": [],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_NO_ACTION_LIVE)
        return state, document, held_cgroup, parsed

    def complete_to_draining(
        self,
    ) -> tuple[contract.ControllerState, dict[str, object], dict[str, object], contract.Session]:
        state, document, held_cgroup, parsed = self.enter_live()
        state = contract.transition(
            state,
            {
                "kind": "NO_ACTION_COMPLETE",
                "packet": contract.control_packet(parsed, "no_action_complete"),
                "credentials": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
                "ancillary_fds": [],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        return state, document, held_cgroup, parsed

    def empty_event(
        self,
        state: contract.ControllerState,
        held_cgroup: dict[str, object],
        parsed: contract.Session,
        *,
        controller_identity: dict[str, object] | None = None,
        terminal_tokens: object | None = None,
    ) -> dict[str, object]:
        return {
            "kind": "CGROUP_EMPTY",
            "controller": (
                identity_from_model(state.controller)
                if controller_identity is None and state.controller is not None
                else controller_identity
            ),
            "cgroup": copy.deepcopy(held_cgroup),
            "populated": False,
            "terminal_pidfd_tokens": [parsed.worker.pidfd_token] if terminal_tokens is None else terminal_tokens,
        }

    def test_canonical_input_and_full_no_action_lifecycle(self) -> None:
        document = valid_session()
        raw = contract.canonical_json_bytes(document) + b"\n"
        parsed_document = contract.parse_canonical_session_bytes(raw)
        parsed = contract.parse_session_contract(parsed_document)
        self.assertEqual(parsed.boot_id, document["boot_id"])
        with self.assertRaisesRegex(contract.GuardianLeaseContractError, "terminal newline"):
            contract.parse_canonical_session_bytes(raw + b" ")
        with self.assertRaisesRegex(contract.GuardianLeaseContractError, "duplicate key"):
            contract.parse_canonical_session_bytes(b'{"x":1,"x":2}\n')

        state, document, held_cgroup, parsed = self.complete_to_draining()
        state = contract.transition(state, self.empty_event(state, held_cgroup, parsed))
        self.assertEqual(state.phase, contract.PHASE_EMPTY_VERIFIED)
        state = contract.transition(
            state,
            {"kind": "CONTROLLER_RELEASE", "controller": identity_from_model(state.controller)},
        )
        self.assertEqual(state.phase, contract.PHASE_IDLE)
        self.assertEqual(state.boot_id, document["boot_id"])
        self.assertEqual(state.highest_generation, document["generation"])
        record = contract.status_record(state)
        self.assertFalse(record["admission_closed"])
        self.assertTrue(record["new_admission_allowed"])
        self.assertTrue(all(value is False for value in record["guarantees"].values()))

    def test_bad_pre_python_input_never_opens_a_lease(self) -> None:
        tamper_cases: list[tuple[str, object]] = [
            ("bootstrap.path_reresolution", True),
            ("bootstrap.filesystem_types", ["overlay"]),
            ("python.secure_exec", False),
            ("launch.environment", {"LD_PRELOAD": "/tmp/evil.so"}),
            ("launch.worker_inherited_fds", [0, 1, 2, 7]),
        ]
        for dotted_field, bad_value in tamper_cases:
            with self.subTest(dotted_field=dotted_field):
                document = valid_session()
                parent, child = dotted_field.split(".")
                document[parent][child] = bad_value
                state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
                self.assertEqual(state.phase, contract.PHASE_IDLE)
                self.assertFalse(state.admission_closed)
                self.assertIsNotNone(state.last_rejection)

        document = valid_session()
        document["bootstrap"]["filesystem_types"] = ["ext4", 7]
        state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
        self.assertEqual(state.phase, contract.PHASE_IDLE)
        self.assertIn("filesystem", state.last_rejection or "")

    def test_preflight_failure_is_explicitly_pre_acquisition_only(self) -> None:
        document = valid_session()
        state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
        self.assertEqual(state.phase, contract.PHASE_PREFLIGHT)
        self.assertIsNone(state.cgroup)
        state = contract.transition(state, {"kind": "PREFLIGHT_FAIL"})
        self.assertEqual(state.phase, contract.PHASE_IDLE)
        self.assertIsNone(state.cgroup)
        self.assertEqual(state.boot_id, document["boot_id"])
        self.assertEqual(state.highest_generation, document["generation"])

    def test_manifest_object_edges_and_sealed_fd_handoffs_are_exact(self) -> None:
        tamper_cases: list[tuple[str, str, object]] = [
            ("anchor_manifest", "bootstrap_sha256", digest("other-bootstrap")),
            ("anchor_manifest", "core_byte_length", 3400),
            ("anchor_manifest", "python_held_fd_token", digest("other-python-fd")),
            ("bootstrap", "held_fd_token", digest("other-bootstrap-fd")),
            ("core", "held_fd_token", digest("other-core-fd")),
            ("launch", "bootstrap_fd_token", digest("other-launch-bootstrap-fd")),
            ("launch", "core_fd_token", digest("other-launch-core-fd")),
        ]
        for parent, field, bad_value in tamper_cases:
            with self.subTest(parent=parent, field=field):
                document = valid_session()
                document[parent][field] = bad_value
                state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
                self.assertEqual(state.phase, contract.PHASE_IDLE)
                self.assertFalse(state.admission_closed)
                self.assertIsNotNone(state.last_rejection)

    def test_warden_or_guardian_loss_never_releases_before_empty_cgroup(self) -> None:
        for phase_name, setup in (
            ("leased", self.enter_leased),
            ("bootstrap", lambda: self._bootstrap_starting()),
            ("live", self.enter_live),
        ):
            for loss in ("GUARDIAN_SIGKILL", "WARDEN_SIGKILL"):
                with self.subTest(phase=phase_name, loss=loss):
                    state, document, held_cgroup, parsed = setup()
                    state = contract.transition(state, {"kind": loss})
                    self.assertEqual(state.phase, contract.PHASE_DRAINING)
                    self.assertTrue(state.admission_closed)
                    state = contract.transition(state, {"kind": "START", "session": document})
                    self.assertEqual(state.phase, contract.PHASE_DRAINING)
                    state = contract.transition(
                        state,
                        {
                            "kind": "CGROUP_POPULATED",
                            "controller": identity_from_model(state.controller),
                            "cgroup": held_cgroup,
                        },
                    )
                    self.assertEqual(state.phase, contract.PHASE_DRAINING)
                    self.assertTrue(state.cgroup_populated)
                    state = contract.transition(state, self.empty_event(state, held_cgroup, parsed))
                    self.assertEqual(state.phase, contract.PHASE_EMPTY_VERIFIED)

    def _bootstrap_starting(self) -> tuple[contract.ControllerState, dict[str, object], dict[str, object], contract.Session]:
        state, document, held_cgroup, parsed = self.enter_leased()
        state = contract.transition(
            state,
            {
                "kind": "BOOTSTRAP_EXECED",
                "worker": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
            },
        )
        self.assertEqual(state.phase, contract.PHASE_BOOTSTRAP_STARTING)
        return state, document, held_cgroup, parsed

    def test_control_channel_replay_credential_and_fd_violations_drain(self) -> None:
        state, document, held_cgroup, parsed = self.enter_live()
        stale = bytearray(contract.control_packet(parsed, "no_action_complete"))
        stale[-2] = ord(" ")
        state = contract.transition(
            state,
            {
                "kind": "NO_ACTION_COMPLETE",
                "packet": bytes(stale),
                "credentials": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
                "ancillary_fds": [],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)

        for mutate in ("credential", "cgroup", "fd", "missing-field"):
            with self.subTest(mutate=mutate):
                state, document, held_cgroup, parsed = self.enter_live()
                event: dict[str, object] = {
                    "kind": "NO_ACTION_COMPLETE",
                    "packet": contract.control_packet(parsed, "no_action_complete"),
                    "credentials": copy.deepcopy(document["worker"]),
                    "cgroup": copy.deepcopy(held_cgroup),
                    "ancillary_fds": [],
                }
                if mutate == "credential":
                    event["credentials"]["starttime_ticks"] += 1
                elif mutate == "cgroup":
                    event["cgroup"]["st_ino"] += 1
                elif mutate == "fd":
                    event["ancillary_fds"] = [99]
                else:
                    del event["credentials"]
                state = contract.transition(state, event)
                self.assertEqual(state.phase, contract.PHASE_DRAINING)
                self.assertTrue(state.admission_closed)

    def test_empty_observation_requires_exact_cgroup_and_worker_pidfd(self) -> None:
        state, document, held_cgroup, parsed = self.complete_to_draining()
        wrong_cgroup = cgroup(inode=9999)
        state = contract.transition(state, self.empty_event(state, wrong_cgroup, parsed))
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertTrue(state.admission_closed)

        state, document, held_cgroup, parsed = self.complete_to_draining()
        state = contract.transition(
            state,
            self.empty_event(state, held_cgroup, parsed, terminal_tokens=[digest("reused-pidfd")]),
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertTrue(state.admission_closed)

        state = contract.transition(state, self.empty_event(state, held_cgroup, parsed))
        self.assertEqual(state.phase, contract.PHASE_EMPTY_VERIFIED)
        state = contract.transition(state, {"kind": "CONTROLLER_RELEASE", "controller": identity_from_model(state.controller)})
        self.assertEqual(state.phase, contract.PHASE_IDLE)

    def test_controller_restart_can_only_recover_to_empty_verified(self) -> None:
        state, document, held_cgroup, parsed = self.enter_live()
        restarted_controller = identity(1, "controller-restarted", starttime=222)
        state = contract.transition(
            state,
            {
                "kind": "CONTROLLER_RESTART",
                "new_controller": restarted_controller,
                "cgroup": copy.deepcopy(held_cgroup),
                "populated": True,
                "terminal_pidfd_tokens": [parsed.worker.pidfd_token],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertEqual(identity_from_model(state.controller), restarted_controller)
        state = contract.transition(state, self.empty_event(state, held_cgroup, parsed))
        self.assertEqual(state.phase, contract.PHASE_EMPTY_VERIFIED)
        self.assertTrue(state.admission_closed)
        state = contract.transition(
            state,
            {"kind": "CONTROLLER_RELEASE", "controller": restarted_controller},
        )
        self.assertEqual(state.phase, contract.PHASE_IDLE)

    def test_durable_ledger_rehydrates_active_lease_and_fences_replay(self) -> None:
        document = valid_session()
        held_cgroup = cgroup()
        restarted_controller = identity(1, "controller-rehydrated", starttime=333)
        ledger = {
            "schema_version": contract.DURABLE_LEDGER_SCHEMA_VERSION,
            "scope": contract.SCOPE,
            "authority": contract.AUTHORITY,
            "installation_status": contract.INSTALLATION_STATUS,
            "storage": "root-controller-authenticated-durable-record",
            "boot_id": document["boot_id"],
            "highest_generation": document["generation"],
            "controller": restarted_controller,
            "active_session": copy.deepcopy(document),
            "held_cgroup": copy.deepcopy(held_cgroup),
            "last_observed_populated": False,
        }
        parsed = contract.parse_session_contract(document)
        state = contract.rehydrate_controller_from_durable_ledger(ledger)
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertTrue(state.admission_closed)
        self.assertEqual(identity_from_model(state.controller), restarted_controller)

        state = contract.transition(
            state,
            {"kind": "CONTROLLER_RELEASE", "controller": restarted_controller},
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)

        state = contract.transition(state, {"kind": "START", "session": document})
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        state = contract.transition(state, self.empty_event(state, held_cgroup, parsed))
        self.assertEqual(state.phase, contract.PHASE_EMPTY_VERIFIED)
        state = contract.transition(
            state,
            {"kind": "CONTROLLER_RELEASE", "controller": restarted_controller},
        )
        self.assertEqual(state.phase, contract.PHASE_IDLE)
        self.assertEqual(state.highest_generation, 7)

        state = contract.transition(state, {"kind": "START", "session": document})
        self.assertEqual(state.phase, contract.PHASE_IDLE)
        self.assertIn("fenced generation", state.last_rejection or "")
        next_document = valid_session()
        next_document["generation"] = 8
        next_document["lease_id"] = digest("lease-id-8")
        next_document["nonce"] = digest("nonce-8")
        state = contract.transition(state, {"kind": "START", "session": next_document})
        self.assertEqual(state.phase, contract.PHASE_PREFLIGHT)

        empty_ledger = copy.deepcopy(ledger)
        empty_ledger["active_session"] = None
        with self.assertRaisesRegex(contract.GuardianLeaseContractError, "empty durable ledger"):
            contract.rehydrate_controller_from_durable_ledger(empty_ledger)

    def test_malformed_control_event_drains_instead_of_raising(self) -> None:
        state, _, _, _ = self.enter_live()
        state = contract.transition(state, {"kind": "NO_ACTION_COMPLETE"})
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertTrue(state.admission_closed)

    def test_deep_json_is_rejected_or_drained_without_recursion_escape(self) -> None:
        nested = b'{"x":' + (b"[" * 1000) + b"0" + (b"]" * 1000) + b"}"
        with self.assertRaises(contract.GuardianLeaseContractError):
            contract.parse_session_contract(contract.parse_canonical_session_bytes(nested + b"\n"))

        state, document, held_cgroup, _ = self.enter_live()
        state = contract.transition(
            state,
            {
                "kind": "NO_ACTION_COMPLETE",
                "packet": nested,
                "credentials": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
                "ancillary_fds": [],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertIn("control-violation", state.drain_reason or "")

    def test_boolean_generation_cannot_match_generation_one(self) -> None:
        document = valid_session()
        document["generation"] = 1
        parsed = contract.parse_session_contract(document)
        state = contract.transition(contract.initial_state(), {"kind": "START", "session": document})
        held_cgroup = cgroup()
        state = contract.transition(state, preflight_event(document, held_cgroup))
        state = contract.transition(
            state,
            {
                "kind": "BOOTSTRAP_EXECED",
                "worker": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
            },
        )
        packet = json.loads(contract.control_packet(parsed, "ready").decode("utf-8"))
        packet["generation"] = True
        state = contract.transition(
            state,
            {
                "kind": "READY",
                "packet": contract.canonical_json_bytes(packet),
                "credentials": copy.deepcopy(document["worker"]),
                "cgroup": copy.deepcopy(held_cgroup),
                "ancillary_fds": [],
            },
        )
        self.assertEqual(state.phase, contract.PHASE_DRAINING)
        self.assertIn("control-violation", state.drain_reason or "")

    def test_action_like_events_are_structurally_rejected(self) -> None:
        for event_name in sorted(contract.ACTION_LIKE_EVENTS):
            with self.subTest(event_name=event_name):
                state, _, _, _ = self.enter_live()
                state = contract.transition(state, {"kind": event_name})
                self.assertEqual(state.phase, contract.PHASE_DRAINING)
                self.assertTrue(state.admission_closed)
                self.assertIn(event_name.lower(), state.drain_reason or "")

    def test_module_remains_a_pure_cpu_only_contract(self) -> None:
        source = (RELEASE_DIRECTORY / "rc3_gate_e_guardian_lease_contract_v1.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "import os",
            "import subprocess",
            "import socket",
            "import fcntl",
            "import signal",
            "O_CREAT",
            "nvidia-smi",
            "docker ",
        ):
            self.assertNotIn(forbidden, source)
        self.assertIn("not-authoritative", source)
        self.assertIn("held cgroup identity", source)


if __name__ == "__main__":
    unittest.main()
