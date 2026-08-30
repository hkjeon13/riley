#!/usr/bin/env python3
"""CPU-only hostile-path tests for the RC3 Gate E v3 private core template."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CORE = REPOSITORY_ROOT / "ci/release/rc3_gate_e_private_raw_core_v1.py"
PYTHON = "/usr/bin/python3.10"


# The isolated driver is intentionally independent from the core.  It copies
# the checkout template into a sealed anonymous descriptor, forks a test-only
# child with exactly FDs 0,1,2,8,9,10, and then execs the pinned interpreter.
# No /opt path, GPU lock, Docker command, or evidence root is involved.
ISOLATED_DRIVER = r'''
import fcntl
import hashlib
import json
import os
import signal
import socket
import struct
import sys

PYTHON = "/usr/bin/python3.10"
CORE_FD = 8
CONFIG_FD = 9
CONTROL_FD = 10
REQUIRED_SEALS = (
    fcntl.F_SEAL_SEAL
    | fcntl.F_SEAL_SHRINK
    | fcntl.F_SEAL_GROW
    | fcntl.F_SEAL_WRITE
)
CONFIG_SCHEMA = "riley.rc3-gate-e-sealed-no-action-core-config.v1"
PROTOCOL_SCHEMA = "riley.rc3-gate-e-sealed-no-action-protocol.v1"
SCOPE = "sealed-no-action-core"
AUTHORITY = "sealed-no-action-protocol-only"
NONCE = "1" * 64


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def starttime(pid):
    with open("/proc/%d/stat" % pid, "r", encoding="ascii") as handle:
        raw = handle.read()
    fields = raw[raw.rfind(") ") + 2:].split()
    return int(fields[19])


def sealed_memfd(name, payload, sealed=True):
    descriptor = os.memfd_create(name, os.MFD_ALLOW_SEALING)
    offset = 0
    while offset < len(payload):
        offset += os.write(descriptor, payload[offset:])
    os.lseek(descriptor, 0, os.SEEK_SET)
    if sealed:
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
    return descriptor


def close_unneeded(allowed):
    scanner = os.open("/proc/self/fd", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        names = os.listdir(scanner)
    finally:
        os.close(scanner)
    for name in names:
        descriptor = int(name)
        if descriptor == scanner or descriptor in allowed:
            continue
        try:
            os.close(descriptor)
        except OSError:
            pass


def receive_packet(control):
    payload, ancillary, flags, address = control.recvmsg(
        4096,
        socket.CMSG_SPACE(struct.calcsize("3i")),
    )
    credential = None
    if len(ancillary) == 1:
        level, kind, data = ancillary[0]
        if (
            level == socket.SOL_SOCKET
            and kind == socket.SCM_CREDENTIALS
            and len(data) == struct.calcsize("3i")
        ):
            credential = list(struct.unpack("3i", data))
    if type(address) is bytes:
        displayed_address = {"bytes_hex": address.hex()}
    else:
        displayed_address = address
    return {
        "packet": json.loads(payload.decode("utf-8")),
        "credential": credential,
        "flags": flags,
        "address": displayed_address,
    }


def child_exit_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    return -999


def main():
    core_path, mode = sys.argv[1:3]
    with open(core_path, "rb") as handle:
        core_bytes = handle.read()
    core_sha256 = hashlib.sha256(core_bytes).hexdigest()
    parent_pid = os.getpid()
    config = {
        "schema_version": CONFIG_SCHEMA,
        "scope": SCOPE,
        "authority": AUTHORITY,
        "nonce": NONCE,
        "parent_pid": parent_pid,
        "parent_starttime_ticks": starttime(parent_pid),
        "parent_uid": os.getuid(),
        "parent_gid": os.getgid(),
        "parent_executable": PYTHON,
        "core_fd": CORE_FD,
        "core_sha256": core_sha256,
        "core_byte_length": len(core_bytes),
        "config_fd": CONFIG_FD,
        "control_fd": CONTROL_FD,
    }
    if mode == "digest_mismatch":
        config["core_sha256"] = "2" * 64
    elif mode == "wrong_parent_pid":
        config["parent_pid"] = parent_pid + 1
    elif mode == "wrong_parent_starttime":
        config["parent_starttime_ticks"] += 1
    config_bytes = canonical(config) + b"\n"
    if mode == "noncanonical_config":
        config_bytes += b" "
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    core_fd = sealed_memfd("rc3-gate-e-test-core", core_bytes, sealed=mode != "unsealed_core")
    config_fd = sealed_memfd(
        "rc3-gate-e-test-config",
        config_bytes,
        sealed=mode != "unsealed_config",
    )
    parent_control, child_control = socket.socketpair(
        socket.AF_UNIX,
        socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC,
    )
    parent_control.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    parent_control.settimeout(3.0)
    error_read, error_write = os.pipe2(os.O_CLOEXEC)
    if mode == "sigterm_ignored":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == "sigterm_blocked":
        signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    child_pid = os.fork()
    if child_pid == 0:
        try:
            parent_control.close()
            os.close(error_read)
            os.dup2(error_write, 2, inheritable=True)
            os.dup2(core_fd, CORE_FD, inheritable=True)
            os.dup2(config_fd, CONFIG_FD, inheritable=True)
            os.dup2(child_control.fileno(), CONTROL_FD, inheritable=True)
            if mode == "fd7":
                os.dup2(core_fd, 7, inheritable=True)
            if mode == "extra_fd":
                os.dup2(config_fd, 11, inheritable=True)
            close_unneeded({0, 1, 2, CORE_FD, CONFIG_FD, CONTROL_FD, 7, 11})
            if mode not in {"fd7", "extra_fd"}:
                # Re-run the cleanup without the deliberate hostile FDs.
                close_unneeded({0, 1, 2, CORE_FD, CONFIG_FD, CONTROL_FD})
            interpreter_argv = [PYTHON, "-I", "-S", "-E", "-B"]
            if mode == "optimized_interpreter":
                interpreter_argv.append("-O")
            interpreter_argv.extend(
                ["/proc/self/fd/8", "--sealed-no-action-core"]
            )
            os.execve(
                PYTHON,
                interpreter_argv,
                {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            )
        except BaseException as error:
            os.write(2, ("driver child error: %r\\n" % (error,)).encode("utf-8"))
        os._exit(127)
    child_control.close()
    os.close(error_write)
    packets = []
    parent_error = None
    try:
        if mode in {"success", "sigterm_ignored", "sigterm_blocked"}:
            init = canonical(
                {
                    "schema_version": PROTOCOL_SCHEMA,
                    "kind": "init",
                    "nonce": NONCE,
                    "config_sha256": config_sha256,
                }
            )
            if parent_control.send(init) != len(init):
                raise RuntimeError("short INIT send")
            packets.append(receive_packet(parent_control))
            if mode in {"sigterm_ignored", "sigterm_blocked"}:
                os.kill(child_pid, signal.SIGTERM)
            else:
                run = canonical(
                    {
                        "schema_version": PROTOCOL_SCHEMA,
                        "kind": "run_no_action",
                        "nonce": NONCE,
                        "config_sha256": config_sha256,
                    }
                )
                if parent_control.send(run) != len(run):
                    raise RuntimeError("short RUN send")
                packets.append(receive_packet(parent_control))
        elif mode == "bad_init":
            bad = canonical(
                {
                    "schema_version": PROTOCOL_SCHEMA,
                    "kind": "init",
                    "nonce": "3" * 64,
                    "config_sha256": config_sha256,
                }
            )
            if parent_control.send(bad) != len(bad):
                raise RuntimeError("short bad INIT send")
        elif mode == "oversized_init":
            oversized = b"x" * 4097
            if parent_control.send(oversized) != len(oversized):
                raise RuntimeError("short oversized INIT send")
        elif mode == "scm_rights":
            init = canonical(
                {
                    "schema_version": PROTOCOL_SCHEMA,
                    "kind": "init",
                    "nonce": NONCE,
                    "config_sha256": config_sha256,
                }
            )
            sent = parent_control.sendmsg(
                [init],
                [
                    (
                        socket.SOL_SOCKET,
                        socket.SCM_RIGHTS,
                        struct.pack("i", config_fd),
                    )
                ],
            )
            if sent != len(init):
                raise RuntimeError("short SCM_RIGHTS INIT send")
    except BaseException as error:
        parent_error = repr(error)
    finally:
        parent_control.close()
    _unused, status = os.waitpid(child_pid, 0)
    error_chunks = []
    while True:
        block = os.read(error_read, 4096)
        if not block:
            break
        error_chunks.append(block)
    os.close(error_read)
    os.close(core_fd)
    os.close(config_fd)
    print(
        json.dumps(
            {
                "child_exit": child_exit_code(status),
                "child_pid": child_pid,
                "child_stderr": b"".join(error_chunks).decode("utf-8", "replace"),
                "config_sha256": config_sha256,
                "packets": packets,
                "parent_error": parent_error,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


main()
'''


class Rc3GateEPrivateRawCoreV1Tests(unittest.TestCase):
    def _run_driver(self, mode: str) -> dict[str, object]:
        completed = subprocess.run(
            [PYTHON, "-I", "-S", "-E", "-B", "-c", ISOLATED_DRIVER, str(CORE), mode],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return json.loads(completed.stdout)

    def test_checkout_path_is_not_a_callable_core(self) -> None:
        completed = subprocess.run(
            [PYTHON, "-I", "-S", "-E", "-B", str(CORE), "--sealed-no-action-core"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("sealed /proc/self/fd/8 descriptor", completed.stderr)

        invalid = subprocess.run(
            [PYTHON, "-I", "-S", "-E", "-B", str(CORE), "--help"],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TZ": "UTC"},
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("only --sealed-no-action-core", invalid.stderr)

    def test_sealed_memfd_protocol_is_nonce_and_credential_bound(self) -> None:
        result = self._run_driver("success")
        self.assertEqual(result["child_exit"], 0, result)
        self.assertEqual(result["child_stderr"], "", result)
        self.assertIsNone(result["parent_error"], result)
        packets = result["packets"]
        self.assertEqual(type(packets), list)
        self.assertEqual(len(packets), 2)
        child_pid = result["child_pid"]
        self.assertEqual(type(child_pid), int)
        for observed in packets:
            self.assertEqual(observed["flags"], 0)
            address = observed["address"]
            self.assertTrue(
                address in (None, "")
                or (
                    type(address) is dict
                    and set(address) == {"bytes_hex"}
                    and type(address["bytes_hex"]) is str
                    and len(address["bytes_hex"]) == 12
                    and address["bytes_hex"].startswith("00")
                ),
                observed,
            )
            self.assertEqual(
                observed["credential"],
                [child_pid, os.getuid(), os.getgid()],
            )
        self.assertEqual(
            packets[0]["packet"],
            {
                "schema_version": "riley.rc3-gate-e-sealed-no-action-protocol.v1",
                "kind": "ready",
                "nonce": "1" * 64,
                "core_sha256": CORE_SHA256(),
                "config_sha256": result["config_sha256"],
            },
        )
        self.assertEqual(
            packets[1]["packet"],
            {
                "schema_version": "riley.rc3-gate-e-sealed-no-action-protocol.v1",
                "kind": "complete",
                "nonce": "1" * 64,
                "core_sha256": CORE_SHA256(),
                "config_sha256": result["config_sha256"],
                "guarantees": {
                    "gpu_lock_acquired": False,
                    "gpu_queried": False,
                    "docker_invoked": False,
                    "evidence_created": False,
                    "semantic_replay_run": False,
                    "receipt_published": False,
                    "qualification_decided": False,
                },
            },
        )

    def test_invalid_sealed_contracts_fail_before_a_completion(self) -> None:
        expected = {
            "unsealed_core": "sealed core does not retain every required seal",
            "unsealed_config": "sealed config does not retain every required seal",
            "digest_mismatch": "sealed core digest differs",
            "wrong_parent_pid": "immediate parent PID differs",
            "wrong_parent_starttime": "parent process start time differs",
            "noncanonical_config": "sealed config must end in exactly one newline",
            "fd7": "inherited descriptor set must be exactly",
            "extra_fd": "inherited descriptor set must be exactly",
            "optimized_interpreter": "must run under",
            "bad_init": "does not match the nonce-bound protocol",
            "oversized_init": "parent control packet was truncated",
            "scm_rights": "parent control packet",
        }
        for mode, diagnostic in expected.items():
            with self.subTest(mode=mode):
                result = self._run_driver(mode)
                self.assertEqual(result["child_exit"], 2, result)
                self.assertEqual(result["packets"], [], result)
                self.assertIn(diagnostic, result["child_stderr"], result)

    def test_child_restores_usable_sigterm_before_waiting_for_the_parent(self) -> None:
        for mode in ("sigterm_ignored", "sigterm_blocked"):
            with self.subTest(mode=mode):
                result = self._run_driver(mode)
                self.assertEqual(result["child_exit"], -signal.SIGTERM, result)
                self.assertEqual(result["child_stderr"], "", result)
                self.assertIsNone(result["parent_error"], result)
                self.assertEqual(len(result["packets"]), 1, result)
                self.assertEqual(result["packets"][0]["packet"]["kind"], "ready", result)

    def test_static_contract_is_no_action_and_keeps_the_parent_lock_out(self) -> None:
        source = CORE.read_text(encoding="utf-8")
        for fragment in (
            'CORE_EXECUTION_PATH: Final = "/proc/self/fd/8"',
            "CORE_FD: Final = 8",
            "CONFIG_FD: Final = 9",
            "CONTROL_FD: Final = 10",
            "PARENT_LOCK_FD: Final = 7",
            "sys.dont_write_bytecode",
            "sys.flags.optimize == 0",
            'getattr(sys, "orig_argv", None)',
            "fcntl.F_GET_SEALS",
            "fcntl.F_SETFD",
            "fcntl.FD_CLOEXEC",
            "socket.SOCK_SEQPACKET",
            "socket.SO_PASSCRED",
            "socket.SCM_CREDENTIALS",
            "socket.SCM_RIGHTS",
            "socket.SO_PEERCRED",
            "PR_SET_PDEATHSIG",
            "signal.SIG_DFL",
            "signal.pthread_sigmask",
            "signal.SIG_UNBLOCK",
            "--sealed-no-action-core",
            "config_sha256",
            "gpu_lock_acquired",
            "sealed-no-action-protocol-only",
        ):
            self.assertIn(fragment, source)
        main_source = source[source.index("def main(arguments:") :]
        self.assertLess(
            main_source.index("_require_sealed_descriptor_invocation(arguments)"),
            main_source.index("_require_descriptor_allowlist()"),
        )
        for forbidden in (
            "fcntl.flock",
            "nvidia-smi",
            "/usr/bin/docker",
            "subprocess.",
            "os.fork",
            "os.exec",
            "write_rc3",
            "replay_rc3",
            "check_rc3_qualification",
        ):
            self.assertNotIn(forbidden, source)


def CORE_SHA256() -> str:
    import hashlib

    return hashlib.sha256(CORE.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
