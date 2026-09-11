#!/usr/bin/env python3
"""CPU-only state contract for the future RC3 Gate E guardian/lease boundary.

This is a pure model: it never installs a guardian, opens a lock, launches a
process, reads a cgroup, signals anything, writes evidence, or queries a GPU.
It fixes the prerequisite that a future native guardian and PID1/system-manager
admission controller must satisfy before a real producer is even considered.

The safety rule is stronger than a parent-held ``flock``: while the previous
session's *held cgroup identity* may be populated, new admission remains
closed.  Guardian or warden loss therefore cannot make lock availability a
release signal.  Only an authenticated empty observation of the same held
cgroup, followed by controller release, returns to IDLE.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Final, NoReturn


SCHEMA_VERSION: Final = "riley.rc3-gate-e-guardian-lease.v1"
CONTROL_SCHEMA_VERSION: Final = "riley.rc3-gate-e-guardian-control.v1"
DURABLE_LEDGER_SCHEMA_VERSION: Final = "riley.rc3-gate-e-durable-admission-ledger.v1"
SCOPE: Final = "guardian-lease-contract-only"
AUTHORITY: Final = "not-authoritative"
INSTALLATION_STATUS: Final = "not-installed"

PHASE_IDLE: Final = "IDLE"
PHASE_PREFLIGHT: Final = "PREFLIGHT"
PHASE_LEASED_EMPTY: Final = "LEASED_EMPTY"
PHASE_BOOTSTRAP_STARTING: Final = "BOOTSTRAP_STARTING"
PHASE_NO_ACTION_LIVE: Final = "NO_ACTION_LIVE"
PHASE_DRAINING: Final = "DRAINING"
PHASE_EMPTY_VERIFIED: Final = "EMPTY_VERIFIED"
ACTIVE_PHASES: Final = frozenset(
    {
        PHASE_PREFLIGHT,
        PHASE_LEASED_EMPTY,
        PHASE_BOOTSTRAP_STARTING,
        PHASE_NO_ACTION_LIVE,
        PHASE_DRAINING,
        PHASE_EMPTY_VERIFIED,
    }
)

PYTHON_PATH: Final = "/usr/bin/python3.10"
PYTHON_SHA256: Final = (
    "7d51cd6b48b521277f5caa4610a82126e315fa2be4df069823a8b1eeb5bd4a86"
)
ANCHOR_ROOT: Final = "/opt/riley/rc3-gate-e-v1"
# This is deliberately a future audit leaf, not the currently checked-in v3
# Python template.  The v3 template cannot safely be retrofitted by changing
# its argv: a native guardian must authenticate this future object before the
# dynamic loader and Python receive it.
FUTURE_BOOTSTRAP_AUDIT_PATH: Final = (
    f"{ANCHOR_ROOT}/rc3_gate_e_guardian_bootstrap_v1.py"
)
FUTURE_ANCHOR_MANIFEST_AUDIT_PATH: Final = f"{ANCHOR_ROOT}/gate-e-v3.manifest.json"
FUTURE_CORE_AUDIT_PATH: Final = f"{ANCHOR_ROOT}/rc3_gate_e_guardian_no_action_core_v1.py"
FUTURE_SEALED_BOOTSTRAP_FD: Final = 31
FUTURE_SEALED_CORE_FD: Final = 32
FUTURE_BOOTSTRAP_ARGV: Final = (
    PYTHON_PATH,
    "-I",
    "-S",
    "-E",
    "-B",
    f"/proc/self/fd/{FUTURE_SEALED_BOOTSTRAP_FD}",
    "--guardian-no-action-bootstrap",
)
FUTURE_BOOTSTRAP_EXEC_FDS: Final = (
    0,
    1,
    2,
    FUTURE_SEALED_BOOTSTRAP_FD,
    FUTURE_SEALED_CORE_FD,
)
FUTURE_CORE_EXEC_FDS: Final = (0, 1, 2)
FULL_INITIAL_ID_MAP: Final = "0 0 4294967295"
ALLOWED_FILESYSTEMS: Final = ("btrfs", "ext4", "xfs")
MAX_CONTRACT_BYTES: Final = 64 * 1024
MAX_CONTROL_BYTES: Final = 4 * 1024
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
U64_MAX: Final = (1 << 64) - 1
DRAIN_EVENTS: Final = frozenset(
    {
        "STOP",
        "TIMEOUT",
        "CONTROL_VIOLATION",
        "BOOTSTRAP_EXIT",
        "GUARDIAN_EXIT",
        "GUARDIAN_SIGKILL",
        "WARDEN_EXIT",
        "WARDEN_SIGKILL",
        "CONTROLLER_EXIT",
        "CONTROLLER_SIGKILL",
        "CGROUP_READ_ERROR",
        "PIDFD_ERROR",
    }
)
ACTION_LIKE_EVENTS: Final = frozenset(
    {
        "RUN_CAPTURE",
        "GPU_QUERY",
        "DOCKER",
        "WRITE_EVIDENCE",
        "WRITE_RECEIPT",
        "QUALIFY",
        "RELEASE_FROM_BOOTSTRAP",
    }
)


class GuardianLeaseContractError(ValueError):
    """A synthetic guardian/lease session violates the closed contract."""


def _fail(reason: str) -> NoReturn:
    raise GuardianLeaseContractError(reason)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        _fail(f"cannot encode canonical guardian lease JSON: {error}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            _fail("guardian lease JSON contains a duplicate key")
        result[key] = value
    return result


def parse_canonical_session_bytes(raw: bytes) -> dict[str, object]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_CONTRACT_BYTES:
        _fail("guardian lease session has an unsafe byte length")
    if not raw.endswith(b"\n") or raw[:-1].endswith(b"\n"):
        _fail("guardian lease session must have exactly one terminal newline")
    encoded = raw[:-1]
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        _fail(f"cannot parse guardian lease session: {error}")
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        _fail("guardian lease session is not canonical JSON")
    return value


def _mapping(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        _fail(f"{label} has an unexpected field set")
    return value


def _positive(value: object, label: str, maximum: int = U64_MAX) -> int:
    if type(value) is not int or value <= 0 or value > maximum:
        _fail(f"{label} must be a positive bounded integer")
    return value


def _nonnegative(value: object, label: str, maximum: int = (1 << 32) - 1) -> int:
    if type(value) is not int or value < 0 or value > maximum:
        _fail(f"{label} must be a nonnegative bounded integer")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be a nonzero lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    starttime_ticks: int
    pidfd_token: str
    uid: int
    gid: int


@dataclass(frozen=True)
class CgroupIdentity:
    st_dev: int
    st_ino: int
    held_fd_token: str


@dataclass(frozen=True)
class Session:
    boot_id: str
    generation: int
    lease_id: str
    nonce: str
    guardian_contract_sha256: str
    guardian: ProcessIdentity
    warden: ProcessIdentity
    controller: ProcessIdentity
    worker: ProcessIdentity


@dataclass(frozen=True)
class ControllerState:
    phase: str
    admission_closed: bool
    session: Session | None = None
    controller: ProcessIdentity | None = None
    cgroup: CgroupIdentity | None = None
    cgroup_populated: bool | None = None
    drain_reason: str | None = None
    last_rejection: str | None = None
    boot_id: str | None = None
    highest_generation: int = 0


def initial_state(
    *,
    last_rejection: str | None = None,
    boot_id: str | None = None,
    highest_generation: int = 0,
) -> ControllerState:
    """Build a synthetic cold-start fixture, never a post-crash recovery API.

    A future controller must call ``rehydrate_controller_from_durable_ledger``
    after any restart.  This pure model cannot stop an arbitrary caller from
    constructing values, so this helper is intentionally non-authoritative.
    """
    if boot_id is not None:
        _digest(boot_id, "controller fence boot_id")
    if type(highest_generation) is not int or highest_generation < 0 or highest_generation > U64_MAX:
        _fail("controller fence highest_generation must be a bounded nonnegative integer")
    if boot_id is None and highest_generation != 0:
        _fail("an unfenced IDLE state cannot claim a prior generation")
    return ControllerState(
        PHASE_IDLE,
        False,
        last_rejection=last_rejection,
        boot_id=boot_id,
        highest_generation=highest_generation,
    )


def _identity(value: object, label: str, *, worker: bool = False) -> ProcessIdentity:
    item = _mapping(value, {"pid", "starttime_ticks", "pidfd_token", "uid", "gid"}, label)
    result = ProcessIdentity(
        _positive(item["pid"], f"{label}.pid", (1 << 31) - 1),
        _positive(item["starttime_ticks"], f"{label}.starttime_ticks"),
        _digest(item["pidfd_token"], f"{label}.pidfd_token"),
        _nonnegative(item["uid"], f"{label}.uid"),
        _nonnegative(item["gid"], f"{label}.gid"),
    )
    if worker and (result.uid == 0 or result.gid == 0):
        _fail(f"{label} must be an unprivileged worker identity")
    return result


def _same_identity(value: object, expected: ProcessIdentity, label: str) -> None:
    if _identity(value, label) != expected:
        _fail(f"{label} differs from its registered PID/start-tick/pidfd identity")


def _root_identity(value: object, label: str, *, pid1: bool = False) -> ProcessIdentity:
    result = _identity(value, label)
    if result.uid != 0 or result.gid != 0:
        _fail(f"{label} must be a root service identity")
    if pid1 and result.pid != 1:
        _fail(f"{label} must be the PID 1 system-manager identity")
    return result


def _cgroup(value: object, label: str) -> CgroupIdentity:
    item = _mapping(value, {"st_dev", "st_ino", "held_fd_token", "non_delegated"}, label)
    if item["non_delegated"] is not True:
        _fail(f"{label} must be controller-owned and non-delegated")
    return CgroupIdentity(
        _positive(item["st_dev"], f"{label}.st_dev"),
        _positive(item["st_ino"], f"{label}.st_ino"),
        _digest(item["held_fd_token"], f"{label}.held_fd_token"),
    )


def _same_cgroup(value: object, expected: CgroupIdentity, label: str) -> None:
    if _cgroup(value, label) != expected:
        _fail(f"{label} differs from the held cgroup device/inode/token identity")


def _root_regular(value: object, label: str, mode: str) -> None:
    item = _mapping(value, {"uid", "gid", "mode", "regular", "single_link", "posix_acl"}, label)
    if item != {
        "uid": 0,
        "gid": 0,
        "mode": mode,
        "regular": True,
        "single_link": True,
        "posix_acl": "absent",
    }:
        _fail(f"{label} differs from the root-owned ACL-free regular-file policy")


def _filesystems(value: object, label: str) -> None:
    if type(value) is not list or not value:
        _fail(f"{label} must be a nonempty sorted unique filesystem list")
    if any(type(item) is not str or item not in ALLOWED_FILESYSTEMS for item in value):
        _fail(f"{label} includes a network, overlay, or unapproved filesystem")
    if value != sorted(value) or len(value) != len(set(value)):
        _fail(f"{label} must be a nonempty sorted unique filesystem list")


def _validate_session(value: object) -> Session:
    item = _mapping(
        value,
        {
            "schema_version", "scope", "authority", "installation_status", "boot_id", "generation", "lease_id",
            "nonce", "guardian_contract_sha256", "anchor_manifest_sha256", "runtime_closure_sha256",
            "anchor_manifest", "bootstrap", "core", "python", "guardian", "warden", "controller", "worker",
            "host_context", "launch",
        },
        "guardian lease session",
    )
    if (
        item["schema_version"] != SCHEMA_VERSION or item["scope"] != SCOPE
        or item["authority"] != AUTHORITY or item["installation_status"] != INSTALLATION_STATUS
    ):
        _fail("session claims unsupported scope, installation state, or authority")
    boot_id = _digest(item["boot_id"], "boot_id")
    generation = _positive(item["generation"], "generation")
    lease_id = _digest(item["lease_id"], "lease_id")
    nonce = _digest(item["nonce"], "nonce")
    guardian_contract_sha256 = _digest(item["guardian_contract_sha256"], "guardian_contract_sha256")
    anchor_manifest_sha256 = _digest(item["anchor_manifest_sha256"], "anchor_manifest_sha256")
    runtime_closure = _digest(item["runtime_closure_sha256"], "runtime_closure_sha256")
    manifest = _mapping(
        item["anchor_manifest"],
        {
            "fixed_path", "sha256", "byte_length", "owner", "filesystem_types", "source",
            "path_reresolution", "authenticated_pre_python", "bootstrap_sha256", "bootstrap_byte_length",
            "bootstrap_held_fd_token", "core_sha256", "core_byte_length", "core_held_fd_token",
            "python_sha256", "python_held_fd_token", "runtime_closure_sha256",
        },
        "anchor_manifest",
    )
    if manifest["fixed_path"] != FUTURE_ANCHOR_MANIFEST_AUDIT_PATH:
        _fail("anchor_manifest.fixed_path differs from the future anchor manifest leaf")
    if _digest(manifest["sha256"], "anchor_manifest.sha256") != anchor_manifest_sha256:
        _fail("anchor_manifest does not bind the supplied manifest digest")
    if _positive(manifest["byte_length"], "anchor_manifest.byte_length") > 2 * 1024 * 1024:
        _fail("anchor_manifest.byte_length exceeds the bounded handoff limit")
    _root_regular(manifest["owner"], "anchor_manifest.owner", "0644")
    _filesystems(manifest["filesystem_types"], "anchor_manifest.filesystem_types")
    if (
        manifest["source"] != "held-fd-no-path-reresolution"
        or manifest["path_reresolution"] is not False
        or manifest["authenticated_pre_python"] is not True
    ):
        _fail("anchor manifest must be pre-Python authenticated from its held object")
    manifest_bootstrap_sha256 = _digest(manifest["bootstrap_sha256"], "anchor_manifest.bootstrap_sha256")
    manifest_bootstrap_length = _positive(manifest["bootstrap_byte_length"], "anchor_manifest.bootstrap_byte_length")
    manifest_bootstrap_token = _digest(manifest["bootstrap_held_fd_token"], "anchor_manifest.bootstrap_held_fd_token")
    manifest_core_sha256 = _digest(manifest["core_sha256"], "anchor_manifest.core_sha256")
    manifest_core_length = _positive(manifest["core_byte_length"], "anchor_manifest.core_byte_length")
    manifest_core_token = _digest(manifest["core_held_fd_token"], "anchor_manifest.core_held_fd_token")
    manifest_python_sha256 = _digest(manifest["python_sha256"], "anchor_manifest.python_sha256")
    manifest_python_token = _digest(manifest["python_held_fd_token"], "anchor_manifest.python_held_fd_token")
    if _digest(manifest["runtime_closure_sha256"], "anchor_manifest.runtime_closure_sha256") != runtime_closure:
        _fail("anchor manifest does not bind the supplied runtime closure digest")
    bootstrap = _mapping(
        item["bootstrap"],
        {
            "fixed_path", "sha256", "byte_length", "owner", "filesystem_types", "source",
            "path_reresolution", "authenticated_pre_python", "held_fd_token",
        },
        "bootstrap",
    )
    if bootstrap["fixed_path"] != FUTURE_BOOTSTRAP_AUDIT_PATH:
        _fail("bootstrap.fixed_path differs from the future anchor leaf")
    bootstrap_sha256 = _digest(bootstrap["sha256"], "bootstrap.sha256")
    bootstrap_length = _positive(bootstrap["byte_length"], "bootstrap.byte_length")
    bootstrap_token = _digest(bootstrap["held_fd_token"], "bootstrap.held_fd_token")
    if bootstrap_length > 2 * 1024 * 1024:
        _fail("bootstrap.byte_length exceeds the bounded handoff limit")
    _root_regular(bootstrap["owner"], "bootstrap.owner", "0755")
    _filesystems(bootstrap["filesystem_types"], "bootstrap.filesystem_types")
    if (
        bootstrap["source"] != "sealed-memfd-derived-from-held-fd"
        or bootstrap["path_reresolution"] is not False
        or bootstrap["authenticated_pre_python"] is not True
    ):
        _fail("bootstrap must be pre-Python authenticated from the same verified object without path re-resolution")
    if (
        bootstrap_sha256 != manifest_bootstrap_sha256 or bootstrap_length != manifest_bootstrap_length
        or bootstrap_token != manifest_bootstrap_token
    ):
        _fail("anchor manifest does not bind the exact bootstrap held object")
    core = _mapping(
        item["core"],
        {"fixed_path", "sha256", "byte_length", "source", "path_reresolution", "held_fd_token"},
        "core",
    )
    if core["fixed_path"] != FUTURE_CORE_AUDIT_PATH:
        _fail("core.fixed_path differs from the future core audit leaf")
    core_sha256 = _digest(core["sha256"], "core.sha256")
    core_length = _positive(core["byte_length"], "core.byte_length")
    core_token = _digest(core["held_fd_token"], "core.held_fd_token")
    if core_length > 2 * 1024 * 1024:
        _fail("core.byte_length exceeds the bounded handoff limit")
    if core["source"] != "sealed-memfd-derived-from-held-fd" or core["path_reresolution"] is not False:
        _fail("core must be handed to the worker from the held verified object")
    if (
        core_sha256 != manifest_core_sha256 or core_length != manifest_core_length
        or core_token != manifest_core_token
    ):
        _fail("anchor manifest does not bind the exact core held object")
    python = _mapping(
        item["python"],
        {
            "path", "sha256", "exec_object", "runtime_closure_sha256", "secure_exec", "execveat",
            "held_fd_token",
        },
        "python",
    )
    if (
        python["path"] != PYTHON_PATH or python["sha256"] != PYTHON_SHA256
        or python["exec_object"] != "same-verified-fd" or python["runtime_closure_sha256"] != runtime_closure
        or python["secure_exec"] is not True or python["execveat"] is not True
        or _digest(python["held_fd_token"], "python.held_fd_token") != manifest_python_token
        or manifest_python_sha256 != PYTHON_SHA256
    ):
        _fail("interpreter must execute from the same verified object and runtime closure")
    guardian_item = _mapping(
        item["guardian"],
        {"identity", "role", "native_static", "pre_python_verifier", "exec_handoff"},
        "guardian",
    )
    if (
        guardian_item["role"] != "native-root-guardian"
        or guardian_item["native_static"] is not True
        or guardian_item["pre_python_verifier"] is not True
        or guardian_item["exec_handoff"] != "same-verified-fd"
    ):
        _fail("guardian must be a static native root pre-Python verifier")
    guardian = _root_identity(guardian_item["identity"], "guardian.identity")
    warden_item = _mapping(
        item["warden"],
        {"identity", "role", "holds_private_lease", "controls_non_delegated_cgroup"},
        "warden",
    )
    if (
        warden_item["role"] != "native-lease-warden"
        or warden_item["holds_private_lease"] is not True
        or warden_item["controls_non_delegated_cgroup"] is not True
    ):
        _fail("warden must be the native holder of the private lease/cgroup boundary")
    warden = _root_identity(warden_item["identity"], "warden.identity")
    controller = _mapping(item["controller"], {"identity", "role", "sole_admission", "survives_warden", "non_delegated_cgroup"}, "controller")
    if (
        controller["role"] != "pid1-system-manager" or controller["sole_admission"] is not True
        or controller["survives_warden"] is not True or controller["non_delegated_cgroup"] is not True
    ):
        _fail("controller must be the sole surviving PID1/system-manager admission authority")
    controller_identity = _root_identity(controller["identity"], "controller.identity", pid1=True)
    worker = _identity(item["worker"], "worker", worker=True)
    if len({guardian, warden, controller_identity, worker}) != 4:
        _fail("guardian, warden, controller, and worker require distinct stable identities")
    host = _mapping(
        item["host_context"],
        {"initial_uid_map", "initial_gid_map", "user_namespace_inode", "mount_namespace_inode", "cgroup_namespace_inode"},
        "host_context",
    )
    if host["initial_uid_map"] != FULL_INITIAL_ID_MAP or host["initial_gid_map"] != FULL_INITIAL_ID_MAP:
        _fail("host context must prove the full initial UID/GID mapping")
    for field in ("user_namespace_inode", "mount_namespace_inode", "cgroup_namespace_inode"):
        _positive(host[field], f"host_context.{field}")
    launch = _mapping(
        item["launch"],
        {
            "argv", "environment", "bootstrap_inherited_fds", "worker_inherited_fds", "bootstrap_fd_number",
            "bootstrap_fd_token", "bootstrap_fd_is_sealed", "bootstrap_fd_carries_lease", "bootstrap_fd_carries_cgroup_control",
            "core_fd_number", "core_fd_token", "core_fd_is_sealed", "core_fd_consumed_before_worker",
            "core_fd_inherited_by_worker",
            "lease_fd_inherited", "cgroup_control_fd_inherited", "no_new_privs", "capabilities",
        },
        "launch",
    )
    if launch["argv"] != list(FUTURE_BOOTSTRAP_ARGV) or type(launch["environment"]) is not dict or launch["environment"]:
        _fail("future sealed bootstrap launch must have fixed argv and empty execve environment")
    if (
        launch["bootstrap_inherited_fds"] != list(FUTURE_BOOTSTRAP_EXEC_FDS)
        or launch["worker_inherited_fds"] != list(FUTURE_CORE_EXEC_FDS)
        or launch["bootstrap_fd_number"] != FUTURE_SEALED_BOOTSTRAP_FD
        or _digest(launch["bootstrap_fd_token"], "launch.bootstrap_fd_token") != bootstrap_token
        or launch["bootstrap_fd_is_sealed"] is not True
        or launch["bootstrap_fd_carries_lease"] is not False
        or launch["bootstrap_fd_carries_cgroup_control"] is not False
        or launch["core_fd_number"] != FUTURE_SEALED_CORE_FD
        or _digest(launch["core_fd_token"], "launch.core_fd_token") != core_token
        or launch["core_fd_is_sealed"] is not True
        or launch["core_fd_consumed_before_worker"] is not True
        or launch["core_fd_inherited_by_worker"] is not False
        or launch["lease_fd_inherited"] is not False
        or launch["cgroup_control_fd_inherited"] is not False or launch["no_new_privs"] is not True
        or launch["capabilities"] != []
    ):
        _fail("worker must be descriptor-isolated, no-new-privileges, and capability-free")
    return Session(boot_id, generation, lease_id, nonce, guardian_contract_sha256, guardian, warden, controller_identity, worker)


def parse_session_contract(value: object) -> Session:
    """Validate the no-action pre-Python binding; this performs no OS action."""
    return _validate_session(value)


def rehydrate_controller_from_durable_ledger(value: object) -> ControllerState:
    """Model PID1 recovery from an authenticated durable admission record.

    This does not read a file or validate a real signature.  It deliberately
    returns DRAINING for every retained active lease, including a record whose
    last observation was empty: a fresh same-object empty observation plus an
    explicit controller release is still required before another START.
    """
    item = _mapping(
        value,
        {
            "schema_version", "scope", "authority", "installation_status", "storage", "boot_id",
            "highest_generation", "controller", "active_session", "held_cgroup", "last_observed_populated",
        },
        "durable admission ledger",
    )
    if (
        item["schema_version"] != DURABLE_LEDGER_SCHEMA_VERSION or item["scope"] != SCOPE
        or item["authority"] != AUTHORITY or item["installation_status"] != INSTALLATION_STATUS
        or item["storage"] != "root-controller-authenticated-durable-record"
    ):
        _fail("durable admission ledger claims unsupported authority or storage")
    boot_id = _digest(item["boot_id"], "durable admission ledger boot_id")
    highest_generation = _nonnegative(item["highest_generation"], "durable admission ledger highest_generation", U64_MAX)
    current_controller = _root_identity(item["controller"], "durable admission ledger controller", pid1=True)
    active_raw = item["active_session"]
    if active_raw is None:
        if item["held_cgroup"] is not None or item["last_observed_populated"] is not None:
            _fail("an empty durable ledger cannot retain a cgroup observation")
        return initial_state(boot_id=boot_id, highest_generation=highest_generation)
    session = _validate_session(active_raw)
    if session.boot_id != boot_id or session.generation > highest_generation:
        _fail("durable active lease does not fit the boot/generation fencing record")
    if current_controller in {session.guardian, session.warden, session.worker}:
        _fail("durable controller must remain distinct from guardian, warden, and worker")
    held_cgroup = _cgroup(item["held_cgroup"], "durable admission ledger held_cgroup")
    populated = item["last_observed_populated"]
    if populated is not None and populated is not True and populated is not False:
        _fail("durable admission ledger populated observation must be boolean or null")
    return ControllerState(
        PHASE_DRAINING,
        True,
        session,
        current_controller,
        held_cgroup,
        populated,
        "durable-ledger-rehydrated",
        "recovery-requires-fresh-empty-observation",
        boot_id,
        highest_generation,
    )


def _check_state(state: ControllerState) -> None:
    if state.phase not in ACTIVE_PHASES | {PHASE_IDLE}:
        _fail("controller state has an unknown phase")
    if type(state.highest_generation) is not int or state.highest_generation < 0 or state.highest_generation > U64_MAX:
        _fail("controller state has an unsafe fencing generation")
    if state.boot_id is not None:
        _digest(state.boot_id, "controller state boot_id")
    elif state.highest_generation != 0:
        _fail("controller state cannot retain a generation without a boot fence")
    if state.phase == PHASE_IDLE:
        if (
            state.admission_closed or state.session is not None or state.controller is not None
            or state.cgroup is not None or state.cgroup_populated is not None or state.drain_reason is not None
        ):
            _fail("IDLE must not retain an admission, session, controller, or held cgroup")
        return
    if not state.admission_closed or state.session is None or state.controller is None:
        _fail("every non-IDLE phase must keep admission closed with a session/controller")
    if state.boot_id != state.session.boot_id or state.highest_generation < state.session.generation:
        _fail("active state must retain the session boot/generation fencing record")
    if state.phase == PHASE_PREFLIGHT:
        if state.cgroup is not None or state.cgroup_populated is not None or state.drain_reason is not None:
            _fail("PREFLIGHT cannot retain a lease/cgroup or terminal reason")
        return
    if state.cgroup is None:
        _fail("every leased phase must retain the same held cgroup identity")
    if state.phase == PHASE_LEASED_EMPTY and state.cgroup_populated is not False:
        _fail("LEASED_EMPTY requires the freshly observed empty held cgroup")
    if state.phase == PHASE_BOOTSTRAP_STARTING and state.cgroup_populated is not False:
        _fail("BOOTSTRAP_STARTING cannot claim a live cgroup before READY")
    if state.phase == PHASE_NO_ACTION_LIVE and state.cgroup_populated is not True:
        _fail("NO_ACTION_LIVE requires the worker cgroup to be populated")
    if state.phase == PHASE_EMPTY_VERIFIED and state.cgroup_populated is not False:
        _fail("EMPTY_VERIFIED requires an authenticated empty cgroup observation")


def _replace(state: ControllerState, **changes: object) -> ControllerState:
    values = {
        "phase": state.phase,
        "admission_closed": state.admission_closed,
        "session": state.session,
        "controller": state.controller,
        "cgroup": state.cgroup,
        "cgroup_populated": state.cgroup_populated,
        "drain_reason": state.drain_reason,
        "last_rejection": state.last_rejection,
        "boot_id": state.boot_id,
        "highest_generation": state.highest_generation,
    }
    values.update(changes)
    return ControllerState(**values)  # type: ignore[arg-type]


def _drain(state: ControllerState, reason: str) -> ControllerState:
    if state.phase == PHASE_PREFLIGHT:
        # PREFLIGHT is deliberately pre-acquisition in this pure model.  A
        # future native implementation must never route an acquired or
        # uncertain lease/cgroup through this branch; it must durably recover
        # that condition as an active DRAINING record instead.
        return initial_state(
            last_rejection=f"preflight-failed:{reason}",
            boot_id=state.boot_id,
            highest_generation=state.highest_generation,
        )
    if state.phase == PHASE_IDLE:
        return _replace(state, last_rejection=reason)
    if state.session is None or state.cgroup is None:
        _fail("active controller lost its session/cgroup before drain")
    return ControllerState(
        PHASE_DRAINING,
        True,
        state.session,
        state.controller,
        state.cgroup,
        state.cgroup_populated,
        reason,
        reason,
        state.boot_id,
        state.highest_generation,
    )


def _event(value: object, fields: set[str], label: str) -> dict[str, object]:
    return _mapping(value, fields, label)


def _control_packet(raw: object, expected_kind: str, session: Session) -> None:
    if type(raw) is not bytes or not raw or len(raw) > MAX_CONTROL_BYTES:
        _fail("control packet has an unsafe byte length")
    try:
        packet = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        _fail(f"cannot parse control packet: {error}")
    expected = {"schema_version", "kind", "boot_id", "lease_id", "generation", "nonce", "contract_sha256"}
    if type(packet) is not dict or canonical_json_bytes(packet) != raw or set(packet) != expected:
        _fail("control packet is not bounded canonical protocol JSON")
    if (
        type(packet["schema_version"]) is not str or type(packet["kind"]) is not str
        or type(packet["boot_id"]) is not str or type(packet["lease_id"]) is not str
        or type(packet["generation"]) is not int or type(packet["nonce"]) is not str
        or type(packet["contract_sha256"]) is not str
        or packet["schema_version"] != CONTROL_SCHEMA_VERSION or packet["kind"] != expected_kind
        or packet["boot_id"] != session.boot_id or packet["lease_id"] != session.lease_id
        or packet["generation"] != session.generation
        or packet["nonce"] != session.nonce or packet["contract_sha256"] != session.guardian_contract_sha256
    ):
        _fail("control packet does not bind the active lease generation/nonce/contract")


def _apply_control(state: ControllerState, raw: object, event_name: str) -> ControllerState:
    event = _event(raw, {"kind", "packet", "credentials", "cgroup", "ancillary_fds"}, event_name)
    required = PHASE_BOOTSTRAP_STARTING if event_name == "READY" else PHASE_NO_ACTION_LIVE
    if state.phase != required or state.session is None or state.cgroup is None:
        return _drain(state, f"{event_name.lower()}-out-of-phase")
    try:
        _control_packet(event["packet"], event_name.lower(), state.session)
        _same_identity(event["credentials"], state.session.worker, f"{event_name}.credentials")
        _same_cgroup(event["cgroup"], state.cgroup, f"{event_name}.cgroup")
        if event["ancillary_fds"] != []:
            _fail("control packet must not carry an FD or release capability")
    except GuardianLeaseContractError as error:
        return _drain(state, f"control-violation:{error}")
    if event_name == "READY":
        return _replace(state, phase=PHASE_NO_ACTION_LIVE, cgroup_populated=True, last_rejection=None)
    return _drain(state, "no-action-complete")


def _terminal_worker_tokens(value: object, session: Session, label: str) -> None:
    if type(value) is not list or value != [session.worker.pidfd_token]:
        _fail(f"{label} must contain exactly the registered terminal worker pidfd token")


def _same_controller(value: object, state: ControllerState, label: str) -> None:
    if state.controller is None:
        _fail("active state has no registered controller identity")
    _same_identity(value, state.controller, label)


def transition(state: ControllerState, raw_event: object) -> ControllerState:
    """Apply one synthetic event. Invalid events drain; none open admission."""
    _check_state(state)
    try:
        if type(raw_event) is not dict or type(raw_event.get("kind")) is not str:
            next_state = _drain(state, "malformed-event")
        else:
            kind = raw_event["kind"]
            if kind == "START":
                event = _event(raw_event, {"kind", "session"}, "START")
                if state.phase != PHASE_IDLE:
                    next_state = _drain(state, "competing-start")
                else:
                    try:
                        session = _validate_session(event["session"])
                        if state.boot_id is not None and state.boot_id != session.boot_id:
                            _fail("START requires durable recovery before accepting a different boot identity")
                        if state.boot_id == session.boot_id and session.generation <= state.highest_generation:
                            _fail("START reuses a fenced generation, lease, or nonce")
                    except GuardianLeaseContractError as error:
                        next_state = initial_state(
                            last_rejection=f"preflight-failed:{error}",
                            boot_id=state.boot_id,
                            highest_generation=state.highest_generation,
                        )
                    else:
                        next_state = ControllerState(
                            PHASE_PREFLIGHT,
                            True,
                            session=session,
                            controller=session.controller,
                            boot_id=session.boot_id,
                            highest_generation=session.generation,
                        )
            elif kind == "NATIVE_PREFLIGHT_OK":
                event = _event(
                    raw_event,
                    {"kind", "guardian", "warden", "controller", "cgroup", "populated"},
                    kind,
                )
                if state.phase != PHASE_PREFLIGHT or state.session is None:
                    next_state = _drain(state, "preflight-ok-out-of-phase")
                else:
                    try:
                        _same_identity(event["guardian"], state.session.guardian, "NATIVE_PREFLIGHT_OK.guardian")
                        _same_identity(event["warden"], state.session.warden, "NATIVE_PREFLIGHT_OK.warden")
                        _same_controller(event["controller"], state, "NATIVE_PREFLIGHT_OK.controller")
                        cgroup = _cgroup(event["cgroup"], "NATIVE_PREFLIGHT_OK.cgroup")
                        if event["populated"] is not False:
                            _fail("fresh controller-reserved cgroup must be empty before lease")
                    except GuardianLeaseContractError as error:
                        next_state = initial_state(
                            last_rejection=f"preflight-failed:{error}",
                            boot_id=state.boot_id,
                            highest_generation=state.highest_generation,
                        )
                    else:
                        next_state = ControllerState(
                            PHASE_LEASED_EMPTY,
                            True,
                            state.session,
                            state.controller,
                            cgroup,
                            False,
                            boot_id=state.boot_id,
                            highest_generation=state.highest_generation,
                        )
            elif kind == "PREFLIGHT_FAIL":
                _event(raw_event, {"kind"}, kind)
                next_state = (
                    initial_state(
                        last_rejection="preflight-failed:explicit",
                        boot_id=state.boot_id,
                        highest_generation=state.highest_generation,
                    )
                    if state.phase == PHASE_PREFLIGHT
                    else _drain(state, "preflight-fail-out-of-phase")
                )
            elif kind == "BOOTSTRAP_EXECED":
                event = _event(raw_event, {"kind", "worker", "cgroup"}, kind)
                if state.phase != PHASE_LEASED_EMPTY or state.session is None or state.cgroup is None:
                    next_state = _drain(state, "bootstrap-exec-out-of-phase")
                else:
                    _same_identity(event["worker"], state.session.worker, "BOOTSTRAP_EXECED.worker")
                    _same_cgroup(event["cgroup"], state.cgroup, "BOOTSTRAP_EXECED.cgroup")
                    next_state = _replace(state, phase=PHASE_BOOTSTRAP_STARTING)
            elif kind in {"READY", "NO_ACTION_COMPLETE"}:
                next_state = _apply_control(state, raw_event, kind)
            elif kind in DRAIN_EVENTS or kind in ACTION_LIKE_EVENTS:
                _event(raw_event, {"kind"}, kind)
                next_state = _drain(state, kind.lower())
            elif kind == "CGROUP_POPULATED":
                event = _event(raw_event, {"kind", "controller", "cgroup"}, kind)
                if state.phase == PHASE_DRAINING and state.cgroup is not None:
                    _same_controller(event["controller"], state, "CGROUP_POPULATED.controller")
                    _same_cgroup(event["cgroup"], state.cgroup, "CGROUP_POPULATED.cgroup")
                    next_state = _replace(state, cgroup_populated=True, last_rejection="cgroup-remains-populated")
                else:
                    next_state = _drain(state, "cgroup-populated-out-of-phase")
            elif kind == "CGROUP_EMPTY":
                event = _event(
                    raw_event,
                    {"kind", "controller", "cgroup", "populated", "terminal_pidfd_tokens"},
                    kind,
                )
                if state.phase != PHASE_DRAINING or state.session is None or state.cgroup is None:
                    next_state = _drain(state, "cgroup-empty-out-of-phase")
                else:
                    _same_controller(event["controller"], state, "CGROUP_EMPTY.controller")
                    _same_cgroup(event["cgroup"], state.cgroup, "CGROUP_EMPTY.cgroup")
                    if event["populated"] is not False:
                        _fail("empty observation requires populated=false")
                    _terminal_worker_tokens(event["terminal_pidfd_tokens"], state.session, "CGROUP_EMPTY.terminal_pidfd_tokens")
                    next_state = _replace(
                        state,
                        phase=PHASE_EMPTY_VERIFIED,
                        cgroup_populated=False,
                        drain_reason="cgroup-empty",
                        last_rejection=None,
                    )
            elif kind == "CONTROLLER_RESTART":
                event = _event(
                    raw_event,
                    {"kind", "new_controller", "cgroup", "populated", "terminal_pidfd_tokens"},
                    kind,
                )
                if (
                    state.phase not in {PHASE_LEASED_EMPTY, PHASE_BOOTSTRAP_STARTING, PHASE_NO_ACTION_LIVE, PHASE_DRAINING}
                    or state.session is None or state.cgroup is None
                ):
                    next_state = _drain(state, "controller-restart-out-of-phase")
                else:
                    new_controller = _root_identity(event["new_controller"], "CONTROLLER_RESTART.new_controller", pid1=True)
                    if new_controller in {state.session.guardian, state.session.warden, state.session.worker}:
                        _fail("restarted controller must stay distinct from guardian, warden, and worker")
                    restarted = _replace(state, controller=new_controller)
                    _same_cgroup(event["cgroup"], restarted.cgroup, "CONTROLLER_RESTART.cgroup")
                    if event["populated"] is not False:
                        next_state = _drain(restarted, "controller-restart-cgroup-populated")
                    else:
                        _terminal_worker_tokens(
                            event["terminal_pidfd_tokens"],
                            restarted.session,
                            "CONTROLLER_RESTART.terminal_pidfd_tokens",
                        )
                        next_state = _replace(
                            restarted,
                            phase=PHASE_EMPTY_VERIFIED,
                            cgroup_populated=False,
                            drain_reason="controller-restart-empty",
                            last_rejection=None,
                        )
            elif kind == "CONTROLLER_RELEASE":
                event = _event(raw_event, {"kind", "controller"}, kind)
                if state.phase != PHASE_EMPTY_VERIFIED:
                    next_state = _drain(state, "controller-release-before-empty")
                else:
                    _same_controller(event["controller"], state, "CONTROLLER_RELEASE.controller")
                    next_state = initial_state(
                        boot_id=state.boot_id,
                        highest_generation=state.highest_generation,
                    )
            else:
                next_state = _drain(state, f"unsupported-event:{kind}")
    except GuardianLeaseContractError as error:
        next_state = _drain(state, f"invalid-event:{error}")
    _check_state(next_state)
    return next_state


def control_packet(session: Session, kind: str) -> bytes:
    if kind not in {"ready", "no_action_complete"}:
        _fail("guardian lease v1 has only READY and NO_ACTION_COMPLETE packets")
    return canonical_json_bytes(
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "kind": kind,
            "boot_id": session.boot_id,
            "lease_id": session.lease_id,
            "generation": session.generation,
            "nonce": session.nonce,
            "contract_sha256": session.guardian_contract_sha256,
        }
    )


def status_record(state: ControllerState) -> dict[str, object]:
    _check_state(state)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "authority": AUTHORITY,
        "installation_status": INSTALLATION_STATUS,
        "phase": state.phase,
        "admission_closed": state.admission_closed,
        "new_admission_allowed": state.phase == PHASE_IDLE,
        "guarantees": {
            "native_guardian_installed": False,
            "admission_controller_installed": False,
            "lease_warden_installed": False,
            "durable_admission_ledger_installed": False,
            "bootstrap_executed": False,
            "gpu_lock_acquired": False,
            "gpu_queried": False,
            "docker_invoked": False,
            "evidence_created": False,
            "semantic_replay_run": False,
            "receipt_published": False,
            "qualification_decided": False,
        },
    }
