#!/usr/bin/env python3
"""Privately bind one RC3 candidate config bridge to fixed rollback paths.

The future authenticated rollback runner needs one current candidate process
bridge at fixed root-relative paths, while the existing endpoint observer
publishes its raw body below ``config-bridge/raw``.  This module is a narrow
private runner helper: it creates ``config/`` before the candidate launches,
then, after the observer completes and the server has written its startup
artifact, create-only materializes that already-observed endpoint body as
``config/endpoint.json``.  It replays the bridge both before and after the
copy through held root FDs.

It is not a path-resume API, a public CLI, a service/GPU producer, or a
qualification/rollback verdict.  A visible fixed endpoint alone never proves
that a candidate process was successfully run; the later same-stack rollback
finalizer must replay this bridge with all other raw evidence.
"""

from __future__ import annotations

import fcntl
import os
import stat
from pathlib import Path
from typing import Callable, NoReturn, TypeVar

import check_c02_config_bridge_v1 as config_bridge
import provenance_v2_common as common


CONFIG_DIRECTORY_NAME = "config"
CONFIG_ENDPOINT_NAME = "endpoint.json"
CONFIG_STARTUP_NAME = "startup.json"
CONFIG_BRIDGE_DIRECTORY_NAME = "config-bridge"
STABLE_DEFAULT_PROFILE = "stable-default"
CAPTURED_ENDPOINT_PATH = f"{CONFIG_BRIDGE_DIRECTORY_NAME}/raw/config-endpoint.json"
CONFIG_ENDPOINT_PATH = f"{CONFIG_DIRECTORY_NAME}/{CONFIG_ENDPOINT_NAME}"
CONFIG_STARTUP_PATH = f"{CONFIG_DIRECTORY_NAME}/{CONFIG_STARTUP_NAME}"
CONFIG_BRIDGE_SESSION_PATH = f"{CONFIG_BRIDGE_DIRECTORY_NAME}/session.json"


class RollbackCandidateConfigMaterializationError(ValueError):
    """The candidate config bridge cannot be bound to fixed rollback paths."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackCandidateConfigMaterializationError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _bridge(call: Callable[[], T]) -> T:
    try:
        return call()
    except config_bridge.ConfigBridgeReplayError as error:
        _fail(getattr(error, "reason_code", "invalid-config-bridge"), str(error))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_external_to_source(evidence_root: Path) -> None:
    try:
        evidence_root.relative_to(_source_root())
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be outside the source checkout",
    )


def _lock_root(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("lock-unavailable", f"cannot acquire exclusive evidence-root lock: {error}")


def _unlock_quietly(root_fd: int | None) -> None:
    if root_fd is not None:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _assert_absent(directory_fd: int, name: str, label: str) -> None:
    try:
        os.lstat(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return
    except OSError as error:
        _fail("output-preflight-failed", f"cannot inspect {label}: {error}")
    _fail("output-name-collision", f"{label} already exists")


def _assert_exact_config_inventory(
    config_fd: int,
    expected_names: frozenset[str],
    label: str,
) -> None:
    """Require the fixed config child to contain only private named leaves.

    The future runner owns this narrow directory across candidate startup and
    endpoint projection.  Refusing a sidecar avoids silently mixing a stale
    startup artifact or an unreviewed runner output into the fixed-path bridge.
    The later bridge replays still read the declared leaves through held root
    FDs; this check closes the directory-layout boundary before and after the
    create-only projection.
    """

    try:
        names = frozenset(os.listdir(config_fd))
    except OSError as error:
        _fail("config-inventory-unreadable", f"cannot list {label}: {error}")
    if names != expected_names:
        _fail(
            "unexpected-config-inventory",
            f"{label} must contain exactly {sorted(expected_names)!r}, got {sorted(names)!r}",
        )
    for name in sorted(expected_names):
        try:
            metadata = os.stat(name, dir_fd=config_fd, follow_symlinks=False)
        except OSError as error:
            _fail("config-inventory-unreadable", f"cannot inspect {label}/{name}: {error}")
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            _fail(
                "unsafe-config-inventory-leaf",
                f"{label}/{name} must be a single-link regular file",
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            _fail(
                "unsafe-config-inventory-leaf",
                f"{label}/{name} must be mode 0600",
            )
        if metadata.st_uid != os.geteuid():
            _fail(
                "unsafe-config-inventory-leaf",
                f"{label}/{name} must be owned by the effective UID",
            )


def _open_locked_root(evidence_root: Path, label: str) -> int:
    _assert_external_to_source(evidence_root)
    root_fd = _common(lambda: common.open_private_evidence_directory(evidence_root, label))
    try:
        _lock_root(root_fd)
        _common(lambda: common.require_private_evidence_directory_fd(root_fd, label))
    except BaseException:
        _close_quietly(root_fd)
        raise
    return root_fd


def _initialize_candidate_config_directory(evidence_root: Path) -> None:
    """Create the fixed private ``config/`` child before candidate launch.

    This deliberately permits no resumed/partial directory: an existing child
    is a collision, so a later invocation cannot mix a different candidate's
    startup artifact with a prior bridge capture.
    """

    root_fd: int | None = None
    config_fd: int | None = None
    try:
        root_fd = _open_locked_root(evidence_root, "rollback candidate config evidence root")
        _assert_absent(root_fd, CONFIG_DIRECTORY_NAME, "fixed candidate config directory")
        config_fd = _common(
            lambda: common.create_private_child_directory(
                root_fd,
                CONFIG_DIRECTORY_NAME,
                "fixed candidate config directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                config_fd,
                CONFIG_DIRECTORY_NAME,
                "fixed candidate config directory",
            )
        )
    finally:
        _close_quietly(config_fd)
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)


def _same_bridge_facts(
    before: config_bridge.ReplayedConfigBridge,
    after: config_bridge.ReplayedConfigBridge,
    expected_endpoint: common.EvidenceDescriptor,
) -> None:
    if after.endpoint != expected_endpoint:
        _fail(
            "projected-endpoint-descriptor-mismatch",
            "fixed candidate endpoint differs from the create-only projected bytes",
        )
    if (
        before.candidate_id != after.candidate_id
        or before.configuration_profile != after.configuration_profile
        or before.configuration_sha256 != after.configuration_sha256
        or before.startup_artifact != after.startup_artifact
        or before.endpoint_observation != after.endpoint_observation
        or before.effective_config != after.effective_config
        or before.effective_config_sha256 != after.effective_config_sha256
        or before.target != after.target
    ):
        _fail(
            "config-bridge-replay-drift",
            "fixed endpoint replay differs from the pre-copy captured config bridge",
        )


def _materialize_candidate_config_bridge(
    evidence_root: Path,
    *,
    candidate_id: str,
    configuration_profile: str,
) -> config_bridge.ReplayedConfigBridge:
    """Create-only project the captured endpoint and replay its fixed bridge.

    The first replay uses the observer-owned raw endpoint path.  It occurs
    before writing ``config/endpoint.json`` so malformed/mismatched observer
    state cannot leave a terminal-looking fixed endpoint.  The second replay
    uses only the fixed paths consumed by the later rollback finalizer.
    """

    if configuration_profile != STABLE_DEFAULT_PROFILE:
        _fail(
            "invalid-configuration-profile",
            "RC3 rollback candidate config must use stable-default",
        )
    root_fd: int | None = None
    config_fd: int | None = None
    try:
        root_fd = _open_locked_root(evidence_root, "rollback candidate config evidence root")
        config_fd = _common(
            lambda: common.open_private_child_directory(
                root_fd,
                CONFIG_DIRECTORY_NAME,
                "fixed candidate config directory",
            )
        )
        _common(
            lambda: common.require_private_child_directory_fd(
                root_fd,
                config_fd,
                CONFIG_DIRECTORY_NAME,
                "fixed candidate config directory",
            )
        )
        _assert_absent(config_fd, CONFIG_ENDPOINT_NAME, "fixed candidate config endpoint")
        _assert_exact_config_inventory(
            config_fd,
            frozenset({CONFIG_STARTUP_NAME}),
            "fixed candidate config directory before endpoint projection",
        )
        preflight = _bridge(
            lambda: config_bridge.replay_config_bridge_v1_fd(
                root_fd,
                candidate_id=candidate_id,
                configuration_profile=configuration_profile,
                endpoint_path=CAPTURED_ENDPOINT_PATH,
                startup_artifact_path=CONFIG_STARTUP_PATH,
                session_path=CONFIG_BRIDGE_SESSION_PATH,
            )
        )
        endpoint_raw = _common(
            lambda: common.read_bounded_regular_relative(
                root_fd,
                CAPTURED_ENDPOINT_PATH,
                "captured candidate config endpoint",
                maximum_bytes=common.DEFAULT_MAX_JSON_BYTES,
            )
        )
        source_descriptor = _common(
            lambda: common.descriptor_for_bytes(
                CAPTURED_ENDPOINT_PATH,
                endpoint_raw,
                "captured candidate config endpoint",
            )
        )
        if source_descriptor != preflight.endpoint:
            _fail(
                "captured-endpoint-descriptor-drift",
                "captured endpoint bytes differ from the held preflight descriptor",
            )
        created = _common(
            lambda: common.write_create_only(
                config_fd,
                CONFIG_ENDPOINT_NAME,
                endpoint_raw,
                "fixed candidate config endpoint",
            )
        )
        expected_endpoint = created.descriptor(
            CONFIG_ENDPOINT_PATH,
            "fixed candidate config endpoint",
        )
        _assert_exact_config_inventory(
            config_fd,
            frozenset({CONFIG_STARTUP_NAME, CONFIG_ENDPOINT_NAME}),
            "fixed candidate config directory after endpoint projection",
        )
        replayed = _bridge(
            lambda: config_bridge.replay_config_bridge_v1_fd(
                root_fd,
                candidate_id=candidate_id,
                configuration_profile=configuration_profile,
                endpoint_path=CONFIG_ENDPOINT_PATH,
                startup_artifact_path=CONFIG_STARTUP_PATH,
                session_path=CONFIG_BRIDGE_SESSION_PATH,
            )
        )
        _same_bridge_facts(preflight, replayed, expected_endpoint)
        return replayed
    finally:
        _close_quietly(config_fd)
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)
