#!/usr/bin/env python3
"""Same-invocation terminal publisher for raw rollback provenance v4.

There is deliberately no path-based CLI or public *reopen-and-bind* API for
an existing preparation or transaction.  The sole public raw producer starts
a new preparation and reaches transaction → v3 → v4 only through lexical
normal-return callbacks while exclusive root and switch locks remain held.  A
fresh structural verifier can inspect raw evidence but cannot recreate that
producer-success edge after an ambiguous completion-pair fsync.
"""

from __future__ import annotations

import fcntl
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar

import bind_raw_rc3_rollback_capture as v3_binder
import capture_rc3_rollback_atomic_transaction_v1 as transaction
import check_rc3_rollback_provenance_v4 as checker
import prepare_rc3_rollback_artifacts_v1 as prepare
import provenance_v2_common as common


sys.dont_write_bytecode = True

_MANIFEST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,122}\.json$")


class RollbackV4TerminalBindError(ValueError):
    """The same-invocation v4 terminal compositor cannot be published."""


def _fail(code: str, message: str) -> NoReturn:
    if code == "ambiguous-terminal-publication":
        message = f"{code}: {message}"
    error = RollbackV4TerminalBindError(message)
    error.reason_code = code  # type: ignore[attr-defined]
    raise error


T = TypeVar("T")


def _common(call: Callable[[], T]) -> T:
    try:
        return call()
    except common.ProvenanceV2Error as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _checker(call: Callable[[], T]) -> T:
    try:
        return call()
    except checker.RollbackV4ProvenanceError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _v3_binder(call: Callable[[], T]) -> T:
    try:
        return call()
    except v3_binder.RollbackBindError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _transaction(call: Callable[[], T]) -> T:
    try:
        return call()
    except transaction.RollbackAtomicTransactionError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _prepare(call: Callable[[], T]) -> T:
    try:
        return call()
    except prepare.RollbackArtifactPreparationError as error:
        _fail(getattr(error, "reason_code", "unsafe-evidence"), str(error))


def _manifest_name(value: Any, label: str) -> str:
    if type(value) is not str or _MANIFEST_NAME.fullmatch(value) is None:
        _fail("invalid-manifest-name", f"{label} must be one direct nonhidden .json root leaf")
    return value


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _assert_external_to_source(evidence_root: Path) -> None:
    try:
        evidence_root.relative_to(_source_root())
    except ValueError:
        return
    _fail("evidence-root-inside-source-checkout", "--evidence-root must be outside the source checkout")


def _lock(descriptor: int, mode: int, label: str) -> None:
    try:
        fcntl.flock(descriptor, mode | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("lock-unavailable", f"cannot acquire {label} lock: {error}")


def _unlock_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass


def _close_quietly(descriptor: int | None) -> None:
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _output_names(name: str) -> tuple[str, str, str]:
    return name, f"{name}.intent", f"{name}.complete"


def _assert_output_absent(root_fd: int, name: str) -> None:
    for output in _output_names(name):
        try:
            os.lstat(output, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail("output-preflight-failed", f"cannot inspect v4 output {output!r}: {error}")
        _fail("output-name-collision", f"v4 output or reserved sibling {output!r} already exists")


def _completion_pair_is_visible(root_fd: int, name: str) -> bool:
    final_name = f"{name}.complete"
    intent_name = f"{name}.intent"
    try:
        final = os.lstat(final_name, dir_fd=root_fd)
        intent = os.lstat(intent_name, dir_fd=root_fd)
    except OSError:
        return False
    return (
        stat.S_ISREG(final.st_mode)
        and stat.S_ISREG(intent.st_mode)
        and stat.S_IMODE(final.st_mode) == 0o600
        and stat.S_IMODE(intent.st_mode) == 0o600
        and final.st_uid == os.geteuid()
        and intent.st_uid == os.geteuid()
        and final.st_nlink == 2
        and intent.st_nlink == 2
        and (final.st_dev, final.st_ino) == (intent.st_dev, intent.st_ino)
    )


def capture_and_bind_rollback_terminal_v4(
    preparation_request: prepare.PreparationRequest,
    bind_request_path: str,
    rollback_v3_manifest_name: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Prepare → transaction → v3 → v4 as one raw-only normal-return chain.

    The preparation request supplies the only evidence root.  This public
    producer rejects source-tree roots, acquires the root lock before any
    fixed child exists, and invokes each successor only from its predecessor's
    normal-return closure.  Neither an earlier ambiguous preparation nor an
    earlier ambiguous transaction can be resumed: their create-only children
    occupy the root and no public or module-level held-FD compositor exists.
    It never starts a service, modifies a deployment path, contacts a GPU, or
    reports rollback success or qualification.
    """

    if not isinstance(preparation_request, prepare.PreparationRequest):
        _fail("invalid-preparation-request", "preparation_request must be a PreparationRequest")
    evidence_root = preparation_request.evidence_root
    _assert_external_to_source(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(evidence_root, "--evidence-root")
    )
    try:
        _lock(root_fd, fcntl.LOCK_EX, "exclusive evidence-root")

        def after_normal_preparation(
            _preparation_replay: prepare.ArtifactPreparationReplay,
            switch_fd: int,
        ) -> dict[str, Any]:
            """Close the prepared session only while its original FD stays live."""

            _lock(switch_fd, fcntl.LOCK_EX, "exclusive rollback switch")
            try:
                v3_name = _manifest_name(rollback_v3_manifest_name, "rollback v3 manifest name")
                name = _manifest_name(manifest_name, "rollback v4 manifest name")
                if name == v3_name:
                    _fail("output-name-collision", "v4 manifest name must differ from the v3 manifest")
                _transaction(lambda: transaction._require_held_switch_fd(root_fd, switch_fd))  # noqa: SLF001
                _v3_binder(lambda: v3_binder._assert_output_absent(root_fd, v3_name))  # noqa: SLF001
                _assert_output_absent(root_fd, name)

                def after_normal_transaction(replay: transaction.AtomicTransactionReplay) -> dict[str, Any]:
                    """Lexical terminal publisher; no replay object crosses module scope."""

                    _v3_binder(
                        lambda: v3_binder._bind_raw_rollback_manifest_held_locked_fd(  # noqa: SLF001
                            root_fd,
                            bind_request_path,
                            v3_name,
                        )
                    )
                    v3_descriptor, v3_report, replayed_transaction = _checker(
                        lambda: checker._replay_inputs_on_held_switch_fd(  # noqa: SLF001
                            root_fd,
                            switch_fd,
                            v3_name,
                        )
                    )
                    if replayed_transaction != replay:
                        _fail(
                            "transaction-capture-drift",
                            "the held transaction replay differs from the normal capture result",
                        )
                    candidate_id = v3_report.get("candidate_id")
                    bindings = v3_report.get("bindings")
                    if type(candidate_id) is not str or not isinstance(bindings, dict):
                        _fail("invalid-v3-report", "held v3 report lacks exact candidate identity bindings")
                    manifest = {
                        "schema_version": checker.ROLLBACK_V4_MANIFEST_VERSION,
                        "capture_status": "captured",
                        "qualification_status": "not-run",
                        "candidate_id": candidate_id,
                        "bindings": dict(bindings),
                        "rollback_v3_manifest": v3_descriptor.as_json(),
                        "atomic_transaction_session": replay.session_descriptor.as_json(),
                    }
                    raw_manifest = _common(lambda: common.canonical_json_bytes(manifest))
                    draft_descriptor = _common(
                        lambda: common.descriptor_for_bytes(name, raw_manifest, "rollback v4 draft manifest")
                    )
                    preflight = _checker(
                        lambda: checker.verify_rollback_provenance_v4_bytes_on_held_switch_fd(
                            root_fd,
                            switch_fd,
                            draft_descriptor,
                            raw_manifest,
                        )
                    )
                    created = _common(
                        lambda: common.write_create_only_json(
                            root_fd,
                            name,
                            manifest,
                            "rollback v4 raw manifest",
                        )
                    )
                    if created.descriptor(name, "rollback v4 created manifest") != draft_descriptor:
                        _fail(
                            "published-manifest-descriptor-mismatch",
                            "create-only publication bytes differ from the held-FD preflight",
                        )
                    replayed = _checker(
                        lambda: checker.verify_rollback_provenance_v4_on_held_switch_fd(
                            root_fd,
                            switch_fd,
                            name,
                        )
                    )
                    if replayed != preflight:
                        _fail(
                            "post-publication-replay-drift",
                            "on-disk v4 replay differs from its held-FD preflight",
                        )
                    marker = {
                        "schema_version": checker.ROLLBACK_V4_COMPLETION_VERSION,
                        "artifact_filename": name,
                        "artifact_sha256": created.sha256,
                    }
                    intent_name = f"{name}.intent"
                    _common(
                        lambda: common.write_create_only_json(
                            root_fd,
                            intent_name,
                            marker,
                            "rollback v4 completion marker intent",
                        )
                    )
                    try:
                        _common(
                            lambda: common.publish_create_only_hardlink(
                                root_fd,
                                intent_name,
                                f"{name}.complete",
                                "rollback v4 completion marker",
                            )
                        )
                    except RollbackV4TerminalBindError:
                        if _completion_pair_is_visible(root_fd, name):
                            _fail(
                                "ambiguous-terminal-publication",
                                "completion marker became visible but its final directory sync failed; "
                                "no later invocation may treat it as producer success",
                            )
                        raise
                    return _checker(
                        lambda: checker.verify_completed_rollback_provenance_v4_on_held_switch_fd(
                            root_fd,
                            switch_fd,
                            name,
                        )
                    )

                return _transaction(
                    lambda: transaction._capture_atomic_transaction_then_on_success_held_switch_fd(  # noqa: SLF001
                        root_fd,
                        switch_fd,
                        after_normal_transaction,
                    )
                )
            finally:
                _unlock_quietly(switch_fd)

        return _prepare(
            lambda: prepare._prepare_artifacts_then_on_success_held_root_fd(  # noqa: SLF001
                preparation_request,
                root_fd,
                after_normal_preparation,
            )
        )
    finally:
        _unlock_quietly(root_fd)
        _close_quietly(root_fd)
