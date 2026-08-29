#!/usr/bin/env python3
"""Bind one reconstructed-RC2 rollback capture from closed path-only input.

This is a local provenance binder, not a rollback runner.  It never starts a
service, opens a network socket, invokes a GPU tool, renames an active
artifact, or decides qualification.  It derives every v3 manifest descriptor
and both process targets through one caller-held private evidence-root FD,
preflights the full raw v3 replay before publication, then create-only
publishes and self-verifies one nonterminal manifest.  The retained v3
manifest schema represents its three binding inputs as SHA-256 scalars rather
than descriptors; this binder derives those scalars from raw leaves at bind
time, but v3 does not preserve those input descriptors for an independent
later replay.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, NoReturn, Sequence, TypeVar

import check_rc3_rollback_provenance_v3 as checker
import provenance_v2_common as common


BIND_REQUEST_VERSION = "riley.rc3-rollback-bind-request.v3"
MAX_BIND_REQUEST_BYTES = common.DEFAULT_MAX_JSON_BYTES
MAX_RAW_LEAF_BYTES = checker.MAX_RAW_LEAF_BYTES
# Keep the same bounded streaming ceiling as the v3 verifier.  A narrower
# producer-only limit would make valid replayable evidence unbindable.
MAX_BIND_ARTIFACT_BYTES = checker.MAX_ARTIFACT_BYTES
MAX_RELATIVE_PATH_BYTES = checker.MAX_RELATIVE_PATH_BYTES
MAX_CANDIDATE_ID_BYTES = 128
MAX_MANIFEST_NAME_BYTES = 128
MANIFEST_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json$")


class RollbackBindError(ValueError):
    """A raw rollback bind request cannot safely produce a manifest."""


def _fail(code: str, message: str) -> NoReturn:
    error = RollbackBindError(message)
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
    except checker.RollbackV3ProvenanceError as error:
        _fail(getattr(error, "reason_code", "invalid-rollback-provenance"), str(error))


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else []
        _fail(
            "unexpected-field-set",
            f"{label} fields differ; expected={sorted(fields)}, actual={actual}",
        )
    return value


def _candidate_id(value: Any, label: str) -> str:
    if (
        type(value) is not str
        or len(value) > MAX_CANDIDATE_ID_BYTES
        or checker.CANDIDATE_RE.fullmatch(value) is None
    ):
        _fail("invalid-candidate-id", f"{label} must be a canonical Riley RC ID")
    return value


def _path(value: Any, label: str) -> str:
    relative = _common(lambda: common.validate_relative_path(value, label))
    if len(relative) > MAX_RELATIVE_PATH_BYTES:
        _fail(
            "invalid-relative-path",
            f"{label} exceeds {MAX_RELATIVE_PATH_BYTES} bytes",
        )
    return relative


def _manifest_name(value: Any) -> str:
    if (
        type(value) is not str
        or len(value) > MAX_MANIFEST_NAME_BYTES
        or MANIFEST_NAME_RE.fullmatch(value) is None
        or "/" in value
    ):
        _fail(
            "invalid-manifest-name",
            "--manifest-name must be a nonhidden root direct-child .json name",
        )
    return value


def _output_names(manifest_name: str) -> tuple[str, str, str]:
    """Reserve terminal-looking siblings even though v3 never writes them.

    A raw v3 binder deliberately has no completion protocol.  Refusing stale
    ``.complete`` or ``.intent`` siblings prevents a later layer or operator
    from visually pairing this nonterminal manifest name with unrelated
    terminal-looking state.
    """

    return (
        manifest_name,
        f"{manifest_name}.complete",
        f"{manifest_name}.intent",
    )


def _assert_external_to_source_checkout(evidence_root: Path) -> None:
    """Reject a lexical source-checkout child before opening evidence."""

    source_root = Path(__file__).resolve().parents[2]
    try:
        evidence_root.relative_to(source_root)
    except ValueError:
        return
    _fail(
        "evidence-root-inside-source-checkout",
        "--evidence-root must be external to the source checkout",
    )


def _reserve_path(value: Any, label: str, reserved_paths: set[str]) -> str:
    relative = _path(value, label)
    if relative in reserved_paths:
        _fail("duplicate-evidence-path", f"{label} reuses evidence path {relative!r}")
    reserved_paths.add(relative)
    return relative


def _read_canonical_request(
    root_fd: int,
    bind_request_path: str,
) -> tuple[str, Mapping[str, Any]]:
    relative = _path(bind_request_path, "rollback v3 bind request path")
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            relative,
            "rollback v3 bind request",
            maximum_bytes=MAX_BIND_REQUEST_BYTES,
        )
    )
    document = _common(
        lambda: common.parse_canonical_json(
            raw,
            "rollback v3 bind request",
            maximum_bytes=MAX_BIND_REQUEST_BYTES,
        )
    )
    assert isinstance(document, Mapping)
    return relative, document


def _raw_descriptor_from_path(
    root_fd: int,
    path: str,
    label: str,
    *,
    maximum_bytes: int = MAX_RAW_LEAF_BYTES,
) -> common.EvidenceDescriptor:
    raw = _common(
        lambda: common.read_bounded_regular_relative(
            root_fd,
            path,
            label,
            maximum_bytes=maximum_bytes,
        )
    )
    if not raw:
        _fail("empty-evidence-leaf", f"{label} must be non-empty")
    return _common(lambda: common.descriptor_for_bytes(path, raw, label))


def _artifact_descriptor_from_path(
    root_fd: int,
    path: str,
    label: str,
) -> common.EvidenceDescriptor:
    descriptor = _common(
        lambda: common.describe_regular_relative(
            root_fd,
            path,
            label,
            maximum_bytes=MAX_BIND_ARTIFACT_BYTES,
        )
    )
    if descriptor.byte_length < 1:
        _fail("empty-evidence-leaf", f"{label} must be non-empty")
    return descriptor


def _path_descriptor_map(
    root_fd: int,
    value: Any,
    fields: frozenset[str],
    label: str,
    *,
    reserved_paths: set[str],
) -> dict[str, common.EvidenceDescriptor]:
    row = _exact(value, {f"{name}_path" for name in fields}, label)
    result: dict[str, common.EvidenceDescriptor] = {}
    for name in sorted(fields):
        path = _reserve_path(
            row[f"{name}_path"],
            f"{label}.{name}_path",
            reserved_paths,
        )
        result[name] = _raw_descriptor_from_path(
            root_fd,
            path,
            f"{label}.{name}",
        )
    return result


def _phase_from_paths(
    root_fd: int,
    value: Any,
    label: str,
    *,
    candidate_phase: bool,
    reserved_paths: set[str],
) -> tuple[checker.Target, dict[str, Any]]:
    fields = {"process_evidence", "health", "generation"}
    if candidate_phase:
        fields |= {
            "generation_audit_index_path",
            "shutdown_artifact_path",
            "shutdown_marker_path",
        }
    row = _exact(value, fields, label)
    process_evidence = _path_descriptor_map(
        root_fd,
        row["process_evidence"],
        checker.RAW_PROCESS_FIELDS,
        f"{label}.process_evidence",
        reserved_paths=reserved_paths,
    )
    target = _checker(
        lambda: checker.derive_phase_target_from_raw_evidence_fd(
            root_fd,
            process_evidence,
            label,
        )
    )
    health = _path_descriptor_map(
        root_fd,
        row["health"],
        checker.HTTP_EXCHANGE_FIELDS,
        f"{label}.health",
        reserved_paths=reserved_paths,
    )
    generation = _path_descriptor_map(
        root_fd,
        row["generation"],
        checker.HTTP_EXCHANGE_FIELDS,
        f"{label}.generation",
        reserved_paths=reserved_paths,
    )
    result: dict[str, Any] = {
        "target": target.as_json(),
        "process_evidence": {name: descriptor.as_json() for name, descriptor in process_evidence.items()},
        "health": {name: descriptor.as_json() for name, descriptor in health.items()},
        "generation": {name: descriptor.as_json() for name, descriptor in generation.items()},
    }
    if candidate_phase:
        audit_path = _reserve_path(
            row["generation_audit_index_path"],
            f"{label}.generation_audit_index_path",
            reserved_paths,
        )
        shutdown_artifact_path = _reserve_path(
            row["shutdown_artifact_path"],
            f"{label}.shutdown_artifact_path",
            reserved_paths,
        )
        shutdown_marker_path = _reserve_path(
            row["shutdown_marker_path"],
            f"{label}.shutdown_marker_path",
            reserved_paths,
        )
        result.update(
            {
                "audit": {
                    "availability": "source-owned",
                    "generation_audit_index": _raw_descriptor_from_path(
                        root_fd,
                        audit_path,
                        f"{label}.generation_audit_index",
                    ).as_json(),
                },
                "shutdown_artifact": _raw_descriptor_from_path(
                    root_fd,
                    shutdown_artifact_path,
                    f"{label}.shutdown_artifact",
                ).as_json(),
                "shutdown_marker": _raw_descriptor_from_path(
                    root_fd,
                    shutdown_marker_path,
                    f"{label}.shutdown_marker",
                ).as_json(),
            }
        )
    else:
        result["audit"] = {"availability": "not-supported"}
    return target, result


def _artifact_map_from_paths(
    root_fd: int,
    value: Any,
    label: str,
    *,
    reserved_paths: set[str],
) -> dict[str, dict[str, Any]]:
    row = _exact(value, {f"{name}_path" for name in checker.ARTIFACT_FIELDS}, label)
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(checker.ARTIFACT_FIELDS):
        path = _reserve_path(row[f"{name}_path"], f"{label}.{name}_path", reserved_paths)
        descriptor = (
            _raw_descriptor_from_path(root_fd, path, f"{label}.{name}")
            if name == "image_inspect"
            else _artifact_descriptor_from_path(root_fd, path, f"{label}.{name}")
        )
        result[name] = descriptor.as_json()
    return result


def _bindings_from_paths(
    root_fd: int,
    value: Any,
    *,
    reserved_paths: set[str],
) -> dict[str, str]:
    row = _exact(
        value,
        {
            "freeze_path",
            "base_release_candidate_report_path",
            "configuration_path",
        },
        "rollback v3 bind request.binding_evidence",
    )
    descriptors: dict[str, common.EvidenceDescriptor] = {}
    for name in sorted(row):
        path = _reserve_path(
            row[name],
            f"rollback v3 bind request.binding_evidence.{name}",
            reserved_paths,
        )
        descriptors[name] = _raw_descriptor_from_path(
            root_fd,
            path,
            f"rollback v3 bind request.binding_evidence.{name}",
        )
    return {
        "freeze_sha256": descriptors["freeze_path"].sha256,
        "base_release_candidate_report_sha256": descriptors[
            "base_release_candidate_report_path"
        ].sha256,
        "configuration_profile": checker.STABLE_DEFAULT_PROFILE,
        "configuration_sha256": descriptors["configuration_path"].sha256,
    }


def _preflight_baseline_aliases(
    root_fd: int,
    descriptor: common.EvidenceDescriptor,
    reserved_paths: set[str],
) -> None:
    """Reject aliases to every transitive A/B baseline leaf before output."""

    baseline_paths = set(reserved_paths)
    baseline_paths.remove(descriptor.path)
    _checker(
        lambda: checker._baseline_manifest(  # noqa: SLF001 - closed v3 replay primitive
            root_fd,
            {"manifest": descriptor.as_json()},
            used_paths=baseline_paths,
        )
    )


def _manifest_from_request(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    request_path, request = _read_canonical_request(root_fd, bind_request_path)
    row = _exact(
        request,
        {
            "schema_version",
            "candidate_id",
            "binding_evidence",
            "reconstructed_baseline",
            "candidate",
            "rollback",
            "candidate_artifacts",
            "rollback_artifacts",
            "atomic_switch",
        },
        "rollback v3 bind request",
    )
    if row["schema_version"] != BIND_REQUEST_VERSION:
        _fail(
            "unsupported-bind-request-version",
            f"rollback v3 bind request must use {BIND_REQUEST_VERSION}",
        )
    candidate_id = _candidate_id(row["candidate_id"], "rollback v3 bind request.candidate_id")
    reserved_paths = {request_path, *_output_names(manifest_name)}
    bindings = _bindings_from_paths(
        root_fd,
        row["binding_evidence"],
        reserved_paths=reserved_paths,
    )
    baseline_row = _exact(
        row["reconstructed_baseline"],
        {"manifest_path"},
        "rollback v3 bind request.reconstructed_baseline",
    )
    baseline_path = _reserve_path(
        baseline_row["manifest_path"],
        "rollback v3 bind request.reconstructed_baseline.manifest_path",
        reserved_paths,
    )
    baseline_descriptor = _raw_descriptor_from_path(
        root_fd,
        baseline_path,
        "rollback v3 reconstructed baseline manifest",
        maximum_bytes=checker.MAX_MANIFEST_BYTES,
    )
    candidate_target, candidate = _phase_from_paths(
        root_fd,
        row["candidate"],
        "rollback v3 bind request.candidate",
        candidate_phase=True,
        reserved_paths=reserved_paths,
    )
    rollback_target, rollback = _phase_from_paths(
        root_fd,
        row["rollback"],
        "rollback v3 bind request.rollback",
        candidate_phase=False,
        reserved_paths=reserved_paths,
    )
    if (candidate_target.pid, candidate_target.start_ticks) == (
        rollback_target.pid,
        rollback_target.start_ticks,
    ):
        _fail(
            "reused-candidate-process",
            "candidate and reconstructed baseline must use distinct PID/start-tick identities",
        )
    candidate_artifacts = _artifact_map_from_paths(
        root_fd,
        row["candidate_artifacts"],
        "rollback v3 bind request.candidate_artifacts",
        reserved_paths=reserved_paths,
    )
    rollback_artifacts = _artifact_map_from_paths(
        root_fd,
        row["rollback_artifacts"],
        "rollback v3 bind request.rollback_artifacts",
        reserved_paths=reserved_paths,
    )
    atomic_switch = _path_descriptor_map(
        root_fd,
        row["atomic_switch"],
        checker.ATOMIC_SWITCH_FIELDS,
        "rollback v3 bind request.atomic_switch",
        reserved_paths=reserved_paths,
    )
    _preflight_baseline_aliases(root_fd, baseline_descriptor, reserved_paths)
    return {
        "schema_version": checker.ROLLBACK_V3_MANIFEST_VERSION,
        "capture_status": "captured",
        "qualification_status": "not-run",
        "candidate_id": candidate_id,
        "bindings": bindings,
        "reconstructed_baseline": {"manifest": baseline_descriptor.as_json()},
        "candidate": candidate,
        "rollback": rollback,
        "candidate_artifacts": candidate_artifacts,
        "rollback_artifacts": rollback_artifacts,
        "atomic_switch": {name: descriptor.as_json() for name, descriptor in atomic_switch.items()},
    }


def _lock_output(root_fd: int) -> None:
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (AttributeError, OSError) as error:
        _fail("output-lock-unavailable", f"cannot acquire exclusive output lock: {error}")


def _assert_output_absent(root_fd: int, manifest_name: str) -> None:
    for name in _output_names(manifest_name):
        try:
            os.lstat(name, dir_fd=root_fd)
        except FileNotFoundError:
            continue
        except OSError as error:
            _fail(
                "output-preflight-failed",
                f"cannot inspect rollback v3 output {name!r}: {error}",
            )
        _fail(
            "output-name-collision",
            f"rollback v3 output or reserved sibling {name!r} already exists",
        )


def _bind_raw_rollback_manifest_held_locked_fd(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Create and self-verify v3 through an already exclusively locked root FD.

    This is intentionally a nonterminal-only held-FD primitive.  An outer
    raw transaction compositor may retain the same root and switch locks
    across a normal transaction return, v3 publication, and a later v4
    finalizer without reopening or relocking this root.  It never writes a
    completion marker and does not itself establish a producer-success edge
    for any earlier transaction.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback v3 bind evidence root",
        )
    )
    name = _manifest_name(manifest_name)
    _assert_output_absent(root_fd, name)
    manifest = _manifest_from_request(root_fd, bind_request_path, name)
    raw_manifest = _common(lambda: common.canonical_json_bytes(manifest))
    draft_descriptor = _common(
        lambda: common.descriptor_for_bytes(name, raw_manifest, "rollback v3 draft manifest")
    )
    preflight = _checker(
        lambda: checker.verify_rollback_provenance_v3_bytes_fd(
            root_fd,
            draft_descriptor,
            raw_manifest,
        )
    )
    created = _common(
        lambda: common.write_create_only_json(
            root_fd,
            name,
            manifest,
            "rollback v3 raw manifest",
        )
    )
    if created.descriptor(name, "rollback v3 created manifest") != draft_descriptor:
        _fail(
            "published-manifest-descriptor-mismatch",
            "create-only publication bytes differ from the preflight manifest",
        )
    replayed = _checker(lambda: checker.verify_rollback_provenance_v3_fd(root_fd, name))
    if replayed != preflight:
        _fail(
            "post-publication-replay-drift",
            "on-disk rollback manifest replay differs from its held-FD preflight",
        )
    return replayed


def bind_raw_rollback_manifest_fd(
    root_fd: int,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Create and self-verify one nonterminal raw v3 rollback manifest.

    This held-FD entry point can prove the root is an EUID-owned private
    directory, but a descriptor alone cannot prove its pathname lies outside
    this source checkout.  Public callers must therefore use
    :func:`bind_raw_rollback_manifest`; internal callers must pass an FD that
    has already received the same external-root guard.
    """

    _common(
        lambda: common.require_private_evidence_directory_fd(
            root_fd,
            "rollback v3 bind evidence root",
        )
    )
    name = _manifest_name(manifest_name)
    _lock_output(root_fd)
    try:
        return _bind_raw_rollback_manifest_held_locked_fd(root_fd, bind_request_path, name)
    finally:
        try:
            fcntl.flock(root_fd, fcntl.LOCK_UN)
        except OSError:
            pass


def bind_raw_rollback_manifest(
    evidence_root: Path,
    bind_request_path: str,
    manifest_name: str,
) -> dict[str, Any]:
    """Open one private evidence root and bind one raw-only v3 manifest."""

    _assert_external_to_source_checkout(evidence_root)
    root_fd = _common(
        lambda: common.open_private_evidence_directory(
            evidence_root,
            "--evidence-root",
        )
    )
    try:
        return bind_raw_rollback_manifest_fd(root_fd, bind_request_path, manifest_name)
    finally:
        os.close(root_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--bind-request", required=True)
    parser.add_argument("--manifest-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = bind_raw_rollback_manifest(
            args.evidence_root,
            args.bind_request,
            args.manifest_name,
        )
    except (RollbackBindError, checker.RollbackV3ProvenanceError, common.ProvenanceV2Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(common.canonical_json_bytes(report) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
